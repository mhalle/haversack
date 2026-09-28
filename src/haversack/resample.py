"""Forward resampling: image intensities from the acquisition grid onto the model grid.

The other leg from :mod:`haversack.restore`. Exactness is the whole point - a network sees what
its training preprocessing produced, so this reproduces the CPU resamplers rather than
approximating them, and does it on the GPU:

* ``convention="corner"`` - ``scipy.ndimage.zoom(grid_mode=False)``, the voxel-corner point
  grid, which is what TotalSegmentator's ``change_spacing`` uses.
* ``convention="center"`` - ``scipy.ndimage.zoom(grid_mode=True)`` == ``skimage.resize``, the
  voxel-center (half-pixel) grid, which is what nnU-Net's own
  ``resample_data_or_seg_to_shape`` uses.

Neither is re-implemented. ``zoom`` is linear and separable, so zooming an identity matrix
along one axis *is* the operator for that axis - spline prefilter, boundary mode and
coordinate convention included, for any order. We build those matrices with scipy once (they
are tiny and cached) and apply them on the GPU, which is exact by construction: there is no
boundary handling here to get subtly wrong.

Ported from the nnU-Net fork's ``resample_data_or_seg_to_shape_gpu`` (tag ``resample-gpu-v1``,
draft PR mhalle/nnUNet#1), keeping only the intensity path so haversack does not pin a fork. The
fork's label / one-hot / anti-aliased paths stay there; anti-aliasing in particular is a
distribution shift for models trained on the scipy pipeline
(``medseg/docs/resampler-parity-finding.md``) and haversack does not want it.

nnU-Net's separate-z path is here (``separate_z_axis``, 2026-09-28): for an anisotropic image
nnU-Net resizes each slice in-plane (skimage, clipped to that slice's range) and then samples the
low-resolution axis with ``map_coordinates`` at ``order_z`` (0 by default: nearest). Before that
date a thick-slice input to a native model was resampled with a cubic spline along z instead.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

CONVENTIONS = ("corner", "center")


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device_raw(spec="auto") -> torch.device:
    """``"auto"`` picks the best available accelerator; anything else is taken literally.

    The default must not name a vendor: haversack exists to be portable, and a hard-coded "mps"
    made it fail on the first CUDA machine it ever met.
    """
    if spec is None or spec == "auto":
        return best_device()
    return torch.device(spec)


@functools.lru_cache(maxsize=128)
def scipy_axis_matrix(n_in: int, n_out: int, order: int, mode: str, grid_mode: bool) -> np.ndarray:
    """The exact 1-D operator of ``scipy.ndimage.zoom`` along one axis, ``(n_out, n_in)`` float64."""
    from scipy import ndimage
    if n_in == n_out:
        return np.eye(n_in)
    probe = ndimage.zoom(np.eye(n_in, dtype=np.float64), (1.0, n_out / n_in),
                         order=order, mode=mode, grid_mode=grid_mode)
    if probe.shape != (n_in, n_out):
        raise RuntimeError(f"scipy zoom probe produced {probe.shape}, expected {(n_in, n_out)}")
    return np.ascontiguousarray(probe.T)


@functools.lru_cache(maxsize=128)
def separate_z_axis_matrix(n_in: int, n_out: int, order_z: int) -> np.ndarray:
    """nnU-Net's separate-z step along the low-resolution axis, ``(n_out, n_in)`` float64.

    ``resample_data_or_seg`` samples that axis with ``map_coordinates(order=order_z,
    mode="nearest")`` at ``float(n_in) / n_out * (j + 0.5) - 0.5``; the other two coordinates are
    integers there, so the step is one-dimensional. It is built by applying that same call to
    the identity with the same coordinates, so a nearest pick at an exact tie is nnU-Net's.
    """
    from scipy.ndimage import map_coordinates
    if n_in == n_out:
        return np.eye(n_in)
    coords = float(n_in) / n_out * (np.arange(n_out) + 0.5) - 0.5
    eye = np.eye(n_in, dtype=np.float64)
    cols = [map_coordinates(eye[i], [coords], order=int(order_z), mode="nearest") for i in range(n_in)]
    return np.ascontiguousarray(np.stack(cols, axis=1))


#: nnU-Net's ``ANISO_THRESHOLD``: separate-z when the coarsest spacing exceeds 3x the finest
ANISO_THRESHOLD = 3


def separate_z_axis(spacing_zyx, new_spacing_zyx, force_separate_z=None) -> int | None:
    """The axis nnU-Net resamples separately, or None: ``determine_do_sep_z_and_axis``
    restated (this module does not import nnU-Net; tests/test_separate_z.py holds the two to
    each other). The image's spacing decides first, then the target's; an image with two (or
    three) equally coarse axes is not resampled separately. Spacings in any consistent axis
    order; the result indexes that order."""
    def coarse(sp):
        sp = np.asarray(sp, dtype=np.float64)
        return np.where(max(sp) / sp == 1)[0]

    def aniso(sp):
        return (np.max(sp) / np.min(sp)) > ANISO_THRESHOLD

    if force_separate_z is not None:
        axis = coarse(spacing_zyx) if force_separate_z else None
    elif aniso(spacing_zyx):
        axis = coarse(spacing_zyx)
    elif aniso(new_spacing_zyx):
        axis = coarse(new_spacing_zyx)
    else:
        axis = None
    return int(axis[0]) if axis is not None and len(axis) == 1 else None


def _apply_axis(x: torch.Tensor, axis: int, w: torch.Tensor) -> torch.Tensor:
    """Apply an ``(n_out, n_in)`` operator along ``axis`` via matmul."""
    x = x.movedim(axis, -1)
    shp = x.shape
    out = (x.reshape(-1, shp[-1]) @ w.t()).reshape(*shp[:-1], w.shape[0])
    return out.movedim(-1, axis)


def compute_nnunet_shape(shape_zyx, spacing_zyx, new_spacing_zyx) -> tuple[int, int, int]:
    """nnU-Net's ``compute_new_shape``: ``int(round(spacing / new_spacing * shape))`` per axis,
    the ratio of spacings taken first (TotalSegmentator's rule, :func:`target_shape`, multiplies
    the shape by the zoom; the two can differ by one voxel where the product lands on .5)."""
    return tuple(int(round(float(a) / float(b) * int(n))) for n, a, b in zip(shape_zyx, spacing_zyx, new_spacing_zyx))


def target_shape(shape_zyx, spacing_zyx, new_spacing_zyx) -> tuple[int, int, int]:
    """``round(shape * spacing / new_spacing)`` - TotalSegmentator's ``change_spacing`` rule."""
    zoom = np.asarray(spacing_zyx, dtype=np.float64) / np.asarray(new_spacing_zyx, dtype=np.float64)
    return tuple(max(1, int(round(float(s) * float(z)))) for s, z in zip(shape_zyx, zoom))


@torch.no_grad()
def resample_data(data_zyx, new_shape=None, *, spacing_zyx=None, new_spacing_zyx=None,
                  convention: str = "corner", order: int = 3, mode: str = "nearest",
                  device=None, out_dtype=None, clip: bool | None = None,
                  separate_z_axis: int | None = None, order_z: int = 0) -> np.ndarray:
    """Resample a 3-D intensity volume to ``new_shape`` (or to ``new_spacing_zyx``).

    ``mode`` is a **scipy** boundary name (``"nearest"`` is what skimage spells ``"edge"``).

    ``clip`` bounds the output to the input's value range, which is what ``skimage.resize``
    does per call and nnU-Net inherits; ``scipy.ndimage.zoom`` does not clip. The default
    follows the convention being reproduced (on for ``center``, off for ``corner``) so that
    each mode matches its reference exactly. ``out_dtype`` applies the caller's cast at the
    end - note that TotalSegmentator uses ``astype``, i.e. truncation, not rounding.

    ``separate_z_axis`` (``convention="center"`` only) is nnU-Net's separate-z path: the other
    two axes are resampled with ``order`` and clipped per slice of that axis to the slice's own
    range (skimage resizes, and clips, one slice at a time there), then that axis is sampled
    with ``order_z`` (:func:`separate_z_axis_matrix`).
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}; got {convention!r}")
    arr = np.asarray(data_zyx)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D volume; got shape {arr.shape}")
    if new_shape is None:
        if spacing_zyx is None or new_spacing_zyx is None:
            raise ValueError("pass new_shape, or both spacing_zyx and new_spacing_zyx")
        new_shape = target_shape(arr.shape, spacing_zyx, new_spacing_zyx)
    new_shape = tuple(int(s) for s in new_shape)
    if len(new_shape) != 3:
        raise ValueError(f"new_shape must be (Z, Y, X); got {new_shape}")
    if clip is None:
        clip = convention == "center"
    if separate_z_axis is not None and convention != "center":
        raise ValueError("separate_z_axis is nnU-Net's path and needs convention='center'")

    dev = torch.device(device) if device is not None else best_device()
    # float64 is unsupported on MPS and unnecessary here: the operators are float32-exact to
    # ~1e-7 relative, far below the intensity quantization these volumes carry.
    work = torch.float32 if dev.type != "cpu" else (torch.float64 if arr.dtype == np.float64 else torch.float32)
    t = torch.as_tensor(np.ascontiguousarray(arr), device=dev, dtype=work)
    grid_mode = convention == "center"
    if separate_z_axis is not None:
        if tuple(t.shape) == new_shape:                    # nnU-Net returns the data untouched
            out = t.cpu().numpy()
            return out.astype(out_dtype) if out_dtype is not None else out
        z = int(separate_z_axis)
        others = tuple(a for a in range(3) if a != z)
        # each slice's own range, as skimage clips each in-plane resize to its input slice
        lo = t.amin(dim=others, keepdim=True) if clip else None
        hi = t.amax(dim=others, keepdim=True) if clip else None
        did_spline = False
        for axis in others:
            n_in, n_out = t.shape[axis], new_shape[axis]
            if n_in == n_out:
                continue
            w = torch.as_tensor(scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), True),
                                device=dev, dtype=work)
            t = _apply_axis(t, axis, w)
            did_spline = did_spline or order >= 2
        if clip and did_spline:
            t = torch.maximum(torch.minimum(t, hi), lo)
        if t.shape[z] != new_shape[z]:
            w = torch.as_tensor(separate_z_axis_matrix(int(t.shape[z]), int(new_shape[z]), int(order_z)),
                                device=dev, dtype=work)
            t = _apply_axis(t, z, w)
    else:
        lo, hi = (t.amin(), t.amax()) if clip else (None, None)
        did_spline = False
        for axis in range(3):
            n_in, n_out = t.shape[axis], new_shape[axis]
            if n_in == n_out:
                continue
            w = torch.as_tensor(scipy_axis_matrix(int(n_in), int(n_out), int(order), str(mode), grid_mode),
                                device=dev, dtype=work)
            t = _apply_axis(t, axis, w)
            did_spline = did_spline or order >= 2
        if clip and did_spline:
            t = torch.clamp(t, lo, hi)
    out = t.cpu().numpy()
    del t
    return out.astype(out_dtype) if out_dtype is not None else out


_MPS_CAP_ARMED = False


def _arm_mps_memory_cap() -> None:
    """Cap the MPS allocator at the device's recommended working set, once per process.

    PyTorch's MPS allocator lets a process grow to 1.7x ``recommendedMaxWorkingSetSize``
    before it raises out-of-memory. Past 1.0x, Metal cannot back the buffers and kernels
    fail SILENTLY: a conv3d returns all zeros, no exception (reproduced 2026-09-03 with
    identity convolutions - exact at 8 GiB of live activations, zeros at 11 GiB on a 10.7
    GiB ceiling; SynthStrip masked the whole image that way). Capped at 1.0x the allocator
    reclaims its cache under pressure and the same forward runs correctly - SynthStrip's
    256^3 fp32 peaks at 6 GiB - and a real shortfall raises. ``HAVERSACK_MPS_MEMORY_FRACTION``
    overrides (``0`` = leave PyTorch's default). Global by nature: the allocator is per process.
    """
    global _MPS_CAP_ARMED
    if _MPS_CAP_ARMED:
        return
    _MPS_CAP_ARMED = True
    import os
    frac = float(os.environ.get("HAVERSACK_MPS_MEMORY_FRACTION", "1.0"))
    if frac > 0 and hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
        torch.mps.set_per_process_memory_fraction(frac)


def resolve_device(spec="auto") -> torch.device:
    """``"auto"`` -> cuda / mps / cpu (or an explicit device). Resolving to MPS also arms
    the allocator cap (:func:`_arm_mps_memory_cap`), so every haversack path that runs on
    Apple Silicon fails loudly on memory rather than silently."""
    dev = _resolve_device_raw(spec)
    if dev.type == "mps":
        _arm_mps_memory_cap()
    return dev
