"""SynthStrip brain extraction (skull-strip) as a haversack engine.

SynthStrip is a contrast-agnostic learned brain-mask generator (a small 3D UNet,
Hoopes et al. 2022) - a different algorithm family from nnU-Net and FastSurfer.

**The model + conform now live in the standalone ``synthstrip-torch`` package**
(mirroring how FastSurfer's CNN lives in ``fastsurfer-lean``): it owns the surfa
trained-conform and the net, and returns the signed distance transform (SDT). This
module is the thin haversack ADAPTER: it keeps the restore geometry (physical-space
``grid_sample``, shared with FastSurfer), the mask cleanup, and the wrapping into an
haversack :class:`~haversack.result.Segmentation`, plus the result-cache weights identity.

Flow: ``synthstrip_torch.predict_sdt`` (surfa conform 1 mm/LIA + net -> conformed
SDT handed back as a SimpleITK image) -> restore the *graded* SDT to the input grid
-> threshold at ``border`` mm + largest filled component -> a 1-label brain mask.
"""
from __future__ import annotations

import os

import numpy as np

from . import registry as _registry
from ..errors import ResourceError
from .geometry import resample_affine as _resample_affine

ENGINE = "synthstrip"

# One literal, in the registry - see the note in engines/fastsurfer.py.
WEIGHTS_ID = ENGINE
WEIGHTS_VERSION = _registry.ENGINES[ENGINE].weights_identity()[0]["version"]
BRAIN_LABEL = 1


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key (from the registry) -
    one source of truth for both the API-side describe and the worker re-key."""
    return _registry.ENGINES[ENGINE].weights_identity()


def _get_model(device: str):
    """The cached SynthStrip net for ``device`` (built + weights loaded once per
    device by ``synthstrip_torch``). ``HAVERSACK_SYNTHSTRIP_MODEL`` overrides the
    weights path; otherwise the package fetches + caches ``synthstrip.1.pt``."""
    import synthstrip_torch

    return synthstrip_torch.load_model(
        path=os.environ.get("HAVERSACK_SYNTHSTRIP_MODEL"), device=device)


def restore_sdt_gpu(sdt, source_ref, target_ref, device="cuda", outside=100.0):
    """Trilinear-resample the 1-channel SDT from ``source_ref``'s grid onto
    ``target_ref``'s grid (physical-space ``grid_sample``, full affine so any
    orientation/rotation is handled). Voxels sampling outside the source get
    ``outside`` (a large positive distance = far from brain, matching surfa's
    ``fill=100``). ``sdt`` is a torch tensor already ``(Zs,Ys,Xs)`` on a device
    (on-GPU path) or a numpy array. Returns the resampled SDT ``(Zt,Yt,Xt)``."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    if isinstance(sdt, torch.Tensor):
        Zs, Ys, Xs = (int(s) for s in sdt.shape)
        field = sdt.to(dev).float()
    else:
        Zs, Ys, Xs = sdt.shape
        field = torch.from_numpy(np.ascontiguousarray(sdt)).to(dev, torch.float32)
    # Shift by -outside so grid_sample's zero-padding REPRESENTS the outside fill:
    # outside/partial-edge taps contribute 0 -> +outside after, matching SimpleITK's
    # constant fill exactly (else zero-padding would drag the edge SDT toward 0 and
    # spuriously threshold as brain). Makes this bit-match restore_sdt_cpu.
    field = (field - float(outside))[None, None]
    tgt = target_ref.GetSize()
    Xt, Yt, Zt = int(tgt[0]), int(tgt[1]), int(tgt[2])
    A, t = _resample_affine(source_ref, target_ref)
    zz, yy, xx = torch.meshgrid(torch.arange(Zt), torch.arange(Yt), torch.arange(Xt),
                                indexing="ij")
    idx = torch.stack([xx, yy, zz], dim=-1).to(dev, torch.float64)
    src = idx @ torch.as_tensor(A, device=dev, dtype=torch.float64).T \
        + torch.as_tensor(t, device=dev, dtype=torch.float64)
    N = torch.as_tensor([Xs, Ys, Zs], device=dev, dtype=torch.float64)
    grid = ((src + 0.5) * 2.0 / N - 1.0).to(torch.float32)[None]
    out = F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)[0, 0] + float(outside)
    return out.float().cpu().numpy()


def restore_sdt_cpu(sdt_zyx, source_ref, target_ref, outside=100.0):
    """Memory-frugal fallback: one SimpleITK linear resample of the SDT (single
    channel, so cheap even on CPU). ``outside`` is the fill for voxels off the
    source grid (surfa uses 100)."""
    import SimpleITK as sitk

    ch = sitk.GetImageFromArray(np.ascontiguousarray(sdt_zyx.astype(np.float32)))
    ch.CopyInformation(source_ref)
    out = sitk.Resample(ch, target_ref, sitk.Transform(), sitk.sitkLinear,
                        float(outside), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)


# Apple Silicon (2026-09-03): the net's first local run masked the whole image because
# PyTorch's MPS allocator let the process grow past the device's recommended working set
# and Metal then returned an all-zero field with no error. haversack caps the allocator at
# 1.0x when it resolves an MPS device (resample._arm_mps_memory_cap), after which the same
# forward runs correctly in fp32 (256^3 peaks at 6 GiB; Dice 0.99997 against the Modal
# CUDA result) and a real shortfall raises - which is what the fallback below catches.


class _HalfNet:
    """Runs a copy of the net in fp16 behind the fp32 interface predict_sdt expects."""

    def __init__(self, model):
        import copy
        import torch
        self.net = copy.deepcopy(model).to(torch.float16).eval()

    def __call__(self, x):
        import torch
        return self.net(x.to(torch.float16)).to(torch.float32)

    def eval(self):
        return self


_HALF: dict = {}


def _capture_sdt(t1_img, device: str):
    """SynthStrip's conform + net via ``synthstrip_torch`` -> ``(sdt_zyx, sdt_sitk)``:
    the SDT array ``(z,y,x)`` and its geometry as a SimpleITK image (so the restore
    needs no manual axis conversion)."""
    import synthstrip_torch

    model = _get_model(device)
    try:
        return synthstrip_torch.predict_sdt(t1_img, model=model, device=device)
    except RuntimeError as e:
        if device != "mps" or "out of memory" not in str(e).lower():
            raise
    # A real MPS out-of-memory (the allocator is capped, so it is real): halve the
    # working set with an fp16 copy of the net (0.015 mm worst case on the SDT), then
    # refuse - never let Metal answer with zeros.
    import torch
    torch.mps.empty_cache()
    half = _HALF.get(id(model)) or _HALF.setdefault(id(model), _HalfNet(model))
    try:
        return synthstrip_torch.predict_sdt(t1_img, model=half, device=device)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        raise ResourceError(
            "synthstrip: this volume does not fit MPS memory even with the net in fp16; "
            "run with device='cpu'") from e


def _refuse_constant_field(sdt, device) -> None:
    """A signed distance field that is the same number everywhere is not a prediction.
    Seen 2026-09-03: torch 2.14 on MPS returned exactly zero for the full 256^3 conform
    while a 64^3 crop matched the CPU to 1e-5 - and "everything is below 1 mm" then masks
    the whole image. Silent garbage is the one thing this engine must not produce."""
    import numpy as np
    a = np.asarray(sdt.detach().cpu().numpy() if hasattr(sdt, "detach") else sdt, dtype=np.float32)
    if not np.isfinite(a).all() or float(a.max() - a.min()) < 1e-6:
        from ..errors import ResourceError
        raise ResourceError(
            f"synthstrip: the net returned a constant field on {device!r} (min {a.min():.3g}, "
            f"max {a.max():.3g}); that is a backend fault, not a brain. Known on MPS with torch "
            "2.14 at the full conformed size - run with device='cpu' (or on CUDA).")


def segment(t1_input, *, out_dir=None, device: str = "cuda", restore: str = "auto",
            border: float = 1.0, self_check: bool = True):
    """Skull-strip a brain image with SynthStrip and return a 1-label brain-mask
    :class:`haversack.result.Segmentation` on the input's grid.

    ``t1_input`` is a SimpleITK image (memory-in) or a path. ``restore`` selects
    the SDT-restore backend (``"gpu"``/``"cpu"``/``"auto"`` = GPU on CUDA). The
    graded SDT is resampled to the input grid, then thresholded at ``border`` mm
    (SynthStrip's default) and reduced to the largest filled connected component -
    thresholding *after* the resample gives a sub-voxel mask boundary. ``out_dir``
    is accepted for call-site compatibility and unused (no temp files)."""
    import time

    import SimpleITK as sitk

    from ..grid import Grid
    from ..result import Segmentation
    from ..values import LabelSchema

    if isinstance(t1_input, sitk.Image):
        t1_img = t1_input
    else:
        from .. import io
        t1_img = io.read_image(str(t1_input))

    use_gpu = restore == "gpu" or (restore == "auto" and str(device).startswith("cuda"))
    timings: dict[str, float] = {}
    _t = time.perf_counter()
    sdt_zyx, conf_orig = _capture_sdt(t1_img, device)     # surfa conform + net (1-channel SDT)
    _refuse_constant_field(sdt_zyx, device)
    timings["capture"] = time.perf_counter() - _t

    _t = time.perf_counter()
    if use_gpu:
        sdt_native = restore_sdt_gpu(sdt_zyx, conf_orig, t1_img, device)
    else:
        sdt_native = restore_sdt_cpu(sdt_zyx, conf_orig, t1_img)
    timings["restore"] = time.perf_counter() - _t

    from scipy import ndimage
    mask = sdt_native < border
    lbl, n = ndimage.label(mask)
    if n:
        biggest = 1 + int(np.argmax(np.bincount(lbl.flat)[1:]))
        mask = ndimage.binary_fill_holes(lbl == biggest)
    nfg = int(mask.sum())
    print(f"[synthstrip] conf={conf_orig.GetSize()} t1={t1_img.GetSize()} "
          f"restore={'gpu' if use_gpu else 'cpu'} components={n} "
          f"brain_voxels={nfg}", flush=True)
    if nfg == 0:
        raise RuntimeError(
            f"synthstrip mask is empty: no voxel had SDT < {border} mm "
            f"(conf={conf_orig.GetSize()}, t1={t1_img.GetSize()}); "
            "conform or restore geometry is likely wrong")

    labels_arr = mask.astype(np.uint16)
    out_img = sitk.GetImageFromArray(labels_arr)
    out_img.CopyInformation(t1_img)
    grid = Grid(shape=tuple(int(s) for s in labels_arr.shape),
                spacing=tuple(float(s) for s in reversed(t1_img.GetSpacing())),
                origin=tuple(float(o) for o in reversed(t1_img.GetOrigin())))
    prov = {"engine": "synthstrip", "synthstrip_version": WEIGHTS_VERSION,
            "network": "SynthStrip UNet (SDT)",
            "restore": f"sdt-graded ({'gpu' if use_gpu else 'cpu'})",
            "border_mm": border, "device": device}
    return Segmentation(labels=out_img, schema=LabelSchema(names={BRAIN_LABEL: "Brain"}),
                        grid=grid, spec=None, timings=timings, provenance=prov)


def run_local(image, *, device="auto", progress=None, cancel=None, **_policy):
    """The in-process entry point (:attr:`Engine.compute`), as for FastSurfer: resolve
    ``device`` the way the nnU-Net path does, report one stage, honor the cancel token
    before the net is built. The SDT restore falls to the CPU path off CUDA by
    ``segment``'s own ``restore="auto"``. nnU-Net policy keys are accepted and ignored."""
    from ..progress import Reporter
    from ..resample import resolve_device
    report = Reporter.of(progress, cancel=cancel)
    dev = resolve_device(device).type
    report.check()
    report.stage("predict", f"synthstrip:mask on {dev}")
    seg = segment(image, device=dev)
    report.check()
    return seg
