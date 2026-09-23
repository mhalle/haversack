"""The nnU-Net family: any nnU-Net network's encoder, its skips at chosen stages as lattices.

Ported 2026-09-23 from feldglas's null adapter (the RADAR study's "null model", EXPLORATION 5.12),
which made its fields with haversack's own reader, resampler and normalization - so a field here
starts from exactly the input a segmentation of the same task would see. The network is run tile
by tile at nnU-Net's own patch, the tiles started on a lattice aligned to the coarsest kept stage
(so every tile's tokens land on whole field tokens), and each lattice is blended with nnU-Net's
patch Gaussian averaged over each token's box. Only the ENCODER runs: the decoder made the old
tool's native mask, which fields no longer carry, and the skips are the same computation without it.

Today the TotalSegmentator lineage (canonical RAS, the corner resampling rule); an nnU-Net catalog of
another lineage is refused by name until its convention is added here as spec data.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..errors import InputError, ModelNotFound
from .radar import Prepared


def _task_spec(spec, root):
    """The TaskSpec of the segmentation task whose network this encoder is."""
    from ..ecosystems import EcosystemCatalog
    eco, short, _, _ = EcosystemCatalog(root=root).resolve(spec.uses_task)
    return eco.spec(short, root)


def load(spec, weights_dir, device, dtype, task_weights=None):
    """haversack's TorchModel for the encoder's dataset, from weights ALREADY installed (the
    task's, fetched by ``haversack weights fetch <task>``) under ``task_weights`` - the root a
    server's Segmenter reads, so the key (its describe) and the compute read ONE install - or
    the default root."""
    from ..network import TorchModel
    from ..tasks import resolve_model_folder, weights_root
    root = Path(task_weights) if task_weights is not None else weights_root("ts")
    task = _task_spec(spec, root)
    if getattr(task, "lineage", "ts") != "ts":
        raise InputError(f"{spec.name}: an nnU-Net encoder of the {task.lineage!r} lineage is not supported yet")
    ds = spec.options["dataset"]
    try:
        folder = resolve_model_folder(ds, model_root=root, **task.model_choice(ds))
    except ModelNotFound:
        raise InputError(f"{spec.name}: its task's weights are not installed - run `haversack weights fetch "
                         f"{spec.uses_task}`") from None
    m = TorchModel(folder, device=device, dtype={"torch.float16": "fp16", "torch.float32": "fp32"}.get(str(dtype), "fp32"))
    if m.transpose_forward != (0, 1, 2):
        raise InputError(f"{folder.name}: a transposed model - the tiling assumes none")
    if any(p % spec.options["align"] for p in m.patch):
        raise InputError(f"{folder.name}: patch {m.patch} is not a multiple of {spec.options['align']}")
    return m


def padded_extent(n: int, patch: int, align: int) -> int:
    """An axis padded at its END: at least a patch, and a multiple of ``align``."""
    return max(patch, int(math.ceil(n / align)) * align)


def tile_starts(n: int, patch: int, align: int) -> list[int]:
    """Every ``step`` (half a patch rounded down to ``align``) from 0, and one flush with the end."""
    step = max(align, (patch // 2) // align * align)
    starts = list(range(0, n - patch + 1, step))
    if starts[-1] != n - patch:
        starts.append(n - patch)
    return starts


def tile_slices(padded_shape, patch, kernels, align):
    """``(voxels, [tokens per lattice])`` per tile: where its input comes from and where each
    lattice's tokens go - exact, since every start is a multiple of every kernel."""
    axes = [tile_starts(int(n), int(p), align) for n, p in zip(padded_shape, patch)]
    out = []
    for z in axes[0]:
        for y in axes[1]:
            for x in axes[2]:
                s = (z, y, x)
                vox = tuple(slice(a, a + int(p)) for a, p in zip(s, patch))
                toks = [tuple(slice(a // k, (a + int(p)) // k) for a, p, k in zip(s, patch, kk)) for kk in kernels]
                out.append((vox, toks))
    return out


def token_weights(gaussian, kernel) -> np.ndarray:
    """nnU-Net's patch Gaussian averaged over each token's box."""
    g = np.asarray(gaussian, np.float64)
    s = [x // k for x, k in zip(g.shape, kernel)]
    return g.reshape(s[0], kernel[0], s[1], kernel[1], s[2], kernel[2]).mean(axis=(1, 3, 5))


def prepare(spec, image, model) -> Prepared:
    """haversack's own segmentation input for this model: canonical orientation, the model's
    spacing by the corner rule, its normalization; padded at the end to the tile lattice."""
    import SimpleITK as sitk
    import torch
    from ..io import CANONICAL, geometry_of, orientation_of
    from ..preprocess import normalize_for, to_model_grid
    from ..ranked_output import model_grid_geometry
    m = model
    original = orientation_of(image)
    im = sitk.DICOMOrient(image, CANONICAL)
    arr, geom = sitk.GetArrayFromImage(im), geometry_of(im)
    grid = to_model_grid(arr, geom, m.spacing_zyx, convention="corner", device=str(m.device.type),
                         original_orientation=original)
    x = normalize_for(grid, m)[0]
    shape = tuple(int(s) for s in x.shape)
    align = spec.options["align"]
    padded = tuple(padded_extent(n, p, align) for n, p in zip(shape, m.patch))
    xp = torch.zeros(padded, dtype=torch.float32)                         # nnU-Net pads with 0 after normalization
    xp[:shape[0], :shape[1], :shape[2]] = x.float().cpu()
    eff, origin, direction, centering = model_grid_geometry(
        {"frame": grid.frame.to_meta(), "model_grid": list(shape), "convention": "corner"})
    D = np.asarray(direction, float).reshape(3, 3)
    rows = [[float(v) for v in D[:, 2 - a] * float(eff[a])] for a in range(3)]   # ITK columns (x, y, z) -> rows (Z, Y, X)
    Dd = np.asarray(image.GetDirection(), float).reshape(3, 3) * np.asarray(image.GetSpacing(), float)
    extra = {"model": m.folder.name, "model_shape": list(shape), "patch": list(m.patch), "centering": centering,
             "stages": list(spec.options["stages"]),
             "input_grid": {"shape": [int(v) for v in image.GetSize()], "directions": Dd.T.round(9).tolist(),
                            "origin": np.round(image.GetOrigin(), 9).tolist(), "space": "left-posterior-superior"}}
    return Prepared(xp, {"shape": list(padded), "directions": rows, "origin": [float(v) for v in origin]},
                    ((0, 0, 0), shape), extra)


def run(spec, model, prepared: Prepared, device, dtype, slab: int = 0) -> list[np.ndarray]:
    """Tokens per lattice: the encoder's chosen skips, tile by tile, blended per token box."""
    import torch
    m = model
    dev = m.device
    o = spec.options
    kernels = [l.kernel for l in spec.lattices]
    widths = [l.width for l in spec.lattices]
    padded = tuple(prepared.grid["shape"])
    xp = prepared.tensor.to(dev, m.dtype)
    gauss = m._gaussian_cpu.double().numpy()
    W = [torch.as_tensor(token_weights(gauss, k), dtype=torch.float32, device=dev) for k in kernels]
    acc = [torch.zeros((w, *(p // k for p, k in zip(padded, kk))), device=dev) for w, kk in zip(widths, kernels)]
    wsum = [torch.zeros(tuple(p // k for p, k in zip(padded, kk)), device=dev) for kk in kernels]
    tiles = tile_slices(padded, m.patch, kernels, o["align"])
    with torch.inference_mode():
        for vox, toks in tiles:
            skips = m.net.encoder(xp[vox][None, None])
            for j, stage in enumerate(o["stages"]):
                f = skips[stage][0].float()
                if tuple(f.shape) != (widths[j], *W[j].shape):
                    raise RuntimeError(f"stage {stage}: {tuple(f.shape)}, expected {(widths[j], *W[j].shape)}")
                acc[j][(slice(None), *toks[j])] += f * W[j]
                wsum[j][toks[j]] += W[j]
    prepared.extra["tiles"] = len(tiles)
    return [(a / s).permute(1, 2, 3, 0).reshape(-1, a.shape[0]).float().cpu().numpy() for a, s in zip(acc, wsum)]
