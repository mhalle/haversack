"""The RADAR family: a PlainConvEncoder with three 1x1 projections, and upstream's input convention.

Ported 2026-09-23 from feldglas's tools (radar_export_modal.py ``_prep``/``_grid`` and
radar_encode_local.py), which made the RADAR study's 1,680 fields; the port is held to them token
for token. Every quirk of upstream's DataFolder is kept, because the weights were trained on it:
the resample target reads the spacing off the affine's DIAGONAL and swaps x and y
(``tgt = [h * sp[1], w * sp[0], dd * sp[2] / 5]``), the crop's upper bound is the last non-air index
itself plus the margin (so with no margin that index is cut), and min-max normalization is over
the whole resampled volume.

The model is built from the installed ``dynamic_network_architectures`` (nnU-Net's), not from
upstream's fork: its encoder is a standard PlainConvEncoder, and on the real checkpoint the tokens
are bit-identical to upstream's (2026-09-23). The fork's light decoder - RADAR's own organ mask - is
not built: fields carry no mask (feldglas docs/embedding-field.md, decision 1).
"""
from __future__ import annotations

import numpy as np

from ..errors import InputError


class Prepared:
    """An input made ready for the network: the tensor (1, 1, Z, Y, X), the model grid in the
    world (rankfield's form), the model voxels that hold image (``data_box``), and what the
    preparation did (provenance ``extra``)."""

    def __init__(self, tensor, grid: dict, data_box, extra: dict):
        self.tensor, self.grid, self.data_box, self.extra = tensor, grid, data_box, extra


def load(spec, weights_dir, device, dtype):
    """The vision encoder and its projections, from the checkpoint - loaded with
    ``weights_only=True`` (no pickle executed), every key accounted for."""
    import torch
    from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
    o = spec.options
    ck = torch.load(weights_dir / spec.weights[0].name, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    a = o["arch"]
    enc = PlainConvEncoder(input_channels=1, n_stages=a["n_stages"], features_per_stage=a["features_per_stage"],
                           conv_op=torch.nn.Conv3d, kernel_sizes=a["kernel_sizes"], strides=a["strides"],
                           n_conv_per_stage=a["n_conv_per_stage"], conv_bias=True, norm_op=torch.nn.BatchNorm3d,
                           norm_op_kwargs={}, dropout_op=None, nonlin=torch.nn.ReLU, nonlin_kwargs={"inplace": True},
                           return_skips=True)
    pre = o["prefix"] + "UNet.encoder."
    enc.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
    projections = torch.nn.ModuleList()
    for key, channels, _ in o["projections"]:
        p = torch.nn.Conv3d(channels, spec.lattices[0].width, kernel_size=1)
        p.load_state_dict({"weight": sd[f"{o['prefix']}{key}.weight"], "bias": sd[f"{o['prefix']}{key}.bias"]}, strict=True)
        projections.append(p)
    model = torch.nn.Module()
    model.encoder, model.projections = enc, projections
    return model.eval().to(device, dtype)


def _las(image):
    """A SimpleITK image as nibabel would hand it over reoriented to LAS: the array in nibabel's
    (i, j, k) order and its RAS affine - the form upstream's DataFolder (MONAI / nibabel) reads."""
    import SimpleITK as sitk
    from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
    arr = np.asarray(sitk.GetArrayFromImage(image), np.float32).transpose(2, 1, 0)     # (k, j, i) -> (i, j, k)
    D = np.asarray(image.GetDirection(), float).reshape(3, 3) * np.asarray(image.GetSpacing(), float)
    A = np.eye(4); A[:3, :3] = D; A[:3, 3] = image.GetOrigin()
    A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ A                                            # LPS -> RAS
    t = ornt_transform(io_orientation(A), axcodes2ornt(("L", "A", "S")))
    return np.ascontiguousarray(apply_orientation(arr, t), np.float32), A @ inv_ornt_aff(t, arr.shape), A


def prepare(spec, image, model=None) -> Prepared:
    """Upstream's DataFolder on a SimpleITK image (any haversack input), on the CPU - one
    interpolation, the same bits on every machine."""
    import torch
    import torch.nn.functional as F
    o = spec.options
    arr, aff, _ = _las(image)
    sp = np.abs(np.diag(aff)[:3])                                  # upstream reads spacing off the diagonal
    h, w, dd = arr.shape
    sy, sx, sz = o["spacing_mm"][1], o["spacing_mm"][0], o["spacing_mm"][2]
    tgt = [int(h * sp[1] / sy), int(w * sp[0] / sx), int(dd * sp[2] / sz)]   # upstream's x/y swap, kept
    if any(s > o["max_resampled"] for s in tgt):
        raise InputError(f"resampled to {tgt}: past {o['max_resampled']} along an axis - upstream's own rule refuses "
                         "this series (a non-axial or very large acquisition)")
    x = F.interpolate(torch.as_tensor(arr)[None, None], size=tgt, mode="trilinear", align_corners=False)[0].permute(0, 3, 2, 1)
    lo_hu, hi_hu = o["window_hu"]
    x = x.clamp(lo_hu, hi_hu)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    nz = x[0] > 0
    lo, hi = [], []
    for ax in range(3):
        idx = torch.nonzero(nz.any(dim=tuple(a for a in range(3) if a != ax))).flatten()
        if not len(idx):
            raise InputError("nothing above the window's floor in this image: no body to encode")
        lo.append(int(idx.min())); hi.append(int(idx.max()))
    lo = [max(l - e, 0) for l, e in zip(lo, o["crop_margin"])]
    hi = [min(m + e, s) for m, e, s in zip(hi, o["crop_margin"], x.shape[1:])]
    x = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    m = o["pad_multiple"]
    want = [int(np.ceil(max(s, mn) / m) * m) for s, mn in zip(x.shape[1:], o["min_shape"])]
    pad = []
    for s, t in zip(reversed(x.shape[1:]), reversed(want)):
        pad += [0, t - s]
    base = F.pad(x, pad)[None].float()
    grid = _grid(aff, (h, w, dd), tgt, lo, base.shape[2:])
    data_box = ((0, 0, 0), tuple(int(b) - int(a) for a, b in zip(lo, hi)))
    extra = {"crop": {"lo": lo, "hi": hi}, "resample_target": tgt}
    size = image.GetSize()
    Dd = np.asarray(image.GetDirection(), float).reshape(3, 3) * np.asarray(image.GetSpacing(), float)
    extra["input_grid"] = {"shape": [int(v) for v in size], "directions": Dd.T.round(9).tolist(),
                           "origin": np.round(image.GetOrigin(), 9).tolist(), "space": "left-posterior-superior"}
    return Prepared(base, grid, data_box, extra)


def _grid(aff, image_shape, tgt, lo, grid_shape) -> dict:
    """The model grid in the world (LPS mm, a direction row per array axis). Model axes (Z, Y, X)
    are the LAS image's axes (2, 1, 0); one model voxel is in/out image voxels
    (``align_corners=False``: output o reads input (o + 0.5) * in / out - 0.5); the first sits at
    the crop's corner."""
    h, w, dd = image_shape
    step = [dd / tgt[2], w / tgt[1], h / tgt[0]]
    first = [(lo[2] + 0.5) * step[2] - 0.5, (lo[1] + 0.5) * step[1] - 0.5, (lo[0] + 0.5) * step[0] - 0.5]   # image (i, j, k)
    ras_to_lps = np.array([-1.0, -1.0, 1.0])
    return {"shape": [int(v) for v in grid_shape],
            "directions": [(aff[:3, 2] * step[0] * ras_to_lps).tolist(), (aff[:3, 1] * step[1] * ras_to_lps).tolist(),
                           (aff[:3, 0] * step[2] * ras_to_lps).tolist()],
            "origin": ((aff @ np.array([*first, 1.0]))[:3] * ras_to_lps).tolist()}


def run(spec, model, prepared: Prepared, device, dtype, slab: int = 16) -> list[np.ndarray]:
    """Tokens per lattice, (N, C) each in C order over the lattice, deep to fine. The encoder's
    full-resolution stages run ``slab`` slices at a time: exact, not an approximation (BatchNorm in
    eval is a per-channel affine, and the leading stages have z-stride 1, so a slice sees only
    its neighbors within the convolutions' half-widths - the halo is read from the network)."""
    import torch
    x = prepared.tensor.to(device, dtype)
    enc = model.encoder
    with torch.inference_mode():
        if slab:
            st = enc.stages
            k = 0
            while k < len(st) and all(m.stride[0] == 1 for m in st[k].modules() if isinstance(m, torch.nn.Conv3d)):
                k += 1
            halo = sum((m.kernel_size[0] - 1) // 2 for m in st[:k].modules() if isinstance(m, torch.nn.Conv3d))
            D, parts = x.shape[2], []
            for z0 in range(0, D, slab):
                z1 = min(z0 + slab, D); a, b = max(z0 - halo, 0), min(z1 + halo, D)
                y = x[:, :, a:b]
                for s in st[:k]:
                    y = s(y)
                parts.append(y[:, :, z0 - a:y.shape[2] - (b - z1)])
            y = torch.cat(parts, 2); del parts
            skips = []
            for s in st[k:]:
                y = s(y); skips.append(y)
        else:
            skips = enc(x)
        toks = [p(skips[-which])[0].flatten(1).T.float().cpu().numpy()
                for p, (_, _, which) in zip(model.projections, spec.options["projections"])]
    if not all(np.isfinite(t).all() for t in toks) or any(np.abs(t).max() == 0 for t in toks):
        raise RuntimeError("the encoder returned zeros or non-finite tokens: out of memory on this device? "
                           "(try a smaller --slab, or --device cpu)")
    return toks
