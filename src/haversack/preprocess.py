"""Source image -> the model's input frame: canonical orientation, forward resample
(the frozen fork resampler, scipy-exact), nnU-Net normalization.

Two orders, one per lineage. TotalSegmentator resamples the image (``change_spacing``) and
nnU-Net then normalizes it: :func:`to_model_grid` + :func:`normalize_for`, which share one
resample between the models at a spacing. A native nnU-Net model was trained on nnU-Net's own
preprocessing, which crops to the nonzero region, normalizes, and only then resamples (with
separate-z for an anisotropic image): :func:`to_nnunet_input` (2026-09-28; before that date
native models were resampled first and normalized after, like TotalSegmentator's).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from .grid import Grid

from .errors import UnsupportedModel
from .frame import Frame


def load_canonical(path):
    """Read through SimpleITK, reoriented to RAS: ``(array (Z, Y, X), geometry, orientation)``."""
    from .io import read
    return read(path)


# nnU-Net's preprocessor resamples training data with a cubic spline, and TotalSegmentator
# matched that at inference until v2.18 dropped the default to linear for speed. Cubic is what
# the models were trained on, and the mismatch grows with the downsampling factor (0.65 mm ->
# 3 mm is 4.6x, where linear reads only the 2 nearest samples per axis). On the GPU resampler
# cubic costs ~2.5-3.4 s instead of scipy's 16.6 s, so the speed argument does not apply here.
# Measured: order 1 vs 3 on the same weights moves small structures by ~1.4 % Dice
# (gallbladder 0.783). See medseg/docs/resampler-parity-finding.md.
DEFAULT_RESAMPLING_ORDER = 3


def forward_resample(data_zyx: np.ndarray, spacing_zyx, new_spacing_zyx, *, convention: str = "corner",
                     order: int = DEFAULT_RESAMPLING_ORDER, device="auto", out_dtype=np.int32):
    """TotalSegmentator's ``change_spacing`` semantics on :mod:`haversack.resample`: new shape =
    round(shape * spacing / new_spacing), scipy.zoom corner rule, edge mode, no anti-aliasing,
    then the dtype conversion TS applies (``astype`` = truncation, not rounding)."""
    from .resample import resample_data, target_shape
    new_shape = target_shape(data_zyx.shape, spacing_zyx, new_spacing_zyx)
    out = resample_data(data_zyx, new_shape, convention=convention, order=order, mode="nearest",
                        device=device, out_dtype=out_dtype)
    return out, new_shape


#: nnU-Net's default resampling (``resample_data_or_seg_to_shape``), as nearly every plans file
#: states it: cubic in-plane for the input, linear for the logits, nearest along a separate z.
NNUNET_RESAMPLING = {"data": {"order": 3, "order_z": 0, "force_separate_z": None},
                     "probabilities": {"order": 1, "order_z": 0, "force_separate_z": None}}


def nnunet_resampling_from_config(configuration: dict, kind: str, name: str = "model") -> dict:
    """``kind``'s resampling from a plans configuration: ``{"order", "order_z",
    "force_separate_z"}``. Raises ``UnsupportedModel`` for a resampling function other than
    nnU-Net's default one, which is the only one haversack reproduces."""
    if kind not in NNUNET_RESAMPLING:
        raise ValueError(f"kind must be one of {tuple(NNUNET_RESAMPLING)}; got {kind!r}")
    fn = configuration.get(f"resampling_fn_{kind}") or "resample_data_or_seg_to_shape"
    if fn != "resample_data_or_seg_to_shape":
        raise UnsupportedModel(f"{name}: plans resample {kind} with {fn!r}; haversack reproduces "
                               "only nnU-Net's resample_data_or_seg_to_shape")
    kw = dict(configuration.get(f"resampling_fn_{kind}_kwargs") or {})
    out = dict(NNUNET_RESAMPLING[kind])
    out.update({k: kw[k] for k in out if k in kw})
    return out


def nnunet_resampling(model, kind: str) -> dict:
    """``model``'s plans resampling for ``kind`` (``"data"`` or ``"probabilities"``); nnU-Net's
    default for a model object that does not state one."""
    get = getattr(model, "resampling", None)
    return get(kind) if callable(get) else dict(NNUNET_RESAMPLING[kind])


def nonzero_box(data_zyx: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """nnU-Net's crop-to-nonzero box: ``(lo, hi)`` half-open, (Z, Y, X).

    nnU-Net crops every case to the bounding box of its nonzero voxels *before* resampling
    (``crop_to_nonzero``: ``data != 0``, holes filled, then the bbox). On CT this is usually
    the whole image; on MRI, where the background outside the acquired volume is exactly 0,
    it removes a real border. An all-zero image yields the whole grid rather than an empty
    box - the same fail-safe direction as the body envelope.
    """
    mask = np.asarray(data_zyx) != 0
    shape = tuple(int(s) for s in mask.shape)
    if not mask.any():
        return (0, 0, 0), shape
    # nnU-Net fills holes in the mask before taking its box; a filled hole lies inside the box
    # already, so the box is the same without it (the fill cost ~2 s on 78 M voxels, 2026-09-28).
    lo, hi = [], []
    for ax in range(mask.ndim):
        present = np.nonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != ax)))[0]
        lo.append(int(present[0]))
        hi.append(int(present[-1]) + 1)
    return tuple(lo), tuple(hi)


def normalize(data: np.ndarray, schemes, props, *, use_mask_for_norm=None, seg=None) -> np.ndarray:
    """Normalize a single-channel image exactly as nnU-Net would, by delegating to nnU-Net's own
    normalization classes (named in ``plans``) - CTNormalization, ZScoreNormalization, etc.

    ``schemes`` is the plans' ``normalization_schemes`` (class names, e.g. "ZScoreNormalization");
    ``props`` the channel's ``foreground_intensity_properties`` (CT needs it, ZScore does not).
    ``use_mask_for_norm`` (from plans) selects masked ZScore for e.g. brain MRI; without a
    provided ``seg`` the nonzero region is used, matching nnU-Net's mask definition.

    Single channel only (Tier A); multi-channel MRI is a later step.
    """
    from nnunetv2.preprocessing.normalization import default_normalization_schemes as N
    names = list(schemes)
    if len(names) != 1:
        raise UnsupportedModel(f"multi-channel normalization {names!r} not supported yet (single channel only)")
    cls = getattr(N, names[0], None)
    if cls is None:
        raise UnsupportedModel(f"unknown normalization scheme {names[0]!r}")
    umn = bool(use_mask_for_norm[0]) if isinstance(use_mask_for_norm, (list, tuple)) else bool(use_mask_for_norm)
    # a fresh C-contiguous copy: run() mutates in place (must not touch the caller's array), and
    # mean/std on a non-contiguous view can round differently. np.array(copy=True) guarantees it.
    x = np.array(data, dtype=np.float32, order="C")[None]                # (C=1, Z, Y, X)
    if umn and seg is None:
        seg = np.where(x != 0, 0, -1).astype(np.int8)     # nnU-Net's nonzero mask: >= 0 is inside
    norm = cls(use_mask_for_norm=umn, intensityproperties=dict(props or {}), target_dtype=np.float32)
    return norm.run(x, seg)[0]


@dataclass(frozen=True)
class ResampledGrid:
    """A source volume cropped and resampled to a model spacing, with NO normalization applied.

    Deliberately a different type from a network input. nnU-Net's normalization is PER MODEL -
    CT clips to that dataset's foreground percentiles and z-scores by its foreground mean/std,
    all read from its own plans - so several models at one spacing can share this resample but
    must not share a normalized array. Keeping the shareable thing normalization-free, and
    making it un-feedable to a network, is what stops that mix-up being expressible.
    """

    data_zyx: np.ndarray
    frame: Frame


def normalization_fingerprint(model) -> tuple:
    """What distinguishes one model's normalization from another's.

    The five parts of TotalSegmentator's ``total`` illustrate the spread: the organs model
    clips CT to [-1024, 276] and z-scores by mean -370 / std 437, while the ribs model clips to
    [-110, 1302] around mean 292 / std 262. Feeding one model's normalization to another
    flattens everything above the first model's upper clip, which is silent and severe.
    """
    props = model.intensity_properties(0) or {}
    numeric = tuple(sorted((str(k), float(v)) for k, v in props.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)))
    umn = model.use_mask_for_norm
    umn = bool(umn[0]) if isinstance(umn, (list, tuple)) else bool(umn)
    return (tuple(model.normalization_schemes), umn, numeric)


def to_model_grid(data_zyx, geometry, spacing_zyx, *, convention: str = "corner", device="auto",
                  order: int = DEFAULT_RESAMPLING_ORDER, original_orientation: str = "RAS",
                  crop_to_nonzero: bool = False, box=None, truncate: bool = True) -> ResampledGrid:
    """Canonical (RAS) array + geometry -> a ``ResampledGrid`` at ``spacing_zyx``.

    ``truncate`` applies TotalSegmentator's own ``astype(int32)`` to the resampled image (it
    truncates, so a TS-lineage model sees exactly what upstream feeds it). Every other model
    keeps the image's values (float32): a PET SUV, a scaled MRI or an ADC map would lose its
    fractional part to it - until 2026-09-28 every lineage was truncated, native nnU-Net models
    (MOOSE's PET included) too.

    ``box`` - ``((z0, y0, x0), (z1, y1, x1))``, source indices, end exclusive - crops the source
    to that box before anything else: a TotalSegmentator cascade's final stage sees only the box
    its crop stage found, as upstream crops the image before resampling it (2026-09-22). It is
    recorded exactly as the nonzero crop is, so the un-crop stays implicit in the mapping.

    Optionally crops to the nonzero box first (``crop_to_nonzero=True``, nnU-Net-native), then
    resamples with the caller's convention (corner = TotalSegmentator's ``change_spacing``,
    center = skimage / nnU-Net's own resampler). Arrays stay in (Z, Y, X) throughout - SimpleITK
    hands them over that way, so there are no transposes.

    Everything here depends only on the geometry and the target spacing, never on which model
    asked, which is what makes the result safe to share between models at one spacing.

    The crop is recorded as ``Frame.model_source`` (a sub-grid with the crop offset as its
    origin), so output grids keep referring to the full source and the un-crop is implicit in
    the mapping - nothing has to be pasted back afterwards.
    """
    spacing_src_zyx = tuple(float(s) for s in geometry.spacing_zyx)
    source, data_zyx, model_source = _crop_source(data_zyx, geometry, box=box,
                                                  crop_to_nonzero=crop_to_nonzero)
    res_zyx, _ = forward_resample(data_zyx, spacing_src_zyx, tuple(spacing_zyx),
                                  convention=convention, device=device, order=order,
                                  out_dtype=np.int32 if truncate else np.float32)
    frame = Frame(source=source, model_shape=tuple(res_zyx.shape), model_spacing=tuple(spacing_zyx),
                  convention=convention, canonical=geometry, original_orientation=original_orientation,
                  model_source=model_source)
    return ResampledGrid(res_zyx, frame)


def _crop_source(data_zyx, geometry, *, box=None, crop_to_nonzero=False):
    """The source grid, the cropped array (float32) and the crop's ``model_source`` sub-grid
    (None when nothing was cropped): ``box`` first, then the nonzero box inside it."""
    spacing_src_zyx = tuple(float(s) for s in geometry.spacing_zyx)
    source_shape_zyx = tuple(int(s) for s in data_zyx.shape)
    source = Grid(source_shape_zyx, spacing_src_zyx, (0.0, 0.0, 0.0))
    data_zyx = np.asarray(data_zyx, dtype=np.float32)
    offset = (0, 0, 0)
    if box is not None:
        lo, hi = (tuple(int(v) for v in box[0]), tuple(int(v) for v in box[1]))
        if any(not 0 <= l < h <= n for l, h, n in zip(lo, hi, source_shape_zyx)):
            raise ValueError(f"crop box {lo}..{hi} is empty or leaves the source {source_shape_zyx}")
        data_zyx = data_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        offset = lo
    if crop_to_nonzero:
        lo, hi = nonzero_box(data_zyx)
        if (lo, hi) != ((0, 0, 0), tuple(data_zyx.shape)):
            data_zyx = data_zyx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            offset = tuple(o + l for o, l in zip(offset, lo))
    model_source = None
    if tuple(data_zyx.shape) != source_shape_zyx:
        model_source = Grid(tuple(int(s) for s in data_zyx.shape), spacing_src_zyx,
                            tuple(float(source.index_to_mm(offset)[a]) for a in range(3)))
    return source, data_zyx, model_source


def to_nnunet_input(data_zyx, geometry, model, *, device="auto", original_orientation: str = "RAS",
                    box=None) -> tuple[torch.Tensor, Frame]:
    """A native nnU-Net model's network input, in nnU-Net's own order (``run_case_npy``):
    crop to the nonzero box, normalize with this model's plans, then resample.

    The order matters. nnU-Net normalizes first ("normalization MUST happen before resampling"):
    a CT clip then applies before the cubic spline, not after its overshoot, and a z-score's
    statistics and mask (the nonzero region with its holes filled, ``crop_to_nonzero``) come
    from the image at its own resolution. The resample follows the plans' ``resampling_fn_data``
    (:func:`nnunet_resampling`): nnU-Net's shape rule (:func:`~haversack.resample.
    compute_nnunet_shape`), the voxel-center grid, and separate-z where nnU-Net would use it
    (:func:`~haversack.resample.separate_z_axis`: an image, or a target, whose coarsest spacing
    is more than 3 times its finest).

    The result is normalized for ``model`` alone, so it is shared only between models whose
    normalization and resampling are the same (the caller's cache key says so). Returns
    ``(x (1, Z, Y, X) float32 CPU, Frame)``; ``x`` carries the normalization fingerprint that
    ``predict_into`` checks.
    """
    from .resample import compute_nnunet_shape, resample_data, separate_z_axis
    source, cropped, model_source = _crop_source(data_zyx, geometry, box=box, crop_to_nonzero=True)
    umn = model.use_mask_for_norm
    umn = bool(umn[0]) if isinstance(umn, (list, tuple)) else bool(umn)
    seg = None
    if umn:                                   # nnU-Net's mask: the nonzero region, holes filled
        from scipy.ndimage import binary_fill_holes
        seg = np.where(binary_fill_holes(cropped != 0), 0, -1).astype(np.int8)[None]
    x = normalize(cropped, model.normalization_schemes, model.intensity_properties(0),
                  use_mask_for_norm=umn, seg=seg)
    r = nnunet_resampling(model, "data")
    spacing_src = tuple(float(s) for s in geometry.spacing_zyx)
    new_shape = compute_nnunet_shape(cropped.shape, spacing_src, model.spacing_zyx)
    axis = separate_z_axis(spacing_src, model.spacing_zyx, r["force_separate_z"])
    res = resample_data(x, new_shape, convention="center", order=int(r["order"]), mode="nearest",
                        device=device, out_dtype=np.float32, separate_z_axis=axis,
                        order_z=int(r["order_z"]))
    frame = Frame(source=source, model_shape=tuple(res.shape), model_spacing=tuple(model.spacing_zyx),
                  convention="center", canonical=geometry, original_orientation=original_orientation,
                  model_source=model_source)
    t = torch.from_numpy(np.ascontiguousarray(res))[None]
    t._haversack_normalization = normalization_fingerprint(model)
    return t, frame


def normalize_for(grid: ResampledGrid, model) -> torch.Tensor:
    """This model's network input: ``grid`` normalized with THIS model's statistics.

    Returns a fresh tensor - ``normalize`` copies, so the grid stays unnormalized and reusable -
    carrying a fingerprint of the normalization applied, which the consumer checks.
    """
    x_zyx = normalize(grid.data_zyx, model.normalization_schemes, model.intensity_properties(0),
                      use_mask_for_norm=model.use_mask_for_norm)
    x = torch.from_numpy(np.ascontiguousarray(x_zyx))[None]
    x._haversack_normalization = normalization_fingerprint(model)
    return x


def to_model_frame(data_zyx, geometry, model, *, convention: str = "corner", device="auto",
                   order: int = DEFAULT_RESAMPLING_ORDER, original_orientation: str = "RAS",
                   crop_to_nonzero: bool = False, truncate: bool = True) -> tuple[torch.Tensor, Frame]:
    """Canonical (RAS) array + geometry -> ``(x (1, Z, Y, X) float32 CPU, Frame)`` for ``model``.

    ``to_model_grid`` then ``normalize_for``, for a single model. Callers that run several
    models at one spacing should use those two directly, sharing the grid and normalizing per
    model; see ``pipeline.segment``.
    """
    grid = to_model_grid(data_zyx, geometry, model.spacing_zyx, convention=convention, device=device,
                         order=order, original_orientation=original_orientation,
                         crop_to_nonzero=crop_to_nonzero, truncate=truncate)
    return normalize_for(grid, model), grid.frame
