"""Restrict inference to the body: the largest single speedup available for CT, off by default.

On a chest CT the labeled anatomy occupies about a third of the volume; the rest is air and
table, and nnU-Net's sliding window tiles all of it. A body envelope - the bounding box of the
patient with a margin - cuts the 1.5 mm patch count from 175 to 42-63 per model on the chest
measured in docs/backend-decision.md.

The envelope comes from a HU threshold, not from a model: air is below -500 HU in any CT, the
largest connected component above it is the patient, and that outline includes skin and fat,
which is the context the fine model's boundary patches need. A coarse *model* could do this
too, but it inherits that model's blind spots; the threshold cannot miss a body.

The envelope is a speedup, not a no-op. Cropping re-tiles the sliding window, and labels move
with the tiles: measured 2026-09-11 against whole-volume inference, a chest CT's 20 mm crop
moved 0.1 % of voxels, all within a few voxels of the skin, and 17 % of TotalSegmentator's
``body_extremities`` (``worth_cropping`` has the numbers). Three rules keep it honest: a crop
must save network tiles or the whole grid runs (``worth_cropping``), a crop narrower than the
patch is grown with real voxels rather than padded with tissue (``at_least``), and ``0`` means
no envelope at every door (``envelope_margin``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR_HU = -500.0


@dataclass(frozen=True)
class Envelope:
    """Half-open voxel range ``[start, stop)`` on the grid the mask was computed on, (Z, Y, X)."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]
    shape: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(int(a), int(b)) for a, b in zip(self.start, self.stop))

    @property
    def extent(self) -> tuple[int, int, int]:
        return tuple(int(b) - int(a) for a, b in zip(self.start, self.stop))

    @property
    def fraction(self) -> float:
        return float(np.prod(np.array(self.stop) - np.array(self.start)) / np.prod(self.shape))

    def is_whole(self) -> bool:
        return all(a == 0 for a in self.start) and tuple(self.stop) == tuple(self.shape)


def body_mask(hu_zyx: np.ndarray, *, threshold: float = AIR_HU, largest_component: bool = True) -> np.ndarray:
    """Voxels that are not air, restricted to the largest connected component (the patient).

    Works on the coarse grid the fine model will run on, so the mask is cheap (the 3 mm grid
    of a chest is 6.6 M voxels). Table and cables are usually thin or disconnected from the
    body and drop out with the component filter; if not, they only enlarge the box slightly.
    """
    mask = np.asarray(hu_zyx) > threshold
    if not mask.any():
        return mask
    if largest_component:
        from scipy import ndimage
        labels, n = ndimage.label(mask)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = labels == int(sizes.argmax())
    return mask


def envelope_of(mask_zyx: np.ndarray, *, margin_voxels) -> Envelope:
    """Bounding box of the mask, padded by ``margin_voxels`` per axis, clipped to the grid.

    An empty mask yields the whole grid: the safe direction is to run the full volume, never
    an empty slab.
    """
    shape = tuple(int(s) for s in mask_zyx.shape)
    if not mask_zyx.any():
        return Envelope((0, 0, 0), shape, shape)
    m = np.broadcast_to(np.asarray(margin_voxels, dtype=np.int64), (3,))
    idx = np.nonzero(mask_zyx)
    start = tuple(int(max(0, i.min() - mm)) for i, mm in zip(idx, m))
    stop = tuple(int(min(n, i.max() + 1 + mm)) for i, n, mm in zip(idx, shape, m))
    return Envelope(start, stop, shape)


def at_least(env: Envelope, extent_zyx) -> Envelope:
    """Grow each axis of ``env`` to at least ``extent_zyx`` voxels, clipped to the grid.

    ``extent_zyx`` is the network's patch on this grid. nnU-Net's sliding window pads anything
    narrower than a patch with 0 *after* normalization, which is the model's mean foreground
    intensity - +120 HU for CADS's head model, -89 HU (fat) for TotalSegmentator's breasts
    model - so a crop narrower than the patch showed the network tissue where the image has
    air, a margin's width outside the skin. Growing the crop fills those voxels from the image
    instead, and costs nothing: along an axis no longer than the patch the window takes one
    step either way, and the grown crop is the same shape as the padded one. Where the grid
    itself is narrower than the patch, the crop spans that axis and pads exactly as
    whole-volume inference (and upstream) does.

    The growth is centered on the box, as the padding is, and shifted to stay on the grid.
    """
    start, stop = list(env.start), list(env.stop)
    for ax in range(3):
        want = min(int(extent_zyx[ax]), env.shape[ax])
        short = want - (stop[ax] - start[ax])
        if short > 0:
            a = min(max(0, start[ax] - short // 2), env.shape[ax] - want)
            start[ax], stop[ax] = a, a + want
    return Envelope(tuple(start), tuple(stop), env.shape)


def worth_cropping(env: Envelope, *, saving: float | None = None,
                   min_saving: float = 0.05) -> Envelope:
    """Collapse a crop that saves too little network work back to the whole grid.

    ``saving`` is the fraction of the network's work the crop removes; the pipeline passes it
    in tiles (``1 - tiles(crop) / tiles(grid)``, see :func:`haversack.network.window_tiles`).
    Without it, the box's volume stands in, which is what the pipeline used until 2026-09-11 -
    and volume is the wrong measure: the sliding window steps half a patch, so a crop that
    removes a third of the volume can still need every tile. On a chest-abdomen-pelvis CT at
    TotalSegmentator's 128^3 patch, the 10-40 mm crops needed all 54 of the whole volume's.

    A crop is never free, because it re-tiles the window, and the labels move with the tiles:
    not only near-ties at the new seams, but whole stretches of a class whose boundary is a
    judgment of context. On a chest CT the 20 mm envelope relabeled 17 % of TotalSegmentator's
    ``body_extremities`` as trunk with no padding involved (0 mm: 5 %; 40 mm: 3 %; not
    monotone, so no margin buys it off). Below ``min_saving`` the crop is not worth that, and
    the whole grid runs: the result is then identical to whole-volume inference.
    """
    if saving is None:
        saving = 1.0 - env.fraction
    if env.is_whole() or saving < min_saving:
        return Envelope((0, 0, 0), env.shape, env.shape)
    return env


def envelope_margin(envelope_mm) -> float | None:
    """What an ``envelope_mm`` asks for: a margin in mm, or ``None`` for the whole volume.

    ``None``, ``0`` and anything below run the whole volume. Until 2026-09-11 that held only on
    the command line (``--envelope 0``); the Python API and the server read 0 as a crop flush to
    the skin, and CADS scored 0.64-0.82 Dice on face, head muscles and mammary glands against
    upstream that way while the whole volume scored 0.95-1.0. One number had two meanings, so
    every door now reads it here: ``segment()`` calls this first, and ``Segmenter``, the server
    and the Modal worker all reach it through ``segment()``.
    """
    if envelope_mm is None:
        return None
    mm = float(envelope_mm)
    if not np.isfinite(mm):
        from .errors import InputError
        raise InputError(f"envelope_mm must be a margin in mm, or 0 / None for the whole volume; "
                         f"got {envelope_mm!r}")
    return mm if mm > 0 else None


def margin_in_voxels(margin_mm: float, spacing_zyx) -> tuple[int, int, int]:
    return tuple(int(np.ceil(float(margin_mm) / float(s))) for s in spacing_zyx)


def label_roi(labels_zyx: np.ndarray, classes, *, margin_voxels) -> Envelope:
    """Bounding box of any of ``classes`` in a labelmap, padded by ``margin_voxels``.

    The cascade counterpart of the body envelope: a coarse model labels everything, and the
    fine model runs only inside the box of the organ(s) it refines. An absent class yields the
    whole grid, so a coarse-model miss (e.g. adrenal on some chests) falls back to the full
    volume rather than an empty crop - the same fail-safe as the body envelope.
    """
    want = np.isin(labels_zyx, np.asarray(list(classes), dtype=labels_zyx.dtype))
    return envelope_of(want, margin_voxels=margin_voxels)


def otsu_threshold(values: np.ndarray, *, bins: int = 256) -> float:
    """Otsu's between-class-variance threshold over a flat array (256-bin histogram).

    Separates a bimodal histogram - background/air vs tissue - at the value that maximizes
    the variance between the two classes. This is the data-driven counterpart to the CT air
    cut: it needs no fixed intensity, so it works on per-image-standardized inputs (ZScore
    MRI) where a dataset-derived HU threshold has no meaning.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    low, high = float(v.min()), float(v.max())
    if high <= low:
        return low
    hist, edges = np.histogram(v, bins=bins, range=(low, high))
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return low
    p /= total
    mids = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)                         # weight of the "below" class at each split
    m0 = np.cumsum(p * mids)                  # its first moment
    mt = m0[-1]                               # global mean
    denom = w0 * (1.0 - w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mt * w0 - m0) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = -1.0
    return float(mids[int(np.argmax(sigma_b2))])


def body_threshold(x_zyx: np.ndarray, *, normalization_schemes, intensity_properties) -> float:
    """The value that separates the patient from surrounding air on a *normalized* model input.

    CT normalization uses dataset statistics, so the air cut (``AIR_HU``, or the channel's
    0.5th foreground percentile when that is higher) maps to a fixed value in normalized units -
    coherent, and the validated CT behavior. Per-image normalizations (ZScore for MRI) standardize
    each volume by its own mean/std, so a dataset-derived HU threshold is meaningless there; the
    air/tissue split is taken from the image itself with Otsu. The largest connected component
    above the threshold is still the body (see :func:`body_mask`), so a stray bright region does
    not move the box.
    """
    schemes = [str(s) for s in (normalization_schemes or [])]
    props = intensity_properties or {}
    if any("CT" in s for s in schemes) and "percentile_00_5" in props:
        return (max(AIR_HU, props["percentile_00_5"]) - props["mean"]) / max(props["std"], 1e-8)
    return otsu_threshold(x_zyx)
