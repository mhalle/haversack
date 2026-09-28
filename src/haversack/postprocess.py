"""TotalSegmentator's label postprocessing, on the grid the labels are on.

Upstream cleans some results after the network (``nnunet.py``, ``_postprocess_multilabel`` and
the ``remove_outside`` step): ``body`` keeps the largest connected piece of ``body_trunc`` and
drops ``body_extremities`` pieces of 50 000 mm3 or less, ``heartchambers_highres`` zeroes
everything outside its crop model's heart, aorta and inferior vena cava dilated by 10 mm, and
``--remove_small_blobs`` drops every class's pieces of 200 mm3 or less. Until 2026-09-28
haversack ran none of it.

A task states its steps as data (``TaskSpec.postprocess``, ``CascadeStep.postprocess``,
``CascadeStep.remove_outside_*``), and :func:`apply` runs them:

* ``{"op": "keep_largest", "classes": [...]}`` - per class, keep its largest 6-connected piece
  (``keep_largest_blob_multilabel``; at equal sizes the piece reached first in a C-order scan,
  as ``ndimage.label`` numbers them and ``np.argmax`` picks).
* ``{"op": "remove_small", "classes": [...] | "all", "max_mm3": v}`` - zero every 6-connected
  piece of those classes of ``v / voxel volume`` voxels or fewer
  (``remove_small_blobs_multilabel``; a piece is the voxels of one class that touch).

Sizes are in mm3 and converted with the grid's own voxel volume, so a threshold means the same
on the model grid (upstream's default) and on the output grid (here, and upstream's smooth
labels): the cut falls at the same physical size, at the output grid's resolution.
"""
from __future__ import annotations

import numpy as np

OPS = ("keep_largest", "remove_small")


def _ndimage():
    """scipy, imported when a step runs: the task registry reads this module's
    :func:`check_ops` on a lean install, which has no scipy."""
    from scipy import ndimage
    return ndimage


def _faces():
    """nnU-Net's and upstream's connectivity: faces only (``ndimage.label``'s default)."""
    return _ndimage().generate_binary_structure(3, 1)


def check_ops(ops, where: str = "postprocess") -> tuple:
    """``ops`` as a tuple of dicts, refused on load if a step is malformed."""
    out = []
    for i, op in enumerate(ops or ()):
        kind = op.get("op") if isinstance(op, dict) else None
        if kind not in OPS:
            raise ValueError(f"{where}: step {i + 1} op {kind!r} is not one of {OPS}")
        classes = op.get("classes")
        if not (classes == "all" or (isinstance(classes, (list, tuple)) and classes
                                     and all(int(c) > 0 for c in classes))):
            raise ValueError(f"{where}: step {i + 1} classes {classes!r} - a list of label values > 0"
                             + (' or "all"' if kind == "remove_small" else ""))
        if kind == "keep_largest" and classes == "all":
            raise ValueError(f"{where}: step {i + 1} keep_largest names its classes")
        if kind == "remove_small" and not float(op.get("max_mm3", -1)) >= 0:
            raise ValueError(f"{where}: step {i + 1} remove_small needs max_mm3 >= 0")
        out.append(dict(op))
    return tuple(out)


def keep_largest(labels: np.ndarray, classes) -> np.ndarray:
    """Per class, keep its largest face-connected piece; in place, returned."""
    for c in classes:
        mask = labels == c
        if not mask.any():
            continue
        pieces, n = _ndimage().label(mask, structure=_faces())
        if n <= 1:
            continue
        sizes = np.bincount(pieces.ravel(), minlength=n + 1)
        sizes[0] = 0
        labels[mask & (pieces != int(np.argmax(sizes)))] = 0
    return labels


def _pieces(masked: np.ndarray):
    """Face-connected pieces of equal nonzero values: cc3d when installed (one pass for every
    class), else one ``ndimage.label`` per value. The two give the same pieces."""
    try:
        import cc3d
    except ImportError:
        cc3d = None
    if cc3d is not None:
        return cc3d.connected_components(masked, connectivity=6, return_N=True)
    out = np.zeros(masked.shape, dtype=np.int64)
    n = 0
    for v in np.unique(masked):
        if v == 0:
            continue
        lab, k = _ndimage().label(masked == v, structure=_faces())
        out[lab > 0] = lab[lab > 0] + n
        n += k
    return out, n


def remove_small(labels: np.ndarray, classes, max_voxels: float) -> np.ndarray:
    """Zero every face-connected piece of ``classes`` (a list, or ``"all"``) of ``max_voxels``
    voxels or fewer; in place, returned."""
    if classes == "all":
        masked = labels
    else:
        masked = np.where(np.isin(labels, np.asarray(list(classes), dtype=labels.dtype)), labels, 0)
    pieces, n = _pieces(np.ascontiguousarray(masked))
    if n == 0:
        return labels
    sizes = np.bincount(pieces.ravel(), minlength=n + 1)
    drop = sizes <= max_voxels
    drop[0] = False
    labels[drop[pieces]] = 0
    return labels


def remove_outside(labels: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """Zero every voxel outside ``mask`` dilated ``iterations`` times with the face structure -
    upstream's ``remove_outside_of_mask`` (``binary_dilation(mask, iterations=addon)``); in
    place, returned. ``iterations`` 0 uses the mask as it is."""
    keep = _ndimage().binary_dilation(mask, structure=_faces(), iterations=int(iterations)) if iterations > 0 \
        else mask.astype(bool)
    labels[~keep] = 0
    return labels


def dilation_iterations(mm: float, spacing_zyx) -> int:
    """Upstream's ``int(remove_outside_dilation / mean(voxel spacing))``."""
    return int(float(mm) / float(np.mean([float(s) for s in spacing_zyx])))


def apply(labels: np.ndarray, ops, spacing_zyx) -> list[dict]:
    """Run ``ops`` on ``labels`` in place; returns what ran, for the provenance."""
    voxel = float(np.prod([float(s) for s in spacing_zyx]))
    ran = []
    for op in ops or ():
        if op["op"] == "keep_largest":
            keep_largest(labels, [int(c) for c in op["classes"]])
            ran.append({"op": "keep_largest", "classes": [int(c) for c in op["classes"]]})
        else:
            classes = op["classes"] if op["classes"] == "all" else [int(c) for c in op["classes"]]
            remove_small(labels, classes, float(op["max_mm3"]) / voxel)
            ran.append({"op": "remove_small", "classes": classes, "max_mm3": float(op["max_mm3"])})
    return ran
