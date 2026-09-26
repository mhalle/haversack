"""Labels from a ranked STORE, on any grid: the store adapter over ``rankfield.restore``.

The restore itself - candidates, the tie rule, paint, the placements, the GPU kernels - is
the library's (``rankfield/docs/format.md``). This module opens a store, turns each part
into a :class:`rankfield.Part` (its planes, its channel table, its array geometry from the
duckn attributes, its frame record when the builder wrote one) and hands the list to the
library in paint order; ``Restored.image`` puts the labels back in the input's orientation,
and ``main_cli`` is the hidden ``haversack restore`` command.

Measured on idc-torso1 (``total_fast`` at 3 mm restored to 1.5 mm, 52.5 M voxels): a 0.3
store differs from the labels the run wrote at 0.0037 % of voxels (a 0.2 store: 0.06 %,
biased); Metal (M2) 0.4 s, Triton (A10) 0.2 s, torch CPU 12-24 s.

UNDOCUMENTED, like the store (see ranked_store).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rankfield as rf
from rankfield import store as rfstore

from .errors import InputError

# Nothing about the stored form is restated here. rankfield reads it - the version
# gate, the geometry, the planes - because rankfield decodes it. A copy of that
# reading drifted twice in one day: the version list said ("0.2", "0.3") while
# `ranked.py` stamped whatever rankfield called current, and the geometry copy
# never learned format 0.4's tails (2026-09-07).


@dataclass
class Restored:
    """Labels on ``grid`` in the canonical frame (Z, Y, X), with what places them."""

    labels: np.ndarray
    grid: rf.Grid
    geometry: object
    frame: rf.Frame | None
    parts: list[str]
    interp: str
    roi: tuple | None = None
    notes: tuple = ()
    #: the store is FastSurfer's (``is_fastsurfer``): the split ran unless a note says otherwise
    fastsurfer: bool = False

    def image(self, orientation: str | None = "input"):
        """A SimpleITK image of the labels: in the input's own orientation when the store
        carries a frame and ``orientation`` is ``"input"``, else canonical."""
        from . import io as nio
        from .values import Geometry
        g = self.geometry
        geo = Geometry.from_record(g)       # rankfield 0.3's Geometry keeps one order
        arr = self.labels
        if orientation == "input" and self.frame is not None:
            arr, geo = nio.reorient(arr, geo, self.frame.original_orientation)
        return nio.to_image(arr, geo)


def _open(store):
    """``(store handle, root, owned)`` - opened here, or handed in already open."""
    from .ranked_store import open_store
    if isinstance(store, (str, Path)):                # a Path has a `.root` of its own
        p = Path(store)
        if not p.exists():
            raise InputError(f"{p}: no such store")
        try:
            st = open_store(p, "r")
        except Exception as e:
            raise InputError(f"{p}: not a ranked store ({e.__class__.__name__})") from None
        return st, st.root, True
    return store, store.root, False


def _places_a_part(env) -> bool:
    """Whether ``env`` is an envelope this reader can actually place a part with.

    duckn's inclusive six-bound list, or the earlier ``{"start": ...}`` dict.
    Anything else - absent, None, a scalar, a string, a ``{lower, upper}`` dict,
    or the oldest ``envelope_start_zyx`` form with no ``envelope`` key at all -
    is refused here, because ``rankfield.store.read_parts`` places such a part at
    the ORIGIN by default. This module used to raise on every one of those; after
    it began delegating (3cf89c2) `haversack restore` exited 0 on a legacy store
    and wrote misplaced labels. Found 2026-09-07 by review. `tools/ranked_verify.py`
    requires `envelope` too, so this is the same contract, enforced where a
    restore will actually read.
    """
    return ((isinstance(env, (list, tuple)) and len(env) == 6)
            or (isinstance(env, dict) and "start" in env))


def parts_of(root) -> list[rf.Part]:
    """The store's parts in paint order. ``rankfield.store.read_parts`` does the
    reading - the version gate, the geometry, the planes, and the ``part_order``
    block, which it looks for under a ``haversack`` key as well as its own. What
    is checked here is the one thing that reader is deliberately lenient about;
    see :func:`_places_a_part`."""
    for i in rfstore.part_indices(root):
        m = (root[f"parts/{i}"].attrs.asdict().get("duckn", {})
             .get("extensions", {}).get("ranked") or {})
        if m and not _places_a_part(m.get("envelope")):
            raise InputError(
                f"parts/{i}: envelope {m.get('envelope')!r} is not a form this reader can "
                "place a part with - the store predates the current layout; upgrade it with "
                "tools/ranked_align_parts.py")
    try:
        return rfstore.read_parts(root)
    except ValueError as e:
        raise InputError(str(e)) from None          # haversack's one-line error contract


#: duckn's space for LPS millimeters: the one a target grid is read in (rankfield's Geometry
#: is LPS; a target in any other space is refused rather than silently mirrored)
_LPS = "left-posterior-superior"


def input_geometry(part: rf.Part):
    """The world grid ``"input"`` means for ``part`` when it has no frame: the ``target_grid``
    its emit recorded, as a rankfield Geometry - or None.

    A frame is how an nnU-Net part restores onto its input: the exact rule its model grid was
    made by. FastSurfer's field lives on a grid FastSurfer resampled in WORLD space from the
    input (its conformed 1 mm grid), which a frame cannot describe; its emit records the
    input grid instead (``target_grid``), and the conformed grid is the part's own array
    geometry. The two geometries are the whole map - input index -> world -> conformed index
    (``rankfield.Affine``) - so nothing else is stored (2026-09-25; before, `haversack
    restore` refused "input" for these stores and restored only onto the conformed grid)."""
    if part.field.frame:
        return None
    t = part.field.meta.get("target_grid")
    if not isinstance(t, dict):
        return None
    if t.get("space") != _LPS:
        raise InputError(f"the part's target grid is in {t.get('space')!r} space; this reader "
                         f"reads it in {_LPS!r} only")
    return rf.Geometry(shape=tuple(int(n) for n in t["samples"]),
                       directions=tuple(tuple(float(v) for v in a["space_direction"]) for a in t["axes"]),
                       origin=tuple(float(v) for v in t["space_origin"]))


def _world(parts, grid):
    """``grid`` as the library takes it: the recorded input geometry for ``"input"`` on an
    unframed store that has one, else as given."""
    if grid in ("input", None):
        geo = input_geometry(parts[0])
        if geo is not None:
            return geo
    return grid


def _grid_of(part, grid):
    """``(output Grid, frame)`` for ``grid`` - a world Geometry included."""
    if isinstance(grid, rf.Geometry):
        return rf.Grid(grid.shape, spacing=grid.spacing), None
    return rf.resolve_grid(part, grid)


def resolve_grid(store, grid="input"):
    st, root, owned = _open(store)
    try:
        parts = parts_of(root)
        return _grid_of(parts[0], _world(parts, grid))
    finally:
        if owned:
            st.close()


def _value(seg: dict):
    """The one label value a stored segment is, or None. Reads the raw ``seg`` block of a
    store of either shape: seg 0.8's ``label_values`` (a class lists exactly one) and the
    ``label_value`` of stores written before it. A segment listing several values is a
    union someone authored, not a class."""
    values = seg.get("label_values")
    if isinstance(values, list):
        return int(values[0]) if len(values) == 1 else None
    value = seg.get("label_value")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _is_background(seg: dict) -> bool:
    return seg.get("role") == "background" or bool(seg.get("background"))


def roi_of(store, labels, *, grid="input", halo: int = 1) -> tuple:
    """The output-index box that should hold ``labels`` (label values), from their stored
    extents - a heuristic, see ``rankfield.roi_of``."""
    st, root, owned = _open(store)
    try:
        parts = parts_of(root)
        grid = _world(parts, grid)
        grid_out, _ = _grid_of(parts[0], grid)
        segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
        want = {int(v) for v in labels}
        extents = []
        for i, p in enumerate(parts):
            lut = p.field.labels
            for s in segs:
                if _value(s) in want and (s.get("layer") or 0) == i and s.get("extent") \
                        and _value(s) in lut:
                    extents.append((i, s["extent"]))
        if not extents:
            raise InputError(f"none of {sorted(want)} has an extent in this store")
        return rf.roi_of(parts, extents, grid_out, halo=halo,
                         **({"world": grid} if isinstance(grid, rf.Geometry) else {}))
    finally:
        if owned:
            st.close()


def restore(store, *, grid="input", interp: str = "linear", roi=None, device="auto",
            slab_voxels: int = 1 << 20, progress=None) -> Restored:
    """Labels on ``grid`` from the store; ``device="auto"`` takes the GPU when there is one."""
    st, root, owned = _open(store)
    try:
        parts = parts_of(root)
        if device == "auto":
            from .resample import best_device
            device = str(best_device())
        if isinstance(grid, (int, float)) and not grid > 0:
            raise InputError(f"spacing must be positive, got {grid}")
        try:
            import torch
            torch.device(device)
        except RuntimeError as e:
            raise InputError(f"--device {device!r}: {e}") from None
        grid = _world(parts, grid)
        try:
            g, _ = _grid_of(parts[0], grid)
        except ValueError as e:
            raise InputError(str(e)) from None
        box = roi or tuple((0, n) for n in g.shape)
        voxels = int(np.prod([b - a for a, b in box]))
        if voxels >= 2 ** 31:
            # the kernels index the output with 32-bit offsets; and a grid this size is
            # asked for by mistake (a spacing in the wrong unit), not on purpose
            raise InputError(f"the output grid {tuple(g.shape)} at {tuple(round(v, 4) for v in g.spacing)} mm "
                             f"is {voxels:,} voxels; the restore holds fewer than 2^31 - ask for a "
                             f"coarser spacing or an roi")
        try:
            r = rf.restore(parts, grid=grid, interp=interp, roi=roi, device=device,
                           slab_voxels=slab_voxels, progress=progress)
        except ValueError as e:
            raise InputError(str(e)) from None
        except (MemoryError, RuntimeError) as e:
            # torch reports an allocation it cannot make as a RuntimeError (OutOfMemoryError
            # on CUDA, "Invalid buffer size" on MPS), not a MemoryError
            if isinstance(e, MemoryError) or any(w in str(e).lower() for w in ("memory", "buffer size", "alloc")):
                raise InputError(f"the output grid {tuple(g.shape)} at {tuple(round(v, 4) for v in g.spacing)} mm "
                                 f"({voxels:,} voxels) does not fit in memory on {device}; ask for a "
                                 f"coarser spacing or an roi") from None
            raise
        labels, notes = _engine_rules(parts, r.labels, roi)
        return Restored(labels=labels, grid=r.grid, geometry=r.geometry, frame=r.frame,
                        parts=r.parts, interp=r.interp, roi=r.roi, notes=notes,
                        fastsurfer=is_fastsurfer(parts))
    finally:
        if owned:
            st.close()


def is_fastsurfer(parts) -> bool:
    """Whether a store's field is FastSurfer's. ``labels_named_by`` says so on stores written
    since 0.13.0; one written before carries only the task and the softmax block's engine, and
    was left unlateralized with no word said (review, 2026-09-25)."""
    meta = parts[0].field.meta
    if str(meta.get("labels_named_by") or "").startswith("fastsurfer:"):
        return True
    if str(meta.get("task") or "").startswith("fastsurfer:"):
        return True
    softmax = meta.get("softmax")
    return isinstance(softmax, dict) and softmax.get("engine") == "fastsurfer"


def _engine_rules(parts, labels, roi):
    """What the engine does to its argmax that the field cannot say: ``(labels, notes)``.

    FastSurfer's network has one channel for each of 17 cortical parcels on BOTH hemispheres,
    and FastSurfer lateralizes them after the argmax (``split_cortex_labels``: each connected
    piece to the nearer hemisphere's white matter). A store holds the network's classes, so
    every restore of one - conformed grid or input grid - named the right hemisphere's parcels
    with the left ids until this applied FastSurfer's own rule (2026-09-25; with it the input
    grid restore of the ds000114 T1 store matches the served labels, see the tests). The rule
    reads the whole brain, so an roi restore is left as the field says, and says so."""
    if not is_fastsurfer(parts):
        return labels, ()
    if roi is not None:
        return labels, ("cortical parcels are not lateralized in an roi restore (FastSurfer's rule "
                        "reads both hemispheres' white matter)",)
    try:
        from FastSurferCNN.data_loader.data_utils import split_cortex_labels
    except ImportError:
        return labels, ("cortical parcels are not lateralized: FastSurfer's rule needs the "
                        "fastsurfer-lean package, which haversack installs (this is a lean or --no-deps install)",)
    wide = labels.astype(np.int32)              # the rule compares against 1003..2035
    try:
        split = split_cortex_labels(wide)
    except Exception as e:                      # noqa: BLE001 - e.g. no white matter to anchor on
        return labels, (f"cortical parcels are not lateralized: FastSurfer's rule failed "
                        f"({type(e).__name__}: {e})",)
    top = int(split.max()) if split.size else 0
    return split.astype(np.uint8 if top < 256 else np.uint16), ()


def main_cli(argv=None) -> int:
    """``haversack restore STORE -o LABELS [--spacing S] [--interp linear|nearest]``."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        prog="haversack restore", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Labels from a ranked store (.duckn or .duckn.zip), on the input's grid or an "
                    "isotropic one, in the input's orientation - the restore, read from the store.")
    ap.add_argument("store", help="a ranked store: STORE.duckn or STORE.duckn.zip")
    ap.add_argument("-o", "--output", required=True, help="labels file (.seg.nrrd, .nrrd, .nii.gz, .mha)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="isotropic output spacing in mm (default: the input's own grid)")
    ap.add_argument("--interp", choices=("linear", "nearest"), default="linear",
                    help="the argmax of the interpolated field, or the nearest model voxel's")
    ap.add_argument("--device", default="auto", help="torch device for the blend (auto: the GPU when there is one)")
    ap.add_argument("--quiet", action="store_true", help="no progress on stderr")
    a = ap.parse_args(argv)
    from . import io as nio
    if nio.image_suffix(a.output) is None:
        raise InputError(f"{a.output}: labels take .seg.nrrd, .nrrd, .nii.gz, .nii or .mha")
    say = (lambda m: None) if a.quiet else (lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    if a.spacing is not None and not a.spacing > 0:
        raise InputError(f"--spacing must be positive, got {a.spacing}")
    res = restore(a.store, grid=(a.spacing if a.spacing is not None else "input"), interp=a.interp,
                  device=a.device, progress=say)
    for n in res.notes:
        print(f"note: {n}", file=sys.stderr)
    img = res.image("input")
    out = Path(a.output)
    from .ranked_store import open_store
    from .result import Segmentation
    from .values import LabelSchema
    with open_store(Path(a.store), "r") as st:
        ext = st.root.attrs.asdict()["duckn"]["extensions"]
    names = {_value(s): s.get("name", "") for s in (ext.get("seg") or {}).get("segments", [])
             if _value(s) is not None and not _is_background(s)}
    if res.fastsurfer and not res.notes:
        # the split made right-hemisphere ids no stored segment names (the store holds the
        # network's channels): name them as FastSurfer's own labels output names them
        from .engines.fastsurfer import output_lut
        lut = output_lut()
        for v in set(np.unique(res.labels).tolist()) - set(names) - {0}:
            if v in lut:
                names[int(v)] = lut[v]["name"]
    prov = {"restored_from": str(a.store), "interp": a.interp, "grid": list(res.grid.shape),
            "spacing": list(res.grid.spacing), "parts": res.parts,
            # a deviation is never only on stderr (house rule): what this restore did not do
            **({"notes": list(res.notes)} if res.notes else {}),
            "haversack": (ext.get("haversack") or {}).get("haversack_version")}
    Segmentation(labels=img, schema=LabelSchema(names=names), grid=res.grid, spec=None,
                 provenance=prov).save(out)
    if not a.quiet:
        print(f"wrote {out}: {tuple(res.labels.shape)} at {tuple(round(v, 3) for v in res.grid.spacing)} mm, "
              f"{a.interp}, from {len(res.parts)} part(s)", file=sys.stderr)
    return 0
