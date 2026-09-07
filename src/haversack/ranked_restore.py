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

    def image(self, orientation: str | None = "input"):
        """A SimpleITK image of the labels: in the input's own orientation when the store
        carries a frame and ``orientation`` is ``"input"``, else canonical."""
        from . import io as nio
        from .values import Geometry
        g = self.geometry
        geo = Geometry(spacing_zyx=g.spacing_zyx, shape_zyx=g.shape_zyx, origin_xyz=g.origin_xyz,
                       direction_xyz=g.direction_xyz)
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


def resolve_grid(store, grid="input"):
    st, root, owned = _open(store)
    try:
        return rf.resolve_grid(parts_of(root)[0], grid)
    finally:
        if owned:
            st.close()


def roi_of(store, labels, *, grid="input", halo: int = 1) -> tuple:
    """The output-index box that should hold ``labels`` (label values), from their stored
    extents - a heuristic, see ``rankfield.roi_of``."""
    st, root, owned = _open(store)
    try:
        parts = parts_of(root)
        grid_out, _ = rf.resolve_grid(parts[0], grid)
        segs = root.attrs.asdict()["duckn"]["extensions"]["seg"]["segments"]
        want = {int(v) for v in labels}
        extents = []
        for i, p in enumerate(parts):
            lut = p.field.labels
            for s in segs:
                if s.get("label_value") in want and (s.get("layer") or 0) == i and s.get("extent") \
                        and s["label_value"] in lut:
                    extents.append((i, s["extent"]))
        if not extents:
            raise InputError(f"none of {sorted(want)} has an extent in this store")
        return rf.roi_of(parts, extents, grid_out, halo=halo)
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
        try:
            g, _ = rf.resolve_grid(parts[0], grid)
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
        return Restored(labels=r.labels, grid=r.grid, geometry=r.geometry, frame=r.frame,
                        parts=r.parts, interp=r.interp, roi=r.roi)
    finally:
        if owned:
            st.close()


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
    img = res.image("input")
    out = Path(a.output)
    from .ranked_store import open_store
    from .result import Segmentation
    from .values import LabelSchema
    with open_store(Path(a.store), "r") as st:
        ext = st.root.attrs.asdict()["duckn"]["extensions"]
    names = {int(s["label_value"]): s.get("name", "") for s in ext["seg"]["segments"]
             if s.get("label_value") is not None and not s.get("background")}
    prov = {"restored_from": str(a.store), "interp": a.interp, "grid": list(res.grid.shape),
            "spacing": list(res.grid.spacing), "parts": res.parts,
            "haversack": (ext.get("haversack") or {}).get("haversack_version")}
    Segmentation(labels=img, schema=LabelSchema(names=names), grid=res.grid, spec=None,
                 provenance=prov).save(out)
    if not a.quiet:
        print(f"wrote {out}: {tuple(res.labels.shape)} at {tuple(round(v, 3) for v in res.grid.spacing)} mm, "
              f"{a.interp}, from {len(res.parts)} part(s)", file=sys.stderr)
    return 0
