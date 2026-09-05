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

from .errors import InputError


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


_SPACE_TO_LPS = {"left-posterior-superior": (1.0, 1.0, 1.0), "right-anterior-superior": (-1.0, -1.0, 1.0),
                 "left-anterior-superior": (1.0, -1.0, 1.0)}


def _array_geometry(arr) -> rf.Geometry:
    """The stored array's own placement, from its duckn attributes, in SimpleITK's terms."""
    d = arr.attrs.asdict()["duckn"]
    axes = [a for a in d["axes"] if a.get("space_direction")]
    if len(axes) != 3:
        raise InputError("ranks array: need three spatial axes with space_direction")
    space = str(d.get("space", "left-posterior-superior")).replace("-time", "")
    if space not in _SPACE_TO_LPS:
        raise InputError(f"ranks array: space {space!r} is not one this reader places")
    flip = np.asarray(_SPACE_TO_LPS[space])
    dirs = [np.asarray(a["space_direction"], float) * flip for a in axes]      # z, y, x in LPS
    spacing = [float(np.linalg.norm(v)) for v in dirs]
    cos = [v / n for v, n in zip(dirs, spacing)]
    D = np.stack([cos[2], cos[1], cos[0]], axis=1)                              # columns x, y, z
    origin = np.asarray(d.get("space_origin", (0.0, 0.0, 0.0)), float) * flip
    return rf.Geometry(spacing_zyx=tuple(spacing), shape_zyx=tuple(int(v) for v in arr.shape[1:]),
                       origin_xyz=tuple(float(v) for v in origin),
                       direction_xyz=tuple(float(v) for v in D.reshape(-1)))


def parts_of(root) -> list[rf.Part]:
    """The store's parts in paint order, as the library sees them; the planes are the zarr
    arrays themselves, read by the restore only where the output needs them."""
    order = (root.attrs.asdict().get("duckn", {}).get("extensions", {}).get("haversack", {})
             .get("part_order"))
    idx = [p["index"] for p in order] if order else []
    if not idx:
        i = 0
        while f"parts/{i}" in root:
            idx.append(i)
            i += 1
    out = []
    for i in idx:
        g = root[f"parts/{i}"]
        m = dict(g.attrs.asdict()["duckn"]["extensions"]["ranked"])
        if str(m.get("version")) not in ("0.2", "0.3"):
            raise InputError(f"parts/{i}: ranked format {m.get('version')!r}; this reader knows 0.2 and 0.3")
        env = m["envelope"]
        field = rf.RankField(ranks=g["ranks"], support=g["support"],
                             tail=g["tail"] if "tail" in g else None, meta=m,
                             labels=[int(v) for v in m["labels"]], geometry=_array_geometry(g["ranks"]),
                             frame=m.get("frame"))
        out.append(rf.Part(field=field, envelope_start=tuple(int(a) for a in env[0::2]),
                           name=str(m.get("part", i))))
    return out


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
            r = rf.restore(parts, grid=grid, interp=interp, roi=roi, device=device,
                           slab_voxels=slab_voxels, progress=progress)
        except ValueError as e:
            raise InputError(str(e)) from None
        except MemoryError:
            g, _ = rf.resolve_grid(parts[0], grid)
            raise InputError(f"the output grid {tuple(g.shape)} at {tuple(g.spacing)} mm does not fit in "
                             f"memory; ask for a coarser spacing or an roi") from None
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
