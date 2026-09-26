"""Run a task through the PRODUCT path and keep its ranked output distribution.

Supersedes `encode_inline.py`, which hand-rolled the codec in this script. That copy predates
the tie-breaking rule, so its ranks can differ from the shipped encoder wherever two logits are
exactly equal - and fp16 logits tie often. A demo store that disagrees with `haversack.ranked` at
ties is a store nothing else reproduces, so this drives `segment(probabilities=RankedSpec(...))`
instead and touches no codec of its own.

The pipeline attaches the geometry itself (envelope start/stop, model grid, the channel -> label
lut, convention, orientation, frame), so each part lands self-describing and the duckn build
needs no second derivation.

`envelope_mm` is passed through: "none" runs the full model grid, which makes a store whose
array IS the model grid - no crop offset to apply, so the origin is the frame's own and a reader
needs no envelope arithmetic to place it. It costs inference time and some size (the extra voxels
are air, which compresses well but not to nothing).

usage: uv run python tools/ranked_emit.py IMAGE TASK OUTDIR [depth] [clip] [envelope_mm|none]
"""
import json
import time
from pathlib import Path

import numpy as np

from haversack.io import STORE_OUTPUT_SUFFIXES, is_store_output  # noqa: F401 - re-exported

# Nothing heavy at module level: haversack.pipeline brings torch and haversack.ranked brings
# rankfield, and a lean install (no torch) or a plain one (no store extra) must still import
# this module - the CLI once did so on every `segment`, and a README-recipe install died here.

DISTANCE_VOXELS = 2.0            # truncation of the emitted distance field, in voxels

#: haversack's own rules for what a store holds, counted: bump when a change to them moves a
#: store's bytes from the same run. 1 (2026-09-24): the TASK's field - a cascade's crop stages
#: left out, a union's parts composed into one field (ranked_compose, painting margins at 1/2,
#: clip 16), the derived layers computed on it.
STORE_RULES = 1


def store_extra_missing() -> list:
    """The packages of the ranked-store extra (rankfield, zarr, duckn - core since 2026-09-25, absent only in a lean install) that
    are not installed - by name, importing nothing. The one answer to "can this environment
    write a store", for the CLI's refusal and a server's ``kind=rankfield`` door alike."""
    import importlib.util

    def absent(name):
        try:
            return importlib.util.find_spec(name) is None
        except ModuleNotFoundError:          # a finder that refuses the name outright
            return True
    return [n for n in ("rankfield", "zarr", "duckn") if absent(n)]


def ranked_tag() -> str:
    """What a cached store's key carries beyond the task's weights: the formats its bytes are
    written in, READ from the libraries that write them - rankfield's encoding and duckn's seg
    metadata - and :data:`STORE_RULES`. A new rankfield or duckn format re-keys every store on
    its own; no number here has to be remembered for it."""
    import duckn
    import rankfield

    from .ranked_compose import rule_tag
    return (f"rankfield=rf{rankfield.FORMAT_VERSION}/seg{duckn.SEG_VERSION}/h{STORE_RULES}"
            f"/{rule_tag()}")


CENTERING = {"corner": "node", "center": "cell"}


def model_grid_geometry(meta):
    """(true spacing zyx, first-voxel-center origin xyz, direction xyz, duckn centering) of a
    part's array on an nnU-Net-path model grid - THE derivation, used by the builder's
    geometry and by the emit's distance scale alike.

    `spacing_zyx` in the meta is the nominal request. The grid the resampler actually produced
    depends on the grid it RAN ON - `frame.model_source` when the source was cropped to
    nonzero first (every nnU-Net-native lineage), else `frame.source`, never the full canonical
    grid when a crop happened - and on the convention:

      corner (TotalSegmentator, scipy.zoom)  holds the first and last sample centers, so
          spacing is (n_src-1)*s_src/(n_model-1) and voxel 0 does not move  -> duckn `node`
      center (nnU-Net native, skimage)       holds the field of view, so spacing is
          n_src*s_src/n_model and voxel 0 moves in by half the spacing change -> duckn `cell`

    The crop's offset (`model_source.origin`, mm along the source axes) and the envelope crop
    (`envelope.start`, model voxels) both move voxel 0. Deriving from the canonical grid alone
    once misplaced a cropped part by 19 mm with a 67 % spacing error, invisibly - the sample
    values are unaffected and only the stated geometry is wrong.
    """
    fr = meta["frame"]
    c = fr["canonical"]
    from .values import Geometry
    g = Geometry.from_record(c)              # rankfield 0.3's one-order record
    ran_on = fr.get("model_source") or fr.get("source")
    if ran_on is None:                       # a frame that states only its canonical grid
        ran_on = {"shape": g.shape_zyx, "spacing": g.spacing_zyx}
    n_src = [int(v) for v in ran_on["shape"]]
    s_src = [float(v) for v in ran_on["spacing"]]
    crop = [float(v) for v in (ran_on.get("origin") or (0.0, 0.0, 0.0))]
    model = [int(v) for v in meta["model_grid"]]
    start = [int(v) for v in (meta.get("envelope") or {}).get("start", (0, 0, 0))]
    convention = meta.get("convention") or fr.get("convention") or "corner"
    centering = CENTERING[convention]
    if centering == "node":
        eff = [(n_s - 1) * s / (n_m - 1) if n_m > 1 else s for n_s, s, n_m in zip(n_src, s_src, model)]
        shift = [0.0, 0.0, 0.0]                                    # voxel 0 stays put
    else:
        eff = [n_s * s / n_m for n_s, s, n_m in zip(n_src, s_src, model)]
        shift = [(e - s) / 2 for e, s in zip(eff, s_src)]
    D = np.asarray(g.direction_xyz, float).reshape(3, 3)
    off_zyx = [cr + sh + st * e for cr, sh, st, e in zip(crop, shift, start, eff)]
    off_xyz = np.asarray([off_zyx[2], off_zyx[1], off_zyx[0]], float)
    origin = np.asarray(g.origin_xyz, float) + D @ off_xyz
    return eff, tuple(float(v) for v in origin), list(g.direction_xyz), centering


def _true_spacing(meta):
    """The spacing the part actually landed on: a distance stated in millimeters has to use
    the grid the samples are really on, not the one that was asked for."""
    return model_grid_geometry(meta)[0]


def _emit_distance(part, code, out):
    """The distance field, computed where the arrays already are - on the CUDA worker.

    CUDA only, by measurement rather than principle: on MPS the dense torch kernel LOSES to
    the optimized numpy band in the builder (6.4 s vs 2.8 s on a 52 Mvoxel part - dense does
    ~40x the band's work and Apple bandwidth does not absorb it), so a local emit skips this
    and the builder computes it at build time instead. Either way the store gets the field;
    this only decides which machine pays.
    """
    import torch
    if not torch.cuda.is_available() or "frame" not in code.meta:
        return {}
    from haversack.ranked import distance_field
    t = time.perf_counter()
    eff = _true_spacing(code.meta)
    truncation = DISTANCE_VOXELS * min(eff)
    from rankfield import levels           # call time: the CLI imports this module on every segment
    dist = distance_field(code.ranks, code.support, clip=float(code.meta["clip"]), levels=levels(code.meta),
                          spacing_zyx=eff, truncation=truncation, device="cuda")
    np.save(out / f"{part}_distance.npy", dist)
    print(f"  {part:<12} distance on {torch.cuda.get_device_name(0)} in "
          f"{time.perf_counter() - t:.1f}s (T={truncation:.3f} mm)", flush=True)
    return {"distance_truncation": round(truncation, 6), "distance_max": 255,
            "distance_voxels": DISTANCE_VOXELS}


def _emit_junction(part, code, out, dist_meta):
    """The triple-line layer, beside the distance field and at its truncation.

    Same rule as the distance: computed on the worker only where a CUDA device holds the
    arrays. Elsewhere the builder computes it, from the numpy reference, which on Apple
    hardware is the fast path anyway (0.8 s against 1.4 s on MPS for a 52 Mvoxel part) -
    the layer gathers only at its tube voxels, so neither device does much work.
    """
    import torch
    if not dist_meta or not torch.cuda.is_available():
        return {}
    from haversack.ranked import junction_field
    t = time.perf_counter()
    eff = _true_spacing(code.meta)
    truncation = float(dist_meta["distance_truncation"])
    from rankfield import levels
    jn, jp = junction_field(code.ranks, code.support, clip=float(code.meta["clip"]), levels=levels(code.meta),
                            spacing_zyx=eff, truncation=truncation, device="cuda")
    np.save(out / f"{part}_junction.npy", jn)
    np.save(out / f"{part}_junction_pair.npy", jp)
    print(f"  {part:<12} junction on {torch.cuda.get_device_name(0)} in "
          f"{time.perf_counter() - t:.1f}s ({100.0 * np.count_nonzero(jn) / jn.size:.2f} % "
          "of voxels)", flush=True)
    return {"junction_truncation": round(truncation, 6), "junction_zero": 128,
            "junction_span": 127}


def main(image, task, outdir, depth=6, clip=8.0, envelope_mm=None, *, quiet=False, run=None,
         image_name=None, **segment_kw):
    """Emit ``task``'s ranked output for ``image`` into ``outdir`` (arrays as ``.npy``, the
    parts' metadata in ``meta.json``) and return the :class:`~haversack.result.Segmentation`.
    ``segment_kw`` goes to :func:`haversack.pipeline.segment` (device, dtype, grid, ...).

    ``run``, when given, is the segmentation to use - ``run(image, task, probabilities=spec,
    progress=..., **kw)``, a server's warm :meth:`Segmenter.segment` - instead of one built here.
    ``image_name`` is what the store calls its input (``source_file``): a server passes the job's
    identity, since its ``image`` is a scratch path or an image read ahead into memory, whose
    ``str()`` is a dump with an address in it (review, 2026-09-25). Default: a path's file name.

    What lands is the TASK's field (2026-09-24): a cascade's crop stages are left out (they only
    decided the final stage's box), and a union's parts are composed into one field
    (:mod:`haversack.ranked_compose`). The distance and junction layers are computed on what
    lands, so a union gets one field of its own surfaces, seams included, not one per model."""
    depth, clip = int(depth), float(clip)
    envelope_mm = (None if str(envelope_mm).lower() in ("none", "null", "")
                   else float(envelope_mm))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    metas, t0 = {}, time.perf_counter()
    say = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, flush=True, **k))

    def sink(part, code):
        for name, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
            if arr is not None:
                np.save(out / f"{part}_{name}.npy", arr)
        metas[part] = dict(code.meta)
        say(f"  {part:<12} {code!r}  ->  {out.name}/{part}_*.npy")

    segment_kw.setdefault("progress", None if quiet else (lambda p: say(f"    {p}")))
    from haversack.ranked import RankedSpec
    spec = RankedSpec(sink=sink, depth=depth, clip=clip)
    if run is not None:
        kw = dict(segment_kw)
        if envelope_mm is not None:
            kw["envelope_mm"] = envelope_mm
        seg = run(image, task, probabilities=spec, **kw)
    elif _runs_on_an_engine(task):
        # An engine with a ranked sink of its own (FastSurfer hands over its pre-argmax field,
        # engines/fastsurfer.emit_probabilities) runs through the Segmenter, which is the door
        # every engine task uses; nnU-Net policy that means nothing to it is left behind.
        from haversack.segmenter import Segmenter
        seg = Segmenter(device=segment_kw.get("device", "auto"),
                        weights=segment_kw.get("weights"),
                        batch_size=segment_kw.get("batch_size", "auto")).segment(
            image, task, probabilities=spec, progress=segment_kw.get("progress"))
    else:
        from haversack.pipeline import segment
        seg = segment(image, task, probabilities=spec, envelope_mm=envelope_mm, **segment_kw)
    if not metas:
        raise RuntimeError(f"{task} produced no ranked output: its engine took no "
                           "probabilities sink, so there is nothing to build a store from")
    metas = _task_field(out, metas, depth, device=segment_kw.get("device"), say=say)
    if not metas:
        # only crop stages were emitted: a cascade whose crop found none of its classes, whose
        # final model therefore never ran (upstream's empty result). The task has no field.
        from .errors import InputError
        raise InputError(f"{task}: its crop found none of the classes it crops to, so the task's "
                         "result is empty (all background) and its final model never ran - there "
                         "is no field to store")
    for part, m in metas.items():
        code = _Code(out, part, m)
        dist_meta = _emit_distance(part, code, out)
        m.update(dist_meta)
        m.update(_emit_junction(part, code, out, dist_meta))

    # the clip and depth the STORED field was encoded at: a composed union's are its own (clip
    # 16), and the store's provenance said 8 for one (review, 2026-09-25)
    stored = next(iter(metas.values()))
    path_like = isinstance(image, (str, Path))
    if image_name is None and path_like:
        image_name = Path(str(image)).name
    (out / "meta.json").write_text(json.dumps(
        {"image": str(image) if path_like else None, "source_file": image_name, "task": task, "depth": int(stored.get("depth", depth)),
         "clip": float(stored.get("clip", clip)),
         "envelope_mm": envelope_mm,
         "parts": metas, "provenance": seg.provenance, "timings": seg.timings},
        indent=1, default=str), encoding="utf-8")
    say(f"done in {time.perf_counter() - t0:.0f}s -> {out}")
    return seg


class _Code:
    """A staged part's arrays (memory-mapped) and meta, in the shape the derived layers read."""

    def __init__(self, out, part, meta):
        self.ranks = np.load(out / f"{part}_ranks.npy", mmap_mode="r")
        self.support = np.load(out / f"{part}_support.npy", mmap_mode="r")
        self.meta = meta


def _drop(out, part):
    for name in ("ranks", "support", "tail"):
        (out / f"{part}_{name}.npy").unlink(missing_ok=True)


def _task_field(out, metas, depth, *, device=None, say=print):
    """The staged parts reduced to the task's own field: crop stages dropped, a union composed.
    Returns the metas of what remains (one part). A part the pipeline marked ``role: crop`` is a
    cascade's crop stage; FastSurfer and single models pass through untouched."""
    for part in [p for p, m in metas.items() if m.get("role") == "crop"]:
        _drop(out, part)
        del metas[part]
        say(f"  {part:<12} crop stage: not stored (its box is in the provenance)")
    if len(metas) < 2:
        return metas
    from .ranked_compose import compose
    t = time.perf_counter()
    parts = [(p, (np.load(out / f"{p}_ranks.npy", mmap_mode="r"),
                  np.load(out / f"{p}_support.npy", mmap_mode="r")), m) for p, m in metas.items()]
    ranks, support, cmeta, labels = compose(parts, depth=depth, device=device)
    first = parts[0][2]
    task = first.get("task")
    name = str(task)
    # the placement is the parts' (one grid, checked); the codec is the composed field's own
    base = {k: v for k, v in first.items()
            if k not in ("softmax", "tail_temperatures", "max_tail_at_temperature", "part")}
    base.update(cmeta)
    base.update(labels=[int(v) for v in labels], part=name, task=task, labels_named_by=task)
    for p, *_ in parts:
        _drop(out, p)
    np.save(out / f"{name}_ranks.npy", ranks)
    np.save(out / f"{name}_support.npy", support)
    say(f"  {name:<12} composed {len(parts)} parts into one field "
        f"({len(labels)} labels, clip {cmeta['clip']:g}) in {time.perf_counter() - t:.1f}s")
    return {name: base}



#: Engines whose runner takes a ranked sink. A store is the whole output distribution, so an
#: engine that returns only labels has nothing to put in one.
RANKED_ENGINES = frozenset({"fastsurfer"})


def _runs_on_an_engine(task) -> bool:
    from haversack.engines import registry
    try:
        return registry.engine_for_task(task).name != registry.NNUNETV2
    except Exception:                                  # noqa: BLE001 - a TaskSpec, a folder
        return False


def supports_store_output(task) -> bool:
    """Whether ``task`` can write a ranked store: every nnU-Net task, and the engines in
    :data:`RANKED_ENGINES`."""
    from haversack.engines import registry
    try:
        name = registry.engine_for_task(task).name
    except Exception:                                  # noqa: BLE001
        return True
    return name == registry.NNUNETV2 or name in RANKED_ENGINES


def input_source(spec) -> dict:
    """A duckn provenance source naming the input: a data-source identifier (``idc:...``,
    ``tcia:...``) as it was given, or a local file by name and format - so the case is
    identifiable from the store alone (README section 6.1)."""
    from .sources import parse_input
    spec = str(spec)
    if parse_input(spec) is not None:
        return {"type": "image", "identifier": spec,
                "description": "resolved by haversack's data sources"}
    p = Path(spec)
    fmt = "DICOM" if p.is_dir() else "".join(p.suffixes[-2:]).lstrip(".") or "unknown"
    return {"type": "image", "format": fmt, "path": p.name}


def segment_to_store(image, task, out, *, case=None, depth=6, clip=8.0, parts="all",
                     distance_voxels=DISTANCE_VOXELS, allow_unnamed=False, names=None,
                     quiet=False, source=None, run=None, image_name=None, **segment_kw):
    """Segment ``image`` with ``task`` and write the ranked store at ``out`` (``.duckn`` or
    ``.duckn.zip``); returns ``(segmentation, out)``.

    UNDOCUMENTED (see ranked_store): the product's labels path is unchanged, this is the
    experiment door. Emits into a staging directory beside ``out`` (the arrays of a whole-body
    run are larger than memory should hold twice), builds the store from it, and removes the
    staging directory whatever happens. Segment names come from the run's own label schema,
    so a stock nnU-Net folder or a TaskSpec needs no catalog lookup. ``segment_kw`` goes to
    :func:`haversack.pipeline.segment` (device, dtype, grid, envelope_mm, ...).
    """
    import shutil
    import tempfile

    from .errors import InputError
    from .ranked_build import build
    out = Path(out).expanduser()
    src = Path(str(image)).expanduser() if isinstance(image, (str, Path)) else None
    if src is not None and src.exists() and src.resolve() == out.resolve():
        raise InputError(f"{out}: the store output is the input; name a different path")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".emit-", dir=out.parent))
    try:
        envelope_mm = segment_kw.pop("envelope_mm", None)          # segment()'s default: none
        seg = main(image, task, staging, depth, clip, envelope_mm, quiet=quiet, run=run,
                   image_name=image_name, **segment_kw)
        # Names the run reports are the model's own, so the store may declare the labeling
        # scheme they belong to - unless the run could not name its classes and fell back to
        # `label <v>` (a MONAI region head), or the caller brought names of its own.
        model_names = names is None and not (seg.provenance or {}).get("labels_unnamed")
        if names is None:
            names = {int(v): str(n) for v, n in seg.schema.names.items()}
        path_like = isinstance(image, (str, Path))
        if case is None:
            case = (Path(str(image)).name.split(".")[0] if path_like else "") or "case"
        if source is None and path_like:        # an in-memory image names nothing; say nothing
            source = input_source(image)
        build(staging, out, case, parts, allow_unnamed, distance_voxels, names=names, quiet=quiet,
              source=source, model_names=model_names)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return seg, out


def main_cli(argv=None):
    """The command line of ``tools/ranked_emit.py``."""
    # argparse, not a positional slice. `main(*sys.argv[1:6])` silently dropped the sixth
    # argument once, so `envelope_mm` kept its default and the run was quietly not the one
    # asked for - visible only because meta.json records what was actually used.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("task")
    ap.add_argument("outdir")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--envelope-mm", default="none",
                    help='margin in mm, or "none" (the default) to run the full model grid')
    a = ap.parse_args(argv)
    main(a.image, a.task, a.outdir, a.depth, a.clip, a.envelope_mm)


if __name__ == "__main__":
    main_cli()
