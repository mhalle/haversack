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

from haversack.pipeline import segment
# NOT haversack.ranked at module level: the CLI imports this module on every `segment` for
# `is_store_output`, and ranked is a shim over rankfield, which a plain install lacks.

DISTANCE_VOXELS = 2.0            # truncation of the emitted distance field, in voxels


def _true_spacing(meta):
    """The grid the part actually landed on, mirroring ranked_build_store.geometry().

    `spacing_zyx` in the meta is the nominal request; under the corner rule the true spacing
    is (n_src-1)*s_src/(n_model-1). A distance stated in millimetres has to use the grid the
    samples are really on, not the one that was asked for.
    """
    c = meta["frame"]["canonical"]
    model = [int(v) for v in meta["model_grid"]]
    if meta.get("convention", "corner") == "corner":
        return [(n_s - 1) * s / (n_m - 1) if n_m > 1 else s
                for n_s, s, n_m in zip(c["shape_zyx"], c["spacing_zyx"], model)]
    return [n_s * s / n_m for n_s, s, n_m in zip(c["shape_zyx"], c["spacing_zyx"], model)]


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


def main(image, task, outdir, depth=6, clip=8.0, envelope_mm=20.0, *, quiet=False,
         **segment_kw):
    """Emit ``task``'s ranked output for ``image`` into ``outdir`` (arrays as ``.npy``, the
    parts' metadata in ``meta.json``) and return the :class:`~haversack.result.Segmentation`.
    ``segment_kw`` goes to :func:`haversack.pipeline.segment` (device, dtype, grid, ...)."""
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
        dist_meta = _emit_distance(part, code, out)
        metas[part] = {**code.meta, **dist_meta, **_emit_junction(part, code, out, dist_meta)}
        say(f"  {part:<12} {code!r}  ->  {out.name}/{part}_*.npy")

    segment_kw.setdefault("progress", None if quiet else (lambda p: say(f"    {p}")))
    from haversack.ranked import RankedSpec
    seg = segment(image, task, probabilities=RankedSpec(sink=sink, depth=depth, clip=clip),
                  envelope_mm=envelope_mm, **segment_kw)

    (out / "meta.json").write_text(json.dumps(
        {"image": str(image), "task": task, "depth": depth, "clip": clip,
         "envelope_mm": envelope_mm,
         "parts": metas, "provenance": seg.provenance, "timings": seg.timings},
        indent=1, default=str))
    say(f"done in {time.perf_counter() - t0:.0f}s -> {out}")
    return seg


STORE_OUTPUT_SUFFIXES = (".duckn.zip", ".duckn")


def is_store_output(path) -> bool:
    """Whether an output path asks for a ranked store rather than labels: ``.duckn`` (a
    directory) or ``.duckn.zip`` (a standard zarr zip). A bare ``.zip`` is not enough - that
    would silently turn a typo into a different kind of output."""
    return str(path).lower().endswith(STORE_OUTPUT_SUFFIXES)


def segment_to_store(image, task, out, *, case=None, depth=6, clip=8.0, parts="all",
                     distance_voxels=DISTANCE_VOXELS, allow_unnamed=False, names=None,
                     quiet=False, **segment_kw):
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
        envelope_mm = segment_kw.pop("envelope_mm", 20.0)
        seg = main(image, task, staging, depth, clip, envelope_mm, quiet=quiet, **segment_kw)
        if names is None:
            names = {int(v): str(n) for v, n in seg.schema.names.items()}
        if case is None:
            case = Path(str(image)).name.split(".")[0] or "case"
        build(staging, out, case, parts, allow_unnamed, distance_voxels, names=names, quiet=quiet)
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
    ap.add_argument("--envelope-mm", default="20.0",
                    help='margin in mm, or "none" to run the full model grid')
    a = ap.parse_args(argv)
    main(a.image, a.task, a.outdir, a.depth, a.clip, a.envelope_mm)


if __name__ == "__main__":
    main_cli()
