"""The one encode path every encoder shares: resolve the encoder, check its weights, fetch and read
the input, let the family prepare and run it, place the lattices, record provenance, and write the
field through feldglas (the format's one author).

A family module supplies ``load(spec, weights_dir, device, dtype)``, ``prepare(spec, image, model)
-> Prepared`` (after the load: an nnU-Net input depends on the model's spacing and normalization)
and ``run(spec, model, prepared, device, dtype, slab) -> [tokens per lattice]``; everything
else is here, so two encoders cannot record provenance, place a field or name its license two ways.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

from ..errors import InputError
from . import weights as W
from .registry import EncoderSpec, resolve


def _identity(input_spec, path: Path) -> dict:
    """What the field records of its input: how it was named, and the bytes' digest."""
    from ..content import digest_dir, digest_file
    from ..sources import parse_input
    parsed = parse_input(str(input_spec))
    ident = {"input": str(input_spec)}
    if parsed:
        ident["source"], ident["identifier"] = parsed[0], parsed[1]
    ident["digest"] = digest_dir(path) if path.is_dir() else digest_file(path)
    return ident


def attribution_of(spec: EncoderSpec) -> dict:
    """What a field says of its encoder's makers: title, repository and the papers they ask to
    be cited (identifiers only - the full record is ``haversack encoders --json``)."""
    from ..attribution import _applies, for_encoder
    rec = for_encoder(spec) or {}
    refs = [r for r in rec.get("cite") or [] if _applies(r, {"modality": spec.modality})]
    return {"title": rec.get("title", ""), "repository": rec.get("repository", ""),
            "cite": [{k: r[k] for k in ("title", "year", "doi", "pmid", "arxiv") if r.get(k)} for r in refs]}


def field_of(spec: EncoderSpec, tokens, prepared, identity: dict):
    """A feldglas ``Field`` for ``tokens``: placed on the prepared model grid, described by the
    spec (layers, reach, look offsets - only what was measured), keyed by the weights' digest."""
    import numpy as np
    from feldglas import Embedding, Field, Provenance
    from rankfield.geometry import Geometry
    from .. import __version__
    g = prepared.grid
    extra = {k: v for k, v in prepared.extra.items() if k != "input_grid"}
    lat = spec.lattices
    emb = Embedding(layers=tuple(l.layer for l in lat), stage=spec.stage, metric=spec.metric, normalized=spec.normalized,
                    receptive_mm=tuple(l.receptive_mm for l in lat) if any(l.receptive_mm for l in lat) else (),
                    support_offset_mm=tuple(l.support_offset_mm for l in lat) if any(l.support_offset_mm for l in lat) else ())
    weights = ",".join(f"sha256:{wf.sha256}" for wf in spec.weights) or f"task {spec.uses_task}"
    prov = Provenance(encoder=spec.name, code=f"haversack {__version__}", weights=weights,
                      preprocessing=f"{spec.family} input convention (haversack.encoders.{spec.family})",
                      license=spec.license, source=identity.get("identifier") or identity.get("input", ""),
                      extra={**extra, "revision": spec.revision, "attribution": attribution_of(spec)},
                      input={"identity": identity, "grid": prepared.extra.get("input_grid")} if prepared.extra.get("input_grid")
                      else {"identity": identity})
    return Field(tokens=[np.asarray(t, np.float32) for t in tokens], kernels=[l.kernel for l in lat],
                 grid=Geometry(shape=tuple(g["shape"]), directions=tuple(tuple(r) for r in g["directions"]),
                               origin=tuple(g["origin"])),
                 provenance=prov, embedding=emb, data_box=prepared.data_box)


def encode(name: str, input_spec, out, *, device: str = "auto", dtype: str | None = None, int8: bool = False,
           slab: int = 16, progress=None) -> dict:
    """Encode one input with encoder ``name`` into ``out`` (``<name>.zarr.zip``). Returns what was done."""
    _check_out(out)
    spec = _ready(name)
    from ..sources import materialize
    t0 = time.time()
    path = Path(materialize(str(input_spec), progress=progress))
    return encode_file(spec.name, path, out, identity=_identity(input_spec, path), device=device, dtype=dtype,
                       int8=int8, slab=slab, progress=progress, started=t0)


def _check_out(out) -> Path:
    out = Path(out)
    if not str(out).endswith(".zarr.zip"):
        raise InputError(f"-o {out}: a field is written as <name>.zarr.zip")
    return out


def _ready(name: str) -> EncoderSpec:
    """The spec, once its weights and the writer are known to be there - before minutes of work."""
    spec = resolve(name)
    if spec.weights and not W.installed(spec):
        total = sum(wf.size for wf in spec.weights) / 1e9
        raise InputError(f"{spec.name}: its weights are not installed ({total:.1f} GB) - run `haversack weights fetch "
                         f"{spec.name}` (or `--from FILE` for a copy you have)")
    try:
        import feldglas.store  # noqa: F401 - the writer
    except ImportError:
        raise InputError("encoding writes through feldglas: install haversack[encode]") from None
    return spec


def encode_file(name: str, path, out, *, identity: dict, device: str = "auto", dtype: str | None = None,
                int8: bool = False, slab: int = 16, progress=None, cancel=None, started: float | None = None,
                task_weights=None) -> dict:
    """Encode an input ALREADY on disk (a file or a DICOM folder) into ``out``. ``identity`` is what
    the field records of its input - a server passes the job's (source, identifier, digest), not
    the scratch path it staged the bytes at. ``cancel``, when given, is checked between steps.
    ``task_weights``: where an nnU-Net encoder's task weights are installed (a server's own root)."""
    import numpy as np
    out = _check_out(out)
    spec = _ready(name)
    from .. import io
    from ..resample import resolve_device
    say = progress or (lambda m: None)
    check = (lambda: cancel.check()) if cancel is not None and hasattr(cancel, "check") else (lambda: None)
    t0 = started if started is not None else time.time()
    path = Path(path)
    image = io.read_image(path)
    family = importlib.import_module(f".{spec.family}", __package__)
    dev = resolve_device(device)
    import torch
    if dtype is None:
        dtype = "fp32" if dev.type == "cpu" else "fp16"
    if dtype == "fp16" and dev.type == "cpu":
        raise InputError("fp16 on the CPU is refused: its Conv3d has no fast fp16 kernel - use --dtype fp32")
    tdt = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
    t1 = time.time()
    check()
    model = family.load(spec, W.directory(spec), dev, tdt, task_weights=task_weights)             # before prepare: an nnU-Net input depends on it
    say(f"{spec.name}: preparing {path.name}")
    prepared = family.prepare(spec, image, model)
    check()
    say(f"{spec.name}: encoding on {dev.type} ({dtype}), model grid {tuple(prepared.grid['shape'])}")
    t2 = time.time()
    tokens = family.run(spec, model, prepared, dev, tdt, slab=slab)
    t3 = time.time()
    check()
    field = field_of(spec, tokens, prepared, identity)
    from feldglas.store import write_field
    out.parent.mkdir(parents=True, exist_ok=True)
    write_field(out, field, token_dtype=np.int8 if int8 else np.float16)
    return {"field": str(out), "bytes": out.stat().st_size, "encoder": spec.name, "revision": spec.revision,
            "license": spec.license, "device": dev.type, "dtype": dtype, "int8": int8,
            "model_grid": list(prepared.grid["shape"]), "tokens": [int(len(t)) for t in tokens],
            "seconds": {"read": round(t1 - t0, 2), "load_and_prepare": round(t2 - t1, 2), "encode": round(t3 - t2, 2),
                        "total": round(time.time() - t0, 2)}}
