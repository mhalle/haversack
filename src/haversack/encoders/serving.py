"""What the server needs to know of an encoder, without torch: its canonical name, the part of
a result key its weights are, the options a job may state, and the record a finished field is
published with. The compute is ``pipeline.encode_file``, in the worker.

An encode job is a job of KIND ``encode`` (2026-09-23; feldglas docs/embedding-field.md, "Encoding
moves into haversack", phase 2): the same queue, cache, lifetimes and routes as a segmentation,
with an embedding field as the primary output instead of labels. Names are resolved HERE, never
through the task catalog: ``ts.v2:total_fast`` is a task and an encoder, and the verb decides.
"""
from __future__ import annotations

from ..errors import InputError
from .registry import EncoderSpec, resolve

#: The encode path's own cache epoch: bumped when the same input and weights would give other
#: tokens (a preparation, tiling or placement change), and seen by no segmentation key.
ENCODE_EPOCH = "1"

#: What an encode job may state, and nothing else is accepted: ``int8`` changes the stored
#: bytes. The device and precision follow the worker, as a segmentation's do.
OPTIONS = {"int8": bool}


def canonical(name) -> str | None:
    """The encoder ``name`` resolves to (aliases and ``@revision`` pins included), or None."""
    try:
        return resolve(str(name)).name
    except InputError:
        return None


def field_versions(segmenter, name) -> list:
    """The key's weights component for an encoder. Downloaded weights are pinned, so their
    digests ARE the version; an nnU-Net encoder's are its task's, read from the install
    sidecars as ``weights_versions_of`` reads them (but not its tile step, crop or auxiliary
    rule: the encoder tiles its own way and never argmaxes). ``unknown`` when not installed,
    exactly as a segmentation keys before its first install."""
    spec = resolve(str(name))
    out = [f"{spec.name}@{spec.revision}"]
    if spec.weights:
        out += [f"{wf.name}=sha256:{wf.sha256}" for wf in spec.weights]
    else:
        try:
            entries = segmenter.describe(spec.uses_task).get("weights_installed") or []
            out += [f"{e.get('id')}={e.get('version') or e.get('sha256') or 'unknown'}"
                    + (f"/{e['model']}" if e.get("model") else "") for e in entries] or ["unknown"]
        except Exception:
            out.append("unknown")
    return out + [f"encode@epoch={ENCODE_EPOCH}"]


def installed_locally(spec: EncoderSpec, task_weights=None) -> bool:
    """Whether ``spec`` can encode on THIS machine: its downloaded weights verified, or an
    nnU-Net encoder's model folder resolvable under ``task_weights`` (default: the default
    root). The CLI's answer; a server answers from its own Segmenter's describe."""
    from . import weights as W
    if spec.weights:
        return W.installed(spec)
    try:
        from ..tasks import resolve_model_folder, weights_root
        resolve_model_folder(spec.options["dataset"], model_root=task_weights or weights_root("ts"))
        return True
    except Exception:
        return False


def input_record(ident: str | None, path) -> dict:
    """What a server's field records of its input, for either executor: the job's identity,
    and the bytes' digest. An input NAMED by its digest (an upload, a stored file or DICOM
    tree) records that digest - never a hash of what staging made of it: a stored tree reaches
    a worker as its decoded copy, whose digest is of bytes nobody sent (review, 2026-09-23)."""
    from pathlib import Path
    from ..content import is_digest
    from .pipeline import _identity
    ident = ident or "upload"
    if is_digest(ident):
        return {"input": ident, "digest": ident}
    return _identity(ident, Path(path))


def ensure_weights(name, progress=None) -> list:
    """A SERVER's first use of an encoder: its pinned weights fetched if missing (digest
    checked, as ``haversack weights fetch`` does), as a server installs a segmentation task's
    weights on first use. The command line does not: there a 1.6 GB download under a license
    is an explicit ``weights fetch``. Nothing for an nnU-Net encoder (its task's weights)."""
    from . import weights as W
    spec = resolve(str(name))
    if spec.weights and not W.installed(spec):
        return W.fetch(spec, progress=progress)
    return []


def validate_options(options: dict) -> dict:
    """The options of an encode job, checked against :data:`OPTIONS`; raises ``RequestError``."""
    from ..errors import RequestError
    extra = sorted(set(options) - set(OPTIONS))
    if extra:
        raise RequestError("unknown_parameter",
                           f"an encode job takes only {', '.join(sorted(OPTIONS))}; not {', '.join(extra)}",
                           parameter=extra[0], known=sorted(OPTIONS))
    for k, t in OPTIONS.items():
        if k in options and not isinstance(options[k], t):
            raise RequestError("invalid_parameter", f"{k} must be {t.__name__}, not {type(options[k]).__name__}",
                               parameter=k)
    return dict(options)


def field_payload(report: dict, path) -> dict:
    """The record a finished field is published with (``result.json``): the output's digest
    first, as every result's is (``etag_of``, ``same_output`` and ``result:`` read
    ``outputs[0]``), with ``kind: field`` so a reference cannot bind it where labels belong."""
    from pathlib import Path
    from ..content import digest_file
    p = Path(path)
    return {"outputs": [{"name": "field", "kind": "field", "sha256": digest_file(p),
                         "bytes": p.stat().st_size}],
            "encoder": report.get("encoder"), "revision": report.get("revision"),
            "license": report.get("license"), "model_grid": report.get("model_grid"),
            "tokens": report.get("tokens"), "int8": report.get("int8"),
            "timings": report.get("seconds"), "device": report.get("device"), "dtype": report.get("dtype")}


def describe(spec: EncoderSpec, installed: bool | None = None) -> dict:
    """An encoder as ``GET /v1/encoders`` and ``haversack encoders --json`` state it."""
    from .pipeline import attribution_of
    return {"name": spec.name, "family": spec.family, "description": spec.description,
            "revision": spec.revision, "license": spec.license, "modality": spec.modality,
            "weights": [{"name": wf.name, "bytes": wf.size, "source": wf.source, "sha256": wf.sha256}
                        for wf in spec.weights],
            "uses_task": spec.uses_task, "installed": installed,
            "lattices": [{"layer": l.layer, "kernel": list(l.kernel), "channels": l.width,
                          "reach_mm": l.receptive_mm} for l in spec.lattices],
            "options": {k: t.__name__ for k, t in OPTIONS.items()},
            "attribution": attribution_of(spec)}
