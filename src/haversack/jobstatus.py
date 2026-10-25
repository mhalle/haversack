"""A job's status record, the one way it is framed on a stream, and the one way it is read back.

The record is what ``GET /v1/jobs/{id}`` answers: ``state``, ``progress`` (a
:class:`~haversack.progress.Progress`'s fields), timestamps, ``input_identity``, ``error``, and
once done a ``result`` built by :func:`result_payload`. ``/v1/jobs/{id}/events`` streams it as
server-sent events, and ``haversack segment --events`` writes the same records, framed the same
way, to stdout - so a program driving haversack, whether it runs the command line or talks to a
server, reads one format with one parser (:func:`read`). Until 2026-10-25 the command line had a
format of its own (``--json-progress``: event names, field names and a result of its own, on
stderr behind a prefix) beside the server's; that was a second description of the same facts,
and two descriptions drift.

What differs between the two is only what a server has and a local run does not: a job ``id``,
a cache ``key``, the server's ``links``. A local run's output location is ``links.result`` as a
``file://`` URI, where a server's is the URL to fetch.

Stdlib only; the command line uses it without the server extra.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

#: The SSE event name every record is sent under.
EVENT = "status"


def result_payload(seg, labels_path) -> dict:
    """A finished job's public result, built ONCE for every producer - both executors and the
    command line.

    They had drifted already - the Modal side reported `timings` and the local
    side did not - which is the same shape as the bug where an engine worker and
    the API disagreed about weights identity: two hand-written copies of one
    thing.

    `outputs` is the content digest of what the job produced. It is what lets a client skip
    re-downloading a result it already holds (it IS the ETag), and what lets one job's output
    become another's input without the bytes taking a round trip out through the client and
    back.
    """
    from .content import digest_file
    out = {"names": {int(k): v for k, v in seg.schema.names.items()},
           "volumes_ml": {k: round(float(v), 2) for k, v in seg.volumes_ml().items()},
           "provenance": seg.provenance}
    timings = getattr(seg, "timings", None)
    if timings:
        out["timings"] = {k: round(float(v), 3) for k, v in timings.items()}
    p = Path(labels_path) if labels_path is not None else None
    if p is not None and p.exists():
        out["outputs"] = [{"name": "labels", "sha256": digest_file(p),
                           "bytes": p.stat().st_size}]
    return out


def ranked_payload(seg, path) -> dict:
    """A finished ranked job's public result (``result.json``): the store's digest first, as every
    result's is (``etag_of``, ``same_output`` and ``result:`` read ``outputs[0]``), with ``kind:
    rankfield`` so a reference cannot bind a store where an image or labels belong - and the run's
    names, volumes and provenance, which the store was computed with."""
    from .content import digest_file
    out = result_payload(seg, None)               # names, volumes, provenance, timings; no labels file
    p = Path(path)
    out["outputs"] = [{"name": "rankfield", "kind": "rankfield", "sha256": digest_file(p),
                       "bytes": p.stat().st_size}]
    return out


def progress_record(p) -> dict:
    """A :class:`~haversack.progress.Progress` as a record's ``progress`` - its fields, as the
    server stores it. Anything else a run reports (a source's download message) is a ``fetch``
    stage, the stage a server shows for the same download."""
    from .progress import Progress
    return asdict(p if is_dataclass(p) else Progress(stage="fetch", detail=str(p)))


def record(task, state: str, **fields) -> dict:
    """A status record for a job with no server behind it - a command-line run. ``fields`` are
    the record's own (``progress``, ``created``, ``started``, ``finished``, ``input_identity``,
    ``error``, ``result``, ``links``, ``kind``); one left as None is left out, as the server
    leaves out what it does not know."""
    return {"task": str(task), "state": state, **{k: v for k, v in fields.items() if v is not None}}


def sse(rec: dict) -> str:
    """One record as server-sent events text: what ``/v1/jobs/{id}/events`` sends and what
    ``segment --events`` writes."""
    return f"event: {EVENT}\ndata: {json.dumps(rec, default=str)}\n\n"


def read(lines):
    """The records in a stream of server-sent-events lines - an HTTP body's or a process's
    stdout, ``str`` or ``bytes``. Other events, comments (a server's keepalives) and anything
    that is not a ``data:`` line are passed over."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("data:"):
            yield json.loads(line[5:].strip())
