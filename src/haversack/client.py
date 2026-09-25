"""A small client for the haversack REST job protocol (`haversack remote ...`).

Talks to any server that implements the contract in :mod:`haversack.serve` - a local
`haversack serve`, a lab GPU box, or the Modal deployment. Progress arrives over the SSE
stream when the server offers it and silently falls back to polling: both surfaces
carry the same idempotent status snapshots, so the fallback changes latency, not
meaning. Needs ``httpx`` (the ``remote`` extra).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .errors import InputError, HaversackError
from .jobpolicy import TERMINAL


class RemoteError(HaversackError):
    """The server refused or failed a request."""


class RemoteClient:
    def __init__(self, server: str, *, token: str | None = None, timeout: float = 60.0,
                 token_source: str | None = None):
        try:
            import httpx
        except ImportError as e:
            raise InputError("the remote client needs httpx: uv sync --extra remote "
                             "(or pip install 'haversack[remote]')") from e
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.token_source = token_source          # named in a 401, so a stale or wrong
                                                  # token is traceable to where it came from
        self._httpx = httpx
        self._http = httpx.Client(base_url=server.rstrip("/"), headers=headers,
                                  timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- plumbing ------------------------------------------------------------
    def _unreachable(self, e: Exception) -> "RemoteError":
        return RemoteError(f"cannot reach {self._http.base_url}: {e.__class__.__name__}: {e}")

    def _json(self, method: str, path: str, *, _retries: int = 3, **kw) -> dict:
        for attempt in range(_retries + 1):
            try:
                r = self._http.request(method, path, **kw)
            except self._httpx.TransportError as e:    # refused, DNS, timeout: one line, no trace
                raise self._unreachable(e) from None
            if r.status_code in (429, 503) and attempt < _retries:
                try:
                    wait = float(r.headers.get("Retry-After", "5"))
                except ValueError:
                    wait = 5.0
                time.sleep(min(max(wait, 0.0), 60.0))
                continue
            break
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            if r.status_code == 401 and self.token_source:
                detail = f"{detail} (the token sent came from {self.token_source})"
            raise RemoteError(f"{method} {path} -> {r.status_code}: {detail}")
        return r.json()

    # -- the protocol --------------------------------------------------------
    def health(self) -> dict:
        return self._json("GET", "/v1/health")

    def tasks(self) -> list[str]:
        return self._json("GET", "/v1/tasks")["tasks"]

    def describe(self, task: str) -> dict:
        return self._json("GET", f"/v1/tasks/{task}")

    def segments(self, query: str, *, mode: str = "words", field: str = "key",
                 catalog: str | None = None, modality: str | None = None,
                 limit: int | None = None, offset: int | None = None,
                 count_only: bool = False) -> dict:
        """Which of the server's tasks produce a segment, and with what label value
        (``GET /v1/segments``): word prefixes by default, or ``mode="glob"``. Paged: pass
        the answer's ``next_offset`` as ``offset`` for the next page; ``count_only`` sizes a
        search first."""
        params = {"q": query, "mode": mode, "field": field}
        for k, v in (("catalog", catalog), ("modality", modality), ("limit", limit),
                     ("offset", offset)):
            if v is not None:
                params[k] = v
        if count_only:
            params["count_only"] = "true"
        return self._json("GET", "/v1/segments", params=params)

    def segmentations(self, *, identity=None, task: str | None = None,
                      limit: int | None = None, cursor: str | None = None) -> dict:
        """One page of the results the server holds (``GET /v1/segmentations``, authorized):
        ``{"segmentations": [...], "next_cursor": ...}``, newest published first.

        ``identity`` is one input or several - ``"idc:<crdc_series_uuid>"``, any
        ``<source>:<identifier>`` the server mounts, or a content digest - and the answer
        is the union of their results; ``task`` keeps one task's. The server computes an
        identity filter rather than searching for it, so it finds what has a path (default
        options and the grid variants, single-input) and is as fast on a cache of thousands
        as on an empty one. Pass ``next_cursor`` back as ``cursor`` for the next page, or
        use :meth:`iter_segmentations`, which does."""
        ids = [identity] if isinstance(identity, str) else list(identity or [])
        params = [("identity", i) for i in ids]
        for k, v in (("task", task), ("limit", limit), ("cursor", cursor)):
            if v is not None:
                params.append((k, v))
        page = self._json("GET", "/v1/segmentations", params=params)
        if "next_cursor" not in page and (ids or task or cursor):
            # a server from before 2026-09-20 ignores the parameters and answers with its
            # whole listing (the newest 500): never hand that back as if it were filtered
            raise RemoteError("this server's /v1/segmentations predates filters and paging "
                              "(its answer has no next_cursor), so it returned its whole "
                              "listing; upgrade the server, or list without a filter")
        return page

    def iter_segmentations(self, *, identity=None, task: str | None = None,
                           page_size: int | None = None):
        """Every row of :meth:`segmentations`, following the server's cursors to the end.
        The cursor is a position, not an offset, so what is published while this runs
        neither repeats a row nor skips one: it sorts ahead of where the walk already is."""
        cursor = None
        while True:
            page = self.segmentations(identity=identity, task=task, limit=page_size,
                                      cursor=cursor)
            yield from page.get("segmentations") or []
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def embeddings(self, *, identity=None, encoder: str | None = None,
                   limit: int | None = None, cursor: str | None = None) -> dict:
        """One page of the embedding fields the server holds (``GET /v1/embeddings``,
        authorized): ``{"embeddings": [...], "next_cursor": ...}``, newest published first.
        ``identity`` and paging as in :meth:`segmentations`; ``encoder`` keeps one encoder's.
        A row with a path has ``links.embedding``, which a plain GET downloads."""
        ids = [identity] if isinstance(identity, str) else list(identity or [])
        params = [("identity", i) for i in ids]
        for k, v in (("encoder", encoder), ("limit", limit), ("cursor", cursor)):
            if v is not None:
                params.append((k, v))
        return self._json("GET", "/v1/embeddings", params=params)

    def iter_embeddings(self, *, identity=None, encoder: str | None = None,
                        page_size: int | None = None):
        """Every row of :meth:`embeddings`, following the server's cursors to the end."""
        cursor = None
        while True:
            page = self.embeddings(identity=identity, encoder=encoder, limit=page_size,
                                   cursor=cursor)
            yield from page.get("embeddings") or []
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def submit(self, image, task: str, *, deliverables=None, kind: str | None = None, **options) -> str:
        """``image`` is a local file to upload, or ``"<source>:<identifier>"``
        (e.g. ``"idc:<crdc_series_uuid>"``) to have the server fetch the input
        from one of its registered data sources. A path that exists locally
        always wins over the shorthand reading.

        ``deliverables`` names what the server renders beside the labels - ``["preview",
        "statistics"]``, any subset, ``[]`` for none; None sends nothing and the server
        renders its own set (``GET /v1/health`` lists it; a name outside it is refused).
        A keyword of its own and a form field of its own, NOT one of ``options``: those
        are part of the result's key, and declining a preview must not name another
        result.

        ``kind="embed"`` makes the job an embedding field of ``image`` with the ENCODER named
        ``task`` (``GET /v1/encoders``), whose only option is ``int8``; ``kind="ranked"`` makes
        it the task's ranked store (2026-09-24; no options, no deliverables); None sends
        nothing, which is a segmentation - the form every server before 2026-09-23 understands."""
        data = {"task": task, "options": json.dumps(options)}
        if kind is not None:
            data["kind"] = kind
        if deliverables is not None:
            data["deliverables"] = json.dumps(list(deliverables))
        img = str(image)
        prefix = img.split(":", 1)[0] if ":" in img else ""
        if prefix.isidentifier() and prefix.islower() and not Path(img).exists():
            ident = img.split(":", 1)[1]
            src = {"kind": prefix, "id": ident}
            if prefix == "idc":
                src["crdc_series_uuid"] = ident
            data["source"] = json.dumps([src])
            r = self._json("POST", "/v1/jobs", data=data)
        else:
            p = Path(image)
            with open(p, "rb") as f:
                r = self._json("POST", "/v1/jobs", data=data,
                               files={"file": (p.name, f, "application/octet-stream")})
        return r["id"]

    def status(self, job_id: str) -> dict:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self._json("DELETE", f"/v1/jobs/{job_id}")

    def fetch(self, job_id: str, output) -> Path:
        """Download a finished job's result to ``output``, in the format its name says.

        The server holds labels as ``.seg.nrrd``, a field as ``.zarr.zip`` and a ranked store
        as ``.duckn.zip`` (named so in its Content-Disposition; both zips), and converts
        labels to NIfTI on request (``?format=nii.gz``; the label values survive, the segment
        names do not). This asked for nothing, so ``-o labels.nii.gz`` wrote NRRD bytes under
        a NIfTI name that every reader then refused (found 2026-09-23; the same in 0.12.4).
        Now a ``.nii.gz`` or ``.nii`` name asks for the conversion, a ``.nrrd``, ``.zarr.zip``
        or ``.duckn.zip`` name must match what the server sends or nothing is written, and
        any other name gets the server's bytes as before."""
        import os
        out = Path(output)
        name = out.name.lower()
        nifti = name.endswith((".nii.gz", ".nii"))
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        n = 0
        try:
            with self._http.stream("GET", f"/v1/jobs/{job_id}/result",
                                   params={"format": "nii.gz"} if nifti else None) as r:
                if r.status_code >= 400:
                    r.read()
                    raise RemoteError(f"result -> {r.status_code}: {r.text}")
                # the server types a field and a store as a zip, labels as anything else, and
                # names a store's download .duckn.zip
                zipped = r.headers.get("Content-Type", "").startswith("application/zip")
                store = zipped and ".duckn.zip" in r.headers.get("Content-Disposition", "")
                field = zipped and not store
                what = ("a ranked store: name the output .duckn.zip" if store else
                        "an embedding field: name the output .zarr.zip" if field else
                        "a label map: name the output .seg.nrrd, or .nii.gz for NIfTI")
                if ((name.endswith(".zarr.zip") and not field)
                        or (name.endswith(".duckn.zip") and not store)
                        or (name.endswith(".nrrd") and zipped)):
                    raise RemoteError(f"job {job_id}'s result is {what}")
                declared = r.headers.get("Content-Length")
                with open(part, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                        n += len(chunk)
        except self._httpx.TransportError as e:
            part.unlink(missing_ok=True)
            raise self._unreachable(e) from None
        if declared is not None and n != int(declared):
            part.unlink(missing_ok=True)   # truncated: leave no partial file
            raise RemoteError(f"result truncated: got {n} of {declared} bytes")
        if name.endswith(".nii"):          # the server sends NIfTI gzipped whatever is asked
            import gzip
            import shutil
            plain = out.with_name(out.name + ".plain")
            try:
                with gzip.open(part, "rb") as src, open(plain, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except BaseException:
                plain.unlink(missing_ok=True)       # no partial file, as for a truncation
                raise
            finally:
                part.unlink(missing_ok=True)
            part = plain
        os.replace(part, out)              # complete: publish atomically
        return out

    # -- progress ------------------------------------------------------------
    def events(self, job_id: str):
        """Status snapshots from the SSE stream; raises on transport failure -
        callers that need robustness use :meth:`wait`, which falls back to polling."""
        try:
            with self._http.stream("GET", f"/v1/jobs/{job_id}/events",
                                   timeout=self._httpx.Timeout(None, read=45.0)) as r:
                if r.status_code >= 400:
                    r.read()
                    raise RemoteError(f"events -> {r.status_code}")
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        yield json.loads(line[5:].strip())
        except self._httpx.TransportError as e:
            raise self._unreachable(e) from None

    def wait(self, job_id: str, *, on_status=None, poll_interval: float = 0.5) -> dict:
        """Block until the job is terminal; returns the final status. Prefers SSE,
        falls back to polling on any stream problem."""
        last = None
        try:
            for snap in self.events(job_id):
                last = snap
                if on_status:
                    on_status(snap)
                if snap["state"] in TERMINAL:
                    return snap
        except Exception:
            pass                                   # stream unavailable - poll instead
        while True:
            snap = self.status(job_id)
            if snap != last and on_status:
                on_status(snap)
            last = snap
            if snap["state"] in TERMINAL:
                return snap
            time.sleep(poll_interval)

    def encoders(self) -> dict:
        """``GET /v1/encoders``: the encoders the server embeds with."""
        return self._json("GET", "/v1/encoders")

    def embed(self, image, encoder: str, output, *, int8: bool = False, on_status=None) -> dict:
        """An embedding field of ``image`` into ``output`` (``<name>.zarr.zip``): submit an
        embedding job, wait, fetch. Returns the final status; raises on a failed job."""
        opts = {"int8": True} if int8 else {}
        jid = self.submit(image, encoder, kind="embed", **opts)
        final = self.wait(jid, on_status=on_status)
        if final["state"] == "done":
            self.fetch(jid, output)
        elif final["state"] == "failed":
            raise RemoteError(f"job {jid} failed: {final.get('error', 'unknown')}")
        return final

    def ranked(self, image, task: str, output, *, on_status=None) -> dict:
        """The task's ranked store of ``image`` into ``output`` (``<name>.duckn.zip``): submit a
        ranked job, wait, fetch. Returns the final status; raises on a failed job."""
        return self.run(image, task, output, on_status=on_status, kind="ranked")

    def run(self, image, task: str, output, *, on_status=None, deliverables=None,
            kind: str | None = None, **options) -> dict:
        """submit + wait + fetch: the whole round trip. Returns the final status.
        ``deliverables`` and ``kind`` as in :meth:`submit`."""
        jid = self.submit(image, task, deliverables=deliverables, kind=kind, **options)
        final = self.wait(jid, on_status=on_status)
        if final["state"] == "done":
            self.fetch(jid, output)
        elif final["state"] == "failed":
            raise RemoteError(f"job {jid} failed: {final.get('error', 'unknown')}")
        return final
