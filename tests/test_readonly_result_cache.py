"""A result cache this process cannot WRITE is still one it can read - result and all.

Found 2026-09-20, smoking the results listing over a cache mounted read-only, and confirmed
with a probe: ``ResultCache.get`` did its LRU touch and its read of ``result.json`` inside ONE
``try``, so where the touch was refused - a read-only filesystem (EROFS), a cache directory
another uid owns (EPERM) - the read was skipped and every hit came back ``(labels, {})``.
A read-only cache is a supported way to run (``_take_lease``: "a lease this process cannot
WRITE ... lets the read proceed unleased"; ``_entry_lock``: "a read-only cache: shared";
``SeriesCache`` has treated its own touch as best effort since 2026-09-06), and nothing
FAILED, which is what made it a defect - three things quietly answered differently:

1. the ETag fell back from the content digest to the key, so ``If-None-Match`` from a client
   holding those very bytes got the whole download again instead of a 304;
2. a ``result:<key>`` reference was refused ``result_unreadable``, with the advice to
   recompute a result that was fine (and that a read-only cache could not republish);
3. the job result route takes an entry only when its digest is the job's (``same_output``),
   so it fell through to the job's scratch copy - 410 "purged" once that was gone, with the
   bytes sitting in the entry.

Two doubles: ``os.utime`` alone refused, which is the probe that found it, and ``_ReadOnly``,
which refuses EVERY write under the cache root the way a read-only mount does - so the
lease, the entry lock and the touch are all refused together, as they really are.
"""
from __future__ import annotations

import builtins
import contextlib
import errno
import io
import json
import os
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve  # noqa: E402
from haversack.content import digest_file  # noqa: E402
from haversack.serve import (RESULT_NAME, LocalExecutor, ResultCache, create_app,  # noqa: E402
                             etag_of, same_output)

from test_job_result_cache import _NrrdSeg, _Segmenter, _done_job, _drop_scratch  # noqa: E402
from test_serve import volume_bytes  # noqa: E402

U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
PATH = f"/v1/idc/{U}/total_fast/labels.seg.nrrd"
KEY = "k" * 64


class _ReadOnly:
    """Every write under ``root`` refused with ``err``, as a read-only mount (EROFS) or
    another user's directory (EPERM) refuses it: touching, creating, opening for write,
    renaming, removing. Reads go through. ``refused`` records what was tried - a test that
    refused nothing proved nothing."""

    WRITES = ("utime", "mkdir", "rename", "replace", "unlink", "remove", "rmdir", "link")

    def __init__(self, root, err: int = errno.EROFS):
        self.root, self.err, self.refused = str(root), err, []
        self._stack = contextlib.ExitStack()

    def _inside(self, path) -> bool:
        try:
            p = os.fspath(path)
        except TypeError:                      # a file descriptor
            return False
        p = p.decode() if isinstance(p, bytes) else p
        return p == self.root or p.startswith(self.root + os.sep)

    def _refuse(self, what: str, path):
        self.refused.append((what, os.path.basename(os.fspath(path))))
        raise OSError(self.err, os.strerror(self.err), os.fspath(path))

    def __enter__(self):
        def guard(name):
            real = getattr(os, name)

            def call(path, *a, **k):
                if self._inside(path) or (a and name in ("rename", "replace", "link")
                                          and self._inside(a[0])):
                    self._refuse(name, path)
                return real(path, *a, **k)
            return call
        for name in self.WRITES:
            self._stack.enter_context(mock.patch.object(os, name, guard(name)))
        real_os_open, real_open = os.open, io.open

        def os_open(path, flags, *a, **k):
            if self._inside(path) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                                               | os.O_TRUNC | os.O_APPEND):
                self._refuse("open", path)
            return real_os_open(path, flags, *a, **k)

        def open_(file, mode="r", *a, **k):
            if self._inside(file) and set(mode) & set("wax+"):
                self._refuse("open", file)
            return real_open(file, mode, *a, **k)
        self._stack.enter_context(mock.patch.object(os, "open", os_open))
        self._stack.enter_context(mock.patch.object(io, "open", open_))
        self._stack.enter_context(mock.patch.object(builtins, "open", open_))
        return self

    def __exit__(self, *exc):
        self._stack.close()


def _published(tmp_path, key=KEY):
    """A cache holding one real publication, and the result it was published with."""
    cache = ResultCache(tmp_path / "rc")
    src = tmp_path / RESULT_NAME
    _NrrdSeg().save(src)
    result = {"outputs": [{"name": "labels", "sha256": digest_file(src)}],
              "volumes_ml": {"spleen": 1.0}}
    cache.put(key, src, result, {"task": "total_fast", "identity": [f"idc:{U}"]})
    return cache, src, result


# -- the cache -------------------------------------------------------------------------

@pytest.mark.parametrize("err", [errno.EROFS, errno.EPERM])
def test_a_refused_lru_touch_does_not_cost_the_result(tmp_path, err):
    """The probe that found it: only the touch refused. The result used to come back ``{}``,
    and with it the ETag fell from the content digest to the key."""
    cache, src, result = _published(tmp_path)
    assert cache.get(KEY)[1] == result                  # a cache it can write: as ever
    with mock.patch("os.utime", side_effect=OSError(err, os.strerror(err))) as touch:
        path, got = cache.get(KEY)
    # of the KEY directory, which eviction orders by - the lease's own touch (Path.touch
    # calls os.utime too) would satisfy a bare `touch.called`
    assert any(c.args and c.args[0] == cache.root / KEY for c in touch.call_args_list), \
        "the LRU touch is still ATTEMPTED - only its failure is forgiven"
    assert got == result
    assert Path(path).read_bytes() == src.read_bytes()
    assert etag_of(KEY, got) == f'"{result["outputs"][0]["sha256"]}"' != etag_of(KEY, {})
    assert same_output(result, got)


@pytest.mark.parametrize("err", [errno.EROFS, errno.EPERM])
def test_a_cache_that_refuses_every_write_reads_whole(tmp_path, err):
    """As it really is: the lease, the entry lock's create and the touch are refused
    TOGETHER. A second reader object, as another process - another user's server, a twin
    over a read-only mount - would be."""
    cache, src, result = _published(tmp_path)
    with _ReadOnly(cache.root, err) as ro:
        path, got = ResultCache(cache.root).get(KEY)
        assert Path(path).read_bytes() == src.read_bytes()
    assert got == result
    assert {what for what, _ in ro.refused} >= {"utime"}, ro.refused
    assert ResultCache(cache.root).get("absent" * 8) is None


def test_a_read_still_touches_a_cache_it_can_write(tmp_path):
    """The other half of "best effort": the touch is forgiven, not dropped. The key
    directory's mtime is what eviction orders by (``evict``), so a read moves it to now."""
    cache, _, _ = _published(tmp_path)
    long_ago = 1_600_000_000
    os.utime(cache.root / KEY, (long_ago, long_ago))
    assert cache.get(KEY) is not None
    assert os.stat(cache.root / KEY).st_mtime > long_ago + 10 ** 6


@pytest.mark.parametrize("body", [b"{not json", b"\xff\xfe\x00 not utf-8", b"[1, 2]", b"null",
                                  None])
def test_an_empty_result_now_means_result_json_itself_is_bad(tmp_path, body):
    """``{}`` is still the answer for a ``result.json`` that is missing, unreadable or not
    what ``put`` writes - which is what ``same_output`` always took it to mean ("an entry
    whose result.json is unreadable reads as {}"). Bytes that are not UTF-8 raised out of
    ``get`` (a 500), and a JSON list crashed ``etag_of``: neither is an object to read."""
    cache, src, _ = _published(tmp_path)
    where = cache._resolve(KEY, lease=False) / "result.json"
    where.unlink() if body is None else where.write_bytes(body)
    path, got = cache.get(KEY)
    assert got == {} and Path(path).read_bytes() == src.read_bytes()
    assert etag_of(KEY, got) == f'"{KEY[:32]}"'


# -- what it cost, route by route ------------------------------------------------------

def _server(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(volume_bytes())
        return d
    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch, artifacts=())
    return ex, TestClient(create_app(ex))


def test_a_client_holding_the_bytes_still_gets_its_304(tmp_path, monkeypatch):
    """(1) The ETag is the content digest, so ``If-None-Match`` spares the download. With
    the result read as ``{}`` the validator became the key, matched nothing the client
    held, and every revalidation was a full transfer."""
    ex, client = _server(tmp_path, monkeypatch)
    first = client.get(PATH, headers={"Prefer": "wait=30"})
    assert first.status_code == 200, first.text
    etag = first.headers["etag"]
    assert etag.startswith('"sha256:')
    with _ReadOnly(ex.cache.root) as ro:
        again = client.get(PATH)
        assert again.status_code == 200 and again.headers["etag"] == etag
        assert again.content == first.content
        assert client.head(PATH).status_code == 200
        fresh = client.get(PATH, headers={"If-None-Match": etag})
        assert fresh.status_code == 304, (fresh.status_code, fresh.headers.get("etag"))
        assert fresh.content == b""
    assert ro.refused, "nothing was refused: this was not a read-only run"
    ex.close()


def test_a_result_reference_resolves_over_a_read_only_cache(tmp_path, monkeypatch):
    """(2) ``result:<key>`` is resolved through the same ``get``: with ``{}`` for a result
    the entry "records no outputs", and the refusal told the caller to recompute it."""
    ex, client = _server(tmp_path, monkeypatch)
    assert client.get(PATH, headers={"Prefer": "wait=30"}).status_code == 200
    key = serve.result_key((f"idc:{U}",), "total_fast", {},
                           serve.weights_versions_of(ex.segmenter, "total_fast"))
    digest = ex.cache_get(key)[1]["outputs"][0]["sha256"]
    with _ReadOnly(ex.cache.root) as ro:
        assert ex.sources["result"].pin(key) == f"{key}!labels@{digest}"
    assert ro.refused, "nothing was refused: this was not a read-only run"
    ex.close()


@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_a_finished_jobs_result_is_not_purged_because_its_cache_is_read_only(tmp_path, fmt):
    """(3) The job route takes the entry only when both digests are known and equal. With
    ``{}`` they never were, so it fell through to the job's scratch copy - and answered
    410 "purged" once that was gone, with the bytes sitting in the entry."""
    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    client = TestClient(create_app(ex))
    jid, s = _done_job(client, ex)
    entry = Path(ex.cache_get(s["key"])[0]).read_bytes()
    _drop_scratch(ex, jid)
    with _ReadOnly(ex.cache.root) as ro:
        r = client.get(f"/v1/jobs/{jid}/result" + (f"?format={fmt}" if fmt else ""))
        assert r.status_code == 200, r.text
        if fmt is None:
            assert r.content == entry
            assert r.headers["etag"] == f'"{s["result"]["outputs"][0]["sha256"]}"'
    assert ro.refused, "nothing was refused: this was not a read-only run"
    ex.close()
