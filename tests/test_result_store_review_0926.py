"""The shared result store's listing, publication and job-result defects a review reproduced
on 2026-09-26 (objectcache.py and the Modal step-7 paths), each pinned here.

Most of them hid behind obstore's MemoryStore, whose last-modified has microseconds: S3 and
R2 report it in WHOLE SECONDS, so two writes to one ref inside a second carry one stamp.
These tests use a store whose clock never ticks at all - every object reports the same
last-modified - which is that coarse clock at its worst, and deterministic: a test that
waits for the start of a second to land its writes inside it is flaky under load.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pytest

from provender.disk import DiskStore

from haversack import objectcache
from haversack.objectcache import SharedResultCache
from haversack.serve import ListingMemo, ResultCache, ResultsNotVisible

FROZEN = dt.datetime(2026, 9, 26, 12, 0, 0, tzinfo=dt.timezone.utc)


class _FrozenClock(DiskStore):
    """A directory whose objects all report one last-modified, as a whole-second store
    reports every write made within one second. Its listing carries no ETag, as provender's
    DiskStore's does not: the memo must then ask a HEAD."""

    def _frozen(self, m):
        return {**m, "last_modified": FROZEN}

    def head(self, path):
        return self._frozen(super().head(path))

    def list(self, prefix=None, *, offset=None, chunk_size=50):
        listing = super().list(prefix, offset=offset, chunk_size=chunk_size)
        listing._items = [self._frozen(o) for o in listing._items]
        return listing


class _FrozenClockTagged(_FrozenClock):
    """The same, with an ETag on every listed object - as S3, R2 and GCS list them."""

    def list(self, prefix=None, *, offset=None, chunk_size=50):
        listing = super().list(prefix, offset=offset, chunk_size=chunk_size)
        listing._items = [{**o, "e_tag": DiskStore.head(self, o["path"])["e_tag"]}
                          for o in listing._items]
        return listing


@pytest.fixture(params=[_FrozenClock, _FrozenClockTagged], ids=["untagged", "tagged"])
def hosts(request, tmp_path):
    store = request.param(tmp_path / "store")

    def host(name):
        return SharedResultCache(store, ResultCache(tmp_path / f"local-{name}"), prefix="pre/")

    def publish(cache, key, body: bytes, *, result=None, preview=None):
        src = tmp_path / "src" / hashlib.sha256(body).hexdigest()
        src.parent.mkdir(exist_ok=True)
        src.write_bytes(body)
        png = None
        if preview is not None:
            png = tmp_path / "src" / "preview.png"
            png.write_bytes(preview)
        return cache.put(key, src, result or {"outputs": [body.decode()]},
                         {"task": "ts.v2:total_fast",
                          "identity": ["idc:0be27d1c-9410-47ff-9c9f-a44b26a4bd55"],
                          "options": {}, "computed": 1.0}, preview_path=png)
    return store, host, publish


KEY = "ab" * 32


def _keys(rows):
    return [r["key"] for r in rows]


# -- 1. a deleted result stays listed ------------------------------------------------------

def test_a_deleted_result_leaves_a_remembered_listing(hosts):
    _, host, publish = hosts
    a, b = host("a"), host("b")
    publish(a, KEY, b"labels")
    memo = ListingMemo()
    assert _keys(b.list(memo=memo)[0]) == [KEY]
    assert _keys(b.list(keys=[KEY], memo=memo)[0]) == [KEY]
    assert a.delete(KEY)
    assert b.get(KEY) is None
    assert _keys(b.list(memo=memo)[0]) == [], "the full listing still lists a deleted result"
    assert _keys(b.list(keys=[KEY], memo=memo)[0]) == [], \
        "the identity listing still lists a deleted result"


def test_a_republication_in_the_same_second_is_read_again(hosts):
    _, host, publish = hosts
    a, b = host("a"), host("b")
    publish(a, KEY, b"first")
    memo = ListingMemo()
    assert b.list(memo=memo)[0][0]["bytes"] == len(b"first")
    publish(a, KEY, b"the second, longer")
    assert b.list(memo=memo)[0][0]["bytes"] == len(b"the second, longer")


# -- 2. an amended artifact stays hidden ---------------------------------------------------

def test_a_preview_amended_in_the_same_second_is_listed(hosts, tmp_path):
    _, host, publish = hosts
    a = host("a")
    gen = publish(a, KEY, b"labels")
    memo = ListingMemo()
    assert "preview" not in a.list(memo=memo)[0][0].get("links", {})
    png = tmp_path / "late.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert a.add_artifact(KEY, "preview.png", png, generation=gen)
    assert "preview" in a.list(memo=memo)[0][0].get("links", {}), \
        "a preview placed after the listing is hidden by the memo"


def test_the_memo_still_spares_the_read_of_an_unchanged_pointer(hosts, monkeypatch):
    """The validator changed; the economy must not have gone with it."""
    _, host, publish = hosts
    a, b = host("a"), host("b")
    publish(a, KEY, b"labels")
    memo = ListingMemo()
    b.list(memo=memo)
    reads = []
    real = SharedResultCache._read_pointer

    def read(cache, key, **kw):
        reads.append(key)
        return real(cache, key, **kw)
    monkeypatch.setattr(SharedResultCache, "_read_pointer", read)
    assert _keys(b.list(memo=memo)[0]) == [KEY]
    assert _keys(b.list(keys=[KEY], memo=memo)[0]) == [KEY]
    assert reads == []


# -- 5. the memo keeps whole pointers ------------------------------------------------------

def test_the_memo_keeps_only_what_a_row_needs(hosts):
    _, host, publish = hosts
    a = host("a")
    big = {"outputs": [{"name": "labels", "sha256": "sha256:" + "0" * 64}],
           "provenance": {"note": "x" * 40000}}
    publish(a, KEY, b"labels", result=big)
    memo = ListingMemo()
    rows, _ = a.list(memo=memo)
    assert _keys(rows) == [KEY]
    kept = memo._rows[KEY][2]
    size = len(json.dumps(kept, default=str))
    assert size < 2048, f"a memo row holds {size} bytes: the result document rode along"


# -- 3. a store fault shortens a listing ---------------------------------------------------

class _Flaky(DiskStore):
    """Faults (a 503) on any path holding one of ``fail``."""
    fail: tuple = ()

    def _check(self, path):
        if any(f in path for f in self.fail):
            raise ConnectionError("503 SlowDown")

    def head(self, path):
        self._check(path)
        return super().head(path)

    def get(self, path):
        self._check(path)
        return super().get(path)

    def list(self, prefix=None, *, offset=None, chunk_size=50):
        self._check(prefix or "")
        return super().list(prefix, offset=offset, chunk_size=chunk_size)


@pytest.fixture
def flaky(tmp_path):
    store = _Flaky(tmp_path / "store")
    cache = SharedResultCache(store, ResultCache(tmp_path / "local"), prefix="pre/")
    keys = [f"{i:02x}" * 32 for i in range(3)]
    for i, k in enumerate(keys):
        src = tmp_path / f"l{i}"
        src.write_bytes(f"v{i}".encode())
        cache.put(k, src, {"outputs": [f"v{i}"]}, {"task": "t", "identity": [f"upload:{i}"]})
    yield store, cache, keys
    store.fail = ()


@pytest.mark.parametrize("memo", [None, "memo"])
def test_a_fault_reading_one_pointer_is_a_503_not_a_shorter_list(flaky, memo):
    store, cache, keys = flaky
    store.fail = (f"results/{keys[1]}",)
    for ask in ({"keys": keys}, {}):
        with pytest.raises(ResultsNotVisible):
            cache.list(memo=ListingMemo() if memo else None, **ask)


def test_a_fault_listing_the_bucket_is_a_503(flaky):
    store, cache, _ = flaky
    store.fail = ("pre/results",)
    with pytest.raises(ResultsNotVisible):
        cache.list()


def test_a_lookup_of_one_key_still_reads_a_fault_as_a_miss(flaky):
    """The listing's rule is the listing's: a single read degrades to a miss, as before."""
    store, cache, keys = flaky
    store.fail = (f"results/{keys[1]}",)
    assert cache.generation(keys[1]) is None


# -- 4. publication gives up under contention ----------------------------------------------

class _Contended(DiskStore):
    """Every conditional write of a ref loses ``losses`` times before one takes - another
    writer moved the ref each time."""
    losses = 0

    def put(self, path, file, *, mode=None):
        if "/results/" in path and self.losses > 0:
            self.losses -= 1
            from obstore.exceptions import PreconditionError
            raise PreconditionError("etag moved")
        return super().put(path, file, mode=mode)


def test_a_publication_outlasts_a_contended_key_and_backs_off(tmp_path, monkeypatch):
    store = _Contended(tmp_path / "store")
    cache = SharedResultCache(store, ResultCache(tmp_path / "local"), prefix="pre/")
    pauses = []
    monkeypatch.setattr(objectcache, "_sleep", pauses.append, raising=False)
    store.losses = 24                          # more than the 16 attempts it used to allow
    src = tmp_path / "labels"
    src.write_bytes(b"labels")
    gen = cache.put(KEY, src, {"outputs": ["labels"]}, {"task": "t"})
    assert cache.generation(KEY) == gen
    assert len(pauses) == 24, "every lost race pauses before reading again"
    assert all(0 <= p <= objectcache.SWAP_BACKOFF_MAX_S for p in pauses)
    # the bound doubles: the late pauses may be far longer than the first could be
    assert max(pauses[10:]) > objectcache.SWAP_BACKOFF_S


def test_a_storm_still_ends_in_an_error_not_a_hang(tmp_path, monkeypatch):
    store = _Contended(tmp_path / "store")
    cache = SharedResultCache(store, ResultCache(tmp_path / "local"), prefix="pre/")
    monkeypatch.setattr(objectcache, "_sleep", lambda s: None, raising=False)
    store.losses = 10 ** 6
    src = tmp_path / "labels"
    src.write_bytes(b"labels")
    with pytest.raises(RuntimeError, match="raced"):
        cache.put(KEY, src, {"outputs": ["labels"]}, {"task": "t"})


# -- Modal: 1 through the real modal_app code, and 6 ----------------------------------------

@pytest.fixture
def modal_store(monkeypatch, tmp_path):
    pytest.importorskip("modal")
    from haversack import modal_app as m
    from test_modal_result_store import _NoVolume
    from test_worker_volume_view import _HidingVolume
    store = _FrozenClockTagged(tmp_path / "bucket")
    jobs = {}
    (tmp_path / "scratch").mkdir()
    scratch = _HidingVolume(tmp_path / "scratch")
    monkeypatch.setattr(m, "RESULT_STORE", "s3://bucket/results")
    novol = _NoVolume()
    monkeypatch.setattr(m, "cache_vol", novol)
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "no-cache-volume"))
    monkeypatch.setattr(m, "MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setattr(m, "jobs_dict", jobs)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(scratch.root))
    monkeypatch.setattr(m, "scratch_vol", scratch)
    monkeypatch.setattr(m, "JOB_LOCAL_ROOT", str(tmp_path / "local"), raising=False)
    monkeypatch.setattr(m, "ARTIFACTS", set())
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_sweep_due", lambda: False)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)
    monkeypatch.setattr(m, "_TWIN", False)
    monkeypatch.setattr(m, "_listing_memo", [])

    def container(name, read_only=False):
        view = SharedResultCache(store, ResultCache(tmp_path / f"copy-{name}"),
                                 check=not read_only, read_only=read_only)
        monkeypatch.setattr(m, "_results_held", {read_only: view})
        return view
    yield m, jobs, container
    assert novol.touched == [], f"the cache volume was touched: {novol.touched}"


def _computed(m, jobs, container, jid="j00"):
    from test_worker_volume_view import _Ctx, _submit
    container("worker")
    body = _submit(m, jobs, jid)
    m._execute_job(_Ctx(), jid)
    assert jobs[jid]["state"] == "done", jobs[jid].get("error")
    return body


def test_modal_a_deleted_result_leaves_the_api_containers_listing(modal_store):
    m, jobs, container = modal_store
    _computed(m, jobs, container)
    container("api")
    assert _keys(m._list_cache(keys=["key-j00"])[0]) == ["key-j00"]
    assert _keys(m._list_cache()[0]) == ["key-j00"]
    assert m.ModalExecutor().cache_delete("key-j00")
    assert m._read_cache("key-j00") is None
    assert m._list_cache(keys=["key-j00"])[0] == []
    assert m._list_cache()[0] == []


def test_modal_another_containers_delete_leaves_this_ones_listing(modal_store):
    """No memo drop can reach another container: the validator must."""
    m, jobs, container = modal_store
    _computed(m, jobs, container)
    container("api")
    assert _keys(m._list_cache()[0]) == ["key-j00"]
    api_memo = m._listing_memo[0]
    elsewhere = container("elsewhere")
    assert elsewhere.delete("key-j00")
    container("api")
    assert m._listing_memo[0] is api_memo
    assert m._list_cache()[0] == []


def test_modal_a_cache_hit_jobs_result_outlives_a_republication(modal_store, tmp_path):
    m, jobs, container = modal_store
    body = _computed(m, jobs, container)
    container("api")
    hit = m.ModalExecutor()._cache_record("key-j00")
    assert hit is not None
    # the record a submit answered from the cache writes (modal_app's ModalExecutor.submit)
    jobs["hit1"] = {"id": "hit1", "task": "ts.v2:total_fast", "state": "done",
                    "cached": True, "created": time.time(), "finished": time.time(),
                    "result": hit[1], "cache_key": "key-j00", "deliverables": []}
    # another container republishes the key with OTHER bytes (a no-cache recompute)
    other = tmp_path / "other.seg.nrrd"
    other.write_bytes(b"labels of a recompute")
    from haversack.content import digest_file
    worker = container("worker2")
    worker.put("key-j00", other, {"outputs": [{"name": "labels",
                                               "sha256": digest_file(other)}]},
               {"task": "ts.v2:total_fast"})
    container("api2")
    state, path = m.ModalExecutor().result_file("hit1")
    assert state == "done"
    assert path is not None, "the job's own bytes are in the store's history: not gone"
    assert Path(path).read_bytes() == b"labels of " + body
    # and again, from what the first answer placed
    assert Path(m.ModalExecutor().result_file("hit1")[1]).read_bytes() == b"labels of " + body


# -- 7. a full local disk is reported as this host's, not the store's -----------------------

def test_a_full_local_disk_is_not_reported_as_the_store(tmp_path, monkeypatch, capsys):
    import errno
    from obstore.store import MemoryStore
    store = MemoryStore()
    a = SharedResultCache(store, ResultCache(tmp_path / "a"), prefix="pre/")
    b = SharedResultCache(store, ResultCache(tmp_path / "b"), prefix="pre/")
    src = tmp_path / "labels"
    src.write_bytes(b"labels")
    a.put(KEY, src, {"outputs": ["labels"]}, {"task": "t"})

    def full(digest, dest):
        raise OSError(errno.ENOSPC, "No space left on device", str(dest))
    monkeypatch.setattr(b.blobs, "fetch", full)
    # a store fault was just reported: its throttle must not swallow the disk's report
    monkeypatch.setattr(objectcache, "_warned_at", time.time())
    monkeypatch.setattr(objectcache, "_local_warned_at", 0.0, raising=False)
    capsys.readouterr()
    assert b.get(KEY) is None                  # still a miss: the bytes cannot be kept
    err = capsys.readouterr().err
    assert "local copy" in err and "unreachable" not in err, err
