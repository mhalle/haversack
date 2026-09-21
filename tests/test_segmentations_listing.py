"""``GET /v1/segmentations``: filters, real paging, parallel reads - and no index.

Measured on Modal 2026-09-20 (`haversack-radar-val-cache`, 2,083 entries): listing the root
0.03 s, a stat 0.4 ms, the FIRST read of any file ~24 ms whatever its size - so a serial walk
of the cache was 161 s, the listing took no filter, and it silently kept the newest 500. The
user's decision the same day: no identity -> key index ("one more thing to keep in sync").
Three stateless mechanisms stand in for one, and each has a test here that a mutant kills:

1. a filter by identity is COMPUTED - the keys are derived and looked up by name;
2. order and paging come from names and the pointer's mtime, content is read for the page
   only, and the cursor is a position, not an offset;
3. a filter by task reads content in PARALLEL, and a long-lived server remembers
   ``meta.json`` per publication (``ListingMemo``) - validated on every use, never believed.

What is counted is what a Modal volume charges for: a filesystem double (``_Disk``) counts
every file OPENED for reading under the cache root, every stat and every directory listing.
The counts are asserted EXACTLY, never as upper bounds: a reader that went around the
double would count zero and pass a bound.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve  # noqa: E402
from haversack.serve import (CURRENT_NAME, RESULT_NAME, ListingMemo, LocalExecutor,  # noqa: E402
                             ResultCache, create_app, create_public_app, decode_cursor,
                             encode_cursor, result_key, weights_versions_of)

from test_serve import FakeSegmenter, volume_bytes  # noqa: E402

U1 = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
U2 = "1be27d1c-9410-47ff-9c9f-a44b26a4bd55"
T0 = 1_700_000_000 * 10 ** 9                      # a publication time, in ns - in the past,
                                                  # so a real publication sorts after it


class _Disk:
    """A filesystem double in front of one cache root.

    ``opened`` counts content reads as ``(key, file name)`` - the ~24 ms operation;
    ``statted`` the key directories something was stat'ed under; ``listed`` directory
    listings of the root. ``read_delay`` makes every content read slow, and ``peak`` is how
    many were in flight at once. ``on_read`` is called inside every counted read."""

    def __init__(self, monkeypatch, root, *, read_delay: float = 0.0, on_read=None):
        self.root = str(root)
        self.opened, self.statted, self.listed = Counter(), Counter(), 0
        self.peak = self._active = 0
        self._lock = threading.Lock()
        real_open, real_stat = io.open, os.stat
        real_scandir, real_listdir = os.scandir, os.listdir

        def under(path):
            try:
                p = os.fspath(path)
            except TypeError:
                return None
            p = p.decode() if isinstance(p, bytes) else p
            if p != self.root and not p.startswith(self.root + os.sep):
                return None                    # `<root>.committed` is a sibling, not inside
            return p[len(self.root):].strip(os.sep)

        def _open(file, mode="r", *a, **k):
            rel = under(file)
            if rel and "w" not in mode and "a" not in mode and "+" not in mode:
                with self._lock:
                    self.opened[(rel.split(os.sep)[0], os.path.basename(rel))] += 1
                    self._active += 1
                    self.peak = max(self.peak, self._active)
                try:
                    if on_read is not None:
                        on_read(rel)
                    if read_delay:
                        time.sleep(read_delay)
                finally:
                    with self._lock:
                        self._active -= 1
            return real_open(file, mode, *a, **k)

        def _stat(path, *a, **k):
            rel = under(path)
            if rel:
                with self._lock:
                    self.statted[rel.split(os.sep)[0]] += 1
            return real_stat(path, *a, **k)

        def _scandir(path="."):
            if under(path) == "":
                with self._lock:
                    self.listed += 1
            return real_scandir(path)

        def _listdir(path="."):
            if under(path) == "":
                with self._lock:
                    self.listed += 1
            return real_listdir(path)

        monkeypatch.setattr(io, "open", _open)
        monkeypatch.setattr(builtins, "open", _open)
        monkeypatch.setattr(os, "stat", _stat)
        monkeypatch.setattr(os, "scandir", _scandir)
        monkeypatch.setattr(os, "listdir", _listdir)

    def reads(self) -> dict:
        """``{key: sorted file names read}`` - what the listing paid ~24 ms apiece for."""
        out: dict = {}
        for (key, name), n in self.opened.items():
            out.setdefault(key, []).extend([name] * n)
        return {k: sorted(v) for k, v in out.items()}


def _entry(root, key, *, task="total_fast", identity=None, options=None, when=T0,
           computed=1.0, artifacts=()):
    """One published entry, made by hand (a real ``put`` evicts and prunes on every call,
    which is O(entries) and not what is under test), its pointer stamped ``when``."""
    d, gen = Path(root) / key, "0" * 32
    g = d / f"g-{gen}"
    g.mkdir(parents=True)
    (g / RESULT_NAME).write_bytes(b"labels")
    (g / "meta.json").write_text(json.dumps({
        "identity": identity if identity is not None else [f"idc:{key}"], "task": task,
        "options": options or {}, "computed": computed, "job": "j"}), encoding="utf-8")
    for name in artifacts:
        (g / name).write_bytes(b"x")
    (d / CURRENT_NAME).write_text(gen, encoding="utf-8")
    os.utime(d / CURRENT_NAME, ns=(when, when))
    return key


def _cache(tmp_path, n=0, **kw):
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    keys = [_entry(cache.root, f"k{i:04d}", when=T0 + i * 10 ** 9, **kw) for i in range(n)]
    return cache, keys


# -- (2) order and paging from names and times, content for the page only --------------

def test_a_page_reads_only_its_own_entries(tmp_path, monkeypatch):
    """Ten entries, a page of three: the pointer and meta.json of the three newest are
    opened, and nothing of the other seven - they cost a stat each. Read-everything-then-
    slice, which is what the listing did, opens all twenty files."""
    cache, keys = _cache(tmp_path, 10)
    disk = _Disk(monkeypatch, cache.root)
    rows, position = cache.list(limit=3)
    assert [r["key"] for r in rows] == keys[:-4:-1]
    assert disk.reads() == {k: [CURRENT_NAME, "meta.json"] for k in keys[:-4:-1]}
    assert disk.listed == 1
    assert set(disk.statted) == set(keys)             # every entry was dated, by a stat
    before = dict(disk.opened)
    rows2, _ = cache.list(limit=3, after=position)
    assert [r["key"] for r in rows2] == keys[-4:-7:-1]
    new = {k: v for k, v in disk.opened.items() if k not in before}
    assert {k for k, _ in new} == set(keys[-4:-7:-1]) and len(new) == 6


def test_the_order_is_the_pointers_mtime_and_traffic_does_not_move_it(tmp_path):
    """Newest PUBLISHED first - not ``computed`` (a job's start on a worker's clock, and a
    content read away), and not the key directory's mtime, which every read moves (the LRU
    touch in ``get``): a listing ordered by that would reshuffle under traffic."""
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    src = tmp_path / RESULT_NAME
    src.write_bytes(volume_bytes())
    for i, (key, computed) in enumerate((("old", 900.0), ("mid", 100.0), ("new", 500.0))):
        cache.put(key, src, {}, {"identity": [f"idc:{key}"], "task": "total_fast",
                                 "options": {}, "computed": computed})
        os.utime(cache.root / key / CURRENT_NAME, ns=(T0 + i * 10 ** 9,) * 2)
    assert [r["key"] for r in cache.list()[0]] == ["new", "mid", "old"]
    assert cache.get("old") is not None                # leases it, and touches <key>/
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG")
    assert cache.add_artifact("mid", "preview.png", png)   # lands inside the generation
    rows = cache.list()[0]
    assert [r["key"] for r in rows] == ["new", "mid", "old"]
    assert [r["published"] for r in rows] == [(T0 + i * 10 ** 9) / 1e9 for i in (2, 1, 0)]


def test_a_republication_moves_the_entry_once_to_the_head(tmp_path):
    """The pointer is replaced by ONE rename, so the entry's time moves exactly when its
    content does, and to a time ahead of every position already handed out."""
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    src = tmp_path / RESULT_NAME
    src.write_bytes(volume_bytes())
    for i, key in enumerate(("a", "b", "c")):
        cache.put(key, src, {}, {"identity": [f"idc:{key}"], "task": "total_fast",
                                 "computed": float(i)})
        os.utime(cache.root / key / CURRENT_NAME, ns=(T0 + i * 10 ** 9,) * 2)
    page1, position = cache.list(limit=2)
    assert [r["key"] for r in page1] == ["c", "b"]
    cache.put("a", src, {}, {"identity": ["idc:a"], "task": "total_fast", "computed": 9.0})
    assert [r["key"] for r in cache.list(after=position)[0]] == []   # it left the tail...
    head = cache.list(limit=1)[0][0]
    assert head["key"] == "a" and head["computed"] == 9.0            # ...for the head


def test_there_is_no_silent_cap(tmp_path):
    """The listing kept the newest 500 and said nothing of the rest: a Modal cache of
    2,083 results listed 500. Every entry is reachable now, by cursor."""
    cache, keys = _cache(tmp_path, 503)
    rows, position = cache.list()
    assert len(rows) == 503 and position is None
    seen, position = [], None
    while True:
        rows, position = cache.list(limit=200, after=position)
        seen += [r["key"] for r in rows]
        if position is None:
            break
    assert seen == keys[::-1]


def test_a_cursor_is_opaque_versioned_and_refused_when_foreign():
    token = encode_cursor((T0, "k" * 64))
    assert decode_cursor(token) == (T0, "k" * 64)
    assert "=" not in token and "k" * 8 not in token
    import base64
    for bad in ("", "nope", "!!!", base64.urlsafe_b64encode(b"[2,1,\"k\"]").decode(),
                base64.urlsafe_b64encode(b"[1,\"1\",\"k\"]").decode(),
                base64.urlsafe_b64encode(b"[1,true,\"k\"]").decode(),
                base64.urlsafe_b64encode(b"[1,1,\"\"]").decode(),
                base64.urlsafe_b64encode(b"{\"t\":1}").decode()):
        with pytest.raises(ValueError):
            decode_cursor(bad)


# -- (1) a filter by identity is computed, not searched --------------------------------

def test_keys_are_looked_up_by_name_and_nothing_else_is_touched(tmp_path, monkeypatch):
    """The cache is never listed, no entry but the named ones is even stat'ed, and only
    what is returned is read."""
    cache, keys = _cache(tmp_path, 12)
    disk = _Disk(monkeypatch, cache.root)
    rows, position = cache.list(keys=[keys[3], "absent" * 8, keys[7], keys[3]])
    assert [r["key"] for r in rows] == [keys[7], keys[3]] and position is None
    assert disk.listed == 0
    assert set(disk.statted) == {keys[3], keys[7], "absent" * 8}
    assert disk.statted["absent" * 8] == 1             # a miss is ONE failed stat
    assert disk.reads() == {k: [CURRENT_NAME, "meta.json"] for k in (keys[3], keys[7])}


def test_a_computed_name_is_never_a_path(tmp_path):
    cache, keys = _cache(tmp_path, 1)
    outside = tmp_path / "elsewhere"
    _entry(tmp_path, "elsewhere")
    assert cache.list(keys=["../elsewhere", ".", "", str(outside)])[0] == []


def test_a_tomb_is_not_an_entry(tmp_path):
    """Eviction moves a WHOLE entry aside, pointer and all, before it asks its second
    question (``_reclaim``): a listing that took the tomb for an entry would show the result
    twice, once under a name nothing resolves. Nor is a stray file, or an empty directory."""
    cache, keys = _cache(tmp_path, 2)
    _entry(cache.root, f".reclaim-1-1-{keys[0]}", when=T0 + 99 * 10 ** 9)
    (cache.root / "stray").write_text("x")
    (cache.root / "empty").mkdir()
    assert [r["key"] for r in cache.list()[0]] == keys[::-1]


# -- (3) a filter that needs content reads in parallel, and may remember ---------------

def test_a_content_filter_reads_entries_in_parallel(tmp_path, monkeypatch):
    """It is round trips, not data: 64 entries at 20 ms a read are 2.6 s one after another,
    and a fraction of that on the pool. Overlap is asserted, not only the clock."""
    cache, keys = _cache(tmp_path, 64)
    disk = _Disk(monkeypatch, cache.root, read_delay=0.02)
    t = time.monotonic()
    rows, _ = cache.list(accept=lambda r: r["key"].endswith("7"))
    took = time.monotonic() - t
    assert len(rows) == 6 and sum(disk.opened.values()) == 128
    assert disk.peak >= 8, f"at most {disk.peak} reads were ever in flight"
    assert took < 1.3, f"{took:.2f} s: the scan ran one read after another"


def test_every_read_happens_inside_the_hold_and_the_filter_outside_it(tmp_path, monkeypatch):
    """The contract Modal relies on (``modal_app._list_cache``): a volume reload hides the
    volume from the container's other threads, the pool's threads ARE other threads, and
    the reload is excluded by a lock the listing holds - so no read may start before the
    hold or outlive it. And the caller's filter, which may block for seconds on another
    volume, is never asked under it."""
    import contextlib
    cache, keys = _cache(tmp_path, 40)
    held, outside = [0], []
    lock = threading.Lock()

    @contextlib.contextmanager
    def hold():
        with lock:
            held[0] += 1
        try:
            yield
        finally:
            with lock:
                held[0] -= 1

    def on_read(rel):
        if not held[0]:
            outside.append(rel)
    disk = _Disk(monkeypatch, cache.root, read_delay=0.005, on_read=on_read)
    real_stat = cache._stamp

    def stamp(key, probe=False):
        if not held[0]:
            outside.append(f"stat {key}")
        return real_stat(key, probe)
    monkeypatch.setattr(cache, "_stamp", stamp)
    asked_under_hold = []
    rows, _ = cache.list(limit=30, hold=hold,
                         match=lambda f: asked_under_hold.append(held[0]) or True,
                         accept=lambda r: asked_under_hold.append(held[0]) or True)
    assert len(rows) == 30 and sum(disk.opened.values()) == 60
    assert outside == []
    assert asked_under_hold == [0] * 60                # 30 entries, asked twice, never held
    assert held[0] == 0


def test_a_batch_that_fails_to_start_still_ends_inside_the_hold(tmp_path, monkeypatch):
    """The exception path: a pool that cannot take the fifth read (no more threads) leaves
    four already running. The hold is not let go until they have ended - a reload let in
    early would hide the volume from them."""
    import contextlib
    from concurrent.futures import ThreadPoolExecutor
    cache, keys = _cache(tmp_path, 12)
    held, outside, reads = [0], [], []

    @contextlib.contextmanager
    def hold():
        held[0] += 1
        try:
            yield
        finally:
            held[0] -= 1

    real_stamp = cache._stamp

    def slow_stamp(key, probe=False):
        time.sleep(0.05)
        reads.append(key)
        if not held[0]:
            outside.append(key)
        return real_stamp(key, probe)
    monkeypatch.setattr(cache, "_stamp", slow_stamp)
    real_submit, n = ThreadPoolExecutor.submit, [0]

    def submit(self, fn, *a, **k):
        n[0] += 1
        if n[0] == 5:
            raise RuntimeError("can't start new thread")
        return real_submit(self, fn, *a, **k)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    with pytest.raises(RuntimeError, match="new thread"):
        cache.list(hold=hold)
    assert len(reads) == 4 and outside == [] and held[0] == 0


def test_a_hold_is_never_longer_than_a_chunk(tmp_path, monkeypatch):
    """A cold scan is seconds long on Modal, a reload waits 2 s for readers and lookups
    queue behind a waiting reload: the view lock is let go between batches."""
    import contextlib
    monkeypatch.setattr(serve, "LIST_CHUNK", 16)
    cache, keys = _cache(tmp_path, 50)
    holds, reads = [], []

    @contextlib.contextmanager
    def hold():
        reads.clear()
        yield
        holds.append(len(set(reads)))

    monkeypatch.setattr(cache, "_fields", lambda k, s, m, real=cache._fields: (
        reads.append(k), real(k, s, m))[1])
    assert len(cache.list(hold=hold)[0]) == 50
    assert max(holds) <= 16 and sum(holds) == 50


def test_a_metadata_filter_is_asked_before_a_rows_files_are_statted(tmp_path, monkeypatch):
    """Measured on Modal 2026-09-20: with every row remembered, a scan of 2,083 entries for
    a task none of them had was still 1.5 s - all stats, which barely overlap there (0.23 ms
    each serially, 0.11 ms on 32 threads). Three of a row's four stats are for its size and
    its artifacts, which a refused row never shows: ``match`` is asked of meta.json's fields
    first, so a refused entry costs the one stat that dated it."""
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    for i in range(12):
        _entry(cache.root, f"k{i:04d}", task="rare" if i in (3, 7) else "common",
               when=T0 + i * 10 ** 9)
    memo = ListingMemo()
    cache.list(memo=memo)                              # remembered: no content read below
    disk = _Disk(monkeypatch, cache.root)
    asked = []
    rows, _ = cache.list(memo=memo, match=lambda f: asked.append(f) or f["task"] == "rare")
    assert [r["key"] for r in rows] == ["k0007", "k0003"]
    assert len(asked) == 12 and set(asked[0]) == {"task", "identity", "options", "computed"}
    assert sum(disk.opened.values()) == 0
    refused = {k: n for k, n in disk.statted.items() if k not in ("k0003", "k0007")}
    assert set(refused.values()) == {1} and len(refused) == 10, refused
    assert min(disk.statted["k0003"], disk.statted["k0007"]) > 1


def test_the_memo_spares_the_reread_and_is_never_believed(tmp_path, monkeypatch):
    """Not an index: every listing still dates every entry by a stat, and a remembered row
    is used only under the pointer stamp it was read at. Second scan: no content read at
    all. A republication: that entry alone is read again. An artifact that arrived: seen,
    with no content read, because it is a stat."""
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    src = tmp_path / RESULT_NAME
    src.write_bytes(volume_bytes())
    for i in range(6):
        cache.put(f"k{i}", src, {}, {"identity": [f"idc:{U1[:-1]}{i}"], "task": "total_fast",
                                     "options": {}, "computed": float(i)})
    memo = ListingMemo()
    disk = _Disk(monkeypatch, cache.root)
    first = cache.list(memo=memo)[0]
    assert sum(disk.opened.values()) == 12 and len(memo) == 6
    disk.opened.clear(), disk.statted.clear()
    assert cache.list(memo=memo)[0] == first
    assert sum(disk.opened.values()) == 0
    assert set(disk.statted) == {f"k{i}" for i in range(6)}      # still dated, every time
    # an artifact arrives in the remembered generation: no pointer change, no content read
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG")
    assert cache.add_artifact("k2", "preview.png", png)
    disk.opened.clear()                                # add_artifact resolved the pointer
    row = next(r for r in cache.list(memo=memo)[0] if r["key"] == "k2")
    assert "preview" in row["links"] and sum(disk.opened.values()) == 0
    # a republication: a new pointer, so the memo does not vouch for it
    cache.put("k4", src, {}, {"identity": [f"idc:{U1[:-1]}4"], "task": "total_fast",
                              "options": {}, "computed": 44.0})
    disk.opened.clear()                                # put read nothing the listing reads,
    row = next(r for r in cache.list(memo=memo)[0] if r["key"] == "k4")   # but to be exact
    assert row["computed"] == 44.0
    assert disk.reads() == {"k4": [CURRENT_NAME, "meta.json"]}


def test_the_memo_gives_way_when_its_generation_is_gone(tmp_path, monkeypatch):
    """A remembered generation that pruning has since taken is read again, not dropped
    from the listing and not answered from memory - even under an unchanged stamp."""
    cache, (key,) = _cache(tmp_path, 1)
    memo = ListingMemo()
    assert len(cache.list(memo=memo)[0]) == 1
    d = cache.root / key
    (d / ("g-" + "0" * 32)).rename(d / ("g-" + "1" * 32))
    meta = json.loads((d / ("g-" + "1" * 32) / "meta.json").read_text())
    (d / ("g-" + "1" * 32) / "meta.json").write_text(json.dumps({**meta, "computed": 7.0}))
    (d / CURRENT_NAME).write_text("1" * 32)
    os.utime(d / CURRENT_NAME, ns=(T0, T0))            # the very stamp the memo holds
    rows = cache.list(memo=memo)[0]
    assert [r["computed"] for r in rows] == [7.0]


def test_the_memo_is_bounded_and_forgets_the_least_recently_used():
    memo = ListingMemo(capacity=3)
    for i in range(5):
        memo.put(f"k{i}", i, "g-x", {"task": "t"})
    assert len(memo) == 3 and memo.get("k0", 0) is None and memo.get("k4", 4) is not None
    assert memo.get("k2", 2) is not None               # used: now the most recent
    memo.put("k5", 5, "g-x", {})
    assert memo.get("k3", 3) is None and memo.get("k2", 2) is not None
    assert memo.get("k4", 99) is None                  # another publication of k4


def test_a_refused_row_does_not_shorten_the_page(tmp_path, monkeypatch):
    """The filter runs while the page fills, not after it: three of six entries refused
    and a page of three still holds three. And once a row has been refused the scan reads
    ahead by chunks, where it would otherwise read one entry per round trip."""
    monkeypatch.setattr(serve, "LIST_CHUNK", 4)
    cache, keys = _cache(tmp_path, 12)
    disk = _Disk(monkeypatch, cache.root)
    rows, position = cache.list(limit=3, accept=lambda r: int(r["key"][1:]) % 2 == 0)
    assert [r["key"] for r in rows] == ["k0010", "k0008", "k0006"]
    assert decode_cursor(encode_cursor(position)) == (T0 + 6 * 10 ** 9, "k0006")
    assert len(disk.reads()) == 3 + 4                  # the first page, then one chunk
    rest = cache.list(after=position, accept=lambda r: int(r["key"][1:]) % 2 == 0)[0]
    assert [r["key"] for r in rest] == ["k0004", "k0002", "k0000"]


# -- the route -------------------------------------------------------------------------

def _app(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(serve, "_idc_enabled", lambda: True)
    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "rc",
                       keep_cached=10 ** 6)
    return seg, ex, TestClient(create_app(ex, **kw))


def _key(seg, identity, task="total_fast", options=None):
    ids = (identity,) if isinstance(identity, str) else tuple(identity)
    return result_key(ids, task, options or {}, weights_versions_of(seg, task))


def _real(seg, ex, identity, task="total_fast", options=None, when=T0, **kw):
    """An entry under the key this server derives NOW for (identity, task, options)."""
    ids = [identity] if isinstance(identity, str) else list(identity)
    return _entry(ex.cache.root, _key(seg, ids, task, options), task=task, identity=ids,
                  options=options, when=when, **kw)


def _listing(client, **params):
    r = client.get("/v1/segmentations", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_identity_is_computed_for_every_task_and_menu_option_set(tmp_path, monkeypatch):
    """What the path surface does for one task, done for all of them: the default options
    and each grid token, under this server's current weights. Entries of other inputs are
    never read; the same input under a STALE key is not found, as the plain listing would
    not show it; and non-menu options have no path, so the plain listing has them only."""
    seg, ex, client = _app(tmp_path, monkeypatch)
    a = _real(seg, ex, f"idc:{U1}", when=T0 + 5 * 10 ** 9)
    b = _real(seg, ex, f"idc:{U1}", task="total", options={"grid": 1.0}, when=T0 + 9 * 10 ** 9)
    c = _real(seg, ex, f"idc:{U1}", options={"interp": "nearest"})
    d = _real(seg, ex, f"idc:{U2}")
    _entry(ex.cache.root, "stale" * 12, identity=[f"idc:{U1}"])
    others = [_real(seg, ex, f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55") for i in range(20)]
    disk = _Disk(monkeypatch, ex.cache.root)

    got = _listing(client, identity=f"idc:{U1}")
    assert [e["key"] for e in got["segmentations"]] == [b, a] and got["next_cursor"] is None
    assert got["segmentations"][0]["links"]["labels"] == \
        f"/v1/idc/{U1}/total/labels_res-1mm.seg.nrrd"
    assert disk.listed == 0
    assert disk.reads() == {k: [CURRENT_NAME, "meta.json"] for k in (a, b)}
    assert len(disk.statted) == 2 * 2                  # tasks x option sets, by name

    disk.statted.clear()
    one = _listing(client, identity=f"idc:{U1}", task="total_fast")
    assert [e["key"] for e in one["segmentations"]] == [a]
    assert len(disk.statted) == 2                      # one key per option set

    both = _listing(client, identity=[f"idc:{U1.upper()}", f" idc:{U2}"])
    assert {e["key"] for e in both["segmentations"]} == {a, b, d}

    plain = {e["key"] for e in _listing(client, limit=1000)["segmentations"]}
    assert plain == {a, b, c, d, *others}              # c is here, the stale key nowhere
    assert "links" not in next(e for e in _listing(client, limit=1000)["segmentations"]
                               if e["key"] == c)


def test_a_tasks_weights_are_read_once_a_request_however_many_keys(tmp_path, monkeypatch):
    """A key's whole cost is its weights versions - a ``describe()`` of the task. Measured
    on Modal 2026-09-20: looking up 188 computed names took 0.05 s, deriving them 2.1 s,
    and asked key by key it would have been once more per identity and per option set."""
    seg, ex, client = _app(tmp_path, monkeypatch)
    a = _real(seg, ex, f"idc:{U1}")
    described = []
    real = seg.describe
    monkeypatch.setattr(seg, "describe", lambda t: described.append(t) or real(t))
    ids = [f"idc:{U1}", f"idc:{U2}", "sha256:" + "e" * 64]
    assert [e["key"] for e in _listing(client, identity=ids)["segmentations"]] == [a]
    assert sorted(described) == ["total", "total_fast"]         # 12 keys, 2 describes


def test_an_executor_that_pins_its_keys_says_how_it_reads_weights(monkeypatch, tmp_path):
    """ModalExecutor's ``resource_key`` hides the versions it keys on (they are reloaded,
    throttled, from a frozen weights volume), so the listing could only ask it key by key.
    ``weights_versions`` is that door: once a task, and the keys the route derives from it
    are the keys ``resource_key`` gives - the entry below is published under one."""
    from test_stale_volume_view import _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    asked = []
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda t: asked.append(t) or ["w=7"])
    key = ex.resource_key(f"idc:{U1}", "total_fast", {"grid": 1.0})
    asked.clear()
    _entry(Path(m.CACHE_ROOT), key, identity=[f"idc:{U1}"], options={"grid": 1.0})
    got = _listing(client, identity=[f"idc:{U1}", f"idc:{U2}", "sha256:" + "e" * 64])
    assert [e["key"] for e in got["segmentations"]] == [key]
    assert sorted(asked) == ["total", "total_fast"]


def test_a_twin_may_say_how_it_reads_weights_too(tmp_path):
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    asked = []
    weights_fn = lambda task: asked.append(task) or ["w=1"]  # noqa: E731
    key_fn = lambda identity, task, opts=None: result_key(  # noqa: E731
        (identity,), task, opts or {}, weights_fn(task))
    a = _entry(cache.root, key_fn(f"idc:{U1}", "total_fast"), identity=[f"idc:{U1}"])
    asked.clear()
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                                          list_fn=cache.list, weights_fn=weights_fn))
    got = _listing(client, identity=[f"idc:{U1}", f"idc:{U2}"])
    assert [e["key"] for e in got["segmentations"]] == [a] and asked == ["total_fast"]


def test_a_task_filter_pages_through_every_match(tmp_path, monkeypatch):
    seg, ex, client = _app(tmp_path, monkeypatch)
    mine, theirs = [], []
    for i in range(14):
        ident = f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55"
        (mine if i % 3 == 0 else theirs).append(
            _real(seg, ex, ident, task="total" if i % 3 == 0 else "total_fast",
                  when=T0 + i * 10 ** 9))
    seen, cursor = [], None
    while True:
        page = _listing(client, task="total", limit=2, **({"cursor": cursor} if cursor else {}))
        seen += [e["key"] for e in page["segmentations"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == mine[::-1]
    assert {e["task"] for e in _listing(client, task="total")["segmentations"]} == {"total"}
    # ...and the route hands the filter over as `match`: a refused entry costs ONE stat
    disk = _Disk(monkeypatch, ex.cache.root)
    assert len(_listing(client, task="total")["segmentations"]) == len(mine)
    assert {disk.statted[k] for k in theirs} == {1} and sum(disk.opened.values()) == 0


def test_the_cursor_is_stable_when_an_entry_is_published_between_two_pages(tmp_path,
                                                                           monkeypatch):
    """A position, not an offset: what is published after page one sorts ahead of it, so
    page two is the page it would have been. With an offset the new entry shifts every
    row down by one - page two repeats the last row of page one."""
    seg, ex, client = _app(tmp_path, monkeypatch)
    keys = [_real(seg, ex, f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55", when=T0 + i * 10 ** 9)
            for i in range(5)]
    page1 = _listing(client, limit=2)
    assert [e["key"] for e in page1["segmentations"]] == [keys[4], keys[3]]
    late = _real(seg, ex, f"idc:{U2}", when=T0 + 60 * 10 ** 9)
    page2 = _listing(client, limit=2, cursor=page1["next_cursor"])
    assert [e["key"] for e in page2["segmentations"]] == [keys[2], keys[1]]
    page3 = _listing(client, limit=2, cursor=page2["next_cursor"])
    assert [e["key"] for e in page3["segmentations"]] == [keys[0]]
    assert page3["next_cursor"] is None
    assert _listing(client, limit=1)["segmentations"][0]["key"] == late


def test_the_default_page_says_there_is_more(tmp_path, monkeypatch):
    seg, ex, client = _app(tmp_path, monkeypatch)
    for i in range(serve.LIST_LIMIT_DEFAULT + 1):
        _real(seg, ex, f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55", when=T0 + i * 10 ** 9)
    page = _listing(client)
    assert len(page["segmentations"]) == serve.LIST_LIMIT_DEFAULT and page["next_cursor"]
    rest = _listing(client, cursor=page["next_cursor"])
    assert len(rest["segmentations"]) == 1 and rest["next_cursor"] is None
    # the local server remembers too: the same page again opens nothing
    disk = _Disk(monkeypatch, ex.cache.root)
    assert _listing(client) == page and sum(disk.opened.values()) == 0


@pytest.mark.parametrize("params, needle", [
    ({"limit": 0}, "limit"), ({"limit": serve.LIST_LIMIT_MAX + 1}, "limit"),
    ({"cursor": "bm9wZQ"}, "cursor"), ({"task": "no_such_task"}, "no_such_task"),
    ({"identity": "nope:thing"}, "idc:"), ({"identity": "idc:not-a-uuid"}, "not a valid idc"),
    ({"identity": "plainword"}, "expected <source>:<identifier>"),
    ({"identity": "result:" + "a" * 64}, "expected <source>:<identifier>"),
    ({"identity": [f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55" for i in range(101)]},
     "at most 100"),
])
def test_what_the_listing_refuses_is_a_422_that_names_it(tmp_path, monkeypatch, params, needle):
    _, _, client = _app(tmp_path, monkeypatch)
    r = client.get("/v1/segmentations", params=params)
    assert r.status_code == 422, r.text
    assert needle in json.dumps(r.json())


def test_filters_do_not_open_the_listing_to_anonymous_callers(tmp_path, monkeypatch):
    """The listing enumerates upload identities, and a filter by digest would CONFIRM one:
    authorized only, whatever the parameters."""
    seg, ex, client = _app(tmp_path, monkeypatch, token="s3cret")
    _real(seg, ex, "sha256:" + "a" * 64)
    for params in ({}, {"identity": "sha256:" + "a" * 64}, {"task": "total_fast"},
                   {"limit": 5}):
        assert client.get("/v1/segmentations", params=params).status_code == 401
    r = client.get("/v1/segmentations", params={"identity": "sha256:" + "a" * 64},
                   headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and len(r.json()["segmentations"]) == 1


def test_a_listing_holding_digest_identities_neither_crashes_nor_mints_links(tmp_path,
                                                                           monkeypatch):
    """Untested until 2026-09-20 (docs/result-references.md): an upload's blob digest, an
    uploaded DICOM series' tree digest, a ``result:`` reference - whose identity IS a
    digest - and a multi-input entry, whose identities are role-tagged. None has a path
    (``resource_links`` asks ``content.is_digest``), all are listed, and a digest can be
    filtered on: the key of an upload is derived exactly as a hosted input's is."""
    seg, ex, client = _app(tmp_path, monkeypatch)
    blob, tree = "sha256:" + "b" * 64, "sha256-tree:" + "c" * 64
    k_blob = _real(seg, ex, blob, when=T0 + 1)
    k_tree = _real(seg, ex, tree, when=T0 + 2)
    k_multi = _real(seg, ex, sorted([f"image=idc:{U1}", f"mask={blob}"]), when=T0 + 3)
    k_hosted = _real(seg, ex, f"idc:{U1}", when=T0 + 4)
    _entry(ex.cache.root, "odd1", identity="idc:not-a-list", when=T0 + 5)
    _entry(ex.cache.root, "odd2", identity=None, task=None, when=T0 + 6)
    (ex.cache.root / "odd3").mkdir()                   # a directory that publishes nothing
    (ex.cache.root / "odd4").write_text("a stray file")
    # an entry eviction moved aside whole, pointer and all: not an entry while it is a tomb
    _entry(ex.cache.root, f".reclaim-1-1-{k_hosted}", identity=[f"idc:{U1}"], when=T0 + 7)
    rows = {e["key"]: e for e in _listing(client)["segmentations"]}
    assert {k_blob, k_tree, k_multi, k_hosted} <= set(rows)
    assert not [k for k in rows if k.startswith(".") or k in ("odd3", "odd4")]
    for k in (k_blob, k_tree, k_multi):
        assert "links" not in rows[k], rows[k]
    assert rows[k_hosted]["links"]["labels"] == f"/v1/idc/{U1}/total_fast/labels.seg.nrrd"
    assert not [r.path for r in client.app.routes if r.path.startswith("/v1/sha256")]
    by_digest = _listing(client, identity=[blob, tree])["segmentations"]
    assert {e["key"] for e in by_digest} == {k_blob, k_tree}
    assert all("links" not in e for e in by_digest)


def test_a_multi_input_result_is_keyed_on_every_identity(tmp_path, monkeypatch):
    """The key round trip asked only the FIRST identity, so every multi-input result read
    as stale-keyed and the listing dropped it. It is held to the same rule now: listed
    under the key this server derives from all of its identities, dropped under another."""
    seg, ex, client = _app(tmp_path, monkeypatch)
    ids = sorted([f"image=idc:{U1}", f"mask=idc:{U2}"])
    fresh = _real(seg, ex, ids)
    _entry(ex.cache.root, "f" * 64, identity=ids)
    _entry(ex.cache.root, _key(seg, ids[0]), identity=ids)      # keyed on the first alone
    assert {e["key"] for e in _listing(client)["segmentations"]} == {fresh}


def test_the_twin_lists_through_its_own_key_function(tmp_path):
    """The operator's opt-in stays one: no lister, no route. With one, the twin filters
    through ``key_fn`` - it has no segmenter to read weights versions from."""
    cache = ResultCache(tmp_path / "rc", keep=10 ** 6)
    key_fn = lambda identity, task, opts=None: result_key(  # noqa: E731
        (identity,) if isinstance(identity, str) else tuple(identity), task, opts or {}, ["w=1"])
    a = _entry(cache.root, key_fn(f"idc:{U1}", "total_fast"), identity=[f"idc:{U1}"])
    _entry(cache.root, key_fn(f"idc:{U2}", "total_fast"), identity=[f"idc:{U2}"], when=T0 + 9)
    _entry(cache.root, "w" * 64, identity=[f"idc:{U1}"])          # other weights: stale here
    bare = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"]))
    assert bare.get("/v1/segmentations").status_code in (404, 405)
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                                          list_fn=cache.list))
    assert [e["key"] for e in _listing(client, identity=f"idc:{U1}")["segmentations"]] == [a]
    page = _listing(client, limit=1)
    assert len(page["segmentations"]) == 1 and page["next_cursor"]
    assert len(_listing(client, cursor=page["next_cursor"])["segmentations"]) == 1


def test_the_listing_never_runs_on_the_event_loop(tmp_path, monkeypatch):
    """On Modal the listing waits out a volume reload and then reads for seconds: on the
    loop thread that froze every request in the container (review, 2026-09-19)."""
    import asyncio
    seg, ex, client = _app(tmp_path, monkeypatch)
    _real(seg, ex, f"idc:{U1}")
    on_loop = []
    real = ex.cache_list

    def cache_list(**kw):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(**kw)
    monkeypatch.setattr(ex, "cache_list", cache_list)
    for params in ({}, {"identity": f"idc:{U1}"}, {"task": "total_fast"}):
        assert len(_listing(client, **params)["segmentations"]) == 1
    assert on_loop == [False, False, False]


# -- the client and the command ---------------------------------------------------------

def _remote(client):
    """A RemoteClient whose transport is the test app."""
    pytest.importorskip("httpx")
    from haversack.client import RemoteClient
    c = RemoteClient("http://testserver")
    c._http = client
    return c


def test_the_client_filters_and_follows_cursors(tmp_path, monkeypatch):
    seg, ex, client = _app(tmp_path, monkeypatch)
    keys = [_real(seg, ex, f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55", when=T0 + i * 10 ** 9)
            for i in range(7)]
    c = _remote(client)
    asked = []
    real = c.segmentations
    monkeypatch.setattr(c, "segmentations", lambda **kw: asked.append(kw) or real(**kw))
    assert [e["key"] for e in c.iter_segmentations(page_size=3)] == keys[::-1]
    assert [a["cursor"] is None for a in asked] == [True, False, False]
    page = real(identity=f"idc:{0:08d}-9410-47ff-9c9f-a44b26a4bd55", task="total_fast")
    assert [e["key"] for e in page["segmentations"]] == [keys[0]]
    assert real(identity=[f"idc:{U1}", f"idc:{U2}"])["segmentations"] == []


def test_the_client_will_not_pass_off_an_old_servers_whole_listing_as_filtered(monkeypatch):
    """A server from before 2026-09-20 ignores the parameters and returns everything."""
    pytest.importorskip("httpx")
    from haversack.client import RemoteClient, RemoteError
    c = RemoteClient("http://old-server")
    monkeypatch.setattr(c, "_json", lambda *a, **k: {"segmentations": [{"key": "x"}]})
    assert c.segmentations() == {"segmentations": [{"key": "x"}]}
    with pytest.raises(RemoteError, match="predates filters"):
        c.segmentations(task="ts.v2:total")


def test_remote_results_prints_a_table_or_json_and_stops_at_the_limit(tmp_path, monkeypatch,
                                                                      capsys):
    from haversack import cli, client as client_mod
    seg, ex, http = _app(tmp_path, monkeypatch)
    for i in range(5):
        _real(seg, ex, f"idc:{i:08d}-9410-47ff-9c9f-a44b26a4bd55", when=T0 + i * 10 ** 9)
    _real(seg, ex, "sha256:" + "d" * 64, when=T0 - 10 ** 9)
    real_init = client_mod.RemoteClient.__init__

    def init(self, server, **kw):
        real_init(self, server, **kw)
        self._http = http
    monkeypatch.setattr(client_mod.RemoteClient, "__init__", init)
    base = ["remote", "--server", "http://testserver", "--token", "t", "results"]
    assert cli.main(base + ["--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    when, task, ident, where = lines[0].split("\t")
    assert (when, task) == ("2023-11-14T22:13:24Z", "total_fast")
    assert where == f"/v1/{ident.replace(':', '/', 1)}/total_fast/labels.seg.nrrd"
    assert cli.main(base + ["--limit", "0", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["segmentations"]
    assert len(rows) == 6
    assert cli.main(base + ["--identity", "sha256:" + "d" * 64]) == 0
    assert capsys.readouterr().out.rstrip().endswith("key:" + rows[-1]["key"])


# -- on Modal: the listing reads the CACHE VOLUME from the api container ---------------
#
# The doubles are test_stale_volume_view's: a volume whose reload hides its root from the
# container's other threads while it runs (measured on Modal 2026-09-19: ENOENT 734,714
# times in 735,326), and one whose view lags what the workers committed until a reload takes.

def _modal_entries(m, n):
    from haversack.serve import ResultCache
    cache = ResultCache(m.CACHE_ROOT, keep=10 ** 6)
    return [_entry(cache.root, f"k{i:04d}", when=T0 + i * 10 ** 9) for i in range(n)]


def test_listings_racing_reloads_never_lose_an_entry(monkeypatch, tmp_path):
    """The listing reads on a pool of threads, and to a reload those are "other threads"
    exactly as another request's is. Every listing here reloads first (a listing is read
    from a view newer than the request), so eight of them race eight reloads: with the
    reads outside ``_cache_view`` a listing comes back short, or empty, and is believed."""
    from test_stale_volume_view import _HidingVolume, _modal
    m, jobs, _, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    keys = _modal_entries(m, 12)
    monkeypatch.setattr(m, "cache_vol", _HidingVolume(tmp_path / "cache"))
    short, errors = [], []

    def ask():
        for _ in range(25):
            try:
                rows, _pos = ex.cache_list()
                if [r["key"] for r in rows] != keys[::-1]:
                    short.append(len(rows))
            except Exception as e:             # noqa: BLE001 - a crash is a failure too
                errors.append(e)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert not short, f"{len(short)} of 200 listings lost entries: {sorted(set(short))}"
    assert m.cache_vol.reloads >= 8                    # the race was really run


@pytest.mark.parametrize("params", [{}, {"identity": f"idc:{U1}"}, {"task": "total_fast"}])
def test_a_listing_from_a_view_that_cannot_be_refreshed_is_503_not_a_shorter_list(
        monkeypatch, tmp_path, params):
    """A result missing from a listing reads as "not computed", and a client that lists to
    decide what to compute would compute it again - the listing's form of the false 410.
    So a refused reload is ``not_visible_yet``; once one takes, the entry is there."""
    from test_stale_volume_view import _commit_elsewhere, _key, _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=10 ** 6)
    monkeypatch.setattr(m, "_listing_memo", [])
    key = _key(ex)
    _commit_elsewhere(m, vol, tmp_path, key)
    hidden_meta = next(vol.hidden.glob(f"{key}/g-*/meta.json"))
    hidden_meta.write_text(json.dumps({"identity": [f"idc:{U1}"], "task": "total_fast",
                                       "options": {}, "computed": 1.0}))
    r = client.get("/v1/segmentations", params=params)
    assert r.status_code == 503, r.text
    assert r.headers["retry-after"] == "5"
    assert r.json()["detail"]["code"] == "not_visible_yet"
    vol.refusals = 1                                   # one more refusal, then it takes
    r = client.get("/v1/segmentations", params=params)
    assert r.status_code == 200, r.text
    assert [e["key"] for e in r.json()["segmentations"]] == [key]


def test_the_view_is_never_held_while_the_callers_filter_is_asked(monkeypatch, tmp_path):
    """The route's filter derives keys, which on Modal can reload the WEIGHTS volume and
    describe a task - seconds. Held across that, the cache view would refuse every reload
    and stall every lookup behind it. A reload can always get in between two batches."""
    from test_stale_volume_view import _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    _modal_entries(m, 9)
    free = []

    def accept(row):
        got = m._cache_view.acquire_exclusive(0.2)
        if got:
            m._cache_view.release_exclusive()
        free.append(got)
        return True
    rows, _ = ex.cache_list(accept=accept)
    assert len(rows) == 9 and free == [True] * 9


def test_the_cache_object_is_built_under_the_view_lock(monkeypatch, tmp_path):
    """``ResultCache()`` mkdirs its root - a touch of the volume, and during a reload one
    that CREATES an empty /cache beside the hidden one. The worker's artifact thread did
    exactly that outside its lock (review, 2026-09-19); the listing must not."""
    from test_stale_volume_view import _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    _modal_entries(m, 5)
    built = []
    real_init = serve.ResultCache.__init__

    def init(self, *a, **k):
        built.append(m._cache_view._readers)
        real_init(self, *a, **k)
    monkeypatch.setattr(serve.ResultCache, "__init__", init)
    assert len(ex.cache_list()[0]) == 5
    assert built == [1]


def test_a_containers_memo_outlives_the_request_and_each_listing_still_reloads(monkeypatch,
                                                                              tmp_path):
    from test_stale_volume_view import _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    keys = _modal_entries(m, 6)
    disk = _Disk(monkeypatch, m.CACHE_ROOT)
    assert [r["key"] for r in ex.cache_list()[0]] == keys[::-1]
    assert sum(disk.opened.values()) == 12
    disk.opened.clear()
    assert [r["key"] for r in ex.cache_list(limit=4)[0]] == keys[:-5:-1]
    assert sum(disk.opened.values()) == 0              # remembered, per container
    assert vol.reloads == 2                            # and never in place of a fresh view


def test_the_task_filter_reaches_the_volume_before_the_stats_do(monkeypatch, tmp_path):
    """On Modal is where it matters: a stat there is 0.2 ms and barely overlaps. The
    executor must hand ``match`` on - dropped, the answer is the same (the route's
    ``accept`` filters again) and only the bill differs, which no other test reads."""
    from test_stale_volume_view import _modal
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    monkeypatch.setattr(m, "_listing_memo", [])
    keys = _modal_entries(m, 6)                        # all of task total_fast
    ex.cache_list()                                    # remembered
    disk = _Disk(monkeypatch, m.CACHE_ROOT)
    assert _listing(client, task="total")["segmentations"] == []
    assert dict(disk.statted) == {k: 1 for k in keys} and sum(disk.opened.values()) == 0
