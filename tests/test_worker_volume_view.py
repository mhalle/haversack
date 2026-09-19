"""A worker's threads must not read a volume while another of its threads reloads it.

Measured on Modal 2026-09-19: while one thread runs ``Volume.reload()``, every path on
that volume is ENOENT to the container's other threads. The api side was fixed first (its
lookups missed committed results and answered 410). A worker container runs up to four
threads over the scratch and cache volumes: the job; the prefetcher, which reloads
scratch; the previous job's artifact overlap, which places into the cache and commits it;
and the retention sweep, which commits scratch. Until the fix the job read its upload
from the volume after dropping ``_vol_lock`` and read its labels back from it - failing
with FileNotFoundError whenever the prefetcher reloaded in between - and put + committed
the cache with no lock at all.

The volume double renames its root away for the length of a reload, as the api's tests
do, and also catches what Modal would discard: anything written under the root while it
was away. Its commit reloads too. Modal's commit did not hide anything in 61 measured
commits, but its source reloads after one whenever the server says to, so the worker
locks commits as it locks reloads, and the double holds it to that.
"""
from __future__ import annotations

import shutil
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("modal")

from haversack.serve import RESULT_NAME, ResultCache  # noqa: E402


class _HidingVolume:
    """A reload as measured on Modal: the volume vanishes from other threads while it
    runs. A write landing in that window recreates the root here; it is recorded in
    ``lost`` and thrown away, as the reload would have."""

    def __init__(self, root: Path):
        self.root, self.away = Path(root), Path(str(root) + ".reloading")
        self.reloads, self.lost = 0, []
        self._one = threading.Lock()           # Modal serializes a process's reloads

    def reload(self):
        with self._one:
            self.reloads += 1
            self.root.rename(self.away)
            time.sleep(0.002)
            if self.root.exists():
                self.lost.append(sorted(str(p.relative_to(self.root))
                                        for p in self.root.rglob("*")))
                shutil.rmtree(self.root)
            self.away.rename(self.root)

    def commit(self):
        self.reload()                          # Modal's commit ends in a reload


class _Seg:
    """What a compute returns: enough for the real result_payload and save."""

    def __init__(self, body: bytes):
        self.body = body
        self.schema = types.SimpleNamespace(names={1: "liver"})
        self.provenance = {}

    def volumes_ml(self):
        return {"liver": 1.0}

    def save(self, path):
        Path(path).write_bytes(self.body)


class _Ctx:
    """A worker as _execute_job sees it."""
    engine = "nnunetv2"
    seg = None
    series_cache = None
    read_ahead = types.SimpleNamespace(pop=lambda key: None)

    def __init__(self):
        self._vol_lock = threading.Lock()
        self.read = []

    def _ensure(self, task):
        pass

    def _compute(self, input_path, meta, on_progress, token):
        # a compute reads its input for a while, not in one go
        seen = set()
        for _ in range(10):
            seen.add(Path(input_path).read_bytes())
            time.sleep(0.001)
        (body,) = seen
        self.read.append(body)
        return _Seg(b"labels of " + body)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    from haversack import modal_app as m
    jobs = {}
    for name in ("scratch", "cache"):
        (tmp_path / name).mkdir()
    scratch, cache = _HidingVolume(tmp_path / "scratch"), _HidingVolume(tmp_path / "cache")
    monkeypatch.setattr(m, "jobs_dict", jobs)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(scratch.root))
    monkeypatch.setattr(m, "CACHE_ROOT", str(cache.root))
    monkeypatch.setattr(m, "JOB_LOCAL_ROOT", str(tmp_path / "local"), raising=False)
    monkeypatch.setattr(m, "scratch_vol", scratch)
    monkeypatch.setattr(m, "cache_vol", cache)
    monkeypatch.setattr(m, "ARTIFACTS", set())       # artifacts come from the thread below
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_sweep_due", lambda: False)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)
    return m, jobs, scratch, cache


def _submit(m, jobs, jid: str) -> bytes:
    body = f"volume {jid}".encode()
    jdir = Path(m.SCRATCH_ROOT) / jid
    jdir.mkdir()
    (jdir / "input_ct.nii.gz").write_bytes(body)
    jobs[jid] = {"id": jid, "task": "ts.v2:total_fast", "state": "queued",
                 "source": [{"kind": "upload"}], "input_identity": [f"sha256:{jid}"],
                 "cache_key": f"key-{jid}", "created": time.time()}
    return body


def _other_threads(m, ctx, stop, key, placed, errors):
    """The threads a job shares its container with, as the worker runs them: the
    previous job's artifact overlap (the real ``_artifact_worker``, placing into an
    entry and committing), and the prefetcher/sweep, which reload and commit scratch
    under ``_vol_lock``."""
    png = Path(m.SCRATCH_ROOT).parent / "preview.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")

    def overlap(pair, task, artifacts, *, preview_out, statistics_out, place, finish):
        out = png.with_name(f"render-{len(placed)}.png")    # /dev/shm is Linux-only
        shutil.copyfile(png, out)
        ok = place("preview.png", out)
        placed.append(ok)
        finish([("preview", 0.0)] if ok else [])

    def artifacts():
        from haversack import serve
        orig = serve.artifact_overlap
        serve.artifact_overlap = overlap
        try:
            while not stop.is_set():
                try:
                    m._WorkerBase._artifact_worker(ctx, object(), key, "prev", "ts.v2:total_fast")
                except Exception as e:             # noqa: BLE001
                    errors.append(e)
        finally:
            serve.artifact_overlap = orig

    def prefetch_and_sweep():
        while not stop.is_set():
            with ctx._vol_lock:
                m.scratch_vol.reload()
            time.sleep(0.0005)

    return [threading.Thread(target=artifacts), threading.Thread(target=prefetch_and_sweep)]


def test_jobs_and_artifacts_racing_reloads_lose_nothing(worker, tmp_path, monkeypatch):
    m, jobs, scratch, cache = worker
    # an entry published by the previous job, which its overlap is still placing into
    src = tmp_path / "prev-labels"
    src.write_bytes(b"previous labels")
    ResultCache(m.CACHE_ROOT).put("key-prev", src, {"outputs": []}, {"job": "prev"})
    ctx = _Ctx()
    stop, placed, errors = threading.Event(), [], []
    threads = _other_threads(m, ctx, stop, "key-prev", placed, errors)
    for t in threads:
        t.start()
    bodies = {}
    try:
        for i in range(30):
            jid = f"j{i:02d}"
            with ctx._vol_lock:                # the api's upload, committed before
                bodies[jid] = _submit(m, jobs, jid)   # the job runs
            m._execute_job(ctx, jid)
    finally:
        stop.set()
        for t in threads:
            t.join()

    failed = {j: jobs[j].get("error", "")[:200] for j in bodies if jobs[j]["state"] != "done"}
    assert not failed, failed
    assert not errors, errors[:3]
    assert ctx.read == list(bodies.values())
    for jid, body in bodies.items():
        hit = ResultCache(m.CACHE_ROOT).get(f"key-{jid}")
        assert hit is not None, jid
        assert Path(hit[0]).read_bytes() == b"labels of " + body
        assert (Path(m.SCRATCH_ROOT) / jid / RESULT_NAME).read_bytes() == b"labels of " + body
    assert placed and all(placed), f"{placed.count(False)} of {len(placed)} artifacts dropped"
    assert not scratch.lost and not cache.lost, (scratch.lost[:2], cache.lost[:2])
    assert scratch.reloads > 100 and cache.reloads > 30      # the race was actually run
    assert not list((tmp_path / "local").iterdir())          # the local copies are gone


def test_a_multi_input_job_reads_its_uploads_from_local_copies(worker, tmp_path):
    """Each role's upload is copied under the lock and the compute gets the copies."""
    m, jobs, scratch, cache = worker
    jdir = Path(m.SCRATCH_ROOT) / "mm"
    jdir.mkdir()
    (jdir / "input_t1_a.nii.gz").write_bytes(b"t1")
    (jdir / "input_flair_b.nii.gz").write_bytes(b"flair")
    jobs["mm"] = {"id": "mm", "task": "ts.v2:total_fast", "state": "queued",
                  "source": [{"kind": "upload", "role": "flair"},
                             {"kind": "upload", "role": "t1"}],
                  "cache_key": "key-mm", "created": time.time()}
    got = {}

    class Ctx(_Ctx):
        def _compute(self, input_path, meta, on_progress, token):
            got.update({r: (p, Path(p).read_bytes()) for r, p in input_path.items()})
            return _Seg(b"x")

    m._execute_job(Ctx(), "mm")
    assert jobs["mm"]["state"] == "done", jobs["mm"].get("error")
    assert {r: b for r, (_, b) in got.items()} == {"t1": b"t1", "flair": b"flair"}
    assert all(not str(p).startswith(m.SCRATCH_ROOT) for p, _ in got.values())


def test_an_upload_that_is_not_visible_fails_the_job_by_name(worker):
    m, jobs, scratch, cache = worker
    (Path(m.SCRATCH_ROOT) / "nx").mkdir()
    jobs["nx"] = {"id": "nx", "task": "ts.v2:total_fast", "state": "queued",
                  "source": [{"kind": "upload"}], "created": time.time()}
    m._execute_job(_Ctx(), "nx")
    assert jobs["nx"]["state"] == "failed"
    assert "no uploaded input is visible" in jobs["nx"]["error"]


def test_a_refused_reload_still_stages_what_is_visible(worker):
    """Modal refuses a reload while a file on the volume is open anywhere in the
    container. That says nothing about the upload, which is usually visible already."""
    m, jobs, scratch, cache = worker
    body = _submit(m, jobs, "rf")

    def refused():
        raise RuntimeError("there are open files preventing the operation")
    scratch.commit = scratch.reload            # commits are not refused (measured)
    scratch.reload = refused
    ctx = _Ctx()
    m._execute_job(ctx, "rf")
    assert jobs["rf"]["state"] == "done", jobs["rf"].get("error")
    assert ctx.read == [body]
