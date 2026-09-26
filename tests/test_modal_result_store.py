"""Modal with its results in an object store (step 7 of docs/cache-consolidation.md, 2026-09-26).

With HAVERSACK_RESULT_STORE the store is the one authority for results and each container keeps
only a local copy on its own disk; the cache volume is neither made nor mounted, and none of
the volume-view machinery (reloads, the view lock, the mirror, commits) is in a result's path.
These drive the real modal_app code - a worker's `_execute_job` and artifact overlap, the
api's lookups, listing, delete and `result:` reader, the twin's read-only view - against an
in-memory object store (obstore's MemoryStore honors conditional writes), with a cache-volume
double that fails the test on ANY use.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("modal")
from obstore.store import MemoryStore                            # noqa: E402

from haversack.objectcache import SharedResultCache              # noqa: E402
from haversack.serve import ResultCache                          # noqa: E402

from test_worker_volume_view import _Ctx, _HidingVolume, _submit  # noqa: E402


class _NoVolume:
    """The cache volume, which a store deployment must never touch. Every touch is RECORDED
    as well as raised: some callers swallow what they raise (the artifact overlap logs and
    carries on), and the fixture fails the test on any touch at all."""

    def __init__(self):
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name):
        self.touched.append(name)
        raise AssertionError(f"the cache volume was used ({name}) with a result store")


@pytest.fixture
def store(monkeypatch, tmp_path):
    from haversack import modal_app as m
    mem = MemoryStore()
    jobs = {}
    (tmp_path / "scratch").mkdir()
    scratch = _HidingVolume(tmp_path / "scratch")
    monkeypatch.setattr(m, "RESULT_STORE", "memory://results")
    novol = _NoVolume()
    monkeypatch.setattr(m, "cache_vol", novol)
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "no-cache-volume"))
    monkeypatch.setattr(m, "jobs_dict", jobs)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(scratch.root))
    monkeypatch.setattr(m, "scratch_vol", scratch)
    monkeypatch.setattr(m, "JOB_LOCAL_ROOT", str(tmp_path / "local"), raising=False)
    monkeypatch.setattr(m, "ARTIFACTS", set())
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_sweep_due", lambda: False)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)
    monkeypatch.setattr(m, "_TWIN", False)

    def container(name, read_only=False):
        """Make this process one container: its own local copy over the one shared store."""
        view = SharedResultCache(mem, ResultCache(tmp_path / f"copy-{name}"), check=not read_only,
                                 read_only=read_only)
        monkeypatch.setattr(m, "_results_held", {read_only: view})
        return view
    yield m, jobs, container
    assert novol.touched == [], f"the cache volume was touched: {novol.touched}"


def _computed(m, jobs, container, jid="j00"):
    container("worker")
    body = _submit(m, jobs, jid)
    m._execute_job(_Ctx(), jid)
    assert jobs[jid]["state"] == "done", jobs[jid].get("error")
    return body


def test_a_worker_publishes_into_the_store_and_another_container_serves_it(store):
    m, jobs, container = store
    body = _computed(m, jobs, container)
    api = container("api")                              # another container: an empty copy
    hit = m._read_cache("key-j00")
    assert hit is not None and Path(hit[0]).read_bytes() == b"labels of " + body
    assert str(api.root) in str(hit[0])                 # served from ITS copy, filled
    assert m.ModalExecutor().cache_get("key-j00")[0] == hit[0]
    rec = m.ModalExecutor()._cache_record("key-j00")
    assert rec is not None and rec[0] == Path(hit[0])


def test_a_miss_is_verified_by_the_pointer_itself(store):
    m, jobs, container = store
    container("api")
    since = time.monotonic()
    assert m._confirm_cache_absent("key-never", since) is None     # no ResultsNotVisible
    assert m.ModalExecutor().confirm_absent("key-never", since) is None


def test_the_artifact_render_lands_in_the_store(store):
    m, jobs, container = store
    _computed(m, jobs, container)
    worker = m._results_held[False]
    gen = worker.generation("key-j00")
    png = Path(m.SCRATCH_ROOT).parent / "p.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    from haversack import serve

    def overlap(pair, task, artifacts, *, preview_out, statistics_out, place, finish,
                unavailable=None):
        ok = place("preview.png", png)
        finish([("preview", 0.0)] if ok else [])
    real = serve.artifact_overlap
    serve.artifact_overlap = overlap
    try:
        m._WorkerBase._artifact_worker(_Ctx(), object(), "key-j00", "j00", "ts.v2:total_fast",
                                       generation=gen)
    finally:
        serve.artifact_overlap = real
    container("api")
    hit = m._read_cache("key-j00")
    assert (Path(hit[0]).parent / "preview.png").read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_the_listing_and_a_delete_are_the_stores(store):
    m, jobs, container = store
    _computed(m, jobs, container)
    container("api")
    rows, _ = m._list_cache()
    assert [r["key"] for r in rows] == ["key-j00"]
    assert m.ModalExecutor().cache_delete("key-j00")
    container("elsewhere")                              # a third container sees the tombstone
    assert m._read_cache("key-j00") is None
    assert m._list_cache()[0] == []


def test_the_twin_reads_through_a_view_that_cannot_write(store):
    m, jobs, container = store
    _computed(m, jobs, container)
    view = container("twin", read_only=True)
    m._TWIN = True
    try:
        assert m._read_cache("key-j00") is not None
        assert [r["key"] for r in m._list_cache()[0]] == ["key-j00"]
        from haversack.errors import InputError
        with pytest.raises(InputError):
            view.delete("key-j00")
    finally:
        m._TWIN = False


def test_a_result_reference_reads_the_store_in_the_api_and_the_worker(store):
    m, jobs, container = store
    body = _computed(m, jobs, container)
    container("api")
    with m._api_result_entry("key-j00", fresh=True) as hit:
        assert Path(hit[0]).read_bytes() == b"labels of " + body
    container("worker2")
    import threading
    with m._worker_result_entry(threading.Lock())("key-j00", fresh=True) as hit:
        assert Path(hit[0]).read_bytes() == b"labels of " + body


def test_the_worker_records_the_tasks_versions_for_readers(store, monkeypatch):
    m, jobs, container = store
    from haversack import serve
    monkeypatch.setattr(serve, "versions_for", lambda seg, task, kind="segment": ["297=v2", "e3"])
    monkeypatch.setattr(serve, "installed_versions", lambda seg, task: ["v2"])
    _computed(m, jobs, container)
    doc = m._results_held[False].task_versions("ts.v2:total_fast")
    assert doc is not None and "297=v2" in str(doc)


def test_the_attach_preflight_does_not_probe_a_cache_volume_there_is_not(store, tmp_path,
                                                                         monkeypatch):
    m, _, _ = store
    for name in ("inputs", "weights"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(m, "INPUTS_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setattr(m, "WEIGHTS_ROOT", str(tmp_path / "weights"))
    m._check_volumes_attached()                         # CACHE_ROOT does not exist: not asked


def test_the_module_imports_as_a_store_deployment_defines_it():
    """Every test above patches a module imported WITHOUT a store; a deploy imports it WITH
    one, where the scheduled sweep and the conditional mounts exist. The first deploy failed at
    that import (modal.Period takes whole hours; a float was passed), which none of them saw."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "HAVERSACK_RESULT_STORE": "s3://bucket/prefix",
           "HAVERSACK_RESULT_SWEEP_HOURS": "0.5",
           "PYTHONPATH": os.pathsep.join(p for p in (str(root / "src"), os.environ.get("PYTHONPATH", "")) if p)}
    code = ("import haversack.modal_app as m; "
            "assert m.cache_vol is None and m._results_mount() == {}; "
            "assert m._STORE_SECRETS and hasattr(m, 'sweep_results'); print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
