"""`result:` references on Modal: both resolutions read the CACHE volume.

A reference is resolved twice - in the api container at submit, which keys the job on the
referenced output's digest, and again in the worker that fetches it, because a lease taken
in the api container does not reach a worker's pruning. Both follow the rules the fixes of
2026-09-19 established (AGENTS.md, "A reload in one thread hides the volume from the
others"): nothing is read from a volume another thread of the container may be reloading,
what is handed out is a local copy made under the lock, and a miss is believed only from a
view newer than the question.

The volume doubles are the ones those fixes were tested with: one whose view LAGS (what
was committed elsewhere enters it only when a reload takes, after a set number of
refusals) and one whose reload HIDES the root from the container's other threads.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
pytest.importorskip("modal")

from haversack import content  # noqa: E402
from haversack.serve import RESULT_NAME, ResultCache  # noqa: E402

from test_stale_volume_view import _LaggingVolume, _commit_elsewhere, _modal  # noqa: E402
from test_worker_volume_view import _HidingVolume, _Seg  # noqa: E402

KEY = "a" * 64


# -- the api container, at submit ------------------------------------------------------

def _api(monkeypatch, tmp_path, *, refusals):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=refusals)
    from dataclasses import replace

    from haversack.engines import registry
    from haversack.schemas import label_input
    monkeypatch.setitem(registry.ENGINES, "nnunetv2",
                        replace(registry.ENGINES["nnunetv2"], multi_input=True))
    seg = ex.segmenter
    seg.tasks = lambda: ["total_fast", "total", "relabel"]
    seg.describe = lambda task: {"name": task, "engine": "nnunetv2",
                                 **({"inputs": [label_input("mask")]} if task == "relabel" else {})}
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None: (
        spawned.append(jid) or types.SimpleNamespace(object_id="fc-1")))
    return m, jobs, vol, client, spawned


def _refer(client, ident=KEY):
    return client.post("/v1/jobs", data={"task": "relabel", "source": json.dumps(
        [{"kind": "result", "id": ident}])})


def test_submit_resolves_a_result_the_stale_view_missed(monkeypatch, tmp_path):
    """The upstream worker committed it; this container's view has not caught up and its
    first reload is refused. A miss read from that view is not believed: the confirming
    reload takes, and the job is keyed on the digest it finds."""
    m, jobs, vol, client, spawned = _api(monkeypatch, tmp_path, refusals=1)
    _, res = _commit_elsewhere(m, vol, tmp_path, KEY)
    r = _refer(client)
    assert r.status_code == 202, r.text
    (jid,) = spawned
    digest = res["outputs"][0]["sha256"]
    assert jobs[jid]["input_identity"] == [digest]
    # what the WORKER is handed: the pinned identifier, so it can ask again and compare
    assert jobs[jid]["source"] == [{"kind": "result", "id": f"{KEY}!labels@{digest}"}]
    assert vol.reloads == 2


def test_a_reference_that_cannot_be_seen_is_503_not_a_refusal(monkeypatch, tmp_path):
    m, jobs, vol, client, spawned = _api(monkeypatch, tmp_path, refusals=10 ** 6)
    _commit_elsewhere(m, vol, tmp_path, KEY)
    r = _refer(client)
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["code"] == "not_visible_yet" and r.headers["retry-after"]
    assert not spawned and not [k for k in jobs if ":" not in k]
    vol.refusals = 0                               # the open files close: it was there
    assert _refer(client).status_code == 202


def test_a_verified_miss_is_refused_with_the_fix(monkeypatch, tmp_path):
    m, jobs, vol, client, spawned = _api(monkeypatch, tmp_path, refusals=0)
    r = _refer(client)
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "result_missing" and not spawned
    assert vol.reloads >= 1                        # believed only once a reload had taken


def test_the_api_reads_the_entry_from_its_local_mirror(monkeypatch, tmp_path):
    """The lookup hands out the container-local copy of the generation, meta.json
    included: nothing on the volume is open once the shared lock is released."""
    m, jobs, vol, client, spawned = _api(monkeypatch, tmp_path, refusals=0)
    _commit_elsewhere(m, vol, tmp_path, KEY)
    seen = []
    with m._api_result_entry(KEY, fresh=True) as hit:
        seen.append(Path(hit[0]))
    (path,) = seen
    assert str(path).startswith(m.MIRROR_ROOT) and (path.parent / "meta.json").exists()


# -- a worker, fetching ----------------------------------------------------------------

class _Ctx:
    """A worker as _execute_job sees it, wired by the helpers ``setup`` itself calls."""
    engine = "nnunetv2"
    seg = None
    read_ahead = types.SimpleNamespace(pop=lambda key: None)

    def __init__(self, m, root: Path):
        self._vol_lock = threading.Lock()
        self._sources = m._worker_sources(self._vol_lock)
        self.series_cache = m._worker_series_cache(self._sources, root / "series", 1 << 30)
        self.read = []

    def _ensure(self, task):
        pass

    def _compute(self, input_path, meta, on_progress, token):
        (file,) = [p for p in Path(input_path).iterdir() if not p.name.startswith(".")]
        seen = set()
        for _ in range(5):                         # a compute reads its input for a while
            seen.add(file.read_bytes())
            time.sleep(0.001)
        (body,) = seen
        self.read.append(body)
        return _Seg(b"computed from " + body)


def _worker(monkeypatch, tmp_path, cache_vol_of):
    from haversack import modal_app as m
    jobs = {}
    for name in ("scratch", "cache"):
        (tmp_path / name).mkdir()
    scratch, cache = _HidingVolume(tmp_path / "scratch"), cache_vol_of(tmp_path / "cache")
    monkeypatch.setattr(m, "jobs_dict", jobs)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(scratch.root))
    monkeypatch.setattr(m, "CACHE_ROOT", str(cache.root))
    monkeypatch.setattr(m, "JOB_LOCAL_ROOT", str(tmp_path / "local"), raising=False)
    monkeypatch.setattr(m, "scratch_vol", scratch)
    monkeypatch.setattr(m, "cache_vol", cache)
    monkeypatch.setattr(m, "ARTIFACTS", set())
    monkeypatch.setattr(m, "CACHE_CONFIRM_DELAYS_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_sweep_due", lambda: False)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)
    return m, jobs, scratch, cache


def _publish(root, key: str, body: bytes, tmp_path: Path) -> str:
    """An upstream result in the cache at ``root``; returns its labels' digest."""
    src = tmp_path / f"labels-{key[:6]}-{len(body)}"
    src.write_bytes(body)
    digest = content.digest_file(src)
    ResultCache(root).put(key, src, {"outputs": [{"name": "labels", "sha256": digest,
                                                  "bytes": len(body)}]},
                          {"task": "ts.v2:total"})
    return digest


def _job(jobs, jid: str, key: str, digest: str):
    jobs[jid] = {"id": jid, "task": "ts.v2:total_fast", "state": "queued",
                 "source": [{"kind": "result", "id": f"{key}!labels@{digest}"}],
                 "input_identity": [digest], "cache_key": f"down-{jid}",
                 "created": time.time()}


def _others(m, ctx, stop):
    """What else runs in a worker container and touches these volumes, as the worker runs
    it: under ``_vol_lock``. A cache commit may end in a reload, which hides the volume."""
    def cache_commits():
        while not stop.is_set():
            with ctx._vol_lock:
                m.cache_vol.commit()
            time.sleep(0.0005)

    def scratch_reloads():
        while not stop.is_set():
            with ctx._vol_lock:
                m.scratch_vol.reload()
            time.sleep(0.0005)
    return [threading.Thread(target=cache_commits), threading.Thread(target=scratch_reloads)]


def test_a_deep_queue_of_references_fetched_while_the_cache_volume_reloads(
        monkeypatch, tmp_path):
    """Thirty jobs, each fetching its own upstream result, beside threads that commit and
    reload the volumes the whole time. A fetch that looked the entry up, or copied it,
    outside the lock meets a volume that is not there: a "missing" result, a
    FileNotFoundError, or a copy cut short - every one of which fails a job."""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    bodies = {}
    for i in range(30):
        key = f"{i:064x}"
        bodies[f"j{i:02d}"] = (key, f"labels of upstream {i} ".encode() * 400)
        _job(jobs, f"j{i:02d}", key, _publish(m.CACHE_ROOT, key, bodies[f"j{i:02d}"][1], tmp_path))
    ctx = _Ctx(m, tmp_path)
    stop = threading.Event()
    threads = _others(m, ctx, stop)
    for t in threads:
        t.start()
    try:
        for jid in bodies:
            m._execute_job(ctx, jid)
    finally:
        stop.set()
        for t in threads:
            t.join()

    failed = {j: jobs[j].get("error", "")[:240] for j in bodies if jobs[j]["state"] != "done"}
    assert not failed, failed
    assert ctx.read == [body for _, body in bodies.values()]
    assert not cache.lost and not scratch.lost, (cache.lost[:2], scratch.lost[:2])
    assert cache.reloads > 60, cache.reloads       # the race was actually run
    for jid, (key, body) in bodies.items():        # and what each published is its own
        hit = ResultCache(m.CACHE_ROOT).get(f"down-{jid}")
        assert hit is not None and Path(hit[0]).read_bytes() == b"computed from " + body


def test_the_volume_is_touched_only_under_the_lock(monkeypatch, tmp_path):
    """Deterministic where the race is not: every reload, every lookup and the copy itself
    happen with ``_vol_lock`` held - ResultCache() too, which mkdirs its root."""
    from haversack import serve
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    digest = _publish(m.CACHE_ROOT, KEY, b"the labels", tmp_path)
    ctx = _Ctx(m, tmp_path)
    unlocked = []

    def watch(name, real):
        def call(*a, **k):
            if not ctx._vol_lock.locked():
                unlocked.append(name)
            return real(*a, **k)
        return call
    monkeypatch.setattr(serve.ResultCache, "__init__", watch("ResultCache()", serve.ResultCache.__init__))
    monkeypatch.setattr(serve.ResultCache, "get", watch("get", serve.ResultCache.get))
    monkeypatch.setattr(cache, "reload", watch("reload", cache.reload))
    real_copy = shutil.copyfile

    def copyfile(src, dst, **k):
        if str(src).startswith(m.CACHE_ROOT) and not ctx._vol_lock.locked():
            unlocked.append("copy")
        return real_copy(src, dst, **k)
    monkeypatch.setattr(shutil, "copyfile", copyfile)

    src = ctx._sources["result"]
    (tmp_path / "e1").mkdir()
    src.fetch(f"{KEY}!labels@{digest}", tmp_path / "e1")            # the stale look is enough
    assert cache.reloads == 0
    (tmp_path / "e2").mkdir()
    with pytest.raises(Exception, match="the referenced result changed"):
        src.fetch(f"{KEY}!labels@sha256:{'0' * 64}", tmp_path / "e2")   # ...and the fresh one
    assert cache.reloads == 1
    assert not unlocked, unlocked
    assert not ctx._vol_lock.locked()


def test_a_result_republished_since_submit_fails_the_job_by_name(monkeypatch, tmp_path):
    """The api keyed the job on one digest; by the time the worker reads the entry it holds
    another. The worker asks a fresh view, finds the same, and computes nothing."""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    was = _publish(m.CACHE_ROOT, KEY, b"what the api saw", tmp_path)
    now = _publish(m.CACHE_ROOT, KEY, b"what a no-cache recompute published since", tmp_path)
    _job(jobs, "jx", KEY, was)
    ctx = _Ctx(m, tmp_path)
    m._execute_job(ctx, "jx")
    assert jobs["jx"]["state"] == "failed"
    assert "the referenced result changed" in jobs["jx"]["error"] and now in jobs["jx"]["error"]
    assert ctx.read == [] and cache.reloads >= 1
    assert ResultCache(m.CACHE_ROOT).get("down-jx") is None


def test_a_worker_whose_view_is_behind_reloads_before_it_calls_a_result_changed(
        monkeypatch, tmp_path):
    """The other way round: the api saw the NEW generation and this worker's view still
    holds the old one. "Other bytes" read from a stale view is not believed either."""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, lambda root: _LaggingVolume(root, 0))
    _publish(m.CACHE_ROOT, KEY, b"the old generation", tmp_path)
    shutil.copytree(m.CACHE_ROOT, cache.hidden)
    now = _publish(cache.hidden, KEY, b"the generation the api saw", tmp_path)
    _job(jobs, "jl", KEY, now)
    ctx = _Ctx(m, tmp_path)
    m._execute_job(ctx, "jl")
    assert jobs["jl"]["state"] == "done", jobs["jl"].get("error")
    assert ctx.read == [b"the generation the api saw"] and cache.reloads == 1


def test_a_worker_whose_view_predates_the_change_computes_from_the_pinned_bytes(
        monkeypatch, tmp_path):
    """Seen on Modal, 2026-09-20: twelve consumers held queued while another container
    republished or evicted their upstream finished DONE, and rightly. A worker's view is a
    snapshot; this one still holds the generation the job was pinned to, the digest matches
    and the copy hashes to the pin, so it computes from exactly the bytes its key was built
    from - and needs no reload to know it. The rule is "never from OTHER bytes"."""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, lambda root: _LaggingVolume(root, 0))
    pinned = _publish(m.CACHE_ROOT, KEY, b"the generation the job was keyed on", tmp_path)
    shutil.copytree(m.CACHE_ROOT, cache.hidden)
    _publish(cache.hidden, KEY, b"republished elsewhere since", tmp_path)
    _job(jobs, "jv", KEY, pinned)
    ctx = _Ctx(m, tmp_path)
    m._execute_job(ctx, "jv")
    assert jobs["jv"]["state"] == "done", jobs["jv"].get("error")
    assert ctx.read == [b"the generation the job was keyed on"] and cache.reloads == 0


def test_a_result_the_worker_cannot_see_is_not_called_missing(monkeypatch, tmp_path):
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path,
                                      lambda root: _LaggingVolume(root, 10 ** 6))
    (tmp_path / "elsewhere").mkdir()
    digest = _publish(tmp_path / "elsewhere", KEY, b"committed by another container", tmp_path)
    _job(jobs, "jn", KEY, digest)
    ctx = _Ctx(m, tmp_path)
    m._execute_job(ctx, "jn")
    assert jobs["jn"]["state"] == "failed"
    assert "cannot see result" in jobs["jn"]["error"] and "no result" not in jobs["jn"]["error"]
    assert ctx.read == []


def test_a_verified_miss_in_the_worker_names_what_to_compute(monkeypatch, tmp_path):
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    _job(jobs, "jm", KEY, "sha256:" + "3" * 64)
    ctx = _Ctx(m, tmp_path)
    m._execute_job(ctx, "jm")
    assert jobs["jm"]["state"] == "failed" and f"no result {KEY}" in jobs["jm"]["error"]


def test_what_the_worker_records_about_the_input(monkeypatch, tmp_path):
    """The Modal worker and the local server answer provenance from the same record."""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    digest = _publish(m.CACHE_ROOT, KEY, b"the labels", tmp_path)
    _job(jobs, "jp", KEY, digest)
    m._execute_job(_Ctx(m, tmp_path), "jp")
    assert jobs["jp"]["state"] == "done", jobs["jp"].get("error")
    (rec,) = jobs["jp"]["result"]["provenance"]["inputs"]
    assert rec["kind"] == "result" and rec["identity"] == digest
    assert rec["content"]["digest"] == digest
    assert rec["origin"]["result"] == KEY and rec["origin"]["task"] == "ts.v2:total"


def test_the_prefetcher_never_stages_a_reference(monkeypatch, tmp_path):
    """So the prefetch thread never reads the cache volume for one. (The fetch takes the
    lock itself; this is the economy, that is the guarantee.)"""
    m, jobs, scratch, cache = _worker(monkeypatch, tmp_path, _HidingVolume)
    _job(jobs, "jq", KEY, "sha256:" + "3" * 64)
    assert m._prefetch_candidate("running-now") is None
    jobs["ju"] = {"id": "ju", "task": "ts.v2:total_fast", "state": "queued",
                  "source": [{"kind": "idc", "id": "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"}],
                  "created": time.time()}
    assert m._prefetch_candidate("running-now")[2] == "ju"


def test_setup_wires_the_worker_through_the_helpers_these_tests_drive():
    """The tests above build their worker with _worker_sources and _worker_series_cache;
    this holds ``setup`` to the same two calls, and to making the lock first."""
    import ast
    import inspect

    from haversack import modal_app as m
    # from the module's source: `setup` itself is a modal PartialFunction, not a function
    module = ast.parse(Path(inspect.getsourcefile(m)).read_text(encoding="utf-8"))
    base = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "_WorkerBase")
    tree = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "setup")
    order = [ast.unparse(n.targets[0]) for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) in ("self._vol_lock", "self._sources",
                                               "self.series_cache")]
    calls = {ast.unparse(n.targets[0]): ast.unparse(n.value.func) for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)}
    assert order == ["self._vol_lock", "self._sources", "self.series_cache"], order
    assert calls["self._sources"] == "_worker_sources"
    assert calls["self.series_cache"] == "_worker_series_cache"
