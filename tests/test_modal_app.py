"""The Modal deployment module is import-safe and shaped as create_app expects."""
import copy
import pathlib
import re

import pytest

modal = pytest.importorskip("modal")


def test_modal_app_imports_and_executor_matches_protocol():
    from haversack import modal_app
    assert modal_app.app.name == modal_app.APP_NAME
    ex = modal_app.ModalExecutor
    for attr in ("new_job_dir", "submit", "status_of", "statuses", "cancel", "result_file"):
        assert callable(getattr(ex, attr)), attr
    assert ex.supports_push is False
    assert modal_app.Worker is not None


def test_worker_uses_series_cache():
    from haversack import modal_app
    import inspect
    src = inspect.getsource(modal_app)
    assert "SeriesCache" in src
    assert "series_cache.get_or_fetch" in src


def test_purgeable_policy():
    from haversack.modal_app import _purgeable
    now = 1000000.0
    ttl = 3600.0
    assert _purgeable({"state": "done", "finished": now - 7200}, now, ttl)
    assert _purgeable({"state": "failed", "finished": now - 7200}, now, ttl)
    assert not _purgeable({"state": "done", "finished": now - 60}, now, ttl)
    # active records are never purged by age
    assert not _purgeable({"state": "queued", "created": now - 10 ** 6}, now, ttl)
    assert not _purgeable({"state": "running", "started": now - 10 ** 6}, now, ttl)
    # garbage records are purgeable
    assert _purgeable(None, now, ttl)


def test_emit_is_terminal_wins():
    """A worker progress emit racing an API cancel must not resurrect a
    terminal record to running - the lost-cancel wedge."""
    from haversack import modal_app

    class FakeDict(dict):
        pass

    orig = modal_app.jobs_dict
    modal_app.jobs_dict = fake = FakeDict()
    try:
        modal_app._emit("j1", {"state": "running", "started": 1.0})
        modal_app._emit("j1", {"state": "cancelled", "finished": 2.0})
        modal_app._emit("j1", {"state": "running", "progress": {"stage": "x"}})
        assert fake["j1"]["state"] == "cancelled"          # cancel survives
        modal_app._emit("j1", {"progress": {"stage": "y"}})
        assert fake["j1"]["state"] == "cancelled"          # stateless merge too
        # a cancel is final (2026-09-19): a worker that ran on past the DELETE, as one
        # did during model loading on a smoke, must not report done
        modal_app._emit("j1", {"state": "done", "finished": 3.0})
        assert fake["j1"]["state"] == "cancelled"
        # other terminal states may still be replaced: a `done` after the orphan
        # rule's `failed` is the job recovering from a false alarm
        modal_app._emit("j2", {"state": "failed", "finished": 2.0})
        modal_app._emit("j2", {"state": "done", "finished": 3.0})
        assert fake["j2"]["state"] == "done"
    finally:
        modal_app.jobs_dict = orig


def _swap_dict(monkeypatch):
    from haversack import modal_app
    fake = {}
    monkeypatch.setattr(modal_app, "jobs_dict", fake)
    return modal_app, fake


def test_prefetch_candidate_follows_jobpolicy(monkeypatch):
    """The scan's exclusions are jobpolicy's, not its own. A job that asked for
    fresh bytes and a multi-input job were both selected before - and the first
    was then pinned by the pre-read, so its own no-cache refresh was refused."""
    m, fake = _swap_dict(monkeypatch)
    fake["inflight:K"] = "x"                     # markers never crash the scan
    fake["cancel:y"] = 1.0
    fake["a"] = {"id": "a", "state": "queued", "created": 1, "refresh_input": True,
                 "source": [{"kind": "s3", "id": "b/a"}]}
    fake["b"] = {"id": "b", "state": "queued", "created": 2,
                 "source": [{"kind": "s3", "id": "b/1", "role": "image"},
                            {"kind": "s3", "id": "b/2", "role": "mask"}]}
    fake["c"] = {"id": "c", "state": "queued", "created": 3, "kind": "prepare"}
    fake["d"] = {"id": "d", "state": "queued", "created": 4,
                 "source": [{"kind": "input", "id": "sha256:0"}]}
    fake["e"] = {"id": "e", "state": "queued", "created": 5,
                 "source": [{"kind": "s3", "id": "b/e"}]}
    fake["f"] = {"id": "f", "state": "running", "created": 0}   # the current job
    assert m._prefetch_candidate("f") == ("s3", "s3:b/e", "e")
    fake["g"] = {"id": "g", "state": "queued", "created": 0.5}  # an upload, older
    assert m._prefetch_candidate("f") == ("upload", None, "g")


def test_a_worker_warms_only_the_jobs_that_will_run_on_it(monkeypatch):
    """Each engine worker is its own container with its own caches, sharing one
    jobs dict. Seen live with five engines deployed: the SynthStrip container
    pre-read the nnU-Net worker's upload, the nnU-Net worker staged a FastSurfer
    job's series and then stopped (one-ahead), and its own next job ran cold."""
    m, fake = _swap_dict(monkeypatch)
    fake["cur"] = {"id": "cur", "state": "running", "created": 0, "task": "ts.v2:total"}
    fake["fs"] = {"id": "fs", "state": "queued", "created": 1, "task": "fastsurfer:asegdkt",
                  "source": [{"kind": "s3", "id": "b/mprage"}]}
    fake["nn"] = {"id": "nn", "state": "queued", "created": 2, "task": "ts.v2:total_fast"}
    assert m._prefetch_candidate("cur", "nnunetv2") == ("upload", None, "nn")
    assert m._prefetch_candidate("cur", "fastsurfer") == ("s3", "s3:b/mprage", "fs")
    assert m._prefetch_candidate("cur", "monai") is None
    assert m._prefetch_candidate("cur") == ("s3", "s3:b/mprage", "fs")   # unfiltered: oldest


def _bound(m, jid):
    """The per-job pass, then its background sweep to completion, if one started."""
    t = m._bound_jobs_store(jid)
    if t is not None:
        t.join(10)
        assert not t.is_alive()


def test_orphaned_records_are_failed_and_live_ones_left_alone(monkeypatch, tmp_path):
    """Records whose spawned call is gone (a `modal app stop` between deploys)
    stayed queued forever: never purged, poisoning single-flight for their key,
    and starving the prefetcher as the oldest candidates. Found live with five
    of them 76 hours old."""
    m, fake = _swap_dict(monkeypatch)
    now = 1_000_000.0
    calls = {"dead": "dead", "live": "live", "fin": "finished"}
    monkeypatch.setattr(m, "_call_state", lambda cid: calls.get(cid, "dead"))
    old = now - 76 * 3600
    fake["stale"] = {"id": "stale", "state": "queued", "created": old, "call_id": "dead",
                     "cache_key": "K"}
    fake["inflight:K"] = "stale"
    fake["alive"] = {"id": "alive", "state": "queued", "created": old, "call_id": "live"}
    fake["young"] = {"id": "young", "state": "queued", "created": now - 5, "call_id": "dead"}
    fake["crashed"] = {"id": "crashed", "state": "running", "created": old,
                       "started": old + 1, "call_id": "fin"}
    fake["nocall"] = {"id": "nocall", "state": "queued", "created": old}
    fake["me"] = {"id": "me", "state": "running", "created": old, "call_id": "dead"}
    fake["done"] = {"id": "done", "state": "done", "created": old, "finished": old}

    assert sorted(m._reconcile_orphans("me", now)) == ["crashed", "nocall", "stale"]
    assert fake["stale"]["state"] == "failed" and "resubmit" in fake["stale"]["error"]
    assert fake["crashed"]["state"] == "failed"
    assert fake["alive"]["state"] == "queued"          # its call answers "still here"
    assert fake["young"]["state"] == "queued"          # too fresh to probe at all
    assert fake["me"]["state"] == "running"            # never the job that is running this
    assert fake["done"]["state"] == "done"

    # ...and the retention pass runs it first, so the orphan's inflight marker
    # is dropped in the same pass instead of surviving to the next job
    class Vol:
        def commit(self):
            pass
    monkeypatch.setattr(m, "scratch_vol", Vol())
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path))
    fake["stale"]["state"] = "queued"                  # reset: prove the pass does it
    monkeypatch.setattr(m.time, "time", lambda: now)
    monkeypatch.setattr(m, "_last_sweep", float("-inf"))   # this container's first job
    monkeypatch.setattr(m, "_sweep_thread", None)
    _bound(m, "me")
    assert fake["stale"]["state"] == "failed"
    assert "inflight:K" not in fake


class _CountingDict(dict):
    """A jobs Dict that counts its RPCs and, like Modal's, hands out COPIES - a
    plain dict shares the record objects `_emit` mutates, so a listing taken
    before a write would show it. `items()` can also be made to answer a stale
    listing, as Modal's unordered, non-atomic DictContents stream may."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rpcs = {"get": 0, "keys": 0, "items": 0}
        self.listing = None

    def get(self, k, default=None):
        self.rpcs["get"] += 1
        return copy.deepcopy(super().get(k, default))

    def __getitem__(self, k):
        self.rpcs["get"] += 1
        return copy.deepcopy(super().__getitem__(k))

    def keys(self):
        self.rpcs["keys"] += 1
        return super().keys()

    def items(self):
        self.rpcs["items"] += 1
        return copy.deepcopy(list(self.listing if self.listing is not None
                                  else super().items()))


def _sweep_env(monkeypatch, tmp_path, fake):
    from haversack import modal_app as m
    commits = []

    class Vol:
        def commit(self):
            commits.append(1)
    monkeypatch.setattr(m, "jobs_dict", fake)
    monkeypatch.setattr(m, "scratch_vol", Vol())
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(m, "_last_sweep", float("-inf"))
    monkeypatch.setattr(m, "_sweep_thread", None)
    monkeypatch.setattr(m, "_call_state", lambda cid: "live")
    return m, commits


def test_the_worker_scans_read_the_jobs_dict_in_one_rpc_whatever_its_size(monkeypatch,
                                                                            tmp_path):
    """Measured 2026-09-19: the prefetch scan (every 2 s per busy worker) and the
    reconcile + purge (after every job) each listed the Dict and `get` every
    record, O(N^2) RPCs over a cohort of N - six workers drained 15 of 200 queued
    jobs during a 30 s submit burst, against 199 after. Each scan is now one
    streamed listing, however many records there are."""
    for n in (10, 200):
        fake = _CountingDict()
        m, _ = _sweep_env(monkeypatch, tmp_path, fake)
        now = m.time.time()
        for i in range(n):
            fake[f"j{i}"] = {"id": f"j{i}", "state": "queued", "created": now - 1,
                             "task": "ts.v2:total", "source": [{"kind": "s3", "id": f"b/{i}"}]}
            fake[f"inflight:K{i}"] = f"j{i}"
            fake[f"artifacts:K{i}"] = {"state": "pending", "t": now, "job": f"j{i}"}
            fake[f"cancel:c{i}"] = now
            fake[f"d{i}"] = {"id": f"d{i}", "state": "done", "created": now - 5,
                             "finished": now - 1}
        assert m._prefetch_candidate("j0", "nnunetv2") == ("s3", "s3:b/1", "j1")
        assert m._reconcile_orphans("j0") == []
        _bound(m, "j0")
        assert fake.rpcs == {"get": 0, "keys": 0, "items": 3}, (n, fake.rpcs)


def test_the_retention_sweep_runs_at_most_once_per_interval_per_container(monkeypatch,
                                                                           tmp_path):
    """The finished job's own upload goes after every job; the whole-Dict sweep
    at most once per JOBS_SWEEP_EVERY_S - and on the first job of a container."""
    fake = _CountingDict()
    m, commits = _sweep_env(monkeypatch, tmp_path, fake)
    clock = [1000.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    old = m.time.time() - m.JOBS_TTL_H * 3600 - 10

    def job(jid):
        (tmp_path / jid).mkdir()
        (tmp_path / jid / "input_x.nii.gz").write_bytes(b"x")
        fake[jid] = {"id": jid, "state": "done", "created": old, "finished": old}
        _bound(m, jid)
        assert not (tmp_path / jid / "input_x.nii.gz").exists(), "the upload stays"

    job("a")
    assert fake.rpcs["items"] == 1 and len(commits) == 1
    job("b")                                           # a moment later: no sweep
    assert "a" in fake and fake.rpcs["items"] == 1 and len(commits) == 2
    clock[0] += m.JOBS_SWEEP_EVERY_S - 1
    job("c")
    assert fake.rpcs["items"] == 1
    clock[0] += 1                                      # the interval has passed
    job("d")                                           # its own commit + the purge's
    assert fake.rpcs["items"] == 2 and len(commits) == 5
    assert sorted(k for k in fake if ":" not in k) == ["d"]   # a b c purged, d is current
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d"]   # and their directories


def test_the_sweep_rereads_before_it_deletes_what_a_stale_listing_shows(monkeypatch,
                                                                         tmp_path):
    """The listing is one stream, unordered and not atomic: a write landing while
    it runs may be missing from it. So a marker is deleted only if a fresh read
    still says it should be - the guard the per-key reads gave for free."""
    fake = _CountingDict()
    m, _ = _sweep_env(monkeypatch, tmp_path, fake)
    now = m.time.time()
    old = now - m.JOBS_TTL_H * 3600 - 10
    fake.listing = [
        # purged in this pass, but a NEWER flight has installed its marker since
        ("old", {"id": "old", "state": "done", "created": old, "finished": old}),
        ("inflight:K", "old"),
        # a marker listed without its job: the job is queued, written mid-stream
        ("inflight:L", "new"),
        # a pending marker listed stale, since replaced by a new flight's
        ("artifacts:A", {"state": "pending", "t": now - 1000, "job": "x"}),
        ("cancel:c", now - 1000),
        # the control: stale in the listing and stale now
        ("inflight:M", "gone"),
        ("artifacts:B", {"state": "pending", "t": now - 1000, "job": "y"}),
    ]
    fake.update({"inflight:K": "newer", "inflight:L": "new",
                 "new": {"id": "new", "state": "queued", "created": now},
                 "newer": {"id": "newer", "state": "queued", "created": now},
                 "old": dict(fake.listing[0][1]),
                 "artifacts:A": {"state": "pending", "t": now, "job": "z"},
                 "cancel:c": now,
                 "inflight:M": "gone",
                 "artifacts:B": {"state": "pending", "t": now - 1000, "job": "y"}})
    _bound(m, "cur")
    assert "old" not in fake                            # the purge itself happened
    assert fake["inflight:K"] == "newer", "a newer flight's marker was deleted"
    assert fake["inflight:L"] == "new", "a live job's marker was deleted"
    assert fake["artifacts:A"]["job"] == "z", "a fresh pending marker was deleted"
    assert "cancel:c" in fake, "a fresh cancel was deleted"
    assert "inflight:M" not in fake and "artifacts:B" not in fake


def test_a_marker_whose_job_ended_is_dropped_in_the_sweep(monkeypatch, tmp_path):
    """The marker rules the snapshot must keep: a terminal (or absent) job's
    inflight marker goes, an active job's stays; a pending marker and a cancel
    go only past 900 s; a job's own record is never purged under it."""
    fake = _CountingDict()
    m, _ = _sweep_env(monkeypatch, tmp_path, fake)
    now = m.time.time()
    old = now - m.JOBS_TTL_H * 3600 - 10
    fake.update({
        "f": {"id": "f", "state": "failed", "created": now - 5, "finished": now - 1},
        "inflight:F": "f",
        "q": {"id": "q", "state": "queued", "created": now - 5},
        "inflight:Q": "q",
        "inflight:X": "nobody",
        "inflight:N": 3.0,                              # garbage value
        "artifacts:Young": {"state": "pending", "t": now - 10, "job": "q"},
        "artifacts:Old": {"state": "pending", "t": now - 901, "job": "q"},
        "artifacts:Legacy": "junk",
        "cancel:young": now - 10,
        "cancel:old": now - 901,
        "cur": {"id": "cur", "state": "done", "created": old, "finished": old},
        "junk": "not a record",                         # purgeable garbage...
        "inflight:J": "junk",                           # ...and its marker
    })
    _bound(m, "cur")
    assert set(fake) == {"f", "q", "inflight:Q", "artifacts:Young", "cancel:young", "cur"}


def test_an_orphan_failed_in_the_sweep_loses_its_marker_in_the_same_sweep(monkeypatch,
                                                                          tmp_path):
    """The listing shows the orphan as queued - it was taken before the
    reconcile failed it - so the marker rule must count what the reconcile
    returned, or the marker outlives its flight to the next sweep."""
    fake = _CountingDict()
    m, _ = _sweep_env(monkeypatch, tmp_path, fake)
    monkeypatch.setattr(m, "_call_state", lambda cid: "dead")
    old = m.time.time() - 3600
    fake.update({"o": {"id": "o", "state": "queued", "created": old, "call_id": "x"},
                 "inflight:K": "o"})
    _bound(m, "cur")
    assert fake["o"]["state"] == "failed"
    assert "inflight:K" not in fake


def test_the_job_does_not_wait_for_the_sweep_and_sweeps_never_overlap(monkeypatch,
                                                                       tmp_path):
    """Observed 2026-09-19: the inline sweep took ~130 s per job at 1342 keys,
    under the volume lock the preview's `place` waits on ("preview 127s"), and
    `run_job` did not return until it was done. The job's own input goes
    inline; the sweep runs in a thread, and a second one never starts while a
    slow one (a deep queue's probes) is still running."""
    import threading
    fake = _CountingDict()
    m, _ = _sweep_env(monkeypatch, tmp_path, fake)
    clock = [1000.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "input_x.nii.gz").write_bytes(b"x")
    ran, release = [], threading.Event()

    def slow_sweep(jid, lock):
        ran.append(jid)
        release.wait(5)
    monkeypatch.setattr(m, "_sweep_jobs_store", slow_sweep)
    lock = threading.Lock()
    t = m._bound_jobs_store("j", lock)
    assert t is not None and t.is_alive()              # returned with the sweep running
    assert not (tmp_path / "j" / "input_x.nii.gz").exists()
    assert not lock.locked()                           # and nothing left holding the lock
    clock[0] += m.JOBS_SWEEP_EVERY_S + 1               # due again, but the first still runs
    assert m._bound_jobs_store("k", lock) is None
    release.set()
    t.join(5)
    assert ran == ["j"]
    assert m._bound_jobs_store("l", lock) is not None  # finished and due: the next may start
    m._sweep_thread.join(5)


def test_a_running_job_is_never_probed_as_an_orphan_by_its_own_container():
    """The container's own job is excluded by id, not by liveness, because
    `_call_state` for it would be a network round trip on every job."""
    import inspect
    from haversack import modal_app
    src = inspect.getsource(modal_app._reconcile_orphans)
    assert "k == current_jid" in src


def test_inflight_marker_ownership(monkeypatch):
    """Opus verification round: the marker operations, unit-reachable at
    module level. Under duplicate flights the marker names the latest job;
    installs never stomp, releases are compare-and-delete."""
    m, fake = _swap_dict(monkeypatch)
    m._install_inflight("K", "A")
    m._install_inflight("K", "B")          # refused: A owns
    assert fake["inflight:K"] == "A"
    m._release_inflight("K", "B")          # not the owner: no-op
    assert fake["inflight:K"] == "A"
    m._release_inflight("K", "A")
    assert "inflight:K" not in fake
    fake["inflight:K"] = "B"               # a NEWER submit installed directly
    m._release_inflight("K", "A")          # the older flight's finally
    assert fake["inflight:K"] == "B", "survivor's marker was clobbered"


def test_pending_marker_ownership(monkeypatch):
    m, fake = _swap_dict(monkeypatch)
    m._set_pending_marker("K", "A")
    m._set_pending_marker("K", "B")        # refuse-if-present: A keeps owning
    assert fake["artifacts:K"]["job"] == "A"
    m._clear_pending_marker("K", "B")      # not the owner: no-op
    assert "artifacts:K" in fake
    m._clear_pending_marker("K", "A")
    assert "artifacts:K" not in fake
    # legacy marker without a job field: unowned, clearable by anyone
    fake["artifacts:K"] = {"state": "pending", "t": 0}
    m._clear_pending_marker("K", "B")
    assert "artifacts:K" not in fake
    # the failure-path wrapper respects ownership too
    m._set_pending_marker("K", "A")
    m._clear_own_artifacts_marker("B", {"cache_key": "K"})
    assert "artifacts:K" in fake
    m._clear_own_artifacts_marker("A", {"cache_key": "K"})
    assert "artifacts:K" not in fake


def test_fresh_weights_versions_reloads_once(monkeypatch):
    """Opus verification round: the freshness reload converges after ONE
    volume reload and is throttled - 'unknown' is also the permanent honest
    answer for weights haversack did not install, and the unthrottled version
    reloaded a multi-GB volume on every HEAD probe forever."""
    import time as _t

    from haversack import modal_app

    class Vol:
        n = 0

        def reload(self):
            Vol.n += 1

    class Seg:
        def describe(self, task):
            if Vol.n:
                return {"weights_installed": [{"id": "297", "version": "v2"}]}
            return {"weights_installed": [{"id": "297"}]}   # -> unknown

        def engine_for(self, task):     # weights_versions_of reads the engine's cache_epoch
            from haversack.engines import registry
            return registry.engine_for_task(str(task))

    monkeypatch.setattr(modal_app, "weights_vol", Vol())
    ex = modal_app.ModalExecutor()
    ex.segmenter = Seg()
    type(ex)._weights_reloaded_at = 0.0
    type(ex)._wv_cache = {}
    wv = ex._fresh_weights_versions("t")
    assert Vol.n == 1 and not any("unknown" in v for v in wv), wv
    # within the cache window the segmenter is not even consulted (the
    # listing derives per entry - this is what keeps it cheap)
    class Boom(Seg):
        def describe(self, task):
            raise AssertionError("described during the cache window")
    ex.segmenter = Boom()
    assert ex._fresh_weights_versions("t") == wv
    # cache aged out but reload throttled: stays unknown without reloading
    class Never(Seg):
        def describe(self, task):
            return {"weights_installed": [{"id": "297"}]}
    ex.segmenter = Never()
    type(ex)._wv_cache = {}
    wv = ex._fresh_weights_versions("t")
    assert Vol.n == 1                       # throttled
    type(ex)._weights_reloaded_at = _t.time() - 60
    type(ex)._wv_cache = {}
    ex._fresh_weights_versions("t")
    assert Vol.n == 2                       # window elapsed: one more


@pytest.mark.parametrize("engine, task", [("fastsurfer", "fastsurfer:asegdkt"),
                                          ("synthstrip", "synthstrip:mask")])
def test_spawn_worker_rejects_an_engine_this_deployment_does_not_run(engine, task, monkeypatch):
    """Dispatch: an engine task on a deployment without that engine enabled fails
    loudly rather than routing to a worker that was never registered.

    The enable state was a module-level flag per engine, patched here by name. It is
    read from the registry now - `_worker_classes()` filters on `enabled()` each call -
    so the switch this patches is the same one an operator sets, not a mirror of it."""
    from haversack import modal_app
    monkeypatch.setenv(f"HAVERSACK_{engine.upper()}", "0")
    with pytest.raises(RuntimeError, match=f"{engine} engine is not enabled"):
        modal_app._spawn_worker(task, "j1")


def test_execute_job_and_hooks_exist():
    """The engine seam: a shared _execute_job + per-worker _ensure/_compute
    hooks (so adding an engine is a compute, not a run_job copy)."""
    from haversack import modal_app
    assert callable(modal_app._execute_job) and callable(modal_app._spawn_worker)
    for h in ("_ensure", "_compute", "_prepare"):
        assert callable(getattr(modal_app.Worker, h))
    import inspect
    assert "progress" in inspect.signature(modal_app.Worker._prepare).parameters  # the job's reporter


def test_all_engine_workers_build_with_the_shared_base():
    """The three @app.cls workers inherit setup/_artifact_worker/run_job from
    _WorkerBase, so Modal must collect those across the MRO. CI never sets the
    engine env vars and never constructs a worker, so this runs the decorated
    class bodies in a subprocess with both engines enabled - the only local
    coverage that the inheritance actually holds together."""
    import subprocess
    import sys
    code = (
        "import haversack.modal_app as M\n"
        "ws = M._worker_classes()\n"
        "assert sorted(ws) == ['fastsurfer', 'nnunetv2', 'synthstrip'], sorted(ws)\n"
        # every worker exposes the full hook set, inherited or not
        "for name, cls in ws.items():\n"
        "    for m in ('setup', 'preload', 'run_job', '_compute', '_prepare', '_ensure'):\n"
        "        assert hasattr(cls, m), (name, m)\n"
        # the engine shim reports the registry's identity, which is what keys the cache
        "from haversack.engines import registry as R\n"
        "for e in ('fastsurfer', 'synthstrip'):\n"
        "    got = M._EngineShim(e).describe('x')['weights_installed']\n"
        "    assert got == R.ENGINES[e].weights_identity(), (e, got)\n"
    )
    env = {"HAVERSACK_FASTSURFER": "1", "HAVERSACK_SYNTHSTRIP": "1", "HAVERSACK_PROXY_AUTH": "0"}
    import os
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, **env})
    assert r.returncode == 0, r.stderr[-1500:]


def test_spawn_worker_routes_by_engine_not_by_task_prefix():
    """Dispatch is a registry lookup: an nnU-Net ecosystem (any of ts/moose/custom)
    routes to the default engine without naming any of them."""
    from haversack import modal_app
    from haversack.engines import registry as R
    assert R.engine_for_task("ts.v2:total_fast").name == R.NNUNETV2
    assert R.engine_for_task("custom:mine").name == R.NNUNETV2
    assert R.engine_for_task("fastsurfer:asegdkt").name == "fastsurfer"
    assert modal_app.ENGINE_WORKERS.keys() <= R.ENGINES.keys()
    # every engine this deployment ENABLES has a composed worker; one that is off has
    # no adapter imported at all, which is the point - its image was never built either
    assert set(modal_app._worker_classes()) == {n for n in R.ENGINES if R.enabled(n)}


def test_the_two_executors_speak_the_same_submit_signature():
    """One protocol, two implementations - and the wire calls whichever it has.
    A parameter added to one and not the other is a 500 that only appears on the
    deployed side, which is exactly how `inputs` first shipped broken."""
    import inspect

    from haversack.modal_app import ModalExecutor
    from haversack.serve import LocalExecutor
    local = inspect.signature(LocalExecutor.submit).parameters
    modal_ = inspect.signature(ModalExecutor.submit).parameters
    assert set(local) - {"self"} <= set(modal_), \
        f"ModalExecutor.submit is missing {sorted(set(local) - set(modal_))}"


def test_the_engine_shim_reports_the_weights_identity_the_api_reports(monkeypatch):
    """The API keys a job at submit; the WORKER re-keys it at publish. If those
    two disagree the finished result lands in a slot nothing looks up, and every
    job of that engine misses its cache forever - silently, because the job
    itself succeeds.

    That is exactly what MONAI did: its identity is per BUNDLE rather than a
    per-engine constant, and the shim answered [] for engines with no constant,
    which weights_versions_of turns into "unknown".
    """
    for var in ("HAVERSACK_FASTSURFER", "HAVERSACK_SYNTHSTRIP", "HAVERSACK_VOXTELL", "HAVERSACK_MONAI"):
        monkeypatch.setenv(var, "1")
    from haversack import Segmenter
    from haversack.ecosystems import default_ecosystems
    from haversack.modal_app import _EngineShim
    from haversack.serve import weights_versions_of

    seg = Segmenter()
    checked = 0
    for eco in default_ecosystems():
        if eco.engine == "nnunetv2":
            continue                       # keyed by the real Segmenter, not a shim
        for task in eco.tasks()[:2]:
            name = f"{eco.name}:{task}"
            worker = weights_versions_of(_EngineShim(eco.engine), name)
            api = weights_versions_of(seg, name)
            assert worker == api, f"{name}: worker {worker} != api {api}"
            assert worker != ["unknown"], f"{name}: no weights identity at all"
            checked += 1
    assert checked, "no engine ecosystems were checked"


def test_every_import_time_knob_reaches_the_container():
    """A Modal container re-imports modal_app, and a HAVERSACK_* variable exists there only
    if the image env forwards it (_RUNTIME_KNOBS). One that is not silently takes its
    default in the container. HAVERSACK_APP_NAME was not (2026-09-12): a deploy with
    --app-name had its worker commit the default app's scratch volume, which was not
    mounted, and write every job record into the DEFAULT app's jobs Dict. Read the
    variables the module actually reads, rather than keeping a second list of them."""
    import ast
    from pathlib import Path

    from haversack import modal_app

    tree = ast.parse(Path(modal_app.__file__).read_text(encoding="utf-8"))
    read = set()
    for stmt in tree.body:                          # module level: what an import runs
        for node in ast.walk(stmt):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                            # (their bodies run later, not at import)
            name = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and ast.unparse(node.func.value) == "os.environ"):
                name = node.args[0].value
            elif (isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ"
                    and isinstance(node.slice, ast.Constant)):
                name = node.slice.value
            if isinstance(name, str) and name.startswith("HAVERSACK_"):
                read.add(name)
    assert "HAVERSACK_APP_NAME" in read, "the scan no longer sees the module's env reads"
    missing = sorted(read - set(modal_app._RUNTIME_KNOBS))
    assert not missing, f"read at import but never forwarded into the container: {missing}"


def test_volume_attach_preflight_fails_with_the_remedy(monkeypatch, tmp_path):
    """A deleted-under-snapshot volume once failed deep in a job with a cryptic
    'volume vo-... not attached' (2026-09-03); the preflight raises at startup with the fix."""
    import haversack.modal_app as m
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "ok"))
    (tmp_path / "ok").mkdir()
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "ok"))
    monkeypatch.setattr(m, "INPUTS_ROOT", str(tmp_path / "ok"))
    monkeypatch.setattr(m, "WEIGHTS_ROOT", str(tmp_path / "ok"))
    m._check_volumes_attached()                               # all writable: no error
    monkeypatch.setattr(m, "SCRATCH_ROOT", "/no/such/volume/path")
    import pytest
    with pytest.raises(RuntimeError, match="HAVERSACK_SNAPSHOT=0"):
        m._check_volumes_attached()


def test_a_disabled_engine_costs_no_image_build():
    """The engine images were module-level expressions, so Modal built every one on
    every deploy. With HAVERSACK_FASTSURFER unset, two deploys still ran the
    FastSurfer image's checkpoint fetch and both died on a Zenodo 504 (2026-09-07) -
    one engine's upstream having a bad day stopped a deploy that did not use it.

    The property, checked where it can fail: the only modal Images built at MODULE
    scope are the two every deployment needs. An engine's image is built inside the
    `if <ENGINE>:` that defines its worker.
    """
    import ast
    import inspect

    from haversack import modal_app

    def builds_an_image(node) -> bool:
        return any(isinstance(c, ast.Call) and "modal.Image" in ast.unparse(c.func)
                   for c in ast.walk(node))

    tree = ast.parse(inspect.getsource(modal_app))
    at_module_scope = {t.id for node in tree.body if isinstance(node, ast.Assign)
                       and builds_an_image(node)
                       for t in node.targets if isinstance(t, ast.Name)}
    assert at_module_scope == {"image", "api_image"}, (
        f"an engine image is built at module scope and will be built on every "
        f"deploy: {sorted(at_module_scope - {'image', 'api_image'})}")
    # ...and each engine's builder now lives in that engine's own adapter module,
    # which modal_app imports only when the engine is enabled - so the image is built
    # exactly when a worker for it is being registered, and never otherwise
    from haversack.engines import registry as R
    adapters = pathlib.Path(modal_app.__file__).parent / "engines"
    for name in R.ENGINES:
        if name == R.NNUNETV2:
            continue                       # its image IS the base image
        mod = adapters / f"modal_{name}.py"
        assert mod.exists(), f"{name} has no Modal adapter"
        atree = ast.parse(mod.read_text(encoding="utf-8"))
        built_at_scope = [n for n in atree.body if isinstance(n, ast.Assign)
                          and builds_an_image(n)]
        assert built_at_scope == [], (
            f"modal_{name}.py builds an image at module scope, so importing the adapter "
            "builds it whether or not this deployment runs the engine")
        assert any(isinstance(n, ast.FunctionDef) and builds_an_image(n)
                   for n in atree.body), f"modal_{name}.py builds no image at all"


def test_the_idc_cloud_knob_is_forwarded_to_the_container():
    """A worker in Google Cloud reads HAVERSACK_IDC_CLOUD at fetch time; a deploy
    shell variable does not exist in the container unless it is forwarded."""
    from haversack import modal_app
    assert "HAVERSACK_IDC_CLOUD" in modal_app._RUNTIME_KNOBS


def test_the_transpose_knob_is_forwarded_to_the_container():
    """The Worker reads HAVERSACK_ALLOW_TRANSPOSE at construction, but the
    container gets only what _RUNTIME_KNOBS forwards at deploy. Unforwarded, the
    three tasks whose plans permute the axes are listed, described, accepted, and
    then refused inside the GPU worker - the exact unreachability the flag ends."""
    from haversack import modal_app
    assert "HAVERSACK_ALLOW_TRANSPOSE" in modal_app._RUNTIME_KNOBS
    # every env var the worker reads at runtime must be in the forwarded set,
    # or the container's copy is simply unset
    src = pathlib.Path(modal_app.__file__).read_text(encoding="utf-8")
    worker = src.split("class Worker")[1].split("\nclass ")[0]
    read_at_runtime = set(re.findall(r'os\.environ(?:\.get)?[(\[]\s*"(HAVERSACK_[A-Z_]+)"', worker))
    missing = read_at_runtime - set(modal_app._RUNTIME_KNOBS)
    assert not missing, f"read by the Worker but never forwarded: {sorted(missing)}"


def test_a_skipped_input_refresh_reaches_a_modal_caller():
    """The worker records that a requested no-cache refresh could not happen; a
    fixed key whitelist then dropped it, so a caller got a result computed from
    bytes it asked not to reuse with no indication. The local executor reports
    it, so this deployment must too."""
    from haversack import modal_app
    src = pathlib.Path(modal_app.__file__).read_text(encoding="utf-8")
    keys = src.split("keys = (")[1].split(")")[0]
    assert '"input_refresh_skipped"' in keys


def test_the_modal_submit_carries_the_pin_and_the_worker_runs_it(monkeypatch, tmp_path):
    """The Modal half of server-side pins (2026-09-12): the job record carries the caller's
    version beside the canonical task, and the worker's weights step and compute both run
    run_name(task, version), so its catalog installs that version or refuses it.

    The worker's compute is driven for real through its plain class. _execute_job runs only
    inside a Modal container, so its weights step is checked in the PARSED source - calls,
    not text: the first version grepped for a string, and a review showed it passing with the
    call reverted and the string left in a comment."""
    import inspect
    import types
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(m, "_spawn_worker",
                        lambda task, jid, tokens=None: types.SimpleNamespace(object_id="fc-1"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: None)
    ex = m.ModalExecutor()
    ex.submit("j1", tmp_path, None, "ts.v2:total_fast", {}, version="v9")
    ex.submit("j2", tmp_path, None, "ts.v2:total_fast", {})
    assert fake["j1"]["task"] == "ts.v2:total_fast" and fake["j1"]["version"] == "v9"
    assert "version" not in fake["j2"]

    assert ex.status_of("j1")["version"] == "v9"      # reported, as the local executor does
    assert "version" not in ex.status_of("j2")

    W = m.Worker._get_user_cls()                      # the nnU-Net worker, plain Python
    w = W.__new__(W)
    seen = []
    w.seg = types.SimpleNamespace(segment=lambda image, task, **kw: seen.append(task))
    W._compute(w, "in.nii.gz", {"task": "ts.v2:total_fast", "version": "v9"}, None, None)
    W._compute(w, "in.nii.gz", {"task": "ts.v2:total_fast"}, None, None)
    assert seen == ["ts.v2:total_fast@v9", "ts.v2:total_fast"]

    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(m._execute_job)))
    ensure_args = [c.args[0] for c in ast.walk(tree)
                   if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "_ensure" and c.args]
    assert ensure_args, "_execute_job no longer calls ctx._ensure"
    for a in ensure_args:
        assert isinstance(a, ast.Call) and getattr(a.func, "id", None) == "run_name", \
            f"ctx._ensure is handed {ast.unparse(a)}, not run_name(...)"


def test_split_run_name_is_run_name_inverted_as_the_catalog_parses():
    from haversack.serve import run_name, split_run_name
    for task, version in [("monai:spleen_ct_segmentation", "0.6.1"), ("ts.v2:total", None),
                          ("fastsurfer:asegdkt", "2.5.4")]:
        assert split_run_name(run_name(task, version)) == (task, version)
    assert split_run_name("a@b@c") == ("a@b", "c")     # the LAST @, as EcosystemCatalog.resolve
    assert split_run_name("a@") == ("a@", None)        # no version: nothing to hand on


def test_an_unverified_pin_leaves_no_modal_inflight_marker(monkeypatch, tmp_path):
    """Review 2026-09-12: the Modal twin of the local join - an unverifiable pin's job took
    the `inflight:<key>` marker for the unpinned key, and the path surface's find_inflight
    then handed its flight to plain asks."""
    import types
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(m, "_spawn_worker",
                        lambda task, jid, tokens=None: types.SimpleNamespace(object_id="fc-1"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: None)
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: ["w=unknown"])
    monkeypatch.setattr(ex, "cache_get", lambda key: None)
    ex.submit("j1", tmp_path, None, "ts.v2:total_fast", {}, identity=("idc:x",),
              no_cache=True, version="v9")
    assert not [k for k, v in fake.items() if k.startswith("inflight:") and v == "j1"]
    ex.submit("j2", tmp_path, None, "ts.v2:total_fast", {}, identity=("idc:x",))
    assert [k for k, v in fake.items() if k.startswith("inflight:") and v == "j2"]


def test_the_pin_check_reloads_a_stale_weights_volume_once(monkeypatch):
    """Review 2026-09-12: the API container's weights volume is frozen at start, and the pin
    check read it without the reload the key's freshness gets - so a task a worker installed
    since read as unknown, and every pinned read was refused. Same throttle as the key."""
    from haversack import modal_app

    class Vol:
        n = 0

        def reload(self):
            Vol.n += 1

    class Seg:
        def describe(self, task):
            if Vol.n:
                return {"weights_installed": [{"id": "297", "version": "v2"}]}
            return {"weights_installed": [{"id": "297"}]}

    monkeypatch.setattr(modal_app, "weights_vol", Vol())
    ex = modal_app.ModalExecutor()
    ex.segmenter = Seg()
    monkeypatch.setattr(type(ex), "_weights_reloaded_at", 0.0)
    assert ex.installed_versions("t") == ["v2"] and Vol.n == 1
    Vol.n = 0                                 # stale again, but inside the throttle window
    assert ex.installed_versions("t") is None and Vol.n == 0


def test_an_image_baked_worker_refuses_a_pin_its_own_build_does_not_run(monkeypatch):
    """Review 2026-09-12: the API checks a pin against ITS registry, and a warm worker from an
    older deploy is not preempted - so the worker checks against its own build too."""
    from haversack import modal_app
    from haversack.engines import registry
    from haversack.errors import ModelNotFound
    monkeypatch.setenv("HAVERSACK_FASTSURFER", "1")
    have = registry.ENGINES["fastsurfer"].weights_identity()[0]["version"]
    ensure = modal_app._WorkerBase._ensure
    with pytest.raises(ModelNotFound, match=f"this build runs fastsurfer {have}"):
        ensure(object(), "fastsurfer:asegdkt@0.0.0")
    ensure(object(), f"fastsurfer:asegdkt@{have}")
    ensure(object(), "fastsurfer:asegdkt")          # unpinned: nothing to check


def test_the_monai_adapter_hands_the_pin_to_the_catalog():
    """Review 2026-09-12 (confirmed): `monai:<bundle>@<v>` reached MonaiEcosystem.ensure whole,
    as an unknown bundle, so every pinned MONAI job failed on Modal. Driven for real through
    the adapter's plain class, in a subprocess: importing an adapter registers a Modal class,
    which this test process must not carry. PYTHONPATH is prepended, never replaced (CI
    gets haversack only from it)."""
    import json
    import os
    import subprocess
    import sys
    code = "\n".join(['import json', 'from haversack import modal_app, ecosystems', 'from haversack.engines import modal_monai as mm', 'seen = []', 'ecosystems.MonaiEcosystem.ensure = (lambda self, bundle, root, progress=None, version=None:', '                                    seen.append([bundle, version]))', "mm.weights_vol = type('V', (), {'commit': lambda self: None})()", 'U = mm.MonaiWorker._get_user_cls()', 'w = U.__new__(U); w._ensured = set()', "U._prepare(w, 'monai:brats_mri_segmentation@0.5.4')", "U._prepare(w, 'monai:spleen_ct_segmentation')", 'print(json.dumps(seen))'])
    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "HAVERSACK_MONAI": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [
        ["brats_mri_segmentation", "0.5.4"], ["spleen_ct_segmentation", None]]


def test_a_pinned_ask_answered_from_the_modal_cache_reports_its_pin(monkeypatch, tmp_path):
    """Seen on Modal (2026-09-12): a verified pin served from the cache came back with no
    `version` - the cache-hit branch builds its own record, which left it out. The local
    executor reports it either way, so this deployment must too."""
    import types
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: ["297=v2.0.0-weights"])
    monkeypatch.setattr(ex, "cache_get", lambda key: ("/cache/labels.seg.nrrd", {"volumes_ml": {}}))
    ex.submit("j1", tmp_path, None, "ts.v2:total_fast", {}, identity=("idc:x",),
              version="v2.0.0-weights")
    ex.submit("j2", tmp_path, None, "ts.v2:total_fast", {}, identity=("idc:x",))
    assert fake["j1"]["cached"] is True and fake["j1"]["version"] == "v2.0.0-weights"
    assert ex.status_of("j1")["version"] == "v2.0.0-weights"
    assert "version" not in fake["j2"]


def test_a_submit_commits_the_scratch_volume_only_when_it_holds_an_upload(monkeypatch,
                                                                          tmp_path):
    """2026-09-19: every submit committed the scratch volume - 0.67 s of the 1.48 each
    POST /v1/jobs spent on Modal, paid by idc: jobs that wrote nothing there. An empty job
    directory is removed instead (the worker's save makes it); one holding an upload is
    committed so the worker sees it. The rest of the record is untouched: meta, the
    inflight marker, the spawn."""
    import types
    m, fake = _swap_dict(monkeypatch)
    commits = []
    monkeypatch.setattr(m, "scratch_vol",
                        types.SimpleNamespace(commit=lambda: commits.append(1)))
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None: (
        spawned.append(jid), types.SimpleNamespace(object_id="fc-" + jid))[1])
    monkeypatch.setattr(m, "_emit", lambda jid, d: fake[jid].update(d))
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: ["w=1"])
    monkeypatch.setattr(ex, "cache_get", lambda key: None)

    remote = tmp_path / "j1"
    remote.mkdir()
    ex.submit("j1", remote, None, "ts.v2:total_fast", {}, identity=("idc:x",),
              source=[{"kind": "idc", "id": "x"}])
    assert commits == [] and not remote.exists()
    key = fake["j1"]["cache_key"]
    assert fake[f"inflight:{key}"] == "j1" and fake["j1"]["call_id"] == "fc-j1"

    upload = tmp_path / "j2"
    upload.mkdir()
    (upload / "input_scan.nii.gz").write_bytes(b"bytes the worker must see")
    ex.submit("j2", upload, upload / "input_scan.nii.gz", "ts.v2:total_fast", {},
              identity=("sha256:ab",))
    assert commits == [1] and (upload / "input_scan.nii.gz").exists()
    assert spawned == ["j1", "j2"]


def test_a_flight_whose_call_is_gone_is_no_flight(monkeypatch):
    """The 1,206-job run (2026-09-19): records a stopped deployment left `running` kept
    their `inflight:` markers until some WORKER reconciled, so after a restart with no new
    jobs every plain GET of such a key joined a dead flight and waited 30 s. The API's
    lookup applies the reconcile's rule itself: an old record with a dead call is failed
    and no flight is returned; a live one is probed once per FLIGHT_LIVE_TTL_S; a young
    one (its call_id may not be written yet) is never probed."""
    import time as _time
    m, fake = _swap_dict(monkeypatch)
    probes = []
    states = {"dead": "dead", "live": "live"}
    monkeypatch.setattr(m, "_call_state", lambda cid: probes.append(cid) or states[cid])
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_flight_seen_live", {})
    old = _time.time() - 3600
    fake["orphan"] = {"id": "orphan", "state": "running", "created": old, "call_id": "dead"}
    fake["inflight:A"] = "orphan"
    fake["alive"] = {"id": "alive", "state": "running", "created": old, "call_id": "live"}
    fake["inflight:B"] = "alive"
    fake["young"] = {"id": "young", "state": "queued", "created": _time.time()}
    fake["inflight:C"] = "young"

    assert ex.find_inflight("A") is None
    assert fake["orphan"]["state"] == "failed" and "orphaned" in fake["orphan"]["error"]
    assert ex.find_inflight("A") is None and probes == ["dead"]   # failed: no second probe

    assert ex.find_inflight("B") == "alive" and ex.find_inflight("B") == "alive"
    assert probes == ["dead", "live"]                              # remembered

    assert ex.find_inflight("C") == "young" and probes == ["dead", "live"]


def test_a_plain_get_on_an_orphaned_flight_answers_at_once(monkeypatch, tmp_path):
    """End to end through the path route: 404 (not materialized), not a 30 s wait."""
    import time as _time
    import types
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from haversack.serve import create_app
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "cache"))
    vol = types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(m, "cache_vol", vol)
    monkeypatch.setattr(m, "_call_state", lambda cid: "dead")
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_flight_seen_live", {})
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: [])

    class Seg:
        def tasks(self):
            return ["total_fast"]

        def describe(self, task):
            return {"name": task}

        def engine_for(self, task):
            from haversack.engines import registry
            return registry.engine_for_task(str(task))

    ex.segmenter = Seg()
    client = TestClient(create_app(ex))
    series = "a05fb365-dfd2-4116-ab8e-a7262d2c169c"
    from haversack.serve import result_key
    key = result_key((f"idc:{series}",), "total_fast", {}, [])
    fake["j"] = {"id": "j", "state": "running", "created": _time.time() - 3600,
                 "call_id": "fc-gone", "cache_key": key}
    fake[f"inflight:{key}"] = "j"
    t0 = _time.monotonic()
    r = client.get(f"/v1/idc/{series}/total_fast/labels.seg.nrrd")
    assert _time.monotonic() - t0 < 5, "joined a flight that will never land"
    assert r.status_code == 404, r.text
    assert fake["j"]["state"] == "failed"


# -- review round 2026-09-19: each finding pinned ------------------------------------

def test_only_an_answer_that_the_call_ended_counts_as_dead(monkeypatch):
    """Every exception used to count as dead: one dropped connection on a read of a
    key failed a running job, and anonymous traffic could trigger it."""
    import modal
    from haversack import modal_app as m

    def probe(exc):
        class FC:
            def get(self, timeout=None):
                raise exc
        monkeypatch.setattr(modal.FunctionCall, "from_id", lambda cid: FC())
        return m._call_state("fc-1")
    assert probe(TimeoutError()) == "live"
    assert probe(ConnectionError("reset")) == "unknown"
    assert probe(modal.exception.ConnectionError("x")) == "unknown"
    assert probe(modal.exception.NotFoundError("x")) == "dead"
    assert probe(modal.exception.RemoteError("x")) == "dead"
    # round 2: a crashed container is InternalFailure, and a container that failed
    # before our code ran re-raises its own exception - both ended calls, never
    # "unknown", which would keep the key's flight alive forever
    assert probe(modal.exception.InternalFailure("x")) == "dead"
    assert probe(RuntimeError("enter failed")) == "dead"
    assert m._call_state(None) == "dead"


def test_a_container_whose_running_input_was_cancelled_takes_no_more(monkeypatch):
    """A cancel raises InputCancellation wherever the job is - mid-import on 2026-09-19,
    leaving torch._dynamo half initialized for the next job in the same warm container."""
    import modal
    import modal.experimental
    from haversack import modal_app as m
    retired = []
    monkeypatch.setattr(modal.experimental, "stop_fetching_inputs",
                        lambda: retired.append(True))

    def cancelled(ctx, jid, tokens):
        raise modal.exception.InputCancellation("Input was cancelled by user")
    monkeypatch.setattr(m, "_execute_job", cancelled)
    W = m.Worker._get_user_cls()
    with pytest.raises(modal.exception.InputCancellation):
        W.run_job._get_raw_f()(W.__new__(W), "j")
    assert retired == [True]
    monkeypatch.setattr(m, "_execute_job", lambda ctx, jid, tokens: None)
    W.run_job._get_raw_f()(W.__new__(W), "j2")
    assert retired == [True]                   # an ordinary finish retires nothing
    # ended by the cooperative marker, with Modal's signal lost somewhere inside: the
    # smoke's case - still retired
    monkeypatch.setattr(m, "_execute_job", lambda ctx, jid, tokens: "cancelled")
    W.run_job._get_raw_f()(W.__new__(W), "j3")
    assert retired == [True, True]


def test_execute_job_reports_a_cooperative_cancel(monkeypatch, tmp_path):
    """run_job retires on this return value: a Cancelled raised mid-job gives it, and a
    job cancelled before it started does not (a batch cancel must not cost a cold start
    per job)."""
    import types
    from haversack.errors import Cancelled
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path))
    fake["c"] = {"id": "c", "state": "cancelled", "task": "ts.v2:total_fast"}
    assert m._execute_job(None, "c") is None
    fake["r"] = {"id": "r", "state": "queued", "task": "ts.v2:total_fast",
                 "source": [{"kind": "upload"}]}
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_bound_jobs_store", lambda *a, **k: None)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)

    def ensure(task):
        raise Cancelled("cancelled")
    ctx = types.SimpleNamespace(_ensure=ensure, series_cache=None, read_ahead=None,
                                _vol_lock=None, engine=None)
    assert m._execute_job(ctx, "r") == "cancelled"
    assert fake["r"]["state"] == "cancelled"


def test_an_unknown_call_state_never_fails_a_job(monkeypatch):
    import time as _time
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "_call_state", lambda cid: "unknown")
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_flight_seen_live", {})
    fake["j"] = {"id": "j", "state": "running", "created": _time.time() - 3600,
                 "call_id": "fc-1"}
    fake["inflight:K"] = "j"
    assert ex.find_inflight("K") == "j" and fake["j"]["state"] == "running"
    assert m._reconcile_orphans(None, _time.time()) == []


def test_a_job_that_finishes_during_the_probe_is_not_failed(monkeypatch):
    """The re-read before the failing emit: a job done between the probe and the
    write must stay done."""
    import time as _time
    m, fake = _swap_dict(monkeypatch)
    fake["j"] = {"id": "j", "state": "running", "created": _time.time() - 3600,
                 "call_id": "fc-1"}

    def finishes(cid):
        fake["j"] = dict(fake["j"], state="done")
        return "dead"
    monkeypatch.setattr(m, "_call_state", finishes)
    assert m._fail_if_orphaned("j", dict(fake["j"], state="running"), _time.time()) is False
    assert fake["j"]["state"] == "done"
    fake["j"] = {"id": "j", "state": "running", "created": _time.time() - 3600,
                 "call_id": "fc-1"}
    assert m._reconcile_orphans(None, _time.time()) == [] and fake["j"]["state"] == "done"


def test_a_young_flight_is_not_remembered_as_live(monkeypatch):
    """An unprobed young record memoized as live would hide its death for another
    FLIGHT_LIVE_TTL_S once it came of age - the 30 s wait this rule removes."""
    import time as _time
    m, fake = _swap_dict(monkeypatch)
    probes = []
    monkeypatch.setattr(m, "_call_state", lambda cid: probes.append(cid) or "dead")
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_flight_seen_live", {})
    t0 = _time.time()
    fake["j"] = {"id": "j", "state": "queued", "created": t0, "call_id": "fc-1"}
    fake["inflight:K"] = "j"
    assert ex.find_inflight("K") == "j" and probes == []
    monkeypatch.setattr(m.time, "time", lambda: t0 + m.ORPHAN_MIN_AGE_S + 1)
    assert ex.find_inflight("K") is None and probes == ["fc-1"]


def test_the_worker_records_its_own_call_and_a_prepare_records_its_spawn(monkeypatch, tmp_path):
    """Both writes of the record are read-modify-writes; a warm worker that read it
    before the API's call_id emit wrote it back without the id, and a prepare never
    had one - the orphan rule reads either as dead."""
    import ast
    import inspect
    import textwrap
    import types
    m, fake = _swap_dict(monkeypatch)
    tree = ast.parse(textwrap.dedent(inspect.getsource(m._execute_job)))
    running = [c for c in ast.walk(tree)
               if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_emit"
               and len(c.args) == 2 and isinstance(c.args[1], ast.Dict)
               and any(isinstance(v, ast.Constant) and v.value == "running"
                       for v in c.args[1].values)]
    assert running, "_execute_job no longer emits running"
    assert all("_own_call_id" in ast.unparse(c.args[1]) for c in running)
    assert callable(getattr(m, "_own_call_id", None))    # named, and still defined
    monkeypatch.setattr(m, "_spawn_worker",
                        lambda task, jid, tokens=None: types.SimpleNamespace(object_id="fc-9"))
    m.ModalExecutor().submit_prepare("p", tmp_path, "ts.v2:total_fast")
    assert fake["p"]["call_id"] == "fc-9"


def test_the_sweep_removes_directories_before_the_records_naming_them(monkeypatch, tmp_path):
    """A sweep waiting on the volume lock with its records already deleted, in a
    container then scaled down, leaked directories no record named."""
    import threading
    import time as _time
    import types
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None,
                                                                reload=lambda: None))
    monkeypatch.setattr(m, "_call_state", lambda cid: "live")
    old = _time.time() - 30 * 24 * 3600
    for j in ("P1", "P2"):
        fake[j] = {"id": j, "state": "done", "created": old, "finished": old}
        (tmp_path / j).mkdir()
    lock = threading.Lock()
    lock.acquire()                              # the next job's save holds it
    t = threading.Thread(target=m._sweep_jobs_store, args=("CUR", lock), daemon=True)
    t.start()
    t.join(0.5)
    assert t.is_alive()
    assert {"P1", "P2"} <= set(fake), "records deleted while their directories remain"
    lock.release()
    t.join(5)
    assert not {"P1", "P2"} & set(fake) and not list(tmp_path.iterdir())


def test_retiring_waits_for_an_earlier_jobs_artifact_overlap(monkeypatch):
    """A retired container exits right after its input; the overlap thread of the job
    before, a daemon, would die mid-way and leave its artifacts marker set."""
    import threading
    import time as _time
    import modal
    import modal.experimental
    from haversack import modal_app as m
    monkeypatch.setattr(modal.experimental, "stop_fetching_inputs", lambda: None)
    done = []
    t = threading.Thread(target=lambda: (_time.sleep(0.5), done.append(True)),
                         name="haversack-artifacts", daemon=True)
    t.start()
    m._retire_container("j")
    assert done == [True]


def test_a_job_cancelled_while_computing_is_not_published(monkeypatch, tmp_path):
    """The smoke of 2026-09-19: DELETE during model loading, no InputCancellation reached
    the worker, the token was never checked again, and the job saved, published and
    reported done. The worker asks for the cancel marker once more after compute."""
    import types
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path))
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "input_ct.nii.gz").write_bytes(b"x")
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None,
                                                                reload=lambda: None))
    monkeypatch.setattr(m, "_prefetch_next", lambda *a, **k: None)
    monkeypatch.setattr(m, "_bound_jobs_store", lambda *a, **k: None)
    monkeypatch.setattr(m, "_own_call_id", lambda: None)
    fake["r"] = {"id": "r", "state": "queued", "task": "ts.v2:total_fast",
                 "source": [{"kind": "upload"}]}
    saved = []

    def compute(input_path, meta, on_progress, token):
        fake["cancel:r"] = 1.0                    # the DELETE lands mid-compute
        fake["r"] = dict(fake["r"], state="cancelled")
        return types.SimpleNamespace(save=lambda p: saved.append(p))
    import threading
    ctx = types.SimpleNamespace(_ensure=lambda task: None, _compute=compute,
                                series_cache=None,
                                read_ahead=types.SimpleNamespace(pop=lambda key: None),
                                _vol_lock=threading.Lock(), engine=None)
    assert m._execute_job(ctx, "r") == "cancelled"
    assert saved == [] and fake["r"]["state"] == "cancelled"


def test_own_call_id_is_modals_current_call(monkeypatch):
    """The name alone in the running emit proved nothing (review round 2): the helper
    must return Modal's id, and None outside a call."""
    import modal
    from haversack import modal_app as m
    monkeypatch.setattr(modal, "current_function_call_id", lambda: "fc-7")
    assert m._own_call_id() == "fc-7"

    def outside():
        raise RuntimeError("not in a function call")
    monkeypatch.setattr(modal, "current_function_call_id", outside)
    assert m._own_call_id() is None
