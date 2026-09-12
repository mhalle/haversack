"""The Modal deployment module is import-safe and shaped as create_app expects."""
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
        modal_app._emit("j1", {"state": "done", "finished": 3.0})
        assert fake["j1"]["state"] == "done"               # terminal->terminal ok
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
    m._bound_jobs_store("me")
    assert fake["stale"]["state"] == "failed"
    assert "inflight:K" not in fake


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
