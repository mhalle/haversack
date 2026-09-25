"""The per-request deliverables on Modal (2026-09-20): tests/test_deliverables.py's rules,
held where the two sides are different processes.

- The api container checks the list at the door and writes it on the job's record; the
  WORKER reads that record once, when the job starts, and renders what it names - so the
  list reaches the worker with no Dict read of its own, and never a scan ("Per-job work on
  Modal must not be O(jobs Dict)").
- The list never reaches ``result_key`` on this side either.
- A cache hit is answered in the api container and reaches no worker, so here it cannot
  render what its list names and the stored generation lacks: it SAYS so - and believes
  the absence only from a view newer than the pending marker's read, because the worker
  places, commits and only then clears that marker (the 2026-09-19 rule for every miss).
- ``_vol_lock``: the list changes what the artifact thread renders and nothing about
  where it touches the volume; test_worker_volume_view.py's race tests run over the same
  code and still hold it to that.

The doubles are the volume and Dict doubles of test_worker_volume_view.py,
test_stale_volume_view.py and test_modal_app.py.
"""
from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
pytest.importorskip("modal")

from haversack.serve import ResultCache  # noqa: E402

from test_modal_app import _CountingDict  # noqa: E402
from test_stale_volume_view import U, _commit_elsewhere, _modal  # noqa: E402
from test_worker_volume_view import _Ctx, _submit, worker  # noqa: E402,F401 - a fixture

IDC = [{"kind": "idc", "crdc_series_uuid": U}]
BOTH = {"preview", "statistics"}


# -- the worker ---------------------------------------------------------------------------

def _run(m, jobs, monkeypatch, jid, *, deliverables="absent", offered=BOTH):
    """One job through the real `_execute_job`, its overlap replaced by a recorder.
    Returns what the overlap was asked to render (None: it never ran), the pending
    marker as the overlap saw it, the pair loads, and every `artifacts:` key written."""
    from haversack import preview, serve
    seen = types.SimpleNamespace(artifacts=None, marker=None, pairs=0, markers=[])
    monkeypatch.setattr(m, "ARTIFACTS", set(offered))

    def pair(image, labels):
        seen.pairs += 1
        return ("pair",)

    def overlap(pair, task, artifacts, *, preview_out, statistics_out, place, finish):
        seen.artifacts = tuple(artifacts)
        seen.marker = jobs.get(f"artifacts:key-{jid}")
        finish([])
    monkeypatch.setattr(preview, "load_oriented_pair", pair)
    monkeypatch.setattr(serve, "artifact_overlap", overlap)
    real_set = m._set_pending_marker

    def set_marker(key, owner, names=None):
        seen.markers.append((key, names))
        return real_set(key, owner, names)
    monkeypatch.setattr(m, "_set_pending_marker", set_marker)
    class Ctx(_Ctx):                       # the double, with the worker's REAL artifact thread
        def _artifact_worker(self, *a, **k):
            return m._WorkerBase._artifact_worker(self, *a, **k)

    _submit(m, jobs, jid)
    if deliverables != "absent":
        jobs[jid] = {**jobs[jid], "deliverables": list(deliverables)}
    if hasattr(jobs, "rpcs"):              # count the JOB's Dict calls, not this setup's
        jobs.rpcs = dict.fromkeys(jobs.rpcs, 0)
    m._execute_job(Ctx(), jid)
    for t in threading.enumerate():
        if t.name == "haversack-artifacts":
            t.join(5)
    seen.rpcs = dict(getattr(jobs, "rpcs", {}))
    assert jobs[jid]["state"] == "done", jobs[jid].get("error")
    return seen


def test_the_worker_renders_the_jobs_own_list(worker, monkeypatch):
    m, jobs, scratch, cache = worker
    seen = _run(m, jobs, monkeypatch, "j1", deliverables=["statistics"])
    assert seen.artifacts == ("statistics",)
    assert seen.marker["names"] == ["statistics"] and seen.marker["job"] == "j1"
    assert "artifacts:key-j1" not in jobs                  # cleared by the overlap's finish


def test_an_empty_list_starts_no_thread_loads_no_pair_and_sets_no_marker(worker, monkeypatch):
    m, jobs, scratch, cache = worker
    seen = _run(m, jobs, monkeypatch, "j2", deliverables=[])
    assert seen.artifacts is None and seen.pairs == 0 and seen.markers == []
    assert ResultCache(m.CACHE_ROOT).get("key-j2") is not None      # published all the same


def test_a_record_with_no_list_gets_the_deployments_set(worker, monkeypatch):
    """A job the previous deploy's api queued, or one a path GET started."""
    m, jobs, scratch, cache = worker
    seen = _run(m, jobs, monkeypatch, "j3")
    assert seen.artifacts == ("preview", "statistics") and seen.pairs == 1


def test_the_worker_holds_the_list_to_what_it_renders(worker, monkeypatch):
    """The api checked the list against ITS setting; a worker still warm from another
    deploy may render less, and must not render what it was deployed not to."""
    m, jobs, scratch, cache = worker
    seen = _run(m, jobs, monkeypatch, "j4", deliverables=["preview", "statistics"],
                offered={"statistics"})
    assert seen.artifacts == ("statistics",)


def test_the_list_costs_the_worker_no_dict_read_of_its_own(worker, monkeypatch):
    """The list rides on the record the job already reads. A job with a list makes
    exactly the Dict calls a job without one makes; neither lists the Dict; and the job
    reads its OWN record once - at its start - apart from `_emit`'s read-modify-writes,
    which is where the list has to come from."""
    m, jobs, scratch, cache = worker
    counts, own = {}, {}
    real_emit = m._emit
    in_emit = threading.local()

    def emit(jid, update):
        in_emit.yes = True
        try:
            return real_emit(jid, update)
        finally:
            in_emit.yes = False
    monkeypatch.setattr(m, "_emit", emit)

    class Dict(_CountingDict):
        def get(self, k, default=None):
            if k == self.watch and not getattr(in_emit, "yes", False):
                self.own_reads += 1
            return super().get(k, default)

    for jid, asked in (("ja", "absent"), ("jb", ["preview", "statistics"])):
        fake = Dict()
        fake.watch, fake.own_reads = jid, 0
        monkeypatch.setattr(m, "jobs_dict", fake)
        seen = _run(m, fake, monkeypatch, jid, deliverables=asked)
        assert seen.artifacts == ("preview", "statistics")
        counts[jid], own[jid] = seen.rpcs, fake.own_reads
    assert counts["ja"] == counts["jb"], counts
    assert counts["jb"]["get"] > 0, "the double counted nothing"
    assert counts["jb"]["items"] == 0 and counts["jb"]["keys"] == 0
    assert own == {"ja": 1, "jb": 1}, own


# -- the api container: the door, the record, the key ---------------------------------------

def _api(monkeypatch, tmp_path, *, refusals=0, offered=BOTH):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=refusals)
    monkeypatch.setattr(m, "ARTIFACTS", set(offered))
    monkeypatch.setattr(type(ex), "artifacts", set(offered))
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None: (
        spawned.append(jid) or types.SimpleNamespace(object_id=f"fc-{jid}")))
    return m, jobs, vol, ex, client, spawned


def _post(client, deliverables="absent", expect=202):
    data = {"task": "total_fast", "source": json.dumps(IDC)}
    if deliverables != "absent":
        data["deliverables"] = json.dumps(deliverables)
    r = client.post("/v1/jobs", data=data)
    assert r.status_code == expect, r.text
    return r.json()


def test_the_list_is_on_the_record_the_worker_reads_and_never_in_the_key(monkeypatch,
                                                                         tmp_path):
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path)
    # one key three times: each job lands before the next ask, or the next JOINS it (a
    # submit joins a flight since 2026-09-25 - this test used to expect three computations)
    def landed(j):
        jobs[j["id"]] = {**jobs[j["id"]], "state": "done", "result": {}}
        return j
    a = landed(_post(client, ["statistics"]))
    b = landed(_post(client, []))
    c = landed(_post(client))
    assert jobs[a["id"]]["deliverables"] == ["statistics"]
    assert jobs[b["id"]]["deliverables"] == []
    assert jobs[c["id"]]["deliverables"] == ["preview", "statistics"]
    assert spawned == [a["id"], b["id"], c["id"]]
    keys = {jobs[j["id"]]["cache_key"] for j in (a, b, c)}
    assert len(keys) == 1 and None not in keys, keys
    assert all("deliverables" not in json.dumps(jobs[j["id"]]["options"]) for j in (a, b, c))
    # the status route reports it, and builds `links` from it
    s = client.get(f"/v1/jobs/{a['id']}").json()
    assert s["deliverables"] == ["statistics"]
    assert "statistics" in s["links"] and "preview" not in s["links"]


def test_the_door_refuses_what_this_deployment_does_not_render(monkeypatch, tmp_path):
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path, offered={"statistics"})
    d = _post(client, ["preview"], expect=422)["detail"]
    assert d["code"] == "deliverable_not_offered" and d["offered"] == ["statistics"]
    assert not jobs and not spawned


# -- a cache hit on Modal -----------------------------------------------------------------

def _hit(monkeypatch, tmp_path, *, refusals=0, place=()):
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path, refusals=refusals)
    key = ex.resource_key(f"idc:{U}", "total_fast", {})
    _commit_elsewhere(m, vol, tmp_path, key)
    for name in place:                                       # a worker rendered these
        g = next(p for p in (vol.hidden / key).iterdir() if p.name.startswith("g-"))
        (g / name).write_bytes(b"x")
    return m, jobs, vol, ex, client, spawned, key


def test_a_hit_that_holds_what_was_asked_for_says_nothing_more(monkeypatch, tmp_path):
    m, jobs, vol, ex, client, spawned, key = _hit(monkeypatch, tmp_path,
                                                  place=("statistics.json",))
    s = _post(client, ["statistics"])
    assert s["state"] == "done" and s["cached"] is True and not spawned
    assert s["deliverables"] == ["statistics"] and "deliverables_unavailable" not in s
    assert "statistics" in client.get(f"/v1/jobs/{s['id']}").json()["links"]


def test_a_hit_cannot_render_here_and_says_so(monkeypatch, tmp_path):
    """No worker is reached, so nothing is rendered: the job names what is missing and
    the way out, `links` leaves it out, and no GPU container is started for it."""
    m, jobs, vol, ex, client, spawned, key = _hit(monkeypatch, tmp_path,
                                                  place=("statistics.json",))
    s = _post(client, ["preview", "statistics"])
    assert s["state"] == "done" and s["cached"] is True and not spawned
    assert s["deliverables_unavailable"] == {"preview": m.DELIVERABLE_NEEDS_A_COMPUTE}
    assert "no-cache" in m.DELIVERABLE_NEEDS_A_COMPUTE
    links = client.get(f"/v1/jobs/{s['id']}").json()["links"]
    assert "statistics" in links and "preview" not in links


def test_a_hit_beside_a_running_render_waits_for_what_it_will_place(monkeypatch, tmp_path):
    """Asked right after `done`, while the computing worker's overlap still renders: what
    that render was asked for is on its way, and only the rest is named."""
    m, jobs, vol, ex, client, spawned, key = _hit(monkeypatch, tmp_path)
    jobs[f"artifacts:{key}"] = {"state": "pending", "t": time.time(), "job": "w",
                                "names": ["statistics"]}
    from haversack.jobpolicy import RENDER_BUSY
    s = _post(client, ["preview", "statistics"])
    assert s["deliverables_unavailable"] == {"preview": RENDER_BUSY}
    assert "statistics" in client.get(f"/v1/jobs/{s['id']}").json()["links"]
    jobs[f"artifacts:{key}"] = {"state": "pending", "t": time.time(), "job": "w"}
    assert "deliverables_unavailable" not in _post(client, ["preview"])   # a marker sans names


def test_an_absence_is_believed_only_from_a_view_newer_than_the_markers_read(monkeypatch,
                                                                             tmp_path):
    """The worker placed and committed the preview, then cleared its marker; this
    container's view predates the commit. The hit must not call it missing."""
    m, jobs, vol, ex, client, spawned, key = _hit(monkeypatch, tmp_path)
    assert _post(client, ["preview"])["deliverables_unavailable"]      # truly absent: said
    # now a worker places it and commits - elsewhere, not yet in this container's view
    g = next(p for p in (Path(m.CACHE_ROOT) / key).iterdir() if p.name.startswith("g-"))
    Path(m.CACHE_ROOT).rename(vol.hidden)
    (vol.hidden / key / g.name / "preview.png").write_bytes(b"\x89PNG")
    # ...while this container still reads its old copy of the generation
    monkeypatch.setattr(m, "CACHE_FRESH_S", 3600.0)
    Path(m.CACHE_ROOT).mkdir()
    import shutil
    shutil.copytree(vol.hidden / key, Path(m.CACHE_ROOT) / key)
    (Path(m.CACHE_ROOT) / key / g.name / "preview.png").unlink()
    s = _post(client, ["preview"])
    assert "deliverables_unavailable" not in s, s
    assert "preview" in client.get(f"/v1/jobs/{s['id']}").json()["links"]


def test_a_view_that_cannot_be_refreshed_is_not_an_absence(monkeypatch, tmp_path):
    m, jobs, vol, ex, client, spawned, key = _hit(monkeypatch, tmp_path)
    assert _post(client)["cached"] is True                   # in view now
    vol.refusals = 99                                        # every reload refused from here
    monkeypatch.setattr(m, "_cache_view_as_of", float("-inf"))
    s = _post(client, ["preview"])
    assert s["cached"] is True
    assert s["deliverables_unavailable"] == {"preview": m.DELIVERABLE_NOT_VISIBLE}


def test_the_pending_state_is_per_deliverable_here_too(monkeypatch, tmp_path):
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path)
    jobs["artifacts:K"] = {"state": "pending", "t": time.time(), "job": "w",
                           "names": ["statistics"]}
    assert ex.artifact_state("K") == "pending"
    assert ex.artifact_state("K", "statistics") == "pending"
    assert ex.artifact_state("K", "preview") == "absent"
    jobs["artifacts:K"] = {"state": "pending", "t": time.time() - 1000, "job": "w",
                           "names": ["statistics"]}
    assert ex.artifact_state("K", "statistics") == "absent" and "artifacts:K" not in jobs


def test_the_twins_read_of_a_dead_marker_writes_nothing(monkeypatch, tmp_path):
    """``sweep=False`` is the twin's promise to write nothing anywhere, housekeeping
    included; the api's read does sweep, or the writer's refuse-if-present rule would
    decline to mark a new flight. Neither half was observed until a mutation pass
    (2026-09-21) flipped each and nothing failed."""
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path)
    dead = {"state": "pending", "t": time.time() - 1000, "job": "w", "names": ["preview"]}
    jobs["artifacts:K"] = dict(dead)
    assert m._artifact_state("K", "preview", sweep=False) == "absent"
    assert jobs.get("artifacts:K") == dead, "the twin's read removed a marker"
    assert m._artifact_state("K", "preview", sweep=True) == "absent"
    assert "artifacts:K" not in jobs


def test_the_twin_is_handed_the_read_that_does_not_sweep():
    """...and ``public()`` is where that read is chosen: parsed, since the function runs
    only in a Modal container."""
    import ast
    import haversack.modal_app as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    sweeps = [kw.value.value for fn in ast.walk(tree)
              if isinstance(fn, ast.FunctionDef) and fn.name == "public"
              for call in ast.walk(fn) if isinstance(call, ast.Call)
              and getattr(call.func, "id", None) == "_artifact_state"
              for kw in call.keywords if kw.arg == "sweep"]
    assert sweeps == [False], sweeps


def test_the_worker_says_what_a_job_with_no_key_cannot_deliver():
    """``_execute_job`` runs only in a worker, so this reads it: it must ask
    ``unkeyed_deliverables`` and emit the answer, or a Modal job with no cache entry
    advertises artifact links no door serves (the local half is driven for real)."""
    import ast
    import haversack.modal_app as mod
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "_execute_job")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
    assert any(getattr(c.func, "id", getattr(c.func, "attr", None)) == "unkeyed_deliverables"
               for c in calls)
    assert any(isinstance(k, ast.Constant) and k.value == "deliverables_unavailable"
               for c in calls if getattr(c.func, "id", None) == "_emit"
               for a in c.args if isinstance(a, ast.Dict) for k in a.keys)

