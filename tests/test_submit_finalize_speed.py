"""Submit and finalize on Modal, without the work nobody reads (2026-09-25).

Profiled on a smoke deployment (`haversack-prof-smoke`, three fresh IDC CTs):

- a submit answered from the cache COPIED the result's generation into the api container
  (``_read_cache`` -> ``_mirror``) to read its record: six concurrent hit-submits waited
  1.0-4.0 s, each behind the others' copies under the one ``_mirror_lock``. It now reads the
  record and stats the artifacts under the view's lock (``_cache_record``): 0.27-0.6 s;
- the route read back from the Dict the record ``submit`` had just written (0.07-0.7 s);
- a worker committed the job's own scratch copy - the api's fallback, read only once the
  entry is evicted or republished - BEFORE ``done``: 0.7-1.5 s of every job's latency;
- the ``.seg.nrrd`` header sorted the whole label map (``np.unique``) to learn which labels
  ``find_objects`` had just reported: 1.0 s of a 418 M-voxel save.
"""
from __future__ import annotations

import threading
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
pytest.importorskip("modal")

from haversack.serve import RESULT_NAME, ResultCache  # noqa: E402

from test_stale_volume_view import U, _commit_elsewhere, _key, _modal  # noqa: E402
from test_worker_volume_view import _Ctx, _submit, worker  # noqa: E402,F401 - a fixture

IDC = '[{"kind": "idc", "crdc_series_uuid": "%s"}]' % U


class _CountingJobs(dict):
    def __init__(self):
        super().__init__()
        self.gets = []

    def get(self, k, default=None):
        self.gets.append(k)
        return super().get(k, default)


def _hit(monkeypatch, tmp_path, *, preview: bool):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    counting = _CountingJobs()
    monkeypatch.setattr(m, "jobs_dict", counting)
    monkeypatch.setattr(m, "ARTIFACTS", {"preview"})
    key = _key(ex)
    _commit_elsewhere(m, vol, tmp_path, key)
    if preview:                                   # rendered into the committed generation
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        committed = ResultCache(str(vol.hidden))
        assert committed.add_artifact(key, "preview.png", png)
    r = client.post("/v1/jobs", data={"task": "total_fast", "source": IDC,
                                      "deliverables": '["preview"]'})
    assert r.status_code == 202, r.text
    return m, counting, key, r.json()


def test_a_submit_answered_from_the_cache_copies_nothing(monkeypatch, tmp_path):
    m, jobs, key, body = _hit(monkeypatch, tmp_path, preview=True)
    assert body["state"] == "done" and body["cached"] is True
    assert not list(Path(m.MIRROR_ROOT).rglob(RESULT_NAME)), "the labels were copied"
    rec = jobs[body["id"]]
    assert rec["cache_path"].startswith(m.CACHE_ROOT + "/" + key + "/")   # the VOLUME's
    assert rec["cache_path"].endswith("/" + RESULT_NAME)
    # the artifact's presence was read off the volume: nothing is said to be unavailable
    assert "deliverables_unavailable" not in rec


def test_a_hit_that_lacks_an_artifact_still_says_so(monkeypatch, tmp_path):
    m, jobs, key, body = _hit(monkeypatch, tmp_path, preview=False)
    assert body["state"] == "done"
    assert set(jobs[body["id"]]["deliverables_unavailable"]) == {"preview"}


def test_the_route_answers_a_submit_from_the_record_in_hand(monkeypatch, tmp_path):
    m, jobs, key, body = _hit(monkeypatch, tmp_path, preview=True)
    assert body["id"] not in jobs.gets, "the route read back the record submit wrote"
    import haversack.modal_app as mm
    assert body["state"] == mm.ModalExecutor().status_of(body["id"])["state"]


def test_status_of_record_says_what_a_status_says(monkeypatch):
    """Against a literal, not against status_of - which calls status_of_record (the first
    version of this test compared the function with itself)."""
    from haversack import modal_app as m
    rec = {"id": "j", "task": "t", "state": "done", "result": {"a": 1}, "kind": "rankfield",
           "cache_key": "k", "version": "v1", "deliverables": ["preview"],
           "deliverables_unavailable": {"preview": "x"}, "secret_field": "x", "tokens_id": "y",
           "call_id": "fc", "source": [{"kind": "idc"}]}
    monkeypatch.setattr(m, "jobs_dict", {"j": rec})
    ex = m.ModalExecutor()
    assert ex.status_of_record(rec) == {
        "id": "j", "task": "t", "state": "done", "result": {"a": 1}, "kind": "rankfield",
        "cache_key": "k", "version": "v1", "deliverables": ["preview"],
        "deliverables_unavailable": {"preview": "x"}}
    assert ex.status_of("j") == ex.status_of_record(rec)


# -- the worker: the job's own scratch copy -------------------------------------------------

def _state_at_scratch_commit(m, jobs, scratch, jid, *, keyed: bool):
    seen = []
    real = scratch.commit

    def commit():
        copy = Path(m.SCRATCH_ROOT) / jid / RESULT_NAME
        seen.append((jobs[jid]["state"], copy.exists()))
        return real()
    scratch.commit = commit
    _submit(m, jobs, jid)
    if not keyed:
        jobs[jid] = {**jobs[jid], "cache_key": None}
    m._execute_job(_Ctx(), jid)
    for t in threading.enumerate():
        if t.name == "haversack-artifacts":
            t.join(5)
    assert jobs[jid]["state"] == "done", jobs[jid].get("error")
    assert (Path(m.SCRATCH_ROOT) / jid / RESULT_NAME).exists()
    return seen


def test_a_keyed_job_places_its_scratch_copy_after_done(worker):
    m, jobs, scratch, cache = worker
    seen = _state_at_scratch_commit(m, jobs, scratch, "k1", keyed=True)
    copies = [s for s in seen if s[1]]
    assert copies and copies[0][0] == "done", seen
    assert ResultCache(m.CACHE_ROOT).get("key-k1") is not None


def test_a_job_with_no_key_places_its_copy_before_done(worker):
    """No entry will exist: the job's own copy is the only one, and must be there first."""
    m, jobs, scratch, cache = worker
    seen = _state_at_scratch_commit(m, jobs, scratch, "u1", keyed=False)
    copies = [s for s in seen if s[1]]
    assert copies and copies[0][0] == "running", seen


# -- the seg.nrrd header ------------------------------------------------------------------

def _header(arr, monkeypatch=None, no_scipy=False):
    from haversack.result import Segmentation
    seg = types.SimpleNamespace(array=arr, provenance={},
                                schema=types.SimpleNamespace(names={2: "a", 5: "b", 3: "c"}))
    if no_scipy:
        import sys
        monkeypatch.setitem(sys.modules, "scipy", None)
    return Segmentation._seg_nrrd_metadata(seg)


def test_the_header_names_the_present_labels_without_sorting_the_volume(monkeypatch):
    arr = np.zeros((4, 5, 6), np.uint8)
    arr[1, 1:3, 2] = 2
    arr[3, 4, 5] = 5                                  # 3 is absent, 1 and 4 too
    with monkeypatch.context() as mp:
        def refuse(*a, **k):
            raise AssertionError("np.unique sorted the label map")
        mp.setattr(np, "unique", refuse)
        md = _header(arr)
    ids = sorted(v for k, v in md.items() if k.endswith("_ID"))
    assert ids == ["Segment_2", "Segment_5"]
    assert md["Segment0_Extent"] == "2 2 1 2 1 1" and md["Segment1_Extent"] == "5 5 4 4 3 3"
    # the same segments as the scipy-free path, which still asks np.unique
    plain = _header(arr, monkeypatch, no_scipy=True)
    strip = lambda d: {k: v for k, v in d.items() if not k.endswith("_Extent")}   # noqa: E731
    assert strip(md) == strip(plain)


def test_a_hit_submit_racing_reloads_never_misses(monkeypatch, tmp_path):
    """`_cache_record` reads under `_cache_view` shared: a reload in another thread hides the
    volume (measured on Modal, 2026-09-19), and a lookup without the lock missed and was believed
    - a hit answered as a miss. The test_stale_volume_view race, aimed at the submit's lookup."""
    import threading

    from haversack.serve import ResultCache
    from test_stale_volume_view import _HidingVolume, _NrrdSeg
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    src = tmp_path / "worker" / RESULT_NAME
    src.parent.mkdir(parents=True)
    _NrrdSeg().save(src)
    ResultCache(m.CACHE_ROOT).put("k" * 64, src, {"outputs": []}, {"task": "total_fast"})
    monkeypatch.setattr(m, "cache_vol", _HidingVolume(tmp_path / "cache"))
    monkeypatch.setattr(m, "CACHE_FRESH_S", 0.0)          # every lookup asks for a reload
    misses, errors = [], []

    def ask():
        for _ in range(40):
            try:
                if ex._cache_record("k" * 64) is None:
                    misses.append(1)
            except Exception as e:             # noqa: BLE001
                errors.append(e)
    threads = [threading.Thread(target=ask) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:3]
    assert not misses, f"{len(misses)} of 320 hit lookups missed"


def test_a_failed_scratch_copy_after_done_leaves_the_job_done(worker):
    """The copy lands after `done`; if it fails, the job is still done and its entry served -
    an exception there reached `_execute_job`'s handler with `done` already emitted."""
    m, jobs, scratch, cache = worker
    real = scratch.commit
    calls = []

    def commit():
        calls.append(jobs["kf"]["state"])
        if jobs["kf"]["state"] == "done":
            raise OSError("volume gone")
        return real()
    scratch.commit = commit
    _submit(m, jobs, "kf")
    m._execute_job(_Ctx(), "kf")
    for t in threading.enumerate():
        if t.name == "haversack-artifacts":
            t.join(5)
    assert "done" in calls and jobs["kf"]["state"] == "done", jobs["kf"].get("error")
    assert ResultCache(m.CACHE_ROOT).get("key-kf") is not None
