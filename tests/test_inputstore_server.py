"""The local server with HAVERSACK_INPUT_STORE=blobs (step 6, docs/cache-consolidation.md).

The server's call sites are unchanged; `inputstore.ServerInputs` answers them. These drive a
real LocalExecutor through its routes and pin what the adapter must keep: an upload referred
to by digest, a fetched source fetched once and recorded, a refresh refused while the input is
in use, a running job's view outliving the eviction of its input, input_gone after eviction,
and no view left behind by a finished job.
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from haversack.serve import LocalExecutor, create_app

from test_serve import FakeSegmenter, submit, volume_bytes, wait_state

UUID = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"


@pytest.fixture(autouse=True)
def blobs(monkeypatch):
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")


def _server(tmp_path, monkeypatch, seg=None, **kw):
    from haversack import serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    calls = []

    def fake_fetch(series, jobdir):
        calls.append(series)
        d = jobdir / "series"
        d.mkdir()
        (d / "ct.nii.gz").write_bytes(volume_bytes(len(calls) + 100))
        # what sources.fetch_recording_origin writes beside a real fetch
        (jobdir / ".input.json").write_text(json.dumps(
            {"kind": "idc", "identity": f"idc:{series}", "origin": {"collection": "test"}}))
        return d
    seg = seg or FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, fetch_idc_fn=fake_fetch, **kw)
    return seg, ex, TestClient(create_app(ex)), calls


def _idc_job(client, headers=None, options=None):
    r = client.post("/v1/jobs", headers=headers or {}, data={
        "task": "total_fast", "options": json.dumps(options or {}),
        "source": json.dumps([{"kind": "idc", "crdc_series_uuid": UUID}])})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def test_the_flag_selects_the_blob_store(tmp_path, monkeypatch):
    from haversack.inputstore import ServerInputs
    _, ex, _, _ = _server(tmp_path, monkeypatch)
    assert isinstance(ex.series_cache, ServerInputs) and ex.content is ex.series_cache


def test_a_fetched_source_is_fetched_once_read_as_its_copy_and_recorded(tmp_path, monkeypatch):
    seg, ex, client, calls = _server(tmp_path, monkeypatch)
    s = wait_state(client, _idc_job(client), ("done",))
    assert s["input_identity"] == [f"idc:{UUID}"]
    assert seg.calls[0][0].endswith("decoded/input.duckn.zip")
    # the record the fetch wrote reached the result's provenance through the view
    assert ex.series_cache.store.record(f"idc:{UUID}")["origin"] == {"collection": "test"}
    assert s["result"]["provenance"]["inputs"][0]["origin"] == {"collection": "test"}
    # another result of the same series (another option, so another key) fetches nothing...
    s2 = wait_state(client, _idc_job(client, options={"interp": "nearest"}), ("done",))
    assert s2["state"] == "done" and len(calls) == 1
    # ...and no-cache refreshes the input too, as it always has
    wait_state(client, _idc_job(client, {"Cache-Control": "no-cache"}), ("done",))
    assert len(calls) == 2


def test_a_finished_job_leaves_no_view(tmp_path, monkeypatch):
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    wait_state(client, _idc_job(client), ("done",))
    views = ex.series_cache._views_root
    assert not views.exists() or not any(views.iterdir())


def test_an_upload_is_referred_to_by_digest(tmp_path, monkeypatch):
    import hashlib
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    raw = volume_bytes(7)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert client.get(f"/v1/inputs/{d}").status_code == 404
    assert client.put(f"/v1/inputs/{d}", content=raw).json()["stored"] is True
    info = client.get(f"/v1/inputs/{d}").json()
    assert info["stored_form"] == "input_copy"
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "input", "sha256": d}])})
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done",))
    assert s["input_identity"] == [d]


def test_an_evicted_upload_is_input_gone(tmp_path, monkeypatch):
    import hashlib
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    raw = volume_bytes(8)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    client.put(f"/v1/inputs/{d}", content=raw)
    ex.series_cache.store.budget = 0
    ex.series_cache.store.evict()
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "input", "sha256": d}])})
    assert r.status_code == 410 and r.json()["detail"]["code"] == "input_gone"


def test_a_running_jobs_input_outlives_its_eviction(tmp_path, monkeypatch):
    """The view is the job's: a budget of zero evicts the ref and sweeps the blobs while the
    job runs, and the job still reads its input to the end."""
    gate = threading.Event()
    seg = FakeSegmenter(gate=gate)
    read = []
    real = seg.segment

    def segment(image, task, **kw):
        from haversack import io
        out = real(image, task, **kw)                 # waits on the gate
        read.append(io.read_image(image).GetSize())    # reads the view AFTER the eviction
        return out
    seg.segment = segment
    _, ex, client, _ = _server(tmp_path, monkeypatch, seg=seg)
    ex.series_cache.store.grace_s = 0
    jid = _idc_job(client)
    # inside segment(), holding its view - "running" is set BEFORE the input is fetched, and
    # evicting then found nothing to evict (the first version of this test, which a symlink
    # mutant survived)
    t0 = time.time()
    while not seg.calls and time.time() - t0 < 5:
        time.sleep(0.01)
    assert ex.series_cache.store.has(f"idc:{UUID}")
    ex.series_cache.store.budget = 0
    ex.series_cache.store.evict()
    assert not ex.series_cache.store.has(f"idc:{UUID}")
    assert ex.series_cache.store.blobs.entries() == []
    gate.set()
    s = wait_state(client, jid, ("done", "failed"))
    assert s["state"] == "done", s
    assert read and read[0]


def test_a_refresh_is_refused_while_the_input_is_in_use(tmp_path, monkeypatch):
    """no-cache on the INPUT (refresh_input) while another job reads it: jobpolicy's rule,
    through the adapter's discard - refused, and said."""
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    wait_state(client, _idc_job(client), ("done",))
    ex.series_cache.pin(f"idc:{UUID}")
    try:
        assert ex.series_cache.discard(f"idc:{UUID}") is False
    finally:
        ex.series_cache.unpin(f"idc:{UUID}")
    assert ex.series_cache.discard(f"idc:{UUID}") is True
    assert not ex.series_cache.has(f"idc:{UUID}")


def test_a_crashed_processs_views_are_reaped(tmp_path, monkeypatch):
    from haversack.inputstore import ServerInputs
    dead = tmp_path / "input_store" / "views" / "999999"
    (dead / "x").mkdir(parents=True)
    ServerInputs(tmp_path / "input_store", None)
    assert not dead.exists()


def test_an_upload_the_legacy_cache_holds_is_adopted_on_first_use(tmp_path, monkeypatch):
    """The migration shim: an upload stored before the flag cannot be fetched again, so it is
    carried into the new store the first time it is asked for - its input copy as it is - and
    a job on it runs. A fetched input is NOT adopted (it is fetched again under the new key)."""
    import hashlib
    raw = volume_bytes(9)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    monkeypatch.delenv("HAVERSACK_INPUT_STORE")
    legacy = LocalExecutor(FakeSegmenter(), workdir=tmp_path)             # the legacy store
    TestClient(create_app(legacy)).put(f"/v1/inputs/{d}", content=raw)
    assert legacy.content.has(d)
    legacy_copy = legacy.content.resolve(d)
    legacy.close()
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    assert not ex.series_cache.store.has(d)
    info = client.get(f"/v1/inputs/{d}")
    assert info.status_code == 200 and info.json()["stored_form"] == "input_copy"
    assert ex.series_cache.store.has(d)
    # the copy was carried byte for byte, not re-encoded
    blob = next(iter(ex.series_cache.store.ref(d)["files"].values()))
    assert blob["digest"] == "sha256:" + hashlib.sha256(legacy_copy.read_bytes()).hexdigest()
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "input", "sha256": d}])})
    assert r.status_code == 202, r.text
    assert wait_state(client, r.json()["id"], ("done",))["input_identity"] == [d]


def test_a_legacy_entry_mid_write_is_not_adopted(tmp_path, monkeypatch):
    import hashlib
    raw = volume_bytes(10)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    monkeypatch.delenv("HAVERSACK_INPUT_STORE")
    legacy = LocalExecutor(FakeSegmenter(), workdir=tmp_path)
    TestClient(create_app(legacy)).put(f"/v1/inputs/{d}", content=raw)
    entry = legacy.series_cache.entry(d)
    legacy.close()
    (entry / legacy.series_cache.MARKER).unlink()                          # uncommitted
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    assert client.get(f"/v1/inputs/{d}").status_code == 404
    assert not ex.series_cache.store.has(d)


def test_a_fetched_input_the_legacy_cache_holds_is_fetched_again(tmp_path, monkeypatch):
    monkeypatch.delenv("HAVERSACK_INPUT_STORE")
    _, legacy, lclient, lcalls = _server(tmp_path, monkeypatch)
    wait_state(lclient, _idc_job(lclient), ("done",))
    assert legacy.series_cache.has(f"idc:{UUID}") and len(lcalls) == 1
    legacy.close()
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    _, ex, client, calls = _server(tmp_path, monkeypatch)
    assert not ex.series_cache.has(f"idc:{UUID}")
    wait_state(client, _idc_job(client, options={"interp": "nearest"}), ("done",))
    assert len(calls) == 1


def test_resolve_alone_adopts_a_legacy_upload(tmp_path, monkeypatch):
    """Every route asks has() first today; resolve() must not depend on that order."""
    import hashlib
    raw = volume_bytes(11)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    monkeypatch.delenv("HAVERSACK_INPUT_STORE")
    legacy = LocalExecutor(FakeSegmenter(), workdir=tmp_path)
    TestClient(create_app(legacy)).put(f"/v1/inputs/{d}", content=raw)
    legacy.close()
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    _, ex, _, _ = _server(tmp_path, monkeypatch)
    assert ex.series_cache.resolve(d).name == "input.duckn.zip"
