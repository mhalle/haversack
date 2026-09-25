"""The 2026-09-25 server review: four defects, each pinned by a test that fails on 9c69466.

1. An upload job's input resolved from a store entry that was not committed - evicted after the
   submit checked it, and being written again by a re-upload of the same bytes - so the job read
   a half-written series and published its labels under the whole content's key.
2. The upload routes (PUT and POST /v1/inputs) ran the store's adopt - since the input copy a
   decode, a compression and a verify of the whole volume - on the event loop, stalling every
   other request to the container.
3. On Modal a submit of a key already computing started a second computation: the executor
   never joined a flight (the local one does, and SERVER.md says a submit joins).
4. A reader failure that is not an InputError (SimpleITK's bare RuntimeError on slices of
   different sizes) escaped the input copy's transcode: a fetched input was torn down and
   downloaded again on every job, an upload was a 500 - where before the copy both were kept.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import shutil
import threading
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("fastapi")
pydicom = pytest.importorskip("pydicom")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import input_copy as ic  # noqa: E402
from haversack.content import ContentStore  # noqa: E402
from haversack.errors import InputError  # noqa: E402
from haversack.serve import LocalExecutor, SeriesCache, create_app  # noqa: E402

from test_input_copy import write_series  # noqa: E402
from test_serve import FakeSegmenter  # noqa: E402


# -- 1. never resolve an entry that is not committed -----------------------------------------

def _half_written(tmp_path):
    """A stored series, discarded, and a re-upload of the same bytes stopped half way."""
    series = write_series(tmp_path / "s", n=6)
    cache = SeriesCache(tmp_path / "cache", lambda k, e: None)
    store = ContentStore(cache)
    digest = store.put_dir(series)
    assert cache.discard(digest)
    gate, half = threading.Event(), threading.Event()
    members = sorted(series.iterdir())

    def slow(key, entry):
        d = Path(entry) / "series"
        d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(members):
            if i == 3:
                half.set()
                gate.wait(10)
            shutil.copyfile(p, d / p.name)
        return d
    t = threading.Thread(target=lambda: cache.get_or_fetch(digest, fetch=slow))
    t.start()
    assert half.wait(5)
    return cache, store, digest, gate, t


def test_the_store_never_resolves_an_entry_that_is_not_committed(tmp_path):
    cache, store, digest, gate, t = _half_written(tmp_path)
    try:
        cache.pin(digest)
        with pytest.raises(FileNotFoundError):
            store.fast_path(digest)
    finally:
        gate.set()
        t.join()
    assert store.has(digest) and store.fast_path(digest).exists()   # once committed, it is


def test_an_input_job_refuses_an_input_evicted_after_its_submit(tmp_path):
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    try:
        digest = ex.content.put_dir(write_series(tmp_path / "s"))
        ex.content.cache.discard(digest)
        with pytest.raises(InputError, match="no longer held"):
            ex._from_store({"kind": "input", "id": digest}, [])
    finally:
        ex.close()


# -- 2. the upload routes store off the event loop ----------------------------------------------

def _spy_on_the_loop(monkeypatch):
    seen = []
    real = ic.transcode

    def spy(*a, **k):
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return real(*a, **k)
    monkeypatch.setattr(ic, "transcode", spy)
    return seen


def _nifti_bytes(tmp_path) -> bytes:
    p = tmp_path / "v.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)),
                    str(p), True)
    return p.read_bytes()


def test_put_inputs_stores_off_the_event_loop(tmp_path, monkeypatch):
    seen = _spy_on_the_loop(monkeypatch)
    raw = _nifti_bytes(tmp_path)
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    try:
        client = TestClient(create_app(ex))
        d = "sha256:" + hashlib.sha256(raw).hexdigest()
        assert client.put(f"/v1/inputs/{d}", content=raw).status_code == 200
    finally:
        ex.close()
    assert seen == [False], seen


def test_post_inputs_stores_off_the_event_loop(tmp_path, monkeypatch):
    seen = _spy_on_the_loop(monkeypatch)
    series = write_series(tmp_path / "s")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in sorted(series.iterdir()):
            z.write(p, p.name)
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    try:
        client = TestClient(create_app(ex))
        r = client.post("/v1/inputs", files={"file": ("s.zip", buf.getvalue())})
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "tree"
    finally:
        ex.close()
    assert seen == [False], seen


# -- 4. a reader failure keeps the original ----------------------------------------------------

def _uneven_rows(d: Path) -> Path:
    """A series SimpleITK's series reader throws a bare RuntimeError on: one slice smaller."""
    write_series(d)
    f = sorted(d.glob("*.dcm"))[2]
    ds = pydicom.dcmread(f)
    ds.Rows = 4
    ds.PixelData = np.zeros((4, 5), np.int16).tobytes()
    ds.save_as(f)
    return d


def test_a_series_the_reader_throws_on_is_kept_and_fetched_once(tmp_path, capsys):
    fetches = []

    def fetch(key, entry):
        fetches.append(1)
        return _uneven_rows(Path(entry) / "series")
    cache = SeriesCache(tmp_path / "cache", fetch)
    for _ in range(2):
        got = cache.get_or_fetch("fixture:uneven-rows")
        assert got.name == "series" and got.is_dir()
    assert len(fetches) == 1
    assert "the original is kept" in capsys.readouterr().err


def test_an_upload_the_reader_throws_on_is_stored_as_it_came(tmp_path):
    series = _uneven_rows(tmp_path / "s")
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    try:
        client = TestClient(create_app(ex))
        files = [("file", (p.name, p.read_bytes())) for p in sorted(series.iterdir())]
        r = client.post("/v1/inputs", files=files)
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "tree" and r.json()["stored"] is True
    finally:
        ex.close()


# -- 3. Modal: a submit joins a flight -------------------------------------------------------

def _modal(monkeypatch):
    pytest.importorskip("modal")
    from test_modal_app import _swap_dict
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None, kind="segment":
                        spawned.append(jid) or types.SimpleNamespace(object_id=f"fc-{jid}"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: fake.__setitem__(jid, {**fake.get(jid, {}), **d}))
    monkeypatch.setattr(m.modal.FunctionCall, "from_id",
                        staticmethod(lambda cid: types.SimpleNamespace(cancel=lambda: None)))
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task, kind="segment": ["v"])
    monkeypatch.setattr(ex, "cache_get", lambda key: None)
    monkeypatch.setattr(ex, "_cache_record", lambda key, wanted=(): None)
    monkeypatch.setattr(ex, "_flight_alive", lambda jid, meta: True)
    return m, fake, ex, spawned


def _submit(ex, tmp_path, jid, **kw):
    return ex.submit(jid, tmp_path / jid, None, "ts.v2:total_fast", {}, identity=("idc:1",), **kw)


def test_a_second_submit_of_a_computing_key_joins_it(monkeypatch, tmp_path):
    m, fake, ex, spawned = _modal(monkeypatch)
    a = _submit(ex, tmp_path, "a", deliverables=["preview"])
    b = _submit(ex, tmp_path, "b", deliverables=["statistics"])
    assert spawned == ["a"]                                    # one computation
    assert b["id"] == "a" and "b" not in fake
    assert set(fake["a"]["deliverables"]) == {"preview", "statistics"}   # queued: the list joins
    # one rider leaving does not end the flight for the other; the last DELETE cancels
    assert ex.cancel("a") == ("released", False)
    assert fake["a"]["state"] == "queued"
    assert ex.cancel("a") == ("cancelled", False)
    assert a["id"] == "a"


def test_a_flight_fetched_with_other_credentials_is_not_joined(monkeypatch, tmp_path):
    m, fake, ex, spawned = _modal(monkeypatch)
    _submit(ex, tmp_path, "a", source_tokens={"tcia": "secret"})
    _submit(ex, tmp_path, "b")
    assert spawned == ["a", "b"]
    assert "secret" not in repr(dict(fake))                    # a digest, never the token


def test_no_cache_never_joins(monkeypatch, tmp_path):
    m, fake, ex, spawned = _modal(monkeypatch)
    _submit(ex, tmp_path, "a")
    _submit(ex, tmp_path, "b", no_cache=True)
    assert spawned == ["a", "b"]


def test_a_finished_flights_marker_is_taken_over(monkeypatch, tmp_path):
    m, fake, ex, spawned = _modal(monkeypatch)
    _submit(ex, tmp_path, "a")
    fake["a"] = {**fake["a"], "state": "done"}
    _submit(ex, tmp_path, "b")
    assert spawned == ["a", "b"]
    key = fake["b"]["cache_key"]
    assert fake[f"inflight:{key}"] == "b"


def test_a_claim_lost_to_a_concurrent_submit_joins_it(monkeypatch, tmp_path):
    """Two submits look, both see no flight, and both try to claim the key: the loser must
    ride the winner's flight, not overwrite its marker and compute again."""
    m, fake, ex, spawned = _modal(monkeypatch)
    _submit(ex, tmp_path, "a")
    real = ex.find_inflight
    calls = []

    def blind_once(key):                       # b's first look races a's claim
        calls.append(key)
        return None if len(calls) == 1 else real(key)
    monkeypatch.setattr(ex, "find_inflight", blind_once)
    b = _submit(ex, tmp_path, "b")
    assert spawned == ["a"] and b["id"] == "a" and "b" not in fake
    assert fake[f"inflight:{fake['a']['cache_key']}"] == "a"


def test_the_route_answers_a_joined_submit_with_the_flights_job(monkeypatch, tmp_path):
    """Through the HTTP door: the second POST's answer is the running job's status, under its
    id - the route asked for the NEW id's status, which a joined submit never recorded."""
    pytest.importorskip("modal")
    from test_deliverables_modal import _api, _post
    m, jobs, vol, ex, client, spawned = _api(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "_flight_alive", lambda jid, meta: True)
    a = _post(client, ["statistics"])
    b = _post(client, ["preview"])
    assert b["id"] == a["id"] and spawned == [a["id"]]
    assert set(jobs[a["id"]]["deliverables"]) == {"preview", "statistics"}
