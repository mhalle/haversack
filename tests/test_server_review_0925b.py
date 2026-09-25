"""The 2026-09-25 server review, second batch - each pinned by a test that fails on c93b699.

5. The embedding worker's image had no pydicom (the `embed` extra lacks it), so it made no input
   copies; and its prefetcher pre-read the next job's image, which the encoder then discarded.
6. The anonymous twin keyed on a weights volume frozen at container start: a task a worker
   installed since keyed "unknown", and its results 404ed on the anonymous path.
7. A read-only result-store server reached a task's rank field only if a segmentation of the task
   had been published too (only segmentations noted their versions), and never an nnU-Net
   encoder's embedding.
8. One try around both deliverables: a preview that failed to render cost the statistics too,
   and neither was said - the job kept linking both.
9. Advice the server refuses: a missing deliverable's 404 advised a plain submit, which on Modal
   renders nothing; a rank field's job was told to ask for deliverables it cannot take; an IDC
   refusal pointed at a /v1/resolve that does not exist.
"""
from __future__ import annotations

import ast
import json
import threading
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import haversack  # noqa: E402
from haversack import serve as serve_mod  # noqa: E402
from haversack.serve import LocalExecutor, artifact_overlap, create_app  # noqa: E402

from test_deliverables import renders  # noqa: E402,F401 - the fixture
from test_serve import FakeSegmenter, submit, volume_bytes, wait_state  # noqa: E402

SRC = Path(haversack.__file__).parent
IDC = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"


# -- 5. the embedding worker ------------------------------------------------------------------

def test_the_embedding_image_can_make_input_copies():
    tree = ast.parse((SRC / "modal_app.py").read_text(encoding="utf-8"))
    syncs = [[e.value for e in kw.value.elts] for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "uv_sync"
             for kw in n.keywords if kw.arg == "extras"]
    embed = [x for x in syncs if "embed" in x]
    assert embed and all("duckn" in x for x in embed), embed      # pydicom rides in `duckn`


def test_the_embedding_worker_stages_the_next_input_and_does_not_read_it(monkeypatch):
    pytest.importorskip("modal")
    from haversack import modal_app as m
    read, staged = [], []
    monkeypatch.setattr(m, "_prefetch_candidate", lambda cur, engine: ("idc", "idc:x", "j2"))
    monkeypatch.setattr(m, "fill_read_ahead", lambda *a, **k: read.append(1) or True)
    cache = types.SimpleNamespace(staging=lambda s: False, has=lambda s: False,
                                  prefetch=lambda s: staged.append(s) or True)
    ra = types.SimpleNamespace(has=lambda s: False)
    for engine, reads in ((m.EMBED_WORKER, 0), ("nnunetv2", 1)):
        read.clear()
        stop = threading.Event()
        before = {t.ident for t in threading.enumerate()}
        m._prefetch_next("j1", stop, cache, ra, threading.Lock(), engine=engine)
        for t in threading.enumerate():
            if t.ident not in before and t.name == "haversack-prefetch":
                t.join(5)
        assert len(read) == reads, (engine, read)
    assert staged == ["idc:x", "idc:x"]                            # both still staged it


# -- 6. the twin's weights ----------------------------------------------------------------------

def test_the_twin_reloads_a_frozen_weights_volume_when_a_key_would_be_unknown(monkeypatch):
    pytest.importorskip("modal")
    from haversack import modal_app as m
    installed = {"now": False}
    reloads = []
    monkeypatch.setattr(serve_mod, "weights_versions_of",
                        lambda seg, task: ["297=v2"] if installed["now"] else ["unknown"])
    monkeypatch.setattr(m, "weights_vol", types.SimpleNamespace(
        reload=lambda: reloads.append(1) or installed.update(now=True)))
    monkeypatch.setattr(m, "_twin_reload", {"at": 0.0})
    assert m._twin_weights_versions("seg", "ts.v2:total_fast") == ["297=v2"]
    assert len(reloads) == 1
    installed["now"] = False                                        # throttled: no second reload
    assert m._twin_weights_versions("seg", "ts.v2:total_fast") == ["unknown"]
    assert len(reloads) == 1


# -- 7. the read-only result-store server --------------------------------------------------------

def _store_server(tmp_path, monkeypatch, **kw):
    pytest.importorskip("obstore")
    from obstore.store import MemoryStore
    from haversack import objectcache
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.nii.gz").write_bytes(volume_bytes(13))
        return d
    store = MemoryStore()
    seg = FakeSegmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c", result_store=store,
                       fetch_idc_fn=fetch, **kw)
    shim = type("S", (), {"tasks": seg.tasks,
                          "resolve_task": staticmethod(lambda t: t if t in seg.tasks() else None)})()
    reader = TestClient(objectcache.read_only_app(store, local_dir=tmp_path / "reader",
                                                  segmenter=shim))
    return seg, ex, TestClient(create_app(ex)), reader


def _job(client, **data):
    r = client.post("/v1/jobs", data={**data, "source": json.dumps(
        [{"kind": "idc", "crdc_series_uuid": IDC}])})
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done", "failed"))
    assert s["state"] == "done", s
    return s


def test_a_read_only_store_server_reaches_a_rank_field_alone(tmp_path, monkeypatch):
    pytest.importorskip("duckn")
    from haversack import ranked_output
    from test_ranked_jobs import FakeStore
    monkeypatch.setattr(ranked_output, "segment_to_store", FakeStore())
    _, ex, client, reader = _store_server(tmp_path, monkeypatch)
    try:
        _job(client, task="total_fast", kind="rankfield")          # no segmentation of it
        path = f"/v1/idc/{IDC}/total_fast/rankfield.duckn.zip"
        assert client.get(path).status_code == 200
        got = reader.get(path)
        assert got.status_code == 200 and got.content == client.get(path).content
        assert len(reader.get("/v1/rankfields").json()["rankfields"]) == 1
    finally:
        ex.close()


def test_a_read_only_store_server_reaches_an_nnunet_encoders_embedding(tmp_path, monkeypatch):
    from test_embed_jobs import FakeEncoder
    seg, ex, client, reader = _store_server(tmp_path, monkeypatch, embed_fn=FakeEncoder())
    try:
        seg.tasks = lambda: ["ts.v2:total_fast", "total"]
        real = seg.describe
        seg.describe = lambda t: ({"name": t, "structures": ["spleen"],
                                   "weights_installed": [{"id": 297, "version": "v2"}]}
                                  if t in seg.tasks() else real(t))
        _job(client, task="ts.v2:total_fast", kind="embed")
        path = f"/v1/idc/{IDC}/ts.v2:total_fast/embedding.zarr.zip"
        assert client.get(path).status_code == 200
        got = reader.get(path)
        assert got.status_code == 200 and got.content == client.get(path).content
    finally:
        ex.close()


# -- 8. each deliverable on its own, and what did not render is said --------------------------

def test_a_failed_preview_costs_nothing_else_and_is_said_before_finish(tmp_path, monkeypatch):
    from haversack import preview, statistics
    monkeypatch.setattr(preview, "render_preview",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("empty label map")))
    out = tmp_path / "s.json"
    monkeypatch.setattr(statistics, "compute_statistics",
                        lambda *a, **k: out.write_text("{}") or out)
    order, placed = [], []
    artifact_overlap(None, "t", ("preview", "statistics"), preview_out=tmp_path / "p.png",
                     statistics_out=out, place=lambda n, p: placed.append(n) or True,
                     finish=lambda done: order.append(("finish", [n for n, _ in done])),
                     unavailable=lambda failed: order.append(("unavailable", failed)))
    assert placed == ["statistics.json"]
    assert order[0][0] == "unavailable" and "empty label map" in order[0][1]["preview"]
    assert order[1] == ("finish", ["statistics"])


def test_a_job_whose_preview_rendered_nothing_stops_linking_it(tmp_path, monkeypatch, renders):
    from haversack import preview
    from test_deliverables import BASE, _post, _quiet, _server
    monkeypatch.setattr(preview, "render_preview", lambda *a, **k: None)   # rendered nothing
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    try:
        s = wait_state(client, _post(client)["id"], ("done",))
        _quiet(ex, s["key"])                   # the overlap runs after done
        s = client.get(f"/v1/jobs/{s['id']}").json()
        assert "nothing to render" in s["deliverables_unavailable"]["preview"]
        assert "preview" not in s["links"] and "statistics" in s["links"]
        r = client.get(f"/v1/jobs/{s['id']}/preview.png")
        assert r.status_code == 404 and "nothing to render" in r.text
        assert client.get(f"{BASE}/statistics.json").status_code == 200
    finally:
        ex.close()


# -- 9. advice the server can follow ------------------------------------------------------------

def test_a_missing_deliverables_advice_is_what_this_server_can_do(tmp_path, monkeypatch, renders):
    from test_deliverables import BASE, _post, _server
    _, ex, client, _ = _server(tmp_path, monkeypatch)
    try:
        wait_state(client, _post(client, deliverables=[])["id"], ("done",))   # none rendered
        local = client.get(f"{BASE}/preview.png")
        assert local.status_code == 404 and 'deliverables ["preview"] renders it' in local.json()["detail"]
        monkeypatch.setattr(LocalExecutor, "renders_on_hit", False)            # as Modal
        modal_like = client.get(f"{BASE}/preview.png")
        assert modal_like.status_code == 404 and "no-cache" in modal_like.text
    finally:
        ex.close()


def test_a_rank_fields_job_is_not_told_to_ask_for_deliverables(tmp_path, monkeypatch):
    pytest.importorskip("duckn")
    from test_ranked_jobs import make, post
    _, _, ex, client = make(tmp_path, monkeypatch)
    try:
        s = wait_state(client, post(client).json()["id"], ("done",))
        r = client.get(f"/v1/jobs/{s['id']}/preview.png")
        assert r.status_code == 404 and "only a segmentation renders deliverables" in r.text
        assert "deliverables [" not in r.text
    finally:
        ex.close()


def test_no_refusal_points_at_a_route_that_does_not_exist(tmp_path):
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    client = TestClient(create_app(ex))
    try:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/resolve" not in paths
        for src in ({"kind": "idc", "crdc_series_uuid": "1.2.840.1"},
                    {"kind": "idc", "series_instance_uid": "1.2.840.1"},
                    {"kind": "idc", "series": "x"}):
            r = client.post("/v1/jobs", data={"task": "total_fast", "source": json.dumps([src])})
            assert r.status_code == 422 and "/v1/resolve" not in r.text, r.text
    finally:
        ex.close()
