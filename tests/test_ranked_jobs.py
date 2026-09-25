"""Ranked jobs on the server (2026-09-24): ``POST /v1/jobs`` with ``kind=ranked``.

A task's ranked store is a third OUTPUT KIND through the queue, the result cache and the job
routes, beside labels and embedding fields. What these hold, each against a way it could fail:

- no existing key moves, and a store's key differs from the labels' and an embedding's of the
  same task; it carries the formats the store is written in (``ranked_output.ranked_tag``);
- the job runs through THIS server's segmenter (its warm models, its cancellation), and the
  store names its input by the job's identity, never a scratch path;
- the store is published and served as a store: named ``.duckn.zip``, typed as a zip, validated
  by its digest, refused a NIfTI conversion, advertised by no label or preview link - and by its
  own path door, which reads and never computes;
- a store takes no options and no deliverables, a server that cannot write one says so before a
  job exists, and ``result:`` of a store is refused where an image belongs.

The store itself is a double (``segment_to_store`` writes a small zip): what a real store holds
is test_ranked_output.py's and test_ranked_compose.py's business.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
pytest.importorskip("duckn")
pytest.importorskip("rankfield")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import ranked_output  # noqa: E402
from haversack import serve as serve_mod  # noqa: E402
from haversack.serve import RANKED_NAME, LocalExecutor, create_app, result_key  # noqa: E402

from test_serve import FakeSegmenter, volume_bytes, wait_state  # noqa: E402

IDC = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"


class FakeStore:
    """``segment_to_store``'s signature: runs the server's own segmenter through ``run`` (as the
    real one does), then writes a small zip whose bytes depend on the task and the input."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, task, out, *, case=None, source=None, quiet=False, run=None,
                 progress=None, **kw):
        seg = run(image, task, probabilities="the-spec", progress=progress)
        self.calls.append({"image": str(image), "task": task, "source": source, "case": case})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            z.writestr("zarr.json", json.dumps({"task": task, "source": source}))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(buf.getvalue())
        return seg, out


def make(tmp_path, monkeypatch, **kw):
    store = FakeStore()
    monkeypatch.setattr(ranked_output, "segment_to_store", store)
    seg = FakeSegmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "work", cache_dir=tmp_path / "cache", **kw)
    return seg, store, ex, TestClient(create_app(ex))


def post(client, *, task="total_fast", kind="ranked", options=None, fill=0, **extra):
    return client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(fill))},
                       data={"task": task, "kind": kind, "options": json.dumps(options or {}), **extra})


# -- keys ----------------------------------------------------------------------

def test_a_store_labels_and_a_field_of_one_name_never_share_a_key():
    args = (("idc:1",), "ts.v2:total_fast", {}, ["297=v2.0.0"])
    keys = {result_key(*args), result_key(*args, kind="embed"), result_key(*args, kind="ranked")}
    assert len(keys) == 3


def test_a_stores_versions_are_its_tasks_and_the_formats_it_is_written_in():
    import duckn
    import rankfield
    seg = FakeSegmenter()
    v = serve_mod.versions_for(seg, "total_fast", "ranked")
    assert v[:-1] == serve_mod.weights_versions_of(seg, "total_fast")
    assert v[-1] == (f"ranked=rf{rankfield.FORMAT_VERSION}/seg{duckn.SEG_VERSION}"
                     f"/h{ranked_output.STORE_RULES}")


# -- the job ---------------------------------------------------------------------

def test_a_ranked_job_publishes_and_serves_a_store(tmp_path, monkeypatch):
    seg, store, ex, client = make(tmp_path, monkeypatch)
    r = post(client)
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done", "failed"))
    assert s["state"] == "done", s
    assert s["kind"] == "ranked" and s["deliverables"] == []
    out = s["result"]["outputs"][0]
    assert out["name"] == "ranked" and out["kind"] == "ranked"
    assert s["result"]["names"]                      # the run's names, as a segmentation reports
    # the server's own segmenter ran it, with the ranked sink - not a second model load
    assert len(seg.calls) == 1 and seg.calls[0][2].get("probabilities") == "the-spec"
    # an upload has no path form: the job is reached through itself
    links = s["links"]
    assert set(links) == {"self", "events", "result", "meta"}, links
    got = client.get(links["result"])
    assert got.status_code == 200
    assert got.headers["content-type"] == "application/zip"
    assert got.headers["content-disposition"].endswith(f'_{s["id"]}.duckn.zip"')
    assert got.headers["etag"] == f'"{out["sha256"]}"'
    assert "sha256:" + hashlib.sha256(got.content).hexdigest() == out["sha256"]
    head = client.head(links["result"])
    assert head.status_code == 200 and head.headers["content-length"] == str(len(got.content))
    assert client.get(links["result"], headers={"If-None-Match": got.headers["etag"]}).status_code == 304
    assert client.get(links["result"] + "?format=nii.gz").status_code == 422
    # the store names the job's identity, never the scratch path it was staged at
    assert store.calls[0]["source"] == {"type": "image", "identifier": s["input_identity"][0]}
    entry = ex.cache.get(s["key"])
    assert entry is not None and entry[0].name == RANKED_NAME
    stored = json.loads((entry[0].parent / "meta.json").read_text())
    assert stored["kind"] == "ranked" and stored["task"] == s["task"]
    ex.close()


def test_the_same_request_again_is_a_cache_hit(tmp_path, monkeypatch):
    seg, store, ex, client = make(tmp_path, monkeypatch)
    a = wait_state(client, post(client).json()["id"], ("done",))
    b = client.get(f"/v1/jobs/{post(client).json()['id']}").json()
    assert b["state"] == "done" and b.get("cached") is True and b["key"] == a["key"]
    assert len(store.calls) == 1
    # the segmentation of the same bytes and task is another result
    lab = client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(0))},
                      data={"task": "total_fast"})
    c = wait_state(client, lab.json()["id"], ("done",))
    assert c["key"] != a["key"] and "kind" not in c
    assert c["result"]["outputs"][0]["name"] == "labels"
    ex.close()


def test_refusals_name_their_cause(tmp_path, monkeypatch):
    seg, store, ex, client = make(tmp_path, monkeypatch)
    before = set((tmp_path / "work").iterdir())
    r = post(client, options={"depth": 8})
    assert r.status_code == 422 and "no_options" in r.text
    r = post(client, deliverables=json.dumps(["preview"]))
    assert r.status_code == 422 and "no_deliverables" in r.text
    r = post(client, task="nope:nothing")
    assert r.status_code == 404
    monkeypatch.setattr(ranked_output, "supports_store_output", lambda task: False)
    r = post(client)
    assert r.status_code == 422 and "no_distribution" in r.text
    assert store.calls == [] and seg.calls == []
    assert set((tmp_path / "work").iterdir()) == before, "a refused submit leaves no job directory"
    ex.close()


def test_a_server_that_cannot_write_a_store_says_so_before_any_job_exists(tmp_path, monkeypatch):
    seg, store, ex, client = make(tmp_path, monkeypatch)
    monkeypatch.setattr(ranked_output, "store_extra_missing", lambda: ["duckn"])
    assert ex.ranked_stores is False
    before = set((tmp_path / "work").iterdir())
    r = post(client)
    assert r.status_code == 501 and ".duckn.zip" in r.text
    assert set((tmp_path / "work").iterdir()) == before and store.calls == []
    ex.close()


def test_a_store_is_not_listed_as_a_segmentation(tmp_path, monkeypatch):
    _, _, ex, client = make(tmp_path, monkeypatch)
    wait_state(client, post(client).json()["id"], ("done",))
    body = client.get("/v1/segmentations").json()
    assert body.get("segmentations", []) == [], body
    ex.close()


def test_a_store_cannot_be_referred_to_as_an_image(tmp_path, monkeypatch):
    seg, _, ex, client = make(tmp_path, monkeypatch)
    s = wait_state(client, post(client).json()["id"], ("done",))
    n = len(seg.calls)
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "result", "id": s["key"]}])})
    assert r.status_code == 422 and "kind" in r.text, r.text
    assert len(seg.calls) == n
    ex.close()


# -- hosted inputs: the path door -----------------------------------------------------

def _idc(tmp_path, monkeypatch, workdir="w"):
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.nii.gz").write_bytes(volume_bytes(13))
        return d
    store = FakeStore()
    monkeypatch.setattr(ranked_output, "segment_to_store", store)
    ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / workdir, cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    return store, ex, TestClient(create_app(ex))


def _submit_idc(client):
    r = client.post("/v1/jobs", data={"task": "total_fast", "kind": "ranked",
                                      "source": json.dumps([{"kind": "idc", "crdc_series_uuid": IDC}])})
    assert r.status_code == 202, r.text
    return wait_state(client, r.json()["id"], ("done", "failed"))


def test_a_hosted_inputs_store_has_its_own_path_and_no_label_paths(tmp_path, monkeypatch):
    store, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)
    assert s["state"] == "done", s
    path = f"/v1/idc/{IDC}/{s['task']}/{RANKED_NAME}"
    assert s["links"]["ranked"] == path
    assert not any(k in s["links"] for k in ("labels", "preview", "statistics")), s["links"]
    job = client.get(s["links"]["result"])
    got = client.get(path)
    assert got.status_code == 200 and got.content == job.content
    assert got.headers["content-type"] == "application/zip"
    assert got.headers["etag"] == job.headers["etag"]
    assert client.head(path).status_code == 200
    assert client.get(path, headers={"If-None-Match": got.headers["etag"]}).status_code == 304
    assert store.calls[0]["source"] == {"type": "image", "identifier": f"idc:{IDC}"}
    ex.close()


def test_the_path_door_reads_and_never_computes(tmp_path, monkeypatch):
    store, ex, client = _idc(tmp_path, monkeypatch)
    path = f"/v1/idc/{IDC}/total_fast/{RANKED_NAME}"
    r = client.get(path, headers={"Prefer": "wait=5"})
    assert r.status_code == 404 and "kind=ranked" in r.text
    assert r.headers.get("cache-control") == "no-store"
    assert store.calls == [] and client.get("/v1/jobs").json()["jobs"] == []
    assert client.get(f"/v1/idc/{IDC}/nope/{RANKED_NAME}").status_code == 404
    ex.close()


def test_an_evicted_or_restarted_ranked_job_is_still_a_store(tmp_path, monkeypatch):
    """The status built from the STORED record says `kind` as the live one does, so the links
    door never mints a store's job label paths."""
    store, ex, client = _idc(tmp_path, monkeypatch)
    jid = _submit_idc(client)["id"]
    ex.close()
    _store2, ex2, client2 = _idc(tmp_path, monkeypatch)
    s = client2.get(f"/v1/jobs/{jid}").json()
    assert s.get("evicted") is True and s["kind"] == "ranked", s
    assert not any(k in s["links"] for k in ("labels", "preview", "statistics")), s["links"]
    assert client2.get(s["links"]["result"]).headers["content-type"] == "application/zip"
    ex2.close()


def test_an_artifact_cannot_take_a_stores_name(tmp_path, monkeypatch):
    _, _, ex, client = make(tmp_path, monkeypatch)
    s = wait_state(client, post(client).json()["id"], ("done",))
    src = tmp_path / "x"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        ex.cache.add_artifact(s["key"], RANKED_NAME, src)
    ex.close()
