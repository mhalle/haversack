"""Encode jobs on the server (2026-09-23): ``POST /v1/jobs`` with ``kind=encode``.

An embedding field is a second OUTPUT KIND through the queue, the result cache and the job
routes; labels stay the default. What these hold, each against a way it has failed or could:

- no existing result key moves (a segmentation's key is the old formula, byte for byte), and a
  field's key differs from a segmentation's of the same name - ``ts.v2:total_fast`` is both;
- an encode job's name is resolved as an ENCODER, never through the task catalog;
- the field is published and served as a field: named ``.zarr.zip``, typed as a zip, validated
  by its digest, refused a NIfTI conversion, advertised by no label or preview link;
- publication re-keys through the encoder's versions, not the segmentation's (a re-key through
  the wrong door publishes a result under a key nothing asks for - Modal's _EngineShim, 09-12).

The encoder itself is a double that writes a real zip: what the NETWORK computes is
``test_encoders.py``'s business.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve as serve_mod  # noqa: E402
from haversack.encoders import serving  # noqa: E402
from haversack.serve import FIELD_NAME, LocalExecutor, create_app, result_key  # noqa: E402

from test_serve import FakeSegmenter, volume_bytes, wait_state  # noqa: E402


class FakeEncoder:
    """``encode_file``'s signature; writes a small zip whose bytes depend on the input and int8."""

    def __init__(self):
        self.calls = []

    def __call__(self, name, path, out, *, identity, device="auto", int8=False, cancel=None,
                 task_weights=None, **_):
        self.calls.append({"name": name, "identity": identity, "int8": int8, "device": device})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            z.writestr("zarr.json", json.dumps({"encoder": name, "input": identity.get("digest"), "int8": int8}))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(buf.getvalue())
        return {"field": str(out), "bytes": out.stat().st_size, "encoder": name, "revision": "r",
                "license": "CC-BY-NC-SA-4.0", "device": device, "dtype": "fp32", "int8": int8,
                "model_grid": [8, 8, 8], "tokens": [1, 8, 64], "seconds": {"total": 0.0}}


def make(tmp_path, token=None):
    seg, enc = FakeSegmenter(steps=1), FakeEncoder()
    ex = LocalExecutor(seg, workdir=tmp_path / "work", cache_dir=tmp_path / "cache", encode_fn=enc)
    return seg, enc, ex, TestClient(create_app(ex, token=token))


def post(client, *, task="radar:pretrain", kind="encode", options=None, fill=0, **extra):
    return client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(fill))},
                       data={"task": task, "kind": kind, "options": json.dumps(options or {}), **extra})


# -- keys ----------------------------------------------------------------------

def _old_key(identity, task, options, weights):
    """``result_key`` as it was before kinds existed - written out, not called."""
    payload = json.dumps({"identity": list(identity), "task": str(task),
                          "options": {k: options[k] for k in sorted(options)},
                          "weights": list(weights), "haversack": serve_mod.CACHE_EPOCH}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_no_segmentation_key_moves():
    args = (("idc:1",), "ts.v2:total_fast", {"grid": 1.0}, ["297=v2.0.0"])
    assert result_key(*args) == _old_key(*args)
    assert result_key(*args, kind="segment") == _old_key(*args)


def test_a_field_and_labels_of_one_name_never_share_a_key():
    args = (("idc:1",), "ts.v2:total_fast", {}, ["297=v2.0.0"])
    assert result_key(*args, kind="encode") != result_key(*args)


def test_an_encoders_versions_are_its_pins_and_its_epoch():
    v = serving.field_versions(None, "radar")
    assert v[0].startswith("radar:pretrain@") and v[-1] == f"encode@epoch={serving.ENCODE_EPOCH}"
    assert any(x.startswith("checkpoint_radar_pretrain.pth=sha256:") for x in v)


def test_an_nnunet_encoder_keys_on_its_tasks_weights_only():
    class Seg:
        def describe(self, task):
            assert task == "ts.v2:total_fast"
            return {"weights_installed": [{"id": 297, "version": "v2.0.0"}], "step_size": 0.5,
                    "crop": "upstream", "auxiliary": [3]}
    v = serving.field_versions(Seg(), "ts.v2:total_fast")
    assert "297=v2.0.0" in v and not any(x.startswith(("step=", "crop=", "auxiliary")) for x in v)


# -- the job ---------------------------------------------------------------------

def test_an_encode_job_publishes_and_serves_a_field(tmp_path):
    _, enc, ex, client = make(tmp_path)
    r = post(client, task="radar")                       # an alias resolves to the canonical name
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done", "failed"))
    assert s["state"] == "done", s
    assert s["task"] == "radar:pretrain" and s["kind"] == "encode" and s["deliverables"] == []
    out = s["result"]["outputs"][0]
    assert out["name"] == "field" and out["kind"] == "field"
    links = s["links"]
    # the job-scoped meta, and nothing of labels: no labels/preview/statistics link, no path form
    assert set(links) == {"self", "events", "result", "meta"}, links
    assert links["meta"] == f"/v1/jobs/{s['id']}/meta.json"
    got = client.get(links["result"])
    assert got.status_code == 200
    assert got.headers["content-type"] == "application/zip"
    assert got.headers["content-disposition"].endswith(f'filename="radar_pretrain_{s["id"]}.zarr.zip"')
    assert got.headers["etag"] == f'"{out["sha256"]}"'
    assert "sha256:" + hashlib.sha256(got.content).hexdigest() == out["sha256"]
    assert zipfile.ZipFile(io.BytesIO(got.content)).namelist() == ["zarr.json"]
    head = client.head(links["result"])
    assert head.status_code == 200 and head.headers["content-length"] == str(len(got.content))
    assert client.get(links["result"], headers={"If-None-Match": got.headers["etag"]}).status_code == 304
    assert client.get(links["result"] + "?format=nii.gz").status_code == 422
    # the field records the job's identity, not the scratch path its bytes were staged at
    ident = enc.calls[0]["identity"]
    # an upload is named by its digest, and that is what the field records - never a hash of
    # what staging made of the bytes (a stored DICOM tree arrives decoded)
    assert ident == {"input": s["input_identity"][0], "digest": s["input_identity"][0]}
    # the cache entry holds the field under its own name, and its meta says what it is
    entry = ex.cache.get(s["key"])
    assert entry is not None and entry[0].name == FIELD_NAME
    stored = json.loads((entry[0].parent / "meta.json").read_text())
    assert stored["kind"] == "encode" and stored["task"] == "radar:pretrain"
    record = client.get(f"/v1/jobs/{s['id']}/meta.json").json()      # the result record
    assert record["outputs"][0]["kind"] == "field" and record["encoder"] == "radar:pretrain"
    ex.close()


def test_the_same_request_again_is_a_cache_hit_and_int8_is_another_result(tmp_path):
    _, enc, ex, client = make(tmp_path)
    a = wait_state(client, post(client).json()["id"], ("done",))
    b = client.get(f"/v1/jobs/{post(client).json()['id']}").json()
    assert b["state"] == "done" and b.get("cached") is True and b["key"] == a["key"]
    assert len(enc.calls) == 1
    c = wait_state(client, post(client, options={"int8": True}).json()["id"], ("done",))
    assert c["key"] != a["key"] and len(enc.calls) == 2 and enc.calls[1]["int8"] is True
    ex.close()


def test_publication_rekeys_through_the_encoders_versions(tmp_path, monkeypatch):
    """Keyed at submit on one version, published after the versions became knowable: the
    entry must sit under the key the ENCODER's versions give, which is the key status reports
    and every later lookup computes."""
    seen = iter([["before"], ["after"], ["after"], ["after"]])
    monkeypatch.setattr(serving, "field_versions", lambda seg, name: next(seen))
    _, _, ex, client = make(tmp_path)
    s = wait_state(client, post(client).json()["id"], ("done",))
    want = result_key(tuple(s["input_identity"]), "radar:pretrain", {}, ["after"], kind="encode")
    assert s["key"] == want
    assert ex.cache.get(want) is not None
    ex.close()


def test_refusals_name_their_cause(tmp_path):
    _, enc, ex, client = make(tmp_path)
    r = post(client, task="nope:nothing")
    assert r.status_code == 404 and "no encoder" in r.text
    r = post(client, task="radar:pretrain@deadbeef")
    assert r.status_code == 404 and "revision" in r.text
    r = post(client, options={"interp": "nearest"})
    assert r.status_code == 422 and "int8" in r.text
    r = post(client, options={"int8": "yes"})
    assert r.status_code == 422
    r = post(client, deliverables=json.dumps(["preview"]))
    assert r.status_code == 422 and "no_deliverables" in r.text
    r = post(client, kind="bogus")
    assert r.status_code == 422 and "job kind" in r.text
    assert enc.calls == []
    assert not any((tmp_path / "work").glob("*/scan*")), "a refused submit leaves no job directory"
    ex.close()


def test_a_task_name_is_still_a_segmentation_by_default(tmp_path):
    """No `kind`: exactly the old door - the FakeSegmenter's task, labels, no encoder call."""
    _, enc, ex, client = make(tmp_path)
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(3))},
                    data={"task": "total_fast"})
    s = wait_state(client, r.json()["id"], ("done",))
    assert "kind" not in s and s["result"]["outputs"][0]["name"] == "labels" and enc.calls == []
    ex.close()


def test_a_field_is_not_listed_as_a_segmentation(tmp_path):
    _, _, ex, client = make(tmp_path)
    wait_state(client, post(client).json()["id"], ("done",))
    rows = client.get("/v1/segmentations").json()
    assert rows.get("results", rows.get("rows", [])) == [], rows
    ex.close()


def test_the_server_lists_its_encoders(tmp_path):
    _, _, ex, client = make(tmp_path)
    rows = {e["name"]: e for e in client.get("/v1/encoders").json()["encoders"]}
    from haversack.encoders import ENCODERS
    assert set(rows) == set(ENCODERS)
    assert client.get("/v1/encoders").json()["encodes"] is True
    r = rows["radar:pretrain"]
    assert r["license"] == "CC-BY-NC-SA-4.0" and r["options"] == {"int8": "bool"}
    assert r["attribution"]["cite"][0]["doi"] == "10.1126/science.aec6129"
    assert r["installed"] in (True, False)
    ex.close()


def test_the_client_encodes_through_the_server(tmp_path):
    """``RemoteClient.encode`` is submit(kind=encode) + wait + fetch; a plain ``submit`` sends
    no ``kind`` at all, so a server from before encoding reads it as it always did."""
    from haversack.client import RemoteClient
    _, enc, ex, client = make(tmp_path)
    sent = []
    real = client.request

    def request(method, url, **kw):
        if method == "POST":
            sent.append(dict(kw.get("data") or {}))
        return real(method, url, **kw)
    client.request = request
    rc = RemoteClient("http://testserver")
    rc._http = client
    src = tmp_path / "scan.nii.gz"
    src.write_bytes(volume_bytes(7))
    out = tmp_path / "f.zarr.zip"
    final = rc.encode(src, "radar", out, int8=True)
    assert final["state"] == "done" and final["kind"] == "encode"
    assert sent[-1]["kind"] == "encode" and json.loads(sent[-1]["options"]) == {"int8": True}
    assert zipfile.is_zipfile(out) and enc.calls[-1]["int8"] is True
    rc.submit(src, "total_fast")
    assert "kind" not in sent[-1]
    ex.close()


def test_a_server_that_cannot_encode_says_so_before_any_job_exists(tmp_path):
    """Modal has no encoder worker yet: its executor says `encodes = False`, and the door
    answers 501 naming the local path - no job, no directory, no worker spawned."""
    _, enc, ex, client = make(tmp_path)
    ex.encodes = False
    before = set((tmp_path / "work").iterdir())
    r = post(client)
    assert r.status_code == 501 and "haversack encode" in r.text
    assert set((tmp_path / "work").iterdir()) == before and enc.calls == []
    ex.close()


def test_the_modal_executor_does_not_encode():
    import pytest as _p
    modal_app = _p.importorskip("haversack.modal_app")
    assert modal_app.ModalExecutor.encodes is False


def test_a_field_cannot_be_referred_to_as_an_image(tmp_path):
    """``result:<key>`` of a field: its output says ``kind: field``, and a segmentation's image
    role takes images, so the submit is refused by name - never run on a zip."""
    seg, enc, ex, client = make(tmp_path)
    s = wait_state(client, post(client).json()["id"], ("done",))
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "result", "id": s["key"]}])})
    assert r.status_code == 422 and "kind" in r.text, r.text
    assert seg.calls == []
    ex.close()


def test_a_hosted_inputs_field_gets_no_label_paths(tmp_path, monkeypatch):
    """An ``idc:`` input HAS a path form - for labels. ``ts.v2``-style links minted for a field
    would send a client to a segmentation's URLs (or to a label map of a task-named encoder)."""
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.nii.gz").write_bytes(volume_bytes(11))
        return d
    seg, enc = FakeSegmenter(steps=1), FakeEncoder()
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c", encode_fn=enc,
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.post("/v1/jobs", data={"task": "radar:pretrain", "kind": "encode",
                                      "source": json.dumps([{"kind": "idc", "crdc_series_uuid": u}])})
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done", "failed"))
    assert s["state"] == "done", s
    assert s["input_identity"] == [f"idc:{u}"]
    assert set(s["links"]) == {"self", "events", "result", "meta"}, s["links"]
    assert all(v.startswith("/v1/jobs/") for v in s["links"].values())
    assert enc.calls[0]["identity"]["source"] == "idc"
    ex.close()


def _idc_executor(tmp_path, monkeypatch, workdir="w", encode_fn=None):
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.nii.gz").write_bytes(volume_bytes(13))
        return d
    return LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / workdir, cache_dir=tmp_path / "c",
                         encode_fn=encode_fn or FakeEncoder(), fetch_idc_fn=fake_fetch)


def test_an_evicted_or_restarted_encode_job_is_still_a_field(tmp_path, monkeypatch):
    """The status built from the STORED record (after a restart, or once the record leaves
    memory) says `kind` as the live one does; without it the links door read the field as a
    segmentation and minted a task-named encoder's label paths (review, 2026-09-23)."""
    ex = _idc_executor(tmp_path, monkeypatch)
    client = TestClient(create_app(ex))
    u = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    r = client.post("/v1/jobs", data={"task": "ts.v2:total_fast", "kind": "encode",
                                      "source": json.dumps([{"kind": "idc", "crdc_series_uuid": u}])})
    jid = wait_state(client, r.json()["id"], ("done",))["id"]
    ex.close()
    ex2 = _idc_executor(tmp_path, monkeypatch)
    client2 = TestClient(create_app(ex2))
    s = client2.get(f"/v1/jobs/{jid}").json()
    assert s.get("evicted") is True and s["kind"] == "encode", s
    assert set(s["links"]) == {"self", "events", "result", "meta"}, s["links"]
    assert client2.get(s["links"]["result"]).headers["content-type"] == "application/zip"
    ex2.close()


def test_the_listing_refuses_a_field_by_what_it_is(tmp_path, monkeypatch):
    """Not by its key failing to recompute as a segmentation's: that check fails open, and a
    field came back as a segmentation row when key derivation raised (review, 2026-09-23)."""
    seg, _, ex, client = make(tmp_path)
    # a server whose catalog serves the task an encoder is named after, as a real one does
    real_describe = seg.describe
    seg.tasks = lambda: ["ts.v2:total_fast", "total"]
    seg.describe = lambda t: ({"name": t, "structures": ["spleen"]} if t in seg.tasks() else real_describe(t))
    wait_state(client, post(client, task="ts.v2:total_fast").json()["id"], ("done",))

    def broken(*a, **k):
        raise RuntimeError("no describe today")
    monkeypatch.setattr(serve_mod, "weights_versions_of", broken)
    body = client.get("/v1/segmentations").json()
    assert body.get("segmentations", []) == [], body
    ex.close()


def test_an_artifact_cannot_take_a_primary_outputs_name(tmp_path):
    _, _, ex, client = make(tmp_path)
    s = wait_state(client, post(client).json()["id"], ("done",))
    src = tmp_path / "x"
    src.write_bytes(b"x")
    for name in ("labels.seg.nrrd", FIELD_NAME):
        with pytest.raises(ValueError):
            ex.cache.add_artifact(s["key"], name, src)
    ex.close()


def test_the_cli_and_the_server_state_one_record(tmp_path, monkeypatch, capsys):
    """`haversack encoders --json` is `/v1/encoders`' record per encoder plus its aliases."""
    from haversack import cli
    monkeypatch.setenv("HAVERSACK_ENCODER_WEIGHTS", str(tmp_path / "ew"))
    assert cli.main(["encoders", "--json"]) == 0
    local = {r["name"]: r for r in json.loads(capsys.readouterr().out)["encoders"]}
    _, _, ex, client = make(tmp_path)
    served = {r["name"]: r for r in client.get("/v1/encoders").json()["encoders"]}
    for name, row in served.items():
        mine = {k: v for k, v in local[name].items() if k != "aliases"}
        assert set(mine) == set(row), name
        assert {k: v for k, v in mine.items() if k != "installed"} == {k: v for k, v in row.items() if k != "installed"}
        assert isinstance(mine["installed"], bool), (name, mine["installed"])
    ex.close()


def test_an_event_is_the_status_snapshot_key_and_links_included(tmp_path):
    """SERVER.md: each event IS the status snapshot. The stream sent the executor's raw record,
    without `key` and `links`, so a client whose wait() ended on the stream held a status with
    no handle (found on the Modal smoke, 2026-09-23). Segmentations and fields alike."""
    _, _, ex, client = make(tmp_path)
    jid = post(client).json()["id"]
    wait_state(client, jid, ("done",))
    with client.stream("GET", f"/v1/jobs/{jid}/events") as r:
        data = next(line for line in r.iter_lines() if line.startswith("data: "))
    snap = json.loads(data[len("data: "):])
    status = client.get(f"/v1/jobs/{jid}").json()
    assert snap["key"] == status["key"] and snap["links"] == status["links"], snap
    ex.close()
