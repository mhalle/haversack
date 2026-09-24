"""Embeddings are listed and fetched by path (2026-09-24).

``GET /v1/embeddings`` is the segmentations listing asking for the other kind - the same cache
scan, cursor and identity filter, a row kept only when its meta says it is an embedding, and a
key round trip through the encoder's versions. A row with a path carries
``links.embedding``, ``/v1/<source>/<id>/<encoder>/embedding[_int8].zarr.zip``: a READ door
beside the labels' (200 / 304, 202 while a job runs, 404 otherwise), which computes nothing.
Driven through the local server with a fake encoder and a fake IDC fetch; the anonymous twin
over the same cache must answer the same paths.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve as serve_mod  # noqa: E402
from haversack.errors import HaversackError  # noqa: E402
from haversack.serve import (ResultsNotVisible, create_app, create_public_app,  # noqa: E402
                             result_key, versions_for, weights_versions_of)

from test_embed_jobs import FakeEncoder, _idc_executor, make, post  # noqa: E402
from test_serve import submit, wait_state  # noqa: E402

U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
PATH = f"/v1/idc/{U}/radar:pretrain/embedding.zarr.zip"


def _embed_idc(client, *, u=U, options=None, encoder="radar:pretrain"):
    r = client.post("/v1/jobs", data={"task": encoder, "kind": "embed",
                                      "options": json.dumps(options or {}),
                                      "source": json.dumps([{"kind": "idc", "crdc_series_uuid": u}])})
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"], ("done", "failed"))
    assert s["state"] == "done", s
    return s


def _server(tmp_path, monkeypatch, **kw):
    enc = FakeEncoder()
    ex = _idc_executor(tmp_path, monkeypatch, embed_fn=enc, **kw)
    return enc, ex, TestClient(create_app(ex))


# -- the listing ------------------------------------------------------------------------

def test_an_embedding_is_listed_with_its_path_and_a_label_map_is_not(tmp_path, monkeypatch):
    enc, ex, client = _server(tmp_path, monkeypatch)
    s = _embed_idc(client)
    wait_state(client, submit(client, task="total_fast"), ("done",))    # a label map beside it
    body = client.get("/v1/embeddings").json()
    assert body["next_cursor"] is None
    rows = body["embeddings"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["key"] == s["key"] and row["encoder"] == "radar:pretrain"
    assert row["identity"] == [f"idc:{U}"] and row["options"] == {} and row["bytes"] > 0
    assert row["links"] == {"embedding": PATH}
    # and the segmentations listing still keeps embeddings out
    assert all(r["key"] != s["key"] for r in client.get("/v1/segmentations").json()["segmentations"])
    ex.close()


def test_an_uploaded_inputs_embedding_is_listed_without_a_path(tmp_path):
    _, _, ex, client = make(tmp_path)
    s = wait_state(client, post(client).json()["id"], ("done",))
    rows = client.get("/v1/embeddings").json()["embeddings"]
    assert [r["key"] for r in rows] == [s["key"]]
    assert "links" not in rows[0] and rows[0]["identity"][0].startswith("sha256")
    ex.close()


def test_the_filters_find_by_identity_and_encoder(tmp_path, monkeypatch):
    _, ex, client = _server(tmp_path, monkeypatch)
    s = _embed_idc(client)

    def keys(**params):
        r = client.get("/v1/embeddings", params=params)
        assert r.status_code == 200, r.text
        return [e["key"] for e in r.json()["embeddings"]]
    assert keys(identity=f"idc:{U}") == [s["key"]]
    assert keys(identity=f"idc:{U.upper()}") == [s["key"]]           # normalized as the path is
    assert keys(identity="idc:11111111-2222-3333-4444-555555555555") == []
    assert keys(encoder="radar") == [s["key"]]                         # an alias resolves
    assert keys(encoder="ts.v2:total_fast") == []
    assert keys(identity=f"idc:{U}", encoder="radar:pretrain") == [s["key"]]
    assert client.get("/v1/embeddings", params={"encoder": "nope"}).status_code == 422
    ex.close()


def test_a_page_at_a_time_by_cursor(tmp_path, monkeypatch):
    _, ex, client = _server(tmp_path, monkeypatch)
    a = _embed_idc(client)["key"]
    b = _embed_idc(client, options={"int8": True})["key"]
    first = client.get("/v1/embeddings", params={"limit": 1}).json()
    second = client.get("/v1/embeddings", params={"limit": 1, "cursor": first["next_cursor"]}).json()
    assert {first["embeddings"][0]["key"], second["embeddings"][0]["key"]} == {a, b}
    assert client.get("/v1/embeddings", params={"cursor": "garbage"}).status_code == 422
    ex.close()


def test_a_label_map_of_an_encoders_name_is_not_an_embedding(tmp_path, monkeypatch):
    """`ts.v2:total_fast` is a task and an encoder. Its segmentation passes the encoder-name
    check, so only the row's KIND keeps it out of this listing (mutation found no test
    reaching the kind check, 2026-09-24)."""
    _, _, ex, client = make(tmp_path)
    seg = ex.segmenter
    real_describe = seg.describe
    seg.tasks = lambda: ["ts.v2:total_fast", "total"]
    seg.describe = lambda t: ({"name": t, "structures": ["spleen"]} if t in seg.tasks()
                              else real_describe(t))
    s = wait_state(client, submit(client, task="ts.v2:total_fast"), ("done",))
    assert any(r["key"] == s["key"] for r in client.get("/v1/segmentations").json()["segmentations"])
    assert client.get("/v1/embeddings").json()["embeddings"] == []
    ex.close()


def test_an_embedding_keyed_under_other_weights_is_not_offered(tmp_path, monkeypatch):
    """The key round trip: once the encoder's versions move (another revision installed), an
    embedding keyed under the old ones would 404 at its own link - so it is not listed."""
    _, ex, client = _server(tmp_path, monkeypatch)
    _embed_idc(client)
    assert len(client.get("/v1/embeddings").json()["embeddings"]) == 1
    real = serve_mod.versions_for
    monkeypatch.setattr(serve_mod, "versions_for",
                        lambda seg, t, kind="segment": real(seg, t, kind) + (["moved"] if kind == "embed" else []))
    assert client.get("/v1/embeddings").json()["embeddings"] == []
    assert client.get(PATH).status_code == 404
    ex.close()


def test_the_listing_needs_the_token(tmp_path, monkeypatch):
    enc = FakeEncoder()
    ex = _idc_executor(tmp_path, monkeypatch, embed_fn=enc)
    client = TestClient(create_app(ex, token="secret"))
    assert client.get("/v1/embeddings").status_code == 401
    assert client.get("/v1/embeddings", headers={"Authorization": "Bearer secret"}).status_code == 200
    ex.close()


# -- the path ---------------------------------------------------------------------------

def test_the_path_serves_the_jobs_bytes_and_head_says_what_get_says(tmp_path, monkeypatch):
    _, ex, client = _server(tmp_path, monkeypatch)
    s = _embed_idc(client)
    job = client.get(f"/v1/jobs/{s['id']}/result")
    got = client.get(PATH)
    assert got.status_code == 200 and got.content == job.content
    assert got.headers["content-type"] == "application/zip"
    assert got.headers["etag"].startswith('"sha256:')
    head = client.head(PATH)
    assert head.status_code == 200 and head.content == b""
    assert head.headers["etag"] == got.headers["etag"]
    assert int(head.headers["content-length"]) == len(got.content)
    again = client.get(PATH, headers={"If-None-Match": got.headers["etag"]})
    assert again.status_code == 304 and again.headers["etag"] == got.headers["etag"]
    ex.close()


def test_int8_has_its_own_path(tmp_path, monkeypatch):
    enc, ex, client = _server(tmp_path, monkeypatch)
    _embed_idc(client, options={"int8": True})
    int8 = PATH.replace("embedding.zarr.zip", "embedding_int8.zarr.zip")
    assert client.get(int8).status_code == 200
    assert client.get(PATH).status_code == 404          # the default was never computed
    row = client.get("/v1/embeddings").json()["embeddings"][0]
    assert row["options"] == {"int8": True} and row["links"] == {"embedding": int8}
    ex.close()


def test_int8_left_off_is_the_default_and_one_result(tmp_path, monkeypatch):
    """`{"int8": false}` and `{}` are the same bytes: they were two keys, and the path - which
    asks for the default by `{}` - could never find the first."""
    enc, ex, client = _server(tmp_path, monkeypatch)
    a = _embed_idc(client, options={"int8": False})
    b = _embed_idc(client)
    assert a["key"] == b["key"] and len(enc.calls) == 1
    assert client.get(PATH).status_code == 200
    ex.close()


def test_a_miss_is_a_404_no_cache_keeps_and_computes_nothing(tmp_path, monkeypatch):
    enc, ex, client = _server(tmp_path, monkeypatch)
    other = f"/v1/idc/11111111-2222-3333-4444-555555555555/radar:pretrain/embedding.zarr.zip"
    for r in (client.get(other), client.head(other), client.get(other, headers={"Prefer": "wait=5"})):
        assert r.status_code == 404 and r.headers.get("cache-control") == "no-store"
    assert "kind=embed" in client.get(other).json()["detail"]
    assert enc.calls == []                               # a path never starts a job
    assert client.get(f"/v1/idc/{U}/nope/embedding.zarr.zip").status_code == 404
    ex.close()


def test_a_path_names_an_encoder_not_a_task(tmp_path, monkeypatch):
    """`ts.v2:total_fast` is a task and an encoder: its label map is not its embedding."""
    _, ex, client = _server(tmp_path, monkeypatch)
    _embed_idc(client)                                    # radar's, not total_fast's
    assert client.get(f"/v1/idc/{U}/ts.v2:total_fast/embedding.zarr.zip").status_code == 404
    ex.close()


def test_a_stale_view_is_a_503_never_a_404(tmp_path, monkeypatch):
    """On Modal a refused reload leaves an old view, and a miss read from it is not believed."""
    enc = FakeEncoder()
    ex = _idc_executor(tmp_path, monkeypatch, embed_fn=enc)

    def stale(key, since):
        raise ResultsNotVisible("no newer view")
    ex.confirm_absent = stale
    client = TestClient(create_app(ex))
    assert client.get(PATH).status_code == 503
    ex.close()


def test_the_anonymous_twin_answers_the_same_path_and_listing(tmp_path, monkeypatch):
    enc, ex, client = _server(tmp_path, monkeypatch)
    s = _embed_idc(client)
    seg = ex.segmenter

    def weights_fn(task, kind="segment"):
        return weights_versions_of(seg, task) if kind == "segment" else versions_for(seg, task, kind)

    def key_fn(identity, task, opts=None):
        ids = (identity,) if isinstance(identity, str) else tuple(identity)
        return result_key(ids, task, opts or {}, weights_fn(task))
    twin = TestClient(create_public_app(key_fn, ex.cache.get, seg.tasks, weights_fn=weights_fn,
                                        list_fn=ex.cache.list))
    got = twin.get(PATH)
    assert got.status_code == 200 and got.content == client.get(PATH).content
    assert [r["key"] for r in twin.get("/v1/embeddings").json()["embeddings"]] == [s["key"]]
    assert twin.post("/v1/jobs", data={"task": "radar:pretrain", "kind": "embed"}).status_code in (401, 404, 405)
    ex.close()


def test_the_modal_executor_keys_an_embedding_with_the_kind(monkeypatch):
    pytest.importorskip("modal")
    from test_embed_modal import _executor
    _, _, ex, _ = _executor(monkeypatch)
    assert ex.weights_versions("radar:pretrain", kind="embed") == ["embed-versions"]
    assert ex.weights_versions("ts.v2:total_fast") == ["segment-versions"]


def test_the_listing_is_documented(tmp_path):
    """The routes are in the OpenAPI document, which is what a client generator reads."""
    _, _, ex, client = make(tmp_path)
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/embeddings" in paths
    assert any(p.endswith("/embedding.zarr.zip") for p in paths)
    ex.close()


def test_links_are_written_in_one_place():
    assert serve_mod.embedding_links("radar:pretrain", [f"idc:{U}"], {}) == {"embedding": PATH}
    assert serve_mod.embedding_links("radar:pretrain", ["sha256:" + "0" * 64], {}) == {}
    assert serve_mod.embedding_links("radar:pretrain", ["idc:a", "idc:b"], {}) == {}
    assert serve_mod.embedding_links("radar:pretrain", [f"idc:{U}"], {"int8": "yes"}) == {}
    assert isinstance(ResultsNotVisible("x"), HaversackError)
