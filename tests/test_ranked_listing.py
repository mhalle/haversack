"""Ranked stores are listed (2026-09-25): ``GET /v1/ranked``.

The third listing over the machinery ``/v1/segmentations`` and ``/v1/embeddings`` share: the
same cache scan, cursor and identity filter, a row kept only when its ``meta.json`` says it is a
ranked store, and a key round trip through the task's store versions (its weights plus the
formats a store is written in). A row with a path carries ``links.ranked``, the read-only door
``test_ranked_jobs`` covers. Driven through the local server with the store double and a fake
IDC fetch; the anonymous twin over the same cache must list the same rows.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("duckn")
pytest.importorskip("rankfield")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import ranked_output  # noqa: E402
from haversack.serve import (RANKED_NAME, create_app, create_public_app, result_key,  # noqa: E402
                             versions_for, weights_versions_of)

from test_ranked_jobs import IDC, FakeStore, _client, _idc, _submit_idc, make, post  # noqa: E402
from test_serve import submit, wait_state  # noqa: E402

PATH = f"/v1/idc/{IDC}/total_fast/{RANKED_NAME}"


def _rows(client, **params):
    r = client.get("/v1/ranked", params=params)
    assert r.status_code == 200, r.text
    return r.json()["ranked"]


def test_a_hosted_store_is_listed_with_its_path_and_nothing_else_is(tmp_path, monkeypatch):
    _, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)
    wait_state(client, submit(client, task="total_fast"), ("done",))   # labels beside it
    body = client.get("/v1/ranked").json()
    assert body["next_cursor"] is None
    rows = body["ranked"]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["key"] == s["key"] and row["task"] == s["task"]
    assert row["identity"] == [f"idc:{IDC}"] and row["bytes"] > 0 and row["published"]
    assert row["links"] == {"ranked": f"/v1/idc/{IDC}/{s['task']}/{RANKED_NAME}"}
    assert client.get(row["links"]["ranked"]).status_code == 200
    # and neither of the other listings takes it
    assert all(r["key"] != s["key"] for r in client.get("/v1/segmentations").json()["segmentations"])
    ex.close()


def test_an_uploads_store_is_listed_without_a_path(tmp_path, monkeypatch):
    _, _, ex, client = make(tmp_path, monkeypatch)
    s = wait_state(client, post(client).json()["id"], ("done",))
    rows = _rows(client)
    assert [r["key"] for r in rows] == [s["key"]]
    assert "links" not in rows[0] and rows[0]["identity"][0].startswith("sha256")
    ex.close()


def test_the_filters_find_by_identity_and_task(tmp_path, monkeypatch):
    _, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)

    def keys(**params):
        return [e["key"] for e in _rows(client, **params)]
    assert keys(identity=f"idc:{IDC}") == [s["key"]]
    assert keys(identity=f"idc:{IDC.upper()}") == [s["key"]]           # normalized as the path is
    assert keys(identity="idc:11111111-2222-3333-4444-555555555555") == []
    assert keys(task="total_fast") == [s["key"]]
    assert keys(task="total") == []
    assert keys(identity=f"idc:{IDC}", task="total_fast") == [s["key"]]
    assert client.get("/v1/ranked", params={"task": "nope"}).status_code == 422
    ex.close()


def test_a_page_at_a_time_by_cursor(tmp_path, monkeypatch):
    _, _, ex, client = make(tmp_path, monkeypatch)
    a = wait_state(client, post(client, fill=0).json()["id"], ("done",))["key"]
    b = wait_state(client, post(client, fill=1).json()["id"], ("done",))["key"]
    first = client.get("/v1/ranked", params={"limit": 1}).json()
    second = client.get("/v1/ranked", params={"limit": 1, "cursor": first["next_cursor"]}).json()
    assert {first["ranked"][0]["key"], second["ranked"][0]["key"]} == {a, b}
    assert client.get("/v1/ranked", params={"cursor": "garbage"}).status_code == 422
    assert client.get("/v1/ranked", params={"limit": 0}).status_code == 422
    ex.close()


def test_labels_of_a_task_are_not_a_store_of_it(tmp_path, monkeypatch):
    """Only the row's KIND keeps a segmentation of the same task out. Its task passes the task
    check, and the key round trip fails OPEN - a key that cannot be derived is not held against
    a row - so it is made to fail here, as the embeddings listing's test does (a first version
    of this test passed with the kind check deleted)."""
    from haversack import serve as serve_mod
    _, _, ex, client = make(tmp_path, monkeypatch)
    wait_state(client, post(client, kind="segment").json()["id"], ("done",))

    def underivable(*a, **k):
        raise RuntimeError("no key today")
    monkeypatch.setattr(serve_mod, "result_key", underivable)
    assert _rows(client) == []
    ex.close()


def test_a_store_written_in_other_formats_is_not_offered(tmp_path, monkeypatch):
    """The key round trip: once the formats a store is written in move (a rankfield or duckn
    format, a composition rule), a store keyed under the old ones would 404 at its own link -
    so it is not listed."""
    _, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)
    assert len(_rows(client)) == 1
    real = ranked_output.ranked_tag
    monkeypatch.setattr(ranked_output, "ranked_tag", lambda: real() + "-moved")
    assert _rows(client) == []
    assert client.get(f"/v1/idc/{IDC}/{s['task']}/{RANKED_NAME}").status_code == 404
    ex.close()


def test_the_listing_needs_the_token(tmp_path, monkeypatch):
    monkeypatch.setattr(ranked_output, "segment_to_store", FakeStore())
    _, _, ex, _ = make(tmp_path, monkeypatch)
    client = TestClient(create_app(ex, token="secret"))
    assert client.get("/v1/ranked").status_code == 401
    assert client.get("/v1/ranked", headers={"Authorization": "Bearer secret"}).status_code == 200
    ex.close()


def test_the_anonymous_twin_lists_the_same_stores(tmp_path, monkeypatch):
    _, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)
    seg = ex.segmenter

    def weights_fn(task, kind="segment"):
        return weights_versions_of(seg, task) if kind == "segment" else versions_for(seg, task, kind)

    def key_fn(identity, task, opts=None):
        ids = (identity,) if isinstance(identity, str) else tuple(identity)
        return result_key(ids, task, opts or {}, weights_fn(task))
    twin = TestClient(create_public_app(key_fn, ex.cache.get, seg.tasks, weights_fn=weights_fn,
                                        list_fn=ex.cache.list))
    rows = twin.get("/v1/ranked").json()["ranked"]
    assert [r["key"] for r in rows] == [s["key"]]
    assert twin.get(rows[0]["links"]["ranked"]).content == client.get(rows[0]["links"]["ranked"]).content
    ex.close()


def test_the_client_and_the_command_line_list_stores(tmp_path, monkeypatch, capsys):
    from haversack import cli
    from haversack import client as client_mod
    _, ex, client = _idc(tmp_path, monkeypatch)
    s = _submit_idc(client)
    rc, _ = _client(client)
    assert [r["key"] for r in rc.iter_ranked_stores(page_size=1)] == [s["key"]]
    monkeypatch.setattr(client_mod, "RemoteClient", lambda *a, **k: rc)
    assert cli.main(["remote", "--server", "http://testserver", "--token", "t", "ranked"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.split("\t")[1:] == [s["task"], f"idc:{IDC}", f"/v1/idc/{IDC}/{s['task']}/{RANKED_NAME}"]
    ex.close()


def test_the_listing_is_documented(tmp_path, monkeypatch):
    _, _, ex, client = make(tmp_path, monkeypatch)
    assert "/v1/ranked" in client.get("/openapi.json").json()["paths"]
    ex.close()
