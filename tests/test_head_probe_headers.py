"""HEAD on a result path describes the representation GET would send.

Found 2026-09-20 by a probe on a writable cache: ``GET .../labels.seg.nrrd`` answered
``ETag: "sha256:<content digest>"`` and ``HEAD`` on the same path ``ETag: "<first 32 hex
of the result key>"``, so a HEAD carrying GET's tag in ``If-None-Match`` answered 200. The
probe route built its headers without the entry's result, and ``etag_of`` then falls back
to the key - though both of the probe's lookups hand the result back. The same 200 said
``Content-Length: 0`` (Starlette's, from the empty body) where GET says the file's size,
which RFC 9110 8.6 forbids.

Why it matters: SERVER.md offers HEAD as the compute-free probe of the same resource, the
validator became the content digest so that a weights bump which leaves the bytes alone
forces no re-download, and a shared cache that forwards a HEAD marks its stored GET stale
when the ETag or the Content-Length differs (RFC 9111 4.3.5) - on ``Cache-Control: public``
responses.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve as serve_mod  # noqa: E402
from haversack.serve import RESULT_NAME, LocalExecutor, create_app  # noqa: E402

from test_job_result_cache import _NrrdSeg, _Segmenter  # noqa: E402
from test_serve import volume_bytes  # noqa: E402
from test_stale_volume_view import PATH, U, _commit_elsewhere, _key, _modal  # noqa: E402


def _computed_by_path(tmp_path, monkeypatch):
    """A path result computed through ``Prefer: wait``, as a client makes one."""
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(volume_bytes())
        return d

    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w",
                       cache_dir=tmp_path / "c", fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    assert client.get(PATH, headers={"Prefer": "wait=30"}).status_code == 200
    return ex, client


def _agrees_with_get(head, get):
    """The validator and the length, and nothing HEAD says that GET would not."""
    assert head.status_code == 200 and get.status_code == 200
    assert get.headers["etag"].startswith('"sha256:'), get.headers["etag"]
    assert head.headers["etag"] == get.headers["etag"]
    assert head.headers["content-length"] == get.headers["content-length"]
    assert int(get.headers["content-length"]) == len(get.content) > 0
    differing = {k: (v, get.headers.get(k)) for k, v in head.headers.items()
                 if get.headers.get(k) != v}
    assert not differing, differing
    assert not head.content


def test_head_answers_with_gets_validator_and_length(tmp_path, monkeypatch):
    """The cache_get branch. Before 2026-09-21 HEAD's ETag was the key's first 32 hex
    characters and its Content-Length 0."""
    ex, client = _computed_by_path(tmp_path, monkeypatch)
    _agrees_with_get(client.head(PATH), client.get(PATH))
    ex.close()


def test_a_conditional_head_is_a_304_where_a_conditional_get_is(tmp_path, monkeypatch):
    """RFC 9110 13.1.2 names GET and HEAD together for the 304. The 2026-09-20 probe
    sent GET's tag and got 200: one request, two answers by verb."""
    ex, client = _computed_by_path(tmp_path, monkeypatch)
    etag = client.get(PATH).headers["etag"]
    for tag in (etag, f'"other", {etag}', "*"):
        head = client.head(PATH, headers={"If-None-Match": tag})
        get = client.get(PATH, headers={"If-None-Match": tag})
        assert head.status_code == get.status_code == 304, tag
        assert head.headers["etag"] == get.headers["etag"] == etag
    stale = client.head(PATH, headers={"If-None-Match": '"sha256:0"'})
    assert stale.status_code == 200 and stale.headers["etag"] == etag
    ex.close()


def test_a_tag_selects_nothing_while_there_is_no_representation(tmp_path):
    """Only the 200 is conditional, as on GET: an absence is still a 404."""
    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    client = TestClient(create_app(ex))
    assert client.head(PATH, headers={"If-None-Match": "*"}).status_code == 404
    ex.close()


def test_the_branch_that_confirms_a_stale_miss_agrees_with_get_too(monkeypatch, tmp_path):
    """On Modal the first lookup can miss from a stale view and the confirming reload find
    the entry: the probe's second 200, which dropped the result exactly as the first did."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    _commit_elsewhere(m, vol, tmp_path, _key(ex))
    head = client.head(PATH)
    assert vol.reloads == 2                        # refused, then the confirm took
    _agrees_with_get(head, client.get(PATH))
    vol.refusals, etag = 1, head.headers["etag"]
    monkeypatch.setattr(m, "_cache_view_as_of", float("-inf"))
    Path(m.CACHE_ROOT).rename(vol.hidden)          # out of view again: the same branch
    assert client.head(PATH, headers={"If-None-Match": etag}).status_code == 304


def test_the_public_twin_probe_agrees_with_its_get(tmp_path):
    """The twin is create_app's own routes, and it is the face a shared cache sits on."""
    from haversack.content import digest_file
    from haversack.serve import ResultCache, create_public_app, result_key
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, [])
    cache = ResultCache(tmp_path / "c")
    src = tmp_path / RESULT_NAME
    _NrrdSeg().save(src)
    cache.put(key_fn(f"idc:{U}", "total_fast"), src,
              {"outputs": [{"name": "labels", "sha256": digest_file(src)}]}, {})
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"]))
    _agrees_with_get(client.head(PATH), client.get(PATH))


def test_an_entry_with_no_digest_still_agrees(tmp_path):
    """``etag_of`` falls back to the key for a legacy or unreadable result: both verbs
    fall back together."""
    from haversack.serve import ResultCache, create_public_app, result_key
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, [])
    key = key_fn(f"idc:{U}", "total_fast")
    cache = ResultCache(tmp_path / "c")
    src = tmp_path / RESULT_NAME
    _NrrdSeg().save(src)
    cache.put(key, src, {"names": {"1": "spleen"}}, {})
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"]))
    head, get = client.head(PATH), client.get(PATH)
    assert head.headers["etag"] == get.headers["etag"] == f'"{key[:32]}"'
    assert head.headers["content-length"] == get.headers["content-length"]


def test_a_file_that_left_since_the_lookup_gets_no_length_rather_than_a_wrong_one(
        tmp_path, monkeypatch):
    """The lookup decides the status, as it always did; the length is the file's or is
    not sent. ``Content-Length: 0`` is what RFC 9110 8.6 forbids."""
    ex, client = _computed_by_path(tmp_path, monkeypatch)
    etag = client.get(PATH).headers["etag"]
    real = ex.cache_get
    monkeypatch.setattr(ex, "cache_get",
                        lambda key: (tmp_path / "gone" / RESULT_NAME, real(key)[1]))
    head = client.head(PATH)
    assert head.status_code == 200 and head.headers["etag"] == etag
    assert "content-length" not in head.headers
    ex.close()
