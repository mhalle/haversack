"""A 304 repeats the caching fields its 200 would have carried.

Found 2026-09-21 by a probe, while HEAD's validator was being fixed: a conditional GET of
``/v1/<source>/<identifier>/<task>/labels.seg.nrrd`` answered 304 with the ETag and nothing
else, while the 200 for the same request carries ``Cache-Control: public, max-age=3600`` and
``Vary: Prefer``. RFC 9110 15.4.5: "The server generating a 304 response MUST generate any
of the following header fields that would have been sent in a 200 (OK) response to the same
request: Content-Location, Date, ETag, and Vary; Cache-Control and Expires". A cache keeps
the stored fields a 304 omits (RFC 9111 4.3.4), so nothing was seen to break - this is the
server saying what the standard has it say, on the responses it invites caches to store.

``Preference-Applied`` stays off the 304, deliberately: see ``serve.not_modified``.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import serve as serve_mod  # noqa: E402
from haversack.serve import RESULT_NAME, LocalExecutor, create_app, not_modified  # noqa: E402

from test_job_result_cache import _NrrdSeg  # noqa: E402
from test_serve import FakeSegmenter, make, submit, volume_bytes, wait_state  # noqa: E402

U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
PATH = f"/v1/idc/{U}/total_fast/labels.seg.nrrd"
CACHING = ("cache-control", "vary")


def _computed_by_path(tmp_path, monkeypatch):
    """A path result computed through ``Prefer: wait``; the server is open (no token), so
    the same client may evict it."""
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)

    def fake_fetch(series, jobdir):
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.dcm").write_bytes(volume_bytes())
        return d

    seg = FakeSegmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       fetch_idc_fn=fake_fetch)
    client = TestClient(create_app(ex))
    ok = client.get(PATH, headers={"Prefer": "wait=30"})
    assert ok.status_code == 200 and ok.headers["etag"].startswith('"sha256:')
    return seg, ex, client, ok


def _repeats_the_200s_caching_fields(fresh, ok):
    assert fresh.status_code == 304 and not fresh.content
    assert fresh.headers["etag"] == ok.headers["etag"]
    for name in CACHING:
        assert ok.headers[name], name              # the 200 has it, or this proves nothing
        assert fresh.headers.get(name) == ok.headers[name], name
    # ...and nothing but those: a 304 exists to say little (RFC 9110 15.4.5, SHOULD NOT)
    assert set(fresh.headers) == {"etag", *CACHING}, dict(fresh.headers)


def test_a_conditional_get_of_a_cached_path_result_keeps_cache_control_and_vary(
        tmp_path, monkeypatch):
    """The cache-hit branch. Before 2026-09-21 this 304 carried the ETag alone."""
    _, ex, client, ok = _computed_by_path(tmp_path, monkeypatch)
    fresh = client.get(PATH, headers={"If-None-Match": ok.headers["etag"]})
    _repeats_the_200s_caching_fields(fresh, ok)
    ex.close()


def test_the_304_that_ends_a_wait_keeps_them_too(tmp_path, monkeypatch):
    """The GET route's second 200 site: a client that holds the bytes, asks with ``Prefer:
    wait`` for a result that has to be computed again, and gets the same bytes back."""
    seg, ex, client, ok = _computed_by_path(tmp_path, monkeypatch)
    assert client.delete(PATH).status_code == 200
    fresh = client.get(PATH, headers={"Prefer": "wait=30",
                                      "If-None-Match": ok.headers["etag"]})
    assert len(seg.calls) == 2                     # it was the wait that answered
    _repeats_the_200s_caching_fields(fresh, ok)
    ex.close()


def test_preference_applied_does_not_cross_onto_the_304(tmp_path, monkeypatch):
    """Decided 2026-09-21. A cache writes a 304's fields onto every stored response with
    that validator (RFC 9111 4.3.4) - picked by the ETag, not by the request's Prefer - so
    a ``wait=30`` echo would land on the variant stored for a plain GET."""
    _, ex, client, ok = _computed_by_path(tmp_path, monkeypatch)
    hit = client.get(PATH, headers={"Prefer": "wait=30"})
    assert hit.headers["preference-applied"] == "wait=30"     # the 200 does echo it
    fresh = client.get(PATH, headers={"Prefer": "wait=30",
                                      "If-None-Match": ok.headers["etag"]})
    _repeats_the_200s_caching_fields(fresh, ok)
    ex.close()


def test_whatever_a_conditional_head_answers_says_what_its_200_says(tmp_path, monkeypatch):
    """On main the probe does not evaluate If-None-Match and this is a 200. Commit d7d9f29
    (branch claude/jovial-bassi-3058c8, not landed when this was written) makes it a 304
    through ``not_modified``: merged, its ``found(hit)`` has to hand ``headers`` over as
    GET's two sites do, and this test is what fails until it does."""
    _, ex, client, ok = _computed_by_path(tmp_path, monkeypatch)
    plain = client.head(PATH)
    asked = client.head(PATH, headers={"If-None-Match": "*"})
    assert plain.status_code == 200 and asked.status_code in (200, 304)
    for name in CACHING:
        assert asked.headers.get(name) == plain.headers[name] == ok.headers[name], \
            (name, asked.status_code)
    ex.close()


def test_the_public_twin_is_the_same_route(tmp_path):
    """The anonymous twin is the face a shared cache sits on."""
    from haversack.content import digest_file
    from haversack.serve import ResultCache, create_public_app, result_key
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, [])
    cache = ResultCache(tmp_path / "c")
    src = tmp_path / RESULT_NAME
    _NrrdSeg().save(src)
    cache.put(key_fn(f"idc:{U}", "total_fast"), src,
              {"outputs": [{"name": "labels", "sha256": digest_file(src)}]}, {})
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"]))
    ok = client.get(PATH)
    _repeats_the_200s_caching_fields(
        client.get(PATH, headers={"If-None-Match": ok.headers["etag"]}), ok)


def test_the_job_results_304_is_what_it_was(tmp_path):
    """Its 200 carries no caching field, so there is nothing to repeat: the ETag, alone."""
    _, _, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    ok = client.get(f"/v1/jobs/{jid}/result")
    assert not any(name in ok.headers for name in CACHING)
    fresh = client.get(f"/v1/jobs/{jid}/result", headers={"If-None-Match": ok.headers["etag"]})
    assert fresh.status_code == 304 and dict(fresh.headers) == {"etag": ok.headers["etag"]}


def test_only_the_fields_the_rfc_lists_cross_and_in_any_spelling():
    """The list is RFC 9110 15.4.5's, less Date (the server's own) and the ETag (passed).
    Two of the four no route sends today; a route that starts to gets them repeated."""
    request = types.SimpleNamespace(headers={"if-none-match": '"x"'})
    sent = {"etag": '"stale"', "cache-control": "public", "VARY": "Prefer",
            "Expires": "Thu, 01 Jan 2026 00:00:00 GMT", "Content-Location": "/v1/elsewhere",
            "Preference-Applied": "wait=5", "Content-Length": "96",
            "Content-Disposition": 'attachment; filename="l.seg.nrrd"',
            "Last-Modified": "Thu, 01 Jan 2026 00:00:00 GMT"}
    fresh = not_modified(request, '"x"', sent)
    assert fresh.status_code == 304
    assert dict(fresh.headers) == {"etag": '"x"', "cache-control": "public", "vary": "Prefer",
                                   "expires": "Thu, 01 Jan 2026 00:00:00 GMT",
                                   "content-location": "/v1/elsewhere"}
    # one ETag, the one matched: the 200's own entry, however spelled, is not a second
    assert fresh.headers.getlist("etag") == ['"x"']
    assert not_modified(request, '"y"', sent) is None
    assert dict(not_modified(request, '"x"').headers) == {"etag": '"x"'}
