"""The ARTIFACTS beside a result - ``meta.json``, ``preview.png``, ``statistics.json`` /
``.tsv`` - have a door for every result, a validator that follows their bytes, and a HEAD
(2026-09-21). Three gaps an adversarial review of the deliverables merge reproduced, all
older than it, which per-request deliverables and the conditional HEAD made matter:

1. a result with NO PATH (an upload, a ``result:`` reference, a multi-input job) rendered
   its deliverables into its cache entry and no route served them - now the job's own
   routes do, ``/v1/jobs/<id>/preview.png`` and the rest, authorized and resolved like
   ``/result``;
2. all four artifacts carried one KEY-derived strong ETag, which a ``no-cache``
   republication under the same key did not move though the bytes did - now each tag is
   the digest of the body sent, and ``If-None-Match`` is answered on them;
3. HEAD on every artifact route was a 405 whose ``Allow`` named DELETE (the greedy
   bare-task alias) or GET alone (the twin) - now HEAD answers as GET does, without the
   body, and a 405 lists the methods of the URL asked about.

The matrix enumerates its URLs from the app's ROUTE TABLE, never a hand list: a file route
added later without a HEAD beside it fails here by name. The segmenter double writes a real
``.seg.nrrd`` and the renderers run for real unless a test gates them (``renders``, from
``test_deliverables``).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
sitk = pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import jobpolicy  # noqa: E402
from haversack import serve as serve_mod  # noqa: E402
from haversack.serve import (ARTIFACT_VIEWS, GRID_TOKENS, LocalExecutor, create_app,  # noqa: E402
                             create_public_app, not_modified, result_key,
                             weights_versions_of)

import test_deliverables as td  # noqa: E402
from test_deliverables import BASE, U, _post, _quiet, _server, renders  # noqa: E402,F401
from test_serve import wait_state  # noqa: E402

CACHING = ("cache-control", "vary")
TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


# -- the harness ------------------------------------------------------------------------

def _file_routes(app) -> dict:
    """``{path template: methods}`` for every /v1 route that names a FILE - read from the
    router, which is the point: nothing here knows the artifacts' names."""
    out = {}
    for r in app.routes:
        path = getattr(r, "path", "")
        last = path.rsplit("/", 1)[-1]
        if path.startswith("/v1/") and "." in last and "{" not in last:
            out.setdefault(path, set()).update(getattr(r, "methods", None) or ())
    return out


def _twin_of(seg, ex):
    """The anonymous twin over the SAME cache, keyed as the writer keys."""
    def key_fn(identity, task, opts=None):
        return result_key((identity,), task, opts or {}, weights_versions_of(seg, task))
    return TestClient(create_public_app(key_fn, ex.cache.get, seg.tasks))


def _wait_all(client, urls, headers=None):
    for url in urls:
        r = None
        t0 = time.time()
        while time.time() - t0 < 10:
            r = client.get(url, headers=headers or {})
            if r.status_code == 200:
                break
            time.sleep(0.02)
        assert r is not None and r.status_code == 200, (url, r.status_code, r.text[:200])


def _everything_rendered(tmp_path, monkeypatch):
    """A server holding one hosted result under EVERY option set that has a path (the
    default and each grid token) and one UPLOAD's result, each with both deliverables
    rendered. Returns ``(seg, ex, client, upload job id)``."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    for opts in ({}, *GRID_TOKENS.values()):
        s = wait_state(client, _post(client, options=opts)["id"], ("done",))
        assert s["state"] == "done", s
        _quiet(ex, s["key"])
    up = wait_state(client, _post(client, source=None, fill=7)["id"], ("done",))
    assert up["state"] == "done", up
    _quiet(ex, up["key"])
    return seg, ex, client, up["id"]


def _urls(app, door: str, jid: str) -> list:
    """The file URLs of one door, from the route table, with the parameters filled in."""
    lead = "/v1/jobs/" if door == "job" else "/v1/idc/"
    # an embedding's path names an ENCODER in the task's place: not one of this result's
    # files (tests/test_embeddings_listing.py holds it to GET/HEAD/304 on its own)
    paths = sorted(p for p in _file_routes(app) if p.startswith(lead) and not p.endswith(".zarr.zip"))
    return [p.replace("{ident:path}", U).replace("{task}", "total_fast").replace("{jid}", jid)
            for p in paths]


# -- 3. HEAD exists wherever a file is served --------------------------------------------

def test_every_route_that_names_a_file_answers_head_as_well_as_get(tmp_path, monkeypatch):
    """The tripwire for a route added later: a file served under GET with no HEAD beside
    it fails here, on the api and on the twin (which drops routes after registration)."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    for name, app in (("api", client.app), ("twin", _twin_of(seg, ex).app)):
        routes = _file_routes(app)
        views = {p.rsplit("/", 1)[-1] for p in routes}
        assert set(ARTIFACT_VIEWS) | {"labels.seg.nrrd"} <= views, (name, views)
        # a job's labels are a file too, under a name with no dot in it: the filter
        # above cannot see `/result`, which is how it kept its 405 through 2026-09-21
        for r in app.routes:
            if getattr(r, "path", "").endswith("/result"):
                routes.setdefault(r.path, set()).update(r.methods or ())
        lacking = sorted(p for p, m in routes.items() if "GET" in m and "HEAD" not in m)
        assert lacking == [], f"{name}: served under GET with no HEAD: {lacking}"
        assert (name == "api") == ("/v1/jobs/{jid}/result" in routes)
    assert any(p.startswith("/v1/jobs/") for p in _file_routes(client.app))
    assert not any(p.startswith("/v1/jobs/") for p in _file_routes(_twin_of(seg, ex).app))
    ex.close()


@pytest.mark.parametrize("door", ["api", "twin", "job"])
def test_get_and_head_say_one_thing_in_their_200_and_in_their_304(tmp_path, monkeypatch, door):
    """Every file route x {GET, HEAD} x {200, 304}, at each door: one ETag - a strong
    content digest - one Cache-Control, one Vary; HEAD's Content-Length is GET's body;
    a 304 repeats the caching fields and never ``Preference-Applied``, whatever the
    request preferred; and the weak form of the tag matches (RFC 9110 13.1.2)."""
    seg, ex, api, jid = _everything_rendered(tmp_path, monkeypatch)
    client = _twin_of(seg, ex) if door == "twin" else api
    urls = _urls(client.app, door, jid)
    per_view = 1 if door == "job" else 1 + len(GRID_TOKENS)
    assert len(urls) >= len(ARTIFACT_VIEWS) * per_view, urls   # the enumeration found them
    prefer = {"Prefer": "wait=5"}
    tags = {}
    for url in urls:
        get = client.get(url)
        assert get.status_code == 200, (url, get.status_code, get.text[:200])
        tag = tags[url] = get.headers["etag"]
        assert tag.startswith('"sha256:') and not tag.startswith("W/"), (url, tag)
        for name in CACHING:
            assert get.headers.get(name), (url, name)
        # the echo exists on a 200, or its absence from a 304 would prove nothing
        assert client.get(url, headers=prefer).headers.get("preference-applied") == "wait=5", url

        head = client.head(url, headers=prefer)
        assert head.status_code == 200 and not head.content, (url, head.status_code)
        assert head.headers["content-length"] == str(len(get.content)), url
        if "/labels" not in url:        # the labels' probe leaves Content-Type to its GET,
            assert head.headers["content-type"] == get.headers["content-type"], url   # on purpose
        for name in ("etag", *CACHING):
            assert head.headers.get(name) == get.headers[name], (url, name)

        for cond in (tag, "W/" + tag, f'"other", {tag}', "*"):
            for verb in (client.get, client.head):
                fresh = verb(url, headers={"If-None-Match": cond, **prefer})
                assert fresh.status_code == 304 and not fresh.content, (url, cond, verb.__name__)
                assert set(fresh.headers) == {"etag", *CACHING}, (url, dict(fresh.headers))
                for name in ("etag", *CACHING):
                    assert fresh.headers[name] == get.headers[name], (url, name)
        for verb in (client.get, client.head):         # a tag it does not hold: the 200
            assert verb(url, headers={"If-None-Match": '"sha256:0"'}).status_code == 200, url

    # one validator per representation: the four artifacts used to share the key's
    plain = [u for u in urls if not any(f"_{t}." in u for t in GRID_TOKENS)]
    assert len({tags[u] for u in plain}) == len(plain), {u: tags[u] for u in plain}
    # and who may store it: anonymous public data by path, a token's own through its job
    for url in urls:
        cc = client.get(url).headers["cache-control"]
        assert ("private" in cc and "no-cache" in cc) if door == "job" else "public" in cc, (url, cc)
    ex.close()


def test_head_never_computes_renders_or_waits(tmp_path, monkeypatch, renders):  # noqa: F811
    """HEAD is a read of what exists. Absent: 404, even from a token with ``Prefer`` -
    the GET that computes. Never asked for: 404 at once, not 202, since no render is
    coming. Rendering: 202 + Retry-After, answered at once whatever ``Prefer`` says."""
    seg, ex, client, fetches = _server(tmp_path, monkeypatch)
    for view in ARTIFACT_VIEWS:
        r = client.head(f"{BASE}/{view}", headers={"Prefer": "wait=30"})
        assert r.status_code == 404, (view, r.status_code)
    assert fetches == [] and client.get("/v1/jobs").json()["jobs"] == []

    renders.gate.clear()                                   # hold the render open
    try:
        s = wait_state(client, _post(client, deliverables=["statistics"])["id"], ("done",))
        t0 = time.time()
        for view in ("statistics.json", "statistics.tsv"):
            for url in (f"{BASE}/{view}", f"/v1/jobs/{s['id']}/{view}"):
                r = client.head(url, headers={"Prefer": "wait=30"})
                assert r.status_code == 202 and r.headers.get("retry-after"), (url, r.status_code)
        assert time.time() - t0 < 5, "a HEAD waited on a render"
        for url in (f"{BASE}/preview.png", f"/v1/jobs/{s['id']}/preview.png"):
            assert client.head(url).status_code == 404, url      # declined: not "wait"
            assert client.get(url).status_code == 404, url
        assert client.head(f"{BASE}/meta.json").status_code == 200   # every entry's
    finally:
        renders.gate.set()
    _quiet(ex, s["key"])
    assert renders.made == ["statistics"]                  # and no HEAD rendered a thing
    assert client.head(f"{BASE}/statistics.tsv").status_code == 200
    # `no-cache` starts a recompute on a GET; a HEAD starts nothing, so it means nothing
    r = client.head(f"{BASE}/statistics.tsv",
                    headers={"Cache-Control": "no-cache", "Prefer": "wait=5"})
    assert r.status_code == 200 and "preference-applied" not in r.headers, dict(r.headers)
    assert len(fetches) == 1 and renders.made == ["statistics"]
    ex.close()


def test_the_twin_says_202_for_a_render_still_running_when_it_is_given_the_signal(
        tmp_path, monkeypatch, renders):  # noqa: F811
    """The twin's executor knew nothing of pending renders, so between `done` and the
    artifact landing it called a preview that was seconds away ABSENT - seen on Modal
    (2026-09-21), where the worker commits it after the api already reports done.
    ``artifact_state`` is the writer's read-only signal, like ``inflight``."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)

    def key_fn(identity, task, opts=None):
        return result_key((identity,), task, opts or {}, weights_versions_of(seg, task))
    twin = TestClient(create_public_app(key_fn, ex.cache.get, seg.tasks,
                                        artifact_state=ex.artifact_state))
    renders.gate.clear()
    try:
        s = wait_state(client, _post(client, deliverables=["statistics"])["id"], ("done",))
        for verb in (twin.get, twin.head):
            r = verb(f"{BASE}/statistics.json")
            assert r.status_code == 202 and r.headers.get("retry-after"), r.status_code
            assert verb(f"{BASE}/preview.png").status_code == 404    # declined: still final
    finally:
        renders.gate.set()
    _quiet(ex, s["key"])
    assert twin.head(f"{BASE}/statistics.json").status_code == 200
    ex.close()


def test_an_artifacts_absence_is_never_for_a_cache_to_keep(tmp_path, monkeypatch, renders):  # noqa: F811
    """An artifact arrives late - after `done`, on a later hit, or (with a shared result
    store) from another host into the same generation - so its 404 and its 202 are
    "not yet, as far as this request saw" and say ``no-store``, at both doors, under both
    verbs, beside a 200 that invites shared caches. Nothing here asserts a 404 STAYS one
    for a deliverable that was asked for: it may be a 200 a moment later."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    renders.gate.clear()
    try:
        s = wait_state(client, _post(client, deliverables=["statistics"])["id"], ("done",))
        doors = (BASE, f"/v1/jobs/{s['id']}")
        for verb in (client.get, client.head):
            for door in doors:
                for view, status in (("statistics.json", 202), ("preview.png", 404)):
                    r = verb(f"{door}/{view}")
                    assert r.status_code == status, (door, view, r.status_code)
                    assert r.headers.get("cache-control") == "no-store", (door, view, status)
            other = verb(BASE.replace("total_fast", "total") + "/preview.png")   # no entry at all
            assert other.status_code == 404 and other.headers.get("cache-control") == "no-store"
    finally:
        renders.gate.set()
    _quiet(ex, s["key"])
    ok = client.get(f"{BASE}/statistics.json")             # and the 200 is still the hour's
    assert ok.status_code == 200 and "max-age" in ok.headers["cache-control"]
    ex.close()


def test_not_materialized_is_never_for_a_cache_to_keep(tmp_path, monkeypatch):
    """The labels' own 404: "not materialized" lasts until somebody computes the result,
    and the 200 that follows is ``public, max-age=3600`` - a shared cache that kept the
    404 on a heuristic (RFC 9111 4.2.2) would hide the result behind it. HEAD and GET,
    anonymous and authorized, the default grid and a token, the api and the twin."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch, token=TOKEN)
    twin = _twin_of(seg, ex)
    labels = [p.rsplit("/", 1)[-1] for p in _file_routes(client.app)
              if p.startswith("/v1/idc/") and "/labels" in p]
    assert len(labels) == 1 + len(GRID_TOKENS), labels     # from the router, tokens included
    asks = [(c, verb, hdr) for c, hdrs in ((client, ({}, AUTH)), (twin, ({},)))
            for hdr in hdrs for verb in ("GET", "HEAD")]
    for name in labels:
        for c, verb, hdr in asks:
            r = c.request(verb, f"{BASE}/{name}", headers=hdr)
            assert r.status_code == 404, (name, verb, hdr, r.status_code)
            assert r.headers.get("cache-control") == "no-store", (name, verb, hdr)
    for c, verb, hdr in asks:                              # a task this server does not know
        r = c.request(verb, BASE.replace("total_fast", "nosuchtask") + "/labels.seg.nrrd",
                      headers=hdr)
        assert r.status_code == 404 and r.headers.get("cache-control") == "no-store", (verb, hdr)

    # ...and then it IS computed: the sequence a kept 404 would have hidden
    ok = client.get(f"{BASE}/labels.seg.nrrd", headers={**AUTH, "Prefer": "wait=30"})
    assert ok.status_code == 200 and ok.headers["cache-control"] == "public, max-age=3600"
    for c, verb, hdr in asks:
        r = c.request(verb, f"{BASE}/labels.seg.nrrd", headers=hdr)
        assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=3600"
    ex.close()


def test_head_says_202_while_the_labels_are_still_computing(tmp_path, monkeypatch):
    """What a GET of the same artifact answers then; ``meta.json`` is 404 until the labels
    are published, under either verb, as its GET has always been."""
    gate = threading.Event()
    seg, ex, client, _ = _server(tmp_path, monkeypatch, gate=gate)
    try:
        jid = _post(client)["id"]
        for view in ("preview.png", "statistics.json", "statistics.tsv"):
            r = client.head(f"{BASE}/{view}")
            assert r.status_code == 202 and r.headers.get("retry-after"), (view, r.status_code)
            assert r.headers.get("cache-control") == "no-store", view    # "not yet" is not kept
            assert client.get(f"{BASE}/{view}").status_code == 202, view
        assert client.head(f"{BASE}/meta.json").status_code == 404
        assert client.get(f"{BASE}/meta.json").status_code == 404
        for view in ARTIFACT_VIEWS:                        # the job's door: not done yet
            assert client.head(f"/v1/jobs/{jid}/{view}").status_code == 409, view
    finally:
        gate.set()
    wait_state(client, jid, ("done",))
    ex.close()


def test_a_405_names_the_methods_of_the_url_asked_about(tmp_path, monkeypatch):
    """``Allow`` was the first path-matching route's: DELETE for every artifact URL (the
    greedy bare-task alias took the file name for a task), and one verb where a URL's
    verbs are separate routes. The twin keeps no DELETE, and says so."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    twin = _twin_of(seg, ex)

    def allow(c, method, url):
        r = c.request(method, url)
        assert r.status_code == 405, (method, url, r.status_code)
        return r.headers["allow"]

    for view in (*ARTIFACT_VIEWS, "preview_res-1mm.png"):
        assert allow(client, "POST", f"{BASE}/{view}") == "GET, HEAD", view
        assert allow(twin, "POST", f"{BASE}/{view}") == "GET, HEAD", view
    assert allow(client, "POST", f"{BASE}/labels.seg.nrrd") == "DELETE, GET, HEAD"
    assert allow(twin, "POST", f"{BASE}/labels.seg.nrrd") == "GET, HEAD"
    assert allow(client, "GET", BASE) == "DELETE"          # the bare task: evict only
    assert allow(client, "POST", "/v1/jobs/abc/preview.png") == "GET, HEAD"
    assert allow(client, "PUT", "/v1/jobs/abc") == "DELETE, GET"
    ex.close()


# -- 2. the validator follows the bytes --------------------------------------------------

class _Growing(td._Labels):
    """Labels whose blob grows with every compute: a recompute that came out differently."""
    n = 0

    def save(self, path):
        a = np.zeros(td.SHAPE, np.uint8)
        a[3:9 + 2 * type(self).n, 4:12, 4:12] = 1
        img = sitk.GetImageFromArray(a)
        img.SetMetaData("Segment0_Name", "blob")
        img.SetMetaData("Segment0_LabelValue", "1")
        img.SetMetaData("Segment0_Color", "0.9 0.3 0.2")
        sitk.WriteImage(img, str(path))
        return path


class _GrowingSegmenter(td._Segmenter):
    def segment(self, image, task, **kw):
        super().segment(image, task, **kw)
        return _Growing()


def test_no_url_serves_new_bytes_under_an_old_tag(tmp_path, monkeypatch):
    """A ``no-cache`` recompute republishes under the SAME key and the same URLs. Every
    file whose bytes moved must move its tag, and a client holding the old tag must get
    the new bytes - a 200, never "not modified" about a picture of the old labels."""
    monkeypatch.setattr(td, "_Segmenter", _GrowingSegmenter)
    monkeypatch.setattr(_Growing, "n", 0)
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    urls = [u for u in _urls(client.app, "api", "-")
            if not any(f"_{t}." in u for t in GRID_TOKENS)]
    assert len(urls) == len(ARTIFACT_VIEWS) + 1            # the four, and the labels

    assert client.get(f"{BASE}/labels.seg.nrrd", headers={"Prefer": "wait=30"}).status_code == 200
    _wait_all(client, urls)
    before = {u: client.get(u) for u in urls}
    key = client.get("/v1/segmentations").json()["segmentations"][0]["key"]

    monkeypatch.setattr(_Growing, "n", 1)
    again = client.get(f"{BASE}/labels.seg.nrrd",
                       headers={"Prefer": "wait=30", "Cache-Control": "no-cache"})
    assert again.status_code == 200
    _quiet(ex, key)
    _wait_all(client, urls)
    assert client.get("/v1/segmentations").json()["segmentations"][0]["key"] == key  # same key

    for u in urls:
        old, new = before[u], client.get(u)
        assert new.content != old.content, f"{u}: the recompute did not change it; proves nothing"
        assert new.headers["etag"] != old.headers["etag"], f"{u}: new bytes under the old tag"
        held = client.get(u, headers={"If-None-Match": old.headers["etag"]})
        assert held.status_code == 200 and held.content == new.content, (u, held.status_code)
        assert client.head(u, headers={"If-None-Match": old.headers["etag"]}).status_code == 200
        assert client.head(u).headers["etag"] == new.headers["etag"], u
    ex.close()


def test_if_none_match_is_compared_weakly():
    """RFC 9110 13.1.2: ``W/"x"`` and ``"x"`` match, either way round; other tags do not."""
    import types

    def ask(sent, etag='"sha256:abc"'):
        req = types.SimpleNamespace(headers={"if-none-match": sent})
        return not_modified(req, etag)

    for sent in ('W/"sha256:abc"', '"sha256:abc"', '"x", W/"sha256:abc"', "*"):
        fresh = ask(sent)
        assert fresh is not None and fresh.status_code == 304, sent
        assert fresh.headers["etag"] == '"sha256:abc"'     # ours, as we issue it
    assert ask('"sha256:abc"', etag='W/"sha256:abc"') is not None
    for sent in ('W/"sha256:abd"', 'sha256:abc', 'w/"sha256:abc"', ""):
        assert ask(sent) is None, sent


# -- 1. a result with no path reaches its artifacts through its job ----------------------

def test_an_uploads_job_links_its_deliverables_and_serves_them(tmp_path, monkeypatch):
    """The reviewers' observation, reversed: ``deliverables: [preview, statistics]``,
    files in the generation, and ``links`` = {self, events, result} with no door at all."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    s = wait_state(client, _post(client, source=None, fill=3)["id"], ("done",))
    base = f"/v1/jobs/{s['id']}"
    assert s["deliverables"] == ["preview", "statistics"] and not s.get("deliverables_unavailable")
    assert s["links"] == {"self": base, "events": f"{base}/events", "result": f"{base}/result",
                          "meta": f"{base}/meta.json", "preview": f"{base}/preview.png",
                          "statistics": f"{base}/statistics.tsv"}
    _wait_all(client, [s["links"][k] for k in ("meta", "preview", "statistics")])
    _quiet(ex, s["key"])
    gen = Path(ex.cache.get(s["key"])[0]).parent
    assert client.get(f"{base}/preview.png").content == (gen / "preview.png").read_bytes()
    assert client.get(f"{base}/statistics.json").json() == \
        json.loads((gen / "statistics.json").read_text())
    assert client.get(f"{base}/statistics.tsv").text.startswith("structure\t")
    assert client.get(f"{base}/meta.json").json() == s["result"]
    # the listing is of results, not of jobs: a path-less row stays without links
    rows = client.get("/v1/segmentations").json()["segmentations"]
    assert [r for r in rows if r["key"] == s["key"]] and \
        all("links" not in r for r in rows if r["key"] == s["key"])
    ex.close()


def test_links_name_only_what_the_job_was_asked_for(tmp_path, monkeypatch):
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    s = wait_state(client, _post(client, source=None, fill=4,
                                 deliverables=["statistics"])["id"], ("done",))
    assert set(s["links"]) == {"self", "events", "result", "meta", "statistics"}
    none = wait_state(client, _post(client, source=None, fill=5, deliverables=[])["id"], ("done",))
    assert set(none["links"]) == {"self", "events", "result", "meta"}
    # a hosted result keeps its PATH links - one name, one URL, whichever kind it is
    hosted = wait_state(client, _post(client)["id"], ("done",))
    assert hosted["links"]["preview"] == f"{BASE}/preview.png"
    assert hosted["links"]["meta"] == f"{BASE}/meta.json"
    # ...and its job's door serves it all the same
    _wait_all(client, [f"/v1/jobs/{hosted['id']}/preview.png"])
    ex.close()


def test_a_job_artifact_answers_by_the_result_routes_rules(tmp_path, monkeypatch):
    """404 no such job, 410 once the bytes are gone from both places ``/result`` looks -
    under both verbs, for every view (409, not done, is in the computing test above)."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    s = wait_state(client, _post(client, source=None, fill=6)["id"], ("done",))
    _quiet(ex, s["key"])
    base = f"/v1/jobs/{s['id']}"
    for view in ARTIFACT_VIEWS:
        for verb in (client.get, client.head):
            assert verb(f"/v1/jobs/nosuchjob/{view}").status_code == 404, view
            assert verb(f"{base}/{view}").status_code == 200, view
    got, probe = client.get(f"{base}/result"), client.head(f"{base}/result")
    assert probe.status_code == 200 and probe.content == b""
    for h in ("etag", "content-length", "content-disposition"):
        assert probe.headers[h] == got.headers[h], h
    assert int(probe.headers["content-length"]) == len(got.content)
    assert client.head(f"{base}/result",
                       headers={"If-None-Match": got.headers["etag"]}).status_code == 304
    converted = client.head(f"{base}/result?format=nii.gz")     # converts nothing, so it
    assert converted.status_code == 200                         # cannot know a length
    assert "content-length" not in converted.headers
    assert ex.cache.delete(s["key"])
    shutil.rmtree(ex.get(s["id"]).dir)
    # gone - until the key is computed again with the same output, when these very URLs
    # say 200: so the 410 is no more for a cache to keep than the 404 is
    for view in ("result", *ARTIFACT_VIEWS):
        for verb in (client.get, client.head):
            r = verb(f"{base}/{view}")
            assert r.status_code == 410 and r.headers["cache-control"] == "no-store", view
    ex.close()


def test_only_an_absence_is_marked_no_store(tmp_path, monkeypatch):
    """404 and 410 are; a 409 (not done) is no absence and no cache keeps it on a
    heuristic, and a 200 keeps its own policy."""
    gate = threading.Event()
    seg, ex, client, _ = _server(tmp_path, monkeypatch, gate=gate)
    try:
        s = _post(client, source=None, fill=9)
        for view in ("result", *ARTIFACT_VIEWS):
            for verb in (client.get, client.head):
                r = verb(f"/v1/jobs/{s['id']}/{view}")
                assert r.status_code == 409 and "cache-control" not in r.headers, view
        # and a job that is not done links nothing that is not there yet
        assert not {"result", "meta", "preview", "statistics"} & set(s["links"]), s["links"]
    finally:
        gate.set()
    done = wait_state(client, s["id"], ("done",))
    _quiet(ex, done["key"])
    r = client.get(f"/v1/jobs/{s['id']}/meta.json")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-cache"
    ex.close()


def test_a_head_response_carries_no_body():
    """At unit level because Starlette's TestClient drops a HEAD's body itself, so no
    route test can see one - and under a real server a body on a HEAD is a protocol
    error (mutation pass, 2026-09-21)."""
    import types
    from fastapi.responses import JSONResponse
    resp = JSONResponse({"a": 1})
    req = types.SimpleNamespace(method="HEAD", headers={})
    out = serve_mod.answer_body(req, resp, {"ETag": '"sha256:x"', "Cache-Control": "private"})
    assert out.body == b"" and out.headers["content-length"] == str(len(resp.body))
    assert out.headers["content-type"] == resp.headers["content-type"]


def test_a_jobs_artifacts_are_its_own_not_a_republications(tmp_path, monkeypatch):
    """The digest-equality rule ``/result`` applies: once the key is republished with
    OTHER bytes, the first job's door must not serve the new entry's picture as its own."""
    monkeypatch.setattr(td, "_Segmenter", _GrowingSegmenter)
    monkeypatch.setattr(_Growing, "n", 0)
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, source=None, fill=8)["id"], ("done",))
    _quiet(ex, first["key"])
    views = ("preview.png", "statistics.json", "meta.json")
    _wait_all(client, [f"/v1/jobs/{first['id']}/{v}" for v in views])
    mine = {v: client.get(f"/v1/jobs/{first['id']}/{v}").content for v in views}

    monkeypatch.setattr(_Growing, "n", 1)
    second = wait_state(client, _post(client, source=None, fill=8,
                                      headers={"Cache-Control": "no-cache"})["id"], ("done",))
    assert second["key"] == first["key"] and second["id"] != first["id"]
    _quiet(ex, second["key"])
    _wait_all(client, [f"/v1/jobs/{second['id']}/{v}" for v in views])
    for v in views:
        theirs = client.get(f"/v1/jobs/{second['id']}/{v}").content
        assert theirs != mine[v], f"{v}: the recompute did not change it; proves nothing"
        assert client.get(f"/v1/jobs/{first['id']}/{v}").content == mine[v], v
    # ...nor through the second look, of a newer view: with the first job's own picture
    # gone, a confirming lookup that finds the REPUBLISHED entry has found somebody
    # else's bytes, and the answer is 404 - not their preview (mutation pass, 2026-09-21)
    theirs = client.get(f"/v1/jobs/{second['id']}/preview.png").content
    (ex.get(first["id"]).dir / "preview.png").unlink()
    ex.confirm_absent = lambda key, since: ex.cache.get(key)
    r = TestClient(create_app(ex)).get(f"/v1/jobs/{first['id']}/preview.png")
    assert r.status_code == 404 and r.content != theirs, r.status_code
    ex.close()


def test_a_stale_view_of_the_entry_is_asked_again_before_a_404(tmp_path, monkeypatch, renders):  # noqa: F811
    """On Modal the path in hand is a container-local copy taken at lookup, and the
    lookup's view may predate the worker's commit of an artifact: the job's door confirms
    against a newer view before it calls the artifact absent, as the path's does."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    renders.gate.clear()
    try:
        s = wait_state(client, _post(client, source=None, fill=9)["id"], ("done",))
        real = ex.cache.get(s["key"])
        stale = tmp_path / "stale-copy"
        shutil.copytree(Path(real[0]).parent, stale)       # the generation, artifacts not yet in
        assert not (stale / "preview.png").exists()
    finally:
        renders.gate.set()
    _quiet(ex, s["key"])
    shutil.rmtree(ex.get(s["id"]).dir / "preview.png", ignore_errors=True)

    asked = []
    ex.cache_get = lambda key: (stale / Path(real[0]).name, real[1])
    ex.confirm_absent = lambda key, since: asked.append(key) or ex.cache.get(key)
    client = TestClient(create_app(ex))
    r = client.get(f"/v1/jobs/{s['id']}/preview.png")
    assert r.status_code == 200 and asked == [s["key"]], (r.status_code, asked)

    # and the cheaper re-ask comes first: the entry as it stands NOW, which on Modal is
    # a fresh local copy - no newer view needed when the second lookup already has it
    looks = []
    ex.cache_get = lambda key: (looks.append(key) or len(looks) > 1) and real or \
        (stale / Path(real[0]).name, real[1])
    del ex.confirm_absent
    r = TestClient(create_app(ex)).get(f"/v1/jobs/{s['id']}/preview.png")
    assert r.status_code == 200 and len(looks) >= 2, (r.status_code, looks)
    ex.close()


def test_a_job_with_no_cache_entry_says_what_it_cannot_deliver(tmp_path, monkeypatch):
    """A deliverable is rendered into a result's entry; with no result cache there is
    none, and the job must say so rather than list - and link - what nothing serves."""
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    ex = LocalExecutor(td._Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=None,
                       artifacts=("preview", "statistics"))
    client = TestClient(create_app(ex))
    s = wait_state(client, _post(client, source=None, fill=2)["id"], ("done",))
    assert s["deliverables_unavailable"] == {d: jobpolicy.NO_CACHE_ENTRY
                                             for d in ("preview", "statistics")}
    assert set(s["links"]) == {"self", "events", "result", "meta"}
    r = client.get(f"/v1/jobs/{s['id']}/preview.png")
    assert r.status_code == 404 and "no entry in the result cache" in r.json()["detail"]
    assert client.get(s["links"]["meta"]).status_code == 200
    ex.close()


# -- who may open the job's door -----------------------------------------------------------

def test_a_jobs_artifacts_need_the_token_and_the_twin_has_no_such_door(tmp_path, monkeypatch):
    """A job's preview shows what was uploaded: never the anonymous tier's. This server
    has ONE token and no per-token ownership of jobs, so "another token" is a token that
    is not the server's - refused, like none at all. The twin registers no job route."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch, token=TOKEN)
    s = _post(client, source=None, fill=1, headers=AUTH)
    jid = s["id"]
    for _ in range(500):
        if client.get(f"/v1/jobs/{jid}", headers=AUTH).json()["state"] == "done":
            break
        time.sleep(0.02)
    key = client.get(f"/v1/jobs/{jid}", headers=AUTH).json()["key"]
    _quiet(ex, key)
    twin = _twin_of(seg, ex)
    job_views = [p.rsplit("/", 1)[-1] for p in _file_routes(client.app) if p.startswith("/v1/jobs/")]
    assert sorted(job_views) == sorted(ARTIFACT_VIEWS)
    for view in job_views:
        url = f"/v1/jobs/{jid}/{view}"
        for verb in ("GET", "HEAD"):
            assert client.request(verb, url, headers=AUTH).status_code == 200, (verb, url)
            assert client.request(verb, url).status_code == 401, (verb, url)
            other = {"Authorization": "Bearer someone-elses"}
            assert client.request(verb, url, headers=other).status_code == 401, (verb, url)
            assert twin.request(verb, url).status_code == 404, (verb, url)
            assert twin.request(verb, url, headers=AUTH).status_code == 404, (verb, url)
    # and the token's 200 is not for a shared cache to keep
    cc = client.get(f"/v1/jobs/{jid}/preview.png", headers=AUTH).headers["cache-control"]
    assert "private" in cc and "public" not in cc
    ex.close()
