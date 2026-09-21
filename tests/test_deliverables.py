"""What is rendered beside a result is the REQUEST's to say (2026-09-20).

`preview` and `statistics` were a deployment setting: every job rendered the deployment's
set. ``deliverables`` on ``POST /v1/jobs`` makes it a per-request list, with the
deployment's set as the default and the ceiling (docs/result-references.md, "Beside it: a
per-request list of light deliverables"). One test per rule, each written against the
mutant it has to kill:

1. a request names its list; absent is the deployment's set; a name this server does not
   render, or has never heard of, is refused at submit with what it offers;
2. a deliverable NEVER enters the labels' result key - same key, same hit, same ETag;
3. a cache hit still honors the list: what the stored generation lacks is rendered then,
   through the artifact path and its single flight, and what cannot be is SAID;
4. ``links`` advertise only what was asked for and is, or will be, there.

The segmenter double writes a real ``.seg.nrrd`` over a real CT-like volume, so where a
test lets the renderers run they run for real; where a test counts or gates a render it
replaces the renderer, which ``artifact_overlap`` looks up at call time.
"""
from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
sitk = pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import jobpolicy, schemas  # noqa: E402
from haversack import serve as serve_mod  # noqa: E402
from haversack.errors import RequestError  # noqa: E402
from haversack.jobpolicy import DELIVERABLES  # noqa: E402
from haversack.serve import LocalExecutor, create_app  # noqa: E402

from test_serve import FakeSegmenter, volume_bytes, wait_artifact, wait_state  # noqa: E402

U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
BASE = f"/v1/idc/{U}/total_fast"
IDC = [{"kind": "idc", "crdc_series_uuid": U}]
SHAPE = (12, 16, 16)


class _Labels:
    """What a compute returns: a save that writes a label map the renderers can read."""
    schema = types.SimpleNamespace(names={1: "blob"})
    provenance = {}

    def volumes_ml(self):
        return {"blob": 1.0}

    def save(self, path):
        a = np.zeros(SHAPE, np.uint8)
        a[3:9, 4:12, 4:12] = 1
        img = sitk.GetImageFromArray(a)
        img.SetMetaData("Segment0_Name", "blob")
        img.SetMetaData("Segment0_LabelValue", "1")
        img.SetMetaData("Segment0_Color", "0.9 0.3 0.2")
        sitk.WriteImage(img, str(path))
        return path


class _Segmenter(FakeSegmenter):
    def segment(self, image, task, *, progress=None, cancel=None, **options):
        super().segment(image, task, progress=progress, cancel=cancel, **options)
        return _Labels()


def _server(tmp_path, monkeypatch, *, artifacts=("preview", "statistics"), token=None,
            gate=None):
    """A local server whose `idc:` fetch writes a CT-like volume and counts itself."""
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    fetches = []

    def fetch(series, jobdir):
        fetches.append(series)
        d = jobdir / "series"
        d.mkdir(parents=True, exist_ok=True)
        a = np.full(SHAPE, -1000, np.int16)
        a[2:10, 3:13, 3:13] = 40
        sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
        return d

    seg = _Segmenter(gate=gate, steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "work", cache_dir=tmp_path / "rc",
                       fetch_idc_fn=fetch, artifacts=artifacts)
    return seg, ex, TestClient(create_app(ex, token=token)), fetches


def _post(client, *, deliverables="absent", source=IDC, fill=None, options=None,
          headers=None, expect=202):
    data = {"task": "total_fast", "options": json.dumps(options or {})}
    files = None
    if source is not None:
        data["source"] = json.dumps(source)
    else:
        files = {"file": ("scan.nii.gz", volume_bytes(fill or 0))}
    if deliverables != "absent":
        data["deliverables"] = deliverables if isinstance(deliverables, str) \
            else json.dumps(deliverables)
    r = client.post("/v1/jobs", data=data, files=files, headers=headers or {})
    assert r.status_code == expect, r.text
    posted = r.json()
    if expect != 202:
        return posted
    # `links` are on the job's status, not on the submit's answer - but what a job
    # delivers, and what a hit could not, is said by the FIRST answer already
    status = client.get(f"/v1/jobs/{posted['id']}", headers=headers or {}).json()
    if posted["state"] == "done":
        for k in ("cached", "deliverables", "deliverables_unavailable"):
            assert posted.get(k) == status.get(k), (k, posted, status)
    return status


def _quiet(ex, key, timeout=5.0):
    """Wait out a render of ``key`` (the overlap runs after `done`)."""
    t0 = time.time()
    while ex.artifact_state(key) == "pending" and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert ex.artifact_state(key) == "absent"


def _generation_files(ex, key) -> set:
    return {p.name for p in Path(ex.cache.get(key)[0]).parent.iterdir()
            if not p.name.startswith(".")}


@pytest.fixture
def renders(monkeypatch):
    """Replace the two renderers with counted, gateable ones that write small files;
    count the pair loads too. ``renders.gate`` holds every render until it is set."""
    import haversack.preview as preview
    import haversack.statistics as statistics
    log = types.SimpleNamespace(pairs=0, made=[], gate=threading.Event())
    log.gate.set()
    real_pair = preview.load_oriented_pair

    def pair(image, labels):
        log.pairs += 1
        return real_pair(image, labels)

    def render_preview(image, labels, out, *, title=None, pair=None, **kw):
        log.gate.wait(10)
        log.made.append("preview")
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n" + b"p" * 32)
        return Path(out)

    def compute_statistics(image, labels, out, *, pair=None, **kw):
        log.gate.wait(10)
        log.made.append("statistics")
        Path(out).write_text(json.dumps({"structures": [{"structure": "blob"}]}))
        return Path(out)

    monkeypatch.setattr(preview, "load_oriented_pair", pair)
    monkeypatch.setattr(preview, "render_preview", render_preview)
    monkeypatch.setattr(statistics, "compute_statistics", compute_statistics)
    return log


# -- 1. the request's list: default, ceiling, refusals ---------------------------------

def test_a_request_that_names_no_list_gets_the_deployments_set(tmp_path, monkeypatch, renders):
    """Absent is exactly what every job did before the field existed."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    s = wait_state(client, _post(client)["id"], ("done",))
    assert s["deliverables"] == ["preview", "statistics"]
    assert s["links"]["preview"] == f"{BASE}/preview.png"
    assert s["links"]["statistics"] == f"{BASE}/statistics.tsv"
    assert wait_artifact(client, f"{BASE}/preview.png").status_code == 200
    assert wait_artifact(client, f"{BASE}/statistics.json").status_code == 200
    assert sorted(renders.made) == ["preview", "statistics"]
    assert client.get("/v1/health").json()["deliverables"] == ["preview", "statistics"]


def test_an_empty_list_renders_nothing_and_does_not_even_load_the_pair(tmp_path, monkeypatch,
                                                                         renders):
    """"Preview off": no render, no pair load, no pending marker, no link - and the
    labels and their metadata are all still there."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    s = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    assert s["deliverables"] == []
    assert set(s["links"]) == {"self", "events", "result", "labels", "meta"}
    _quiet(ex, s["key"])
    assert renders.pairs == 0 and renders.made == []
    assert _generation_files(ex, s["key"]) == {"labels.seg.nrrd", "meta.json", "result.json"}
    assert client.get(f"{BASE}/labels.seg.nrrd").status_code == 200
    # absent, and definitively: nothing is pending, so a poller is never told to wait
    assert client.get(f"{BASE}/preview.png").status_code == 404
    assert client.get(f"{BASE}/statistics.json").status_code == 404


def test_a_list_renders_what_it_names_and_a_declined_artifact_is_absent_at_once(
        tmp_path, monkeypatch, renders):
    """While the statistics are still rendering, a GET of them says 202 - and a GET of
    the preview this job declined says 404, not "wait": that render will never place it."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    renders.gate.clear()                                   # hold the render open
    try:
        s = wait_state(client, _post(client, deliverables=["statistics"])["id"], ("done",))
        assert s["deliverables"] == ["statistics"]
        assert "statistics" in s["links"] and "preview" not in s["links"]
        assert ex.artifact_state(s["key"], "statistics") == "pending"
        assert ex.artifact_state(s["key"], "preview") == "absent"
        assert client.get(f"{BASE}/statistics.json").status_code == 202
        assert client.get(f"{BASE}/preview.png").status_code == 404
    finally:
        renders.gate.set()
    assert wait_artifact(client, f"{BASE}/statistics.json").status_code == 200
    _quiet(ex, s["key"])
    assert renders.made == ["statistics"]
    assert "preview.png" not in _generation_files(ex, s["key"])


def test_a_name_this_server_does_not_render_is_refused_with_what_it_offers(tmp_path,
                                                                           monkeypatch):
    """The deployment's set is the ceiling, and the refusal names the fix in one line."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch, artifacts=("statistics",))
    d = _post(client, deliverables=["preview"], expect=422)["detail"]
    assert d["code"] == "deliverable_not_offered" and d["offered"] == ["statistics"]
    assert "'preview'" in d["message"] and "this server offers statistics" in d["message"]
    assert "\n" not in d["message"]
    assert client.get("/v1/jobs").json()["jobs"] == [] and seg.calls == []
    # what it does offer is accepted, and is all that a default request gets
    assert wait_state(client, _post(client)["id"], ("done",))["deliverables"] == ["statistics"]


def test_an_unknown_name_is_refused_with_what_this_server_offers(tmp_path, monkeypatch):
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    d = _post(client, deliverables=["statistics", "mesh"], expect=422)["detail"]
    assert d["code"] == "unknown_deliverable" and d["deliverable"] == "mesh"
    assert d["offered"] == ["preview", "statistics"]
    assert "this server offers preview, statistics" in d["message"]
    assert client.get("/v1/jobs").json()["jobs"] == [] and seg.calls == []


@pytest.mark.parametrize("sent", ['"preview"', "[1]", '{"preview": true}', '[["preview"]]'])
def test_a_list_that_is_not_a_list_of_names_is_refused(tmp_path, monkeypatch, sent):
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    d = _post(client, deliverables=sent, expect=422)["detail"]
    assert d["code"] == "bad_deliverables" and d["offered"] == ["preview", "statistics"]


def test_a_server_that_renders_nothing_refuses_by_saying_so(tmp_path, monkeypatch):
    seg, ex, client, _ = _server(tmp_path, monkeypatch, artifacts=())
    d = _post(client, deliverables=["preview"], expect=422)["detail"]
    assert d["offered"] == [] and "this server renders none" in d["message"]
    assert wait_state(client, _post(client, deliverables=[])["id"], ("done",))["deliverables"] == []


def test_deliverables_inside_options_is_refused_naming_the_field(tmp_path, monkeypatch):
    """Options are hashed into the result key; the list must never ride in them."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    d = _post(client, options={"deliverables": ["preview"]}, expect=422)["detail"]
    assert d["code"] == "misplaced_deliverables" and "form field" in d["message"]
    assert seg.calls == []


# -- 2. never in the labels' result key --------------------------------------------------

def test_the_list_never_enters_the_result_key(tmp_path, monkeypatch, renders):
    """Declining a preview and then asking for one is ONE result: the same key, a cache
    hit, no second segmentation, the same labels ETag - and nothing `result_key` is ever
    handed knows a deliverable's name."""
    keyed = []
    real = serve_mod.result_key

    def result_key(identity, task, options, weights_versions, epoch=None):
        keyed.append(json.dumps([list(identity), task, options, list(weights_versions)]))
        return real(identity, task, options, weights_versions, epoch)
    monkeypatch.setattr(serve_mod, "result_key", result_key)
    seg, ex, client, _ = _server(tmp_path, monkeypatch)

    first = wait_state(client, _post(client, deliverables=[], source=None, fill=7)["id"],
                       ("done",))
    etag = client.get(f"/v1/jobs/{first['id']}/result").headers["etag"]
    again = _post(client, deliverables=["preview", "statistics"], source=None, fill=7)
    assert again["state"] == "done" and again["cached"] is True
    assert again["key"] == first["key"]
    assert client.get(f"/v1/jobs/{again['id']}/result").headers["etag"] == etag
    default = _post(client, source=None, fill=7)
    assert default["cached"] is True and default["key"] == first["key"]
    assert len(seg.calls) == 1, "asking for a deliverable recomputed the segmentation"

    # the same for a hosted input, whose labels are addressed by path
    h1 = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    tag = client.get(f"{BASE}/labels.seg.nrrd").headers["etag"]
    h2 = _post(client, deliverables=["statistics"])
    assert h2["cached"] is True and h2["key"] == h1["key"]
    assert client.get(f"{BASE}/labels.seg.nrrd").headers["etag"] == tag
    assert len(seg.calls) == 2

    assert keyed, "the spy saw no key being built"
    leaked = [k for k in keyed if "deliverables" in k or any(n in k for n in DELIVERABLES)]
    assert not leaked, leaked


def test_no_option_may_be_named_deliverables(monkeypatch):
    """The name is the request's own field, and `RemoteClient.submit` takes it as a keyword
    beside `**options` - so an option of that name could neither be sent nor stay out of
    the key. Every shipped parameter model is held to it, and `wire_params` refuses one."""
    from pydantic import Field
    from haversack.engines.registry import ENGINES
    assert schemas.DELIVERABLES_FIELD == "deliverables"
    for name, eng in ENGINES.items():
        model = schemas.wire_params(eng.parameters, eng.processing_knobs)
        assert schemas.DELIVERABLES_FIELD not in model.model_fields, name

    class Sly(schemas.Params):
        deliverables: list[str] | None = Field(None)
    for processing in (True, False):
        with pytest.raises(TypeError, match="may not be named 'deliverables'"):
            schemas.wire_params(Sly, processing)


# -- 3. a cache hit still honors the list -----------------------------------------------

def test_a_cache_hit_renders_what_its_list_names_and_the_result_lacks(tmp_path, monkeypatch):
    """Declined when the result was computed, asked for later: rendered THEN, by the real
    renderer, into the generation that already holds the labels - no recompute, no fetch,
    no new generation."""
    seg, ex, client, fetches = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    key = first["key"]
    gen, labels = ex.cache.generation(key), Path(ex.cache.get(key)[0]).read_bytes()
    assert client.get(f"{BASE}/statistics.json").status_code == 404

    hit = _post(client, deliverables=["statistics"])
    assert hit["state"] == "done" and hit["cached"] is True and hit["key"] == key
    assert hit["deliverables"] == ["statistics"] and "deliverables_unavailable" not in hit
    assert hit["links"]["statistics"] == f"{BASE}/statistics.tsv"
    assert "preview" not in hit["links"]
    r = wait_artifact(client, f"{BASE}/statistics.json")
    assert r.status_code == 200 and r.json()["structures"][0]["structure"] == "blob"
    _quiet(ex, key)
    assert len(seg.calls) == 1 and fetches == [U]
    assert ex.cache.generation(key) == gen
    assert Path(ex.cache.get(key)[0]).read_bytes() == labels
    assert _generation_files(ex, key) == {"labels.seg.nrrd", "meta.json", "result.json",
                                          "statistics.json"}
    # the path surface serves what exists, to anyone; what was never asked for is absent
    assert client.get(f"{BASE}/statistics.tsv").status_code == 200
    assert client.get(f"{BASE}/preview.png").status_code == 404
    listed = next(e for e in client.get("/v1/segmentations").json()["segmentations"]
                  if e["key"] == key)
    assert "statistics" in listed["links"] and "preview" not in listed["links"]


def test_a_hit_on_an_upload_renders_from_the_bytes_just_sent(tmp_path, monkeypatch, renders):
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, deliverables=[], source=None, fill=3)["id"],
                       ("done",))
    hit = _post(client, deliverables=["preview"], source=None, fill=3)
    assert hit["cached"] is True and "deliverables_unavailable" not in hit
    t0 = time.time()
    while "preview.png" not in _generation_files(ex, first["key"]) and time.time() - t0 < 5:
        time.sleep(0.01)
    assert "preview.png" in _generation_files(ex, first["key"])
    assert renders.made == ["preview"] and len(seg.calls) == 1


def test_two_hits_wanting_one_deliverable_render_it_once_through_the_artifact_path(
        tmp_path, monkeypatch, renders):
    """The single flight is the artifact path's own pending marker, and the placement is
    the cache's `add_artifact` into the generation the hit read: a GET during the render
    says 202, as it does after a compute, and a second hit rides the first's render."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    key, gen = first["key"], ex.cache.generation(first["key"])
    placed = []
    real = ex.cache.add_artifact

    def add_artifact(k, name, src, generation=None):
        placed.append((k, name, generation))
        return real(k, name, src, generation=generation)
    monkeypatch.setattr(ex.cache, "add_artifact", add_artifact)
    renders.gate.clear()
    try:
        a = _post(client, deliverables=["statistics"])
        assert ex.artifact_state(key, "statistics") == "pending"
        assert ex._artifacts_pending[key][0] == a["id"]
        assert client.get(f"{BASE}/statistics.json").status_code == 202
        # what a running render will place is on its way whatever has become of the
        # input since: the second hit must not go looking for it, let alone call it gone
        monkeypatch.setattr(ex, "_reference_on_hand", lambda rec: (None, lambda: None))
        b = _post(client, deliverables=["statistics"])
        assert ex._artifacts_pending[key][0] == a["id"], "the second hit took the flight"
        for s in (a, b):
            assert s["cached"] is True and "statistics" in s["links"]
            assert "deliverables_unavailable" not in s
    finally:
        renders.gate.set()
    assert wait_artifact(client, f"{BASE}/statistics.json").status_code == 200
    _quiet(ex, key)
    assert renders.made == ["statistics"] and renders.pairs == 1
    assert placed == [(key, "statistics.json", gen)]


def test_two_hits_racing_for_the_flight_start_one_render(tmp_path, monkeypatch, renders):
    """Hits are answered on request threads, several at once, so the claim is one step:
    both asks here are past "is a render running?" before either claims. One renders; the
    other lets go of the input it had pinned and rides."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    both_looked = threading.Barrier(2, timeout=10)
    real = ex._reference_on_hand

    def reference_on_hand(rec):
        found = real(rec)
        both_looked.wait()                     # neither claims until both have looked
        return found
    monkeypatch.setattr(ex, "_reference_on_hand", reference_on_hand)
    renders.gate.clear()
    out, errors = [], []

    def ask():
        try:
            jid, jdir = ex.new_job_dir()
            out.append(ex.submit(jid, jdir, None, "total_fast", {}, source=IDC,
                                 identity=(f"idc:{U}",), deliverables=("statistics",)))
        except Exception as e:                 # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=ask) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
    finally:
        renders.gate.set()
    assert not errors and len(out) == 2, errors
    assert all(r.cached and r.cache_key == first["key"] for r in out)
    assert all(r.deliverables_unavailable is None for r in out)
    _quiet(ex, first["key"])
    assert renders.made == ["statistics"] and renders.pairs == 1
    assert not ex.series_cache._pins, "the ask that lost the claim kept its pin"


def test_a_hit_whose_input_is_gone_says_so_and_fetches_nothing(tmp_path, monkeypatch, renders):
    """The input would have to be fetched again, so nothing is rendered - and the job SAYS
    so, with the way out, instead of leaving the link off without a word."""
    seg, ex, client, fetches = _server(tmp_path, monkeypatch)
    first = wait_state(client, _post(client, deliverables=[])["id"], ("done",))
    assert ex.series_cache.discard(f"idc:{U}") is True       # evicted since
    hit = _post(client, deliverables=["preview", "statistics"])
    assert hit["state"] == "done" and hit["cached"] is True and hit["key"] == first["key"]
    assert hit["deliverables"] == ["preview", "statistics"]
    why = hit["deliverables_unavailable"]
    assert set(why) == {"preview", "statistics"}
    assert "no longer staged" in why["preview"] and "no-cache" in why["preview"]
    assert "preview" not in hit["links"] and "statistics" not in hit["links"]
    assert hit["links"]["labels"] == f"{BASE}/labels.seg.nrrd"
    assert fetches == [U], "a cache hit fetched its input again to render a deliverable"
    assert ex.artifact_state(first["key"]) == "absent"
    assert renders.pairs == 0 and renders.made == [] and len(seg.calls) == 1
    assert client.get(f"{BASE}/preview.png").status_code == 404
    # the record keeps saying so after it leaves memory
    assert client.get(f"/v1/jobs/{hit['id']}").json()["deliverables_unavailable"] == why
    stored = ex.jobs_db.get(hit["id"])
    assert stored["deliverables"] == ["preview", "statistics"]
    assert stored["deliverables_unavailable"] == why


def test_a_hit_that_wants_more_than_a_running_render_brings_says_which(tmp_path, monkeypatch,
                                                                        renders):
    """One render per result at a time: what the running one will place is on its way,
    and the rest is named as not coming from it."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    renders.gate.clear()
    try:
        first = wait_state(client, _post(client, deliverables=["statistics"])["id"], ("done",))
        hit = _post(client, deliverables=["preview", "statistics"])
        assert hit["cached"] is True
        assert hit["deliverables_unavailable"] == {"preview": jobpolicy.RENDER_BUSY}
        assert "statistics" in hit["links"] and "preview" not in hit["links"]
        assert ex._artifacts_pending[first["key"]][0] == first["id"]
    finally:
        renders.gate.set()
    _quiet(ex, first["key"])
    assert renders.made == ["statistics"]
    # once it has finished, the same ask renders the rest
    later = _post(client, deliverables=["preview", "statistics"])
    assert "deliverables_unavailable" not in later and "preview" in later["links"]
    assert wait_artifact(client, f"{BASE}/preview.png").status_code == 200


def test_a_read_of_a_missing_artifact_never_renders_it(tmp_path, monkeypatch, renders):
    """Decided: a GET serves what exists. Anonymous never computes; an authorized GET of
    a stored result's missing artifact does not either, with Prefer or without - the
    answer is a definitive 404 that names the door which renders it."""
    seg, ex, client, _ = _server(tmp_path, monkeypatch, token="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    done = wait_state(TestClient(create_app(ex)), _post(client, deliverables=[],
                                                        headers=auth)["id"], ("done",))
    for headers in ({}, {"Prefer": "wait=2"}, auth, {**auth, "Prefer": "wait=2"}):
        r = client.get(f"{BASE}/preview.png", headers=headers)
        assert r.status_code == 404, (headers, r.text)
        assert "POST /v1/jobs" in r.json()["detail"] and '["preview"]' in r.json()["detail"]
        assert client.get(f"{BASE}/statistics.tsv", headers=headers).status_code == 404
    assert renders.pairs == 0 and renders.made == [] and len(seg.calls) == 1
    assert ex.artifact_state(done["key"]) == "absent"
    assert client.get(f"{BASE}/labels.seg.nrrd").status_code == 200    # anonymous read


# -- a joined flight, and a restart ------------------------------------------------------

def test_a_submit_that_joins_a_flight_adds_to_its_list_until_publication(tmp_path,
                                                                         monkeypatch, renders):
    gate = threading.Event()
    seg, ex, client, _ = _server(tmp_path, monkeypatch, gate=gate)
    a = _post(client, deliverables=[])
    b = _post(client, deliverables=["statistics"])
    assert b["id"] == a["id"], "the second ask did not join the flight"
    assert client.get(f"/v1/jobs/{a['id']}").json()["deliverables"] == ["statistics"]
    rec = ex.get(a["id"])
    rec.deliverables_sealed = True                  # as publication leaves it
    c = _post(client, deliverables=["preview"])
    assert c["id"] == a["id"] and c["deliverables"] == ["statistics"]
    rec.deliverables_sealed = False
    gate.set()
    s = wait_state(client, a["id"], ("done",))
    assert s["deliverables"] == ["statistics"] and "preview" not in s["links"]
    assert wait_artifact(client, f"{BASE}/statistics.json").status_code == 200
    _quiet(ex, s["key"])
    assert renders.made == ["statistics"] and len(seg.calls) == 1


def test_a_queued_job_keeps_its_list_across_a_restart(tmp_path, monkeypatch, renders):
    """What `jobs.db` persists is the list itself, so a job re-queued by a new process
    still declines what its caller declined; a record from before the list gets the
    deployment's set."""
    gate = threading.Event()
    seg, ex, client, _ = _server(tmp_path, monkeypatch, gate=gate)
    _post(client, source=None, fill=11)                              # holds the dispatcher
    queued = _post(client, deliverables=["statistics"], source=None, fill=12)
    older = _post(client, deliverables=[], source=None, fill=13)
    assert ex.jobs_db.get(queued["id"])["deliverables"] == ["statistics"]
    legacy = ex.jobs_db.get(older["id"])
    legacy.pop("deliverables")                                       # written by 0.12
    ex.jobs_db.put(legacy)
    ex.close()
    gate.set()
    ex2 = LocalExecutor(_Segmenter(gate=threading.Event()), workdir=tmp_path / "work",
                        cache_dir=tmp_path / "rc")
    try:
        assert ex2.get(queued["id"]).deliverables == ("statistics",)
        assert ex2.get(older["id"]).deliverables == ("preview", "statistics")
    finally:
        ex2.close()


# -- one table ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DELIVERABLES))
def test_every_deliverable_of_the_table_is_rendered_under_its_file_name(name, tmp_path,
                                                                        renders):
    """`artifact_overlap` owns the renderers, so it spells the names a second time; this
    holds that spelling, and the file each lands as, to `jobpolicy.DELIVERABLES`."""
    placed, finished = [], []
    serve_mod.artifact_overlap(
        object(), "t", (name,), preview_out=tmp_path / "p.png",
        statistics_out=tmp_path / "s.json",
        place=lambda fname, path: placed.append(fname) or True, finish=finished.append)
    assert placed == [DELIVERABLES[name]]
    assert [n for n, _ in finished[0]] == [name] and renders.made == [name]


def test_the_vocabulary_has_one_home():
    assert serve_mod.ARTIFACT_NAMES == tuple(DELIVERABLES.values())
    assert jobpolicy.wanted_deliverables(None, {"statistics", "preview"}) == tuple(DELIVERABLES)
    assert jobpolicy.wanted_deliverables(["statistics", "preview"], {"preview"}) == ("preview",)
    # the ORDER is the catalog's, never the ask's: a submit that joins a flight hands over a
    # SET of names, so the ask's order there is arbitrary, and the list is persisted and
    # reported. A mutant that kept the ask's order passed every test until 2026-09-21.
    assert jobpolicy.wanted_deliverables(["statistics", "preview"],
                                         {"statistics", "preview"}) == tuple(DELIVERABLES)
    assert jobpolicy.wanted_deliverables({"statistics", "preview"},
                                         {"statistics", "preview"}) == tuple(DELIVERABLES)
    assert jobpolicy.wanted_deliverables([], {"preview"}) == ()
    assert jobpolicy.wanted_deliverables(None, ()) == ()
    assert jobpolicy.pending_covers(None, "preview") is True        # a marker without names
    assert jobpolicy.pending_covers(("statistics",), "preview") is False
    with pytest.raises(RequestError):
        schemas.requested_deliverables(["Preview"], DELIVERABLES)   # names are exact
    assert schemas.requested_deliverables(None, DELIVERABLES) is None
    assert schemas.requested_deliverables(["statistics", "preview", "preview"],
                                          DELIVERABLES) == ("preview", "statistics")


# -- the client and the command line -----------------------------------------------------

def test_the_remote_client_sends_the_list_beside_the_options_never_in_them(tmp_path,
                                                                           monkeypatch, renders):
    from haversack.client import RemoteClient
    seg, ex, client, _ = _server(tmp_path, monkeypatch)
    sent = []
    real = client.request

    def request(method, url, **kw):
        if method == "POST":
            sent.append(dict(kw.get("data") or {}))
        return real(method, url, **kw)
    monkeypatch.setattr(client, "request", request)
    rc = RemoteClient("http://testserver")
    rc._http = client                          # starlette's TestClient is an httpx.Client
    jid = rc.submit(f"idc:{U}", "total_fast", deliverables=["statistics"], interp="nearest")
    assert json.loads(sent[-1]["deliverables"]) == ["statistics"]
    assert json.loads(sent[-1]["options"]) == {"interp": "nearest"}
    final = rc.wait(jid)
    assert final["deliverables"] == ["statistics"] and final["options"] == {"interp": "nearest"}
    rc.submit(f"idc:{U}", "total_fast", interp="nearest")            # no list: nothing sent
    assert "deliverables" not in sent[-1]
    rc.submit(f"idc:{U}", "total_fast", deliverables=[], interp="nearest")
    assert json.loads(sent[-1]["deliverables"]) == []
    assert len({c[2].get("interp") for c in seg.calls}) == 1 and len(seg.calls) == 1


def test_remote_submit_passes_its_deliverables_flag_to_the_client(monkeypatch, capsys):
    from haversack import cli, client as client_mod
    from haversack.errors import InputError
    calls = []

    class Fake:
        def __init__(self, *a, **k):
            pass

        def submit(self, image, task, **kw):
            calls.append(("submit", kw))
            return "abc123"

        def run(self, image, task, output, **kw):
            calls.append(("run", {k: v for k, v in kw.items() if k != "on_status"}))
            return {"state": "done",
                    "deliverables_unavailable": {"preview": "the input is gone"}}
    monkeypatch.setattr(client_mod, "RemoteClient", Fake)
    base = ["remote", "--server", "http://127.0.0.1:9", "--token", "t", "submit",
            "scan.nii.gz", "--task", "ts.v2:total_fast"]
    assert cli.main(base + ["--no-wait"]) == 0
    assert cli.main(base + ["--no-wait", "--deliverables", "none"]) == 0
    assert cli.main(base + ["--no-wait", "--deliverables", "statistics, preview"]) == 0
    assert [kw for _, kw in calls] == [{"deliverables": None}, {"deliverables": []},
                                       {"deliverables": ["statistics", "preview"]}]
    capsys.readouterr()
    assert cli.main(base + ["--deliverables", "preview", "-o", "out.seg.nrrd"]) == 0
    assert calls[-1] == ("run", {"deliverables": ["preview"]})
    assert "note: no preview: the input is gone" in capsys.readouterr().err
    assert cli._deliverables_arg(None) is None
    with pytest.raises(InputError, match='"none"'):
        cli._deliverables_arg(" , ")
