"""The job result route reads the job's PUBLISHED cache entry, not its scratch copy.

Found on Modal 2026-09-19 (haversack-radar-val, 0.12.2, ts.v2:total on 440 IDC series):
``GET /v1/jobs/{id}/result?format=nii.gz`` answered 500 for 162 jobs - SimpleITK could not
find ``/scratch/<jid>/labels.seg.nrrd`` in the api container - while the same result by
path, ``/v1/idc/<uuid>/ts.v2:total/labels.seg.nrrd``, answered 200 for every one. The by-path
route resolves the key through the leased cache lookup; the job route read the worker's
scratch file. SERVER.md promises 410 for bytes that are gone, never a 500.

Here the scratch copy is removed after the job publishes, which is what the api container
saw: the route must answer from the entry, with and without ``?format``, and 410 once the
entry is gone too.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
sitk = pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack.serve import RESULT_NAME, LocalExecutor, create_app  # noqa: E402

from test_serve import FakeSegmenter, submit, wait_state  # noqa: E402


class _NrrdSeg:
    """A Segmentation double whose saved labels SimpleITK can read, so the format
    conversion runs for real."""

    class _Schema:
        names = {0: "background", 1: "spleen"}

    schema = _Schema()
    provenance = {"device": "fake"}

    def __init__(self, fill=1):
        self.fill = fill

    def volumes_ml(self):
        return {"spleen": 1.0}

    def save(self, path):
        img = sitk.GetImageFromArray(np.full((2, 3, 4), self.fill, np.uint8))
        sitk.WriteImage(img, str(path), True)
        return path


class _Segmenter(FakeSegmenter):
    fill = 1

    def segment(self, image, task, *, progress=None, cancel=None, **options):
        super().segment(image, task, progress=progress, cancel=cancel, **options)
        return _NrrdSeg(self.fill)


def _make(tmp_path):
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c")
    return seg, ex, TestClient(create_app(ex))


def _done_job(client, ex):
    jid = submit(client)
    s = wait_state(client, jid, ("done",))
    assert s.get("key"), "an uploaded input has a content identity, so a cache key"
    assert ex.cache_get(s["key"]) is not None
    return jid, s


def _drop_scratch(ex, jid):
    scratch = ex.get(jid).dir / RESULT_NAME
    assert scratch.exists()
    scratch.unlink()


@pytest.mark.parametrize("fmt", [None, "nii.gz", "nii"])
def test_result_is_served_from_the_published_entry_when_scratch_is_gone(tmp_path, fmt):
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    entry = ex.cache_get(s["key"])[0]
    _drop_scratch(ex, jid)
    url = f"/v1/jobs/{jid}/result" + (f"?format={fmt}" if fmt else "")
    r = client.get(url)
    assert r.status_code == 200, r.text
    if fmt is None:
        assert r.content == Path(entry).read_bytes()
        assert r.headers["etag"] == f'"{s["result"]["outputs"][0]["sha256"]}"'
    ex.close()


def test_the_read_takes_a_lease_on_the_entry(tmp_path, monkeypatch):
    """The route resolves through cache_get - the leased lookup - and not a path it
    remembered: reclamation must know a reader holds the generation."""
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    asked = []
    real = ex.cache_get
    monkeypatch.setattr(ex, "cache_get", lambda key: asked.append(key) or real(key))
    lease = Path(real(s["key"])[0]).parent / ex.cache.LEASE_NAME
    lease.unlink()
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 200
    assert asked == [s["key"]]
    assert lease.exists()
    ex.close()


@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_neither_copy_is_410_not_500(tmp_path, fmt):
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    _drop_scratch(ex, jid)
    assert ex.cache_delete(s["key"])
    url = f"/v1/jobs/{jid}/result" + (f"?format={fmt}" if fmt else "")
    r = client.get(url)
    assert r.status_code == 410, r.text
    ex.close()


def test_scratch_still_serves_where_there_is_no_entry(tmp_path):
    """A server without a result cache has only the job's copy."""
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w")
    client = TestClient(create_app(ex))
    jid = submit(client)
    wait_state(client, jid, ("done",))
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 200
    assert client.get(f"/v1/jobs/{jid}/result?format=nii.gz").status_code == 200
    ex.close()


def test_a_republished_entry_with_other_bytes_is_not_this_jobs_result(tmp_path):
    """The key outlives the job: a no-cache recompute republishes under it. When those
    bytes differ, the job's result is the job's own copy - and gone once that is."""
    seg, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    job_bytes = (ex.get(jid).dir / RESULT_NAME).read_bytes()
    other = tmp_path / "other.seg.nrrd"
    _NrrdSeg(fill=7).save(other)
    ex.cache.put(s["key"], other, {"outputs": [{"name": "labels", "sha256": "f" * 64}]},
                 {"task": s["task"]})
    r = client.get(f"/v1/jobs/{jid}/result")
    assert r.status_code == 200 and r.content == job_bytes
    _drop_scratch(ex, jid)
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 410
    ex.close()


def test_from_job_promotes_the_published_entry(tmp_path):
    """POST /v1/inputs from_job reads the job's result the same way."""
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    _drop_scratch(ex, jid)
    r = client.post("/v1/inputs", data={"from_job": jid})
    assert r.status_code == 200, r.text
    assert r.json()["digest"] == s["result"]["outputs"][0]["sha256"]
    ex.close()


def test_an_evicted_record_whose_entry_survives_still_advertises_its_result(tmp_path):
    """The durable record's result_available looks where the route looks."""
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       keep_finished=1)
    client = TestClient(create_app(ex))
    first = submit(client)
    wait_state(client, first, ("done",))
    for _ in range(2):
        wait_state(client, submit(client), ("done",))
    assert ex.get(first) is None
    s = client.get(f"/v1/jobs/{first}").json()
    assert s["evicted"] is True and s["result_available"] is True, json.dumps(s)
    assert "result" in s["links"]
    assert client.get(f"/v1/jobs/{first}/result").status_code == 200
    ex.close()


# -- the Modal executor: the api container, where the defect was seen ----------------

def _modal(monkeypatch, tmp_path):
    """A ModalExecutor behind create_app over tmp volumes and a plain-dict job store. The
    scratch volume's reload is where the api container learns of the worker's file; here
    it never does, as on 2026-09-19."""
    import types
    pytest.importorskip("modal")
    from haversack import modal_app as m
    fake = {}
    monkeypatch.setattr(m, "jobs_dict", fake)
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "scratch"))
    vol = types.SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setattr(m, "cache_vol", vol)
    monkeypatch.setattr(m, "scratch_vol", vol)
    ex = m.ModalExecutor()
    ex.segmenter = _Segmenter(steps=1)
    return m, fake, ex, TestClient(create_app(ex))


def _published(m, tmp_path, key, jid):
    """What a worker leaves behind: an entry under the key, a record naming it."""
    from haversack.content import digest_file
    from haversack.serve import ResultCache
    src = tmp_path / "worker" / RESULT_NAME
    src.parent.mkdir(parents=True)
    _NrrdSeg().save(src)
    res = {"outputs": [{"name": "labels", "sha256": digest_file(src)}]}
    ResultCache(m.CACHE_ROOT).put(key, src, res, {"task": "total_fast", "job": jid})
    return src, res


@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_modal_job_result_reads_the_entry_not_the_workers_scratch(monkeypatch, tmp_path, fmt):
    m, fake, ex, client = _modal(monkeypatch, tmp_path)
    src, res = _published(m, tmp_path, "k" * 64, "402631ac7d0e")
    fake["402631ac7d0e"] = {"id": "402631ac7d0e", "task": "total_fast", "state": "done",
                            "cache_key": "k" * 64, "result": res}
    assert not (Path(m.SCRATCH_ROOT) / "402631ac7d0e" / RESULT_NAME).exists()
    url = "/v1/jobs/402631ac7d0e/result" + (f"?format={fmt}" if fmt else "")
    r = client.get(url)
    assert r.status_code == 200, r.text
    if fmt is None:
        assert r.content == src.read_bytes()


@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_modal_job_result_with_neither_copy_is_410(monkeypatch, tmp_path, fmt):
    m, fake, ex, client = _modal(monkeypatch, tmp_path)

    def refuses():                        # Modal's reload can raise (open files)
        raise RuntimeError("there are open files preventing the operation")
    monkeypatch.setattr(m, "scratch_vol", type("V", (), {"reload": staticmethod(refuses)})())
    fake["j"] = {"id": "j", "task": "total_fast", "state": "done", "cache_key": "q" * 64,
                 "result": {"outputs": [{"name": "labels", "sha256": "sha256:0"}]}}
    r = client.get("/v1/jobs/j/result" + (f"?format={fmt}" if fmt else ""))
    assert r.status_code == 410, r.text


def test_a_modal_cache_hit_records_its_key(monkeypatch, tmp_path):
    """A job answered from the cache at submit named only the generation it was handed
    (cache_path); the result route needs the key to resolve - and lease - the entry."""
    m, fake, ex, client = _modal(monkeypatch, tmp_path)
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: [])
    from haversack.serve import result_key
    key = result_key(("idc:x",), "total_fast", {}, [])
    _published(m, tmp_path, key, "earlier")
    meta = ex.submit("j2", tmp_path / "jd", None, "total_fast", {}, identity=("idc:x",))
    assert meta["cached"] is True and fake["j2"]["cache_key"] == key
    assert client.get("/v1/jobs/j2").json()["key"] == key
    assert client.get("/v1/jobs/j2/result").status_code == 200
