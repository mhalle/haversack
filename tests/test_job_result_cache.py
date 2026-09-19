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


# -- review round 2026-09-19: each finding pinned ------------------------------------

def test_an_entry_without_a_digest_is_not_taken_for_the_jobs_bytes(tmp_path):
    """An unreadable result.json reads as {}, and a legacy entry may carry no outputs:
    the first version matched either and served another publication's bytes."""
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    job_bytes = (ex.get(jid).dir / RESULT_NAME).read_bytes()
    other = tmp_path / "other.seg.nrrd"
    _NrrdSeg(fill=7).save(other)
    ex.cache.put(s["key"], other, {}, {"task": s["task"]})
    r = client.get(f"/v1/jobs/{jid}/result")
    assert r.status_code == 200 and r.content == job_bytes
    assert r.headers["etag"] == f'"{s["result"]["outputs"][0]["sha256"]}"'
    _drop_scratch(ex, jid)
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 410
    ex.close()


def test_an_evicted_record_does_not_advertise_an_entry_holding_other_bytes(tmp_path):
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       keep_finished=1)
    client = TestClient(create_app(ex))
    first = submit(client)
    s = wait_state(client, first, ("done",))
    (ex.get(first).dir / RESULT_NAME).unlink()
    for _ in range(2):
        wait_state(client, submit(client), ("done",))
    other = tmp_path / "other.seg.nrrd"
    _NrrdSeg(fill=7).save(other)
    ex.cache.put(s["key"], other, {"outputs": [{"name": "labels", "sha256": "sha256:f"}]},
                 {"task": s["task"]})
    st = client.get(f"/v1/jobs/{first}").json()
    assert st["result_available"] is False and "result" not in st["links"]
    assert client.get(f"/v1/jobs/{first}/result").status_code == 410
    ex.close()


def test_a_file_that_leaves_after_the_check_is_410_not_500(tmp_path, monkeypatch):
    """FileResponse opened the path at send time; a purge or a scratch reload in
    between raised mid-response."""
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w")               # no cache: job copy only
    client = TestClient(create_app(ex), raise_server_exceptions=False)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    real = ex.result_file

    def then_gone(j):
        state, p = real(j)
        Path(p).unlink()
        return state, p
    monkeypatch.setattr(ex, "result_file", then_gone)
    assert client.get(f"/v1/jobs/{jid}/result").status_code == 410
    ex.close()


def test_the_conversion_answers_410_when_its_input_leaves_and_cleans_up(tmp_path, monkeypatch):
    import tempfile
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w")
    client = TestClient(create_app(ex), raise_server_exceptions=False)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    conv = tmp_path / "tmp"
    conv.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(conv))
    real_read = sitk.ReadImage
    leave = {"on": True}

    def read(path, *a, **k):
        if leave["on"]:
            Path(path).unlink()
        raise RuntimeError("ImageFileReader_Execute: does not exist")
    monkeypatch.setattr(sitk, "ReadImage", read)
    assert client.get(f"/v1/jobs/{jid}/result?format=nii.gz").status_code == 410
    assert not list(conv.glob("haversack-conv-*"))
    # a read that fails with the file still there is a real error, not a 410
    jid2 = submit(client)
    wait_state(client, jid2, ("done",))
    leave["on"] = False
    assert client.get(f"/v1/jobs/{jid2}/result?format=nii.gz").status_code == 500
    monkeypatch.setattr(sitk, "ReadImage", real_read)
    ex.close()


def test_two_concurrent_uploads_through_a_guarded_executor_both_finish(tmp_path):
    """The volume guard is a threading.Lock, and the route held it across `await
    upload.read()`: Starlette reads a part over 1 MB in a thread, a second upload's
    acquire() blocked the event loop, and the first could never resume (review
    2026-09-19; the Modal api container)."""
    import asyncio
    import tempfile
    import threading
    httpx = pytest.importorskip("httpx")
    seg = _Segmenter(steps=1)
    ex = LocalExecutor(seg, workdir=tmp_path / "w")
    ex.volume_guard = threading.Lock()
    app = create_app(ex)
    rng = np.random.default_rng(0)

    def big(seed):
        img = sitk.GetImageFromArray(rng.integers(0, 30000, (40, 128, 128), np.int16))
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "v.nii.gz"
            sitk.WriteImage(img, str(f), True)
            return f.read_bytes()
    bodies = [big(0), big(1)]
    assert all(len(b) > 1 << 20 for b in bodies)

    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await asyncio.gather(*(
                c.post("/v1/jobs", data={"task": "total_fast"},
                       files={"file": ("ct.nii.gz", b, "application/octet-stream")})
                for b in bodies))
    out = {}
    t = threading.Thread(target=lambda: out.update(r=asyncio.run(main())), daemon=True)
    t.start()
    t.join(20)
    assert not t.is_alive(), "two concurrent uploads deadlocked on the volume guard"
    assert [r.status_code for r in out["r"]] == [202, 202], [r.text for r in out["r"]]
    ex.close()


# -- review round 2 (mutation testing): the gaps it found ------------------------------

def test_same_output_needs_the_jobs_own_digest():
    from haversack.serve import same_output
    d = {"outputs": [{"name": "labels", "sha256": "sha256:x"}]}
    assert same_output(d, d)
    assert not same_output({}, d) and not same_output({}, {}) and not same_output(d, {})
    assert not same_output(d, {"outputs": [{"name": "labels", "sha256": "sha256:y"}]})


def test_an_unreadable_entry_result_is_not_this_jobs_result(tmp_path):
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    entry = Path(ex.cache_get(s["key"])[0]).parent
    (entry / "result.json").write_text("{not json", encoding="utf-8")
    assert ex._entry_holds(s["key"], s["result"]) is False
    ex.close()


def test_the_download_length_is_the_files(tmp_path):
    _, ex, client = _make(tmp_path)
    jid, s = _done_job(client, ex)
    r = client.get(f"/v1/jobs/{jid}/result")
    size = Path(ex.cache_get(s["key"])[0]).stat().st_size
    assert int(r.headers["content-length"]) == len(r.content) == size
    ex.close()
