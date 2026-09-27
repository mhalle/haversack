"""A black-box review of the local server (2026-09-26) - each finding pinned by a test that
fails on 4b5ecb2.

1. A label map named by its digest (or uploaded) was accepted into an IMAGE role and ran; the
   same bytes named as `result:` were refused `wrong_input_kind`.
2. A malformed multipart body to POST /v1/inputs was a 500 (POST /v1/jobs said 400).
3. POST /v1/inputs stored a "tree" of text files; job errors carried absolute server paths.
4. Options SERVER.md says are validated at submit were accepted and failed later, or ran
   unbounded: folds, configuration, grid, envelope_mm, a string `no_cache`.
5. `?format=` on a job's result served seg.nrrd for any unknown value, `nii` sent `.nii.gz`,
   and the converted file had a new ETag every request.
6. After a restart GET /v1/jobs listed nothing, though each old job still answered by id.
8. HTTP details: 401 without WWW-Authenticate, a case-sensitive auth scheme, no HEAD on the
   GET-only service routes, upper-case digests, a foreign cursor, `file` beside a `source`.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from haversack.serve import LocalExecutor, create_app  # noqa: E402

from test_serve import FakeSegmenter, make, submit, volume_bytes, wait_state  # noqa: E402


def _seg_nrrd_bytes() -> bytes:
    """A tiny label map with its names, as haversack writes one."""
    import SimpleITK as sitk
    img = sitk.Image(4, 4, 4, sitk.sitkUInt8)
    img[1, 1, 1] = 1
    img.SetMetaData("Segment0_Name", "spleen")
    img.SetMetaData("Segment0_LabelValue", "1")
    img.SetMetaData("Segment0_Layer", "0")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.seg.nrrd"
        sitk.WriteImage(img, str(p))
        return p.read_bytes()


def _digest(b: bytes) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(b).hexdigest()


# -- 1. the kind of a stored input -----------------------------------------------------------

def test_a_label_map_named_by_digest_is_refused_from_an_image_role(tmp_path):
    seg, ex, client = make(tmp_path)
    b = _seg_nrrd_bytes()
    r = client.put(f"/v1/inputs/{_digest(b)}", content=b)
    assert r.status_code == 200, r.text
    r = client.post("/v1/jobs", data={"task": "total_fast", "source": json.dumps(
        [{"kind": "input", "sha256": _digest(b)}])})
    assert r.status_code == 422, r.text
    d = r.json()["detail"]
    assert d["code"] == "wrong_input_kind" and d["input_kind"] == "labels", d
    assert not seg.calls


def test_a_label_map_uploaded_into_an_image_role_is_refused(tmp_path):
    seg, ex, client = make(tmp_path)
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", _seg_nrrd_bytes())},
                    data={"task": "total_fast"})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "wrong_input_kind"


def test_an_image_named_by_digest_still_runs(tmp_path):
    seg, ex, client = make(tmp_path)
    b = volume_bytes(4242)
    assert client.put(f"/v1/inputs/{_digest(b)}", content=b).status_code == 200
    r = client.post("/v1/jobs", data={"task": "total_fast", "source": json.dumps(
        [{"kind": "input", "sha256": _digest(b)}])})
    assert r.status_code == 202, r.text
    assert wait_state(client, r.json()["id"])["state"] == "done"


# -- 4. options validated at submit ------------------------------------------------------------

@pytest.mark.parametrize("opts,param", [
    ({"grid": 1e-6}, "grid"), ({"grid": 0.05}, "grid"), ({"grid": 1e9}, "grid"),
    ({"grid": "1.5"}, "grid"), ({"grid": True}, "grid"),
    ({"envelope_mm": -5}, "envelope_mm"), ({"envelope_mm": 1e308}, "envelope_mm"),
    ({"folds": []}, "folds"), ({"folds": [0, 0]}, "folds"),
    ({"resampling_order": "3"}, "resampling_order"),
    ({"no_cache": "yes"}, "no_cache"), ({"no_cache": "false"}, "no_cache"),
])
def test_an_option_out_of_its_documented_range_is_refused_at_submit(tmp_path, opts, param):
    seg, ex, client = make(tmp_path)
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(7))},
                    data={"task": "total_fast", "options": json.dumps(opts)})
    assert r.status_code == 422, (opts, r.status_code, r.text)
    assert r.json()["detail"]["parameter"] == param
    assert not seg.calls


@pytest.mark.parametrize("opts", [{"grid": 1}, {"grid": 0.5}, {"envelope_mm": 0},
                                  {"envelope_mm": 20}, {"envelope_mm": None},
                                  {"folds": [0]}, {"no_cache": True}, {"no_cache": False}])
def test_documented_values_still_pass(tmp_path, opts):
    seg, ex, client = make(tmp_path)
    jid = submit(client, options=opts)
    assert wait_state(client, jid)["state"] == "done"


def test_folds_and_configuration_are_checked_against_the_installed_model(tmp_path):
    from haversack.segmenter import Segmenter
    d = tmp_path / "Dataset297_TotalSegmentator_total_3mm_1559subj" / \
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    (d / "fold_0").mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "a": 1},
        "numTraining": 1, "file_ending": ".nii.gz"}))
    (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
    s = Segmenter(weights=tmp_path, device="cpu")
    t = "ts.v2:total_fast"
    assert s.option_problem(t, {"folds": [99]})[0] == "folds"
    assert s.option_problem(t, {"folds": [0, 99]})[0] == "folds"      # never run as [0] alone
    assert s.option_problem(t, {"configuration": "3d_lowres"})[0] == "configuration"
    assert "3d_fullres" in s.option_problem(t, {"configuration": "3d_lowres"})[1]
    assert s.option_problem(t, {"folds": [0], "configuration": "3d_fullres"}) is None
    assert s.option_problem(t, {}) is None
    assert s.option_problem("ts.v2:total", {"folds": [7]}) is None      # not installed: unknowable


def test_the_submit_asks_the_segmenter_about_folds(tmp_path):
    class Picky(FakeSegmenter):
        def option_problem(self, task, options):
            return ("folds", "fold(s) [99] are not in it") if options.get("folds") == [99] else None
    seg = Picky()
    client = TestClient(create_app(LocalExecutor(seg, workdir=tmp_path)))
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(8))},
                    data={"task": "total_fast", "options": json.dumps({"folds": [99]})})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["parameter"] == "folds"


def test_a_fine_grid_over_a_large_field_of_view_is_refused_before_it_allocates():
    from haversack.errors import InputError
    from haversack.frame import MAX_OUTPUT_VOXELS
    from haversack.grid import Grid
    import haversack.frame as F
    fr = F.Frame.__new__(F.Frame)
    object.__setattr__(fr, "source", Grid((512, 512, 2000), (0.8, 0.8, 0.8), (0.0, 0.0, 0.0)))
    with pytest.raises(InputError, match="coarser grid"):
        fr.resolve_grid(0.1)
    assert int(__import__("numpy").prod(fr.resolve_grid(1.0).shape)) < MAX_OUTPUT_VOXELS


# -- 2. a malformed multipart body -------------------------------------------------------------

def test_a_part_header_past_the_parser_limit_is_a_400_not_a_500(tmp_path):
    seg, ex, client = make(tmp_path)
    name = "a" * 5000
    body = (f'--abc\r\nContent-Disposition: form-data; name="f"; filename="{name}"\r\n'
            'Content-Type: application/octet-stream\r\n\r\nxyz\r\n--abc--\r\n').encode()
    r = client.post("/v1/inputs", content=body,
                    headers={"content-type": "multipart/form-data; boundary=abc"})
    assert r.status_code == 400, (r.status_code, r.text)


def test_a_delimiter_other_than_the_declared_boundary_is_a_400(tmp_path):
    seg, ex, client = make(tmp_path)
    body = (b'--xyz\r\nContent-Disposition: form-data; name="f"; filename="a.nii"\r\n\r\n'
            b'abc\r\n--xyz--\r\n')
    r = client.post("/v1/inputs", content=body,
                    headers={"content-type": "multipart/form-data; boundary=abc"})
    assert r.status_code in (400, 422), (r.status_code, r.text)   # never a 500


# -- 3. what the input store accepts, and what an error says ----------------------------------

def test_a_tree_of_files_nothing_can_read_is_not_stored(tmp_path):
    seg, ex, client = make(tmp_path)
    r = client.post("/v1/inputs", files=[("f", ("a.txt", b"hello")), ("f", ("b.txt", b"world"))])
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "unknown_format"


def test_a_put_the_server_cannot_identify_does_not_name_its_temporary_file(tmp_path):
    seg, ex, client = make(tmp_path)
    b = b"not an image at all" * 10
    r = client.put(f"/v1/inputs/{_digest(b)}", content=b)
    assert r.status_code == 422
    assert "the upload" in r.json()["detail"]["message"], r.text
    assert "tmp" not in r.json()["detail"]["message"]


def test_a_failed_job_does_not_name_the_servers_paths(tmp_path, capsys):
    class Leaky(FakeSegmenter):
        def segment(self, image, task, **kw):
            raise RuntimeError(f"cannot read {image}: truncated")
    ex = LocalExecutor(Leaky(), workdir=tmp_path / "w")
    client = TestClient(create_app(ex))
    jid = submit(client)
    s = wait_state(client, jid, ("failed",))
    assert str(tmp_path) not in s["error"] and str(tmp_path.resolve()) not in s["error"], s
    assert "scan.nii.gz: truncated" in s["error"], s        # the file's own name stays
    assert str(tmp_path / "w") in capsys.readouterr().err   # the whole text is in the log


def test_scrubbing_leaves_api_routes_alone():
    from haversack.serve import scrub_server_paths
    t = "see /v1/jobs/abc/result; /srv/w/series_cache/e1.r2!idc:x/series: bad."
    assert scrub_server_paths(t, ["/srv/w"]) == "see /v1/jobs/abc/result; series: bad."


# -- 5. ?format= on a job's result -------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["bogus", "zarr.zip", "../../etc/passwd", ""])
def test_an_unknown_format_is_refused_not_served_as_nrrd(tmp_path, fmt):
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    r = client.get(f"/v1/jobs/{jid}/result", params={"format": fmt})
    assert r.status_code == 422, (fmt, r.status_code)
    assert "nii.gz" in r.json()["detail"]["message"]


def _real(tmp_path):
    """A server whose results are real NRRDs, so a conversion can be read."""
    from test_job_result_cache import _make
    return _make(tmp_path)


def test_nii_is_served_uncompressed_and_nii_gz_compressed(tmp_path):
    seg, ex, client = _real(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    gz = client.get(f"/v1/jobs/{jid}/result", params={"format": "nii.gz"})
    raw = client.get(f"/v1/jobs/{jid}/result", params={"format": "nii"})
    assert gz.status_code == raw.status_code == 200
    assert gz.content[:2] == b"\x1f\x8b" and raw.content[:2] != b"\x1f\x8b"
    assert raw.content[344:348] == b"n+1\x00"
    assert 'filename="' in raw.headers["content-disposition"]
    assert raw.headers["content-disposition"].rstrip('"').endswith(".nii")
    assert client.get(f"/v1/jobs/{jid}/result",
                      params={"format": "seg.nrrd"}).content[:4] == b"NRRD"


def test_a_converted_result_has_a_stable_validator_and_revalidates(tmp_path):
    seg, ex, client = _real(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    url = f"/v1/jobs/{jid}/result"
    a = client.get(url, params={"format": "nii.gz"})
    b = client.get(url, params={"format": "nii.gz"})
    assert a.headers["etag"] == b.headers["etag"]
    assert "last-modified" not in a.headers
    assert a.headers["etag"] != client.get(url).headers["etag"]        # not the NRRD's tag
    assert a.headers["etag"] != client.get(url, params={"format": "nii"}).headers["etag"]
    again = client.get(url, params={"format": "nii.gz"}, headers={"If-None-Match": a.headers["etag"]})
    assert again.status_code == 304
    h = client.head(url, params={"format": "nii.gz"})
    assert h.status_code == 200 and h.headers["etag"] == a.headers["etag"]


# -- 6. the job listing after a restart --------------------------------------------------------

def test_the_job_listing_survives_a_restart(tmp_path):
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    ex.close()
    ex2 = LocalExecutor(FakeSegmenter(), workdir=tmp_path)      # same workdir: a restart
    client2 = TestClient(create_app(ex2))
    assert client2.get(f"/v1/jobs/{jid}").json()["evicted"] is True
    listed = {d["id"]: d for d in client2.get("/v1/jobs").json()["jobs"]}
    assert jid in listed, listed
    assert listed[jid]["state"] == "done" and listed[jid]["evicted"] is True
    ex2.close()


# -- 8. HTTP details ---------------------------------------------------------------------------

def _tokened(tmp_path):
    ex = LocalExecutor(FakeSegmenter(), workdir=tmp_path)
    return ex, TestClient(create_app(ex, token="s3cret"))


def test_a_401_names_the_scheme_that_would_do(tmp_path):
    ex, client = _tokened(tmp_path)
    r = client.get("/v1/jobs")
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_the_auth_scheme_is_case_insensitive(tmp_path, scheme):
    ex, client = _tokened(tmp_path)
    assert client.get("/v1/jobs", headers={"Authorization": f"{scheme} s3cret"}).status_code == 200
    assert client.get("/v1/jobs", headers={"Authorization": f"{scheme} S3CRET"}).status_code == 401


@pytest.mark.parametrize("path", ["/v1/health", "/v1/tasks", "/v1/sources"])
def test_head_answers_where_get_does(tmp_path, path):
    seg, ex, client = make(tmp_path)
    got, probe = client.get(path), client.head(path)
    assert got.status_code == probe.status_code == 200, (path, probe.status_code)
    assert probe.content == b""
    assert probe.headers["content-length"] == got.headers["content-length"]


def test_head_on_a_job_status(tmp_path):
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    wait_state(client, jid, ("done",))
    assert client.head(f"/v1/jobs/{jid}").status_code == 200
    assert client.head("/v1/jobs/nosuch").status_code == 404


def test_the_head_routes_stay_out_of_the_published_schema(tmp_path):
    seg, ex, client = make(tmp_path)
    ops = client.get("/openapi.json").json()["paths"]
    assert "head" not in ops["/v1/health"] and "get" in ops["/v1/health"]


def test_delete_on_an_artifact_url_is_405_with_allow(tmp_path, monkeypatch):
    import haversack.serve as serve_mod
    monkeypatch.setattr(serve_mod, "_idc_enabled", lambda: True)
    seg, ex, client = make(tmp_path)
    r = client.delete("/v1/idc/0be27d1c-9410-47ff-9c9f-a44b26a4bd55/total_fast/preview.png")
    assert r.status_code == 405, r.text
    assert set(r.headers["allow"].replace(" ", "").split(",")) >= {"GET", "HEAD"}
    assert "DELETE" not in r.headers["allow"]


def test_an_upper_case_digest_is_the_same_digest(tmp_path):
    seg, ex, client = make(tmp_path)
    b = volume_bytes(4343)
    d = _digest(b)
    up = "sha256:" + d.split(":", 1)[1].upper()
    assert client.put(f"/v1/inputs/{up}", content=b).status_code == 200
    assert client.get(f"/v1/inputs/{up}").status_code == 200
    assert client.get(f"/v1/inputs/{d}").status_code == 200
    r = client.post("/v1/jobs", data={"task": "total_fast", "source": json.dumps(
        [{"kind": "input", "sha256": up}])})
    assert r.status_code == 202, r.text
    assert wait_state(client, r.json()["id"])["input_identity"] == [d]


def test_a_cursor_from_another_listing_or_filter_is_a_422(tmp_path):
    from test_job_result_cache import _make
    seg, ex, client = _make(tmp_path)
    for _ in range(2):
        wait_state(client, submit(client), ("done",))
    page = client.get("/v1/segmentations", params={"limit": 1}).json()
    cur = page["next_cursor"]
    assert cur
    assert client.get("/v1/segmentations", params={"limit": 1, "cursor": cur}).status_code == 200
    assert client.get("/v1/segmentations",
                      params={"limit": 1, "cursor": cur, "task": "total"}).status_code == 422
    assert client.get("/v1/rankfields", params={"limit": 1, "cursor": cur}).status_code == 422


def test_a_file_beside_a_source_that_names_no_upload_is_refused(tmp_path):
    seg, ex, client = make(tmp_path)
    b = volume_bytes(4444)
    client.put(f"/v1/inputs/{_digest(b)}", content=b)
    r = client.post("/v1/jobs", files={"file": ("scan.nii.gz", volume_bytes(4445))},
                    data={"task": "total_fast", "source": json.dumps(
                        [{"kind": "input", "sha256": _digest(b)}])})
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "ambiguous_input"


def test_a_done_job_reports_its_progress_complete(tmp_path):
    seg, ex, client = make(tmp_path)
    jid = submit(client)
    s = wait_state(client, jid, ("done",))
    assert s["progress"]["fraction"] == 1.0 and s["progress"]["stage"] == "done", s["progress"]


def test_serve_exits_quietly_on_ctrl_c(tmp_path, capsys):
    import sys
    import types
    from unittest import mock

    from haversack import cache_admin, serve

    class _Server:
        def __init__(self, config):
            pass

        def run(self, sockets=None):
            raise KeyboardInterrupt        # what asyncio's runner re-raises after uvicorn stops

    class _Config:
        def __init__(self, *a, **k):
            pass

        def bind_socket(self):
            return types.SimpleNamespace(close=lambda: None)
    uv = types.ModuleType("uvicorn")
    uv.Server, uv.Config = _Server, _Config
    args = types.SimpleNamespace(
        token="t", no_token=False, host="127.0.0.1", port=0, device="cpu", dtype="auto",
        model_root=None, cache_models=1, workdir=str(tmp_path), no_result_cache=True,
        max_pending=4, keep_finished=4)
    with mock.patch.dict(sys.modules, {"uvicorn": uv}), \
            mock.patch.object(serve, "create_app", lambda ex, **k: object()), \
            mock.patch.object(serve, "LocalExecutor"), \
            mock.patch("haversack.segmenter.Segmenter"), \
            mock.patch.object(cache_admin, "serve_token_path", lambda port: tmp_path / "tok.json"), \
            mock.patch.object(cache_admin, "check_cache_root"):
        try:
            rc = serve.main_serve(args)
        except KeyboardInterrupt:          # caught here, or it would stop the whole session
            pytest.fail("the Ctrl-C escaped main_serve, to be printed as a traceback")
        assert rc == 130
    assert "stopped" in capsys.readouterr().err
