"""An input's DICOM tags, handed back as JSON (2026-09-27).

``inputs.Input.tags(select=, withhold=, per_slice=)`` and ``GET|HEAD
/v1/<source>/<identifier>/dicom.json``. The user's decisions: tags for HOSTED sources only - an
upload's are never handed back, as its bytes are not; select by dicom-spec §5's group names or
PS3.6 keywords; what the operator withholds is ``null`` with ``anonymized: true``, never left out
silently. And not content a credential may have reached: the input cache files such bytes under
the source's identity whoever fetched them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydicom")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import sources as src_mod  # noqa: E402
from haversack.inputs import Input  # noqa: E402
from haversack.serve import LocalExecutor, create_app  # noqa: E402

from test_input_copy import write_series  # noqa: E402
from test_serve import FakeSegmenter  # noqa: E402

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SERIES = "a05fb365-dfd2-4116-ab8e-a7262d2c169c"
URL = f"/v1/idc/{SERIES}/dicom.json"


class StubSource(src_mod.DataSource):
    """A hosted source that writes a small CT series - one whose credential could reach
    private bytes, as a Hugging Face repo's can."""
    prefix = "stub"
    id_pattern = r"[a-z0-9]+"
    description = "test source"
    credentials_reach_private = True

    def __init__(self, fetches):
        self.fetches = fetches

    def fetch(self, identifier, dest_dir, *, credentials=None):
        self.fetches.append((identifier, credentials))
        return write_series(Path(dest_dir) / "series")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "hc"))
    fetches = []

    def fetch_idc(series, entry):
        fetches.append((series, None))
        return write_series(Path(entry) / "series")

    made = []

    def make(**kw):
        ex = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / f"w{len(made)}",
                           cache_dir=tmp_path / f"c{len(made)}", fetch_idc_fn=fetch_idc,
                           sources=[src_mod.IDCSource(), StubSource(fetches)], **kw)
        made.append(ex)
        return ex, TestClient(create_app(ex, token=TOKEN))

    yield make, fetches
    for ex in made:
        ex.close()


def test_a_get_fetches_and_answers_the_series_tags(server):
    make, fetches = server
    ex, client = make()
    r = client.get(URL, headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["series"]["KVP"] == 120 and body["series"]["Modality"] == "CT"
    assert body["series"]["PatientName"] == "Fixture^Patient"
    assert len(body["slices"]) == 6
    assert [s["XRayTubeCurrent"] for s in body["slices"]] == [100 + 10 * i for i in range(6)]
    assert body["tags_version"] == 2 and "anonymized" not in body
    assert r.headers["cache-control"] == "private, no-cache"
    assert r.headers["etag"].startswith('"sha256:')
    assert fetches == [(SERIES, None)]
    assert client.get(URL, headers=AUTH).status_code == 200
    assert len(fetches) == 1                          # held now: read, not fetched again


def test_head_never_fetches_and_then_says_what_get_says(server):
    make, fetches = server
    ex, client = make()
    r = client.head(URL, headers=AUTH)
    assert r.status_code == 404 and r.headers["cache-control"] == "no-store"
    assert fetches == []
    g = client.get(URL, headers=AUTH)
    h = client.head(URL, headers=AUTH)
    assert h.status_code == 200
    assert h.headers["etag"] == g.headers["etag"]
    assert int(h.headers["content-length"]) == len(g.content)
    assert client.get(URL, headers={**AUTH, "If-None-Match": g.headers["etag"]}).status_code == 304
    assert client.head(URL, headers={**AUTH, "If-None-Match": g.headers["etag"]}).status_code == 304


def test_select_by_group_and_keyword_and_leave_the_slices_out(server):
    make, _ = server
    ex, client = make()
    r = client.get(URL, params=[("select", "ct"), ("select", "Modality")], headers=AUTH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["series"]) <= {"KVP", "XRayTubeCurrent", "Modality"}
    assert body["series"]["KVP"] == 120 and body["series"]["Modality"] == "CT"
    assert all(set(s) <= {"KVP", "XRayTubeCurrent", "Modality"} for s in body["slices"])
    assert any("XRayTubeCurrent" in s for s in body["slices"])
    r = client.get(URL, params={"slices": "false"}, headers=AUTH)
    assert "slices" not in r.json() and r.json()["series"]["KVP"] == 120


def test_an_unknown_name_is_refused_not_ignored(server):
    make, _ = server
    ex, client = make()
    r = client.get(URL, params={"select": "Kvp"}, headers=AUTH)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_name"


def test_the_token_is_required(server):
    make, fetches = server
    ex, client = make()
    assert client.get(URL).status_code == 401
    assert client.head(URL).status_code == 401
    assert fetches == []


def test_the_operator_withholds_as_null_and_says_so(server):
    make, _ = server
    ex, client = make(dicom_withhold=("patient",))
    body = client.get(URL, headers=AUTH).json()
    assert body["series"]["PatientName"] is None and body["series"]["PatientID"] is None
    assert "PatientBirthDate" not in body["series"]        # never there: not claimed removed
    assert body["anonymized"] is True
    assert body["series"]["KVP"] == 120
    # a selection cannot get around the policy
    body = client.get(URL, params={"select": "PatientName"}, headers=AUTH).json()
    assert body["series"] == {"PatientName": None} and body["anonymized"] is True


def test_a_misspelled_policy_stops_the_server(server):
    make, _ = server
    with pytest.raises(ValueError, match="neither a module"):
        make(dicom_withhold=("patients",))


def test_content_a_credential_fetched_is_not_handed_back(server):
    make, fetches = server
    ex, client = make()
    key = "stub:private1"
    ex.series_cache.pin(key)
    try:
        ex.series_cache.get_or_fetch(key, credentials="secret")
    finally:
        ex.series_cache.unpin(key)
    record = src_mod.read_input_record(ex.series_cache.entry(key))
    assert record["credentialed"] is True and "secret" not in json.dumps(record)
    r = client.get("/v1/stub/private1/dicom.json", headers=AUTH)
    assert r.status_code == 403 and r.json()["detail"]["code"] == "input_withheld"


def test_an_anonymous_fetch_is_recorded_so_and_served(server):
    make, fetches = server
    ex, client = make()
    r = client.get("/v1/stub/public1/dicom.json", headers=AUTH)
    assert r.status_code == 200, r.text
    assert fetches[-1] == ("public1", None)
    record = src_mod.read_input_record(ex.series_cache.entry("stub:public1"))
    assert record["credentialed"] is False


def test_a_record_from_before_the_fact_is_refused_where_a_credential_could_reach(server):
    make, fetches = server
    ex, client = make()
    assert client.get("/v1/stub/old1/dicom.json", headers=AUTH).status_code == 200
    entry = ex.series_cache.entry("stub:old1")
    record = src_mod.read_input_record(entry)
    del record["credentialed"]                               # as every record before today
    (Path(entry) / src_mod.INPUT_SIDECAR).write_text(json.dumps(record))
    r = client.get("/v1/stub/old1/dicom.json", headers=AUTH)
    assert r.status_code == 403
    # ... and served where no credential can reach anything private (idc: none is ever sent)
    assert client.get(URL, headers=AUTH).status_code == 200
    assert "credentialed" not in (src_mod.read_input_record(ex.series_cache.entry(f"idc:{SERIES}"))
                                  or {})                     # the idc seam writes no record


def test_an_upload_has_no_tags_route(server, tmp_path):
    make, _ = server
    ex, client = make()
    digest = "sha256:" + "0" * 64
    for path in (f"/v1/inputs/{digest}/dicom.json", f"/v1/sha256/{'0' * 64}/dicom.json"):
        assert client.get(path, headers=AUTH).status_code in (404, 405), path


def test_the_read_only_twin_has_no_tags_route(server):
    make, _ = server
    ex, _ = make()
    twin = TestClient(create_app(ex, read_only=True))
    assert twin.get(URL).status_code in (404, 405)
    assert not any(getattr(r, "path", "").endswith("/dicom.json") for r in twin.app.routes)


def test_a_non_dicom_input_has_no_tags(tmp_path):
    import numpy as np
    import SimpleITK as sitk
    p = tmp_path / "x.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((3, 4, 5), np.int16)), str(p))
    assert Input(None, p, None).tags() == {}
    with pytest.raises(ValueError):                  # a bad name is refused all the same
        Input(None, p, None).tags(select=["nope"])


def test_one_name_is_one_name_not_its_letters(tmp_path, server):
    make, _ = server
    ex, client = make()
    client.get(URL, headers=AUTH)
    key = f"idc:{SERIES}"
    ex.series_cache.pin(key)
    try:
        x = Input(key, ex.series_cache.path(key), None)
        tags = x.tags(select="ct")
        assert tags["series"] == {"KVP": 120}           # the tube current varies: per slice
        assert all(set(s) == {"XRayTubeCurrent"} for s in tags["slices"])
    finally:
        ex.series_cache.unpin(key)


def test_every_source_says_truly_whether_a_credential_can_reach_private_bytes():
    """The flag is held to what each archive source's ``_headers`` does with a credential:
    sends it (True) or refuses it (False). A source without ``_headers`` never sends one."""
    for s in src_mod.default_sources():
        headers = getattr(s, "_headers", None)
        if headers is None:
            assert s.credentials_reach_private is False, s.prefix
            continue
        try:
            sent = bool(headers("probe"))
        except src_mod.InputError:
            sent = False
        assert s.credentials_reach_private is sent, s.prefix


def test_a_folder_read_in_place_answers_as_its_copy_does(tmp_path, monkeypatch):
    """One conversion whichever form is on hand: a local series not stored as a copy is
    converted by the copy's own code, so its tags equal those of its copy."""
    from haversack import inputs
    folder = write_series(tmp_path / "ct")
    in_place = Input(None, folder, None)
    assert not in_place.is_copy
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    copy = inputs.open(str(folder), cache_dir=tmp_path / "cache")
    assert copy.is_copy
    assert in_place.tags() == copy.tags()
    assert in_place.tags(select=["ct"], per_slice=False) == copy.tags(select=["ct"], per_slice=False)
    assert in_place.tags()["series"]["PatientName"] == "Fixture^Patient"


def test_a_deployment_without_input_tags_says_501(server):
    """Modal's executor has no ``input_tags``: its inputs live in the workers' caches, which the
    api container cannot read. The route answers 501 naming the local command, never a 500."""
    make, fetches = server
    ex, _ = make()

    class NoTags(LocalExecutor):
        input_tags = None

    ex.__class__ = NoTags
    client = TestClient(create_app(ex, token=TOKEN))
    r = client.get(URL, headers=AUTH)
    assert r.status_code == 501 and "haversack tags" in r.text
    assert fetches == []


def test_the_client_asks_the_route_and_refuses_an_upload(server):
    from haversack.client import RemoteClient
    from haversack.errors import InputError
    make, _ = server
    ex, client = make()
    client.headers.update(AUTH)
    rc = RemoteClient("http://testserver")
    rc._http = client                         # starlette's TestClient is an httpx.Client
    body = rc.tags(f"idc:{SERIES}", select="ct", slices=False)
    assert body["series"] == {"KVP": 120} and "slices" not in body
    with pytest.raises(InputError, match="upload"):
        rc.tags("sha256:" + "0" * 64)
    with pytest.raises(InputError, match="<source>:<identifier>"):
        rc.tags("not-a-spec")


def test_the_command_prints_json_and_refuses_a_bad_name(tmp_path, monkeypatch, capsys):
    from haversack.cli import main
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "hc"))
    folder = write_series(tmp_path / "ct")
    assert main(["tags", str(folder), "--select", "ct", "--no-slices"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["series"] == {"KVP": 120} and "slices" not in out
    assert main(["tags", str(folder), "--select", "Kvp"]) == 2
    assert "neither a module" in capsys.readouterr().err
