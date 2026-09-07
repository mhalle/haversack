"""What a result was computed from: the bytes, their origin, their license,
and what to cite.

Each source answers `describe_input()` from its repository's own metadata; the
fetch door records that plus the content's digest beside the bytes; both job
bodies and the command line write it into provenance as `inputs`. No network
here: every repository is a canned answer keyed on the request, so what is
tested is the shape each source builds from what its API actually returns
(captured live on 2026-09-06).
"""
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import pytest

from haversack import fetchlib, jobpolicy, sources

UUID = "19ecafc9-d05a-4c6c-8727-ce1a78190d11"
SID = "1.2.840.113654.2.55.97114726565566537928831413367474015470"

ANSWERS = {
    ("POST", "/v3/sql"): {"rows": [{"collection_id": "nlst", "license_short_name": "CC BY 4.0",
                                    "SeriesInstanceUID": SID, "series_revised_idc_version": 4}]},
    ("POST", "/v3/citations"): {"citations": ["NLST Research Team. (2013). <i>Data from the NLST</i> [Dataset]. https://doi.org/10.7937/TCIA.HMQ8-J677"],
                                "idc_acknowledgment": "Fedorov, A., et al. (2023). IDC. https://doi.org/10.1148/rg.230180"},
    ("GET", "/getSeries"): [{"Collection": "CPTAC-CCRCC", "CollectionURI": "https://doi.org/10.7937/K9/TCIA.2018.OBLAMN27",
                             "LicenseName": "Creative Commons Attribution 4.0 International License",
                             "LicenseURI": "https://creativecommons.org/licenses/by/4.0/", "DateReleased": "2018-10-24"}],
    ("GET", "/api/records/7262581"): {"doi": "10.5281/zenodo.7262581", "metadata": {
        "title": "Amos", "license": {"id": "cc-by-4.0"}, "creators": [{"name": "JI YUANFENG"}]}},
    ("GET", "/api/datasets/org/data"): {"author": "org", "cardData": {"license": "cc-by-nc-4.0"}},
    ("GET", "/repos/o/licensed"): {"license": {"spdx_id": "Apache-2.0", "url": "https://api.github.com/licenses/apache-2.0"}},
    ("GET", "/repos/o/bare"): {"license": None},
}


class _Resp(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(req, timeout=None):
    from urllib.parse import urlparse
    path = urlparse(req.full_url).path
    for (method, tail), answer in ANSWERS.items():
        if req.get_method() == method and path.endswith(tail):
            return _Resp(json.dumps(answer).encode())
    raise AssertionError(f"unexpected request {req.get_method()} {req.full_url}")


def _dicom_dir(root, n=3, uid="1.2.3.4"):
    """A tiny but real DICOM series, so SimpleITK genuinely recognizes it."""
    sitk = pytest.importorskip("SimpleITK")
    import numpy as np
    d = pathlib.Path(root) / "series"
    d.mkdir()
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    for i in range(n):
        sl = sitk.GetImageFromArray(np.full((1, 4, 4), i + 1, np.int16))
        for tag, val in (("0008|0060", "MR"), ("0020|000e", uid), ("0020|000d", "9.8.7"),
                         ("0008|103e", "T1w"), ("0020|0013", str(i + 1)), ("0008|0018", f"{uid}.{i}"),
                         ("0020|0032", f"0\\0\\{i}"), ("0020|0037", "1\\0\\0\\0\\1\\0")):
            sl.SetMetaData(tag, val)
        w.SetFileName(str(d / f"IM{i}.dcm"))
        w.Execute(sl)
    return d


class EachSourceAnswersFromItsRepository(unittest.TestCase):

    def setUp(self):
        self.patch = mock.patch.object(fetchlib, "urlopen", _fake_urlopen)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_idc_answers_per_series_with_release_license_and_both_citations(self):
        r = sources.IDCSource().describe_input(UUID)
        self.assertEqual(r["origin"]["collection"], "nlst")
        self.assertEqual(r["origin"]["uid"], SID)
        self.assertEqual(r["origin"]["version"], "IDC data release v4")
        self.assertEqual(r["origin"]["doi"], "10.7937/TCIA.HMQ8-J677")
        self.assertEqual(r["license"], {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"})
        self.assertEqual([c["for"] for c in r["cite"]], ["dataset", "IDC"])
        self.assertNotIn("<i>", r["cite"][0]["text"])
        self.assertEqual(r["cite"][1]["doi"], "10.1148/rg.230180")

    def test_an_idc_bucket_prefix_is_the_idc_series_it_names(self):
        self.assertEqual(sources.S3Source().describe_input(f"idc-open-data/{UUID}/")["origin"]["collection"], "nlst")
        self.assertEqual(sources.GCSSource().describe_input(f"idc-open-data/{UUID}/")["origin"]["collection"], "nlst")
        self.assertIsNone(sources.S3Source().describe_input("fcp-indi/data/x.nii.gz"))   # a bucket is not a license

    def test_tcia_answers_from_nbia_with_the_collection_doi(self):
        r = sources.TCIASource().describe_input(SID)
        self.assertEqual(r["origin"]["collection"], "CPTAC-CCRCC")
        self.assertEqual(r["origin"]["doi"], "10.7937/K9/TCIA.2018.OBLAMN27")
        self.assertEqual(r["license"]["url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertEqual(r["cite"][0]["doi"], "10.7937/K9/TCIA.2018.OBLAMN27")

    def test_zenodo_answers_from_the_record(self):
        r = sources.ZenodoSource().describe_input("7262581/amos22.zip")
        self.assertEqual(r["license"]["name"], "cc-by-4.0")
        self.assertEqual(r["license"]["url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertEqual(r["origin"]["doi"], "10.5281/zenodo.7262581")
        self.assertEqual(r["origin"]["version"], "record 7262581")
        self.assertIn("Zenodo. https://doi.org/10.5281/zenodo.7262581", r["cite"][0]["text"])

    def test_openneuro_is_cc0_by_policy_without_a_request(self):
        self.patch.stop()                       # no network: a policy, not a lookup
        try:
            r = sources.openneuro_source().describe_input("ds000114/sub-01/anat/sub-01_T1w.nii.gz")
        finally:
            self.patch.start()
        self.assertEqual(r["license"]["name"], "CC0-1.0")
        self.assertEqual(r["origin"]["dataset"], "ds000114")
        self.assertEqual(r["origin"]["url"], "https://openneuro.org/datasets/ds000114")

    def test_hugging_face_reports_what_the_uploader_declared_and_the_commit(self):
        r = sources.HuggingFaceSource().describe_input("org/data@abc123/x.nii.gz")
        self.assertEqual(r["license"]["name"], "cc-by-nc-4.0")
        self.assertEqual(r["origin"]["version"], "abc123")
        self.assertIn("declared", r["origin"]["determined_by"])

    def test_github_reports_the_repository_license_and_tag_or_nothing(self):
        r = sources.GitHubReleaseSource().describe_input("o/licensed@v1/a.zip")
        self.assertEqual(r["license"]["name"], "Apache-2.0")
        self.assertEqual(r["origin"]["version"], "v1")
        self.assertIsNone(sources.GitHubReleaseSource().describe_input("o/bare@v1/a.zip"))


class TheFetchDoorRecordsIt(unittest.TestCase):

    class _Src:
        prefix = "toy"

        def __init__(self, said=None, raise_=False, dicom=False):
            self._said, self._raise, self._dicom = said, raise_, dicom

        def fetch(self, ident, dest_dir):
            if self._dicom:
                return _dicom_dir(dest_dir)
            d = pathlib.Path(dest_dir) / "series"
            d.mkdir(exist_ok=True)
            (d / "x.bin").write_bytes(b"x" * 10)
            (d / "y.bin").write_bytes(b"y" * 5)
            return d

        def describe_input(self, ident, fetched=None, credentials=None):
            if self._raise:
                raise RuntimeError("repository down")
            return self._said

    def test_the_record_pins_the_bytes_and_lands_beside_them(self):
        from haversack.content import digest_dir
        with tempfile.TemporaryDirectory() as td:
            got = sources.fetch_recording_origin(
                self._Src({"origin": {"dataset": "d"}, "license": {"name": "CC BY 4.0"}, "cite": []}), "a", td)
            rec = sources.read_input_record(td)
            self.assertEqual(rec["identity"], "toy:a")
            # the content store's own digest, so a fetched tree and a stored one hash alike
            self.assertEqual(rec["content"], {"digest": digest_dir(got), "bytes": 15, "files": 2})
            self.assertTrue(rec["content"]["digest"].startswith("sha256-tree:"))
            self.assertEqual(rec["origin"]["dataset"], "d")
            self.assertRegex(rec["origin"]["fetched"], r"^\d{4}-\d{2}-\d{2}T")
            self.assertEqual(rec["license"]["name"], "CC BY 4.0")

    def test_a_dicom_series_reports_what_its_files_say(self):
        with tempfile.TemporaryDirectory() as td:
            sources.fetch_recording_origin(self._Src(None, dicom=True), "a", td)
            rec = sources.read_input_record(td)
            self.assertEqual(rec["content"]["files"], 3)
            self.assertEqual(rec["content"]["dicom"]["series_instance_uid"], "1.2.3.4")
            self.assertEqual(rec["content"]["dicom"]["study_instance_uid"], "9.8.7")
            self.assertEqual(rec["content"]["dicom"]["modality"], "MR")
            self.assertEqual(rec["content"]["dicom"]["series_description"], "T1w")
            self.assertIn("could not determine", rec["note"])   # the source said nothing

    def test_a_failed_lookup_never_undoes_a_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            got = sources.fetch_recording_origin(self._Src(raise_=True), "a", td)
            self.assertTrue((got / "x.bin").exists())
            rec = sources.read_input_record(td)
            self.assertIsNone(rec["license"])
            self.assertIn("repository down", rec["error"])
            self.assertIsNotNone(rec["content"])              # the digest still stands

    def test_a_source_without_the_hook_is_undetermined(self):
        class Bare:
            prefix = "bare"
            fetch = self._Src.fetch
            _dicom = False
        with tempfile.TemporaryDirectory() as td:
            sources.fetch_recording_origin(Bare(), "a", td)
            self.assertIsNone(sources.read_input_record(td)["origin"].get("dataset"))
            self.assertIsNone(sources.read_input_record(pathlib.Path(td) / "nowhere"))


class TheResultSaysWhatItWasComputedFrom(unittest.TestCase):

    def test_input_records_come_back_in_binding_order(self):
        class Cache:
            def __init__(self, root):
                self.root = pathlib.Path(root)

            def entry(self, key):
                d = self.root / key.replace(":", "_").replace("/", "_")
                d.mkdir(exist_ok=True)
                return d

        with tempfile.TemporaryDirectory() as td:
            cache = Cache(td)
            (cache.entry("s3:b/k") / sources.INPUT_SIDECAR).write_text(json.dumps(
                {"kind": "s3", "identity": "s3:b/k", "content": {"digest": "sha256:aa"},
                 "origin": {"collection": "c"}, "license": {"name": "CC BY 4.0"}, "cite": []}))
            (cache.entry("zenodo:1/x") / sources.INPUT_SIDECAR).write_text(json.dumps(
                {"kind": "zenodo", "identity": "zenodo:1/x", "content": None, "origin": None,
                 "license": None, "cite": [], "error": "HTTPError: 503"}))
            out = jobpolicy.input_records(
                [{"kind": "s3", "id": "b/k", "role": "T1"}, {"kind": "upload", "role": "T2"},
                 {"kind": "zenodo", "id": "1/x", "role": "FLAIR"}, {"kind": "input", "id": "sha256:0"}],
                ["T1=s3:b/k", "T2=sha256:abc", "FLAIR=zenodo:1/x", "sha256:0"], cache)
            self.assertEqual([o["identity"] for o in out], ["s3:b/k", "sha256:abc", "zenodo:1/x", "sha256:0"])
            self.assertEqual([o.get("role") for o in out], ["T1", "T2", "FLAIR", None])
            self.assertEqual(out[0]["license"]["name"], "CC BY 4.0")
            self.assertEqual(out[0]["content"]["digest"], "sha256:aa")
            self.assertEqual(out[1]["content"], {"digest": "sha256:abc"})   # an upload's digest IS its identity
            self.assertIn("uploaded by the caller", out[1]["note"])
            self.assertEqual(out[2]["error"], "HTTPError: 503")
            self.assertIn("not known", out[3]["note"])

    def test_the_local_server_writes_it_into_the_result(self):
        pytest.importorskip("fastapi")
        from test_serve import FakeSegmenter, LocalExecutor, TestClient, create_app, volume_bytes, wait_state

        class Src:
            prefix, id_pattern, description = "toy", r"[a-z0-9]+", "toy"

            def enabled(self):
                return True

            def identity(self, ident):
                return f"toy:{ident}"

            def fetch(self, ident, dest_dir):
                d = pathlib.Path(dest_dir) / "series"
                d.mkdir(parents=True, exist_ok=True)
                (d / "img.nii.gz").write_bytes(volume_bytes())
                return d

            def describe_input(self, ident, fetched=None, credentials=None):
                return {"origin": {"collection": "toyset"}, "license": {"name": "CC BY-NC 4.0"}, "cite": []}

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ex = LocalExecutor(FakeSegmenter(), workdir=td, cache_dir=td / "rc", sources=[Src()])
            try:
                client = TestClient(create_app(ex))
                r = client.post("/v1/jobs", data={"task": "total_fast",
                                                  "source": json.dumps([{"kind": "toy", "id": "sp042"}])})
                self.assertEqual(r.status_code, 202, r.text)
                s = wait_state(client, r.json()["id"], ("done",))
                inp = s["result"]["provenance"]["inputs"][0]
                self.assertEqual(inp["identity"], "toy:sp042")
                self.assertEqual(inp["license"]["name"], "CC BY-NC 4.0")
                self.assertEqual(inp["origin"]["collection"], "toyset")
                # ONE file: a blob digest, the same one the upload below gets for the
                # same bytes. This asserted "sha256-tree:" while the two paths disagreed.
                self.assertTrue(inp["content"]["digest"].startswith("sha256:"))
                r = client.post("/v1/jobs", data={"task": "total_fast"},
                                files={"file": ("v.nii.gz", volume_bytes(), "application/gzip")})
                self.assertEqual(r.status_code, 202, r.text)
                s = wait_state(client, r.json()["id"], ("done",))
                up = s["result"]["provenance"]["inputs"][0]
                self.assertIsNone(up["license"])
                self.assertTrue(up["content"]["digest"].startswith("sha256:"))
                self.assertIn("uploaded", up["note"])
            finally:
                ex.close()

    def test_the_same_bytes_fetched_and_uploaded_carry_the_same_digest(self):
        """A fetch always writes a DIRECTORY, even for one object, so a single
        fetched file read `sha256-tree:` over a directory of one while the same
        bytes uploaded read `sha256:` - two identities for one input, and a
        content-store lookup that could never match. `sole_file` is the rule
        `materialize` already used to hand the pipeline that one file."""
        from haversack.content import digest_file
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            payload = b"\x1f\x8b" + b"m" * 500

            class One:
                prefix = "toy"

                def fetch(self, ident, dest_dir):
                    d = pathlib.Path(dest_dir) / "series"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "mprage.nii.gz").write_bytes(payload)
                    return d

            entry = td / "fetched"
            entry.mkdir()
            sources.fetch_recording_origin(One(), "a", entry)
            fetched = sources.read_input_record(entry)["content"]

            loose = td / "mprage.nii.gz"
            loose.write_bytes(payload)
            uploaded = sources.input_record(str(loose), cache_dir=td)["content"]

            self.assertEqual(fetched["digest"], uploaded["digest"])
            self.assertEqual(fetched["digest"], digest_file(loose))
            self.assertEqual(fetched["files"], 1)
            # ...and a real series still hashes as the tree it is
            multi = td / "multi"
            multi.mkdir()
            sources.fetch_recording_origin(TheFetchDoorRecordsIt._Src(None), "b", multi)
            self.assertTrue(sources.read_input_record(multi)["content"]["digest"]
                            .startswith("sha256-tree:"))

    def test_it_is_written_in_place_because_the_result_is_frozen(self):
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Frozen:
            provenance: dict = dataclasses.field(default_factory=dict)

        class Cache:
            def entry(self, key):
                return pathlib.Path("/nonexistent")

        seg = Frozen(provenance={"task": "t"})
        jobpolicy.record_inputs(seg, [{"kind": "upload"}], ["sha256:1"], Cache())
        self.assertEqual(seg.provenance["task"], "t")
        self.assertEqual(seg.provenance["inputs"][0]["identity"], "sha256:1")
        jobpolicy.record_inputs(object(), [], [], Cache())          # no provenance: no error

    def test_both_job_bodies_and_the_cli_write_it(self):
        import ast
        import inspect
        import textwrap

        from haversack import cli, modal_app, serve
        for fn in (serve.LocalExecutor._dispatch, modal_app._execute_job):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
            self.assertIn("record_inputs", calls, fn.__qualname__)
        self.assertIn("input_record", inspect.getsource(cli._run))

    def test_the_command_line_pins_a_local_file_and_names_it_for_what_it_is(self):
        from haversack.content import digest_file
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "scan.nii.gz"
            f.write_bytes(b"\x1f\x8b" + b"z" * 100)
            rec = sources.input_record(str(f), cache_dir=td)
            self.assertEqual(rec["kind"], "file")
            self.assertEqual(rec["content"], {"digest": digest_file(f), "bytes": 102, "files": 1})
            self.assertIsNone(rec["origin"])
            rec = sources.input_record("zenodo:7262581/amos22.zip", cache_dir=td)   # not fetched yet
            self.assertEqual(rec["identity"], "zenodo:7262581/amos22.zip")
            self.assertEqual(rec["note"], "no record of this fetch")


class TheCommandPrintsIt(unittest.TestCase):

    def test_rights_prints_the_record_without_fetching(self):
        from contextlib import redirect_stdout

        from haversack import cli
        with mock.patch.object(fetchlib, "urlopen", _fake_urlopen):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["rights", "zenodo:7262581/amos22.zip"])
            self.assertEqual(rc, 0)
            self.assertIn("cc-by-4.0", buf.getvalue())
            self.assertIn("10.5281/zenodo.7262581", buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["rights", "--json", f"idc:{UUID}"])
            rec = json.loads(buf.getvalue())
            self.assertEqual(rec["origin"]["collection"], "nlst")
            self.assertEqual(rec["origin"]["version"], "IDC data release v4")
        # a local file: refused in one line (main() turns an InputError into exit code 2)
        self.assertEqual(cli.main(["rights", "/no/such/local.nii.gz"]), 2)


if __name__ == "__main__":
    unittest.main()
