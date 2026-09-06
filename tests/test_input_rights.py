"""What a result was computed from, and under what terms.

Each source answers `rights()` from its repository's own metadata; the fetch
door records it beside the bytes; both job bodies and the command line write it
into provenance as `inputs`. No network here: every repository is a canned
answer keyed on the request, so what is tested is the shape each source builds
from what its API actually returns (captured live on 2026-09-06).
"""
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from haversack import fetchlib, jobpolicy, sources

UUID = "19ecafc9-d05a-4c6c-8727-ce1a78190d11"
SID = "1.2.840.113654.2.55.97114726565566537928831413367474015470"

#: The repositories' answers, as captured.
ANSWERS = {
    ("POST", "/v3/sql"): {"columns": ["collection_id", "license_short_name", "SeriesInstanceUID"],
                          "rows": [{"collection_id": "nlst", "license_short_name": "CC BY 4.0",
                                    "SeriesInstanceUID": SID}]},
    ("POST", "/v3/citations"): {"citations": ["National Lung Screening Trial Research Team. (2013). <i>Data from the National Lung Screening Trial (NLST)</i> (Version 3) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.HMQ8-J677"],
                                "idc_acknowledgment": "Fedorov, A., et al. (2023). National Cancer Institute Imaging Data Commons. RadioGraphics. https://doi.org/10.1148/rg.230180"},
    ("GET", "/getSeries"): [{"Collection": "CPTAC-CCRCC", "CollectionURI": "https://doi.org/10.7937/K9/TCIA.2018.OBLAMN27",
                             "LicenseName": "Creative Commons Attribution 4.0 International License",
                             "LicenseURI": "https://creativecommons.org/licenses/by/4.0/", "DateReleased": "2018-10-24"}],
    ("GET", "/api/records/7262581"): {"doi": "10.5281/zenodo.7262581", "metadata": {
        "title": "Amos", "license": {"id": "cc-by-4.0"}, "creators": [{"name": "JI YUANFENG"}]}},
    ("GET", "/api/datasets/org/data"): {"author": "org", "cardData": {"license": "cc-by-nc-4.0"}, "tags": ["license:cc-by-nc-4.0"]},
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


class EachSourceAnswersFromItsRepository(unittest.TestCase):

    def setUp(self):
        self.patch = mock.patch.object(fetchlib, "urlopen", _fake_urlopen)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_idc_answers_per_series_with_the_datasets_citation_and_its_own(self):
        r = sources.IDCSource().rights(UUID)
        self.assertEqual(r["collection"], "nlst")
        self.assertEqual(r["license"], {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"})
        self.assertEqual(r["citation_doi"], "10.7937/TCIA.HMQ8-J677")
        self.assertNotIn("<i>", r["citation"])
        self.assertEqual(r["acknowledge"]["doi"], "10.1148/rg.230180")

    def test_an_idc_bucket_prefix_is_the_idc_series_it_names(self):
        self.assertEqual(sources.S3Source().rights(f"idc-open-data/{UUID}/")["collection"], "nlst")
        self.assertEqual(sources.GCSSource().rights(f"idc-open-data/{UUID}/")["collection"], "nlst")
        self.assertIsNone(sources.S3Source().rights("fcp-indi/data/x.nii.gz"))   # a bucket is not a license

    def test_tcia_answers_from_nbia(self):
        r = sources.TCIASource().rights(SID)
        self.assertEqual(r["collection"], "CPTAC-CCRCC")
        self.assertEqual(r["license"]["url"], "https://creativecommons.org/licenses/by/4.0/")

    def test_zenodo_answers_from_the_record(self):
        r = sources.ZenodoSource().rights("7262581/amos22.zip")
        self.assertEqual(r["license"]["name"], "cc-by-4.0")
        self.assertEqual(r["license"]["url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertEqual(r["doi"], "10.5281/zenodo.7262581")
        self.assertIn("Zenodo. https://doi.org/10.5281/zenodo.7262581", r["citation"])

    def test_openneuro_is_cc0_by_policy_without_a_request(self):
        self.patch.stop()                       # no network: a policy, not a lookup
        try:
            r = sources.openneuro_source().rights("ds000114/sub-01/anat/sub-01_T1w.nii.gz")
        finally:
            self.patch.start()
        self.assertEqual(r["license"]["name"], "CC0-1.0")
        self.assertEqual(r["dataset"], "ds000114")

    def test_hugging_face_reports_what_the_uploader_declared(self):
        r = sources.HuggingFaceSource().rights("org/data@abc123/x.nii.gz")
        self.assertEqual(r["license"]["name"], "cc-by-nc-4.0")
        self.assertIn("declared", r["determined_by"])

    def test_github_reports_the_repository_license_or_nothing(self):
        self.assertEqual(sources.GitHubReleaseSource().rights("o/licensed@v1/a.zip")["license"]["name"], "Apache-2.0")
        self.assertIsNone(sources.GitHubReleaseSource().rights("o/bare@v1/a.zip"))


class TheFetchDoorRecordsIt(unittest.TestCase):

    class _Src:
        prefix = "toy"

        def __init__(self, rights=None, raise_=False):
            self._rights, self._raise = rights, raise_

        def fetch(self, ident, dest_dir):
            d = pathlib.Path(dest_dir) / "series"
            d.mkdir(exist_ok=True)
            (d / "x.bin").write_bytes(b"x")
            return d

        def rights(self, ident, fetched=None, credentials=None):
            if self._raise:
                raise RuntimeError("repository down")
            return self._rights

    def test_the_record_lands_beside_the_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            got = sources.fetch_recording_rights(self._Src({"license": {"name": "CC BY 4.0"}}), "a", td)
            self.assertTrue((got / "x.bin").exists())
            rec = sources.read_rights(td)
            self.assertEqual(rec["identity"], "toy:a")
            self.assertEqual(rec["rights"]["license"]["name"], "CC BY 4.0")
            self.assertRegex(rec["checked"], r"^\d{4}-\d{2}-\d{2}T")

    def test_a_failed_lookup_never_undoes_a_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            got = sources.fetch_recording_rights(self._Src(raise_=True), "a", td)
            self.assertTrue((got / "x.bin").exists())
            rec = sources.read_rights(td)
            self.assertIsNone(rec["rights"])
            self.assertIn("repository down", rec["error"])

    def test_a_source_without_the_hook_is_undetermined(self):
        class Bare:
            prefix = "bare"
            fetch = self._Src.fetch
        with tempfile.TemporaryDirectory() as td:
            sources.fetch_recording_rights(Bare(), "a", td)
            self.assertIsNone(sources.read_rights(td)["rights"])
            self.assertIsNone(sources.read_rights(pathlib.Path(td) / "nowhere"))


class TheResultSaysWhatItWasComputedFrom(unittest.TestCase):

    def test_input_rights_reads_the_sidecars_in_binding_order(self):
        class Cache:
            def __init__(self, root):
                self.root = pathlib.Path(root)

            def entry(self, key):
                d = self.root / key.replace(":", "_").replace("/", "_")
                d.mkdir(exist_ok=True)
                return d

        with tempfile.TemporaryDirectory() as td:
            cache = Cache(td)
            (cache.entry("s3:b/k") / sources.RIGHTS_SIDECAR).write_text(json.dumps(
                {"rights": {"license": {"name": "CC BY 4.0"}}}))
            (cache.entry("zenodo:1/x") / sources.RIGHTS_SIDECAR).write_text(json.dumps(
                {"rights": None, "error": "HTTPError: 503"}))
            out = jobpolicy.input_rights(
                [{"kind": "s3", "id": "b/k", "role": "T1"}, {"kind": "upload", "role": "T2"},
                 {"kind": "zenodo", "id": "1/x", "role": "FLAIR"}, {"kind": "input", "id": "sha256:0"}],
                ["T1=s3:b/k", "T2=sha256:abc", "FLAIR=zenodo:1/x", "sha256:0"], cache)
            self.assertEqual([o["identity"] for o in out], ["s3:b/k", "sha256:abc", "zenodo:1/x", "sha256:0"])
            self.assertEqual(out[0]["rights"]["license"]["name"], "CC BY 4.0")
            self.assertIn("uploaded by the caller", out[1]["note"])
            self.assertEqual(out[2]["rights_error"], "HTTPError: 503")
            self.assertIn("not known", out[3]["note"])
            self.assertEqual(out[0]["role"], "T1")

    def test_the_local_server_writes_it_into_the_result(self):
        import pytest
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

            def rights(self, ident, fetched=None, credentials=None):
                return {"collection": "toyset", "license": {"name": "CC BY-NC 4.0"}}

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            ex = LocalExecutor(FakeSegmenter(), workdir=td, cache_dir=td / "rc", sources=[Src()])
            try:
                client = TestClient(create_app(ex))
                r = client.post("/v1/jobs", data={"task": "total_fast",
                                                  "source": json.dumps([{"kind": "toy", "id": "sp042"}])})
                self.assertEqual(r.status_code, 202, r.text)
                s = wait_state(client, r.json()["id"], ("done",))
                inputs = s["result"]["provenance"]["inputs"]
                self.assertEqual(inputs[0]["identity"], "toy:sp042")
                self.assertEqual(inputs[0]["rights"]["license"]["name"], "CC BY-NC 4.0")
                # an upload: not determined, and said so
                r = client.post("/v1/jobs", data={"task": "total_fast"},
                                files={"file": ("v.nii.gz", volume_bytes(), "application/gzip")})
                self.assertEqual(r.status_code, 202, r.text)
                s = wait_state(client, r.json()["id"], ("done",))
                up = s["result"]["provenance"]["inputs"][0]
                self.assertIsNone(up["rights"])
                self.assertIn("uploaded", up["note"])
            finally:
                ex.close()

    def test_it_is_written_in_place_because_the_result_is_frozen(self):
        """Segmentation is a frozen dataclass. Assigning `provenance` raises,
        and the unfrozen doubles in the server tests let that slip through to
        the first real run."""
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

        from haversack import cli, modal_app, serve
        import textwrap
        for fn in (serve.LocalExecutor._dispatch, modal_app._execute_job):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
            self.assertIn("record_inputs", calls, fn.__qualname__)
        self.assertIn("input_record", inspect.getsource(cli._run))

    def test_the_command_line_names_a_local_file_for_what_it_is(self):
        with tempfile.TemporaryDirectory() as td:
            rec = sources.input_record(str(pathlib.Path(td) / "scan.nii.gz"), cache_dir=td)
            self.assertEqual(rec["kind"], "file")
            self.assertIsNone(rec["rights"])
            rec = sources.input_record("zenodo:7262581/amos22.zip", cache_dir=td)   # not fetched yet
            self.assertEqual(rec, {"kind": "zenodo", "identity": "zenodo:7262581/amos22.zip", "rights": None})


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
            self.assertEqual(json.loads(buf.getvalue())["rights"]["collection"], "nlst")
        # a local file: refused in one line (main() turns an InputError into exit code 2)
        self.assertEqual(cli.main(["rights", "/no/such/local.nii.gz"]), 2)


if __name__ == "__main__":
    unittest.main()
