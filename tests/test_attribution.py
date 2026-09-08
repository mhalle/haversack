"""Every model haversack runs is someone else's work, and the credit must be
findable from any layer: the task, its ecosystem, the engine that runs it.

The facts themselves live in data/attribution.json and were read from each
project's own README, LICENSE or documentation; these tests check that the
record is complete for every ecosystem that ships, that it reaches every
place a user looks (describe(), the wire, a result's provenance, the CLI), and
that the per-task rules (TotalSegmentator's licensed models, the MRI paper for
MR tasks) are applied.
"""
import json
import unittest
from unittest import mock

from haversack import attribution

#: Derived, never hand-listed: this tuple decides which catalogs `_every_ecosystem` can
#: see, so a hand-written copy that missed a new engine would quietly stop checking that
#: engine's catalog for a record - the test would still pass, having looked at less.
from haversack.engines.registry import engine_env_vars  # noqa: E402

ENGINE_VARS = engine_env_vars()


def _every_ecosystem():
    from haversack.ecosystems import default_ecosystems
    with mock.patch.dict("os.environ", {v: "1" for v in ENGINE_VARS}):
        return default_ecosystems()


class TheRecordIsComplete(unittest.TestCase):

    def test_every_shipped_ecosystem_has_a_record(self):
        names = {e.name for e in _every_ecosystem()} | {"custom"}
        for name in sorted(names):
            rec = attribution.for_ecosystem(name)
            self.assertIsNotNone(rec, f"no attribution record for ecosystem {name!r}")
            self.assertTrue(rec.get("title"), name)
            self.assertTrue(rec.get("description"), name)
            if name == "custom":
                continue                     # the user's own models: nothing to say
            self.assertTrue(rec.get("repository"), f"{name}: no repository")
            self.assertTrue(rec.get("group"), f"{name}: no group")
            self.assertIsNotNone(rec.get("license"), f"{name}: no license")

    def test_every_engine_has_a_record(self):
        from haversack.engines.registry import ENGINES
        for name in ENGINES:
            rec = attribution.for_engine(name)
            self.assertIsNotNone(rec, f"no attribution record for engine {name!r}")
            self.assertTrue(rec.get("cite"), f"{name}: an engine's paper is what it asks for")

    def test_every_reference_is_identifiable(self):
        """A reference is only useful if a reader can find it: a DOI, and a
        PubMed ID whenever the venue is indexed there."""
        data = attribution.load()
        recs = list(data["ecosystems"].values()) + [r for r in data["engines"].values()
                                                    if "same_as_ecosystem" not in r]
        n = 0
        for rec in recs:
            for ref in rec.get("cite") or []:
                n += 1
                self.assertTrue(ref.get("title") and ref.get("authors") and ref.get("year"), ref)
                self.assertTrue(ref.get("doi") or ref.get("arxiv"), f"unidentifiable: {ref['title']}")
                indexed = ref.get("journal") not in ("arXiv", "medRxiv") and "CVPR" not in str(ref.get("journal"))
                if ref.get("doi") and indexed:
                    self.assertRegex(ref.get("pmid") or "", r"^\d{7,9}$", f"no PMID: {ref['title']}")
        self.assertGreaterEqual(n, 14)

    def test_totalsegmentators_licensed_models_all_exist_in_the_catalog(self):
        """The list of license-gated tasks is copied from upstream's README; a
        renamed or dropped task must fail here, not silently un-gate."""
        from haversack.ecosystems import TSEcosystem
        offered = set(TSEcosystem().tasks())
        gated = attribution.for_ecosystem("ts")["licensed_tasks"]
        self.assertTrue(gated)
        self.assertEqual(sorted(set(gated) - offered), [])
        self.assertIn("brain_aneurysm", offered)


class TheThreeLayersMerge(unittest.TestCase):

    def test_a_task_carries_its_ecosystem_and_its_engine(self):
        rec = attribution.for_task("ts:total_fast", {"ecosystem": "ts", "engine": "nnunetv2",
                                                     "modality": "CT"})
        self.assertEqual(rec["ecosystem"], "ts")
        self.assertEqual(rec["engine"], "nnunetv2")
        self.assertEqual(rec["ecosystem_info"]["title"], "TotalSegmentator")
        dois = [r.get("doi") for r in rec["cite"]]
        self.assertEqual(dois, ["10.1148/ryai.230024", "10.1038/s41592-020-01008-z"])
        self.assertEqual([r["for"] for r in rec["cite"]], ["ecosystem", "engine"])

    def test_the_mri_paper_is_asked_for_by_the_mr_tasks_only(self):
        ct = attribution.for_task("ts:total_fast", {"ecosystem": "ts", "modality": "CT"})
        mr = attribution.for_task("ts:total_mr", {"ecosystem": "ts", "modality": "MR"})
        self.assertNotIn("10.1148/radiol.241613", [r.get("doi") for r in ct["cite"]])
        self.assertIn("10.1148/radiol.241613", [r.get("doi") for r in mr["cite"]])
        self.assertIn("39964271", [r.get("pmid") for r in mr["cite"]])

    def test_a_licensed_totalsegmentator_model_says_so(self):
        rec = attribution.for_task("ts:appendicular_bones", {"ecosystem": "ts"})
        self.assertIn("non-commercial", rec["task"]["license"]["weights"])
        self.assertTrue(rec["task"]["license"]["url"].startswith("https://backend.totalsegmentator.com"))
        aneurysm = attribution.for_task("ts:brain_aneurysm", {"ecosystem": "ts"})
        self.assertEqual(aneurysm["task"]["license"]["weights"], "CC-BY-NC-4.0")
        plain = attribution.for_task("ts:total_fast", {"ecosystem": "ts"})
        self.assertNotIn("license", plain["task"])

    def test_a_manifests_facts_become_the_tasks_own(self):
        """A bundle's authors and references, a per-model license: the catalog
        manifest's record, lifted into the task layer and its references into
        the cite list ahead of the ecosystem's."""
        info = {"ecosystem": "monai", "engine": "monai", "authors": "Vanderbilt University + MONAI team",
                "copyright": "Copyright (c) MONAI Consortium",
                "references": ["Yu, X. et al. UNesT. arXiv preprint arXiv:2209.14378 (2022).",
                               "Tang, Y. et al. https://doi.org/10.1007/978-3-030-87199-4_40"]}
        rec = attribution.for_task("monai:x", info)
        self.assertEqual(rec["task"]["authors"], "Vanderbilt University + MONAI team")
        self.assertEqual(rec["cite"][0]["arxiv"], "2209.14378")
        self.assertEqual(rec["cite"][1]["doi"], "10.1007/978-3-030-87199-4_40")
        self.assertEqual(rec["cite"][-1]["arxiv"], "2211.02701")          # the MONAI framework
        dental = attribution.for_task("dentalsegmentator:base",
                                      {"ecosystem": "dentalsegmentator", "license": "cc-by-4.0"})
        self.assertEqual(dental["task"]["license"], {"weights": "cc-by-4.0"})
        self.assertEqual(dental["cite"][0]["pmid"], "38878813")
        self.assertEqual(dental["cite"][1]["title"][:7], "nnU-Net")

    def test_a_redistributed_model_credits_its_real_makers(self):
        """MOOSE's registry offers DentalSegmentator's checkpoint on its own host.
        The output is governed by Dot et al.'s CC BY 4.0, and their paper leads."""
        rec = attribution.for_task("moose:clin_ct_dental", {"ecosystem": "moose", "engine": "nnunetv2"})
        self.assertEqual(rec["task"]["derived_from"], "dentalsegmentator")
        self.assertEqual(rec["task"]["license"]["weights"], "CC-BY-4.0")
        self.assertEqual(rec["cite"][0]["pmid"], "38878813")          # Dot et al. first
        self.assertIn("35772962", [r.get("pmid") for r in rec["cite"]])  # MOOSE still cited
        block = attribution.provenance_block("moose:clin_ct_dental", {"ecosystem": "moose"})
        self.assertEqual(block["license"]["weights"], "CC-BY-4.0")
        other = attribution.for_task("moose:clin_ct_organs", {"ecosystem": "moose"})
        self.assertNotIn("derived_from", other["task"])

    def test_the_non_commercial_weights_are_named_as_such(self):
        for eco, expected in (("voxtell", "CC-BY-NC-SA-4.0"),):
            rec = attribution.for_ecosystem(eco)
            self.assertEqual(rec["license"]["weights"], expected)
        self.assertEqual(attribution.for_task("ts:brain_aneurysm", {"ecosystem": "ts"})
                         ["task"]["license"]["weights"], "CC-BY-NC-4.0")

    def test_an_engine_that_is_its_own_ecosystem_is_cited_once(self):
        rec = attribution.for_task("synthstrip:mask", {"ecosystem": "synthstrip", "engine": "synthstrip"})
        dois = [r.get("doi") for r in rec["cite"]]
        self.assertEqual(len(dois), len(set(dois)))
        self.assertIn("10.1016/j.neuroimage.2022.119474", dois)


class ItReachesEveryPlaceAUserLooks(unittest.TestCase):

    def test_the_catalog_record_and_describe_carry_it_installed_or_not(self):
        """The bug this closes: the installed path of describe() rebuilt its
        answer from the checkpoint and dropped the manifest's license and the
        attribution - the credit vanished the moment the model was usable."""
        import tempfile

        from haversack import Segmenter
        with tempfile.TemporaryDirectory() as td:
            seg = Segmenter(device="cpu", weights=td)
            installed = seg.describe("ts:total_fast")          # TS is always materialized
            self.assertEqual(installed["attribution"]["ecosystem_info"]["title"], "TotalSegmentator")
            self.assertIn("37795137", json.dumps(installed["attribution"]))
            not_installed = seg.describe("dentalsegmentator:base")
            self.assertFalse(not_installed["materialized"])
            self.assertEqual(not_installed["license"], "cc-by-4.0")
            self.assertEqual(not_installed["attribution"]["task"]["license"], {"weights": "cc-by-4.0"})

    def test_the_wire_publishes_it(self):
        import pytest
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from haversack.serve import LocalExecutor, create_app
        from haversack import Segmenter
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ex = LocalExecutor(Segmenter(device="cpu", weights=td), workdir=td)
            try:
                d = TestClient(create_app(ex)).get("/v1/tasks/total_fast").json()
            finally:
                ex.close()
            self.assertIn("attribution", d)
            self.assertEqual(d["attribution"]["engine"], "nnunetv2")
            self.assertEqual(d["attribution"]["cite"][0]["pmid"], "37795137")

    def test_every_result_provenance_names_the_license_and_the_papers(self):
        block = attribution.provenance_block("moose:clin_ct_organs", {"ecosystem": "moose"})
        self.assertEqual(block["license"]["weights"], "CC-BY-4.0")
        self.assertIn("35772962", [c.get("pmid") for c in block["cite"]])
        self.assertIn("33288961", [c.get("pmid") for c in block["cite"]])
        for c in block["cite"]:
            self.assertFalse(set(c) - {"doi", "pmid", "arxiv", "title"}, "identifiers only")
        # ...and both compute paths write it
        import inspect

        from haversack import pipeline, segmenter
        self.assertIn('"attribution"', inspect.getsource(pipeline.segment))
        self.assertIn("provenance_block", inspect.getsource(segmenter.Segmenter._run_engine))

    def test_the_cli_prints_a_pasteable_list(self):
        import io
        from contextlib import redirect_stdout

        from haversack import cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["cite", "total_fast"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("University Hospital Basel", out)
        self.assertIn("doi:10.1148/ryai.230024", out)
        self.assertIn("PMID 37795137", out)
        self.assertIn("nnU-Net", out)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["cite", "totalvibe:body_regions", "--json"])
        rec = json.loads(buf.getvalue())
        self.assertEqual(rec["cite"][0]["pmid"], "41068435")


if __name__ == "__main__":
    unittest.main()
