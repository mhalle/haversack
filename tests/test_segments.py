"""The segments index: every catalog answers what its tasks produce - each segment's id, label
value and layer - mined the way an install reads it, merged without touching what was not
asked for, stale the moment a version moves, and searchable.
"""
import io
import json
import os
import re
import shutil
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from haversack import ecosystems as E
from haversack import segments
from haversack.engines import registry as R
from haversack.errors import InputError, ModelNotFound
from haversack.tasks import TaskSpec
from haversack.weights_fetch import _write_sidecar

CFG = "nnUNetTrainer__nnUNetPlans__3d_fullres"
GENERIC = {"a": 1, "b": 2}


def _dataset(labels, channel="CT"):
    return {"channel_names": {"0": channel}, "labels": {"background": 0, **labels},
            "numTraining": 1, "file_ending": ".nii.gz"}


def _zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in members.items():
            z.writestr(name, body if isinstance(body, (bytes, str)) else json.dumps(body))
    return buf.getvalue()


def _monai_meta(version, channel_def, modality="CT"):
    return {"version": version, "network_data_format": {
        "inputs": {"image": {"modality": modality, "num_channels": 1}},
        "outputs": {"pred": {"channel_def": channel_def}}}}


def _table(listing_or_record) -> dict:
    """``{value: id}`` of the layer-0 segments - what a labelmap's segment table says."""
    return {s["value"]: s["id"] for s in listing_or_record["segments"] if not s.get("layer")}


class FakeReader:
    """Archives and documents by URL, offline. Anything not registered gets a generic answer
    in every packaging the catalogs use, so the whole catalog can be walked."""

    def __init__(self, zips=None, jsons=None, fail=()):
        self.zips, self.jsons, self.fail = dict(zips or {}), dict(jsons or {}), set(fail)

    def zip(self, url):
        if url in self.fail:
            raise urllib.error.URLError("unreachable")
        # flat (MRSegmentator) and a configuration folder at the top level - where an install
        # reads an archive that does not carry its manifest's Dataset folder
        data = self.zips.get(url) or _zip({"dataset.json": _dataset(GENERIC),
                                           f"{CFG}/dataset.json": _dataset(GENERIC)})
        return zipfile.ZipFile(io.BytesIO(data))

    def json(self, url):
        if url in self.fail:
            raise urllib.error.URLError("unreachable")
        if url in self.jsons:
            return self.jsons[url], {}
        m = re.search(r"/resolve/([^/]+)/configs/metadata\.json$", url)
        if m:
            return _monai_meta(m.group(1), {"0": "background", "1": "spleen"}), {}
        if "/api/models/" in url:
            return {"sha": "0" * 40}, {}
        raise AssertionError(f"unexpected URL {url}")


def _tmpdir(case) -> Path:
    d = Path(tempfile.mkdtemp())
    case.addCleanup(shutil.rmtree, d, True)
    return d


class EveryCatalogAnswers(unittest.TestCase):
    """The modularity half. Mining is the ecosystem's job, so every catalog this build knows
    must answer both questions: one on a shared shape passes by inheritance, and one that does
    not fails here with the method it has to write."""

    def test_every_task_of_every_known_catalog_is_versioned_and_listed(self):
        ecos = E.known_ecosystems()
        self.assertGreaterEqual(len(ecos), 10)
        walked = 0
        for eco in ecos:
            for task in eco.tasks():
                with self.subTest(task=f"{eco.name}:{task}"):
                    version = eco.label_version(task)
                    self.assertTrue(version)
                    json.dumps(version)
                    listing = eco.label_listing(task, None, FakeReader())
                    self.assertIn(listing["kind"], segments.KINDS)
                    if listing["kind"] == "segments":
                        self.assertTrue(listing["segments"])
                        for s in listing["segments"]:
                            self.assertTrue(str(s["id"]).strip())
                            self.assertTrue(isinstance(s["value"], int) and s["value"] != 0)
                    walked += 1
        self.assertGreaterEqual(walked, 100, "the walk is not reaching the catalog")

    def test_the_mined_catalogs_are_the_served_ones_with_every_engine_on(self):
        env = {e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}
        with mock.patch.dict(os.environ, env):
            served = {e.name for e in E.default_ecosystems()}
        self.assertEqual(served, {e.name for e in E.known_ecosystems()})

    def test_a_catalog_that_answers_nothing_names_the_method_to_write(self):
        class Bare(E.ModelEcosystem):
            name = "bare"
        with self.assertRaisesRegex(NotImplementedError, "label_version"):
            Bare().label_version("x")
        with self.assertRaisesRegex(NotImplementedError, "label_listing"):
            Bare().label_listing("x", None, FakeReader())

    def test_an_engine_without_a_label_table_is_not_quietly_open(self):
        class Quiet(E.ImageBakedEcosystem):
            name, engine, task_names = "quiet", "voxtell", ("t",)
        with self.assertRaisesRegex(NotImplementedError, "label_names"):
            Quiet().label_listing("t", None, FakeReader())
        with self.assertRaisesRegex(NotImplementedError, "label_names"):
            Quiet().label_version("t")

    def test_an_image_baked_catalogs_structures_are_its_engines_table(self):
        """Two sources for one list: the catalog's `structures` (what /v1/tasks says) and the
        engine row's table (what the index and the ranked builder read)."""
        checked = 0
        for eco in E.known_ecosystems():
            if isinstance(eco, E.ImageBakedEcosystem) and not eco.open_vocabulary:
                for task in eco.tasks():
                    with self.subTest(task=f"{eco.name}:{task}"):
                        table = _table(eco.label_listing(task, None, FakeReader()))
                        self.assertEqual(sorted(eco.info(task, None)["structures"]),
                                         sorted(table.values()))
                        checked += 1
        self.assertGreaterEqual(checked, 2)


class ArchivesAreReadTheWayAnInstallReadsThem(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.eco = E.DentalSegmentatorEcosystem()
        self.folder = self.eco._entries["base"]["folder"]

    def listing(self, members, eco=None, task="base"):
        eco = eco or self.eco
        url = eco._entries[task]["url"]
        return eco.label_listing(task, None, FakeReader(zips={url: _zip(members)}))

    def test_a_mined_list_is_the_list_the_installed_model_reports(self):
        """The same checkpoint, read out of its archive and installed: one segment table, and
        one modality - the manifest's, not the channel name this checkpoint misstates."""
        ds = _dataset({"liver": 1, "spleen": 2, "ignore": 3, "far": 99}, channel="PET")
        moose = E.MooseEcosystem()
        task = sorted(moose.tasks())[0]
        folder = moose._entries[task]["folder"]
        d = moose._folder(task, self.tmp) / CFG
        (d / "fold_0").mkdir(parents=True)
        (d / "dataset.json").write_text(json.dumps(ds))
        (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
        spec = moose.spec(task, self.tmp)
        got = self.listing({f"{folder}/{CFG}/dataset.json": ds}, eco=moose, task=task)
        self.assertEqual(_table(got), dict(spec.label_map))
        self.assertEqual(_table(got), {1: "liver", 2: "spleen", 99: "far"})    # no `ignore`
        self.assertEqual(got["modality"], spec.modality)
        self.assertEqual(spec.modality, moose._entries[task]["modality"])
        self.assertNotEqual(spec.modality, TaskSpec.from_model_folder(d).modality)
        self.assertEqual(got["source"]["member"], f"{folder}/{CFG}/dataset.json")

    def test_the_manifests_modality_wins_where_it_states_one(self):
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": _dataset(GENERIC, "any")})
        self.assertEqual(got["modality"], self.eco._entries["base"]["modality"])

    def test_the_preferred_configuration_is_read_and_macos_litter_is_not(self):
        got = self.listing({
            f"__MACOSX/{self.folder}/{CFG}/dataset.json": _dataset({"junk": 1}),
            f"{self.folder}/nnUNetTrainer__nnUNetPlans__2d/dataset.json": _dataset({"flat": 1}),
            f"{self.folder}/{CFG}/dataset.json": _dataset({"volume": 1}),
        })
        self.assertEqual(got["segments"], [{"id": "volume", "value": 1}])

    def test_a_nested_copy_of_a_configuration_is_not_read(self):
        """Configuration folders are the Dataset folder's direct children, as an install sees
        them; a copy deeper in the archive is not one (a review mined `stale` from one)."""
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": _dataset({"tooth": 1}),
                            f"{self.folder}/old/{CFG}/dataset.json": _dataset({"stale": 1})})
        self.assertEqual(got["segments"], [{"id": "tooth", "value": 1}])

    def test_a_preferred_configuration_without_dataset_json_is_refused(self):
        """An install takes the preferred configuration whether or not it holds a dataset.json,
        and then cannot load; reading the next one instead answered for a model no install
        runs."""
        with self.assertRaisesRegex(ModelNotFound, "has no dataset.json"):
            self.listing({f"{self.folder}/{CFG}/plans.json": {},
                          f"{self.folder}/nnUNetTrainer__nnUNetPlans__2d/dataset.json":
                              _dataset({"flat": 1})})

    def test_another_dataset_folder_in_the_archive_is_not_read(self):
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": _dataset({"tooth": 1}),
                            f"Dataset999_other/{CFG}/dataset.json": _dataset({"other": 1})})
        self.assertEqual(got["segments"], [{"id": "tooth", "value": 1}])

    def test_a_dot_prefixed_folder_is_not_a_configuration(self):
        """Dot-prefixed directories are an installer's scratch (`.staging`, `.unzip-*`), never
        a model; TotalVibe's archives are read from their top level, where one could sit."""
        eco = E.TotalVibeEcosystem()
        task = sorted(eco.tasks())[0]
        got = self.listing({f".{CFG}/dataset.json": _dataset({"staging": 1}),
                            "nnUNetTrainer__nnUNetPlans__2d/dataset.json": _dataset({"real": 1})},
                           eco=eco, task=task)
        self.assertEqual(got["segments"], [{"id": "real", "value": 1}])

    def test_configurations_the_preference_cannot_choose_between_are_refused(self):
        with self.assertRaisesRegex(ModelNotFound, "none is preferred"):
            self.listing({f"{self.folder}/T__P__alpha/dataset.json": _dataset({"a": 1}),
                          f"{self.folder}/T__P__beta/dataset.json": _dataset({"b": 1})})

    def test_an_archive_without_a_configuration_folder_is_refused(self):
        with self.assertRaisesRegex(ModelNotFound, "no <trainer>__<plans>__<config>"):
            self.listing({f"{self.folder}/dataset.json": _dataset(GENERIC)})

    def test_the_ignore_label_is_a_role_not_a_segment(self):
        """nnU-Net never predicts `ignore`: its label manager skips the key by name. Listed as a
        segment it named something no result can contain - TotalVibe's vibe said 73 for 72.
        The key is matched exactly, as nnU-Net matches it."""
        ds = _dataset({"liver": 1, "Ignore": 2, "ignore": 3})
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": ds})
        self.assertEqual(got["segments"], [{"id": "liver", "value": 1}, {"id": "Ignore", "value": 2}])
        d = self.tmp / "Dataset9_x" / CFG
        (d / "fold_0").mkdir(parents=True)
        (d / "dataset.json").write_text(json.dumps(ds))
        (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
        self.assertEqual(dict(TaskSpec.from_model_folder(self.tmp / "Dataset9_x").label_map),
                         {1: "liver", 2: "Ignore"})

    def test_the_ignore_label_is_no_region_either(self):
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": {
            "channel_names": {"0": "MR"},
            "labels": {"background": 0, "whole": [1, 2], "ignore": 3}}})
        self.assertEqual(got["segments"], [{"id": "whole", "layer": 0, "value": 1}])

    def test_overlapping_regions_are_segments_in_layers_of_their_own(self):
        """Segments need not be disjoint: a region-based model's regions overlap by design, so
        each is a segment in its own layer - not a different kind of thing."""
        got = self.listing({f"{self.folder}/{CFG}/dataset.json": {
            "channel_names": {"0": "MR"},
            "labels": {"background": 0, "whole": [1, 2, 3], "core": [2, 3]}}})
        self.assertEqual(got["kind"], "segments")
        self.assertEqual(got["segments"], [{"id": "whole", "layer": 0, "value": 1},
                                           {"id": "core", "layer": 1, "value": 1}])

    def test_a_flat_archive_whose_version_file_disagrees_is_refused(self):
        eco = E.MRSegmentatorEcosystem()
        with self.assertRaisesRegex(ModelNotFound, "weights_version"):
            self.listing({"dataset.json": _dataset(GENERIC),
                          "version.json": {"weights_version": "0.0-not-this"}}, eco=eco)
        got = self.listing({"dataset.json": _dataset(GENERIC),
                            "version.json": {"weights_version": eco._entries["base"]["tag"]}},
                           eco=eco)
        self.assertEqual(got["segments"], [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
        self.assertEqual(got["modality"], "MR")

    def test_a_monai_head_of_overlapping_outputs_is_one_layer_per_channel(self):
        eco, task = E.MonaiEcosystem(), "brats_mri_segmentation"
        v = eco._entry(task)["version"]
        url = eco.METADATA_URL.format(bundle=task, version=v)
        head = {"0": "Tumor core", "1": "Whole tumor", "2": "Enhancing tumor"}
        got = eco.label_listing(task, None, FakeReader(jsons={url: _monai_meta(v, head, "MRI")}))
        self.assertEqual(got["segments"], [{"id": "Tumor core", "layer": 0, "value": 1},
                                           {"id": "Whole tumor", "layer": 1, "value": 1},
                                           {"id": "Enhancing tumor", "layer": 2, "value": 1}])
        self.assertEqual(got["source"]["commit"], "0" * 40)

    def test_monai_metadata_at_another_version_is_refused(self):
        eco, task = E.MonaiEcosystem(), "spleen_ct_segmentation"
        url = eco.METADATA_URL.format(bundle=task, version=eco._entry(task)["version"])
        with self.assertRaisesRegex(ModelNotFound, "curates"):
            eco.label_listing(task, None, FakeReader(
                jsons={url: _monai_meta("9.9.9", {"0": "background", "1": "spleen"})}))


class RecordsAreSegmentTables(unittest.TestCase):
    """A record's segments are an id and a label value each, with a layer only where it is not
    0 - duckn's seg leaf, and DICOM's."""

    def test_layer_zero_is_not_written_and_order_is_layer_then_value(self):
        got = segments._segment_table([{"id": "b", "value": 2, "layer": 1},
                                       {"id": "a", "value": 5, "layer": 0},
                                       {"id": "c", "value": 1, "layer": 1}])
        self.assertEqual(got, [{"id": "a", "value": 5}, {"id": "c", "value": 1, "layer": 1},
                               {"id": "b", "value": 2, "layer": 1}])

    def test_one_value_may_repeat_across_layers_but_not_within_one(self):
        segments._segment_table([{"id": "a", "value": 1}, {"id": "b", "value": 1, "layer": 1}])
        with self.assertRaisesRegex(ValueError, "share layer 0 value 1"):
            segments._segment_table([{"id": "a", "value": 1}, {"id": "b", "value": 1}])

    def test_a_segment_needs_an_id_and_a_value(self):
        for bad in ([{"value": 1}], [{"id": "a"}], [{"id": " ", "value": 1}], []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                segments._segment_table(bad)


class MergingTouchesOnlyWhatWasAsked(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.path = self.tmp / "segments.json"
        self.ecos = [E.DentalSegmentatorEcosystem(), E.MRSegmentatorEcosystem()]
        r = segments.mine(self.plan(), self.path, reader=FakeReader(), today="2000-01-01")
        self.assertFalse(r["failed"], r["results"])

    def plan(self, *targets):
        if not targets:
            return segments.plan((), all_=True, ecosystems=self.ecos)
        return segments.plan(targets, ecosystems=self.ecos)

    def records(self):
        return json.loads(self.path.read_text())["tasks"]

    def test_the_first_run_records_every_task_with_what_pins_it(self):
        recs = self.records()
        self.assertEqual(set(recs), {"dentalsegmentator:base", "mrsegmentator:base",
                                     "mrsegmentator:body_comp"})
        rec = recs["dentalsegmentator:base"]
        self.assertEqual(rec["version"], self.ecos[0].label_version("base"))
        self.assertEqual(rec["segments"], [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
        self.assertEqual(rec["mined"]["at"], "2000-01-01")
        self.assertEqual(segments.check(self.path, ecosystems=self.ecos),
                         [(n, "ok", segments._summary(recs[n])) for n in
                          ("dentalsegmentator:base", "mrsegmentator:base",
                           "mrsegmentator:body_comp")])

    def test_mining_one_task_leaves_every_other_record_as_it_was(self):
        before = self.records()
        url = self.ecos[1]._entries["base"]["url"]
        r = segments.mine(self.plan("mrsegmentator:base"), self.path, today="2001-01-01",
                          reader=FakeReader(zips={url: _zip({"dataset.json": _dataset({"liver": 1})})}))
        after = self.records()
        self.assertEqual([(n, s) for n, s, _ in r["results"]], [("mrsegmentator:base", "changed")])
        self.assertEqual(after["mrsegmentator:base"]["segments"], [{"id": "liver", "value": 1}])
        self.assertEqual(after["mrsegmentator:base"]["mined"]["at"], "2001-01-01")
        for name in ("dentalsegmentator:base", "mrsegmentator:body_comp"):
            self.assertEqual(after[name], before[name])

    def test_an_unchanged_remine_rewrites_nothing(self):
        text = self.path.read_text()
        r = segments.mine(self.plan(), self.path, reader=FakeReader(), today="2002-02-02")
        self.assertFalse(r["written"])
        self.assertEqual(self.path.read_text(), text)
        self.assertEqual({s for _, s, _ in r["results"]}, {"unchanged"})

    def test_a_dry_run_writes_nothing(self):
        text = self.path.read_text()
        url = self.ecos[0]._entries["base"]["url"]
        r = segments.mine(self.plan("dentalsegmentator"), self.path, write=False,
                          reader=FakeReader(zips={url: _zip({f"{CFG}/dataset.json": _dataset({"x": 1})})}))
        self.assertEqual([s for _, s, _ in r["results"]], ["changed"])
        self.assertFalse(r["written"])
        self.assertEqual(self.path.read_text(), text)

    def test_a_failed_fetch_keeps_the_previous_record_and_fails_the_run(self):
        before = self.records()
        url = self.ecos[0]._entries["base"]["url"]
        r = segments.mine(self.plan("dentalsegmentator"), self.path, reader=FakeReader(fail={url}))
        self.assertTrue(r["failed"])
        self.assertEqual(self.records(), before)
        self.assertIn("kept the record mined 2000-01-01", r["results"][0][2])

    def test_a_failed_task_keeps_its_record_while_another_changes(self):
        """The failure path in a run that still writes: one task's fetch fails, another's list
        changes, and the failed task's record must come through the write unchanged. Mining a
        single failing task writes nothing, so it could not tell."""
        before = self.records()
        dental = self.ecos[0]._entries["base"]["url"]
        mrseg = self.ecos[1]._entries["base"]["url"]
        r = segments.mine(self.plan(), self.path, today="2003-03-03", reader=FakeReader(
            fail={dental}, zips={mrseg: _zip({"dataset.json": _dataset({"liver": 1})})}))
        self.assertTrue(r["failed"])
        self.assertTrue(r["written"])
        after = self.records()
        self.assertEqual(after["dentalsegmentator:base"], before["dentalsegmentator:base"])
        self.assertEqual(after["mrsegmentator:base"]["segments"], [{"id": "liver", "value": 1}])

    def _plant(self, **records):
        data = json.loads(self.path.read_text())
        base = data["tasks"]["dentalsegmentator:base"]
        for name, eco in records.items():
            data["tasks"][name.replace("__", ":")] = dict(base, task=name.replace("__", ":"),
                                                          ecosystem=eco)
        self.path.write_text(json.dumps(data))

    def test_naming_a_catalog_drops_its_vanished_tasks_and_nothing_else(self):
        self._plant(mrsegmentator__gone="mrsegmentator", cads__elsewhere="cads")
        r = segments.mine(self.plan("mrsegmentator"), self.path, reader=FakeReader())
        recs = self.records()
        self.assertNotIn("mrsegmentator:gone", recs)
        self.assertIn("cads:elsewhere", recs)
        self.assertIn(("mrsegmentator:gone", "removed"), [(n, s) for n, s, _ in r["results"]])

    def test_naming_one_task_drops_nothing(self):
        self._plant(mrsegmentator__gone="mrsegmentator")
        segments.mine(self.plan("mrsegmentator:base"), self.path, reader=FakeReader())
        self.assertIn("mrsegmentator:gone", self.records())

    def test_all_with_prune_drops_catalogs_this_build_no_longer_has(self):
        self._plant(cads__elsewhere="cads")
        segments.mine(self.plan(), self.path, reader=FakeReader(), prune=True)
        self.assertNotIn("cads:elsewhere", self.records())

    def _install_dental(self, root, labels, tag=None):
        eco = self.ecos[0]
        entry = eco._entries["base"]
        folder = eco._folder("base", root)
        d = folder / CFG
        (d / "fold_0").mkdir(parents=True)
        (d / "dataset.json").write_text(json.dumps(_dataset(labels)))
        (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
        _write_sidecar(folder, "base", tag or entry["tag"], {"url": entry["url"]}, None)

    def test_an_installed_copy_that_disagrees_keeps_the_list_out(self):
        root = self.tmp / "weights"
        self._install_dental(root, {"other": 1})
        before = self.records()
        r = segments.mine(self.plan("dentalsegmentator:base"), self.path, root=root,
                          reader=FakeReader(zips={self.ecos[0]._entries["base"]["url"]:
                                                  _zip({f"{CFG}/dataset.json": _dataset({"x": 1})})}))
        self.assertTrue(r["failed"])
        self.assertIn("installed copy", r["results"][0][2])
        self.assertEqual(self.records(), before)

    def test_an_installed_copy_that_agrees_is_said_to(self):
        root = self.tmp / "weights"
        self._install_dental(root, GENERIC)
        r = segments.mine(self.plan("dentalsegmentator:base"), self.path, root=root,
                          reader=FakeReader())
        self.assertFalse(r["failed"])
        self.assertIn("installed copy agrees", r["results"][0][2])

    def test_an_installed_copy_at_another_version_is_not_compared(self):
        root = self.tmp / "weights"
        self._install_dental(root, {"other": 1}, tag="v0-older")
        r = segments.mine(self.plan("dentalsegmentator:base"), self.path, root=root,
                          reader=FakeReader())
        self.assertFalse(r["failed"])
        self.assertIn("v0-older, not compared", r["results"][0][2])


class StalenessFollowsTheCatalog(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)
        self.path = self.tmp / "segments.json"
        self.eco = E.DentalSegmentatorEcosystem()
        segments.mine(segments.plan((), all_=True, ecosystems=[self.eco]), self.path,
                      reader=FakeReader())

    def test_a_moved_tag_makes_the_record_stale_and_says_which_field(self):
        self.assertEqual([s for _, s, _ in segments.check(self.path, ecosystems=[self.eco])],
                         ["ok"])
        self.eco._entries = {**self.eco._entries,
                             "base": {**self.eco._entries["base"], "tag": "v200"}}
        [(name, status, detail)] = segments.check(self.path, ecosystems=[self.eco])
        self.assertEqual((name, status), ("dentalsegmentator:base", "stale"))
        self.assertIn("tag", detail)

    def test_a_moved_manifest_modality_makes_the_record_stale(self):
        self.eco._entries = {**self.eco._entries,
                             "base": {**self.eco._entries["base"], "modality": "MR"}}
        [(_, status, detail)] = segments.check(self.path, ecosystems=[self.eco])
        self.assertEqual(status, "stale")
        self.assertIn("modality", detail)

    def test_missing_and_orphan_are_reported(self):
        data = json.loads(self.path.read_text())
        data["tasks"]["dentalsegmentator:gone"] = data["tasks"]["dentalsegmentator:base"]
        self.path.write_text(json.dumps(data))
        rows = {n: s for n, s, _ in segments.check(
            self.path, ecosystems=[self.eco, E.MRSegmentatorEcosystem()])}
        self.assertEqual(rows["mrsegmentator:base"], "missing")
        self.assertEqual(rows["dentalsegmentator:gone"], "orphan")
        self.assertEqual(rows["dentalsegmentator:base"], "ok")

    def test_a_modality_the_catalog_class_states_moves_its_records(self):
        """MRSegmentator and the engines state modality on the class, not in a manifest. Their
        records carry it, so an edit there must make the records stale - a review changed it
        and watched check() stay green (2026-09-13)."""
        for eco in (E.MRSegmentatorEcosystem(), E.FastSurferEcosystem(), E.SynthStripEcosystem()):
            task = eco.tasks()[0]
            before = eco.label_version(task)
            with mock.patch.object(eco, "modality", "CT (edited)"):
                self.assertNotEqual(eco.label_version(task), before, eco.name)

    def test_check_refuses_a_file_that_is_not_there(self):
        """A mistyped --file read as an empty index: every task 'never mined'."""
        with self.assertRaisesRegex(InputError, "no such index file"):
            segments.check(self.tmp / "typo.json", ecosystems=[self.eco])
        with self.assertRaisesRegex(InputError, "no such index file"):
            segments.check(self.tmp, ecosystems=[self.eco])

    def test_check_without_a_file_reads_the_packaged_index_with_the_users_over_it(self):
        """What search reads is what is checked: from an installed package, checking only the
        file `mine` would write reported the whole index missing."""
        shipped = segments.load(segments.PACKAGED)["tasks"]["dentalsegmentator:base"]
        user = self.tmp / "user.json"
        segments.dump({"dentalsegmentator:base": {**shipped, "version": {"tag": "old"}}}, user)
        with mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": str(user)}):
            rows = {n: s for n, s, _ in segments.check(targets=["dentalsegmentator"])}
        self.assertEqual(rows, {"dentalsegmentator:base": "stale"})
        with mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": str(self.tmp / "none.json")}):
            rows = {n: s for n, s, _ in segments.check(targets=["dentalsegmentator"])}
        self.assertEqual(rows, {"dentalsegmentator:base": "ok"})

    def test_check_narrows_to_a_catalog_or_a_task(self):
        """A named catalog is checked whole, its vanished tasks included; a named task alone."""
        ecos = [self.eco, E.MRSegmentatorEcosystem()]
        data = json.loads(self.path.read_text())
        data["tasks"]["dentalsegmentator:gone"] = data["tasks"]["dentalsegmentator:base"]
        self.path.write_text(json.dumps(data))
        rows = segments.check(self.path, ecosystems=ecos, targets=["dentalsegmentator"])
        self.assertEqual({n: s for n, s, _ in rows},
                         {"dentalsegmentator:base": "ok", "dentalsegmentator:gone": "orphan"})
        rows = segments.check(self.path, ecosystems=ecos, targets=["dentalsegmentator:base"])
        self.assertEqual([(n, s) for n, s, _ in rows], [("dentalsegmentator:base", "ok")])

    def test_an_edit_to_one_ts_task_moves_that_task_alone(self):
        ts = E.TSEcosystem()
        total, fast = ts.label_version("total"), ts.label_version("total_fast")
        _, entries = ts._registry_entries()
        entries["total"] = {**entries["total"], "label_map": {"1": "renamed"}}
        self.assertNotEqual(ts.label_version("total"), total)
        self.assertEqual(ts.label_version("total_fast"), fast)

    def test_an_engines_table_is_versioned_by_its_contents(self):
        eco = E.SynthStripEcosystem()
        before = eco.label_version("mask")
        with mock.patch.object(eco, "_label_table", return_value={1: "Brain", 2: "Skull"}):
            self.assertNotEqual(eco.label_version("mask"), before)

    def test_an_index_of_another_schema_is_refused(self):
        self.path.write_text(json.dumps({"_meta": {"schema_version": 99}, "tasks": {}}))
        with self.assertRaisesRegex(InputError, "schema 99"):
            segments.check(self.path, ecosystems=[self.eco])


class TargetsAreExplicit(unittest.TestCase):
    def test_neither_or_both_is_refused(self):
        with self.assertRaisesRegex(InputError, "--all"):
            segments.plan(())
        with self.assertRaisesRegex(InputError, "not both"):
            segments.plan(("moose",), all_=True)

    def test_a_pinned_version_is_refused(self):
        with self.assertRaisesRegex(InputError, "not mined"):
            segments.plan(("ts.v2:total@v2.0.0",))

    def test_a_renamed_catalog_or_task_names_its_new_form(self):
        with self.assertRaisesRegex(InputError, "ts.v2"):
            segments.plan(("ts:total",))
        with self.assertRaisesRegex(InputError, "asegdkt"):
            segments.plan(("fastsurfer:brain",))

    def test_an_unknown_catalog_or_task_is_refused(self):
        with self.assertRaisesRegex(InputError, "unknown catalog"):
            segments.plan(("nosuch",))
        with self.assertRaisesRegex(InputError, "unknown task"):
            segments.plan(("moose:nosuch",))

    def test_a_catalog_named_with_one_of_its_tasks_is_mined_whole_once(self):
        for order in (("mrsegmentator:base", "mrsegmentator"), ("mrsegmentator", "mrsegmentator:base")):
            [(eco, tasks, whole)] = segments.plan(order)
            self.assertEqual((eco.name, sorted(tasks), whole),
                             ("mrsegmentator", ["base", "body_comp"], True))


class TheShippedIndexIsCurrent(unittest.TestCase):
    """The drift guard: the index ships beside the manifests it describes, and a catalog change
    that is not re-mined fails here - each manifest's version against the record's, two files
    written by two different tools."""

    def test_every_task_of_every_catalog_has_a_current_record(self):
        rows = segments.check(segments.PACKAGED)
        self.assertGreaterEqual(len(rows), 100)
        bad = [f"{n} {s}: {d}" for n, s, d in rows if s != "ok"]
        self.assertFalse(bad, "re-mine with `haversack catalog mine <catalog>`:\n  "
                         + "\n  ".join(bad[:20]))

    def test_every_shipped_id_is_a_valid_duckn_id(self):
        """duckn reserves `.`, `/` and `#` in an id; a model token that used one could not
        become an entity id without a mapping, so the index says so the day one appears."""
        records = segments.load(segments.PACKAGED)["tasks"]
        bad = [(t, s["id"]) for t, r in records.items() for s in r.get("segments") or ()
               if re.search(r"[./#]", s["id"])]
        self.assertEqual(bad, [])


@pytest.mark.slow
class TheShippedIndexMatchesTheArchives(unittest.TestCase):
    """The network half of the drift guard: one task of each remote shape re-mined live must
    give its shipped record back. Out of CI like the other upstream checks."""

    def test_one_task_of_each_shape_re_mines_to_its_shipped_record(self):
        shipped = segments.load(segments.PACKAGED)["tasks"]
        ecos = {e.name: e for e in E.known_ecosystems()}
        reader = segments.HttpReader()
        for name in ("dentalsegmentator:base", "mrsegmentator:body_comp",
                     "totalvibe:pancreas", "monai:spleen_ct_segmentation"):
            with self.subTest(task=name):
                eco_name, _, task = name.partition(":")
                record, problem, _ = segments._mine_one(ecos[eco_name], task, None, reader)
                self.assertIsNone(problem)
                self.assertEqual(record, segments._content(shipped[name]))


SAMPLE = {
    "ts.v2:total": {"ecosystem": "ts.v2", "kind": "segments", "modality": "CT", "segments": [
        {"id": "kidney_right", "value": 2}, {"id": "kidney_left", "value": 3},
        {"id": "pancreas", "value": 7}, {"id": "vertebrae_L1", "value": 27},
        {"id": "vertebrae_C7", "value": 31}, {"id": "vertebrae_T12", "value": 40}]},
    "fastsurfer:asegdkt": {"ecosystem": "fastsurfer", "kind": "segments", "modality": "MR (T1)",
                           "segments": [{"id": "Left-Cerebral-White-Matter", "value": 2}]},
    "moose:x": {"ecosystem": "moose", "kind": "segments", "modality": "CT",
                "segments": [{"id": "Left Kidney", "value": 1}, {"id": "1", "value": 2}]},
    "monai:brats": {"ecosystem": "monai", "kind": "segments", "modality": "MRI",
                    "segments": [{"id": "Tumor core", "value": 1},
                                 {"id": "Whole tumor", "value": 1, "layer": 1}]},
    "voxtell:text": {"ecosystem": "voxtell", "kind": "open", "modality": "CT / MR / PET"},
}


class IdsFoldAndSplit(unittest.TestCase):
    def test_a_key_folds_case_spaces_and_hyphens(self):
        self.assertEqual(segments.fold("Left-Cerebral-White-Matter"), "left_cerebral_white_matter")
        self.assertEqual(segments.fold("  Left  Kidney "), "left_kidney")
        self.assertEqual(segments.fold("a__b--c"), "a_b_c")

    def test_a_letter_digit_token_stays_whole(self):
        self.assertEqual(segments.words("vertebrae_l1"), ["vertebrae", "l1"])


class SearchingTheIndex(unittest.TestCase):
    def setUp(self):
        self.idx = segments.Index(SAMPLE)

    def keys(self, *a, **kw):
        return [g["key"] for g in self.idx.search(*a, **kw)["results"]]

    def test_words_are_prefixes_in_any_order(self):
        self.assertEqual(self.keys("kid left"), ["kidney_left", "left_kidney"])
        self.assertEqual(self.keys("left kid"), ["kidney_left", "left_kidney"])
        self.assertEqual(self.keys("left cereb"), ["left_cerebral_white_matter"])

    def test_the_models_spelling_is_kept_beside_the_key(self):
        groups = {g["key"]: g for g in self.idx.search("left kidney")["results"]}
        self.assertEqual(groups["left_kidney"]["ids"], ["Left Kidney"])
        self.assertEqual(groups["kidney_left"]["ids"], ["kidney_left"])

    def test_a_short_prefix_does_not_reach_inside_a_letter_digit_token(self):
        self.assertEqual(self.keys("vertebrae l"), ["vertebrae_l1"])

    def test_an_exact_key_comes_first(self):
        self.assertEqual(self.keys("vertebrae l1")[0], "vertebrae_l1")
        self.assertEqual(self.keys("pancreas"), ["pancreas"])

    def test_a_glob_is_the_whole_key_and_keeps_its_ranges(self):
        self.assertEqual(self.keys("kidney", mode="glob"), [])
        self.assertEqual(self.keys("kidney*", mode="glob"), ["kidney_left", "kidney_right"])
        # [s-u] is a range: folded like an id it would be [s_u] and match nothing here
        self.assertEqual(self.keys("vertebrae_[s-u]*", mode="glob"), ["vertebrae_t12"])
        self.assertEqual(self.keys("Left-Cerebral*", mode="glob"), ["left_cerebral_white_matter"])

    def test_a_regex_searches_anywhere_ignoring_case(self):
        self.assertEqual(self.keys(r"^vertebrae_[ct]\d+$", mode="regex"),
                         ["vertebrae_c7", "vertebrae_t12"])
        self.assertEqual(self.keys("PANCREAS", mode="regex"), ["pancreas"])

    def test_field_id_matches_the_models_spelling(self):
        self.assertEqual(self.keys("Left *", mode="glob", field="id"), ["left_kidney"])
        self.assertEqual(self.keys("left *", mode="glob", field="id"), [])

    def test_filters_narrow_by_catalog_modality_and_served_tasks(self):
        self.assertEqual(self.keys("left", catalog="moose"), ["left_kidney"])
        self.assertEqual(self.keys("left", modality="mr"), ["left_cerebral_white_matter"])
        self.assertEqual(self.keys("kid", tasks={"moose:x"}), ["left_kidney"])
        self.assertEqual(self.keys("kid", tasks=set()), [])

    def test_a_layered_segment_is_found_with_its_layer(self):
        [g] = self.idx.search("whole tumor")["results"]
        [hit] = g["segments"]
        self.assertEqual((hit["task"], hit["value"], hit["layer"]), ("monai:brats", 1, 1))
        [g] = self.idx.search("tumor core")["results"]
        self.assertNotIn("layer", g["segments"][0])

    def test_open_vocabulary_tasks_are_named_apart(self):
        res = self.idx.search("liver")
        self.assertEqual((res["results"], res["open_vocabulary"]), ([], ["voxtell:text"]))
        self.assertIn("prompt", res["open_vocabulary_note"])
        self.assertEqual(self.idx.search("liver", tasks={"ts.v2:total"})["open_vocabulary"], [])

    def test_limit_truncates_and_says_so(self):
        res = self.idx.search("vertebrae", limit=1)
        self.assertEqual((len(res["results"]), res["truncated"], res["key_count"]), (1, True, 3))

    def test_bad_queries_are_refused_with_the_fix(self):
        for kw, words in (({"query": ""}, "empty"), ({"query": "x" * 201}, "limit"),
                          ({"query": "x", "mode": "fuzzy"}, "use words"),
                          ({"query": "x", "field": "id"}, "glob and regex"),
                          ({"query": "x", "mode": "glob", "field": "name"}, "use key"),
                          ({"query": "(", "mode": "regex"}, "regex"),
                          ({"query": "x", "catalog": "nosuch"}, "unknown catalog"),
                          ({"query": "---"}, "no words"), ({"query": "x", "limit": 0}, "limit")):
            with self.subTest(kw=kw), self.assertRaisesRegex(InputError, words):
                self.idx.search(**kw)

    def test_a_caller_without_regex_is_told_where_regex_is(self):
        with self.assertRaisesRegex(InputError, "--regex"):
            self.idx.search("^x$", mode="regex", allowed_modes=segments.WIRE_MODES)

    def test_exact_regex_minds_case_and_the_folded_key_does_not(self):
        self.assertEqual(self.keys("^Left", mode="regex", field="id"),
                         ["left_cerebral_white_matter", "left_kidney"])
        self.assertEqual(self.keys("^left", mode="regex", field="id"), [])
        self.assertEqual(self.keys("^LEFT", mode="regex"),
                         ["left_cerebral_white_matter", "left_kidney"])

    def test_a_catalog_the_caller_cannot_see_is_not_named(self):
        """A server answers from what it serves; naming the rest of the index in an error told
        its callers about catalogs it does not serve."""
        with self.assertRaisesRegex(InputError, "has no tasks here") as cm:
            self.idx.search("left", catalog="moose", tasks={"ts.v2:total"})
        self.assertNotIn("fastsurfer", str(cm.exception))
        with self.assertRaises(InputError) as cm:
            self.idx.search("left", catalog="nosuch", tasks={"ts.v2:total"})
        self.assertNotIn("moose", str(cm.exception))

    def test_a_records_note_rides_along_with_its_answers(self):
        idx = segments.Index({**SAMPLE, "monai:brats": {**SAMPLE["monai:brats"],
                                                        "note": "its own encoding"}})
        self.assertEqual(idx.search("tumor core")["notes"], {"monai:brats": "its own encoding"})
        self.assertNotIn("notes", idx.search("pancreas"))

    def test_globs_do_not_fill_fnmatchs_pattern_cache(self):
        """fnmatch caches every distinct pattern, 32768 of them; an anonymous caller could fill
        it with 75-200 MiB of compiled globs. The search compiles its own."""
        import fnmatch
        cache = getattr(fnmatch, "_compile_pattern", None)
        if cache is None or not hasattr(cache, "cache_info"):
            self.skipTest("this Python's fnmatch keeps no pattern cache")
        before = cache.cache_info().currsize
        for i in range(50):
            self.idx.search(f"*{i}q*", mode="glob")
        self.assertEqual(cache.cache_info().currsize, before)


class TheIndexIsBuiltFromTheFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmpdir(self)

    def write(self, name, records):
        p = self.tmp / name
        segments.dump(records, p)
        return p

    def test_a_user_record_replaces_the_packaged_one_for_its_task(self):
        a = self.write("a.json", {"ts.v2:total": SAMPLE["ts.v2:total"], "moose:x": SAMPLE["moose:x"]})
        b = self.write("b.json", {"moose:x": {**SAMPLE["moose:x"],
                                              "segments": [{"id": "spleen", "value": 1}]}})
        idx = segments.index(paths=[a, b])
        self.assertEqual([g["key"] for g in idx.search("spleen")["results"]], ["spleen"])
        self.assertEqual([g["key"] for g in idx.search("kid", catalog="moose")["results"]], [])

    def test_a_file_that_does_not_parse_is_refused_naming_the_fix(self):
        """A broken user index reached the caller as a traceback, whose exit status read as
        'nothing matched'."""
        p = self.tmp / "broken.json"
        p.write_text("{not json")
        with self.assertRaisesRegex(InputError, "catalog mine"):
            segments.load(p)
        with self.assertRaisesRegex(InputError, "not a file"):
            segments.load(self.tmp)

    def test_a_malformed_record_names_the_task_to_re_mine(self):
        with self.assertRaisesRegex(InputError, "moose:x.*catalog mine moose:x"):
            segments.Index({"moose:x": {**SAMPLE["moose:x"], "segments": [{"id": "a"}]}})

    def test_haversack_segments_is_where_mine_writes_even_in_a_checkout(self):
        """Setting it is how a run stays off the shipped file; in a checkout it was ignored."""
        with mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": str(self.tmp / "mine.json")}):
            self.assertEqual(segments.target(), self.tmp / "mine.json")

    def test_it_is_built_once_and_again_when_a_file_changes(self):
        a = self.write("a.json", {"moose:x": SAMPLE["moose:x"]})
        first = segments.index(paths=[a])
        self.assertIs(segments.index(paths=[a]), first)
        self.write("a.json", {"ts.v2:total": SAMPLE["ts.v2:total"]})
        second = segments.index(paths=[a])
        self.assertIsNot(second, first)
        self.assertEqual([g["key"] for g in second.search("pancreas")["results"]], ["pancreas"])


class TheCommandLine(unittest.TestCase):
    """`tasks --find` searches the index, `catalog check` checks it, and the flags that only
    narrow a search say so when there is none."""

    def run_cli(self, *argv):
        import contextlib
        from haversack import cli
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": "/nonexistent/segments.json"}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_tasks_find_answers_from_the_index(self):
        rc, out, _ = self.run_cli("tasks", "--find", "pancreas")
        self.assertEqual(rc, 0)
        self.assertIn("ts.v2:total", out)

    def test_tasks_find_with_a_task_searches_that_task_alone(self):
        rc, out, _ = self.run_cli("tasks", "ts.v2:total", "--find", "kid left")
        self.assertEqual(rc, 0)
        tasks = {line.split()[0] for line in out.splitlines() if line.startswith("  ")}
        self.assertEqual(tasks, {"ts.v2:total"})

    def test_nothing_found_exits_1(self):
        rc, out, _ = self.run_cli("tasks", "--find", "zzqqxx")
        self.assertEqual((rc, out), (1, ""))

    def test_a_search_modifier_without_find_is_refused(self):
        rc, _, err = self.run_cli("tasks", "--glob")
        self.assertEqual(rc, 2)
        self.assertIn("--find", err)

    def test_catalog_check_narrows_to_one_catalog(self):
        rc, _, err = self.run_cli("catalog", "check", "cads")
        self.assertEqual(rc, 0, err)
        self.assertIn("9 ok", err)

    def test_catalog_check_refuses_a_file_that_is_not_there(self):
        rc, _, err = self.run_cli("catalog", "check", "--file", "/nonexistent/typo.json")
        self.assertEqual(rc, 2)
        self.assertIn("no such index file", err)

    def test_the_search_modifiers_are_refused_where_they_mean_nothing(self):
        for argv, words in ((("tasks", "--limit", "3"), "--find"),
                            (("tasks", "--find", "x", "--glob", "--regex"), "give one"),
                            (("tasks", "--find", "kidney", "--exact"), "--exact")):
            with self.subTest(argv=argv):
                rc, _, err = self.run_cli(*argv)
                self.assertEqual(rc, 2, err)
                self.assertIn(words, err)

    def test_find_answers_from_the_tasks_this_catalog_lists(self):
        """Engines are off in the suite, so their catalogs are not listed - and --find must not
        name what `tasks TASK` would then call unknown."""
        rc, out, _ = self.run_cli("tasks", "--find", "*tumor*", "--glob")
        self.assertNotIn("monai:", out)

    def test_a_task_not_installed_prints_its_segments_from_the_index(self):
        empty = _tmpdir(self)
        rc, out, err = self.run_cli("tasks", "cads:organs", "--model-root", str(empty))
        self.assertEqual(rc, 0, err)
        self.assertIn("from the segments index", err)
        self.assertTrue(out.splitlines()[0].startswith("1\t"), out)


class TheShippedIndexAnswersRealQuestions(unittest.TestCase):
    """A floor on the real data, not a snapshot of it: questions whose answer cannot go away
    without a catalog losing a segment everyone relies on."""

    def setUp(self):
        self.idx = segments.index(paths=[segments.PACKAGED])

    def tasks_for(self, *a, **kw):
        return {s["task"] for g in self.idx.search(*a, **kw)["results"] for s in g["segments"]}

    def test_the_pancreas_is_produced_by_several_catalogs(self):
        tasks = self.tasks_for("pancreas", mode="glob")
        self.assertIn("ts.v2:total", tasks)
        self.assertGreaterEqual(len({t.partition(":")[0] for t in tasks}), 4)

    def test_word_order_finds_both_conventions(self):
        keys = {g["key"] for g in self.idx.search("kid left")["results"]}
        self.assertIn("kidney_left", keys)

    def test_voxtell_is_listed_as_open_vocabulary(self):
        self.assertIn("voxtell:text", self.idx.search("liver")["open_vocabulary"])
