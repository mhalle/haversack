"""A result key names a catalog's model folder the same way on every machine (2026-09-26).

A task that installs a model FOLDER - MOOSE, MRSegmentator, DentalSegmentator, TotalVibe, CADS -
names its weights by the folder's absolute path, and ``weights_versions_of`` put that path into
every such result's key: ``/Users/<someone>/.totalsegmentator/nnunet/results/moose/Dataset444_Ribs/
...=08052025`` on one laptop, ``/weights/moose/...`` on Modal. A result cache is portable (keys
hold no app name, AGENTS.md), and these keys were not: two machines with different weights roots
computed the same bytes under two keys. Under the weights root a folder is now named relative to
it; a dataset id, and a caller's own folder outside every root, are left as they were.
"""
import json
import os
import unittest
from pathlib import Path

from haversack import serve
from haversack.segmenter import Segmenter
from haversack.weights_fetch import _write_sidecar


def _install_ribs(root: Path) -> Path:
    """What a MOOSE install leaves: the dataset folder, one model folder, its version sidecar."""
    dataset = root / "moose" / "Dataset444_Ribs"
    model = dataset / "nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    (model / "fold_all").mkdir(parents=True)
    (model / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "rib_left_1": 1},
        "numTraining": 1, "file_ending": ".nii.gz"}))
    (model / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
    _write_sidecar(dataset, "clin_ct_ribs", "08052025", {"url": "https://example.invalid/x.zip"}, None)
    return dataset


class TestThePortableKey(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def _key(self, root, task="moose:clin_ct_ribs"):
        return serve.weights_versions_of(Segmenter(device="cpu", weights=root), task)

    def test_one_install_under_two_roots_is_one_key(self):
        a, b = self.tmp / "laptop" / "results", self.tmp / "weights"
        _install_ribs(a), _install_ribs(b)
        ka, kb = self._key(a), self._key(b)
        self.assertEqual(ka, kb)
        self.assertIn("moose/Dataset444_Ribs/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__"
                      "3d_fullres=08052025", ka)
        self.assertFalse([v for v in ka if str(self.tmp) in v or v.startswith("/")], ka)

    def test_a_root_reached_through_a_symlink_is_the_same_root(self):
        """macOS's /tmp is /private/tmp: a root spelled one way and a folder resolved the
        other must still be one tree."""
        real = self.tmp / "real"
        _install_ribs(real)
        link = self.tmp / "link"
        os.symlink(real, link)
        self.assertEqual(self._key(link), self._key(real))
        self.assertFalse([v for v in self._key(link) if v.startswith("/")])

    def test_the_version_still_decides(self):
        a, b = self.tmp / "a", self.tmp / "b"
        _install_ribs(a)
        d = _install_ribs(b)
        _write_sidecar(d, "clin_ct_ribs", "09092099", {"url": "https://example.invalid/y.zip"}, None)
        self.assertNotEqual(self._key(a), self._key(b))

    def test_a_dataset_id_is_left_as_it_was(self):
        root = self.tmp / "ts"
        (root / "Dataset297_TotalSegmentator_total_3mm_1559subj" /
         "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres").mkdir(parents=True)
        key = self._key(root, "ts.v2:total_fast")
        self.assertTrue(key[0].startswith("297="), key)

    def test_a_folder_outside_every_root_keeps_its_path(self):
        """A caller's own model folder is identified by nothing but where it is."""
        seg = Segmenter(device="cpu", weights=self.tmp / "root")
        outside = self.tmp / "mine" / "Dataset900_Toy" / "trainer__plans__3d_fullres"
        self.assertEqual(seg._portable_id(str(outside)), str(outside))
        self.assertEqual(seg._portable_id(297), "297")
        self.assertEqual(seg._portable_id("Dataset297_x"), "Dataset297_x")
