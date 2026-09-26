"""MOOSE models get the orientation they were trained in, and moosez's crop workflow (2026-09-26).

Until that day every MOOSE model was fed the input's own axis order, as a stock nnU-Net folder
is - and none was trained that way. moosez fed dicom2nifti's LAS until 2025-07-18 and RAS since,
when six models were retrained for it. A user found it: on LPS DICOM ``clin_ct_ribs`` swapped
left and right and lost the posterior ribs (170 ml of rib where MOOSE's own result has 344).
Measured on an NLST chest CT against ``ts.v2:total``: organs mean Dice 0.00 as fed, 0.91 in RAS;
lungs 0.57 as fed, 0.99 in LAS and 0.00 in RAS - so neither "always RAS" (what current moosez
does, and upstream issue #232 reports lungs flipped) nor the old behavior is right. Each task's
orientation is stated in the manifest, never read off a file name.

``clin_ct_body_composition`` is moosez's one crop workflow: ``clin_ct_fast_vertebrae`` labels the
image, the image is cut to the superior-inferior extent of L1-L5, body composition runs on the
cut, and its result keeps only the band of L3's largest connected piece.
"""
import json
import unittest

import numpy as np
import pytest

from haversack.ecosystems import MooseEcosystem
from haversack.pipeline import band_of, crop_axes_only, keep_band
from haversack.grid import Grid
from haversack.tasks import CascadeStep, TaskSpec, _check_cascade, array_axis


def _fake_model_folder(root, name, labels, trainer="nnUNetTrainer"):
    d = root / name / f"{trainer}__nnUNetPlans__3d_fullres"
    (d / "fold_all").mkdir(parents=True)
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"}, "labels": {"background": 0, **labels},
        "numTraining": 1, "file_ending": ".nii.gz"}))
    (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
    return root / name


VERTEBRAE = {**{f"vertebra_T{i}": 7 + i for i in range(1, 13)},
             **{f"vertebra_L{i}": 19 + i for i in range(1, 6)}}


class TestTheAnatomicalAxis(unittest.TestCase):
    """An axis is named anatomically and found in the orientation the labels are in: which
    ARRAY axis runs head to foot depends on it, so an index would be right for one only."""

    def test_the_array_axis_follows_the_orientation(self):
        # DICOMOrient codes name where image axes x, y, z point; array axis k is image axis 2 - k
        self.assertEqual(array_axis("LPS", "SI"), 0)
        self.assertEqual(array_axis("RAS", "SI"), 0)
        self.assertEqual(array_axis("RAS", "RL"), 2)
        self.assertEqual(array_axis("LAS", "AP"), 1)
        self.assertEqual(array_axis("SLP", "SI"), 2)
        self.assertEqual(array_axis("PIR", "SI"), 1)
        self.assertEqual(array_axis("ipr", "SI"), 2)       # any spelling of a code

    def test_what_is_not_an_axis_or_an_orientation_is_refused(self):
        for code, axis in (("LPS", "XY"), ("LPQ", "SI"), ("LP", "SI"), (None, "SI"), ("SSP", "SI"), ("IAS", "SI")):
            with self.subTest(code=code, axis=axis), self.assertRaises(ValueError):
                array_axis(code, axis)


class TestTheBox(unittest.TestCase):
    def test_only_the_named_axes_are_cut(self):
        box = ((3, 4, 5), (7, 8, 9))
        self.assertEqual(crop_axes_only(box, (10, 11, 12), [0]), ((3, 0, 0), (7, 11, 12)))
        self.assertEqual(crop_axes_only(box, (10, 11, 12), [2]), ((0, 0, 5), (10, 11, 9)))

    def test_no_axes_is_the_whole_box_and_no_box_stays_none(self):
        box = ((3, 4, 5), (7, 8, 9))
        self.assertEqual(crop_axes_only(box, (10, 11, 12), []), box)
        self.assertIsNone(crop_axes_only(None, (10, 11, 12), [0]))


class TestTheBand(unittest.TestCase):
    """moosez's restrict_fov: the band the classes span, of their largest connected piece."""

    def test_the_band_of_the_largest_piece(self):
        labels = np.zeros((20, 6, 6), np.uint8)
        labels[2:4, 1:3, 1:3] = 22          # 8 voxels
        labels[10:15, 1:4, 1:4] = 22        # 45 voxels: the largest
        self.assertEqual(band_of(labels, [22], [0], largest_component=True), {0: (10, 15)})
        self.assertEqual(band_of(labels, [22], [0], largest_component=False), {0: (2, 15)})

    def test_pieces_that_touch_only_at_an_edge_are_two(self):
        """SimpleITK's ConnectedComponent defaults to face connectivity, as scipy's label does:
        two blocks meeting at an edge are separate pieces."""
        labels = np.zeros((10, 10, 10), np.uint8)
        labels[1:3, 1:3, 1:3] = 1                    # 8 voxels
        labels[3:6, 3:6, 1:4] = 1                    # 27, touching the first along an edge only
        self.assertEqual(band_of(labels, [1], [0], largest_component=True), {0: (3, 6)})

    def test_absent_classes_are_no_band(self):
        self.assertIsNone(band_of(np.zeros((4, 4, 4), np.uint8), [22], [0],
                                  largest_component=True))

    def test_the_band_keeps_exactly_its_slices_on_its_own_grid(self):
        g = Grid((10, 4, 4), (1.25, 0.7, 0.7), (5.0, 0.0, 0.0))
        out = keep_band(np.ones(g.shape, np.uint8), {0: (3, 7)}, g, g)
        self.assertEqual(np.nonzero(out.any(axis=(1, 2)))[0].tolist(), [3, 4, 5, 6])

    def test_the_band_carries_over_in_millimeters_to_another_grid(self):
        src = Grid((10, 4, 4), (2.0, 1.0, 1.0), (0.0, 0.0, 0.0))     # slice k spans 2k-1 .. 2k+1
        fine = Grid((20, 4, 4), (1.0, 1.0, 1.0), (-0.5, 0.0, 0.0))   # voxel j centered at j - 0.5
        out = keep_band(np.ones(fine.shape, np.uint8), {0: (3, 5)}, src, fine)   # 5 mm .. 9 mm
        self.assertEqual(np.nonzero(out.any(axis=(1, 2)))[0].tolist(), [6, 7, 8, 9])

    def test_a_torch_tensor_is_restricted_in_place(self):
        torch = pytest.importorskip("torch")
        g = Grid((6, 2, 2))
        out = keep_band(torch.ones(g.shape, dtype=torch.uint8), {0: (2, 4)}, g, g)
        self.assertEqual(out.any(dim=2).any(dim=1).nonzero().flatten().tolist(), [2, 3])


class TestTheCascadeRules(unittest.TestCase):
    def test_a_final_stage_cannot_cut_or_band(self):
        for kw in ({"crop_axes": ("SI",)}, {"band_classes": (1,)}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                _check_cascade((CascadeStep(weights_id=1, crop_to_classes=(1,)),
                                CascadeStep(weights_id=2, **kw)), "t")

    def test_an_unknown_axis_or_a_repeated_one_is_refused(self):
        for axes in (("XY",), ("SI", "SI")):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                _check_cascade((CascadeStep(weights_id=1, crop_to_classes=(1,), crop_axes=axes),
                                CascadeStep(weights_id=2)), "t")

    def test_a_largest_component_of_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            _check_cascade((CascadeStep(weights_id=1, crop_to_classes=(1,),
                                        band_largest_component=True),
                            CascadeStep(weights_id=2)), "t")


class TestTheCatalog(unittest.TestCase):
    """The orientation is the manifest's, per task; the workflow is built from it."""

    def setUp(self):
        import tempfile
        self.root = __import__("pathlib").Path(tempfile.mkdtemp())
        self.eco = MooseEcosystem()

    def _install(self, task, labels):
        _fake_model_folder(self.root / "moose", self.eco._entries[task]["folder"], labels)

    def test_each_task_runs_in_the_orientation_its_manifest_states(self):
        for task, want in (("clin_ct_ribs", "RAS"), ("clin_ct_lungs", "LAS"),
                           ("clin_ct_dental", None)):
            with self.subTest(task=task):
                self._install(task, {"a": 1})
                self.assertEqual(self.eco.spec(task, self.root).orientation, want)
                info = self.eco.info(task, self.root)
                self.assertEqual(info["model_orientation"], want or "native")
                self.assertTrue(info["orientation_basis"])

    def test_every_shipped_task_states_one(self):
        for task, entry in self.eco._entries.items():
            with self.subTest(task=task):
                self.assertIn(entry.get("model_orientation"), {"RAS", "LAS", "native"})

    def test_a_manifest_stating_none_is_refused_rather_than_guessed(self):
        self._install("clin_ct_ribs", {"a": 1})
        del self.eco._entries["clin_ct_ribs"]["model_orientation"]
        from haversack.errors import ModelNotFound
        with self.assertRaisesRegex(ModelNotFound, "states no model orientation"):
            self.eco.spec("clin_ct_ribs", self.root)

    def test_body_composition_is_moosez_workflow(self):
        self._install("clin_ct_fast_vertebrae", VERTEBRAE)
        self.assertFalse(self.eco.materialized("clin_ct_body_composition", self.root))
        self._install("clin_ct_body_composition", {"skeletal_muscle": 1, "subcutaneous_fat": 2,
                                                   "visceral_fat": 3})
        self.assertTrue(self.eco.materialized("clin_ct_body_composition", self.root))
        spec = self.eco.spec("clin_ct_body_composition", self.root)
        self.assertEqual(spec.shape, "cascade")
        self.assertEqual(spec.orientation, "LAS")
        crop, final = spec.cascade
        self.assertIn("Dataset112_FastVertebrae", str(crop.weights_id))
        self.assertIn("Dataset778_Body_composition", str(final.weights_id))
        self.assertEqual(crop.crop_to_classes, (20, 21, 22, 23, 24))
        self.assertEqual((crop.dilation_mm, crop.crop_axes), (0.0, ("SI",)))
        self.assertEqual((crop.band_classes, crop.band_largest_component), ((22,), True))
        self.assertEqual(spec.label_map, {1: "skeletal_muscle", 2: "subcutaneous_fat",
                                          3: "visceral_fat"})
        self.assertEqual(self.eco.stage_task("clin_ct_body_composition", 0),
                         "clin_ct_fast_vertebrae")

    def test_a_crop_checkpoint_that_names_the_values_otherwise_is_refused(self):
        """moosez states values (20-24, 22); a crop model whose release numbers its vertebrae
        otherwise would cut to something else - refused, never followed."""
        shifted = {n: v + 1 for n, v in VERTEBRAE.items()}
        self._install("clin_ct_fast_vertebrae", shifted)
        self._install("clin_ct_body_composition", {"skeletal_muscle": 1})
        from haversack.errors import ModelNotFound
        with self.assertRaisesRegex(ModelNotFound, "crop_classes"):
            self.eco.spec("clin_ct_body_composition", self.root)

    def test_a_crop_model_stated_in_another_orientation_is_refused(self):
        """moosez runs both models of a workflow on one reoriented image; a manifest that
        states them differently cannot be run as one cascade and is refused."""
        self._install("clin_ct_fast_vertebrae", VERTEBRAE)
        self._install("clin_ct_body_composition", {"skeletal_muscle": 1})
        self.eco._entries["clin_ct_fast_vertebrae"]["model_orientation"] = "RAS"
        from haversack.errors import ModelNotFound
        with self.assertRaisesRegex(ModelNotFound, "one cascade runs one orientation"):
            self.eco.spec("clin_ct_body_composition", self.root)

    def test_installing_the_workflow_installs_its_crop_model(self):
        calls = []
        from haversack import ecosystems
        orig = ecosystems.ZipManifestEcosystem.ensure
        try:
            ecosystems.ZipManifestEcosystem.ensure = (
                lambda self, task, root, progress=None, version=None: calls.append((task, version)))
            self.eco.ensure("clin_ct_body_composition", self.root, version="05092024")
        finally:
            ecosystems.ZipManifestEcosystem.ensure = orig
        self.assertEqual(calls, [("clin_ct_fast_vertebrae", None),
                                 ("clin_ct_body_composition", "05092024")])


class TestTheKey(unittest.TestCase):
    """A stated reorientation joins the key, so a result computed in the wrong orientation is
    never served for the right one; nothing else's key moves."""

    def _versions(self, info):
        from haversack import serve

        class _S:
            def describe(self, task):
                return {"weights_installed": [{"id": "444", "version": "08052025"}], **info}

            def engine_for(self, task):
                from haversack.engines.registry import ENGINES, NNUNETV2
                return ENGINES[NNUNETV2]
        return serve.weights_versions_of(_S(), "moose:clin_ct_ribs")

    def test_a_stated_orientation_joins_the_key(self):
        self.assertIn("orient=RAS", self._versions({"model_orientation": "RAS"}))
        self.assertIn("orient=LAS", self._versions({"model_orientation": "LAS"}))

    def test_native_and_unstated_keep_the_key_they_had(self):
        base = self._versions({})
        self.assertEqual(self._versions({"model_orientation": "native"}), base)
        self.assertFalse([v for v in base if v.startswith("orient=")])


# --- through segment(): the axis is resolved against the task's orientation ------------------

pytest.importorskip("nnunetv2")
from test_cascade_union import _Part, _run, _StubModel          # noqa: E402
from test_normalization_sharing import ORGANS, RIBS              # noqa: E402


class _Layout(_StubModel):
    """A crop stage whose classes sit where the test says, on the canonical model grid."""

    def __init__(self, layout, K=4):
        super().__init__(ORGANS._props, K=K)
        self.layout = layout

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((self.K, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        for cls, where in self.layout.items():
            logits[cls][where] = 2.0
        return logits


def _moose_cascade(orientation, *, band=(), largest=False):
    return TaskSpec(name="stub_moose", lineage="nnunetv2", shape="cascade", label_map={1: "a"},
                    orientation=orientation,
                    cascade=(CascadeStep(weights_id=1, crop_to_classes=(1, 2), dilation_mm=0.0,
                                         crop_axes=("SI",), band_classes=band,
                                         band_largest_component=largest),
                             CascadeStep(weights_id=2)))


ALL = (slice(None),) * 3


def test_the_cut_is_along_the_superior_inferior_axis_of_the_tasks_orientation(tmp_path, monkeypatch):
    """In RAS head-to-foot is array axis 0; in SLP it is array axis 2. The stub CT is 16 x 20 x
    18 (z, y, x); the crop class spans 4 of the SI axis's voxels and the final stage must see
    exactly those 4 across the full extent of the other two."""
    for orientation, si in (("RAS", 0), ("SLP", 2)):
        where = [slice(None)] * 3
        where[si] = slice(4, 8)
        crop = _Layout({1: tuple(where)})
        final = _Part(RIBS._props, cover=ALL)
        res, shape = _run(tmp_path, monkeypatch, [crop, final], _moose_cascade(orientation))
        full = tuple(crop.received.shape[1:])
        want = list(full)
        want[si] = 4
        assert tuple(final.received.shape[1:]) == tuple(want), (orientation, full)
        assert res.provenance["crops"][0]["axes"] == ["SI"]


def test_the_result_keeps_only_the_band_of_the_largest_piece(tmp_path, monkeypatch):
    """Crop class 1 spans z 2..14; the band class 2 has two pieces, z 3..4 (small) and z 9..12
    (large); the final stage labels everything it sees. Only z 9..12 survives."""
    crop = _Layout({1: (slice(2, 15), slice(0, 2), slice(0, 2)),
                    2: (slice(9, 13), slice(8, 14), slice(8, 14))})
    crop.layout = {**crop.layout}
    small = (slice(3, 5), slice(16, 18), slice(15, 17))

    class _TwoPieces(_Layout):
        def predict_logits(self, x, report=None):
            logits = super().predict_logits(x, report)
            logits[2][small] = 2.0
            return logits

    crop = _TwoPieces(crop.layout)
    final = _Part(RIBS._props, cover=ALL)
    res, _ = _run(tmp_path, monkeypatch, [crop, final],
                  _moose_cascade("RAS", band=(2,), largest=True))
    rows = np.nonzero(res.array.any(axis=(1, 2)))[0].tolist()
    assert rows == [9, 10, 11, 12], rows
    band = res.provenance["crops"][0]["band"]
    assert band["largest_component"] is True and band["slices"] == {"0": [9, 13]}


def test_without_largest_component_the_band_spans_every_piece(tmp_path, monkeypatch):
    small = (slice(3, 5), slice(16, 18), slice(15, 17))

    class _TwoPieces(_Layout):
        def predict_logits(self, x, report=None):
            logits = super().predict_logits(x, report)
            logits[2][small] = 2.0
            return logits

    crop = _TwoPieces({1: (slice(2, 15), slice(0, 2), slice(0, 2)),
                       2: (slice(9, 13), slice(8, 14), slice(8, 14))})
    final = _Part(RIBS._props, cover=ALL)
    res, _ = _run(tmp_path, monkeypatch, [crop, final], _moose_cascade("RAS", band=(2,)))
    rows = np.nonzero(res.array.any(axis=(1, 2)))[0].tolist()
    assert rows == list(range(3, 13)), rows


def test_absent_band_classes_restrict_nothing(tmp_path, monkeypatch):
    crop = _Layout({1: (slice(2, 15), slice(0, 2), slice(0, 2))})
    final = _Part(RIBS._props, cover=ALL)
    res, _ = _run(tmp_path, monkeypatch, [crop, final],
                  _moose_cascade("RAS", band=(3,), largest=True))
    rows = np.nonzero(res.array.any(axis=(1, 2)))[0].tolist()
    assert rows == list(range(2, 15))              # the cut, whole
    assert res.provenance["crops"][0]["band"]["slices"] is None
