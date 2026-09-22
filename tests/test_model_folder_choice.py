"""A dataset holding two model folders of one configuration is refused, never chosen.

TotalSegmentator's ``v3.0.0-weights`` (2026-09) ships Datasets 831-836 with TWO ``3d_fullres``
folders each, read out of the zips' central directories on 2026-09-21:

    nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres               upstream's default
    nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans_8__3d_fullres  upstream's model_size="small"

``resolve_model_folder`` keyed the folders by configuration alone
(``{c.name.rsplit("__", 1)[1]: c}``), so the second replaced the first and the small ResEnc
model ran under the default's name with nothing said. Every test here builds that layout and
fails against the old resolver: it either expects a refusal the old code never made, or passes
the ``plans``/``trainer`` choice the old code had no way to take.

The choice must reach EVERY resolve of a weights id - describe (which feeds the result key's
weights component), warm, the orientation decision, and the pipeline's own model load - so
each of those doors has a test that tells the two folders apart by what they hold.
"""
import json
import tempfile
import unittest
from pathlib import Path

from haversack.errors import AmbiguousModel, ModelNotFound
from haversack.tasks import TaskCatalog, TaskSpec, resolve_model_folder
from haversack.weights import WeightsStore

T = "nnUNetTrainerNoMirroring"
DEFAULT = f"{T}__nnUNetPlans__3d_fullres"
SMALL = f"{T}__nnUNetResEncUNetLPlans_8__3d_fullres"


def _dataset(root: Path, folders, *, wid=831, readers=None, tags=None) -> Path:
    """``Dataset<wid>_x/<folder>/`` for each folder, each holding a dataset.json, a plans.json
    naming ``readers[folder]`` and a version sidecar tagged ``tags[folder]`` - so a test can
    tell from what comes back WHICH folder a door resolved."""
    ds = root / f"Dataset{wid}_TotalSegmentator_part1_organs_1830subj"
    for name in folders:
        f = ds / name
        (f / "fold_0").mkdir(parents=True)
        (f / "dataset.json").write_text(json.dumps(
            {"labels": {"background": 0, "spleen": 1}, "channel_names": {"0": "CT"}}))
        (f / "plans.json").write_text(json.dumps(
            {"image_reader_writer": (readers or {}).get(name, "NibabelIOWithReorient")}))
        (f / ".haversack-version.json").write_text(json.dumps(
            {"id": str(wid), "tag": (tags or {}).get(name, name), "sha256": None}))
    return ds


class ResolverRefusesAmbiguity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_plans_of_one_configuration_are_refused_by_id(self):
        _dataset(self.root, [DEFAULT, SMALL])
        with self.assertRaises(AmbiguousModel) as cm:
            resolve_model_folder(831, model_root=self.root)
        msg = str(cm.exception)
        self.assertIn(DEFAULT, msg)
        self.assertIn(SMALL, msg)
        self.assertIn("plans", msg)            # the remedy is named

    def test_refused_when_the_configuration_is_named_too(self):
        # naming the configuration does not narrow two folders that share it
        _dataset(self.root, [DEFAULT, SMALL])
        with self.assertRaises(AmbiguousModel):
            resolve_model_folder(831, model_root=self.root, configuration="3d_fullres")

    def test_refused_through_a_dataset_folder_path(self):
        ds = _dataset(self.root, [DEFAULT, SMALL])
        with self.assertRaises(AmbiguousModel):
            resolve_model_folder(ds)

    def test_ambiguity_is_a_model_not_found(self):
        # so info()'s `unresolved` and describe()'s not-installed paths report it
        self.assertTrue(issubclass(AmbiguousModel, ModelNotFound))

    def test_plans_picks_each_folder(self):
        ds = _dataset(self.root, [DEFAULT, SMALL])
        self.assertEqual(resolve_model_folder(831, model_root=self.root, plans="nnUNetPlans"),
                         ds / DEFAULT)
        self.assertEqual(resolve_model_folder(831, model_root=self.root,
                                              plans="nnUNetResEncUNetLPlans_8"), ds / SMALL)

    def test_trainer_narrows_when_plans_are_shared(self):
        other = f"nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
        ds = _dataset(self.root, [DEFAULT, other])
        with self.assertRaises(AmbiguousModel):
            resolve_model_folder(831, model_root=self.root, plans="nnUNetPlans")
        self.assertEqual(resolve_model_folder(831, model_root=self.root, trainer=T,
                                              plans="nnUNetPlans"), ds / DEFAULT)

    def test_a_stated_choice_the_dataset_lacks_is_not_found(self):
        _dataset(self.root, [DEFAULT, SMALL])
        with self.assertRaises(ModelNotFound) as cm:
            resolve_model_folder(831, model_root=self.root, plans="nnUNetResEncUNetMPlans")
        self.assertNotIsInstance(cm.exception, AmbiguousModel)
        self.assertIn(DEFAULT, str(cm.exception))   # says what IS there

    def test_a_model_folder_contradicting_the_choice_is_refused(self):
        ds = _dataset(self.root, [DEFAULT, SMALL])
        self.assertEqual(resolve_model_folder(ds / SMALL), ds / SMALL)      # no choice: as given
        with self.assertRaises(ModelNotFound):
            resolve_model_folder(ds / SMALL, plans="nnUNetPlans")

    def test_one_folder_still_resolves_without_a_choice(self):
        ds = _dataset(self.root, [DEFAULT])
        self.assertEqual(resolve_model_folder(831, model_root=self.root), ds / DEFAULT)

    def test_the_store_passes_the_choice_through(self):
        ds = _dataset(self.root, [DEFAULT, SMALL])
        store = WeightsStore(self.root, fetch=False)
        with self.assertRaises(AmbiguousModel):
            store.resolve(831)
        self.assertEqual(store.resolve(831, plans="nnUNetPlans"), ds / DEFAULT)
        self.assertEqual(store.resolve(ds, plans="nnUNetResEncUNetLPlans_8"), ds / SMALL)


class RegistryStatesTheModel(unittest.TestCase):
    def _catalog(self, entry) -> TaskCatalog:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"tasks": [entry]}, tmp)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return TaskCatalog("ts", path=tmp.name)

    def test_models_is_read_and_keyed_canonically(self):
        cat = self._catalog({"name": "t", "shape": "single", "single": 831,
                             "models": {"0831": {"trainer": T, "plans": "nnUNetPlans"}},
                             "label_map": {"1": "spleen"}})
        spec = cat.get("t")
        self.assertEqual(spec.model_choice(831), {"trainer": T, "plans": "nnUNetPlans"})
        self.assertEqual(spec.model_choice("831"), {"trainer": T, "plans": "nnUNetPlans"})
        self.assertEqual(spec.model_choice("0831"), {"trainer": T, "plans": "nnUNetPlans"})
        self.assertEqual(spec.model_choice(832), {})

    def test_an_unknown_key_is_refused_on_load(self):
        with self.assertRaises(ValueError) as cm:
            self._catalog({"name": "t", "shape": "single", "single": 831,
                           "models": {"831": {"plan": "nnUNetPlans"}}})
        self.assertIn("plan", str(cm.exception))


def _spec(**kw) -> TaskSpec:
    base = dict(name="v3probe", lineage="ts", shape="single", single=831,
                label_map={1: "spleen"},
                models={"831": {"trainer": T, "plans": "nnUNetPlans"}})
    base.update(kw)
    return TaskSpec(**base)


class EveryDoorCarriesTheChoice(unittest.TestCase):
    """Each door that resolves a weights id, told apart by what the chosen folder holds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = _dataset(self.root, [DEFAULT, SMALL],
                           tags={DEFAULT: "v3.0.0-weights", SMALL: "SMALL-RAN"},
                           readers={DEFAULT: "NibabelIOWithReorient", SMALL: "SimpleITKIO"})

    def tearDown(self):
        self.tmp.cleanup()

    def _segmenter(self):
        from haversack.segmenter import Segmenter

        class _Catalog:                      # a TaskSpec passes through _resolve_spec
            def get(self, name):
                return name
        return Segmenter(weights=WeightsStore(self.root, fetch=False), catalog=_Catalog())

    def test_describe_reports_the_stated_folder_installed(self):
        # describe() feeds the result key's weights component: an unresolvable id there is
        # "unknown", and the small model's sidecar would key a result as the default's
        entries = self._segmenter().describe(_spec())["weights_installed"]
        self.assertEqual(entries, [{"id": "831", "installed": True,
                                    "version": "v3.0.0-weights", "sha256": None}])

    def test_describe_without_a_choice_reports_not_installed_rather_than_guess(self):
        entries = self._segmenter().describe(_spec(models={}))["weights_installed"]
        self.assertEqual(entries, [{"id": "831", "installed": False}])

    def test_warm_loads_the_stated_folder(self):
        seg = self._segmenter()
        loaded = []

        class _Models:
            def get(self, folder, **kw):
                loaded.append(Path(folder))

            def __len__(self):
                return len(loaded)
        seg.models = _Models()
        seg.policy["weights"] = WeightsStore(self.root, fetch=False)
        seg.warm(_spec())
        self.assertEqual(loaded, [self.ds / DEFAULT])

    def test_orientation_follows_the_stated_folders_reader(self):
        from haversack import io as nio
        from haversack.pipeline import canonical_orientation_for
        store = WeightsStore(self.root, fetch=False)
        spec = _spec(lineage="nnunetv2")
        self.assertEqual(canonical_orientation_for(spec, store), nio.CANONICAL)
        small = _spec(lineage="nnunetv2",
                      models={"831": {"trainer": T, "plans": "nnUNetResEncUNetLPlans_8"}})
        self.assertIsNone(canonical_orientation_for(small, store))

    def test_the_pipeline_loads_the_stated_folder(self):
        import numpy as np
        import SimpleITK as sitk
        from haversack.pipeline import segment

        class _Stop(Exception):
            pass
        loaded = []

        class _Models:
            def get(self, folder, **kw):
                loaded.append(Path(folder))
                raise _Stop

        img = self.root / "ct.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), np.int16)), str(img))
        with self.assertRaises(_Stop):
            segment(img, _spec(), weights=WeightsStore(self.root, fetch=False),
                    models=_Models(), device="cpu", dtype="fp32")
        self.assertEqual(loaded, [self.ds / DEFAULT])

    def test_a_cascade_stage_uses_its_own_tasks_choice(self):
        # a crop-from task is resolved from the catalog and run under ITS spec; the choice
        # for its model is that spec's, not the outer task's (which here states nothing)
        import numpy as np
        import SimpleITK as sitk
        from haversack.pipeline import segment
        from haversack.tasks import CascadeStep

        class _Stop(Exception):
            pass
        loaded = []

        class _Models:
            def get(self, folder, **kw):
                loaded.append(Path(folder))
                raise _Stop
        sub = _spec(name="sub")

        class _Catalog:
            def get(self, name, progress=None):
                return sub if name == "sub" else name
        outer = _spec(name="outer", shape="cascade", single=None, models={},
                      cascade=(CascadeStep(crop_from_task="sub", crop_to_classes=(1,)),
                               CascadeStep(weights_id=831)))
        img = self.root / "ct.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), np.int16)), str(img))
        with self.assertRaises(_Stop):
            segment(img, outer, catalog=_Catalog(), weights=WeightsStore(self.root, fetch=False),
                    models=_Models(), device="cpu", dtype="fp32")
        self.assertEqual(loaded, [self.ds / DEFAULT])

    def test_the_pipeline_refuses_without_a_choice(self):
        import numpy as np
        import SimpleITK as sitk
        from haversack.pipeline import segment

        class _Models:
            def get(self, folder, **kw):
                raise AssertionError(f"loaded {folder} from an ambiguous dataset")

        img = self.root / "ct.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), np.int16)), str(img))
        with self.assertRaises(AmbiguousModel):
            segment(img, _spec(models={}), weights=WeightsStore(self.root, fetch=False),
                    models=_Models(), device="cpu", dtype="fp32")


if __name__ == "__main__":
    unittest.main()
