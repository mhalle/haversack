"""A task may state its sliding-window tile step; ts.v3 states upstream's 0.8.

TotalSegmentator tiles ``total``, ``total_v3`` and ``total_mr`` at 0.8 and everything else at
nnU-Net's 0.5 (nnunet.py). haversack ran 0.5 everywhere; matching 0.8 took ts.v3:total from
99.86 % to 99.98 % voxel agreement with upstream on an IDC CT (2026-09-21), so ts.v3's
registry states it. ts.v2 does NOT - its cached results were computed at 0.5 - and every
test here that sets a step also checks a task stating none is exactly as before.

The step changes the output, so it must reach the network (through the pipeline and warm),
separate warm models built at different steps, and enter the result key.
"""
import json
import tempfile
import unittest
from pathlib import Path

from haversack.tasks import TaskCatalog, TaskSpec
from haversack.weights import WeightsStore

FOLDER = "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"


def _install(root: Path) -> Path:
    f = root / "Dataset836_TotalSegmentator_total_3mm_1559subj" / FOLDER
    (f / "fold_0").mkdir(parents=True)
    (f / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}}))
    (f.parent / ".haversack-version.json").write_text(json.dumps({"tag": "v3.0.0-weights"}))
    return f


def _spec(step=0.8) -> TaskSpec:
    return TaskSpec(name="stepprobe", lineage="ts", shape="single", single=836,
                    label_map={1: "spleen"}, step_size=step)


class TheRegistries(unittest.TestCase):
    def test_ts_v3_states_upstreams_step_and_ts_v2_states_none(self):
        from haversack.ecosystems import TSEcosystem, TSv3Ecosystem
        v3, v2 = TSv3Ecosystem(), TSEcosystem()
        self.assertEqual({t: v3.spec(t, None).step_size for t in v3.tasks()},
                         dict.fromkeys(v3.tasks(), 0.8))
        self.assertEqual({t for t in v2.tasks() if v2.spec(t, None).step_size is not None}, set())

    def test_a_step_outside_zero_to_one_is_refused_on_load(self):
        for bad in (0, 1.5, -0.8):
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                json.dump({"tasks": [{"name": "t", "single": 1, "step_size": bad}]}, tmp)
            self.addCleanup(Path(tmp.name).unlink)
            with self.assertRaises(ValueError):
                TaskCatalog("ts", path=tmp.name)


class TheWarmModelCache(unittest.TestCase):
    def setUp(self):
        import haversack.network as network
        self.built = []
        built = self.built

        class _Model:
            def __init__(self, folder, **kw):
                built.append(kw)

            def to_device(self):
                return self
        self._orig, network.TorchModel = network.TorchModel, _Model
        self.addCleanup(setattr, network, "TorchModel", self._orig)

    def test_the_step_reaches_the_model_and_separates_warm_entries(self):
        from haversack.cache import ModelCache
        cache = ModelCache(capacity=4)
        a = cache.get("/m", step_size=0.8)
        b = cache.get("/m")
        c = cache.get("/m", step_size=0.8)
        self.assertEqual([k["step_size"] for k in self.built], [0.8, 0.5])
        self.assertIsNot(a, b)
        self.assertIs(a, c)


class EveryDoorCarriesTheStep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = _install(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _segmenter(self):
        from haversack.segmenter import Segmenter

        class _Catalog:
            def get(self, name):
                return name
        return Segmenter(weights=WeightsStore(self.root, fetch=False), catalog=_Catalog())

    def _pipeline_kwargs(self, spec) -> dict:
        import numpy as np
        import SimpleITK as sitk
        from haversack.pipeline import segment

        class _Stop(Exception):
            pass
        seen = []

        class _Models:
            def get(self, folder, **kw):
                seen.append(kw)
                raise _Stop
        img = self.root / "ct.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 8, 8), np.int16)), str(img))
        with self.assertRaises(_Stop):
            segment(img, spec, weights=WeightsStore(self.root, fetch=False), models=_Models(),
                    device="cpu", dtype="fp32")
        return seen[0]

    def test_the_pipeline_builds_the_model_at_the_stated_step(self):
        self.assertEqual(self._pipeline_kwargs(_spec())["step_size"], 0.8)

    def test_a_task_stating_no_step_is_called_as_before(self):
        self.assertNotIn("step_size", self._pipeline_kwargs(_spec(None)))

    def test_warm_builds_at_the_stated_step(self):
        seg = self._segmenter()
        seen = []

        class _Models:
            def get(self, folder, **kw):
                seen.append(kw)

            def __len__(self):
                return len(seen)
        seg.models = _Models()
        seg.policy["weights"] = WeightsStore(self.root, fetch=False)
        seg.warm(_spec())
        seg.warm(_spec(None))
        self.assertEqual([k.get("step_size") for k in seen], [0.8, None])
        self.assertNotIn("step_size", seen[1])

    def test_the_result_key_carries_a_stated_step_only(self):
        from haversack.serve import weights_versions_of
        seg = self._segmenter()
        self.assertEqual(seg.describe(_spec())["step_size"], 0.8)
        self.assertNotIn("step_size", seg.describe(_spec(None)))
        self.assertEqual(weights_versions_of(seg, _spec()), ["836=v3.0.0-weights", "step=0.8"])
        self.assertEqual(weights_versions_of(seg, _spec(None)), ["836=v3.0.0-weights"])
        self.assertNotEqual(weights_versions_of(seg, _spec(0.5)),
                            weights_versions_of(seg, _spec(None)))   # stated, so keyed


if __name__ == "__main__":
    unittest.main()
