"""A TotalSegmentator result carries only the classes its task names (2026-09-22).

Some TotalSegmentator models are trained with classes their task does not report - "auxiliary"
classes, there to help the model learn. Upstream zeroes them after prediction
(postprocessing.remove_auxiliary_labels, called from nnunet.py on the model's labels, before any
resample back): for task X, every value in ``class_map["X_auxiliary"]`` becomes 0. haversack wrote
them: compared with upstream 2.18.0 on Modal, ts.v2:kidney_cysts wrote values 3 and 4 (~115k
voxels each, Dataset 789's kidney_left / kidney_right) beside upstream's 248 voxels of cyst,
under a label map that names only 1 and 2.

The rule: a TS-lineage result maps every model value its label map does not name to 0, after
the argmax, as upstream does. The registry states each task's auxiliary classes (upstream's
list); a model emitting a value that is neither named nor stated auxiliary is refused as a
catalog that does not match its weights, so no result changes without its key saying so.
"""
import json
import unittest

import numpy as np
import pytest

from haversack.tasks import TaskCatalog

# upstream's "<task>_auxiliary" maps, TotalSegmentator 2.13.0 map_to_binary.py (ts.v2's release),
# identical in 2.18.0; read 2026-09-22. renal_arteries_auxiliary ({4: aorta}) has no ts.v2 task.
UPSTREAM_AUXILIARY = {
    "appendicular_bones": {12: "humerus", 13: "femur", 14: "liver", 15: "spleen"},
    "face_mr": {2: "brain", 3: "liver"},
    "kidney_cysts": {3: "kidney_left", 4: "kidney_right"},
}


class TheRegistry(unittest.TestCase):
    def setUp(self):
        self.cat = TaskCatalog("ts")

    def test_each_task_states_upstreams_auxiliary_classes(self):
        for name, aux in UPSTREAM_AUXILIARY.items():
            self.assertEqual(dict(self.cat.get(name).auxiliary), aux, name)

    def test_no_other_task_states_any(self):
        stated = {n for n in self.cat.names() if self.cat.get(n).auxiliary}
        self.assertEqual(stated, set(UPSTREAM_AUXILIARY))

    def test_an_auxiliary_class_is_never_a_named_one(self):
        for name in UPSTREAM_AUXILIARY:
            spec = self.cat.get(name)
            self.assertFalse(set(spec.auxiliary) & set(spec.label_map), name)

    def test_ts_v3_states_none(self):
        from haversack.ecosystems import TS_V3_TASKS
        cat = TaskCatalog("ts", path=TS_V3_TASKS)
        self.assertFalse([n for n in cat.names() if cat.get(n).auxiliary])

    def test_a_registry_naming_an_auxiliary_class_does_not_load(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reg.json"
            p.write_text(json.dumps({"tasks": [{
                "name": "bad", "single": 1, "label_map": {"1": "a", "2": "b"},
                "auxiliary": {"2": "b"}}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad: auxiliary"):
                TaskCatalog("ts", path=p)


# --- through segment() ------------------------------------------------------------------------

pytest.importorskip("nnunetv2")
from test_cascade_union import _Crop, _run                  # noqa: E402
from test_normalization_sharing import ORGANS, _StubModel    # noqa: E402
from haversack.tasks import CascadeStep, TaskSpec            # noqa: E402

KIDNEY = (slice(2, 10), slice(3, 12), slice(4, 12))           # where the model says "kidney"
CYST = (slice(5, 7), slice(6, 8), slice(7, 9))                # a cyst inside it


class _KidneyCysts(_StubModel):
    """Dataset 789's shape: 1-2 cysts, 3-4 kidneys (auxiliary); kidney wins where it lies."""

    def __init__(self, K=5):
        super().__init__(ORGANS._props, K=K)

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((self.K, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        logits[self.K - 1][KIDNEY] = 2.0
        logits[2][CYST] = 3.0
        return logits


def _single(aux=None, K=5):
    return TaskSpec(name="stub_kidney_cysts", shape="single", single=1,
                    label_map={1: "kidney_cyst_left", 2: "kidney_cyst_right"},
                    auxiliary=aux if aux is not None else {3: "kidney_left", 4: "kidney_right"})


def test_auxiliary_classes_are_zeroed(tmp_path, monkeypatch):
    res, _ = _run(tmp_path, monkeypatch, [_KidneyCysts()], _single())
    lab = res.array
    assert set(np.unique(lab)) == {0, 2}
    assert int((lab == 2).sum()) > 0


def test_where_the_auxiliary_class_won_is_background_not_the_runner_up(tmp_path, monkeypatch):
    """Upstream zeroes the argmax's labels; it does not drop the channel before the argmax,
    which would hand the kidney's voxels to whichever named class came second."""
    class _RunnerUp(_KidneyCysts):
        def predict_logits(self, crop, report=None):
            logits = super().predict_logits(crop, report)
            logits[1][KIDNEY] = 1.5                             # second to the kidney everywhere
            return logits
    res, _ = _run(tmp_path, monkeypatch, [_RunnerUp()], _single())
    assert set(np.unique(res.array)) == {0, 2}


def test_a_cascade_final_stage_zeroes_them_too(tmp_path, monkeypatch):
    """kidney_cysts is a cascade (body 300, then 789): the final stage is where they appear."""
    spec = TaskSpec(name="stub_kidney_cysts", shape="cascade",
                    label_map={1: "kidney_cyst_left", 2: "kidney_cyst_right"},
                    auxiliary={3: "kidney_left", 4: "kidney_right"},
                    cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,), dilation_mm=30.0),
                             CascadeStep(weights_id=2)))
    res, _ = _run(tmp_path, monkeypatch, [_Crop(), _KidneyCysts()], spec)
    assert set(np.unique(res.array)) == {0, 2}


def test_a_value_neither_named_nor_auxiliary_is_refused(tmp_path, monkeypatch):
    from haversack.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match=r"\[3, 4\]"):
        _run(tmp_path, monkeypatch, [_KidneyCysts()], _single(aux={}))


def test_an_auxiliary_class_the_model_lacks_is_refused(tmp_path, monkeypatch):
    """The registry's list must be the model's - a stale one is a catalog mismatch."""
    from haversack.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="auxiliary"):
        _run(tmp_path, monkeypatch, [_KidneyCysts(K=4)], _single())


def test_a_model_with_only_named_classes_is_untouched(tmp_path, monkeypatch):
    spec = TaskSpec(name="stub", shape="single", single=1,
                    label_map={1: "a", 2: "b", 3: "c", 4: "d"})
    res, _ = _run(tmp_path, monkeypatch, [_KidneyCysts()], spec)
    assert set(np.unique(res.array)) == {0, 2, 4}


class TestTheKey:
    """Only the three tasks stating auxiliary classes are re-keyed: no other task's labels move."""

    def _seg(self, tmp_path):
        from haversack.segmenter import Segmenter
        from haversack.weights import WeightsStore
        (tmp_path / "w").mkdir()
        return Segmenter(device="cpu", weights=WeightsStore(tmp_path / "w", fetch=False))

    def test_a_task_with_auxiliary_classes_names_the_rule(self, tmp_path):
        from haversack.serve import weights_versions_of
        seg = self._seg(tmp_path)
        for task in ("ts.v2:kidney_cysts", "ts.v2:appendicular_bones", "ts.v2:face_mr"):
            d = seg.describe(task)
            assert d["auxiliary"] == {str(k): v for k, v in UPSTREAM_AUXILIARY[task[6:]].items()}
            assert "auxiliary=0" in weights_versions_of(seg, task), task

    def test_no_other_task_does(self, tmp_path):
        from haversack.serve import weights_versions_of
        seg = self._seg(tmp_path)
        for task in ("ts.v2:total_fast", "ts.v2:total", "ts.v3:total", "ts.v2:body",
                     "ts.v2:face", "ts.v2:appendicular_bones_mr", "ts.v2:head_muscles"):
            assert "auxiliary" not in seg.describe(task)
            assert not any(v.startswith("auxiliary=") for v in weights_versions_of(seg, task))


def test_the_rule_is_the_ts_lineages(tmp_path, monkeypatch):
    """A stock nnU-Net task's label map is its dataset.json's, so it names every value the model
    emits; the rule and its refusal are TotalSegmentator's, and leave other lineages alone."""
    spec = TaskSpec(name="stub", lineage="nnunetv2", shape="single", single=1,
                    label_map={1: "kidney_cyst_left", 2: "kidney_cyst_right"})
    res, _ = _run(tmp_path, monkeypatch, [_KidneyCysts()], spec)
    assert 4 in set(np.unique(res.array))
