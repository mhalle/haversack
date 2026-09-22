"""A cascade whose last stage is a label union (2026-09-22): TotalSegmentator's
``headneck_muscles``.

Upstream crops like ``headneck_bones_vessels`` (the 6 mm ``total`` model, Dataset 298, boxing
both clavicles and C1/C5/T1/T4, +40 mm) and then runs Datasets 778 and 779 on that one crop,
combining them as it combines ``total``'s parts: each part in turn writes its classes, a later
part over an earlier one (nnunet.py, ``seg_combined[img_part][seg == jdx] = ...``). haversack
had cascades ending in one model and unions over the whole volume, never both, so the task was
left out of the registry. These tests hold the registry entry to upstream's facts, the loader to
what a stage may state, and ``segment()`` to running every part on the box the crop stage found.
"""
import json
import unittest

import numpy as np
import pytest

from haversack.tasks import CascadeStep, TaskCatalog, UnionPart, _check_cascade

# upstream's class_map["headneck_muscles"] and class_map_parts_headneck_muscles, TotalSegmentator
# 2.13.0 map_to_binary.py, read 2026-09-22 (the part maps are the global map split at 11)
UPSTREAM = ["sternocleidomastoid_right", "sternocleidomastoid_left",
            "superior_pharyngeal_constrictor", "middle_pharyngeal_constrictor",
            "inferior_pharyngeal_constrictor", "trapezius_right", "trapezius_left",
            "platysma_right", "platysma_left", "levator_scapulae_right", "levator_scapulae_left",
            "anterior_scalene_right", "anterior_scalene_left", "middle_scalene_right",
            "middle_scalene_left", "posterior_scalene_right", "posterior_scalene_left",
            "sterno_thyroid_right", "sterno_thyroid_left", "thyrohyoid_right", "thyrohyoid_left",
            "prevertebral_right", "prevertebral_left"]


class TheRegistryEntry(unittest.TestCase):
    def setUp(self):
        self.cat = TaskCatalog("ts")
        self.spec = self.cat.get("headneck_muscles")

    def test_labels_are_upstreams(self):
        self.assertEqual([self.spec.label_map[k] for k in sorted(self.spec.label_map)], UPSTREAM)
        self.assertEqual(sorted(self.spec.label_map), list(range(1, 24)))

    def test_it_crops_as_headneck_bones_vessels_does(self):
        crop, final = self.spec.cascade
        self.assertEqual(crop, self.cat.get("headneck_bones_vessels").cascade[0])
        self.assertEqual(crop.weights_id, 298)
        total = self.cat.get("total_fast").label_map         # 298 is total's 6 mm model
        self.assertEqual({total[c] for c in crop.crop_to_classes},
                         {"clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5",
                          "vertebrae_T1", "vertebrae_T4"})
        self.assertEqual(crop.dilation_mm, 40.0)
        self.assertIsNone(final.weights_id)

    def test_the_union_is_778_then_779_split_at_eleven(self):
        a, b = self.spec.cascade[-1].union
        self.assertEqual((a.weights_id, b.weights_id), (778, 779))
        self.assertEqual(dict(a.label_remap), {i: i for i in range(1, 12)})
        self.assertEqual(dict(b.label_remap), {i: i + 11 for i in range(1, 13)})

    def test_every_model_is_provisioned(self):
        self.assertEqual(self.spec.weights_ids, [298, 778, 779])

    def test_the_manifest_has_both_parts(self):
        from importlib import resources
        manifest = json.loads(resources.files("haversack").joinpath("data/ts_weights.json")
                              .read_text(encoding="utf-8"))
        text = json.dumps(manifest)
        for name in ("Dataset778_headneck_muscles_part1_492subj.zip",
                     "Dataset779_headneck_muscles_part2_492subj.zip"):
            self.assertIn(name, text)

    def test_the_meta_counts_the_tasks(self):
        from importlib import resources
        raw = json.loads(resources.files("haversack").joinpath("data/ts_tasks.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(raw["_meta"]["task_count"], len(raw["tasks"]))

    def test_it_is_not_a_licensed_task(self):
        from importlib import resources
        att = json.loads(resources.files("haversack").joinpath("data/attribution.json")
                         .read_text(encoding="utf-8"))
        self.assertNotIn("headneck_muscles", att["ecosystems"]["ts.v2"]["licensed_tasks"])


class WhatAStageMayState(unittest.TestCase):
    U = (UnionPart(weights_id=1, label_remap={1: 1}), UnionPart(weights_id=2, label_remap={1: 2}))

    def test_a_final_union_is_accepted(self):
        _check_cascade((CascadeStep(weights_id=9, crop_to_classes=(1,)), CascadeStep(union=self.U)), "t")

    def test_a_union_before_the_last_stage_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not the last"):
            _check_cascade((CascadeStep(union=self.U, crop_to_classes=(1,)),
                            CascadeStep(weights_id=9)), "t")

    def test_a_stage_stating_two_things_is_refused(self):
        with self.assertRaisesRegex(ValueError, "stage 2 states"):
            _check_cascade((CascadeStep(weights_id=9), CascadeStep(weights_id=8, union=self.U)), "t")

    def test_a_stage_stating_nothing_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing"):
            _check_cascade((CascadeStep(weights_id=9), CascadeStep()), "t")

    def test_every_shipped_cascade_loads(self):
        cat = TaskCatalog("ts")
        cascades = [n for n in cat.names() if cat.get(n).shape == "cascade"]
        self.assertGreater(len(cascades), 10)
        for n in cascades:
            _check_cascade(cat.get(n).cascade, n)

    def test_a_registry_with_a_misplaced_union_does_not_load(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reg.json"
            p.write_text(json.dumps({"tasks": [{
                "name": "bad", "shape": "cascade", "label_map": {"1": "a"},
                "cascade": [{"union": [{"weights_id": 1, "label_remap": {"1": 1}}],
                             "crop_to_classes": [1]}, {"weights_id": 2}]}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad: cascade stage 1"):
                TaskCatalog("ts", path=p)


# --- through segment() ------------------------------------------------------------------------

pytest.importorskip("nnunetv2")
from test_normalization_sharing import ORGANS, RIBS, _StubModel, _write_ct   # noqa: E402

BOX = (slice(4, 8), slice(5, 9), slice(6, 10))     # where the crop stage finds its class, model grid


class _Crop(_StubModel):
    """Stage 1: class 1 in BOX, background elsewhere - the crop stage's answer."""

    def __init__(self, found=True):
        super().__init__(ORGANS._props, K=2)
        self.found = found

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((2, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        if self.found:
            logits[1][BOX] = 2.0
        return logits


class _Part(_StubModel):
    """A union part: class 1 over ``cover`` of whatever crop it is handed."""

    def __init__(self, props, cover):
        super().__init__(props, K=2)
        self.cover = cover

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((2, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        logits[1][self.cover] = 2.0
        return logits


def _run(tmp_path, monkeypatch, models, spec):
    from haversack import pipeline
    folder = tmp_path / "Dataset000_stub" / "trainer__plans__3d_fullres"
    (folder / "fold_0").mkdir(parents=True, exist_ok=True)

    class _Store:
        root = folder.parent.parent

        def resolve(self, weights_id, *, configuration=None, **kw):
            return folder

        def describe(self, *a, **k):
            return {}

    class _Cache:
        def __init__(self):
            self.order = list(models)

        def get(self, folder, **kw):
            return self.order.pop(0)

        def release(self, model):
            pass

    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    store = _Store()
    shape = (16, 20, 18)
    res = pipeline.segment(str(_write_ct(tmp_path, shape)), spec, models=_Cache(), device="cpu",
                           envelope_mm=None, convention="corner", folds=(0,), interp="nearest")
    return res, shape


def _spec():
    from haversack.tasks import TaskSpec
    return TaskSpec(name="stub_crop_union", shape="cascade", label_map={1: "a", 2: "b"},
                    cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,), dilation_mm=0.0),
                             CascadeStep(union=(UnionPart(weights_id=2, label_remap={1: 1}, name="first"),
                                                UnionPart(weights_id=3, label_remap={1: 2}, name="second")))))


def test_every_part_runs_on_the_box_the_crop_stage_found(tmp_path, monkeypatch):
    crop = _Crop()
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(0, 2), slice(None), slice(None)))
    _run(tmp_path, monkeypatch, [crop, a, b], _spec())
    full = tuple(crop.received.shape[1:])
    got_a, got_b = tuple(a.received.shape[1:]), tuple(b.received.shape[1:])
    assert got_a == got_b, "the two parts ran on different crops"
    assert got_a != full and all(x <= y for x, y in zip(got_a, full)), (got_a, full)
    # its own normalization each, as every union's parts get - the tripwire would have raised
    assert not np.allclose(a.received.numpy(), b.received.numpy())


def test_the_later_part_paints_over_the_earlier_and_nothing_lands_outside_the_box(tmp_path, monkeypatch):
    crop = _Crop()
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(0, 2), slice(None), slice(None)))
    res, _ = _run(tmp_path, monkeypatch, [crop, a, b], _spec())
    lab = res.array
    cz, cy, cx = a.received.shape[1:]
    assert set(np.unique(lab)) == {0, 1, 2}
    assert int((lab == 2).sum()) == 2 * cy * cx
    assert int((lab == 1).sum()) == cz * cy * cx - 2 * cy * cx
    assert int((lab > 0).sum()) < lab.size


def test_nothing_found_runs_every_part_on_the_whole_volume(tmp_path, monkeypatch):
    crop = _Crop(found=False)
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(0, 2), slice(None), slice(None)))
    _run(tmp_path, monkeypatch, [crop, a, b], _spec())
    full = tuple(crop.received.shape[1:])
    assert tuple(a.received.shape[1:]) == tuple(b.received.shape[1:]) == full


def test_provenance_names_all_three_models(tmp_path, monkeypatch):
    crop = _Crop()
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(0, 2), slice(None), slice(None)))
    res, _ = _run(tmp_path, monkeypatch, [crop, a, b], _spec())
    assert [m["weights"] for m in res.provenance["models"]] == ["1", "2", "3"]
