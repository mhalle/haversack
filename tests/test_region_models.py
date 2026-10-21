"""Region-based nnU-Net models run, decoded as nnU-Net's own export decodes them (2026-10-20).

A region-based model (a ``labels`` entry listing several values, e.g. PanTS's
``"pancreas": [17, 18, 19, 20, 21, 28]``) has one sigmoid head per foreground region, in
``labels`` order, and nnU-Net writes ``regions_class_order[k]`` wherever channel ``k`` is above
0.5, later channels winning (``LabelManager.convert_probabilities_to_segmentation``). haversack
refused them until labelfield's ``mode="regions"`` restore was wired in: the same rule, on the
logits (sigmoid > 0.5 is logit > 0), fused with the resample back to the input grid.
"""
import json

import numpy as np
import pytest

from haversack.errors import UnsupportedModel
from haversack.tasks import TaskSpec, dataset_labels, dataset_regions

# PanTS's dataset.json, abridged to the shape that matters: a region whose painted value (17) is
# no other label's own, sub-regions painted after it, and a lesion painted last of all
PANTS_LIKE = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "liver": 1, "pancreas": [2, 3, 4, 6],
               "pancreas_head": 3, "pancreas_tail": 4, "spleen": 5, "pancreatic_lesion": 6},
    "regions_class_order": [1, 2, 3, 4, 5, 6],
}


class TestLabels:
    def test_each_region_is_named_by_the_value_it_paints(self):
        assert dataset_labels(PANTS_LIKE) == {1: "liver", 2: "pancreas", 3: "pancreas_head",
                                              4: "pancreas_tail", 5: "spleen",
                                              6: "pancreatic_lesion"}
        assert dataset_regions(PANTS_LIKE) == (1, 2, 3, 4, 5, 6)

    def test_the_order_not_the_label_value_names_a_region(self):
        """nnU-Net paints channel k as regions_class_order[k], whatever the labels entry says."""
        ds = {"labels": {"background": 0, "whole": [1, 2, 3], "core": [2, 3], "enh": 3},
              "regions_class_order": [1, 2, 3]}
        assert dataset_labels(ds) == {1: "whole", 2: "core", 3: "enh"}
        ds["regions_class_order"] = [3, 2, 1]
        assert dataset_labels(ds) == {3: "whole", 2: "core", 1: "enh"}

    def test_ignore_is_no_region(self):
        ds = {"labels": {"background": 0, "whole": [1, 2], "core": 2, "ignore": 3},
              "regions_class_order": [1, 2]}
        assert dataset_labels(ds) == {1: "whole", 2: "core"}

    def test_a_softmax_model_has_no_regions(self):
        assert dataset_regions({"labels": {"background": 0, "a": 1}}) == ()

    @pytest.mark.parametrize("order", [None, [1, 2], [1, 1, 2], [0, 1, 2]])
    def test_regions_without_a_decode_are_refused(self, order):
        ds = {"labels": {"background": 0, "whole": [1, 2, 3], "core": [2, 3], "enh": 3}}
        if order is not None:
            ds["regions_class_order"] = order
        with pytest.raises(UnsupportedModel, match="regions_class_order"):
            dataset_labels(ds)

    def test_a_model_folder_carries_its_decode(self, tmp_path):
        d = tmp_path / "Dataset003_x" / "T__P__3d_fullres"
        (d / "fold_all").mkdir(parents=True)
        (d / "dataset.json").write_text(json.dumps(PANTS_LIKE))
        (d / "plans.json").write_text(json.dumps({"configurations": {"3d_fullres": {}}}))
        spec = TaskSpec.from_model_folder(d)
        assert spec.regions == (1, 2, 3, 4, 5, 6)
        assert spec.label_map[2] == "pancreas"


class TestListing:
    def test_a_decodable_region_model_lists_the_label_map_it_writes(self):
        from haversack.ecosystems import _dataset_listing
        got = _dataset_listing(PANTS_LIKE, where="x")
        assert got["segments"][:2] == [{"id": "liver", "value": 1}, {"id": "pancreas", "value": 2}]
        assert not any("layer" in s for s in got["segments"])


# --- through segment() ------------------------------------------------------------------------

pytest.importorskip("nnunetv2")
from test_cascade_union import _run                          # noqa: E402
from test_normalization_sharing import ORGANS, _StubModel    # noqa: E402

PANCREAS = (slice(2, 12), slice(3, 14), slice(4, 14))
HEAD = (slice(2, 7), slice(3, 14), slice(4, 14))              # inside the pancreas region
LESION = (slice(5, 9), slice(6, 10), slice(7, 11))            # across head and the rest


class _Regions(_StubModel):
    """Six sigmoid heads, no background channel: logit > 0 is "in the region"."""

    def __init__(self, K=6):
        super().__init__(ORGANS._props, K=K)

    def predict_logits(self, crop, report=None):
        import torch
        logits = torch.full((self.K, *crop.shape[1:]), -4.0, dtype=torch.float32)
        logits[1][PANCREAS] = 3.0
        logits[2][HEAD] = 3.0
        logits[5][LESION] = 3.0
        return logits


def _spec():
    return TaskSpec(name="stub_regions", lineage="nnunetv2", shape="single", single=1,
                    label_map=dataset_labels(PANTS_LIKE), regions=dataset_regions(PANTS_LIKE))


def _expected(shape):
    want = np.zeros(shape, dtype=np.uint8)
    want[PANCREAS] = 2
    want[HEAD] = 3
    want[LESION] = 6
    return want


def test_regions_paint_in_order_later_winning(tmp_path, monkeypatch):
    res, shape = _run(tmp_path, monkeypatch, [_Regions()], _spec())
    assert np.array_equal(np.asarray(res.array), _expected(shape))


def test_the_decode_is_nnunets_own(tmp_path, monkeypatch):
    """The same logits through nnU-Net's LabelManager give the same labels."""
    import torch
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    res, shape = _run(tmp_path, monkeypatch, [_Regions()], _spec())
    lm = LabelManager(PANTS_LIKE["labels"], PANTS_LIKE["regions_class_order"])
    ref = lm.convert_logits_to_segmentation(_Regions().predict_logits(torch.zeros((1, *shape))))
    assert np.array_equal(np.asarray(res.array), np.asarray(ref))


def test_the_decode_is_in_the_provenance(tmp_path, monkeypatch):
    res, _ = _run(tmp_path, monkeypatch, [_Regions()], _spec())
    assert res.provenance["decode"] == {"mode": "regions", "threshold": 0.0,
                                        "order": [1, 2, 3, 4, 5, 6]}


def test_a_head_count_the_labels_do_not_declare_is_refused(tmp_path, monkeypatch):
    from haversack.errors import ModelNotFound
    with pytest.raises(ModelNotFound, match="6 regions"):
        _run(tmp_path, monkeypatch, [_Regions(K=7)], _spec())


def test_a_ranked_store_is_refused(tmp_path, monkeypatch):
    from haversack.ranked import RankedSpec
    with pytest.raises(UnsupportedModel, match="softmax"):
        _run(tmp_path, monkeypatch, [_Regions()], _spec(), probabilities=RankedSpec(sink=lambda *a: None))
