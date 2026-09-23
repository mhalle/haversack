"""A TotalSegmentator cascade crops as upstream does (2026-09-22).

Upstream (cropping.py, python_api.py, nnunet.py; 2.13.0 and 2.18.0 alike) builds a mask of the
crop classes from the crop model's labels on the ORIGINAL image, takes its bounding box widened
by ``int(mm / zoom)`` voxels per axis (upper end one past the last voxel, clipped), cuts the image
to that box, runs the final model on the cut alone and pastes the labels back into zeros; an
empty mask returns an empty segmentation. haversack's crop was, until this date, a speed
approximation of whole-volume inference - grown to the patch, collapsed when it saved little -
and labeled beyond upstream's box: headneck_bones_vessels scored mean Dice 0.738 against
upstream on a neck CT, the zygomatic arches found by haversack only.
"""
import numpy as np
import pytest

from haversack.pipeline import upstream_crop_box


def _upstream_bbox(mask, margin_mm, zooms):
    """cropping.get_bbox_from_mask(mask, outside_value=0, addon=int(mm / zoom)), restated from
    TotalSegmentator 2.18.0 - the reference the box is checked against."""
    addon = (np.array([margin_mm] * 3) / np.asarray(zooms)).astype(int)
    if (mask > 0).sum() == 0:
        return None
    c = np.where(mask > 0)
    out = []
    for ax in range(3):
        lo = int(np.min(c[ax])) - addon[ax]
        hi = int(np.max(c[ax])) + 1 + addon[ax]
        out.append((max(0, lo), min(mask.shape[ax], hi)))
    return tuple(o[0] for o in out), tuple(o[1] for o in out)


@pytest.mark.parametrize("seed", range(12))
def test_the_box_is_upstreams(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in rng.integers(8, 40, 3))
    labels = np.zeros(shape, dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        lo = [int(rng.integers(0, s)) for s in shape]
        hi = [min(s, l + int(rng.integers(1, 6))) for l, s in zip(lo, shape)]
        labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = int(rng.integers(1, 4))
    classes = [1, 3]
    spacing = tuple(float(v) for v in rng.choice([0.49, 0.75, 1.0, 1.5, 3.0, 5.0], 3))
    margin = float(rng.choice([0.0, 5.0, 10.0, 20.0, 40.0]))
    want = _upstream_bbox(np.isin(labels, classes), margin, spacing)
    assert upstream_crop_box(labels, classes, margin, spacing) == want


def test_the_margin_truncates_as_upstream_does():
    labels = np.zeros((90, 90, 90), np.uint8)
    labels[40:42, 40:42, 40:42] = 1
    # 20 mm at 0.75 mm is 26.67 voxels: upstream's astype(int) makes it 26, not 27
    lo, hi = upstream_crop_box(labels, [1], 20.0, (0.75, 0.75, 0.75))
    assert lo == (40 - 26,) * 3 and hi == (42 + 26,) * 3


def test_an_absent_class_is_no_box():
    assert upstream_crop_box(np.zeros((5, 5, 5), np.uint8), [1], 20.0, (1.0, 1.0, 1.0)) is None


# --- through segment() ----------------------------------------------------------------------

pytest.importorskip("nnunetv2")
from test_cascade_union import BOX, _Crop, _Part, _run, _spec   # noqa: E402
from test_normalization_sharing import ORGANS, RIBS                # noqa: E402
from haversack.tasks import CascadeStep, TaskSpec, UnionPart       # noqa: E402

SPACING = 1.5                                   # the stub CT's and the stub models' spacing


def _cascade(margin_mm):
    return TaskSpec(name="stub_crop", shape="cascade", label_map={1: "a", 2: "b"},
                    cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,), dilation_mm=margin_mm),
                             CascadeStep(union=(UnionPart(weights_id=2, label_remap={1: 1}, name="first"),
                                                UnionPart(weights_id=3, label_remap={1: 2}, name="second")))))


def _box_shape(margin_mm, grid=(16, 20, 18)):
    add = int(margin_mm / SPACING)
    return tuple(min(n, s.stop + add) - max(0, s.start - add) for s, n in zip(BOX, grid))


def test_the_final_stage_sees_exactly_the_box(tmp_path, monkeypatch):
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(0, 1), slice(None), slice(None)))
    res, _ = _run(tmp_path, monkeypatch, [_Crop(), a, b], _cascade(3.0))
    want = _box_shape(3.0)                                  # 3 mm at 1.5 mm = 2 voxels a side
    assert tuple(a.received.shape[1:]) == tuple(b.received.shape[1:]) == want
    box = res.provenance["crops"][0]["box"]
    assert tuple(h - l for l, h in zip(*box)) == want


def test_nothing_is_labeled_outside_the_box(tmp_path, monkeypatch):
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)       # everything it sees is class 1
    b = _Part(RIBS._props, cover=(slice(0, 1), slice(None), slice(None)))
    res, _ = _run(tmp_path, monkeypatch, [_Crop(), a, b], _cascade(3.0))
    lab = res.array
    want = _box_shape(3.0)
    assert int((lab > 0).sum()) == int(np.prod(want))       # the box, all of it, and no more
    nz = np.nonzero(lab)
    assert tuple(int(i.max() - i.min() + 1) for i in nz) == want


def test_a_box_narrower_than_the_patch_is_not_grown(tmp_path, monkeypatch):
    """The old crop grew to the patch (4 voxels here) with real image; upstream's network pads
    the cut instead, and so does the final stage now."""
    a = _Part(ORGANS._props, cover=(slice(None),) * 3)
    b = _Part(RIBS._props, cover=(slice(None),) * 3)
    import test_cascade_union as tcu
    monkeypatch.setattr(tcu, "BOX", (slice(4, 6), slice(5, 7), slice(6, 8)))
    _run(tmp_path, monkeypatch, [_Crop(), a, b], _cascade(0.0))
    assert tuple(a.received.shape[1:]) == (2, 2, 2)


def test_a_crop_from_another_task_uses_that_tasks_labels_on_the_input(tmp_path, monkeypatch):
    from haversack import pipeline
    sub = TaskSpec(name="sub", shape="single", single=1, label_map={1: "c"})
    outer = TaskSpec(name="outer", shape="cascade", label_map={1: "a"},
                     cascade=(CascadeStep(crop_from_task="sub", crop_to_classes=(1,), dilation_mm=3.0),
                              CascadeStep(weights_id=2)))

    class _Catalog:
        def get(self, name, progress=None):
            return sub if name == "sub" else name
    real = pipeline.segment

    def with_catalog(*a, **k):
        return real(*a, catalog=_Catalog(), **k)
    monkeypatch.setattr(pipeline, "segment", with_catalog)
    part = _Part(ORGANS._props, cover=(slice(None),) * 3)
    res, _ = _run(tmp_path, monkeypatch, [_Crop(), part], outer)
    assert tuple(part.received.shape[1:]) == _box_shape(3.0)
    assert int((res.array > 0).sum()) == int(np.prod(_box_shape(3.0)))


class _CoarseCrop(_Crop):
    """A crop model at twice the image's spacing, as 298 (6 mm) is to most CTs. Its class sits
    in model voxels ``box``; restored nearest-neighbor - upstream's order-0 resample of its
    labels - each covers exactly two image voxels a side (corner convention, factor 2). The
    class wins by a narrow margin (1.1 over 1.0), so a linear restore of the field pulls its
    edges in - (2, 5), (5, 7), (5, 7) where nearest gives (2, 6), (4, 8), (4, 8), measured - and
    a crop stage restored that way would cut a different box."""

    def __init__(self, box):
        super().__init__()
        self.spacing_zyx = (2 * SPACING,) * 3
        self.box = box

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((2, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        logits[1][self.box] = 1.1
        return logits


def test_the_crop_stage_is_restored_nearest_as_upstream_restores_it(tmp_path, monkeypatch):
    part = _Part(ORGANS._props, cover=(slice(None),) * 3)
    spec = TaskSpec(name="coarse", shape="cascade", label_map={1: "a"},
                    cascade=(CascadeStep(weights_id=1, crop_to_classes=(1,), dilation_mm=0.0),
                             CascadeStep(weights_id=2)))
    box = (slice(1, 3), slice(2, 4), slice(2, 4))
    _run(tmp_path, monkeypatch, [_CoarseCrop(box), part], spec, interp="linear")
    assert tuple(part.received.shape[1:]) == (4, 4, 4)       # 2 model voxels -> 4 image voxels


def test_a_crop_source_task_is_read_on_the_input_grid_whatever_grid_was_asked(tmp_path, monkeypatch):
    """The box is in input voxels, so the crop task's labels must be too - not on the grid the
    caller asked the RESULT on."""
    from haversack import pipeline
    sub = TaskSpec(name="sub", shape="single", single=1, label_map={1: "c"})
    outer = TaskSpec(name="outer", shape="cascade", label_map={1: "a"},
                     cascade=(CascadeStep(crop_from_task="sub", crop_to_classes=(1,), dilation_mm=3.0),
                              CascadeStep(weights_id=2)))

    class _Catalog:
        def get(self, name, progress=None):
            return sub if name == "sub" else name
    real = pipeline.segment
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: real(*a, catalog=_Catalog(), **k))
    part = _Part(ORGANS._props, cover=(slice(None),) * 3)
    _run(tmp_path, monkeypatch, [_Crop(), part], outer, grid=3.0)
    assert tuple(part.received.shape[1:]) == _box_shape(3.0)


class TestTheKey:
    """Only cascades' results are re-keyed: their labels moved on 2026-09-22, no other task's."""

    def _seg(self, tmp_path):
        from haversack.segmenter import Segmenter
        from haversack.weights import WeightsStore
        (tmp_path / "w").mkdir()
        return Segmenter(device="cpu", weights=WeightsStore(tmp_path / "w", fetch=False))

    def test_a_cascade_names_the_rule(self, tmp_path):
        from haversack.serve import weights_versions_of
        seg = self._seg(tmp_path)
        assert seg.describe("ts.v2:head_muscles")["crop"] == "upstream"
        assert "crop=upstream" in weights_versions_of(seg, "ts.v2:head_muscles")

    def test_no_other_task_does(self, tmp_path):
        from haversack.serve import weights_versions_of
        seg = self._seg(tmp_path)
        for task in ("ts.v2:total_fast", "ts.v2:total", "ts.v3:total", "ts.v2:body"):
            assert "crop" not in seg.describe(task)
            assert not any(v.startswith("crop=") for v in weights_versions_of(seg, task))
