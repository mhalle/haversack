"""TotalSegmentator's postprocessing, reproduced (2026-09-28): body's largest trunk and small
extremities, heartchambers_highres' remove-outside, --remove_small_blobs.

The references below are upstream's functions (totalsegmentator/postprocessing.py, v2.13-v3),
transcribed because TotalSegmentator is not a haversack dependency."""
import builtins

import numpy as np
import pytest
from scipy import ndimage

from haversack import postprocess as pp


# --- upstream, transcribed ---------------------------------------------------------------------
def ref_keep_largest_blob(data):
    blob_map, nr_of_blobs = ndimage.label(data)
    counts = [np.sum(blob_map == i) for i in range(1, nr_of_blobs + 1)]
    if len(counts) == 0:
        return data
    return (blob_map == np.argmax(counts) + 1).astype(np.uint8)


def ref_keep_largest_blob_multilabel(data, rois):
    for idx in rois:
        data_roi = data == idx
        cleaned_roi = ref_keep_largest_blob(data_roi) > 0.5
        data[data_roi] = 0
        data[cleaned_roi] = idx
    return data


def ref_remove_small_blobs(img, interval):
    mask, number_of_blobs = ndimage.label(img)
    counts = np.bincount(mask.flatten())
    if len(counts) <= 1:
        return img
    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1
    return mask


def ref_remove_small_blobs_multilabel(data, rois, interval):
    for idx in rois:
        data_roi = data == idx
        cleaned_roi = ref_remove_small_blobs(data_roi, interval) > 0.5
        data[data_roi] = 0
        data[cleaned_roi] = idx
    return data


def ref_remove_outside_of_mask(seg, mask, addon):
    mask = ndimage.binary_dilation(mask, iterations=addon)
    seg[mask == 0] = 0
    return seg


# --- fixtures ----------------------------------------------------------------------------------
def speckled(shape, n_labels, seed, density=0.35):
    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.normal(size=shape), 1.2)
    lab = rng.integers(1, n_labels + 1, size=shape).astype(np.uint8)
    lab = ndimage.median_filter(lab, 3)
    return np.where(noise > np.quantile(noise, 1 - density), lab, 0).astype(np.uint8)


@pytest.fixture(params=["cc3d", "scipy"])
def backend(request, monkeypatch):
    if request.param == "scipy":
        real = builtins.__import__

        def no_cc3d(name, *a, **k):
            if name == "cc3d":
                raise ImportError("cc3d hidden for this test")
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", no_cc3d)
    else:
        pytest.importorskip("cc3d")
    return request.param


@pytest.mark.parametrize("seed", range(4))
def test_keep_largest_is_upstreams(seed):
    data = speckled((30, 32, 34), 4, seed)
    want = ref_keep_largest_blob_multilabel(data.copy(), [1, 3])
    got = pp.keep_largest(data.copy(), [1, 3])
    np.testing.assert_array_equal(got, want)
    assert (want != data).any()


def test_keep_largest_ties_pick_the_first_piece():
    data = np.zeros((5, 5, 12), np.uint8)
    data[1:3, 1:3, 1:3] = 1
    data[1:3, 1:3, 8:10] = 1                      # the same size, later in a C-order scan
    np.testing.assert_array_equal(pp.keep_largest(data.copy(), [1]),
                                  ref_keep_largest_blob_multilabel(data.copy(), [1]))


@pytest.mark.parametrize("seed", range(4))
def test_remove_small_is_upstreams(seed, backend):
    data = speckled((30, 32, 34), 5, seed)
    for rois, thr in (([1, 2, 3, 4, 5], 20.0), ([2], 7.5), ([1, 4], 0.0)):
        want = ref_remove_small_blobs_multilabel(data.copy(), rois, [thr, 1e10])
        got = pp.remove_small(data.copy(), rois, thr)
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(pp.remove_small(data.copy(), "all", 20.0),
                                  ref_remove_small_blobs_multilabel(data.copy(), [1, 2, 3, 4, 5], [20.0, 1e10]))


def test_remove_outside_is_upstreams():
    data = speckled((20, 22, 24), 3, 9, density=0.8)
    mask = np.zeros(data.shape, bool)
    mask[8:12, 9:13, 10:14] = True
    for it in (0, 1, 3):
        want = ref_remove_outside_of_mask(data.copy(), mask, it) if it else np.where(mask, data, 0)
        np.testing.assert_array_equal(pp.remove_outside(data.copy(), mask, it), want)
    assert pp.dilation_iterations(10, (1.5, 1.5, 1.5)) == 6       # int(10 / 1.5)
    assert pp.dilation_iterations(10, (3.0, 0.8, 0.8)) == 6       # int(10 / mean 1.533)


def test_apply_converts_mm3_with_the_grids_voxel():
    data = np.zeros((10, 10, 10), np.uint8)
    data[1, 1, 1:5] = 2                            # 4 voxels = 32 mm3 at 2 mm
    data[5, 5, 1:6] = 2                            # 5 voxels = 40 mm3
    ops = pp.check_ops([{"op": "remove_small", "classes": [2], "max_mm3": 32}])
    out = data.copy()
    pp.apply(out, ops, (2.0, 2.0, 2.0))
    assert int((out == 2).sum()) == 5


def test_malformed_steps_are_refused():
    for bad in ([{"op": "dilate", "classes": [1]}], [{"op": "keep_largest", "classes": "all"}],
                [{"op": "remove_small", "classes": [1]}], [{"op": "keep_largest", "classes": []}]):
        with pytest.raises(ValueError):
            pp.check_ops(bad)


# --- the registry says what upstream does -----------------------------------------------------
def test_the_registry_states_upstreams_rules():
    from haversack.tasks import TaskCatalog
    c = TaskCatalog()
    body = ({"op": "keep_largest", "classes": [1]},
            {"op": "remove_small", "classes": [2], "max_mm3": 50000})
    assert c.get("body").postprocess == body
    assert c.get("body_fast").postprocess == body
    assert c.get("body").label_map == {1: "body_trunc", 2: "body_extremities"}
    am = c.get("abdominal_muscles").cascade[0]
    assert am.weights_id == 300 and am.postprocess == body            # upstream crop_task = "body"
    hc = c.get("heartchambers_highres").cascade[0]
    fast = c.get("total_fast")
    assert hc.weights_id == fast.single
    assert [fast.label_map[k] for k in hc.remove_outside_classes] == ["heart", "aorta", "inferior_vena_cava"]
    assert hc.remove_outside_mm == 10.0
    # every other task is left as the network made it
    assert sorted(n for n in c.names() if c.get(n).postprocess) == ["body", "body_fast"]


def test_a_final_stage_may_not_postprocess():
    from haversack.tasks import CascadeStep, _check_cascade
    with pytest.raises(ValueError):
        _check_cascade((CascadeStep(weights_id=1, crop_to_classes=(1,)),
                        CascadeStep(weights_id=2, remove_outside_classes=(1,))), "t")


# --- through segment() -------------------------------------------------------------------------
SPACING = (1.5, 1.5, 1.5)
CT = {"mean": 0.0, "std": 100.0, "percentile_00_5": -1000.0, "percentile_99_5": 1000.0}


class _Stub:
    """A TS-lineage model whose logits pick ``pattern(shape)`` (a label volume on its grid)."""

    def __init__(self, K, pattern):
        self.spacing_zyx = SPACING
        self.normalization_schemes = ("CTNormalization",)
        self.use_mask_for_norm = (False,)
        self.K = K
        self.patch = (4, 4, 4)
        self.transpose_forward = (0, 1, 2)
        self.accumulate_choice = {"on_device": False}
        self.pattern = pattern

    def intensity_properties(self, channel):
        return dict(CT)

    def tiles(self, extent_zyx):
        from haversack.network import window_tiles
        return window_tiles(extent_zyx, self.patch, self.transpose_forward)

    def predict_logits(self, crop, report=None):
        import torch
        lab = torch.as_tensor(self.pattern(tuple(crop.shape[1:])), dtype=torch.long)
        return torch.nn.functional.one_hot(lab, self.K).permute(3, 0, 1, 2).float() * 10


def _run(tmp_path, monkeypatch, spec, models, shape=(30, 32, 34), **kw):
    import SimpleITK as sitk
    from haversack import pipeline
    folder = tmp_path / "Dataset000_stub" / "trainer__plans__3d_fullres"
    (folder / "fold_0").mkdir(parents=True, exist_ok=True)

    class _Store:
        root = folder.parent.parent

        def resolve(self, weights_id, *, configuration=None, **k):
            return folder

        def describe(self, *a, **k):
            return {}

    class _Cache:
        order = list(models)

        def get(self, folder, **k):
            return self.order.pop(0)

        def release(self, model):
            pass

    img = sitk.GetImageFromArray(np.zeros(shape, np.int16))
    img.SetSpacing(SPACING[::-1])
    img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))   # RAS: nothing to reorient
    p = tmp_path / "ct.nii.gz"
    sitk.WriteImage(img, str(p))
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    store = _Store()
    r = pipeline.segment(str(p), spec, models=_Cache(), device="cpu", envelope_mm=None, folds=(0,), **kw)
    return sitk.GetArrayFromImage(r.labels), r


def test_body_keeps_its_trunk_and_drops_small_extremities(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK")
    from haversack.tasks import TaskSpec

    def body(shape):
        lab = np.zeros(shape, np.int64)
        lab[5:25, 5:25, 5:25] = 1                  # the trunk
        lab[1:3, 1:3, 1:3] = 1                     # a stray trunk piece
        lab[27:29, 27:30, 27:30] = 2               # a small extremity piece (18 vox = 60.75 mm3)
        lab[5:25, 26:31, 5:10] = 2                 # a large one
        return lab

    spec = TaskSpec(name="body", single=299, label_map={1: "body_trunc", 2: "body_extremities"},
                    postprocess=pp.check_ops([{"op": "keep_largest", "classes": [1]},
                                              {"op": "remove_small", "classes": [2], "max_mm3": 100}]))
    got, r = _run(tmp_path, monkeypatch, spec, [_Stub(3, body)])
    want = body(got.shape).astype(np.uint8)
    want[1:3, 1:3, 1:3] = 0
    want[27:29, 27:30, 27:30] = 0
    np.testing.assert_array_equal(got, want)
    assert [s["op"] for s in r.provenance["postprocessing"]] == ["keep_largest", "remove_small"]

    # --remove_small_blobs on top: the 5x5x20 extremity is 500 vox = 1687.5 mm3
    got2, _ = _run(tmp_path, monkeypatch, spec, [_Stub(3, body)], remove_small_blobs=2000)
    assert not (got2 == 2).any() and (got2 == 1).sum() == 8000


def test_heartchambers_removes_what_lies_beyond_its_mask(tmp_path, monkeypatch):
    pytest.importorskip("SimpleITK")
    from haversack.tasks import CascadeStep, TaskSpec

    def crop_model(shape):                         # total_fast: a heart cube
        lab = np.zeros(shape, np.int64)
        lab[10:16, 13:19, 16:22] = 51              # off center: a flip would show
        return lab

    def target(shape):                             # heartchambers: labels everywhere it runs
        return np.ones(shape, np.int64)

    spec = TaskSpec(name="heartchambers_highres", shape="cascade", label_map={1: "heart_myocardium"},
                    cascade=(CascadeStep(weights_id=297, crop_to_classes=(51,), dilation_mm=20.0,
                                         remove_outside_classes=(51, 52, 63), remove_outside_mm=10.0),
                             CascadeStep(weights_id=301)))
    got, r = _run(tmp_path, monkeypatch, spec, [_Stub(64, crop_model), _Stub(2, target)])
    mask = np.zeros(got.shape, bool)
    mask[10:16, 13:19, 16:22] = True
    keep = ndimage.binary_dilation(mask, iterations=6)          # int(10 mm / 1.5 mm)
    assert got[keep].all() and not got[~keep].any()
    assert r.provenance["postprocessing"][-1]["op"] == "remove_outside"


def test_mask_on_grid_is_nearest():
    from haversack.grid import Grid
    from haversack.pipeline import mask_on_grid
    src = Grid((4, 4, 4), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    m = np.zeros((4, 4, 4), bool)
    m[1:3, 1:3, 1:3] = True
    np.testing.assert_array_equal(mask_on_grid(m, src, src), m)
    half = Grid((8, 8, 8), (0.5, 0.5, 0.5), (-0.25, -0.25, -0.25))
    np.testing.assert_array_equal(mask_on_grid(m, src, half), m.repeat(2, 0).repeat(2, 1).repeat(2, 2))
