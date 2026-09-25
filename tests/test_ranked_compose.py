"""A union's parts composed into the task's one ranked field (haversack.ranked_compose).

The properties the store rests on, each on synthetic parts encoded by the real encoder:
the composed argmax IS the painted label on the model grid; a gap between two parts' labels
is the later part's own claim (not twice it - the construction that doubled it swelled thin
structures under a linear restore); slab-by-slab encoding is the whole field's encoding byte
for byte; and parts on different grids are refused rather than composed voxel by voxel wrongly.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rankfield = pytest.importorskip("rankfield")

from haversack import ranked_compose as rc   # noqa: E402
from haversack.errors import InputError      # noqa: E402

SHAPE = (7, 9, 8)


def _part(name, logits, lut, depth=4, clip=8.0, grid=SHAPE):
    code = rankfield.encode(torch.as_tensor(logits, dtype=torch.float32), depth=depth, clip=clip)
    meta = dict(code.meta, labels=[int(v) for v in lut], model_grid=list(grid),
                envelope={"start": [0, 0, 0], "stop": list(grid)}, spacing_zyx=[1.5, 1.5, 1.5],
                softmax={"weights": name, "classes": len(lut)})
    return name, (code.ranks, code.support), meta, np.asarray(logits)


def _painted(parts):
    out = np.zeros(SHAPE, np.int64)
    for _n, _a, meta, logits in parts:
        win = logits.argmax(0)
        lut = np.asarray(meta["labels"])
        out = np.where(win != 0, lut[win], out)      # a later claim paints over an earlier one
    return out


def _union(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 2, (4,) + SHAPE)
    b = rng.normal(0, 2, (3,) + SHAPE)
    c = rng.normal(0, 2, (3,) + SHAPE)
    a[0] += 1.0                                       # background a little more likely
    b[0] += 1.5
    c[0] += 2.0
    # part a's channel 3 maps to 0 (a class the union drops: it paints background), and
    # label 5 is painted by two parts, as a remap can make it
    return [_part("a", a, [0, 1, 2, 0]), _part("b", b, [0, 5, 6]), _part("c", c, [0, 5, 7])]


def _compose(parts, **kw):
    return rc.compose([(n, arrs, m) for n, arrs, m, _l in parts], depth=4, device="cpu", **kw)


def _argmax_labels(ranks, labels):
    return np.asarray(labels)[ranks[0].astype(np.int64) - 1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_composed_argmax_is_the_painted_label(seed):
    parts = _union(seed)
    ranks, _support, meta, labels = _compose(parts)
    got = _argmax_labels(ranks, labels)
    np.testing.assert_array_equal(got, _painted(parts))
    assert labels == [0, 1, 2, 5, 6, 7]
    assert meta["scores"] == "composed" and meta["clip"] == rc.COMPOSED_CLIP
    assert meta["tail_temperatures"] == [] and meta.get("max_tail") is None
    assert [p["part"] for p in meta["composed"]["parts"]] == ["a", "b", "c"]
    assert meta["composed"]["margin_scale"] == rc.MARGIN_SCALE


def test_a_gap_between_two_parts_is_the_later_parts_own_claim():
    """Where the first part paints firmly and the later part declines by c, the composed gap
    from the winner to the later part's label is |c| - the size of the later part's own
    field there - not 2|c|, which reached the clip in half the distance and floored."""
    a = np.zeros((2,) + SHAPE)
    a[1] = 6.0                                        # part a claims everywhere by 6
    b = np.zeros((2,) + SHAPE)
    b[0] = np.linspace(0.5, 3.0, SHAPE[2])            # part b declines by 0.5 .. 3.0 along x
    parts = [_part("a", a, [0, 1]), _part("b", b, [0, 2])]
    ranks, support, meta, labels = _compose(parts)
    field = rankfield.RankField(ranks=ranks, support=support, tail=None, meta=meta, labels=labels)
    d2 = rankfield.deficit(field, labels.index(2))
    np.testing.assert_allclose(-d2[0, 0], b[0][0, 0], rtol=0.03)     # the log curve's quantum


def test_encoding_slab_by_slab_is_encoding_the_whole_field():
    parts = _union(3)
    whole = _compose(parts, slab=SHAPE[0])
    for slab in (1, 2, 3):
        got = _compose(parts, slab=slab)
        np.testing.assert_array_equal(got[0], whole[0])
        np.testing.assert_array_equal(got[1], whole[1])


def test_the_batch_decode_is_rankfields_deficit():
    name, (ranks, support), meta, _l = _union(4)[0]
    field = rankfield.RankField(ranks=ranks, support=support, tail=None, meta=meta,
                                labels=meta["labels"])
    got = rc.deficits(ranks, support, meta, 0, SHAPE[0], "cpu").numpy()
    for ch in range(meta["classes"]):
        np.testing.assert_array_equal(got[ch], rankfield.deficit(field, ch))


def test_parts_on_different_grids_are_refused():
    parts = _union(5)
    other = dict(parts[1][2], envelope={"start": [1, 0, 0], "stop": list(SHAPE)})
    moved = [parts[0], (parts[1][0], parts[1][1], other, parts[1][3])]
    with pytest.raises(InputError, match="same grid|share one grid"):
        _compose(moved)


def test_one_part_is_not_a_union():
    with pytest.raises(ValueError):
        _compose(_union(6)[:1])


def test_a_single_model_store_restores_exactly_under_a_linear_restore(tmp_path, monkeypatch):
    """Composition is for unions only: a one-model task's store is the model's own field, and
    restoring it linearly reproduces the run's labels voxel for voxel, as it always did."""
    pytest.importorskip("SimpleITK")
    pytest.importorskip("zarr")
    pytest.importorskip("duckn")
    import SimpleITK as sitk
    from test_normalization_sharing import ORGANS, _two_part_task, _write_ct
    from test_ranked_output import _ShapedStub

    from haversack import pipeline
    from haversack import ranked_restore as rr
    from haversack.ranked_output import segment_to_store
    from haversack.tasks import TaskSpec, UnionPart
    _spec, store, cache = _two_part_task(tmp_path, [_ShapedStub(ORGANS._props)])
    spec = TaskSpec(name="stub_one", shape="union",
                    union=(UnionPart(weights_id=1, label_remap={1: 1}, name="only"),),
                    label_map={1: "a"})
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    seg, path = segment_to_store(str(_write_ct(tmp_path)), spec, tmp_path / "one.duckn",
                                 case="one", models=cache, device="cpu", envelope_mm=None,
                                 grid=1.0, interp="linear", convention="corner", folds=(0,),
                                 quiet=True)
    res = rr.restore(path, grid=1.0, interp="linear")
    got = sitk.GetArrayFromImage(res.image("input"))
    assert res.parts and len(res.parts) == 1
    np.testing.assert_array_equal(got, seg.array)
