"""`grid="model"`: the network's own grid (found by the input-store soak, 2026-09-26).

The job schema and SERVER.md have offered it since the package was named; nothing resolved it,
and every such job failed "could not convert string to float: 'model'". The model grid is the
forward resampler's rule inverted and placed in the source's millimeters, so the mapping from
it to the model grid is the identity: restoring onto it returns the network's own voxels.
"""
import numpy as np
import pytest

from haversack.frame import Frame
from haversack.grid import Grid


def _frame(convention, *, crop=None, src=(40, 50, 60), model=(17, 23, 31)):
    source = Grid(src, (1.25, 0.7, 0.8), (0.0, 0.0, 0.0))
    model_source = None
    if crop is not None:
        lo, hi = crop
        model_source = Grid(tuple(h - l for l, h in zip(lo, hi)), source.spacing,
                            tuple(l * s for l, s in zip(lo, source.spacing)))
    return Frame(source=source, model_shape=model, model_spacing=(3.0, 1.5, 1.5),
                 convention=convention, canonical=None,
                 model_source=model_source)


@pytest.mark.parametrize("convention", ["corner", "center"])
@pytest.mark.parametrize("crop", [None, ((3, 5, 7), (37, 45, 50))])
def test_the_model_grid_maps_onto_the_model_grid_exactly(convention, crop):
    fr = _frame(convention, crop=crop)
    g = fr.resolve_grid("model")
    assert g.shape == fr.model_shape
    m = fr.mapping(g)
    np.testing.assert_allclose(m.a, (1.0, 1.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(m.b, (0.0, 0.0, 0.0), atol=1e-9)


def test_an_axis_collapsed_to_one_sample_keeps_the_source_spacing():
    fr = _frame("corner", src=(1, 50, 60), model=(1, 23, 31))
    g = fr.resolve_grid("model")
    assert g.spacing[0] == pytest.approx(1.25)


def test_segment_on_the_model_grid_returns_the_models_shape(tmp_path, monkeypatch):
    pytest.importorskip("nnunetv2")
    from test_cascade_union import _Part, _run
    from test_normalization_sharing import ORGANS
    from haversack.tasks import TaskSpec
    spec = TaskSpec(name="stub_single", shape="single", single=1, label_map={1: "a"})
    res, shape = _run(tmp_path, monkeypatch, [_Part(ORGANS._props, cover=(slice(None),) * 3)],
                      spec, grid="model")
    assert tuple(res.grid.shape) == tuple(res.provenance["output_grid"])
    assert res.array.shape == tuple(res.grid.shape)
