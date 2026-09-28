"""Two defects found by the 2026-09-28 audit: every lineage's input was truncated to int32, and a
TotalSegmentator cascade's final stage labeled one voxel past its crop box."""
import numpy as np
import pytest
import torch

from haversack.pipeline import keep_band
from haversack.preprocess import to_model_grid
from haversack.restore import to_labels
from haversack.values import Geometry


def _geometry(spacing, shape):
    return Geometry(spacing_zyx=(spacing,) * 3, shape_zyx=shape, origin_xyz=(0, 0, 0),
                    direction_xyz=(1, 0, 0, 0, 1, 0, 0, 0, 1))


@pytest.mark.parametrize("convention", ["corner", "center"])
def test_only_the_ts_lineage_truncates_the_resampled_image(convention):
    """A PET SUV or scaled MRI keeps its fractional part; TotalSegmentator's astype(int32) is
    applied only when asked (truncate=True, the TS lineage)."""
    n = (12, 14, 16)
    data = np.random.default_rng(0).uniform(0, 3, size=n).astype(np.float32)
    kept = to_model_grid(data, _geometry(1.0, n), (1.5,) * 3, convention=convention, device="cpu",
                         order=1, truncate=False)
    cut = to_model_grid(data, _geometry(1.0, n), (1.5,) * 3, convention=convention, device="cpu",
                        order=1, truncate=True)
    kept_values = np.asarray(kept.data_zyx)
    assert kept_values.dtype == np.float32
    assert np.any(np.abs(kept_values - np.round(kept_values)) > 0.1)          # fractions survive
    assert set(np.unique(np.asarray(cut.data_zyx))) <= {0, 1, 2}                # truncated, as TS does
    same = to_model_grid(data, _geometry(1.5, n), (1.5,) * 3, convention=convention, device="cpu",
                         order=1, truncate=False)                            # no resample needed
    np.testing.assert_allclose(np.asarray(same.data_zyx), data, atol=1e-6)


@pytest.mark.parametrize("spacing", [0.5, 0.7, 0.75, 1.0])
@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_nothing_outside_a_cascade_crop_box_is_labeled(spacing, interp):
    n = (120, 200, 200)
    box = ((30, 50, 60), (90, 150, 140))
    grid = to_model_grid(np.zeros(n, np.float32), _geometry(spacing, n), (1.5,) * 3, convention="corner",
                         device="cpu", order=1, box=box)
    fr = grid.frame
    logits = torch.zeros((2, *fr.model_shape))
    logits[1] = 1.0                                                          # class 1 all over the box
    out = torch.zeros(n, dtype=torch.uint8)
    to_labels(logits, fr.source, fr.mapping(fr.source), interp=interp, outside="background",
              lut=np.arange(2), out=out, backend="torch")
    out = keep_band(out, {k: (box[0][k], box[1][k]) for k in range(3)}, fr.source, fr.source)
    inside = np.zeros(n, bool)
    inside[30:90, 50:150, 60:140] = True
    labeled = out.numpy() > 0
    assert not (labeled & ~inside).any()
    assert labeled[inside].mean() > 0.99                                     # and the box is still labeled
