"""nnU-Net's separate-z resample, reproduced: a native model given an anisotropic image sees
what nnU-Net's own preprocessing feeds it (per-slice in-plane cubic, then nearest along the
low-resolution axis), not a cubic spline along all three axes."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
drs = pytest.importorskip("nnunetv2.preprocessing.resampling.default_resampling")
from haversack.resample import compute_nnunet_shape, resample_data, separate_z_axis, separate_z_axis_matrix

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
          (["cuda"] if torch.cuda.is_available() else [])

# (current spacing, target spacing, shape): thick slices -> iso, iso -> thick target, a
# downsample by exactly 2 along z (every nearest pick an exact tie), the low-res axis in-plane
CASES = [
    ((5.0, 0.7, 0.7), (1.5, 1.5, 1.5), (9, 40, 44)),
    ((1.0, 1.0, 1.0), (4.0, 1.0, 1.0), (21, 12, 14)),
    ((6.0, 1.5, 1.5), (3.0, 1.5, 1.5), (10, 16, 18)),
    ((3.0, 1.5, 1.5), (12.0, 1.5, 1.5), (24, 11, 13)),
    ((0.8, 4.0, 0.8), (1.0, 1.0, 1.0), (30, 8, 26)),
]


def _reference(data, cur, new, order=3, order_z=0):
    shape = drs.compute_new_shape(data.shape, cur, new)
    out = drs.resample_data_or_seg_to_shape(data[None].astype(np.float32), shape, cur, new, is_seg=False,
                                            order=order, order_z=order_z, force_separate_z=None)
    return out[0], tuple(int(s) for s in shape)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cur,new,shape", CASES)
def test_matches_nnunet_resample(device, cur, new, shape):
    rng = np.random.default_rng(sum(shape))
    data = (rng.normal(size=shape) * 300 + 40).astype(np.float32)
    want, want_shape = _reference(data, cur, new)
    axis = separate_z_axis(cur, new)
    assert axis is not None                                   # every case here is separate-z
    assert compute_nnunet_shape(shape, cur, new) == want_shape
    got = resample_data(data, want_shape, convention="center", order=3, device=device,
                        separate_z_axis=axis, order_z=0)
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-3 * np.abs(want).max())


def test_nearest_ties_are_nnunets():
    """Downsampling by exactly 2 puts every sample on a tie; the picks must be nnU-Net's."""
    from scipy.ndimage import map_coordinates
    for n_in, n_out in [(8, 4), (10, 5), (7, 14), (9, 3)]:
        m = separate_z_axis_matrix(n_in, n_out, 0)
        coords = float(n_in) / n_out * (np.arange(n_out) + 0.5) - 0.5
        want = map_coordinates(np.arange(n_in, dtype=float), [coords], order=0, mode="nearest")
        np.testing.assert_array_equal(m @ np.arange(n_in, dtype=float), want)


def test_isotropic_is_not_separate():
    assert separate_z_axis((1.5, 0.8, 0.8), (1.5, 1.5, 1.5)) is None
    assert separate_z_axis((2.0, 0.5, 0.5), (1.0, 1.0, 1.0)) == 0             # ratio 4
    assert separate_z_axis((3.0, 3.0, 0.9), (1.0, 1.0, 1.0)) is None          # two low-res axes
    assert separate_z_axis((3.0, 1.0, 1.0), (1.0, 1.0, 1.0)) is None          # ratio 3: not above
    assert separate_z_axis((3.0, 1.0, 1.0), (1.0, 1.0, 1.0), force_separate_z=True) == 0


def test_the_rule_is_nnunets():
    """separate_z_axis restates nnU-Net's determine_do_sep_z_and_axis; they must agree."""
    rng = np.random.default_rng(0)
    choices = [0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0, 4.5, 5.0, 6.0]
    for _ in range(3000):
        cur = tuple(float(v) for v in rng.choice(choices, 3))
        new = tuple(float(v) for v in rng.choice(choices, 3))
        force = [None, True, False][int(rng.integers(3))]
        do, axis = drs.determine_do_sep_z_and_axis(force, list(cur), list(new))
        want = int(axis) if do and axis is not None else None
        assert separate_z_axis(cur, new, force) == want, (cur, new, force)
