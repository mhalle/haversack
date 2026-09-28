"""A native nnU-Net model's input is what nnU-Net's own preprocessor feeds it: crop to the
nonzero box, normalize, then resample (separate-z for an anisotropic image). Until 2026-09-28
haversack resampled first and normalized after, TotalSegmentator's order, for every lineage."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from haversack.preprocess import nnunet_resampling_from_config, to_nnunet_input
from haversack.errors import UnsupportedModel
from haversack.values import Geometry

CT_PROPS = {"mean": 60.0, "std": 180.0, "percentile_00_5": -900.0, "percentile_99_5": 1200.0}


def _plans(spacing, scheme, use_mask):
    cfg = {"data_identifier": "x", "preprocessor_name": "DefaultPreprocessor", "spacing": list(spacing),
           "normalization_schemes": [scheme], "use_mask_for_norm": [use_mask], "patch_size": [32, 32, 32],
           "architecture": {"network_class_name": "unused", "arch_kwargs": {}, "_kw_requires_import": []},
           "resampling_fn_data": "resample_data_or_seg_to_shape",
           "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
           "resampling_fn_seg": "resample_data_or_seg_to_shape",
           "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
           "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
           "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0,
                                                  "force_separate_z": None}}
    return {"dataset_name": "Dataset999_T", "plans_name": "nnUNetPlans", "transpose_forward": [0, 1, 2],
            "transpose_backward": [0, 1, 2], "configurations": {"3d_fullres": cfg},
            "foreground_intensity_properties_per_channel": {"0": CT_PROPS},
            "original_median_spacing_after_transp": list(spacing), "image_reader_writer": "SimpleITKIO",
            "label_manager": "LabelManager", "experiment_planner_used": "x"}


class _Model:
    def __init__(self, plans):
        self.cm = PlansManager(plans).get_configuration("3d_fullres")
        self.spacing_zyx = tuple(self.cm.spacing)
        self.normalization_schemes = tuple(self.cm.normalization_schemes)
        self.use_mask_for_norm = tuple(self.cm.use_mask_for_norm)
        self._props = plans["foreground_intensity_properties_per_channel"]["0"]

    def intensity_properties(self, channel):
        return dict(self._props)

    def resampling(self, kind):
        return nnunet_resampling_from_config(self.cm.configuration, kind)


def _image(shape, seed, zero_border):
    rng = np.random.default_rng(seed)
    img = (rng.normal(size=shape) * 300 + 50).astype(np.float32)
    if zero_border:                      # MRI-like: exact zeros outside, a hole inside
        img[:2] = 0
        img[:, -3:] = 0
        img[:, :, :1] = 0
        c = tuple(s // 2 for s in shape)
        img[c[0], c[1] - 1:c[1] + 1, c[2] - 1:c[2] + 1] = 0
    return img


CASES = [   # (target spacing, image spacing, shape, scheme, use_mask, zero border)
    ((1.5, 1.5, 1.5), (1.0, 0.8, 0.8), (20, 26, 24), "CTNormalization", False, False),
    ((1.5, 1.5, 1.5), (5.0, 0.7, 0.7), (8, 36, 40), "CTNormalization", False, True),      # separate-z
    ((2.0, 2.0, 2.0), (1.2, 1.0, 1.0), (18, 22, 20), "ZScoreNormalization", False, True),
    ((2.0, 2.0, 2.0), (4.5, 1.0, 1.0), (9, 24, 26), "ZScoreNormalization", True, True),    # masked, sep-z
]


@pytest.mark.parametrize("target,spacing,shape,scheme,use_mask,zero_border", CASES)
def test_matches_nnunet_preprocessor(target, spacing, shape, scheme, use_mask, zero_border):
    plans = _plans(target, scheme, use_mask)
    model = _Model(plans)
    img = _image(shape, sum(shape), zero_border)
    pm = PlansManager(plans)
    want, _, props = DefaultPreprocessor(verbose=False).run_case_npy(
        img[None].copy(), None, {"spacing": list(spacing)}, pm, pm.get_configuration("3d_fullres"), {})
    geometry = Geometry(spacing_zyx=spacing, shape_zyx=shape, origin_xyz=(0.0, 0.0, 0.0),
                        direction_xyz=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    x, frame = to_nnunet_input(img, geometry, model, device="cpu")
    got = x.numpy()
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3 * max(1.0, float(np.abs(want).max())))
    # the crop is recorded, so the restore undoes it
    box = props["bbox_used_for_cropping"]
    if frame.model_source is not None:
        assert frame.model_source.shape == tuple(int(b[1] - b[0]) for b in box)


def test_another_resampling_function_is_refused():
    cfg = {"resampling_fn_data": "no_resampling_hack", "resampling_fn_data_kwargs": {}}
    with pytest.raises(UnsupportedModel):
        nnunet_resampling_from_config(cfg, "data")
    assert nnunet_resampling_from_config({}, "probabilities") == {"order": 1, "order_z": 0,
                                                                  "force_separate_z": None}
