"""Which restore backend "auto" takes, decided by what each fused kernel can address.

Until 2026-09-11 "auto" took the device's fused kernel by device type alone, and the Triton
kernel refused any field of K x Zs x Ys x Xs >= 2^31 with a ValueError nothing caught: the
open CADS head-and-neck model (K=30) on a whole-body PET/CT's attenuation CT (334x334x678 at
1.5 mm, 2.27e9 logits) with ``--envelope 0`` failed ``segment`` outright on an A10G. The kernel
now takes a 64-bit channel base, as the Metal one always has, so that field stays on it; what
a fused kernel still cannot address - a channel, or for Triton the output, of 2^31 voxels or
more - "auto" hands to the torch backend, and the run records it.

The decision is a function of shapes alone, so it is tested here without a GPU and without
the memory such a field needs; `tools/restore_limits_modal.py` runs the real fields on CUDA.
"""
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import haversack as lg
from haversack import Mapping, backends
from haversack.backends import metal, torch_gather, triton_gpu

from conftest import voronoi_logits

CUDA, MPS, CPU = torch.device("cuda"), torch.device("mps"), torch.device("cpu")
LIMIT = 2 ** 31
# the field that failed: K=30 on 334x334x678 (ZYX 678, 334, 334), restored to 512x512x311
REPORTED = ((30, 678, 334, 334), (311, 512, 512))
CHANNEL_AT_LIMIT = (2, 2048, 1024, 1024)             # one channel of exactly 2^31 voxels
CHANNEL_SHORT = (2, 1, 1, LIMIT - 1)
SMALL_OUT = (64, 64, 64)


class AutoOnCuda(unittest.TestCase):

    def setUp(self):
        p = mock.patch.object(triton_gpu, "available", lambda: True)
        p.start()
        self.addCleanup(p.stop)

    def test_the_reported_field_stays_on_triton(self):
        assert REPORTED[0][0] * np.prod(REPORTED[0][1:]) >= LIMIT        # the case really is past it
        self.assertEqual(backends.select("auto", CUDA, *REPORTED), ("triton", triton_gpu, None))

    def test_a_channel_of_2_31_voxels_goes_to_torch_and_says_why(self):
        c = backends.select("auto", CUDA, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertEqual((c.name, c.module), ("torch", torch_gather))
        self.assertIn("triton", c.fallback)
        self.assertIn("2,147,483,648", c.fallback)

    def test_a_channel_one_voxel_short_of_2_31_stays_on_triton(self):
        self.assertEqual(backends.select("auto", CUDA, CHANNEL_SHORT, SMALL_OUT).name, "triton")

    def test_an_output_of_2_31_voxels_goes_to_torch(self):
        """Triton's output offsets are 32-bit: past 2^31 ``pid * BLOCK`` wraps negative, passes
        the ``offs < n_out`` mask and writes before the buffer. It is refused, not launched."""
        c = backends.select("auto", CUDA, (30, 100, 100, 100), (2048, 1024, 1024))
        self.assertEqual(c.name, "torch")
        self.assertIn("output", c.fallback)
        self.assertEqual(backends.select("auto", CUDA, (30, 100, 100, 100), (1, 1, LIMIT - 1)).name, "triton")

    def test_asking_for_triton_by_name_refuses_what_it_cannot_take(self):
        with self.assertRaises(ValueError) as e:
            backends.select("triton", CUDA, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertIn("backend='torch'", str(e.exception))

    def test_without_triton_auto_is_torch_and_nothing_fell_back(self):
        with mock.patch.object(triton_gpu, "available", lambda: False):
            self.assertEqual(backends.select("auto", CUDA, *REPORTED), ("torch", torch_gather, None))


class AutoOnMps(unittest.TestCase):

    def setUp(self):
        p = mock.patch.object(metal, "available", lambda: True)
        p.start()
        self.addCleanup(p.stop)

    def test_the_reported_field_stays_on_metal(self):
        self.assertEqual(backends.select("auto", MPS, *REPORTED), ("metal", metal, None))

    def test_a_channel_of_2_31_voxels_goes_to_torch(self):
        c = backends.select("auto", MPS, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertEqual(c.name, "torch")
        self.assertIn("metal", c.fallback)
        self.assertEqual(backends.select("auto", MPS, CHANNEL_SHORT, SMALL_OUT).name, "metal")

    def test_a_large_output_stays_on_metal(self):
        """Metal launches the output in z-slabs and indexes it with 64-bit offsets."""
        self.assertEqual(backends.select("auto", MPS, (30, 100, 100, 100), (2048, 1024, 1024)).name, "metal")


class AutoOnCpu(unittest.TestCase):

    def test_cpu_is_torch_whatever_the_field(self):
        for shape in (REPORTED, (CHANNEL_AT_LIMIT, SMALL_OUT)):
            self.assertEqual(backends.select("auto", CPU, *shape), ("torch", torch_gather, None))


class _PretendCpuHasTriton:
    """Make "auto" on the CPU prefer the Triton backend, with a limit small enough that a test
    field crosses it, and a ``run`` that fails if it is ever called - so a fallback that did not
    happen is a failure, not a CUDA error this machine cannot raise."""

    def setUp(self):
        self.triton_runs = []
        for p in (mock.patch.dict(backends.FUSED, {"cpu": "triton"}),
                  mock.patch.object(triton_gpu, "available", lambda: True),
                  mock.patch.object(triton_gpu, "run", lambda *a, **k: self.triton_runs.append(a))):
            p.start()
            self.addCleanup(p.stop)

    def shrink_the_limit(self, limit=100):
        p = mock.patch.object(triton_gpu, "OFFSET_LIMIT", limit)
        p.start()
        self.addCleanup(p.stop)


class ToLabelsFallsBack(_PretendCpuHasTriton, unittest.TestCase):

    SRC, OUT = (5, 6, 7), (9, 10, 11)

    def _logits(self):
        return torch.from_numpy(voronoi_logits(K=4, shape=self.SRC, seed=3))

    def test_auto_past_the_limit_runs_torch_and_warns(self):
        self.shrink_the_limit()
        mapping = Mapping.center(self.OUT, self.SRC)
        with pytest.warns(RuntimeWarning, match="triton"):
            got = lg.to_labels(self._logits(), self.OUT, mapping, backend="auto")
        self.assertEqual(self.triton_runs, [])
        want = lg.to_labels(self._logits(), self.OUT, mapping, backend="torch")
        np.testing.assert_array_equal(got.numpy(), want.numpy())
        self.assertTrue(len(np.unique(want.numpy())) > 1)                  # a field with something in it

    def test_auto_within_the_limit_still_takes_the_fused_kernel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lg.to_labels(self._logits(), self.OUT, Mapping.center(self.OUT, self.SRC), backend="auto")
        self.assertEqual(len(self.triton_runs), 1)


class _Model:
    """Just enough of a model for segment(), with logits that put label 1 in the upper half."""

    spacing_zyx = (1.5, 1.5, 1.5)
    normalization_schemes = ("CTNormalization",)
    use_mask_for_norm = (False,)
    transpose_forward = (0, 1, 2)
    K = 2
    accumulate_choice = {"on_device": False, "why": ""}

    def intensity_properties(self, channel):
        return {"mean": 0.0, "std": 100.0, "percentile_00_5": -1000.0, "percentile_99_5": 1000.0}

    def predict_logits(self, crop, report=None):
        logits = torch.zeros((self.K, *crop.shape[1:]))
        logits[1, crop.shape[1] // 2:] = 1.0
        return logits


class SegmentRecordsTheFallback(_PretendCpuHasTriton, unittest.TestCase):

    def _segment(self, tmp):
        sitk = pytest.importorskip("SimpleITK")
        from haversack import pipeline
        from haversack.tasks import TaskSpec, UnionPart

        folder = Path(tmp) / "Dataset000_stub" / "trainer__plans__3d_fullres"
        (folder / "fold_0").mkdir(parents=True)

        class _Store:
            root = folder.parent.parent

            def resolve(self, weights_id, *, configuration=None):
                return folder

            def describe(self, *a, **k):
                return {}

        class _Cache:
            def get(self, folder, **kw):
                return _Model()

            def release(self, model):
                pass

        spec = TaskSpec(name="stub", shape="union",
                        union=(UnionPart(weights_id=1, label_remap={1: 1}, name="first"),),
                        label_map={1: "a"})
        img = sitk.GetImageFromArray(np.full((12, 14, 16), -1000, dtype=np.int16))
        img.SetSpacing((1.5, 1.5, 1.5))
        ct = Path(tmp) / "ct.nii.gz"
        sitk.WriteImage(img, str(ct))
        stages = []
        with mock.patch.object(pipeline, "as_store", lambda *a, **k: _Store()), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)       # the pipeline records; it does not warn
            seg = pipeline.segment(str(ct), spec, models=_Cache(), device="cpu", envelope_mm=None,
                                   convention="corner", folds=(0,), progress=stages.append)
        return seg, stages

    def test_the_deviation_is_in_the_provenance_and_the_progress(self):
        import tempfile
        self.shrink_the_limit()
        with tempfile.TemporaryDirectory() as tmp:
            seg, stages = self._segment(tmp)
        (d,) = [d for d in seg.provenance["deviations"] if d["what"] == "restore backend"]
        self.assertEqual((d["requested"], d["effective"]), ("auto", "torch"))
        self.assertIn("triton", d["why"])
        self.assertTrue(any(p.stage == "restore" and "torch" in p.detail for p in stages),
                        [(p.stage, p.detail) for p in stages])
        self.assertEqual(self.triton_runs, [])
        import SimpleITK as sitk
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(seg.labels))), {0, 1})

    def test_a_field_the_kernel_takes_records_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            seg, _ = self._segment(tmp)
        self.assertEqual([d for d in seg.provenance["deviations"] if d["what"] == "restore backend"], [])
        self.assertEqual(len(self.triton_runs), 1)


if __name__ == "__main__":
    unittest.main()
