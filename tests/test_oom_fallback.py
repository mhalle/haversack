"""The on-device out-of-memory fallback must let go of the failed attempt before it retries.

Until 2026-09-11 ``TorchModel.predict_logits`` ran its host retry INSIDE the ``except`` block
that caught the out-of-memory error. While an exception is being handled, its traceback holds
every frame between the handler and the failed allocation, and each frame holds its locals -
the device accumulator, the batched input, the activations. So ``empty_cache()`` had nothing
it could release, and the retry ran beside all of it. Seen on Modal: a CADS ResEnc-L
checkpoint in fp32 on a 22 GiB A10 raised on an 864 MiB allocation with 19.08 GiB still
allocated by PyTorch, where the same model in fp16 on the same card ran. The PyTorch FAQ names
the pitfall ("My out of memory exception handler can't allocate memory").

The pinning is Python's semantics, not the device's, so these run on the CPU: a stand-in
network raises out-of-memory where the real one would, and a weak reference says whether what
the failed attempt allocated is still alive when the retry begins.
"""
import sys
import tempfile
import unittest
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("acvl_utils")        # predict_logits pads with nnU-Net's own helper
import haversack.network as N  # noqa: E402

K = 3
PATCH = (8, 8, 8)
SHAPE = (24, 8, 8)                        # five patches along z at nnU-Net's half-patch step
OOM = "CUDA out of memory. Tried to allocate 864.00 MiB"


def _slicers(shape):
    return [(slice(None), slice(z, z + PATCH[0]), slice(0, PATCH[1]), slice(0, PATCH[2]))
            for z in range(0, shape[0] - PATCH[0] + 1, PATCH[0] // 2)]


class _Net:
    """Stands in for the network. It is a pointwise function of its input, so every patch
    agrees on every voxel and the accumulated logits are known exactly: ``x + k`` in channel
    ``k``. On the calls numbered in ``fail`` it raises, first taking a weak reference to the
    accumulator of the ``_accumulate`` frame that called it, if one did. On every call it
    records which of those accumulators are still alive."""

    def __init__(self, fail=(), error=None):
        self.fail = set(fail)
        self.error = error or (lambda: torch.OutOfMemoryError(OOM))
        self.calls, self.refs, self.alive = 0, [], {}

    def __call__(self, x):
        self.calls += 1
        self.alive[self.calls] = [r() is not None for r in self.refs]
        if self.calls in self.fail:
            f = sys._getframe(1)
            while f is not None and f.f_code.co_name != "_accumulate":
                f = f.f_back
            if f is not None:
                self.refs.append(weakref.ref(f.f_locals["acc"]))
            del f                         # this frame is in the traceback too: hold nothing through it
            raise self.error()
        return x.expand(-1, K, -1, -1, -1) + torch.arange(K, dtype=x.dtype).view(1, K, 1, 1, 1)


def _model(net, *, batch_size="auto"):
    """A TorchModel with no weights behind it: only what predict_logits and the sliding window
    read, on the CPU."""
    m = N.TorchModel.__new__(N.TorchModel)
    m.device, m.dtype, m.K, m.patch = torch.device("cpu"), torch.float32, K, PATCH
    m.transpose_forward = m.transpose_backward = (0, 1, 2)
    m.accumulate, m.activation_reserve_gb, m.batch_size = "auto", N.DEFAULT_ACTIVATION_RESERVE_GB, batch_size
    m.accumulate_choice = m.batch_choice = None
    m.fold_params, m._load_fold = [None], lambda i: None
    m.predictor = SimpleNamespace(_internal_get_sliding_window_slicers=_slicers)
    m.net, m._gaussian_cpu = net, torch.ones(PATCH)
    m.gaussian, m._on_device = m._gaussian_cpu, True
    return m


def _on_device(*a, **k):
    # the CPU policy always answers "host"; these tests need the placement that can fall back
    return True, "test: the accumulator fits on the device"


def _volume():
    return torch.randint(0, 8, (1, *SHAPE), generator=torch.Generator().manual_seed(0)).float()


def _expected(x):
    return torch.stack([x[0] + k for k in range(K)])


def _run(m, x):
    with warnings.catch_warnings(record=True) as caught, mock.patch.object(N, "choose_accumulate", _on_device):
        warnings.simplefilter("always")
        out = m.predict_logits(x)
    return out, [str(w.message) for w in caught]


class TheFailedAttemptIsReleasedBeforeTheRetry(unittest.TestCase):

    def test_a_tensor_only_the_failed_attempt_held_is_gone_when_the_retry_starts(self):
        """The mechanism alone: the first attempt holds a tensor only in its own frame and runs
        out of memory; by the time the host retry starts, nothing may reference that tensor.
        Before the fix the handler's traceback did, through the failed attempt's frame."""
        seen = {}
        m = _model(_Net())

        def sliding_window(padded, slicers, *, force_host=False, report=None, **kw):
            if not seen:
                held = torch.empty(1 << 20)                  # what the failed attempt allocated
                seen["ref"] = weakref.ref(held)
                m.accumulate_choice = {"on_device": True, "why": "test"}
                m.batch_choice = {"batch": 1, "why": "test"}
                raise torch.OutOfMemoryError(OOM)
            seen["alive at the retry"] = seen["ref"]() is not None
            seen["force_host"] = force_host
            m.accumulate_choice = {"on_device": False, "why": "forced host"}
            return torch.zeros((K, *padded.shape[1:]), dtype=torch.half)

        m._sliding_window = sliding_window
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m.predict_logits(_volume())
        self.assertTrue(seen["force_host"], "the retry after an out-of-memory error at batch 1 is the host")
        self.assertFalse(seen["alive at the retry"],
                         "the retry ran while the failed attempt's tensor was still referenced")

    def test_the_real_accumulator_is_gone_when_the_retry_starts(self):
        """The same through the real sliding window: the accumulator ``_accumulate`` allocated
        on the device must be garbage by the retry's first forward pass."""
        net = _Net(fail={2})                  # call 1 is the first patch; call 2 the first batch of four
        m = _model(net, batch_size=4)
        x = _volume()
        out, _ = _run(m, x)
        self.assertEqual(len(net.refs), 1, "the stand-in never saw the accumulator; the test proves nothing")
        self.assertEqual(net.alive[3], [False],
                         "the retry's first patch ran beside the failed attempt's accumulator")
        torch.testing.assert_close(out.float(), _expected(x))


class TheFallbackStepsDown(unittest.TestCase):

    def test_an_oom_at_batch_4_is_retried_at_batch_1_on_the_device(self):
        """Batch 1 on the device is what the measured placement policy actually approved; only
        the batch above it was the estimate. So a batch that runs out of memory costs a retry at
        1, not the host path's slower loop."""
        net = _Net(fail={2})
        m = _model(net, batch_size=4)
        x = _volume()
        out, warned = _run(m, x)
        self.assertTrue(m.accumulate_choice["on_device"], m.accumulate_choice)
        self.assertEqual(m.batch_choice["batch"], 1, m.batch_choice)
        self.assertIn("batch 4 ran out of memory", m.batch_choice["why"])
        self.assertIn(OOM, m.batch_choice["why"])
        self.assertEqual(net.calls, 2 + 5, "the failed attempt's two calls, then five patches at batch 1")
        self.assertTrue(any("retrying at batch 1" in w and OOM in w for w in warned), warned)
        torch.testing.assert_close(out.float(), _expected(x))

    def test_an_oom_at_batch_1_moves_the_accumulator_to_the_host(self):
        """Today's fallback, kept: the warning, and a placement whose why says what happened."""
        net = _Net(fail={2})                  # auto: batch 1 off CUDA; call 2 is its second patch
        m = _model(net)
        x = _volume()
        out, warned = _run(m, x)
        self.assertFalse(m.accumulate_choice["on_device"], m.accumulate_choice)
        self.assertTrue(m.accumulate_choice["why"].startswith("forced host after an out-of-memory fallback"),
                        m.accumulate_choice)
        self.assertIn(OOM, m.accumulate_choice["why"])
        self.assertTrue(any(w.startswith("on-device accumulation ran out of memory") and OOM in w
                            and w.endswith("falling back to host") for w in warned), warned)
        self.assertEqual(net.alive[3], [False])
        torch.testing.assert_close(out.float(), _expected(x))

    def test_batch_4_then_batch_1_then_the_host(self):
        net = _Net(fail={2, 4})               # batch 4 fails at its first batch, batch 1 at its second patch
        m = _model(net, batch_size=4)
        x = _volume()
        out, warned = _run(m, x)
        self.assertFalse(m.accumulate_choice["on_device"], m.accumulate_choice)
        self.assertEqual(net.alive[5], [False, False],
                         "the host run's first patch ran beside a failed attempt's accumulator")
        self.assertEqual(sum("retrying at batch 1" in w for w in warned), 1, warned)
        self.assertEqual(sum("falling back to host" in w for w in warned), 1, warned)
        torch.testing.assert_close(out.float(), _expected(x))

    def test_when_the_host_retry_fails_too_its_error_names_what_failed_before(self):
        """Retrying inside the handler chained the errors for free. Outside it, the last error
        has to carry the earlier ones itself, or the log shows only the last allocation."""
        net = _Net(fail={2, 4})               # the device run fails at call 2, the host run at call 4
        m = _model(net)
        with self.assertRaises(torch.OutOfMemoryError) as cm:
            _run(m, _volume())
        notes = getattr(cm.exception, "__notes__", [])
        self.assertTrue(any(OOM in n and "batch 1" in n for n in notes), notes)

    def test_an_error_that_is_not_out_of_memory_is_raised_without_a_retry(self):
        net = _Net(fail={2}, error=lambda: RuntimeError("non-finite logits after accumulation"))
        m = _model(net, batch_size=4)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _run(m, _volume())
        self.assertEqual(net.calls, 2)

    def test_an_oom_in_the_first_patch_is_raised_as_itself(self):
        """The first patch runs before either policy has decided anything, so there is no
        placement to fall back from, and every fallback runs that same patch the same way first.
        The handler read the model's accumulate_choice regardless: on a fresh model it was None,
        so the out-of-memory error surfaced as a TypeError; on a model the cache had already used,
        it was the previous volume's placement, and steered a retry that could not help."""
        for previous in (None, {"on_device": True, "why": "the previous volume's"}):
            with self.subTest(previous=previous):
                net = _Net(fail={1})
                m = _model(net)
                m.accumulate_choice = previous
                with self.assertRaises(torch.OutOfMemoryError):
                    _run(m, _volume())
                self.assertEqual(net.calls, 1)


class _StubModel:
    """Just enough of a model for segment(): it reports that the fallback moved its accumulator."""

    spacing_zyx = (1.5, 1.5, 1.5)
    normalization_schemes = ("CTNormalization",)
    use_mask_for_norm = (False,)
    transpose_forward = (0, 1, 2)
    K = 2
    WHY = f"forced host after an out-of-memory fallback (batch 1, accumulator on cuda: {OOM})"

    def __init__(self):
        self.accumulate_choice = {"on_device": True, "why": "the policy's"}

    def intensity_properties(self, channel):
        return {"mean": 0.0, "std": 100.0, "percentile_00_5": -1000.0, "percentile_99_5": 1000.0}

    def predict_logits(self, crop, report=None):
        self.accumulate_choice = {"on_device": False, "why": self.WHY}
        logits = torch.zeros((self.K, *crop.shape[1:]))
        logits[0] = 1.0
        return logits


class TheDeviationSaysWhy(unittest.TestCase):

    def test_a_forced_device_run_that_fell_back_records_the_reason(self):
        """segment() read the reason from a key the model never writes ("reason"; the model
        writes "why"), so the deviation a forced ``--accumulate device`` run records when the
        fallback moves it to the host had an empty why since it was written."""
        pytest.importorskip("SimpleITK")
        import numpy as np
        import SimpleITK as sitk

        from haversack import pipeline
        from haversack.tasks import TaskSpec, UnionPart

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Dataset000_stub" / "trainer__plans__3d_fullres"
            (folder / "fold_0").mkdir(parents=True)

            class _Store:
                root = folder.parent.parent

                def resolve(self, weights_id, *, configuration=None):
                    return folder

                def describe(self, *a, **k):
                    return {}

            class _Cache:
                def __init__(self):
                    self.order = [_StubModel(), _StubModel()]

                def get(self, folder, **kw):
                    return self.order.pop(0)

                def release(self, model):
                    pass

            spec = TaskSpec(name="stub_union", shape="union",
                            union=(UnionPart(weights_id=1, label_remap={1: 1}, name="first"),
                                   UnionPart(weights_id=2, label_remap={1: 2}, name="second")),
                            label_map={1: "a", 2: "b"})
            img = sitk.GetImageFromArray(np.full((12, 14, 16), -1000, dtype=np.int16))
            img.SetSpacing((1.5, 1.5, 1.5))
            ct = Path(tmp) / "ct.nii.gz"
            sitk.WriteImage(img, str(ct))
            with mock.patch.object(pipeline, "as_store", lambda *a, **k: _Store()):
                seg = pipeline.segment(str(ct), spec, models=_Cache(), device="cpu", accumulate="device",
                                       envelope_mm=None, convention="corner", folds=(0,))
        placement = [d for d in seg.provenance["deviations"] if d["what"] == "accumulator placement"]
        self.assertEqual(len(placement), 2, seg.provenance["deviations"])
        for d in placement:
            self.assertEqual((d["requested"], d["effective"], d["why"]), ("device", "host", _StubModel.WHY))


if __name__ == "__main__":
    unittest.main()
