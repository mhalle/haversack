"""The accumulator's non-finite check, done with two reductions instead of a full-size mask.

``network.all_finite`` replaced ``torch.isfinite(acc).all()`` on 2026-10-05: on the CPU that
expression materialized intermediates of 2.25x an fp16 accumulator before reducing them, about
1.7 GB on a 755 MB field. The replacement is finite exactly when the tensor's max and min are, so
it must still say no to a NaN, a +inf or a -inf wherever it sits - first, in the middle, last -
in every float dtype the accumulator could be. Each case is checked against the expression it
replaced, not only against a hand-written answer, so the two cannot drift apart quietly.
"""
import contextlib
import unittest
import warnings
from unittest import mock

import pytest
import torch

import haversack.network as N

from conftest import device_names

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
BAD = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
SHAPE = (3, 5, 6, 7)


def _supported(device, dtype):
    return not (device == "mps" and dtype == torch.float64)       # MPS has no float64


def _cases():
    for device in device_names():
        for dtype in DTYPES:
            if _supported(device, dtype):
                yield pytest.param(device, dtype, id=f"{device}-{str(dtype)[6:]}")


def _field(device, dtype):
    # signed values well inside fp16's range, so the extremes are ordinary numbers of both signs
    g = torch.Generator().manual_seed(0)
    return (torch.rand(SHAPE, generator=g) * 20 - 10).to(device=device, dtype=dtype)


@pytest.mark.parametrize("device, dtype", list(_cases()))
def test_a_finite_tensor_passes(device, dtype):
    t = _field(device, dtype)
    assert N.all_finite(t) is True
    assert N.all_finite(t) == bool(torch.isfinite(t).all())


@pytest.mark.parametrize("bad", list(BAD))
@pytest.mark.parametrize("where", ["first", "middle", "last"])
@pytest.mark.parametrize("device, dtype", list(_cases()))
def test_one_non_finite_value_anywhere_is_caught(device, dtype, where, bad):
    t = _field(device, dtype)
    flat = t.view(-1)
    flat[{"first": 0, "middle": flat.numel() // 2, "last": -1}[where]] = BAD[bad]
    assert N.all_finite(t) is False
    assert N.all_finite(t) == bool(torch.isfinite(t).all())


@pytest.mark.parametrize("device, dtype", list(_cases()))
def test_nan_beside_both_infinities_is_caught(device, dtype):
    """A NaN must not hide behind an infinity of either sign in the reduction, nor they behind it."""
    t = _field(device, dtype)
    flat = t.view(-1)
    flat[1], flat[2], flat[3] = float("inf"), float("nan"), float("-inf")
    assert N.all_finite(t) is False


@pytest.mark.parametrize("device, dtype", list(_cases()))
def test_a_view_is_judged_by_its_own_elements(device, dtype):
    """A slice of a tensor holding a NaN elsewhere is still finite: the check reads the view."""
    t = _field(device, dtype)
    t[0, 0, 0, 0] = float("nan")
    assert N.all_finite(t[1:]) is True
    assert N.all_finite(t[:, :, :, :1]) is False


def test_an_empty_tensor_is_finite():
    """``isfinite(empty).all()`` is True, while ``amax`` of an empty tensor raises."""
    t = torch.empty((3, 0, 4), dtype=torch.float16)
    assert N.all_finite(t) is True
    assert N.all_finite(t) == bool(torch.isfinite(t).all())


K = 3
PATCH = (8, 8, 8)
GRID = (24, 8, 8)                       # five patches along z at nnU-Net's half-patch step
POISONS = [("nan", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf")),
           # past fp16's largest value (65504): the half accumulator itself turns these into
           # infinities, the way a real overflowing logit would arrive
           ("fp16 overflow", 1e5), ("fp16 negative overflow", -1e5)]


def _slicers(shape):
    return [(slice(None), slice(z, z + PATCH[0]), slice(0, PATCH[1]), slice(0, PATCH[2]))
            for z in range(0, shape[0] - PATCH[0] + 1, PATCH[0] // 2)]


def _model(poison, *, device="cpu", batch_size=1):
    """A TorchModel with no weights behind it, whose stand-in network writes ``poison`` into
    channel 1 of one voxel of its second forward pass - not the first patch, which the sliding
    window runs before choosing a placement, and at batch 4 the only other pass there is."""
    m = N.TorchModel.__new__(N.TorchModel)
    m.device, m.dtype, m.K, m.patch = torch.device(device), torch.float32, K, PATCH
    m.accumulate, m.activation_reserve_gb, m.batch_size = "auto", N.DEFAULT_ACTIVATION_RESERVE_GB, batch_size
    m.accumulate_choice = m.batch_choice = None
    m._gaussian_cpu = torch.ones(PATCH)
    m.gaussian = m._gaussian_cpu.to(device)
    m.calls, m.poisoned = 0, False

    def net(x):
        m.calls += 1
        out = x.expand(-1, K, -1, -1, -1) + torch.arange(K, dtype=x.dtype, device=x.device).view(1, K, 1, 1, 1)
        if poison is not None and m.calls == 2:
            out = out.clone()
            out[0, 1, 4, 4, 4] = poison
            m.poisoned = True
        return out

    m.net = net
    return m


def _on_device(*a, **k):
    # the CPU policy always answers "host"; the device path is what these tests need
    return True, "test: the accumulator fits on the device"


def _placements():
    """(device, on_device, batch): the host path, and the device path at batch 1 and in batches
    of four - on the CPU with the placement forced, and on each real accelerator present."""
    yield "cpu", False, 1
    for device in device_names():
        for batch in (1, 4):
            yield device, True, batch


class TheAccumulatorRefusesNonFiniteLogits(unittest.TestCase):
    """Through the real sliding window and its out-of-memory fallback, wherever the accumulator
    lives. Until 2026-10-05 only the host accumulator was checked."""

    def _run(self, poison, device, on_device, batch):
        m = _model(poison, device=device, batch_size=batch)
        x = torch.rand((1, *GRID), generator=torch.Generator().manual_seed(0))
        policy = mock.patch.object(N, "choose_accumulate", _on_device) if on_device else contextlib.nullcontext()
        try:
            with policy, warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return m._sliding_window_with_fallback(x, _slicers(GRID))
        finally:
            self.assertEqual(m.accumulate_choice["on_device"], on_device, "the test ran on the other path")
            self.assertEqual(m.batch_choice["batch"], batch if on_device else 1)
            self.assertEqual([str(w.message) for w in caught], [], "the run was retried")
            self.assertEqual(m.poisoned, poison is not None, "the stand-in never wrote the poison")

    def test_finite_logits_pass(self):
        for device, on_device, batch in _placements():
            with self.subTest(device=device, on_device=on_device, batch=batch):
                out = self._run(None, device, on_device, batch)
                self.assertEqual(out.dtype, torch.half)
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_nan_and_both_infinities_are_refused(self):
        """Raised once, where it arose: the error is not an out-of-memory one, so the fallback
        must not retry it on the host (which would raise again, a second run later)."""
        for device, on_device, batch in _placements():
            for name, poison in POISONS:
                with self.subTest(name, device=device, on_device=on_device, batch=batch), \
                        self.assertRaisesRegex(RuntimeError, "non-finite logits"):
                    self._run(poison, device, on_device, batch)


if __name__ == "__main__":
    unittest.main()
