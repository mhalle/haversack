"""The sliding window must let go of the first patch's output once it has been accumulated.

``TorchModel._sliding_window`` runs the first patch before either memory policy has decided
anything: the accumulator's placement and the batch are chosen from what the device holds with
that patch's output alive. It then hands the output to ``_accumulate``, which adds it and
``del``s it. Until 2026-09-11 the ``del`` freed nothing - the caller's frame still bound the
tensor, and so did the ``args`` tuple of the ``@torch.inference_mode()`` wrapper around
``_accumulate`` - so the first patch's output stayed allocated on the device for the whole
accumulation loop. It is a view of the network's whole output, K x patch x dtype bytes: about
510 MB for an fp32 K=18 model at a 192^3 patch.

Which references are alive is Python's business, not the device's, so these run on the CPU: a
stand-in network records, on every forward pass, whether a weak reference to the first patch's
output is still alive.
"""
import contextlib
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("acvl_utils")        # predict_logits pads with nnU-Net's own helper
import haversack.network as N  # noqa: E402

K = 3
PATCH = (8, 8, 8)
SHAPE = (24, 8, 8)                        # five patches along z at nnU-Net's half-patch step


def _slicers(shape):
    return [(slice(None), slice(z, z + PATCH[0]), slice(0, PATCH[1]), slice(0, PATCH[2]))
            for z in range(0, shape[0] - PATCH[0] + 1, PATCH[0] // 2)]


class _Net:
    """Stands in for the network. It is a pointwise function of its input, so every patch
    agrees on every voxel and the accumulated logits are known exactly: ``x + k`` in channel
    ``k``. On every call it records whether the first patch's output is still alive."""

    def __init__(self):
        self.calls, self.first, self.alive = 0, None, {}

    def first_alive(self):
        return self.first is not None and self.first() is not None

    def __call__(self, x):
        self.calls += 1
        self.alive[self.calls] = self.first_alive()
        return x.expand(-1, K, -1, -1, -1) + torch.arange(K, dtype=x.dtype).view(1, K, 1, 1, 1)


def _model(net, *, batch_size):
    """A TorchModel with no weights behind it: only what predict_logits and the sliding window
    read, on the CPU. Its ``_patch`` hands the net a weak reference to the first tensor it
    returns - that tensor, not the net's own output, because a view taken under inference mode
    keeps no reference to its base: the net's output is garbage as soon as ``_patch`` has indexed
    it, whatever the sliding window still holds, so a reference to it would prove nothing."""
    m = N.TorchModel.__new__(N.TorchModel)
    m.device, m.dtype, m.K, m.patch = torch.device("cpu"), torch.float32, K, PATCH
    m.transpose_forward = m.transpose_backward = (0, 1, 2)
    m.accumulate, m.activation_reserve_gb, m.batch_size = "auto", N.DEFAULT_ACTIVATION_RESERVE_GB, batch_size
    m.accumulate_choice = m.batch_choice = None
    m.fold_params, m._load_fold = [None], lambda i: None
    m.predictor = SimpleNamespace(_internal_get_sliding_window_slicers=_slicers)
    m.net, m._gaussian_cpu = net, torch.ones(PATCH)
    m.gaussian, m._on_device = m._gaussian_cpu, True
    patch = m._patch

    def watched(padded, sl):
        out = patch(padded, sl)
        if net.first is None:
            net.first = weakref.ref(out)
        return out

    m._patch = watched
    return m


def _copying_to():
    """Off a device, ``.to("cpu")`` is a copy. On the CPU it returns the tensor itself, which the
    host path's add worker then holds until the next patch reaches it; this makes it a copy, as a
    transfer off a device is, so that what the worker holds is not the tensor under test."""
    to = torch.Tensor.to

    def copying(self, *a, **k):
        out = to(self, *a, **k)
        return out.clone() if out is self else out

    return mock.patch.object(torch.Tensor, "to", copying)


def _run(m, x, *, on_device):
    """``predict_logits``, with each memory measurement and the placement policy recording
    whether the first patch's output was alive when it ran."""
    measured = {}

    def watch(name, fn):
        def watched(*a, **k):
            measured.setdefault(name, []).append(m.net.first_alive())
            return fn(*a, **k)
        return watched

    # the CPU policy always answers "host"; on the device path this stands in for one that fits
    policy = (lambda *a, **k: (True, "test: the accumulator fits on the device")) if on_device else N.choose_accumulate
    with contextlib.ExitStack() as stack:
        for name, fn in (("device_working_set_bytes", N.device_working_set_bytes),
                         ("device_budget_bytes", N.device_budget_bytes), ("choose_accumulate", policy)):
            stack.enter_context(mock.patch.object(N, name, watch(name, fn)))
        if not on_device:
            stack.enter_context(_copying_to())
        out = m.predict_logits(x)
    return out, measured


def _volume():
    return torch.randint(0, 8, (1, *SHAPE), generator=torch.Generator().manual_seed(0)).float()


def _expected(x):
    return torch.stack([x[0] + k for k in range(K)])


class TheFirstPatchIsFreedOnceAccumulated(unittest.TestCase):

    def _check(self, *, on_device, batch_size):
        net = _Net()
        m = _model(net, batch_size=batch_size)
        x = _volume()
        out, measured = _run(m, x, on_device=on_device)
        self.assertEqual(m.accumulate_choice["on_device"], on_device, m.accumulate_choice)
        self.assertEqual(m.batch_choice["batch"], batch_size, m.batch_choice)
        self.assertIsNotNone(net.first, "the first patch never went through _patch; the test proves nothing")
        # The order the policies depend on: they decide from what the device holds with the first
        # patch's output alive, so freeing it earlier would change their figures. It also shows
        # the weak reference does see the tensor alive while something holds it.
        self.assertIn("device_working_set_bytes", measured)
        for name, alive in measured.items():
            self.assertTrue(all(alive), f"{name} ran without the first patch's output alive")
        self.assertFalse(net.alive[2], "the second forward pass ran beside the first patch's output")
        torch.testing.assert_close(out.float(), _expected(x))

    def test_on_the_device_at_batch_1(self):
        self._check(on_device=True, batch_size=1)

    def test_on_the_device_at_batch_4(self):
        """The batched loop's second forward pass is its first batch of four, which goes straight
        through the network rather than through ``_patch``."""
        self._check(on_device=True, batch_size=4)

    def test_on_the_host(self):
        """The host path copies the first patch off the device for the add worker; the device
        tensor must go once the copy is made."""
        self._check(on_device=False, batch_size=1)


if __name__ == "__main__":
    unittest.main()
