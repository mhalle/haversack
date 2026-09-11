"""The sliding window must let go of each patch's output before the next forward pass runs.

``TorchModel._accumulate`` adds each patch's output into the accumulator and goes on to the
next. Until 2026-09-11 the output stayed bound while the next was computed: on the batch-1
device loop and the host loop the name ``pred`` was rebound only when the next ``_patch``
returned, and the batched loop left ``pred`` on the batch's last row, a view that keeps the
whole batch's storage after ``del preds``. So every forward pass after the second ran beside the
previous one's output - K x patch x dtype bytes, about 510 MB for an fp32 K=18 model at a 192^3
patch, B times that at batch B - while both memory policies measure the device after the FIRST
forward pass, with one output resident.

Which references are alive is Python's business, not the device's, so these run on the CPU with
a stand-in network and two detectors. One is a weak reference to every tensor ``_patch``
returns. The other is a weak reference to the numpy array behind every output the network
allocates, which dies only with the storage: a view outlives its tensor object, and on the
batched path no ``_patch`` is called and what held on was a view. The first patch's output is
``_sliding_window``'s to hand over, a separate fix, and is not checked here.
"""
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

pytest.importorskip("acvl_utils")        # predict_logits pads with nnU-Net's own helper
import haversack.network as N  # noqa: E402

K = 3
PATCH = (8, 8, 8)
SHAPE = (40, 8, 8)                        # nine patches along z at nnU-Net's half-patch step: two batches of 4


def _slicers(shape):
    return [(slice(None), slice(z, z + PATCH[0]), slice(0, PATCH[1]), slice(0, PATCH[2]))
            for z in range(0, shape[0] - PATCH[0] + 1, PATCH[0] // 2)]


def _alive(refs):
    return [k for k, ref in refs if ref() is not None]


class _Net:
    """Stands in for the network. It is a pointwise function of its input, so every patch agrees
    on every voxel and the accumulated logits are known exactly: ``x + k`` in channel ``k``. Its
    output's storage is a numpy array's - ``torch.from_numpy`` keeps the array referenced until
    the storage is freed - so a weak reference to the array says whether the memory is still
    held, whatever views of it are left. On every call it records which earlier calls' outputs
    are alive, by storage (``held``) and by the tensor ``_patch`` returned (``bound``)."""

    def __init__(self):
        self.calls, self.storages, self.returned = 0, [], []    # (the call that made it, weak reference)
        self.held, self.bound = {}, {}                          # call -> earlier calls still alive

    def __call__(self, x):
        self.calls += 1
        self.held[self.calls], self.bound[self.calls] = _alive(self.storages), _alive(self.returned)
        arr = np.empty((x.shape[0], K, *x.shape[2:]), dtype=np.float32)
        self.storages.append((self.calls, weakref.ref(arr)))
        out = torch.from_numpy(arr)
        out.copy_(x.expand(-1, K, -1, -1, -1) + torch.arange(K, dtype=x.dtype).view(1, K, 1, 1, 1))
        return out


def _model(net, *, batch_size, hold=None):
    """A TorchModel with no weights behind it: only what predict_logits and the sliding window
    read, on the CPU. Its ``_patch`` gives the net a weak reference to every tensor it returns -
    that tensor, not the net's own output, because a view taken under inference mode keeps no
    reference to its base: the net's output object is garbage as soon as ``_patch`` has indexed
    it. ``hold``, if given, is called with each of them: the controls keep things alive with it."""
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
        net.returned.append((net.calls, weakref.ref(out)))
        if hold is not None:
            hold(out)
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
    # the CPU policy always answers "host"; on the device path this stands in for one that fits
    policy = (lambda *a, **k: (True, "test: the accumulator fits on the device")) if on_device else N.choose_accumulate
    with mock.patch.object(N, "choose_accumulate", policy):
        if on_device:
            return m.predict_logits(x)
        with _copying_to():
            return m.predict_logits(x)


def _volume():
    return torch.randint(0, 8, (1, *SHAPE), generator=torch.Generator().manual_seed(0)).float()


def _expected(x):
    return torch.stack([x[0] + k for k in range(K)])


def _beside(alive):
    """forward pass -> the outputs of earlier passes still alive while it ran, the first patch's
    excepted (see the module docstring)"""
    return {j: [k for k in ks if k != 1] for j, ks in alive.items()}


class EachPatchIsFreedBeforeTheNextForwardPass(unittest.TestCase):

    def _check(self, *, on_device, batch_size):
        net = _Net()
        m = _model(net, batch_size=batch_size)
        x = _volume()
        out = _run(m, x, on_device=on_device)
        self.assertEqual(m.accumulate_choice["on_device"], on_device, m.accumulate_choice)
        self.assertEqual(m.batch_choice["batch"], batch_size, m.batch_choice)
        rest = len(_slicers(SHAPE)) - 1
        self.assertEqual(net.calls, 1 + -(-rest // batch_size), "the first patch, then the rest in batches")
        self.assertGreaterEqual(net.calls, 3, "no forward pass after the second; the test proves nothing")
        self.assertEqual([k for k, _ in net.storages], list(range(1, net.calls + 1)), "an output went unwatched")
        nothing = {j: [] for j in net.held}
        self.assertEqual(_beside(net.held), nothing, "forward pass -> earlier outputs whose storage it ran beside")
        torch.testing.assert_close(out.float(), _expected(x))
        return net

    def _check_patch_outputs(self, net):
        self.assertEqual([k for k, _ in net.returned], list(range(1, net.calls + 1)),
                         "a forward pass did not go through _patch; the test proves nothing")
        self.assertEqual(_beside(net.bound), {j: [] for j in net.bound},
                         "forward pass -> earlier _patch outputs it ran beside")

    def test_on_the_device_at_batch_1(self):
        self._check_patch_outputs(self._check(on_device=True, batch_size=1))

    def test_on_the_device_at_batch_4(self):
        """No ``_patch`` here: each batch goes straight through the network, and the tensor that
        outlived ``del preds`` was a row of it, so only the storage says whether the batch was
        freed."""
        self._check(on_device=True, batch_size=4)

    def test_on_the_host(self):
        """The host path copies each output off the device for the add worker; the device tensor
        must go once the copy is enqueued."""
        self._check_patch_outputs(self._check(on_device=False, batch_size=1))


class TheDetectorsSeeWhatIsHeld(unittest.TestCase):
    """Controls: a check that nothing is alive passes just as well when the detector is blind."""

    def test_an_output_kept_whole(self):
        net, kept = _Net(), []
        _run(_model(net, batch_size=1, hold=kept.append), _volume(), on_device=True)
        for j in range(2, net.calls + 1):
            self.assertEqual(net.held[j], list(range(1, j)), j)
            self.assertEqual(net.bound[j], list(range(1, j)), j)

    def test_a_row_kept_after_its_output_is_gone(self):
        """What the batched loop did: a view outlives the tensor it was taken from and keeps the
        whole storage. The weak reference to the tensor calls it dead; the storage's does not."""
        net, kept = _Net(), []
        _run(_model(net, batch_size=1, hold=lambda out: kept.append(out[:1])), _volume(), on_device=True)
        last = net.calls
        self.assertEqual(net.held[last], list(range(1, last)))
        self.assertNotIn(2, net.bound[last])


if __name__ == "__main__":
    unittest.main()
