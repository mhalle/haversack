"""Shared fixtures for the kernel-layer tests: the device matrix, synthetic logits, and the
tie-aware comparison. Auto-loaded by pytest.
"""
import threading
import time

import numpy as np
import pytest
import torch
from scipy.ndimage import uniform_filter

from haversack import reference


def device_names():
    names = ["cpu"]
    if torch.backends.mps.is_available():
        names.append("mps")
    if torch.cuda.is_available():
        names.append("cuda")
    return names


@pytest.fixture(params=device_names())
def device(request):
    return torch.device(request.param)


def voronoi_logits(K=9, shape=(14, 18, 22), n_regions=20, seed=0, noise=0.3, smooth=True):
    """Small anatomically-shaped synthetic logits: Voronoi blobs, +4 inside / -4
    outside per class, box-smoothed so boundaries are soft, plus noise."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    seeds = rng.uniform(0, 1, size=(n_regions, 3)) * np.array([z, y, x])
    seed_label = rng.integers(1, K, n_regions)
    seed_label[: max(1, n_regions // 3)] = 0
    gz, gy, gx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    pts = np.stack([gz, gy, gx], -1).reshape(-1, 3).astype(np.float64)
    d = ((pts[:, None, :] - seeds[None]) ** 2).sum(-1)
    lab = seed_label[d.argmin(1)].reshape(shape)
    logits = np.where(lab[None] == np.arange(K)[:, None, None, None], 4.0, -4.0)
    if smooth:
        logits = uniform_filter(logits, size=(1, 3, 3, 3), mode="nearest")
        logits = uniform_filter(logits, size=(1, 3, 3, 3), mode="nearest")
    if noise:
        logits = logits + rng.normal(0, noise, logits.shape)
    return logits.astype(np.float32)


def assert_agree_up_to_ties(got, want, values, *, tol=1e-4, max_fraction=2e-3, what=""):
    """Backends compute in float32 and may round differently; every
    disagreement must be a genuine near-tie of the float64 reference."""
    got = np.asarray(got).astype(np.int64)
    want = np.asarray(want).astype(np.int64)
    assert got.shape == want.shape, (got.shape, want.shape)
    mism = got != want
    n = int(mism.sum())
    if n == 0:
        return 0
    m = reference.margins(values)
    bad = mism & (m > tol)
    assert not bad.any(), (f"{what}: {int(bad.sum())} of {n} mismatches are not ties "
                           f"(largest margin {m[mism].max():.3g})")
    assert mism.mean() <= max_fraction, f"{what}: {n} tie mismatches = {mism.mean():.2e} > {max_fraction}"
    return n


@pytest.fixture(autouse=True)
def _engines_off_by_default(monkeypatch):
    """Pin every engine's enable flag OFF unless a test opts in (2026-09-03).

    Engines now enable on an installed runtime when their flag is unset, so a dev environment
    with `--extra fastsurfer` synced would silently flip the default ecosystem/registry/version
    that many tests assert on. Setting each flag to "0" gives every test a deterministic
    nnU-Net-only baseline; a test that wants an engine sets its flag (or, for the local-runner
    tests, deletes it to exercise the runtime-based path)."""
    from haversack.engines.registry import engine_env_vars
    for var in engine_env_vars():
        monkeypatch.setenv(var, "0")


#: The server's threads that do finite work, or stop when their executor is closed. Not
#: "haversack-view" (serve_forever until shut down) nor a job.py worker, whose fn may block.
_SERVER_THREADS = ("haversack-serve", "haversack-artifacts", "haversack-prefetch",
                   "haversack-sweep", "haversack-mirror-trim")
#: How long a test's server threads get to finish after it. They finish in milliseconds
#: here; the bound only turns a hang into a teardown error instead of a stuck suite.
_SERVER_THREADS_JOIN_S = 30.0


@pytest.fixture(autouse=True)
def _server_threads_end_with_their_test():
    """Close every LocalExecutor a test started and wait for the server threads it started
    (2026-10-01).

    `LocalExecutor.close()` only sets a stop flag, and most tests never call it, so their
    threads outlived them: a "done" job's daemon artifact thread published into the shared
    store (`SharedResultCache._swap`) during a LATER test - eight such publications per
    fast-suite run, in TestGapsFromMutation, TestBoundedHistory, TestFormat1IsReadAndConverted.
    One broke TestPush's publication count in CI (run 36786031441), and any class-wide patch
    of the store could see another test's write. Closing here is the existing shutdown, not
    a new one: a running job still finishes and publishes - before this test ends, not after.

    The executor is reached through its dispatcher thread's target (CPython's
    `Thread._target`, there until the target returns), not by wrapping `__init__`: a patched
    constructor is itself a class-wide change, and `test_executor_contract` reads the real
    one's code.
    """
    before = set(threading.enumerate())
    yield
    for t in threading.enumerate():
        if t not in before and t.name == "haversack-serve":
            close = getattr(getattr(getattr(t, "_target", None), "__self__", None), "close", None)
            if close is not None:
                close()
    deadline = time.monotonic() + _SERVER_THREADS_JOIN_S
    while True:
        # again after each round: a dispatcher finishing its job starts an artifact thread
        left = [t for t in threading.enumerate() if t not in before
                and t.name.startswith(_SERVER_THREADS) and t.is_alive()]
        if not left:
            return
        for t in left:
            t.join(max(0.0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            still = sorted(t.name for t in left if t.is_alive())
            if still:
                pytest.fail(f"server threads outlived their test by {_SERVER_THREADS_JOIN_S:.0f}"
                            f" s: {', '.join(still)}", pytrace=False)
            return
