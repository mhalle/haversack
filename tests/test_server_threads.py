"""The autouse fixture that ends a test's server threads with the test (conftest.py,
2026-10-01) - held to its promise here, since a fixture that silently did nothing would
pass every other test in the suite.

Before it, a "done" job's daemon artifact thread published into the shared store during a
LATER test, and one broke TestPush's publication count in CI (run 36786031441).
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")

from conftest import _SERVER_THREADS, server_threads_end_here  # noqa: E402

#: Long enough that a thread left running is plainly still running when the block exits.
SLOW_S = 0.3


def _server_threads(before):
    return sorted(t.name for t in threading.enumerate()
                  if t not in before and t.name.startswith(_SERVER_THREADS) and t.is_alive())


def test_a_job_running_at_the_end_publishes_before_the_block_is_left(tmp_path):
    """The hard case: the job is still RUNNING when the body ends. Closing lets it finish,
    and the dispatcher then starts an artifact thread the first look did not see - so the
    block is left only after that thread has placed its artifacts too."""
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_job_result_cache import _Segmenter
    from test_serve import submit, wait_state

    before = set(threading.enumerate())
    gate, placed = threading.Event(), []
    with server_threads_end_here(join_s=10.0):
        ex = LocalExecutor(_Segmenter(gate=gate, steps=1), workdir=tmp_path / "w",
                           cache_dir=tmp_path / "c")
        real = ex.cache.add_artifact

        def slow_add_artifact(*a, **kw):       # this executor's cache only, not the class
            time.sleep(SLOW_S)
            got = real(*a, **kw)
            placed.append(a[1])
            return got
        ex.cache.add_artifact = slow_add_artifact
        client = TestClient(create_app(ex))
        wait_state(client, submit(client), ("running",))
        threading.Timer(SLOW_S, gate.set).start()   # the job ends after the body does
    assert _server_threads(before) == [], "a server thread outlived the block"
    assert placed, "the job's artifacts were placed before the block was left"
    assert ex._stop, "the executor the block started was closed"
