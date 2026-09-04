"""Jobs, progress and cancellation - the properties a UI actually depends on.

Driven by fake work rather than a real network: what is under test is that the caller's thread
stays free, that cancellation is honored promptly, and that runs on one device serialize.
"""
import threading
import time

import pytest

from haversack.errors import Cancelled
from haversack.job import Job, device_lock
from haversack.progress import CancelToken, Progress, Reporter


def _slow(n=20, sleep=0.005):
    """Fake work shaped like the sliding window: ticks a reporter per 'patch'."""
    def run(reporter):
        for i in range(n):
            reporter.tick(i + 1, n)
            time.sleep(sleep)
        return "finished"
    return run


# -- cancellation -----------------------------------------------------------------------
def test_cancel_token_raises_at_the_next_check():
    t = CancelToken()
    t.check()                                   # no-op while active
    t.cancel()
    assert t.cancelled
    with pytest.raises(Cancelled):
        t.check()


def test_a_reporter_tick_is_the_cancellation_point():
    t = CancelToken()
    r = Reporter(cancel=t)
    r.tick(1, 10)
    t.cancel()
    with pytest.raises(Cancelled):
        r.tick(2, 10)


def test_cancelling_a_job_stops_it_promptly_and_is_reported():
    job = Job(_slow(n=2000, sleep=0.001), device="cpu")
    while job.progress is None or job.progress.step < 2:
        time.sleep(0.005)
    job.cancel()
    assert job.wait(timeout=5), "job did not stop after cancel"
    assert job.cancelled and job.done
    with pytest.raises(Cancelled):
        job.result()
    assert job.progress.step < 2000             # stopped early, did not run to completion


def test_a_job_that_finishes_returns_its_value():
    job = Job(_slow(n=3), device="cpu")
    assert job.result(timeout=5) == "finished"
    assert job.done and not job.cancelled


def test_an_exception_is_re_raised_from_result_not_swallowed():
    def boom(reporter):
        raise ValueError("kaboom")
    job = Job(boom, device="cpu")
    with pytest.raises(ValueError, match="kaboom"):
        job.result(timeout=5)
    assert job.done


def test_result_times_out_rather_than_blocking_forever():
    job = Job(_slow(n=1000, sleep=0.01), device="cpu")
    with pytest.raises(TimeoutError):
        job.result(timeout=0.05)
    job.cancel()


# -- the point of the exercise: the caller's thread stays free ---------------------------
def test_the_caller_keeps_running_while_the_job_works():
    job = Job(_slow(n=40, sleep=0.005), device="cpu")
    spins = 0
    while not job.done:                          # a UI event loop would be doing this
        spins += 1
        time.sleep(0.001)
    assert spins > 5, "the caller was blocked instead of looping"
    assert job.result() == "finished"


def test_progress_is_pollable_without_a_callback():
    job = Job(_slow(n=30, sleep=0.003), device="cpu")
    seen = []
    while not job.done:
        p = job.progress
        if p is not None:
            seen.append(p.fraction)
        time.sleep(0.002)
    job.result()
    assert seen and max(seen) > min(seen)        # it moved


def test_a_progress_callback_that_raises_does_not_kill_the_run():
    def bad(_p):
        raise RuntimeError("a broken UI callback")
    job = Job(_slow(n=5), device="cpu", on_progress=bad)
    assert job.result(timeout=5) == "finished"


def test_done_callback_fires_and_is_immediate_if_already_finished():
    job = Job(_slow(n=2), device="cpu")
    job.result(timeout=5)
    fired = []
    job.add_done_callback(fired.append)          # already done: called at once
    assert fired == [job]


# -- the device is serial ----------------------------------------------------------------
def test_two_jobs_on_one_device_do_not_overlap():
    """TorchModel mutates its weights in place to load a fold and models are shared through the
    cache, so overlapping runs would corrupt each other - and the memory policy sizes the
    accumulator from free memory a concurrent run is about to take."""
    active = []
    overlapped = []

    def work(reporter):
        active.append(1)
        if len(active) > 1:
            overlapped.append(1)
        time.sleep(0.05)
        active.pop()
        return "ok"

    a = Job(work, device="cpu:test-serial")
    b = Job(work, device="cpu:test-serial")
    assert a.result(timeout=5) == "ok" and b.result(timeout=5) == "ok"
    assert not overlapped, "two jobs ran on the same device at once"


def test_a_queued_job_says_so():
    started = threading.Event()
    release = threading.Event()

    def hold(reporter):
        started.set()
        release.wait(timeout=5)
        return "ok"

    first = Job(hold, device="cpu:test-queued")
    started.wait(timeout=5)
    second = Job(_slow(n=1), device="cpu:test-queued")
    time.sleep(0.02)
    assert second.progress is not None and second.progress.stage == "queued"
    release.set()
    assert first.result(timeout=5) == "ok" and second.result(timeout=5) == "finished"


def test_different_devices_do_not_block_each_other():
    assert device_lock("cuda:0") is not device_lock("cuda:1")
    assert device_lock("cuda:0") is device_lock("cuda:0")


# -- Progress itself ---------------------------------------------------------------------
def test_progress_prints_readably_so_a_print_callback_still_works():
    p = Progress(stage="predict", detail="organs", part=1, n_parts=5, step=3, n_steps=10,
                 fraction=0.34)
    text = str(p)
    assert "34%" in text and "predict" in text and "[2/5]" in text and "3/10" in text


def test_fraction_advances_across_parts():
    r = Reporter(n_parts=2)
    r.enter_part(0, "a"); r.tick(10, 10)
    first = r.last.fraction
    r.enter_part(1, "b"); r.tick(10, 10)
    assert 0.0 < first < r.last.fraction <= 1.0


def test_stages_advance_the_fraction_without_patches():
    """A one-patch volume used to read 5 %, 5 %, 5 %, 95 %: every stage but restore
    started at the same number and only patch ticks moved it (2026-09-03). Now each
    stage starts where the previous one's work ends, and the last patch lands on restore."""
    from haversack.progress import Reporter
    seen = []
    r = Reporter(progress=lambda p: seen.append((p.stage, round(p.fraction, 2))))
    r.stage("read", "x"); r.enter_part(0, "m"); r.stage("preprocess"); r.stage("predict")
    r.tick(1, 1); r.stage("restore"); r.stage("finalize")
    fracs = [f for _, f in seen]
    assert fracs == sorted(fracs) and len(set(fracs)) == len(fracs) - 1   # only the last tick == restore
    assert dict(seen)["read"] == 0.0 and 0.05 < dict(seen)["loading"] < dict(seen)["preprocess"] < dict(seen)["predict"]
    assert seen[-3] == ("predict", 0.9) and seen[-2] == ("restore", 0.9) and seen[-1][1] > 0.9
    # a union of two parts: the second part's stages sit in the upper half
    r2 = Reporter(progress=lambda p: seen.append(p), n_parts=2)
    r2.enter_part(1, "second"); assert 0.5 < r2.last.fraction < 0.6


# -- InstallProgress: what a weights installer reports through ---------------------------------

def test_install_progress_with_a_message_callback_keeps_the_cli_contract():
    """Positions (item / download / finished) are reporter-only; messages still reach the
    callback, including through the legacy ``progress(msg)`` call."""
    from haversack.progress import InstallProgress
    lines = []
    n = InstallProgress.of(lines.append)
    n.begin(2); n.item(0, "Dataset8"); n.download(0, 100, "x"); n.download(100, 100, "x")
    n("downloading Dataset8 from a.zip"); n.unpack("unpacking Dataset8"); n.finished("done")
    assert lines == ["downloading Dataset8 from a.zip", "unpacking Dataset8"]
    assert InstallProgress.of(n) is n
    InstallProgress.of(None).say("silent")          # nothing to report to; must not raise


def test_install_progress_drives_a_reporter_by_parts_and_bytes():
    from haversack.progress import InstallProgress, Reporter
    seen = []
    n = InstallProgress.of(Reporter(progress=seen.append))
    n.begin(2)
    n.item(0, "Dataset298"); n.finished("Dataset298 present")
    n.item(1, "Dataset570")
    n.download(0, 1000, "downloading"); n.download(1000, 1000, "downloading")
    n.unpack("unpacking Dataset570"); n.finished("Dataset570 installed")
    assert {p.stage for p in seen} == {"weights"} and seen[0].n_parts == 2
    fr = [p.fraction for p in seen]
    assert fr == sorted(fr) and fr[-1] == 1.0
    assert seen[1].fraction == 0.5                   # first part complete
    mid = [p for p in seen if p.part == 1 and p.step == 1000 and p.detail == "downloading"][0]
    assert mid.n_steps == 1000 and 0.94 < mid.fraction < 0.96      # 0.5 + 0.9/2
    assert [p.detail for p in seen if p.part == 1][-2:] == ["unpacking Dataset570",
                                                             "Dataset570 installed"]


def test_install_progress_throttles_byte_snapshots():
    """A 1 MiB read loop must not be a snapshot per chunk: emit first, at completion, and
    every 2 % (at least 4 MiB) in between."""
    from haversack.progress import InstallProgress, Reporter
    seen = []
    n = InstallProgress.of(Reporter(progress=seen.append))
    total = 100 << 20
    for done in range(0, total + 1, 1 << 20):
        n.download(done, total)
    steps = [p.step >> 20 for p in seen]
    assert steps[0] == 0 and steps[-1] == 100 and len(steps) < 40, steps
    assert all(b - a >= 4 for a, b in zip(steps, steps[1:]))
    seen.clear()
    n.item(1); n.download(0, 0); n.download(7, 7)   # unknown size: still opens and closes
    assert [(p.step, p.n_steps) for p in seen[1:]] == [(0, 0), (7, 7)]


def test_reporter_advance_is_a_cancellation_point():
    from haversack.errors import Cancelled
    from haversack.progress import CancelToken, Reporter
    t = CancelToken(); r = Reporter(cancel=t)
    r.advance(0.5, step=1, n_steps=2)
    t.cancel()
    with pytest.raises(Cancelled):
        r.advance(0.6)
