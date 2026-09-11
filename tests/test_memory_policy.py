"""The accumulator-placement policy: a runtime decision from the device's budget."""
import pytest
import torch

pytest.importorskip("haversack")
from haversack.network import ACCUMULATE, choose_accumulate, device_budget_bytes

CPU = torch.device("cpu")
MPS = torch.device("mps")
CHEST_3MM = (236, 167, 167)          # padded model grid of a chest at 3 mm


def test_forced_modes():
    on, why = choose_accumulate("host", device=MPS, K=118, shape=CHEST_3MM)
    assert on is False and "forced host" in why
    on, _ = choose_accumulate("device", device=MPS, K=118, shape=CHEST_3MM)
    assert on is True
    with pytest.raises(ValueError):
        choose_accumulate("gpu", device=MPS, K=118, shape=CHEST_3MM)


def test_cpu_is_always_host():
    on, why = choose_accumulate("auto", device=CPU, K=118, shape=CHEST_3MM)
    assert on is False and "cpu" in why


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_auto_scales_with_the_budget(monkeypatch):
    """Same volume, different budgets: the policy is about the machine, not the model.
    Both host readings are pinned so the test measures the policy and not the moment it ran.
    Pinning only the available bytes was not enough: the pressure gate in front of the budget
    read the live kern.memorystatus_level, so on 2026-09-10 this failed at 34 % (the gate
    refuses below 35) and passed minutes later at 37 %."""
    import haversack.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # roomy host: the Metal ceiling binds
    monkeypatch.setattr(N, "host_memory_health", lambda: 80)               # healthy host: the budget decides, not the gate
    budget = N.device_budget_bytes(MPS)
    assert budget > 4e9
    # K=118 chest at 3 mm: 1.57 GB accumulator. Whether it fits depends on what else must fit.
    big, why_big = N.choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM, activation_reserve_gb=0.5)
    small, why_small = N.choose_accumulate("auto", device=MPS, K=118, shape=CHEST_3MM,
                                           activation_reserve_gb=budget / 1e9 + 1.0)
    assert big is True and small is False, (why_big, why_small)
    # a tiny volume fits anywhere
    on, _ = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), activation_reserve_gb=0.5)
    assert on is True
    # ... and a busy host takes the option away, whatever the model
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(3.2e9))
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), activation_reserve_gb=0.5)
    assert on is False, why


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_measured_mode_uses_what_the_network_actually_holds(monkeypatch):
    """After the first patch the network is resident, so the budget already reflects it and only
    a transient margin is reserved - instead of a constant that is 2 GB wrong for some models.
    The pressure level is pinned too, as in test_auto_scales_with_the_budget: left live, this
    failed on 2026-09-10 at 34 %, the gate refusing before either branch under test ran."""
    import haversack.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # take the Metal ceiling
    monkeypatch.setattr(N, "host_memory_health", lambda: 80)               # healthy host: the budget decides, not the gate
    monkeypatch.setattr(N, "device_working_set_bytes", lambda device: int(2.5e9))
    budget = N.device_budget_bytes(MPS)
    on, why = N.choose_accumulate("auto", device=MPS, K=25, shape=(480, 340, 340), measured=True)
    assert "measured" in why and "network holds 2.50 GB" in why
    # the accumulator is (K + 1) x voxels x 2 B = 2.89 GB - the weight map is a channel too, which
    # this missed until 2026-09-11 (2.78 GB) - plus a 0.62 GB margin, against the real budget
    assert "accumulator 2.89 GB" in why, why
    assert on is ((25 + 1) * 480 * 340 * 340 * 2 + 0.625e9 <= budget), why
    # the unmeasured path would have demanded a further 4.5 GB on top
    on_est, why_est = N.choose_accumulate("auto", device=MPS, K=25, shape=(480, 340, 340))
    assert "unmeasured" in why_est
    assert not (on_est and not on)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_budget_honors_the_watermark_env(monkeypatch):
    """The watermark caps the Metal side of the budget. The host figure is pinned high so that
    side binds - otherwise on a busy machine both values are host-limited and drift between
    calls, which says nothing about the watermark."""
    import haversack.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))
    full = N.device_budget_bytes(MPS)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.05")
    capped = N.device_budget_bytes(MPS)
    ceiling = torch.mps.recommended_max_memory()
    assert capped <= 0.05 * ceiling
    assert capped < full


def test_host_available_is_reported_here():
    from haversack.network import host_available_bytes
    host = host_available_bytes()
    assert host is not None and 0 < host < 10_000e9


def test_host_memory_health_is_reported_here():
    """The gate is only as good as this reading, and it fails open: any error comes back as None,
    which choose_accumulate takes as "no veto". So a sysctl that stopped answering would remove
    the gate without a word - probed 2026-09-11 with a failing sysctl first on PATH: None, no
    warning, and nothing about the gate in the policy's why. On macOS it must be a percentage."""
    import platform
    from haversack.network import host_memory_health
    level = host_memory_health()
    if platform.system() == "Darwin":
        assert isinstance(level, int) and 0 <= level <= 100, level
    else:
        assert level is None, level


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_budget_is_host_limited_not_the_metal_ceiling(monkeypatch):
    """Unified memory: the Metal ceiling is hardware, not availability. Taking it drives the
    machine into swap (measured 2026-08-22: a 9 GB budget on a 16 GB M2 Air pushed
    kern.memorystatus_level to 10 % and tripped the bench guard)."""
    import haversack.network as N
    ceiling = torch.mps.recommended_max_memory()
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(6e9))       # a busy machine
    # only a fraction of "available" is durably ours on unified memory (taking it causes swapping)
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(1.5e9, rel=1e-6)
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0, unified_fraction=1.0) == pytest.approx(3e9, rel=1e-6)
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))     # a big workstation
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == pytest.approx(
        ceiling - torch.mps.driver_allocated_memory(), rel=1e-6)           # then the Metal ceiling binds
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(2e9))
    assert N.device_budget_bytes(MPS, host_headroom_gb=3.0) == 0           # nothing to spare -> host


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_tight_host_declines_the_device_whatever_the_budget_says(monkeypatch):
    """A snapshot of free memory is optimistic on unified memory: taking the budget is what
    destroys it. Measured 2026-08-22 - 6.9 GB looked available, the policy took a 1.6 GB
    accumulator on device, and the machine then swapped at 4.9 GB/min."""
    import haversack.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))
    monkeypatch.setattr(N, "device_working_set_bytes", lambda device: int(1e9))
    monkeypatch.setattr(N, "host_memory_health", lambda: 20)               # kernel says memory is tight
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is False and "already tight" in why
    monkeypatch.setattr(N, "host_memory_health", lambda: 70)               # healthy host
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is True, why


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_the_pressure_gate_refuses_at_34_and_admits_at_35(monkeypatch):
    """The gate's edge, pinned where it was met on 2026-09-10. The tight-host test's 20 and 70
    would pass a `<=`, or any threshold from 21 to 70; this holds it at 35. Refusing at 34 is
    the policy working - that day's failures were tests reading the machine, not the gate."""
    import haversack.network as N
    monkeypatch.setattr(N, "host_available_bytes", lambda: int(400e9))
    monkeypatch.setattr(N, "device_working_set_bytes", lambda device: int(1e9))
    monkeypatch.setattr(N, "host_memory_health", lambda: 34)
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is False and "already tight" in why, why
    monkeypatch.setattr(N, "host_memory_health", lambda: 35)
    on, why = N.choose_accumulate("auto", device=MPS, K=4, shape=(32, 32, 32), measured=True)
    assert on is True, why


def test_accumulate_names():
    assert ACCUMULATE == ("auto", "device", "host")


def test_choose_batch_policy():
    from haversack.network import CUDA_AUTO_BATCH, choose_batch
    cuda = torch.device("cuda")
    assert choose_batch(3, device=cuda, on_device=True, held_bytes=1, budget_bytes=1, accumulator_bytes=1)[0] == 3
    assert choose_batch("auto", device=MPS, on_device=True, held_bytes=int(1e9), budget_bytes=int(20e9), accumulator_bytes=int(2e9))[0] == 1
    assert choose_batch("auto", device=cuda, on_device=False, held_bytes=int(1e9), budget_bytes=int(20e9), accumulator_bytes=int(2e9))[0] == 1
    # an A10 whole-body part: 1.5 GB working set, 2.9 GB accumulator, ~20 GB free -> 4
    b, why = choose_batch("auto", device=cuda, on_device=True, held_bytes=int(1.5e9), budget_bytes=int(20e9), accumulator_bytes=int(2.9e9))
    assert b == CUDA_AUTO_BATCH and "fits" in why
    # a small card: 1.5 GB working set, 2.9 GB accumulator, 3 GB free -> stay at 1
    b, why = choose_batch("auto", device=cuda, on_device=True, held_bytes=int(1.5e9), budget_bytes=int(3e9), accumulator_bytes=int(2.9e9))
    assert b == 1 and "would not fit" in why


def test_a_budget_query_for_an_absent_backend_returns_none_rather_than_raising(monkeypatch):
    """CI is Linux with no MPS. A budget query must answer "unknown", not raise - naming a
    device the host does not have is a legitimate question with a legitimate answer."""
    from haversack.network import choose_accumulate, device_budget_bytes
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device_budget_bytes(torch.device("mps")) is None
    on, why = choose_accumulate("device", device=torch.device("mps"), K=118, shape=CHEST_3MM)
    assert on is True and "forced device" in why            # forced still works without a budget
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_budget_bytes(torch.device("cuda")) is None


def test_host_available_works_on_linux_without_psutil(monkeypatch):
    """The CUDA path runs on Linux, where there is no vm_stat - /proc/meminfo is the fallback."""
    import builtins
    import haversack.network as N
    real_import = builtins.__import__

    def no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    got = N.host_available_bytes()
    import platform
    if platform.system() == "Linux":
        assert got is not None and got > 0
