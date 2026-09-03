"""Engines run in-process when their runtime is installed (2026-09-03).

FastSurfer-free: the engine is a fake `Engine.compute`, and availability is what
`find_spec` says. What is pinned: the runtime being installed is the local switch,
an explicit env flag still wins, the Segmenter routes an engine task to its compute
(and never to the nnU-Net pipeline), the refusal names the per-engine environment,
the CLI's stack check is per engine, and the view-aggregation placement reads the host.
"""
import dataclasses
import importlib.util

import pytest


class _Seg:
    provenance: dict = {}

from haversack.engines import registry
from haversack.errors import UnsupportedModel


def _spec_says(monkeypatch, present: bool):
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "FastSurferCNN":
            return object() if present else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_installed_runtime_enables_the_engine_locally(monkeypatch):
    monkeypatch.delenv("HAVERSACK_FASTSURFER", raising=False)
    _spec_says(monkeypatch, True)
    assert registry.available("fastsurfer") and registry.enabled("fastsurfer")
    monkeypatch.setenv("HAVERSACK_FASTSURFER", "0")
    assert not registry.enabled("fastsurfer")            # an explicit off wins


def test_no_runtime_and_no_flag_means_disabled(monkeypatch):
    monkeypatch.delenv("HAVERSACK_FASTSURFER", raising=False)
    _spec_says(monkeypatch, False)
    assert not registry.available("fastsurfer") and not registry.enabled("fastsurfer")
    monkeypatch.setenv("HAVERSACK_FASTSURFER", "1")       # the Modal deploy's switch still works
    assert registry.enabled("fastsurfer")


def test_segmenter_routes_an_engine_task_to_its_compute(monkeypatch, tmp_path):
    calls = {}

    def fake_compute(image, **kw):
        calls.update(kw, image=image)
        return "SEGMENTATION"

    monkeypatch.setitem(registry.ENGINES, "fastsurfer",
                        dataclasses.replace(registry.ENGINES["fastsurfer"], compute=fake_compute))
    _spec_says(monkeypatch, True)
    from haversack import pipeline
    from haversack.segmenter import Segmenter

    def never(*a, **k):
        raise AssertionError("an engine task must not reach the nnU-Net pipeline")

    monkeypatch.setattr(pipeline, "segment", never)
    seg = Segmenter(weights=tmp_path, device="cpu", batch_size=4)
    assert seg.segment("t1.nii.gz", "fastsurfer:brain", progress=print) == "SEGMENTATION"
    assert calls["image"] == "t1.nii.gz" and calls["device"] == "cpu" and calls["batch_size"] == 4
    assert calls["progress"] is print and calls["probabilities"] is None
    # the off-thread form takes the same route
    job = seg.submit("t1.nii.gz", "fastsurfer:brain")
    assert job.result() == "SEGMENTATION"


def test_segmenter_refuses_an_engine_task_without_its_runtime(monkeypatch, tmp_path):
    monkeypatch.delenv("HAVERSACK_FASTSURFER", raising=False)
    _spec_says(monkeypatch, False)
    from haversack.segmenter import Segmenter
    with pytest.raises(UnsupportedModel, match="uv sync --extra fastsurfer"):
        Segmenter(weights=tmp_path).segment("t1.nii.gz", "fastsurfer:brain")


def test_nnunet_tasks_still_take_the_pipeline(monkeypatch, tmp_path):
    from haversack import pipeline
    from haversack.segmenter import Segmenter
    seen = {}
    monkeypatch.setattr(pipeline, "segment", lambda image, task, **kw: seen.update(task=task) or "OK")
    assert Segmenter(weights=tmp_path).segment("ct.nii.gz", "ts:total_fast") == "OK"
    assert seen["task"] == "ts:total_fast"


def test_cli_stack_check_is_per_engine(monkeypatch, tmp_path, capsys):
    from haversack import cli
    _spec_says(monkeypatch, False)
    (tmp_path / "t1.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "t1.nii.gz"), "--task", "fastsurfer:brain",
                   "-o", str(tmp_path / "out.seg.nrrd")])
    err = capsys.readouterr().err
    assert rc == 2 and "fastsurfer engine" in err and "--extra fastsurfer" in err
    assert "lean install" not in err                      # the wrong remedy would mislead


def test_viewagg_placement_reads_the_host():
    from haversack.engines import fastsurfer as fs
    assert fs.local_viewagg("cuda") == "auto"             # FastSurfer's own 4 GB rule on the card
    assert fs.local_viewagg("cuda:1") == "auto"
    assert fs.local_viewagg("mps", host_gb=16) == "cpu"   # unified memory: keep the field off the GPU
    assert fs.local_viewagg("mps", host_gb=64) == "mps"
    assert fs.local_viewagg("cpu", host_gb=256) == "cpu"
    assert fs._host_memory_gb() > 1                       # readable on this host


def test_run_local_resolves_device_and_reports(monkeypatch):
    from haversack.engines import fastsurfer as fs
    seen = {}

    def fake_segment(image, **kw):
        seen.update(kw, image=image)
        return "SEG"

    monkeypatch.setattr(fs, "segment", lambda image, **kw: (fake_segment(image, **kw), _Seg())[1])
    monkeypatch.setattr(fs, "_host_memory_gb", lambda: 16.0)
    stages = []

    class Rep:
        def __call__(self, p):
            stages.append((p.stage, p.detail))

    assert isinstance(fs.run_local("t1.nii.gz", device="cpu", batch_size="auto", progress=Rep(),
                                   grid="input", interp="linear"), _Seg)  # nnU-Net keys ignored
    assert seen["device"] == "cpu" and seen["batch_size"] == 8 and seen["viewagg_device"] == "cpu"
    assert stages and stages[0][0] == "predict" and "fastsurfer:brain" in stages[0][1]


def test_synthstrip_has_a_local_runner_too(monkeypatch, tmp_path):
    """Same seam, second engine (2026-09-03): synthstrip:mask routes to Engine.compute."""
    monkeypatch.delenv("HAVERSACK_SYNTHSTRIP", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: object() if n == "synthstrip_torch" else real(n, *a, **k))
    assert registry.enabled("synthstrip")
    calls = {}
    monkeypatch.setitem(registry.ENGINES, "synthstrip",
                        dataclasses.replace(registry.ENGINES["synthstrip"],
                                            compute=lambda image, **kw: calls.update(kw, image=image) or "MASK"))
    from haversack.segmenter import Segmenter
    assert Segmenter(weights=tmp_path, device="cpu").segment("t1.nii.gz", "synthstrip:mask") == "MASK"
    assert calls["device"] == "cpu"


def test_synthstrip_run_local_resolves_device(monkeypatch):
    from haversack.engines import synthstrip as ss
    seen = {}
    monkeypatch.setattr(ss, "segment", lambda image, **kw: seen.update(kw, image=image) or _Seg())
    assert isinstance(ss.run_local("t1.nii.gz", device="cpu", grid="input"), _Seg)
    assert seen == {"image": "t1.nii.gz", "device": "cpu"}


def test_synthstrip_refuses_a_constant_field():
    """MPS + torch 2.14 returned an all-zero SDT for the full conform (2026-09-03); the
    engine must refuse rather than mask the whole image."""
    import numpy as np
    from haversack.engines import synthstrip as ss
    from haversack.errors import ResourceError
    with pytest.raises(ResourceError, match="constant field"):
        ss._refuse_constant_field(np.zeros((4, 4, 4), np.float32), "mps")
    ss._refuse_constant_field(np.linspace(-5, 5, 64, dtype=np.float32).reshape(4, 4, 4), "cpu")


def test_synthstrip_falls_back_to_fp16_on_a_real_mps_oom(monkeypatch):
    """With the allocator capped, an MPS shortfall raises instead of zeroing; the engine then
    retries with the net in fp16, and refuses (naming the CPU) if that is short too."""
    import types
    import torch
    from haversack.engines import synthstrip as ss
    from haversack.errors import ResourceError
    net = torch.nn.Conv3d(1, 1, 3, padding=1)
    monkeypatch.setattr(ss, "_get_model", lambda device: net)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None, raising=False)
    calls = []

    def predict_sdt(img, *, model, device):
        calls.append(type(model).__name__)
        if len(calls) == 1:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 10.01 GiB ...)")
        return "SDT", "GEOM"

    monkeypatch.setitem(__import__("sys").modules, "synthstrip_torch", types.SimpleNamespace(predict_sdt=predict_sdt))
    ss._HALF.clear()
    sdt, geom, run = ss._capture_sdt("img", "mps")
    assert (sdt, geom) == ("SDT", "GEOM") and calls == ["Conv3d", "_HalfNet"]
    assert run["precision"] == "fp16"
    (d,) = run["deviations"]
    assert d["what"] == "precision" and d["requested"] == "fp32" and d["effective"] == "fp16" and "out of memory" in d["why"]
    calls.clear()

    def always_oom(img, *, model, device):
        calls.append(1); raise RuntimeError("MPS backend out of memory")

    monkeypatch.setitem(__import__("sys").modules, "synthstrip_torch", types.SimpleNamespace(predict_sdt=always_oom))
    with pytest.raises(ResourceError, match="device='cpu'"):
        ss._capture_sdt("img", "mps")
    assert len(calls) == 2                                     # fp32, then fp16, then refuse

    def other_error(img, *, model, device):
        raise RuntimeError("something else entirely")

    monkeypatch.setitem(__import__("sys").modules, "synthstrip_torch", types.SimpleNamespace(predict_sdt=other_error))
    with pytest.raises(RuntimeError, match="something else"):  # not an OOM: not ours to catch
        ss._capture_sdt("img", "mps")


def test_resolving_mps_arms_the_allocator_cap(monkeypatch):
    """Past 1.0x the recommended working set Metal fails silently (zeros); PyTorch's default
    lets a process grow to 1.7x. Resolving an MPS device caps it at 1.0x, once."""
    import torch
    from haversack import resample
    seen = []
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", lambda f: seen.append(f), raising=False)
    monkeypatch.setattr(resample, "_MPS_CAP_ARMED", False)
    monkeypatch.setattr(resample, "_resolve_device_raw", lambda spec="auto": torch.device("mps"))
    resample.resolve_device("auto"); resample.resolve_device("mps")
    assert seen == [1.0]                                       # once per process
    monkeypatch.setattr(resample, "_MPS_CAP_ARMED", False)
    monkeypatch.setenv("HAVERSACK_MPS_MEMORY_FRACTION", "0")
    resample.resolve_device("mps")
    assert seen == [1.0]                                       # 0 = leave PyTorch's default


def test_synthstrip_half_net_keeps_the_fp32_interface():
    import torch
    from haversack.engines import synthstrip as ss
    net = torch.nn.Conv3d(1, 1, 3, padding=1)
    half = ss._HalfNet(net)
    x = torch.rand(1, 1, 8, 8, 8)
    y = half(x)
    assert y.dtype == torch.float32 and (y - net(x)).abs().max() < 1e-2
    assert next(net.parameters()).dtype == torch.float32          # the original is untouched


def test_a_deviation_reaches_the_cli_and_the_provenance(monkeypatch, tmp_path, capsys):
    """A user who asked for one thing and got another must be told: in the provenance
    (and so the seg.nrrd header and the server payload) and on the CLI's closing lines."""
    from haversack import cli, pipeline
    from haversack.result import deviation

    class R:
        timings = {}
        grid = type("G", (), {"shape": (1, 1, 1)})()
        schema = type("S", (), {"names": []})()
        provenance = {"deviations": [deviation("precision", "fp32", "fp16", "MPS out of memory")]}

        def save(self, path):
            return path

        def present(self):
            return {}

    (tmp_path / "in.nii.gz").touch()
    monkeypatch.setattr(pipeline, "segment", lambda image, task, **kw: R())
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast", "-o", str(tmp_path / "o.nii.gz")])
    err = capsys.readouterr().err
    assert rc == 0 and "note: precision: asked fp32, ran fp16 - MPS out of memory" in err


def test_fastsurfer_records_the_field_moving_to_cpu():
    from haversack.result import deviation
    d = deviation("device (view-aggregation field)", "mps", "cpu", "would not fit")
    assert d == {"what": "device (view-aggregation field)", "requested": "mps", "effective": "cpu", "why": "would not fit"}
