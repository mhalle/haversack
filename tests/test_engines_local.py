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

    monkeypatch.setattr(fs, "segment", fake_segment)
    monkeypatch.setattr(fs, "_host_memory_gb", lambda: 16.0)
    stages = []

    class Rep:
        def __call__(self, p):
            stages.append((p.stage, p.detail))

    assert fs.run_local("t1.nii.gz", device="cpu", batch_size="auto", progress=Rep(),
                        grid="input", interp="linear") == "SEG"          # nnU-Net keys ignored
    assert seen["device"] == "cpu" and seen["batch_size"] == 8 and seen["viewagg_device"] == "cpu"
    assert stages and stages[0][0] == "predict" and "fastsurfer:brain" in stages[0][1]
