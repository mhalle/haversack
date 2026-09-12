"""`envelope_mm=0` runs the whole volume, at every door.

Until 2026-09-11 one number meant two things: `--envelope 0` on the command line ran the whole
volume, while `segment(envelope_mm=0)`, `Segmenter(envelope_mm=0)` and the server's
`{"envelope_mm": 0}` cropped inference flush to the skin. A CADS parity study passed 0 believing
it meant "no envelope" and measured mean Dice 0.64-0.82 on face, head muscles and mammary
glands where the whole volume gives 0.95-1.0. `envelope.envelope_margin` is now the one reading,
applied by `segment()`; each test here drives one door into the REAL `segment()` with stub
models and checks the grid the network was handed, against a run that crops (so a door that
drops the option altogether fails too) and a run with no envelope at all.
"""
from __future__ import annotations

import itertools
import time

import pytest

pytest.importorskip("nnunetv2")
pytest.importorskip("SimpleITK")
from test_normalization_sharing import ORGANS, _StubModel, _two_part_task, _write_ct  # noqa: E402

from haversack import pipeline  # noqa: E402
from haversack.errors import InputError  # noqa: E402


def test_zero_and_below_are_the_whole_volume_and_a_margin_is_a_margin():
    from haversack.envelope import envelope_margin
    assert envelope_margin(None) is None
    assert envelope_margin(0) is None and envelope_margin(0.0) is None
    assert envelope_margin(-5) is None
    assert envelope_margin(20) == 20.0 and envelope_margin("7.5") == 7.5
    for bad in (float("nan"), float("inf")):
        with pytest.raises(InputError):
            envelope_margin(bad)


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Every door ends in `pipeline.segment`. This keeps the real one and swaps only the task,
    the models and the image, forwarding `envelope_mm` exactly as the door handed it over (and
    not at all when the door passed none, so segment's own default applies)."""
    real = pipeline.segment
    ct = _write_ct(tmp_path)
    seen = []
    runs = itertools.count()

    def through_stubs(image, task, **kw):
        first, second = _StubModel(ORGANS._props), _StubModel(ORGANS._props)
        d = tmp_path / f"run{next(runs)}"
        d.mkdir()
        spec, store, cache = _two_part_task(d, [first, second])
        monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
        forward = {"envelope_mm": kw["envelope_mm"]} if "envelope_mm" in kw else {}
        r = real(str(ct), spec, models=cache, device="cpu", convention="corner", folds=(0,),
                 **forward)
        seen.append({"grid": tuple(first.received.shape[1:]),
                     "recorded": r.provenance["envelope_mm"]})
        return r

    monkeypatch.setattr(pipeline, "segment", through_stubs)
    through_stubs(ct, None, envelope_mm=None)
    whole = seen.pop()["grid"]
    through_stubs(ct, None, envelope_mm=3.0)
    cropped = seen.pop()["grid"]
    assert cropped != whole, "the fixture's body must crop at 3 mm, or no test here can fail"
    return {"seen": seen, "ct": ct, "whole": whole, "cropped": cropped}


def _check(stubbed, zero, three):
    assert zero["grid"] == stubbed["whole"], "0 cropped instead of running the whole volume"
    assert zero["recorded"] is None, "provenance must record what ran: no envelope"
    assert three["grid"] == stubbed["cropped"], "the door dropped the option"


def test_the_python_api(stubbed):
    for mm in (0, 3.0):
        pipeline.segment(str(stubbed["ct"]), "total_fast", envelope_mm=mm)
    _check(stubbed, *stubbed["seen"])


# -- and nothing asked is the whole volume too (the default was 20 mm until 2026-09-11) ------

def test_every_door_defaults_to_the_whole_volume(stubbed, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from haversack import Segmenter, cli
    from haversack.serve import LocalExecutor, create_app
    ct = stubbed["ct"]
    pipeline.segment(str(ct), "total_fast")
    Segmenter(weights=tmp_path).segment(str(ct), "total_fast")
    assert cli.main(["segment", str(ct), "--task", "total_fast",
                     "-o", str(tmp_path / "labels.nii.gz"), "--quiet"]) == 0
    client = TestClient(create_app(LocalExecutor(Segmenter(weights=tmp_path / "w"),
                                                 workdir=tmp_path / "work")))
    r = client.post("/v1/jobs", files={"file": ("ct.nii.gz", ct.read_bytes())},
                    data={"task": "total_fast", "options": "{}"})
    assert r.status_code == 202, r.text
    jid, t0 = r.json()["id"], time.time()
    while (s := client.get(f"/v1/jobs/{jid}").json())["state"] not in ("done", "failed"):
        assert time.time() - t0 < 30, s
        time.sleep(0.02)
    assert s["state"] == "done", s
    doors = ("segment()", "Segmenter", "the command line", "the server")
    for door, run in zip(doors, stubbed["seen"], strict=True):
        assert run["grid"] == stubbed["whole"], f"{door} cropped by default"
        assert run["recorded"] is None, door


def test_the_segmenter_policy_and_its_per_call_override(stubbed, tmp_path):
    from haversack import Segmenter
    ct = str(stubbed["ct"])
    Segmenter(weights=tmp_path, envelope_mm=0).segment(ct, "total_fast")
    Segmenter(weights=tmp_path, envelope_mm=3.0).segment(ct, "total_fast")
    Segmenter(weights=tmp_path).segment(ct, "total_fast", envelope_mm=0)
    Segmenter(weights=tmp_path).segment(ct, "total_fast", envelope_mm=3.0)
    policy_zero, policy_three, call_zero, call_three = stubbed["seen"]
    _check(stubbed, policy_zero, policy_three)
    _check(stubbed, call_zero, call_three)


def test_the_server(stubbed, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from haversack import Segmenter
    from haversack.serve import LocalExecutor, create_app
    client = TestClient(create_app(LocalExecutor(Segmenter(weights=tmp_path / "w"),
                                                 workdir=tmp_path / "work")))
    body = stubbed["ct"].read_bytes()
    for mm in (0, 3.0):
        r = client.post("/v1/jobs", files={"file": ("ct.nii.gz", body)},
                        data={"task": "total_fast", "options": f'{{"envelope_mm": {mm}}}'})
        assert r.status_code == 202, r.text
        jid, t0 = r.json()["id"], time.time()
        while (s := client.get(f"/v1/jobs/{jid}").json())["state"] not in ("done", "failed"):
            assert time.time() - t0 < 30, s
            time.sleep(0.02)
        assert s["state"] == "done", s
    _check(stubbed, *stubbed["seen"])


def test_the_command_line(stubbed, tmp_path):
    from haversack import cli
    for mm in ("0", "3"):
        rc = cli.main(["segment", str(stubbed["ct"]), "--task", "total_fast", "--envelope", mm,
                       "-o", str(tmp_path / f"labels{mm}.nii.gz"), "--quiet"])
        assert rc == 0
    _check(stubbed, *stubbed["seen"])
