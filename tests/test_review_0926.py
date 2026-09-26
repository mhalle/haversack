"""The 2026-09-25/26 adversarial review of the day's work, each finding pinned.

Four reviewers (world-geometry restore, packaging and Python, server and transcode, mutation
testing). What the server/transcode reviewer found is pinned in tests/test_input_stream.py
beside the code; this file holds the rest:

- packaging: engine flags decided on the deploying side (a lean deploy left FastSurfer half on);
  install hints that sent people to plain pip and to a FastSurfer venv; the core dependencies,
  the Python range and CI's matrix, none of which any test held;
- restore: FastSurfer stores written before 0.13.0 left unlateralized in silence; the
  "not lateralized" note never reaching the written file; the 17 right-hemisphere ids the split
  creates written unnamed; the lateralized labels' dtype; `ranked_restore.roi_of` on a frameless
  store; the rule inside an roi, the rule failing, the CLI's notes;
- mutation gaps: `_cache_record` racing reloads, the after-done scratch copy failing,
  `_check_streamed`'s geometry and layout checks, NIfTI intercepts, extensions and offsets.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = "git+https://github.com/mhalle/haversack"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _names(specs) -> set:
    return {re.split(r"[<>=!~\[; @]", s.strip(), maxsplit=1)[0].lower() for s in specs}


# -- packaging ---------------------------------------------------------------------------

def test_the_packages_haversack_cannot_do_without_are_core():
    """Input copies (duckn, zarr, pydicom), rank fields (rankfield) and FastSurfer degrade to
    'off' or a 501 without a word when their package is missing - so nothing else would notice
    one leaving the core list (mutants M6-M9, M12 survived every test)."""
    core = _names(_pyproject()["project"]["dependencies"])
    assert {"duckn", "zarr", "rankfield", "pydicom", "fastsurfer-lean", "provender"} <= core


def _ci() -> str:
    return (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")


def test_ci_runs_both_ends_of_the_supported_python_range():
    """CI never installs haversack, so requires-python was checked by nothing: dropping 3.14
    from it, or from the matrix, passed everything (M8, M10)."""
    from packaging.specifiers import SpecifierSet
    spec = SpecifierSet(_pyproject()["project"]["requires-python"])
    matrix = re.search(r'python:\s*\[([^\]]*)\]', _ci()).group(1)
    legs = [v.strip().strip('"\'') for v in matrix.split(",")]
    supported = [f"3.{m}" for m in range(8, 20) if f"3.{m}.0" in spec]
    assert legs and all(f"{v}.0" in spec for v in legs), (legs, str(spec))
    assert supported[0] in legs and supported[-1] in legs, (supported, legs)


def test_ci_installs_every_core_package_that_comes_from_git():
    """CI hand-lists its installs; a core package from a git tag that it forgets is one whose
    tests importorskip to nothing - FastSurfer's, the day it became core (M3)."""
    data = _pyproject()
    sources = data["tool"]["uv"]["sources"]
    core = _names(data["project"]["dependencies"])
    ci = _ci()
    missing = [n for n in sorted(core) if n in sources and "git" in sources[n]
               and f"'{n} @ git+" not in ci]
    assert not missing, missing


def test_no_message_sends_anyone_to_plain_pip():
    """haversack and its own packages are not on PyPI: `pip install 'haversack[...]'` fails, and
    names anyone could register are the worst thing to tell a user to install."""
    bad = []
    for p in (ROOT / "src" / "haversack").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(?<!uv )pip install ['\"]?haversack", line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, bad


def test_a_missing_core_engine_is_a_reinstall_and_an_optional_one_its_own_env():
    from haversack.engines import registry as R
    fs, ss = R.ENGINES["fastsurfer"], R.ENGINES["synthstrip"]
    assert fs.core and not ss.core
    assert f"@ {GIT}" in R.install_hint(fs) and ".venvs/" not in R.install_hint(fs)
    assert ".venvs/synthstrip" in R.install_hint(ss) and "--extra synthstrip" in R.install_hint(ss)


_DEPLOY_SIDE = r'''
import os, sys
if sys.argv[1] == "lean":
    sys.modules["FastSurferCNN"] = None     # find_spec answers None: not installed
for k in list(os.environ):
    if k.startswith("HAVERSACK_") and k.endswith(("FASTSURFER", "SYNTHSTRIP", "VOXTELL", "MONAI")):
        del os.environ[k]
import haversack.modal_app as m
print(os.environ.get("HAVERSACK_FASTSURFER"), "HAVERSACK_FASTSURFER" in m._RUNTIME_KNOBS,
      sorted(m.ENGINE_WORKERS))
'''


@pytest.mark.parametrize("side,flag", [("full", "1"), ("lean", "0")])
def test_the_deploying_side_decides_every_engine_flag(side, flag):
    """With the flag unset each container asked `enabled()` of its OWN environment: a deploy
    from a lean install built no FastSurfer worker while the api image, which has FastSurfer
    since it became core, turned it on - and spawned a worker class that was never deployed.
    The deploying import now writes its decision into the environment it forwards."""
    pytest.importorskip("modal")
    if side == "full":
        pytest.importorskip("FastSurferCNN")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__import__("haversack").__file__).parents[1])]
        + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    r = subprocess.run([sys.executable, "-c", _DEPLOY_SIDE, side], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    got, forwarded, workers = r.stdout.strip().splitlines()[-1].split(" ", 2)
    assert got == flag and forwarded == "True"
    assert ("'fastsurfer'" in workers) == (flag == "1")


# -- restore -----------------------------------------------------------------------------

rf = pytest.importorskip("rankfield")
torch = pytest.importorskip("torch")

from haversack import ranked_restore as R  # noqa: E402

from test_world_restore import CENTER, _store  # noqa: E402


def _fake_split(monkeypatch, fn):
    import types
    mod = types.ModuleType("FastSurferCNN.data_loader.data_utils")
    mod.split_cortex_labels = fn
    monkeypatch.setitem(sys.modules, "FastSurferCNN.data_loader.data_utils", mod)


def test_the_lateralized_labels_keep_their_values(monkeypatch, tmp_path):
    """The split's ids run to 2035; a uint8 choice made on the wrong bound wraps them mod 256
    in silence - the fake that returned `a + 1` never reached 256 (RR2)."""
    _fake_split(monkeypatch, lambda a: np.where(a == 17, 2003, a))
    r = R.restore(_store(tmp_path), device="cpu")
    assert r.labels.dtype == np.uint16 and r.labels.max() == 2003 and r.notes == ()


def _old_store(tmp_path, meta):
    """A FastSurfer store as 0.13.0's predecessors wrote it: no `labels_named_by`."""
    import rankfield.store as rs
    s = _store(tmp_path)
    parts = rs.read_parts(rs.open_group(s))
    f = parts[0].field
    m = {k: v for k, v in f.meta.items() if k != "labels_named_by"}
    m.update(meta)
    f.meta = m
    f.ranks, f.support = np.asarray(f.ranks), np.asarray(f.support)
    out = tmp_path / "old.duckn.zip"
    rs.write_parts(out, parts)
    return out


@pytest.mark.parametrize("meta", [{"task": "fastsurfer:asegdkt"},
                                  {"softmax": {"engine": "fastsurfer"}}],
                         ids=["task", "softmax-engine"])
def test_a_store_written_before_labels_named_by_is_still_fastsurfers(monkeypatch, tmp_path, meta):
    seen = []
    _fake_split(monkeypatch, lambda a: seen.append(1) or a)
    r = R.restore(_old_store(tmp_path, meta), device="cpu")
    assert seen == [1] and r.fastsurfer


def test_the_rule_in_an_roi_and_the_rule_failing_each_say_so(monkeypatch, tmp_path):
    _fake_split(monkeypatch, lambda a: a + 1)
    s = _store(tmp_path)
    r = R.restore(s, device="cpu", roi=((0, 10), (0, 10), (0, 10)))
    assert "roi" in r.notes[0]
    _fake_split(monkeypatch, lambda a: (_ for _ in ()).throw(IndexError("no white matter")))
    r = R.restore(s, device="cpu")
    assert "failed" in r.notes[0] and "IndexError" in r.notes[0]
    assert r.labels.max() == 17                        # as the field says, not half-split


def test_the_written_file_says_what_the_restore_did_not_do(monkeypatch, tmp_path, capsys):
    """The note went to stderr only; the .seg.nrrd - what outlives the terminal - had none."""
    import json

    import SimpleITK as sitk
    monkeypatch.setitem(sys.modules, "FastSurferCNN.data_loader.data_utils", None)
    out = tmp_path / "labels.seg.nrrd"
    assert R.main_cli([str(_store(tmp_path)), "-o", str(out), "--device", "cpu"]) == 0
    assert "note: cortical parcels are not lateralized" in capsys.readouterr().err
    img = sitk.ReadImage(str(out))
    prov = next(json.loads(img.GetMetaData(k)) for k in img.GetMetaDataKeys() if "provenance" in k)
    assert "not lateralized" in prov["notes"][0]


def test_the_ids_the_split_creates_are_named(monkeypatch, tmp_path):
    """The store names the network's channels; 2003 is not one of them, and the restore wrote it
    as `label_2003` - as FastSurfer's own labels output did (the LUT is FastSurfer's 79)."""
    import SimpleITK as sitk

    from haversack.engines.fastsurfer import output_lut
    _fake_split(monkeypatch, lambda a: np.where(a == 17, 2003, a))
    out = tmp_path / "labels.seg.nrrd"
    assert R.main_cli([str(_store(tmp_path)), "-o", str(out), "--device", "cpu", "--quiet"]) == 0
    img = sitk.ReadImage(str(out))
    names = [img.GetMetaData(k) for k in img.GetMetaDataKeys() if k.endswith("_Name")]
    assert output_lut()[2003]["name"] in names and "label_2003" not in names
    assert output_lut()[2003]["name"] == output_lut()[1003]["name"].replace("ctx-lh-", "ctx-rh-")


def test_output_lut_names_exactly_the_split_ids():
    from haversack.engines.fastsurfer import SPLIT_AFTER_THE_NETWORK, load_lut, output_lut
    added = set(output_lut()) - set(load_lut())
    assert added == {v + 1000 for v in SPLIT_AFTER_THE_NETWORK}
    assert all(output_lut()[v]["name"].startswith("ctx-rh-") for v in added)


def test_roi_of_a_frameless_store_bounds_it_on_the_input_grid(tmp_path):
    """`ranked_restore.roi_of` on a FastSurfer store: the test that checked it called rankfield's
    `roi_of` directly, so the adapter dropping `world=` went unseen (RR7)."""
    import json
    import zipfile
    s = _store(tmp_path)
    whole = R.restore(s, device="cpu").labels
    parts = R.parts_of(R._open(s)[1])
    geo = parts[0].field.geometry
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in geo.shape], indexing="ij"), -1)
    inside = np.argwhere(np.linalg.norm(geo.world(idx) - CENTER, axis=-1) < 9.0)
    ext = [int(v) for pair in zip(inside.min(0), inside.max(0)) for v in pair]
    # give the bare store the segments block a built store has
    with zipfile.ZipFile(s) as z:
        members = {n: z.read(n) for n in z.namelist()}
    root = json.loads(members["zarr.json"])
    duckn = root.setdefault("attributes", {}).setdefault("duckn", {})
    duckn.setdefault("extensions", {})["seg"] = {"segments": [
        {"id": "s17", "name": "ball", "label_values": [17], "layer": 0, "extent": ext}]}
    members["zarr.json"] = json.dumps(root).encode()
    with zipfile.ZipFile(s, "w", zipfile.ZIP_STORED) as z:
        for n, b in members.items():
            z.writestr(n, b)
    box = R.roi_of(s, [17])
    ball = np.argwhere(whole == 17)
    for ax, (a, b) in enumerate(box):
        assert a <= ball[:, ax].min() and ball[:, ax].max() < b, (box, ball.min(0), ball.max(0))


def test_fastsurfers_labels_output_names_the_split_ids():
    """What `run_local` writes into the header: before, the 17 split ids were `label_20xx`."""
    from haversack.engines.fastsurfer import SPLIT_AFTER_THE_NETWORK, segment_names
    names = segment_names([0, 2, 1003, 2003, 2002, 9999])
    assert 0 not in names and names[9999] == "label_9999"
    assert names[2003] == "ctx-rh-caudalmiddlefrontal" and names[1003] == "ctx-lh-caudalmiddlefrontal"
    assert all(not n.startswith("label_") for n in segment_names([v + 1000 for v in SPLIT_AFTER_THE_NETWORK]).values())
