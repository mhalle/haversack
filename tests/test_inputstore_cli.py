"""The command line's inputs with HAVERSACK_INPUT_STORE=blobs (step 6b).

Store-neutral versions of what the legacy cache's tests pin, plus what the store must add:
one fetch across processes; `get` printing a path that is still there after the command
exits, of the ORIGINAL files (the command line keeps no input copy - a viewer opens what
`get` hands out); provenance from the stored record; `cache usage` and `clean` counting refs.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from haversack import cli, sources

from test_get_and_cache import FakeSource


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    monkeypatch.setattr(sources, "default_sources", lambda: [FakeSource()])
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    from haversack import inputstore
    monkeypatch.setattr(inputstore, "_COMMAND", {})
    return FakeSource


def _child(argv, tmp_path):
    """`haversack <argv>` in a real process, the FakeSource its only source."""
    code = ("import haversack.cli as c, haversack.sources as s;"
            "s.default_sources=lambda:[__import__('test_get_and_cache',fromlist=['FakeSource']).FakeSource()];"
            f"raise SystemExit(c.main({argv!r}))")
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "HAVERSACK_CACHE_DIR": str(tmp_path / "cache"), "HAVERSACK_INPUT_STORE": "blobs",
           "PYTHONPATH": os.pathsep.join(p for p in (str(root / "tests"), str(root / "src"),
                                                      os.environ.get("PYTHONPATH", "")) if p)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, env=env)


def test_get_prints_the_original_and_it_outlives_the_command(fake, tmp_path):
    r = _child(["get", "fake:case1"], tmp_path)
    assert r.returncode == 0, r.stderr
    printed = Path(r.stdout.strip())
    assert printed.name == "case1.nrrd"                   # the original, not an input copy
    assert printed.is_file()                               # the process is gone; the path is not
    assert (tmp_path / "cache" / "input-store") in printed.parents


def test_two_commands_fetch_once_and_print_one_path(fake, tmp_path):
    a, b = _child(["get", "fake:case1"], tmp_path), _child(["get", "fake:case1"], tmp_path)
    assert a.stdout.strip() == b.stdout.strip()
    assert "fetching" in a.stderr and "fetching" not in b.stderr


def test_provenance_comes_from_the_stored_record(fake, tmp_path, monkeypatch):
    class Recording(FakeSource):
        def fetch(self, identifier, dest_dir, *, credentials=None):
            out = super().fetch(identifier, dest_dir, credentials=credentials)
            (Path(dest_dir) / sources.INPUT_SIDECAR).write_text(json.dumps(
                {"kind": "fake", "content": {"digest": "sha256:current"},
                 "origin": {"collection": "stored"}, "license": None, "cite": []}))
            return out
    monkeypatch.setattr(sources, "fetch_recording_origin",
                        lambda src, ident, entry, cred=None: src.fetch(ident, entry))
    sources.materialize("fake:case1", sources=[Recording()])
    rec = sources.input_record("fake:case1", sources=[Recording()])
    assert rec["content"]["digest"] == "sha256:current" and rec["identity"] == "fake:case1"


def test_cache_usage_and_clean_count_refs(fake, tmp_path, capsys):
    from haversack import cache_admin
    cli.main(["get", "fake:case1"]); cli.main(["get", "fake:case2"]); capsys.readouterr()
    rows = {r["name"]: r for r in cache_admin.usage()}
    assert rows["inputs"]["items"] == 2 and rows["inputs"]["bytes"] > 0
    assert cli.main(["cache", "clean", "inputs", "--dry-run"]) == 0
    assert {r["name"]: r for r in cache_admin.usage()}["inputs"]["items"] == 2
    assert cli.main(["cache", "clean", "inputs", "fake:case1", "--yes"]) == 0
    assert {r["name"]: r for r in cache_admin.usage()}["inputs"]["items"] == 1
    exports = tmp_path / "cache" / "input-store" / "exports"
    assert len([p for p in exports.iterdir() if not p.name.startswith(".")]) == 1
    assert cli.main(["cache", "clean", "inputs", "--yes"]) == 0
    assert {r["name"]: r for r in cache_admin.usage()}["inputs"]["items"] == 0
    blobs = tmp_path / "cache" / "input-store" / "store" / "blobs" / "sha256"
    assert not blobs.exists() or not any(blobs.iterdir())   # an explicit clean frees the bytes


def test_a_batch_materializes_every_input_and_each_path_stays(fake, tmp_path):
    paths = [sources.materialize(f"fake:case{i}") for i in range(3)]
    assert all(p.is_file() and p.name == f"case{i}.nrrd" for i, p in enumerate(paths))


def test_an_input_the_server_would_copy_is_kept_as_fetched(fake, tmp_path):
    """A compressed NIfTI is what the server keeps as an input copy; the command line keeps
    the file (`get` hands it to the user, and a viewer opens a NIfTI, not a duckn copy)."""
    import SimpleITK as sitk

    class Gz(FakeSource):
        def fetch(self, identifier, dest_dir, *, credentials=None):
            d = Path(dest_dir) / "series"
            d.mkdir()
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((6, 6, 6), np.int16)),
                            str(d / f"{identifier}.nii.gz"))
            return d
    from haversack import input_copy
    assert input_copy.wanted(Path(tmp_path / "x.nii.gz"))    # what the server would copy
    got = sources.materialize("fake:gz", sources=[Gz()])
    assert got.name == "gz.nii.gz" and got.is_file()
