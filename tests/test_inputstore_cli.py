"""The command line's inputs with HAVERSACK_INPUT_STORE=blobs (step 6b).

Store-neutral versions of what the legacy cache's tests pin, plus what the store must add:
one fetch across processes; `get` printing a path that is still there after the command
exits; provenance from the stored record; `cache usage` and `clean` counting refs. And the
user's decision of 2026-09-26: no export of the bytes as they came off the wire - `get` hands
out haversack's form (the input copy, or an input already in its own efficient form), and an
input the reader refused has nothing to export.
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
    assert printed.name == "case1.nrrd"                   # a raw NRRD is stored as it is
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


def _gz_source(FakeSource):
    import SimpleITK as sitk

    class Gz(FakeSource):
        def fetch(self, identifier, dest_dir, *, credentials=None):
            d = Path(dest_dir) / "series"
            d.mkdir()
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((6, 6, 6), np.int16)),
                            str(d / f"{identifier}.nii.gz"))
            return d
    return Gz()


def test_the_command_line_keeps_the_copy_as_the_server_does(fake, tmp_path):
    got = sources.materialize("fake:gz", sources=[_gz_source(FakeSource)])
    assert got.name == "input.duckn.zip" and got.is_file()


def test_get_into_a_directory_writes_the_copy_named_by_the_source(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sources, "default_sources", lambda: [_gz_source(FakeSource)])
    out = tmp_path / "out"
    assert cli.main(["get", "fake:scan", "fake:other", "-o", str(out) + "/"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["other.duckn.zip", "scan.duckn.zip"]
    from haversack import io
    assert io.read_image(out / "scan.duckn.zip").GetSize() == (6, 6, 6)


def test_get_converts_from_the_copy(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sources, "default_sources", lambda: [_gz_source(FakeSource)])
    dst = tmp_path / "scan.nrrd"
    assert cli.main(["get", "fake:scan", "-o", str(dst)]) == 0
    import SimpleITK as sitk
    assert sitk.ReadImage(str(dst)).GetSize() == (6, 6, 6)


def test_an_input_in_its_own_efficient_form_is_handed_out_as_stored(fake, tmp_path, capsys):
    """A raw NRRD is never copied (it is already a mapped read): stored as it is, and that IS
    haversack's form of it."""
    out = tmp_path / "out"
    assert cli.main(["get", "fake:case1", "-o", str(out) + "/"]) == 0
    assert [p.name for p in out.iterdir()] == ["case1.nrrd"]


def test_an_input_the_reader_refused_is_not_exported_as_fetched(fake, tmp_path, monkeypatch, capsys):
    class Broken(FakeSource):
        def fetch(self, identifier, dest_dir, *, credentials=None):
            d = Path(dest_dir) / "series"
            d.mkdir()
            (d / "a.dcm").write_bytes(b"DICM but not really")
            (d / "b.dcm").write_bytes(b"DICM but not really either")
            return d
    monkeypatch.setattr(sources, "default_sources", lambda: [Broken()])
    assert cli.main(["get", "fake:bad", "-o", str(tmp_path / "out") + "/"]) != 0
    assert "does not hand out the files as fetched" in capsys.readouterr().err
    assert not (tmp_path / "out" / "bad").exists()


# -- the cached form as the standard form of input (2026-09-26) ---------------------------------

def test_a_local_file_is_ingested_and_read_as_its_copy(fake, tmp_path):
    import SimpleITK as sitk
    src = tmp_path / "local" / "ct.nii.gz"
    src.parent.mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(np.full((6, 6, 6), 3, np.int16)), str(src))
    got = sources.materialize(str(src))
    assert got.name == "input.duckn.zip"
    assert sources.materialize(str(src)) == got                 # one identity: its bytes
    from haversack import io
    assert (sitk.GetArrayFromImage(io.read_image(got)) == 3).all()


def test_the_same_bytes_anywhere_are_one_input(fake, tmp_path):
    import shutil
    import SimpleITK as sitk
    a = tmp_path / "a" / "ct.nii.gz"
    a.parent.mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(np.full((6, 6, 6), 4, np.int16)), str(a))
    b = tmp_path / "b" / "other-name.nii.gz"
    b.parent.mkdir()
    shutil.copyfile(a, b)
    assert sources.materialize(str(a)) == sources.materialize(str(b))


def test_a_local_series_folder_is_one_tree_input(fake, tmp_path):
    from test_several_series import THREE, write_series
    d = write_series(tmp_path / "dcm", 3, THREE, value=6)
    got = sources.materialize(str(d))
    assert got.name == "input.duckn.zip"


def test_what_is_not_an_image_and_haversacks_own_form_pass_through(fake, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    assert sources.materialize(str(txt)) == txt
    copy = sources.materialize(str(_nifti_at(tmp_path / "n.nii.gz")))
    assert sources.materialize(str(copy)) == copy


def test_without_the_flag_a_local_path_is_read_where_it_is(tmp_path, monkeypatch):
    monkeypatch.delenv("HAVERSACK_INPUT_STORE", raising=False)
    p = _nifti_at(tmp_path / "n.nii.gz")
    assert sources.materialize(str(p)) == p


def _nifti_at(path):
    import SimpleITK as sitk
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((5, 5, 5), np.int16)), str(path))
    return Path(path)


def test_cli_segment_reads_a_local_input_as_its_copy(fake, tmp_path, monkeypatch, capsys):
    from haversack import pipeline
    from test_local_sources import _Saved
    src = _nifti_at(tmp_path / "local" / "ct.nii.gz")
    got = {}
    monkeypatch.setattr(pipeline, "segment", lambda image, task, **kw: got.update(image=image) or _Saved())
    rc = cli.main(["segment", str(src), "--task", "total_fast", "-o", str(tmp_path / "out.nii.gz")])
    assert rc == 0, capsys.readouterr().err
    assert Path(got["image"]).name == "input.duckn.zip"


def test_a_duckn_folder_store_is_read_where_it_is_not_stored_as_a_tree(fake, tmp_path):
    """A duckn/zarr directory is haversack's own form already. Without the guard it would be
    ingested as a tree of chunk files - a new identity for bytes nobody sent as an input."""
    import SimpleITK as sitk
    from duckn.io import write as duckn_write
    from duckn.sitk_adapter import from_sitk
    store = tmp_path / "vol.zarr"
    duckn_write(from_sitk(sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16))), store, format="zarr")
    assert sources.materialize(str(store)) == store
    assert sources.materialize(str(store)) == store
    from haversack import inputstore
    assert inputstore._COMMAND == {} or all(not c.store._refs() for c in inputstore._COMMAND.values())
