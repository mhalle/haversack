"""`haversack get` and the cache utilities (2026-09-03).

A user segmenting a remote dataset usually wants the data too. `get` fetches a source into
the cache (or out to a file, converting a DICOM series to one volume), and `cache`/`weights`
list and clean what is on disk. Fakes and a local DICOM-ish series - nothing leaves the machine.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from haversack import cli, io, sources
from haversack.errors import InputError


class FakeSource(sources.DataSource):
    prefix = "fake"
    id_pattern = r"[a-z0-9_]+"
    description = "test double: writes one nrrd"

    def fetch(self, identifier, dest_dir, *, credentials=None):
        import SimpleITK as sitk
        d = Path(dest_dir) / "series"; d.mkdir()
        img = sitk.GetImageFromArray(np.arange(8 * 8 * 8, dtype=np.int16).reshape(8, 8, 8))
        img.SetSpacing((1.0, 1.2, 1.5))
        sitk.WriteImage(img, str(d / f"{identifier}.nrrd"))
        return d


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "default_sources", lambda: [FakeSource()])
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    return FakeSource


def _run(argv, tmp_path):
    monkeyenv = {"HAVERSACK_CACHE_DIR": str(tmp_path / "cache")}
    code = ("import haversack.cli as c, haversack.sources as s, haversack.engines.fastsurfer as f;"
            "s.default_sources=lambda:[__import__('tests.test_get_and_cache',fromlist=['FakeSource']).FakeSource()];"
            f"raise SystemExit(c.main({argv!r}))")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, env={**__import__("os").environ, **monkeyenv})


def test_get_no_output_prints_the_cache_path(fake, tmp_path, capsys):
    assert cli.main(["get", "fake:case1"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("case1.nrrd") and (tmp_path / "cache" / "inputs" / "fake") in Path(out).parents


def test_get_converts_a_series_to_one_volume(fake, tmp_path, capsys):
    dst = tmp_path / "out" / "case.nii.gz"
    assert cli.main(["get", "fake:case1", "-o", str(dst)]) == 0
    assert dst.is_file()
    import SimpleITK as sitk
    assert sitk.ReadImage(str(dst)).GetSpacing() == pytest.approx((1.0, 1.2, 1.5))   # geometry preserved
    assert (tmp_path / "cache" / "inputs" / "fake").is_dir()                          # cached by default


def test_get_format_into_a_directory_auto_names(fake, tmp_path, capsys):
    assert cli.main(["get", "fake:case1", "--format", "nrrd", "-o", str(tmp_path / "d") + "/"]) == 0
    assert (tmp_path / "d" / "case1.nrrd").is_file()


class MultiFileSource(sources.DataSource):
    prefix = "multi"
    id_pattern = r"[a-z0-9_]+"
    description = "test double: a two-file series (a directory)"

    def fetch(self, identifier, dest_dir, *, credentials=None):
        d = Path(dest_dir) / "series"; d.mkdir()
        (d / "001.dcm").write_bytes(b"a"); (d / "002.dcm").write_bytes(b"b")
        return d


def test_get_raw_copy_of_a_series_names_the_dir_by_source(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sources, "default_sources", lambda: [MultiFileSource()])
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    assert cli.main(["get", "multi:seriesx", "-o", str(tmp_path / "raw") + "/"]) == 0
    assert (tmp_path / "raw" / "seriesx" / "001.dcm").is_file()                       # named by source, not "series"
    assert (tmp_path / "raw" / "seriesx" / "002.dcm").is_file()


def test_get_no_cache_leaves_nothing_cached(fake, tmp_path):
    assert cli.main(["get", "fake:case1", "--no-cache", "-o", str(tmp_path / "x.nrrd")]) == 0
    assert (tmp_path / "x.nrrd").is_file()
    assert not (tmp_path / "cache" / "inputs" / "fake").exists()


def test_get_no_cache_without_output_is_refused(fake, tmp_path):
    with pytest.raises(InputError, match="no-cache needs -o"):
        cli._run(["get", "fake:case1", "--no-cache"])


def test_get_local_path_is_a_noop(tmp_path, capsys):
    (tmp_path / "scan.nii.gz").touch()
    assert cli.main(["get", str(tmp_path / "scan.nii.gz")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "scan.nii.gz")


def sitk_series(dirpath, zs):
    """A CT series as SimpleITK writes it: one 4x4 slice per z, its position in the IPP tag.
    ``KeepOriginalImageUIDOn`` so the UIDs given are the ones written."""
    import SimpleITK as sitk
    dirpath.mkdir(parents=True)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.9"
    for i, z in enumerate(zs):
        sl = sitk.GetImageFromArray(np.full((4, 4), i, np.int16))
        for tag, val in (("0008|0060", "CT"), ("0020|000d", f"{uid}.1"), ("0020|000e", f"{uid}.2"),
                         ("0008|0018", f"{uid}.3.{i}"), ("0020|0013", str(i + 1)),
                         ("0020|0032", f"-266\\-138\\{z:g}"), ("0020|0037", "1\\0\\0\\0\\1\\0")):
            sl.SetMetaData(tag, val)
        w.SetFileName(str(dirpath / f"slice_{i:03d}.dcm"))
        w.Execute(sl)
    return dirpath


GAPPED = (31.0, 32.0, 33.0, 35.0, 36.0)          # 34 is missing


class Gapped(sources.DataSource):
    prefix = "gapped"
    id_pattern = r"[a-z0-9]+"
    description = "test double: a CT series with one slice missing"

    def fetch(self, identifier, dest_dir, *, credentials=None):
        return sitk_series(Path(dest_dir) / "series", GAPPED)


class ConvertRefusesWhatSegmentRefuses(unittest.TestCase):
    """`get SOURCE -o scan.nii.gz` converts through io.convert, which until 2026-09-11 read a
    series with a bare ImageSeriesReader: a series with a missing slice came out regridded onto
    ITK's mean step, with a warning on stderr and nothing else (IDC eay131, 3 mm slices with
    four 6 mm gaps, was written at 3.0577 mm with slices up to ~2.9 mm off). The NIfTI is then a
    clean uniform grid, so `segment` on it cannot see what it refuses on the source."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        for p in (mock.patch.object(sources, "default_sources", lambda: [Gapped()]),
                  mock.patch.dict(os.environ, {"HAVERSACK_CACHE_DIR": str(self.tmp / "cache")})):
            p.start()
            self.addCleanup(p.stop)

    def segment_says(self, series) -> str:
        with self.assertRaises(InputError) as said:      # `segment` reads through read_image
            io.read_image(series)
        return str(said.exception)

    def test_convert_raises_segments_error_and_writes_nothing(self):
        src = sitk_series(self.tmp / "series", GAPPED)
        dst = self.tmp / "out" / "scan.nii.gz"
        with self.assertRaises(InputError) as said:
            io.convert(src, dst)
        self.assertEqual(str(said.exception), self.segment_says(src))
        self.assertIn("missing or duplicate slices", str(said.exception))
        self.assertFalse(dst.exists())

    def test_get_refuses_in_one_line_that_names_the_raw_copy(self):
        dst = self.tmp / "scan.nii.gz"
        err = StringIO()
        with redirect_stderr(err):
            rc = cli.main(["get", "gapped:case1", "-o", str(dst)])
        self.assertEqual(rc, 2)
        self.assertFalse(dst.exists())
        said = [ln for ln in err.getvalue().splitlines() if ln.startswith("haversack:")]
        self.assertEqual(len(said), 1)
        self.assertIn(self.segment_says(sources.materialize("gapped:case1")), said[0])
        self.assertIn("-o <directory>/", said[0])

    def test_the_raw_copy_it_names_still_takes_the_series(self):
        """The way out the refusal names must work on the very series it refused."""
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            rc = cli.main(["get", "gapped:case1", "-o", str(self.tmp / "raw") + "/"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(list((self.tmp / "raw" / "case1").glob("*.dcm"))), len(GAPPED))


def test_cache_list_and_clean(fake, tmp_path, capsys):
    cli.main(["get", "fake:case1"]); cli.main(["get", "fake:case2"]); capsys.readouterr()
    from haversack import cache_admin
    rows = {r["name"]: r for r in cache_admin.usage()}
    assert rows["inputs"]["items"] == 2 and rows["weights"]["sweepable"] is False
    # dry run removes nothing
    assert cli.main(["cache", "clean", "inputs", "--dry-run"]) == 0
    assert (tmp_path / "cache" / "inputs" / "fake").is_dir()
    # one entry by spec
    assert cli.main(["cache", "clean", "inputs", "fake:case1", "--yes"]) == 0
    assert cache_admin.usage()[0]["items"] == 1
    # whole category
    assert cli.main(["cache", "clean", "inputs", "--yes"]) == 0
    assert cache_admin.usage()[0]["items"] == 0


def test_cache_clean_never_sweeps_weights():
    from haversack import cache_admin
    with pytest.raises(InputError, match="weights remove"):
        cache_admin.clean("weights")


def test_older_than_parsing(fake, tmp_path, capsys):
    cli.main(["get", "fake:fresh"]); capsys.readouterr()
    # everything is fresh, so a 30d cutoff removes nothing
    assert cli.main(["cache", "clean", "inputs", "--older-than", "30d", "--yes"]) == 0
    from haversack import cache_admin
    assert cache_admin.usage()[0]["items"] == 1
    with pytest.raises(InputError, match="use a number then"):
        cli._run(["cache", "clean", "inputs", "--older-than", "30x", "--yes"])


# -- batch mode (multiple inputs) --------------------------------------------------------
def test_get_batch_converts_each_into_a_directory(fake, tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["get", "fake:case1", "fake:case2", "--format", "nrrd", "-o", str(out)]) == 0
    assert (out / "case1.nrrd").is_file() and (out / "case2.nrrd").is_file()


def test_get_batch_no_output_caches_each_and_prints_paths(fake, tmp_path, capsys):
    assert cli.main(["get", "fake:case1", "fake:case2"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 2 and all("inputs/fake" in l for l in lines)


def test_get_batch_defaults_output_dir_to_cwd(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["get", "fake:case1", "fake:case2", "--format", "nrrd"]) == 0
    assert (tmp_path / "case1.nrrd").is_file() and (tmp_path / "case2.nrrd").is_file()


def test_get_batch_one_failure_does_not_sink_the_rest(fake, tmp_path, capsys):
    rc = cli.main(["get", "fake:good", "fake:BAD-ID", "--format", "nrrd", "-o", str(tmp_path / "o")])
    assert rc == 1                                                # one failed
    assert (tmp_path / "o" / "good.nrrd").is_file()               # the good one still landed


def test_segment_batch_writes_named_outputs(monkeypatch, tmp_path, capsys):
    from haversack import cli, pipeline
    import numpy as np

    class R:
        timings = {}
        grid = type("G", (), {"shape": (2, 2, 2)})()
        schema = type("S", (), {"names": ["liver"]})()
        provenance = {}

        def save(self, path):
            Path(path).write_text("labels"); return path

        def present(self):
            return {1: "liver"}

    monkeypatch.setattr(pipeline, "segment", lambda image, task, **kw: R())
    (tmp_path / "a.nii.gz").touch(); (tmp_path / "b.nii.gz").touch()
    out = tmp_path / "labels"
    rc = cli.main(["segment", str(tmp_path / "a.nii.gz"), str(tmp_path / "b.nii.gz"),
                   "--task", "total_fast", "--format", "seg.nrrd", "-o", str(out)])
    assert rc == 0
    assert (out / "a_total_fast.seg.nrrd").is_file() and (out / "b_total_fast.seg.nrrd").is_file()


def test_segment_batch_requires_format(tmp_path):
    from haversack import cli
    (tmp_path / "a.nii.gz").touch(); (tmp_path / "b.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "a.nii.gz"), str(tmp_path / "b.nii.gz"),
                   "--task", "total_fast", "-o", str(tmp_path / "o")])
    assert rc == 2                                                # InputError -> one line, status 2


def test_segment_single_input_unchanged(monkeypatch, tmp_path, capsys):
    from haversack import cli, pipeline

    class R:
        timings = {}; grid = type("G", (), {"shape": (1, 1, 1)})(); schema = type("S", (), {"names": []})()
        provenance = {}
        def save(self, path): return path
        def present(self): return {}

    saved = {}
    def fake_seg(image, task, **kw):
        saved["img"] = image; return R()
    monkeypatch.setattr(pipeline, "segment", fake_seg)
    (tmp_path / "scan.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "scan.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.seg.nrrd")])
    assert rc == 0 and saved["img"] == str(tmp_path / "scan.nii.gz")
