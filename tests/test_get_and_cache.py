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


# -- a local path is written as a fetched one is (2026-09-11) -----------------------------
def dicom_series(dirpath, n=4, dz=2.5):
    """A CT series as a scanner's folder holds it: one 4x4 slice per file, pixels 0.5 mm,
    each slice placed by its ImagePositionPatient ``dz`` mm along z and filled with its
    index. SimpleITK writes it, keeping the UIDs given, so the files are one series."""
    import SimpleITK as sitk
    dirpath.mkdir(parents=True)
    w = sitk.ImageFileWriter()
    w.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.7"
    for i in range(n):
        sl = sitk.GetImageFromArray(np.full((4, 4), i, np.int16))
        sl.SetSpacing((0.5, 0.5))
        for tag, val in (("0008|0060", "CT"), ("0020|000d", f"{uid}.1"), ("0020|000e", f"{uid}.2"),
                         ("0008|0018", f"{uid}.3.{i}"), ("0020|0013", str(i + 1)),
                         ("0020|0032", f"0\\0\\{i * dz:g}"), ("0020|0037", "1\\0\\0\\0\\1\\0")):
            sl.SetMetaData(tag, val)
        w.SetFileName(str(dirpath / f"slice_{i:03d}.dcm"))
        w.Execute(sl)
    return dirpath


def nrrd(path, fill):
    """A 4x4x4 int16 volume of ``fill`` (a number, or an array that shape), spacing (1, 1.2, 1.5)."""
    import SimpleITK as sitk
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.broadcast_to(np.asarray(fill, np.int16), (4, 4, 4)).copy())
    img.SetSpacing((1.0, 1.2, 1.5))
    sitk.WriteImage(img, str(path))
    return path


def voxels(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


class _Get(unittest.TestCase):
    """`haversack get` in process, with the fake remote source and a cache of the test's own:
    a probe must never reach the real one (AGENTS.md - one deleted 421 MB of it)."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        for p in (mock.patch.object(sources, "default_sources", lambda: [FakeSource()]),
                  mock.patch.dict(os.environ, {"HAVERSACK_CACHE_DIR": str(self.tmp / "cache")})):
            p.start()
            self.addCleanup(p.stop)

    def get(self, *argv):
        """``(exit status, stdout lines, stderr)``."""
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(["get", *map(str, argv)])
        return rc, out.getvalue().splitlines(), err.getvalue()

    def chdir(self, where):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(where)


class LocalSourceIsWritten(_Get):
    """`get ./series -o x.nii.gz` exited 0 having written nothing until 2026-09-11: a local path
    was printed and returned before -o was looked at, and a batch with --format printed each
    local path and converted none. A local path is a fetch already done, and is written as a
    fetched source is - converted for an image extension or --format, copied for a directory -o."""

    def setUp(self):
        super().setUp()
        self.series = dicom_series(self.tmp / "cases" / "ct_series")
        self.scan = nrrd(self.tmp / "cases" / "scan.nrrd", np.arange(64).reshape(4, 4, 4))

    def test_a_series_converts_to_the_file_named(self):
        dst = self.tmp / "out" / "x.nii.gz"
        rc, printed, _ = self.get(self.series, "-o", dst)
        self.assertEqual((rc, printed), (0, [str(dst)]))
        import SimpleITK as sitk
        img = sitk.ReadImage(str(dst))
        self.assertEqual(img.GetSize(), (4, 4, 4))
        np.testing.assert_allclose(img.GetSpacing(), (0.5, 0.5, 2.5))
        self.assertEqual(voxels(dst)[:, 0, 0].tolist(), [0, 1, 2, 3])        # every slice, in order

    def test_a_file_converts_by_extension(self):
        dst = self.tmp / "x.nii.gz"
        rc, printed, _ = self.get(self.scan, "-o", dst)
        self.assertEqual((rc, printed), (0, [str(dst)]))
        np.testing.assert_array_equal(voxels(dst), voxels(self.scan))

    def test_format_into_a_directory_names_it_by_the_source(self):
        d = self.tmp / "d"
        rc, printed, _ = self.get(self.series, "--format", "nrrd", "-o", f"{d}/")
        self.assertEqual((rc, printed), (0, [str(d / "ct_series.nrrd")]))
        self.assertTrue((d / "ct_series.nrrd").is_file())

    def test_format_alone_converts_into_the_current_directory(self):
        self.chdir(self.tmp)
        rc, printed, _ = self.get(self.scan, "--format", "nifti")
        self.assertEqual((rc, printed), (0, ["scan.nii.gz"]))
        self.assertTrue((self.tmp / "scan.nii.gz").is_file())

    def test_a_directory_output_copies_the_series(self):
        rc, printed, _ = self.get(self.series, "-o", f"{self.tmp / 'raw'}/")
        copied = self.tmp / "raw" / "ct_series"
        self.assertEqual((rc, printed), (0, [str(copied)]))
        self.assertEqual(sorted(p.read_bytes() for p in copied.iterdir()),
                         sorted(p.read_bytes() for p in self.series.iterdir()))

    def test_a_batch_converts_local_and_fetched_sources_alike(self):
        out = self.tmp / "out"
        rc, printed, _ = self.get(self.scan, self.series, "fake:case1", "--format", "nifti", "-o", out)
        want = [out / "scan.nii.gz", out / "ct_series.nii.gz", out / "case1.nii.gz"]
        self.assertEqual((rc, printed), (0, [str(p) for p in want]))
        self.assertTrue(all(p.is_file() for p in want))

    def test_a_batch_copies_local_and_fetched_sources_alike(self):
        out = self.tmp / "raw"
        rc, printed, _ = self.get(self.scan, self.series, "fake:case1", "-o", out)
        self.assertEqual((rc, printed), (0, [str(out / n) for n in ("scan.nrrd", "ct_series", "case1.nrrd")]))
        self.assertEqual(len(list((out / "ct_series").iterdir())), 4)

    def test_dot_is_named_by_the_folder_it_is(self):
        """`.` was named `.`, so this would have written `d/..nrrd` - a hidden file."""
        self.chdir(self.series)
        rc, printed, _ = self.get(".", "--format", "nrrd", "-o", f"{self.tmp / 'd'}/")
        self.assertEqual((rc, printed), (0, [str(self.tmp / "d" / "ct_series.nrrd")]))

    def test_a_bang_is_part_of_a_local_name(self):
        """Only a remote spec has `!member`; a local `a!b.nrrd` was named `b`."""
        self.assertEqual(sources.source_stem(self.tmp / "a!b.nrrd"), "a!b")
        self.assertEqual(sources.source_stem("zenodo:1/c.zip!d.nii.gz"), "d")

    def test_a_missing_path_is_refused_in_one_line(self):
        rc, printed, err = self.get(self.tmp / "nope", "-o", self.tmp / "x.nii.gz")
        self.assertEqual((rc, printed), (2, []))
        self.assertEqual(err.splitlines(),
                         [f"haversack: not a remote source and not a local path: {self.tmp / 'nope'}"])

    def test_a_refused_series_is_refused_in_segments_words_alone(self):
        """io.convert refuses a series `segment` would refuse (a missing slice, a tilted gantry)
        with `segment`'s own error. A fetched series refused that way can still be had as a
        directory - `-o <dir>/` copies it as fetched - and saying so is the fix for it. A local
        series is a directory already, the user's own: that copy would be of their folder, which
        `segment` refuses just the same, so nothing may be appended for a local source."""
        refusal = "non-uniform slice spacing (steps 2.500-5.000 mm): missing or duplicate slices"

        def refuse(src, dst, **kw):
            raise InputError(refusal)
        with mock.patch.object(io, "convert", refuse):
            rc, printed, err = self.get(self.series, "-o", self.tmp / "x.nii.gz")
        self.assertEqual((rc, printed, err.splitlines()), (2, [], [f"haversack: {refusal}"]))


class NeverIntoItself(_Get):
    """Once a local path is written, -o can name the source itself. Refused in one line, and
    decided by the filesystem (`os.path.samefile`): a symlink, a hard link or - on APFS and
    FAT - a spelling in another case reaches the same bytes."""

    def setUp(self):
        super().setUp()
        self.series = dicom_series(self.tmp / "cases" / "ct_series")
        self.scan = nrrd(self.tmp / "cases" / "scan.nrrd", 7)
        self.before = self.scan.read_bytes()

    def assert_refused(self, *argv):
        rc, printed, err = self.get(*argv)
        self.assertEqual((rc, printed), (2, []))
        self.assertEqual(len(err.splitlines()), 1, err)
        self.assertIn("into itself", err)

    def test_a_file_copied_onto_itself(self):
        """Ends in shutil's SameFileError, a traceback, without the refusal."""
        self.assert_refused(self.scan, "-o", f"{self.scan.parent}/")
        self.assertEqual(self.scan.read_bytes(), self.before)

    def test_a_file_converted_onto_itself(self):
        """Would rewrite the input in place, compressed, through SimpleITK's header."""
        self.assert_refused(self.scan, "-o", self.scan)
        self.assertEqual(self.scan.read_bytes(), self.before)

    def test_a_folder_copied_into_itself(self):
        """Nests a copy of the folder inside itself, one level deeper on every run."""
        self.assert_refused(self.series, "-o", f"{self.series}/")
        self.assertFalse((self.series / "ct_series").exists())

    def test_a_folder_copied_into_itself_from_below(self):
        """A relative -o from inside the source: only its absolute path shows where it lands."""
        below = self.series / "sub"
        below.mkdir()
        self.chdir(below)
        self.assert_refused(self.series, "-o", "out/")
        self.assertFalse((below / "out").exists())

    def test_the_same_file_by_another_name(self):
        alias = self.tmp / "alias"
        alias.symlink_to(self.scan.parent, target_is_directory=True)
        self.assert_refused(self.scan, "-o", f"{alias}/")
        self.assertEqual(self.scan.read_bytes(), self.before)

    def test_writing_beside_the_source_is_not_into_it(self):
        """Not a ban on the source's own folder: a conversion may land in it."""
        dst = self.series / "scan.nii.gz"
        rc, printed, _ = self.get(self.series, "-o", dst)
        self.assertEqual((rc, printed), (0, [str(dst)]))


class GetDoesWhatItIsAsked(_Get):
    """Three more ways `get` exited 0 having done less than it was asked, found beside the
    local-path fix (2026-09-11)."""

    def test_format_alone_converts_one_source_into_the_current_directory(self):
        """As it always did for several; for one it printed the cache path and converted nothing."""
        self.chdir(self.tmp)
        rc, printed, _ = self.get("fake:case1", "--format", "nifti")
        self.assertEqual((rc, printed), (0, ["case1.nii.gz"]))
        self.assertTrue((self.tmp / "case1.nii.gz").is_file())

    def test_no_cache_holds_for_a_raw_copy_into_a_directory(self):
        """Every raw copy fetched into the cache and left it there; only conversions honored it."""
        rc, printed, _ = self.get("fake:case1", "--no-cache", "-o", f"{self.tmp / 'raw'}/")
        self.assertEqual((rc, printed), (0, [str(self.tmp / "raw" / "case1.nrrd")]))
        self.assertFalse((self.tmp / "cache" / "inputs" / "fake").exists())

    def test_no_cache_holds_for_a_raw_copy_to_a_file(self):
        rc, printed, _ = self.get("fake:case1", "--no-cache", "-o", self.tmp / "case1.bin")
        self.assertEqual((rc, printed), (0, [str(self.tmp / "case1.bin")]))
        self.assertFalse((self.tmp / "cache" / "inputs" / "fake").exists())

    def test_no_cache_holds_for_a_batch_of_raw_copies(self):
        out = self.tmp / "raw"
        rc, printed, _ = self.get("fake:case1", "fake:case2", "--no-cache", "-o", out)
        self.assertEqual((rc, printed), (0, [str(out / "case1.nrrd"), str(out / "case2.nrrd")]))
        self.assertFalse((self.tmp / "cache" / "inputs" / "fake").exists())

    def test_two_sources_onto_one_name_fail_the_second(self):
        """`a/scan.nrrd b/scan.nrrd --format nifti -o out` wrote out/scan.nii.gz twice, b over a,
        and exited 0 - the ordinary layout of a folder of cases, once local paths are written."""
        a, b = nrrd(self.tmp / "a" / "scan.nrrd", 1), nrrd(self.tmp / "b" / "scan.nrrd", 2)
        out = self.tmp / "out"
        rc, printed, err = self.get(a, b, "--format", "nifti", "-o", out)
        self.assertEqual((rc, printed), (1, [str(out / "scan.nii.gz")]))
        self.assertTrue((voxels(out / "scan.nii.gz") == 1).all())            # a's, kept
        self.assertIn(f"FAILED {b}: {out / 'scan.nii.gz'} was already written for {a}", err)

    def test_names_differing_only_in_case_are_one_name(self):
        """On APFS and FAT `Scan.nii.gz` IS `scan.nii.gz`: a check by `==` lets b over a."""
        a, b = nrrd(self.tmp / "a" / "scan.nrrd", 1), nrrd(self.tmp / "b" / "Scan.nrrd", 2)
        rc, printed, _ = self.get(a, b, "--format", "nifti", "-o", self.tmp / "out")
        self.assertEqual((rc, len(printed)), (1, 1))
        self.assertTrue((voxels(self.tmp / "out" / "scan.nii.gz") == 1).all())

    def test_two_series_are_not_copied_into_one_folder(self):
        """A raw copy merged the second series into the first's folder, file over file."""
        a = dicom_series(self.tmp / "a" / "series")
        b = dicom_series(self.tmp / "b" / "series", dz=3.0)
        raw = self.tmp / "raw"
        rc, printed, _ = self.get(a, b, "-o", raw)
        self.assertEqual((rc, printed), (1, [str(raw / "series")]))
        self.assertEqual(sorted(p.read_bytes() for p in (raw / "series").iterdir()),
                         sorted(p.read_bytes() for p in a.iterdir()))


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
