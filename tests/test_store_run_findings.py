"""Two findings from running `haversack serve --result-store file://...` (2026-09-26).

1. Every NIfTI upload put GDCM's "No Series were found" into the server log: the server asks
   `io.dicom_series_ids` of every folder an upload is staged in, and GDCM prints two ITK
   warnings on the way to the right answer (no series) for a folder with no DICOM in it.
   GDCM is now asked only when a file there could be DICOM.
2. The startup lines never said which result store a writer publishes into.
"""
import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from test_several_series import THREE, write_series


def _series_ids_in_a_child(folder):
    """``dicom_series_ids(folder)`` in a subprocess, with what ITK printed: its warnings are
    C++ writes to fd 2, which no Python-level capture sees."""
    code = textwrap.dedent(f"""
        from haversack.io import dicom_series_ids
        print(repr(dicom_series_ids({str(folder)!r})))
    """)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(Path(__file__).parent.parent / "src"), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    return eval(r.stdout.strip()), r.stderr                  # noqa: S307 - our own repr


class TheSeriesCheckIsQuiet(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def test_a_folder_holding_a_nifti_is_no_series_and_says_nothing(self):
        d = self.tmp / "staged"
        d.mkdir()
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16)), str(d / "ct.nii.gz"))
        ids, err = _series_ids_in_a_child(d)
        self.assertEqual(ids, [])
        self.assertNotIn("No Series", err)
        self.assertNotIn("WARNING", err)

    def test_reading_a_staged_nifti_folder_says_nothing(self):
        """How a single-file upload arrives: a folder holding the one file, read as it."""
        d = self.tmp / "staged2"
        d.mkdir()
        sitk.WriteImage(sitk.GetImageFromArray(np.arange(120, dtype=np.int16).reshape(4, 5, 6)),
                        str(d / "ct.nii.gz"))
        code = f"from haversack import io; print(io.read_image({str(d)!r}).GetSize())"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).parent.parent / "src"), os.environ.get("PYTHONPATH", "")])}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "(6, 5, 4)")
        self.assertNotIn("No Series", r.stderr)

    def test_a_dicom_series_is_still_found(self):
        ids, _ = _series_ids_in_a_child(write_series(self.tmp / "s", 3, THREE, value=1))
        self.assertEqual(ids, [THREE])

    def test_a_dicom_series_beside_a_nifti_is_still_found(self):
        d = write_series(self.tmp / "mixed", 3, THREE, value=1)
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16)), str(d / "a.nii.gz"))
        ids, _ = _series_ids_in_a_child(d)
        self.assertEqual(ids, [THREE])

    def test_dicom_without_its_preamble_still_reaches_gdcm(self):
        """pydicom refuses a file with no preamble unless forced; GDCM reads it. The check
        must not skip such a folder, or its series would read as none."""
        import pydicom
        d = write_series(self.tmp / "bare", 3, THREE, value=1)
        for f in d.iterdir():
            f.write_bytes(f.read_bytes()[132:])           # drop the preamble and "DICM"
            with self.assertRaises(Exception):            # plainly, pydicom refuses it
                pydicom.dcmread(f, stop_before_pixels=True)
        from haversack import io
        self.assertTrue(io._may_hold_dicom(d))

    def test_without_pydicom_gdcm_is_asked_as_before(self):
        from haversack import io
        with mock.patch.dict(sys.modules, {"pydicom": None}):
            self.assertTrue(io._may_hold_dicom(self.tmp))


class TheStartupLinesNameTheStore(unittest.TestCase):
    """`main_serve` all the way to its prints, with uvicorn stood in for (CI installs none)."""

    def _serve(self, store):
        import io as _io
        import tempfile

        from haversack import cache_admin, serve
        fake = types.ModuleType("uvicorn")

        class Config:
            def __init__(self, *a, **k):
                pass

            def bind_socket(self):
                return mock.Mock()

        class Server:
            def __init__(self, config):
                pass

            def run(self, sockets=None):
                pass
        fake.Config, fake.Server = Config, Server
        out = _io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                token="t", no_token=False, host="127.0.0.1", port=0, device="cpu",
                dtype="auto", model_root=None, cache_models=1, workdir=tmp,
                cache_dir=str(Path(tmp) / "cache"), no_result_cache=False, max_pending=4,
                keep_finished=4, result_store=store)
            with mock.patch.dict(sys.modules, {"uvicorn": fake}), \
                    mock.patch.object(serve, "create_app", lambda *a, **k: object()), \
                    mock.patch.object(serve, "LocalExecutor"), \
                    mock.patch("haversack.segmenter.Segmenter"), \
                    mock.patch.object(cache_admin, "check_cache_root"), \
                    mock.patch.object(cache_admin, "serve_token_path",
                                      lambda port: Path(tmp) / "token.json"), \
                    mock.patch("sys.stdout", out), mock.patch("sys.stderr", _io.StringIO()):
                serve.main_serve(args)
            return out.getvalue(), str(Path(tmp) / "cache")

    def test_the_store_and_its_local_copy_are_named(self):
        text, cache = self._serve("file:///srv/store")
        self.assertIn(f"result store: file:///srv/store (local copy {cache})", text)

    def test_credentials_in_the_url_are_not_printed(self):
        text, _ = self._serve("s3://KEY:SECRET@minio.local:9000/bucket/prefix")
        self.assertIn("result store: s3://minio.local:9000/bucket/prefix", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("KEY", text)

    def test_no_store_no_line(self):
        text, _ = self._serve(None)
        self.assertNotIn("result store:", text)
