"""`segment`'s batch never writes two inputs onto one output (2026-09-11).

Inference is faked (`pipeline.segment` patched, as `test_get_and_cache.py` does) and so is a
remote fetch: nothing leaves the machine and no model runs.
"""
import os
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from haversack import cli, pipeline, sources
from haversack.errors import InputError


class _Labels:
    """What `pipeline.segment` returns, as far as the batch reads it. `save` writes which
    input it was, so a test can tell whose labels landed."""
    timings = {}
    grid = type("G", (), {"shape": (2, 2, 2)})()
    schema = type("S", (), {"names": ["liver"]})()

    def __init__(self, image):
        self.image = image
        self.provenance = {}

    def save(self, path):
        Path(path).write_text(f"labels of {self.image}")
        return path

    def present(self):
        return {1: "liver"}


class NeverTwoInputsOntoOneOutput(unittest.TestCase):
    """A batch names each output `<stem>_<task><ext>` in -o. Until 2026-09-11 two inputs sharing
    a stem - `a/scan.nii.gz` and `b/scan.nii.gz`, the ordinary layout of a folder of cases - were
    both written to `out/scan_total_fast.seg.nrrd`, and the run exited 0 with only b's labels
    there. The names depend on the specs alone, so the pair is refused before anything is
    fetched or inferred, in one line naming the fix. Names are compared as the filesystem
    compares them: on APFS or FAT `Scan` and `scan` are one file (AGENTS.md)."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.out = self.tmp / "out"
        self.inferred = []

        def segment(image, task, **kw):
            self.inferred.append(image)
            return _Labels(image)
        # a cache of the test's own: a probe must never reach the real one (AGENTS.md)
        for p in (mock.patch.object(pipeline, "segment", segment),
                  mock.patch.dict(os.environ, {"HAVERSACK_CACHE_DIR": str(self.tmp / "cache")})):
            p.start()
            self.addCleanup(p.stop)

    def inputs(self, *names):
        """Create each (empty) file under the test's directory; their paths, as given on argv."""
        paths = []
        for n in names:
            p = self.tmp / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            paths.append(str(p))
        return paths

    def segment(self, *specs):
        """``(exit status, stderr lines)`` of a quiet batch into ``self.out``."""
        err = StringIO()
        with redirect_stdout(StringIO()), redirect_stderr(err):
            rc = cli.main(["segment", *specs, "--task", "total_fast", "--format", "seg.nrrd",
                           "-o", str(self.out), "--quiet"])
        return rc, err.getvalue().splitlines()

    def assert_refused(self, specs, first, second, name):
        """Refused in one line - the output, both inputs, the fix - with nothing run, not even
        the first of the pair, and the output directory not made."""
        rc, err = self.segment(*specs)
        self.assertEqual(rc, 2, err)
        self.assertEqual(err, [f"haversack: {self.out / name} would be written for both {first} "
                               f"and {second}; segment {second} on its own, with -o naming "
                               "another file"])
        self.assertEqual(self.inferred, [])
        self.assertFalse(self.out.exists())

    def test_distinct_names_all_run(self):
        """The harness itself: with no collision every input is inferred and written, so a
        refusal below is the check's doing and not the fake's."""
        a, b = self.inputs("a/scan.nii.gz", "b/scan2.nii.gz")
        rc, err = self.segment(a, b)
        self.assertEqual((rc, err), (0, []))
        self.assertEqual(self.inferred, [a, b])
        self.assertEqual((self.out / "scan_total_fast.seg.nrrd").read_text(), f"labels of {a}")
        self.assertEqual((self.out / "scan2_total_fast.seg.nrrd").read_text(), f"labels of {b}")

    def test_two_inputs_sharing_a_stem(self):
        a, b = self.inputs("a/scan.nii.gz", "b/scan.nii.gz")
        self.assert_refused([a, b], a, b, "scan_total_fast.seg.nrrd")

    def test_one_case_in_two_formats(self):
        """The extension is not part of the stem: `scan.nii.gz` and `scan.nrrd` are one name."""
        a, b = self.inputs("scan.nii.gz", "scan.nrrd")
        self.assert_refused([a, b], a, b, "scan_total_fast.seg.nrrd")

    def test_names_differing_only_in_case(self):
        """A check by `==` passes these, and APFS then writes b's labels over a's."""
        a, b = self.inputs("a/Scan.nii.gz", "b/scan.nii.gz")
        self.assert_refused([a, b], a, b, "Scan_total_fast.seg.nrrd")

    def test_names_differing_only_in_unicode_normalization(self):
        """APFS folds normalization too: `café` composed and `café` decomposed are one file."""
        nfc, nfd = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
        self.assertNotEqual(nfc, nfd)
        a, b = self.inputs(f"a/{nfc}.nii.gz", f"b/{nfd}.nii.gz")
        self.assert_refused([a, b], a, b, f"{nfc}_total_fast.seg.nrrd")

    def test_remote_inputs_are_refused_before_either_is_fetched(self):
        """Two Zenodo records each holding a `scan.nii.gz`, a local case between them: the pair
        need not be adjacent, and neither is downloaded."""
        (c,) = self.inputs("c.nii.gz")
        fetched = []

        def materialize(spec, **kw):
            fetched.append(spec)
            return self.tmp / "c.nii.gz"
        specs = ["zenodo:111/scan.nii.gz", c, "zenodo:222/scan.nii.gz"]
        with mock.patch.object(sources, "materialize", materialize), \
                mock.patch.object(sources, "input_record", lambda spec, **kw: {}):
            self.assert_refused(specs, specs[0], specs[2], "scan_total_fast.seg.nrrd")
        self.assertEqual(fetched, [])

    def test_a_lean_install_hears_about_the_pair_not_about_torch(self):
        """One of the cheap mistakes, so it is found before the inference stack is asked for:
        an install without torch is told what is wrong with the run it asked for."""
        a, b = self.inputs("a/scan.nii.gz", "b/scan.nii.gz")

        def no_stack(task):
            raise InputError("segment needs the inference stack: pip install 'haversack[torch]'")
        with mock.patch.object(cli, "_need_inference_stack", no_stack):
            self.assert_refused([a, b], a, b, "scan_total_fast.seg.nrrd")


if __name__ == "__main__":
    unittest.main()
