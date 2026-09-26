"""The input cache on the content-addressed store (step 6, docs/cache-consolidation.md).

Each test pins a guarantee the SeriesCache/ContentStore protocol gave its callers, or one the
new store adds: a fetch stores its input copy (or its original, refused) as blobs + one ref;
a job's view has the old entry's layout and survives eviction; eviction and the sweep refuse
what they cannot account for; keys differing only in case never share a ref.
"""
import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

from haversack import inputstore
from haversack.content import is_digest
from haversack.inputstore import InputGone, InputStore, key_for, ref_name

from test_several_series import THREE, write_series


def _nifti(path, value=7, shape=(6, 5, 4)):
    img = sitk.GetImageFromArray(np.full(shape, value, np.int16))
    img.SetSpacing((0.8, 0.9, 2.0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return Path(path)


class _Fetch:
    """A source's fetch: build <entry>/series and a record beside it; counts its calls."""

    def __init__(self, make):
        self.make, self.calls = make, 0

    def __call__(self, identity, entry, credentials=None):
        self.calls += 1
        series = Path(entry) / "series"
        series.mkdir(parents=True)
        self.make(series)
        (Path(entry) / ".input.json").write_text(json.dumps(
            {"kind": "test", "identity": identity, "content": {"digest": "sha256:" + "a" * 64}}))
        return series


class _Base(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.fetch = _Fetch(lambda d: _nifti(d / "ct.nii.gz"))
        self.store = InputStore(self.tmp / "inputs", self.fetch, grace_s=0)

    def job(self, name="job"):
        d = self.tmp / "jobs" / name
        d.mkdir(parents=True, exist_ok=True)
        return d


class FetchAndRead(_Base):
    def test_a_fetch_stores_the_copy_and_the_view_reads_as_the_original(self):
        path = self.store.get_or_fetch("idc:abc", self.job())
        self.assertEqual(path.name, "input.duckn.zip")
        self.assertEqual(path.parent.name, "decoded")
        from haversack import io
        got = sitk.GetArrayFromImage(io.read_image(path))
        self.assertTrue((got == 7).all() and got.shape == (6, 5, 4))
        self.assertEqual(json.loads((path.parent.parent / ".input.json").read_text())["identity"],
                         "idc:abc")
        doc = self.store.ref("idc:abc")
        self.assertEqual(list(doc["files"]), ["decoded/input.duckn.zip"])   # the original is gone
        self.assertEqual(self.store.record("idc:abc")["kind"], "test")

    def test_a_second_use_fetches_nothing(self):
        self.store.get_or_fetch("idc:abc", self.job("a"))
        self.store.get_or_fetch("idc:abc", self.job("b"))
        self.assertEqual(self.fetch.calls, 1)

    def test_an_input_the_reader_refuses_keeps_its_original(self):
        store = InputStore(self.tmp / "i2", _Fetch(lambda d: (d / "notes.txt").write_text("x")),
                           grace_s=0)
        path = store.get_or_fetch("s3:bucket/notes", self.job())
        self.assertEqual(path.name, "series")
        self.assertEqual((path / "notes.txt").read_text(), "x")

    def test_what_is_not_stored_is_gone(self):
        with self.assertRaises(InputGone):
            self.store.materialize("idc:nothing", self.job())
        self.assertFalse(self.store.has("idc:nothing"))
        self.assertIsNone(self.store.record("idc:nothing"))

    def test_a_blob_swept_under_a_ref_is_fetched_again_once(self):
        self.store.ensure("idc:abc")
        for b in self.store.ref("idc:abc")["files"].values():
            (self.store.store.root / self.store.blobs.path(b["digest"])).unlink()
        self.assertFalse(self.store.has("idc:abc"))
        self.store.get_or_fetch("idc:abc", self.job())
        self.assertEqual(self.fetch.calls, 2)

    def test_a_reader_version_change_makes_it_absent(self):
        self.store.ensure("idc:abc")
        with mock.patch("haversack.input_copy.READER_VERSION", 99):
            self.assertFalse(self.store.has("idc:abc"))
        self.assertTrue(self.store.has("idc:abc"))

    def test_forget_forgets_it(self):
        self.store.ensure("idc:abc")
        self.assertTrue(self.store.forget("idc:abc"))
        self.assertFalse(self.store.has("idc:abc"))
        self.store.ensure("idc:abc")
        self.assertEqual(self.fetch.calls, 2)


class Uploads(_Base):
    def test_a_file_is_stored_under_its_digest_and_read_as_its_copy(self):
        src = _nifti(self.tmp / "up" / "scan.nii.gz", value=3)
        digest = self.store.put_file(src)
        self.assertTrue(is_digest(digest) and self.store.has(digest))
        path = self.store.materialize(digest, self.job())
        self.assertEqual(path.name, "input.duckn.zip")

    def test_a_file_that_is_no_medical_image_is_refused_and_nothing_stored(self):
        from haversack.content import UnidentifiedContent
        src = self.tmp / "up" / "blob.bin"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"\x00not an image")
        with self.assertRaises(UnidentifiedContent):
            self.store.put_file(src)
        self.assertEqual(self.store.blobs.entries(), [])
        self.assertEqual(self.store._refs(), [])

    def test_an_upload_the_reader_refuses_is_handed_back_as_the_one_file(self):
        """No copy (the reader refused it): the view is series/<file>, and the reader gets the
        FILE - a directory would make SimpleITK read it as a DICOM series."""
        src = _nifti(self.tmp / "up" / "scan.nii.gz", value=4)
        with mock.patch("haversack.input_copy.transcode", return_value=None):
            digest = self.store.put_file(src)
        path = self.store.materialize(digest, self.job())
        self.assertTrue(path.is_file(), path)
        self.assertEqual(path.parent.name, "series")

    def test_an_uploads_key_names_the_reader_version(self):
        digest = self.store.put_file(_nifti(self.tmp / "up" / "v.nii.gz"))
        with mock.patch("haversack.input_copy.READER_VERSION", 99):
            self.assertFalse(self.store.has(digest))
        self.assertTrue(self.store.has(digest))

    def test_a_series_is_a_tree_and_read_as_its_copy(self):
        d = write_series(self.tmp / "up" / "dcm", 3, THREE, value=5)
        digest = self.store.put_dir(d)
        self.assertTrue(digest.startswith("sha256-tree:"))
        path = self.store.materialize(digest, self.job())
        from haversack import io
        self.assertTrue((sitk.GetArrayFromImage(io.read_image(path)) == 5).all())

    def test_a_folder_of_one_file_is_that_file(self):
        d = self.tmp / "up" / "one"
        _nifti(d / "x.nii.gz")
        self.assertEqual(self.store.put_dir(d), self.store.put_file(d / "x.nii.gz"))

    def test_a_claimed_digest_is_checked_not_trusted(self):
        from haversack.content import DigestMismatch
        src = _nifti(self.tmp / "up" / "s.nii.gz")
        with self.assertRaises(DigestMismatch):
            self.store.put_file(src, expect="sha256:" + "0" * 64)


class Eviction(_Base):
    def _fill(self, n):
        for i in range(n):
            f = _Fetch(lambda d, i=i: _nifti(d / "ct.nii.gz", value=i + 1))
            self.store.ensure(f"idc:{i}", fetch=f)
            time.sleep(0.01)                    # distinct mtimes: the LRU's clock

    def test_the_least_recently_used_go_first_and_their_blobs_with_them(self):
        self._fill(3)
        one = self.store.ref("idc:0")["bytes"]
        self.store.budget = one * 2 + 1
        self.store.touch("idc:0")               # used last: stays
        self.store.evict()
        self.assertTrue(self.store.has("idc:0"))
        self.assertFalse(self.store.has("idc:1"))
        self.assertTrue(self.store.has("idc:2"))
        kept = {b["digest"] for k in ("idc:0", "idc:2")
                for b in self.store.ref(k)["files"].values()}
        self.assertEqual({b["digest"] for b in self.store.blobs.entries()}, kept)

    def test_a_jobs_view_survives_the_eviction_of_its_input(self):
        for linked in (True, False):
            with self.subTest(linked=linked):
                self.store.forget("idc:abc")
                ctx = (mock.patch("os.link", side_effect=OSError(45, "not supported"))
                       if not linked else mock.patch.object(os, "getpid", os.getpid))
                with ctx:
                    path = self.store.get_or_fetch("idc:abc", self.job(f"v{linked}"))
                self.store.budget = 0
                self.store.evict()
                self.store.budget = 8 << 30
                self.assertFalse(self.store.has("idc:abc"))
                self.assertEqual(self.store.blobs.entries(), [])
                from haversack import io
                self.assertTrue((sitk.GetArrayFromImage(io.read_image(path)) == 7).all())

    def test_a_ref_that_cannot_be_read_stops_the_sweep(self):
        self._fill(2)
        (self.store.store.root / "inputs" / "junk.json").write_text("{not json")
        before = self.store.blobs.entries()
        self.store.forget("idc:0")             # its blobs are now unreferenced
        self.store.evict()
        self.assertEqual(self.store.blobs.entries(), before)

    def test_young_unreferenced_blobs_wait_out_the_grace(self):
        store = InputStore(self.tmp / "g", self.fetch, grace_s=3600)
        store.ensure("idc:abc")
        store.forget("idc:abc")
        store.evict()
        self.assertEqual(len(store.blobs.entries()), 1)


class Keys(_Base):
    def test_keys_differing_only_in_case_never_share_a_ref(self):
        a, b = ref_name(key_for("s3:bucket/Scan")), ref_name(key_for("s3:bucket/scan"))
        self.assertNotEqual(a.lower(), b.lower())

    def test_a_long_key_is_hashed_and_still_found(self):
        ident = "s3:bucket/" + "x" * 400
        self.assertTrue(ref_name(key_for(ident)).startswith("inputs/h_"))
        self.store.ensure(ident)
        self.assertTrue(self.store.has(ident))

    def test_a_fetched_key_names_the_epoch_and_an_upload_does_not(self):
        self.assertTrue(key_for("idc:x").startswith("e"))
        self.assertTrue(key_for("sha256:" + "1" * 64).startswith("r"))

    def test_a_ref_naming_a_path_outside_its_view_is_not_an_input(self):
        self.store.ensure("idc:abc")
        f = self.store._ref_file(key_for("idc:abc"))
        doc = json.loads(f.read_text())
        doc["files"] = {"../../escape": next(iter(doc["files"].values()))}
        f.write_text(json.dumps(doc))
        self.assertIsNone(self.store.ref("idc:abc"))
        with self.assertRaises(InputGone):
            self.store.materialize("idc:abc", self.job())
        self.assertFalse((self.tmp / "jobs" / "escape").exists())

    def test_a_ref_for_another_key_is_not_this_input(self):
        self.store.ensure("idc:abc")
        f = self.store._ref_file(key_for("idc:abc"))
        doc = json.loads(f.read_text())
        doc["key"] = key_for("idc:other")
        f.write_text(json.dumps(doc))
        self.assertIsNone(self.store.ref("idc:abc"))


class OneFetchPerHost(_Base):
    def test_concurrent_jobs_fetch_one_key_once(self):
        slow = _Fetch(lambda d: (time.sleep(0.3), _nifti(d / "ct.nii.gz")))
        errors = []

        def use(i):
            try:
                self.store.get_or_fetch("idc:abc", self.job(f"t{i}"), fetch=slow)
            except Exception as e:              # noqa: BLE001
                errors.append(e)
        ts = [threading.Thread(target=use, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errors, [])
        self.assertEqual(slow.calls, 1)

    def test_a_cancelled_wait_raises_from_the_check(self):
        import fcntl
        key = key_for("idc:abc")
        import hashlib
        stripe = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % 256
        self.store.locks.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.store.locks / f"{stripe:03d}", os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)          # another process's fetch, as far as it knows
        try:
            def check():
                raise RuntimeError("cancelled")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.store.ensure("idc:abc", check=check)
        finally:
            os.close(fd)
