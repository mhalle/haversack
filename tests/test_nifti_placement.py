"""A NIfTI with both transforms is placed by its sform, as nibabel and duckn place it (2026-09-30,
the user's call). SimpleITK 2.5.6 took the qform beside an MNI or aligned sform; haversack's own
fallback for ITK-refused cosines already placed by nibabel's affine, so one file could be placed
by either rule. These hold every read to nibabel's ``get_best_affine``."""

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
sitk = pytest.importorskip("SimpleITK")

from haversack.errors import InputError  # noqa: E402
from haversack.io import read_image  # noqa: E402

LPS = np.diag([-1.0, -1.0, 1.0])


def _oblique(origin=(-118.40123, -97.6123456, 63.25), spacing=(0.9375123, 0.9375, 3.3)):
    """An affine as a scanner converter writes it: rotated 13.7 degrees, offsets of ~100 mm."""
    th = np.deg2rad(13.7)
    r = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    aff = np.eye(4)
    aff[:3, :3] = r @ np.diag(spacing)
    aff[:3, 3] = origin
    return aff


def _corners(image):
    """Every corner voxel's physical point, in RAS."""
    size = image.GetSize()
    pts = []
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                pts.append(LPS @ np.array(image.TransformIndexToPhysicalPoint((i, j, k))))
    return np.array(pts)


def _expected(aff, shape):
    return np.array([(aff @ [i, j, k, 1])[:3]
                     for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                     for k in (0, shape[2] - 1)])


class TestSformFirst(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _save(self, sform, scode, qform, qcode, name="a.nii", shape=(5, 4, 3), klass=None):
        img = (klass or nib.Nifti1Image)(np.arange(np.prod(shape), dtype=np.int16).reshape(shape),
                                         sform if sform is not None else qform)
        img.set_sform(sform, code=scode)
        img.set_qform(qform, code=qcode)
        p = self.tmp / name
        nib.save(img, str(p))
        return p

    def _differing(self):
        s = _oblique()
        q = s.copy()
        q[:3, 3] += [5.0, -3.0, 2.0]
        return s, q

    def test_an_mni_or_aligned_sform_places_the_file_not_the_qform(self):
        s, q = self._differing()
        for code in (2, 3, 4, 5):
            p = self._save(s, code, q, 1, name=f"c{code}.nii")
            assert not np.allclose(sitk.ReadImage(str(p)).GetOrigin(), LPS @ s[:3, 3], atol=1e-3), \
                "SimpleITK no longer takes the qform here: this test no longer tests the change"
            img = read_image(p)
            np.testing.assert_allclose(_corners(img), _expected(nib.load(str(p)).affine, (5, 4, 3)),
                                       atol=1e-4, err_msg=f"sform code {code}")

    def test_a_scanner_sform_beside_a_template_qform_is_the_sform(self):
        s, q = self._differing()
        p = self._save(s, 1, q, 4)
        np.testing.assert_allclose(_corners(read_image(p)),
                                   _expected(nib.load(str(p)).affine, (5, 4, 3)), atol=1e-4)

    def test_the_sform_is_read_at_the_headers_precision(self):
        # SimpleITK prints srow to six significant digits (-118.401): the header's float32 is used
        s, q = self._differing()
        p = self._save(s, 4, q, 1)
        stored = nib.load(str(p)).header.get_sform()
        np.testing.assert_array_equal(np.array(read_image(p).GetOrigin()), LPS @ stored[:3, 3])

    def test_a_scanner_converters_file_reads_exactly_as_simpleitk_reads_it(self):
        # sform and qform agree up to the qform's float32 quaternion: nothing is touched
        s = _oblique()
        p = self._save(s, 1, s, 1)
        plain = sitk.ReadImage(str(p))
        img = read_image(p)
        self.assertEqual(img.GetOrigin(), plain.GetOrigin())
        self.assertEqual(img.GetSpacing(), plain.GetSpacing())
        self.assertEqual(img.GetDirection(), plain.GetDirection())

    def test_one_transform_reads_as_simpleitk_reads_it(self):
        s, q = self._differing()
        for name, args in {"s.nii": (s, 1, None, 0), "q.nii": (None, 0, q, 1)}.items():
            p = self._save(*args, name=name)
            plain = sitk.ReadImage(str(p))
            img = read_image(p)
            self.assertEqual(img.GetOrigin(), plain.GetOrigin(), name)
            self.assertEqual(img.GetDirection(), plain.GetDirection(), name)

    def test_a_singular_sform_is_passed_over_for_the_qform(self):
        s, q = self._differing()
        p = self._save(s, 4, q, 1)
        raw = bytearray(p.read_bytes())
        raw[280:328] = bytes(48)                       # srow_x, y, z all zero
        p.write_bytes(bytes(raw))
        img = read_image(p)
        np.testing.assert_allclose(_corners(img), _expected(q, (5, 4, 3)), atol=1e-3)

    def test_a_pair_named_by_its_image_is_placed_by_its_header(self):
        s, q = self._differing()
        p = self._save(s, 4, q, 1, name="pair.img", klass=nib.Nifti1Pair)
        self.assertTrue((self.tmp / "pair.hdr").is_file())
        np.testing.assert_allclose(_corners(read_image(p)), _expected(s, (5, 4, 3)), atol=1e-4)

    def test_a_compressed_file_is_placed_by_its_header(self):
        s, q = self._differing()
        p = self._save(s, 4, q, 1, name="a.nii.gz")
        np.testing.assert_allclose(_corners(read_image(p)), _expected(s, (5, 4, 3)), atol=1e-4)

    def test_a_big_endian_header_is_read(self):
        s, q = self._differing()
        img = nib.Nifti1Image(np.zeros((5, 4, 3), ">i2"), s)
        img.set_sform(s, code=4)
        img.set_qform(q, code=1)
        hdr = img.header.as_byteswapped(">")
        p = self.tmp / "be.nii"
        nib.save(nib.Nifti1Image(np.zeros((5, 4, 3), ">i2"), None, hdr), str(p))
        self.assertEqual(p.read_bytes()[:4], (348).to_bytes(4, "big"))
        np.testing.assert_allclose(_corners(read_image(p)), _expected(s, (5, 4, 3)), atol=1e-4)

    def test_a_sheared_sform_beside_a_qform_is_refused(self):
        # SimpleITK took the qform; the sform states a shear its image cannot hold
        q = _oblique()
        s = q.copy()
        s[:3, 2] += [0.0, 0.5, 0.0]
        p = self._save(s, 4, q, 1)
        sitk.ReadImage(str(p))                          # SimpleITK reads it (by the qform)
        with pytest.raises(InputError, match="sheared|orthonormal"):
            read_image(p)

    def test_a_dicom_or_other_file_is_not_asked(self):
        from haversack.io import _nifti1_transforms
        p = self.tmp / "x.nrrd"
        p.write_bytes(b"NRRD0004\n")
        self.assertIsNone(_nifti1_transforms(p))
        q = self.tmp / "analyze.hdr"
        img = nib.AnalyzeImage(np.zeros((2, 2, 2), np.int16), np.eye(4))
        nib.save(img, str(q))
        self.assertIsNone(_nifti1_transforms(q))

    def test_gzip_is_judged_by_its_bytes(self):
        from haversack.io import _nifti1_transforms
        s, q = self._differing()
        p = self._save(s, 4, q, 1)
        z = self.tmp / "b.nii.gz"
        z.write_bytes(gzip.compress(p.read_bytes()))
        u = self.tmp / "c.nii.gz"                       # named .gz, not compressed
        u.write_bytes(p.read_bytes())
        for f in (z, u):
            code, _, srow = _nifti1_transforms(f)
            self.assertEqual(code, 4)
            np.testing.assert_allclose(srow, s[:3], atol=1e-4)
