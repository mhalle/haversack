"""DICOM series geometry comes from IOP + IPP, never from SpacingBetweenSlices.

Regression for the 2026-08-24 finding: on Philips head-to-foot series (0018,0088)
is negative and ITK's GDCM layer takes the slice axis's sign from it, while origin
and stacking follow the spatially sorted list - the assembled volume then claims a
physical span the scan never occupies and the network sees a head-down body
(CPTAC-CCRCC a05fb365: 28 of 111 structures lost, including a kidney). haversack now
constructs series geometry from the geometric tags; these fixtures encode the
failure and the grid contract without shipping patient data.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from haversack import io
from haversack.errors import InputError


CT_SOP = "1.2.840.10008.5.1.4.1.1.2"


def write_series(dirpath, ipps, *, sbs=None, iop=(1, 0, 0, 0, 1, 0), pixel_spacing=(0.7, 0.5)):
    """Minimal CT slices: 4x4 int16, geometry per arguments, InstanceNumber
    descending with file index (the Philips pattern that exposed the bug)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    for i, ipp in enumerate(ipps):
        sop = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_SOP
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CT_SOP
        ds.SOPInstanceUID = sop
        ds.Modality = "CT"
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PatientID = "fixture"
        ds.PatientName = "fixture"
        ds.InstanceNumber = len(ipps) - i
        ds.ImagePositionPatient = [float(v) for v in ipp]
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.PixelSpacing = [float(v) for v in pixel_spacing]
        ds.SliceThickness = 2.0
        if sbs is not None:
            ds.SpacingBetweenSlices = float(sbs)
        ds.Rows = ds.Columns = 4
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.full((4, 4), i, dtype=np.int16).tobytes()
        ds.save_as(dirpath / f"slice_{i:03d}.dcm", enforce_file_format=True)
    return dirpath


def ascending(n=4, z0=31.0, dz=1.0):
    return [(-266.0, -138.0, z0 + i * dz) for i in range(n)]


def geometry(tmp_path, **kw):
    write_series(tmp_path, kw.pop("ipps"), **kw)
    _, geo, _ = io.read(tmp_path, reorient=False)
    return geo


def test_negative_sbs_gets_ipp_geometry(tmp_path):
    """THE regression: Philips negative (0018,0088) must not flip the slice axis."""
    geo = geometry(tmp_path, ipps=ascending(), sbs=-1.0)
    d = np.array(geo.direction_xyz).reshape(3, 3)
    assert np.allclose(d, np.eye(3)), d                      # +z, from the IPPs
    assert geo.origin_xyz == (-266.0, -138.0, 31.0)          # first sorted slice
    assert np.allclose(geo.spacing_zyx, (1.0, 0.7, 0.5))     # (dz, row, col)


def test_positive_sbs_same_answer(tmp_path):
    geo = geometry(tmp_path, ipps=ascending(), sbs=1.0)
    assert np.allclose(np.array(geo.direction_xyz).reshape(3, 3), np.eye(3))
    assert geo.origin_xyz == (-266.0, -138.0, 31.0)


def test_absent_sbs_same_answer(tmp_path):
    geo = geometry(tmp_path, ipps=ascending())
    assert np.allclose(np.array(geo.direction_xyz).reshape(3, 3), np.eye(3))


def test_descending_storage_reads_identically(tmp_path):
    """File creation order must not matter - GDCM sorts spatially either way."""
    geo = geometry(tmp_path, ipps=list(reversed(ascending())), sbs=-1.0)
    assert np.allclose(np.array(geo.direction_xyz).reshape(3, 3), np.eye(3))
    assert geo.origin_xyz == (-266.0, -138.0, 31.0)


def test_gantry_tilt_rejected(tmp_path):
    ipps = [(-266.0 + 0.4 * i, -138.0, 31.0 + i) for i in range(4)]   # x drifts with z
    write_series(tmp_path, ipps)
    with pytest.raises(InputError, match="tilt|normal"):
        io.read(tmp_path, reorient=False)


def test_gap_rejected(tmp_path):
    ipps = [(-266.0, -138.0, z) for z in (31.0, 32.0, 33.0, 35.0)]    # missing slice
    write_series(tmp_path, ipps)
    with pytest.raises(InputError, match="spacing|missing"):
        io.read(tmp_path, reorient=False)


def test_duplicate_position_rejected(tmp_path):
    ipps = [(-266.0, -138.0, z) for z in (31.0, 32.0, 32.0, 33.0)]
    write_series(tmp_path, ipps)
    with pytest.raises(InputError, match="[Dd]uplicate|monotonic"):
        io.read(tmp_path, reorient=False)


def test_single_slice_rejected(tmp_path):
    write_series(tmp_path, ascending(n=1))
    with pytest.raises(InputError, match="fewer than 2|3D"):
        io.read(tmp_path, reorient=False)


def test_reorient_lands_in_ras(tmp_path):
    """End to end through the canonical path: RAS out, regardless of the SBS sign."""
    write_series(tmp_path, ascending(), sbs=-1.0)
    _, geo, original = io.read(tmp_path, reorient=True)
    img = io.to_image(np.zeros(geo.shape_zyx, dtype=np.uint8), geo)
    assert io.orientation_of(img) == io.CANONICAL
    assert original == "LPS"                                 # the stored frame


def test_directory_with_one_image_file_reads_as_that_file(tmp_path):
    """Staged single-file sources arrive as a directory holding one file."""
    import numpy as np
    import pytest
    import SimpleITK as sitk

    from haversack.errors import InputError
    from haversack.io import read_image

    d = tmp_path / "series"
    d.mkdir()
    img = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16))
    img.SetSpacing((2.0, 3.0, 4.0))
    sitk.WriteImage(img, str(d / "vol.nii.gz"))
    got = read_image(d)
    assert got.GetSize() == (6, 5, 4) and got.GetSpacing() == (2.0, 3.0, 4.0)

    (d / "second.txt").write_text("x")      # two files: ambiguous, still an error
    with pytest.raises(InputError):
        read_image(d)


def test_near_orthonormal_affine_snaps_and_sheared_refuses(tmp_path):
    """Published datasets (TS training data) carry affines off by ~1e-4: snap
    via SVD polar and read. Genuinely sheared geometry still refuses."""
    import nibabel as nib
    import numpy as np
    import pytest

    from haversack.errors import InputError
    from haversack.io import read_image

    a = np.zeros((10, 12, 14), np.int16)
    a[3:7, 4:8, 5:9] = 100

    aff = np.diag([1.5, 1.5, 2.0, 1.0])
    aff[:3, :3] += np.array([[0, 5e-5, 0], [-4e-5, 0, 0], [0, 0, 0]])
    p1 = tmp_path / "near.nii.gz"
    nib.save(nib.Nifti1Image(a, aff), str(p1))
    # integration: read_image succeeds whichever side of ITK's own tolerance
    # the perturbation lands on
    img = read_image(p1)
    assert img.GetSize() == (10, 12, 14)
    assert np.allclose(img.GetSpacing(), (1.5, 1.5, 2.0), atol=1e-3)
    D = np.array(img.GetDirection()).reshape(3, 3)
    assert np.allclose(D @ D.T, np.eye(3), atol=1e-4)
    assert np.linalg.det(D) > 0.99                       # not mirrored
    # the snap path itself, forced: exactly orthonormal out, sign preserved
    from haversack.io import _read_with_snapped_affine
    snapped = _read_with_snapped_affine(p1, RuntimeError("orthonormal"))
    Ds = np.array(snapped.GetDirection()).reshape(3, 3)
    assert np.allclose(Ds @ Ds.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(Ds) > 0.999
    assert snapped.GetSize() == (10, 12, 14)

    sheared = np.diag([1.5, 1.5, 2.0, 1.0])
    sheared[0, 1] = 0.3                                  # real shear
    p2 = tmp_path / "sheared.nii.gz"
    nib.save(nib.Nifti1Image(a, sheared), str(p2))
    with pytest.raises(InputError, match="sheared|orthonormal"):
        read_image(p2)


class ConvertTakesTheSameGeometry(unittest.TestCase):
    """`get -o scan.nii.gz` converts through io.convert, which until 2026-09-11 read a series
    with a bare ImageSeriesReader and so never met the override above: the Philips series came
    out with its slice axis reversed about the first slice. SimpleITK cannot write this
    fixture - GDCM fills (0018,0088) from the image's own spacing, whatever the metadata says,
    and SimpleITK refuses a negative spacing - so it is pydicom's (write_series)."""

    def converts_to_where_its_slices_are(self, ipps):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "series"
            src.mkdir()
            write_series(src, ipps, sbs=-1.0)
            dst = Path(td) / "scan.nii.gz"
            io.convert(src, dst)
            out = sitk.ReadImage(str(dst))
            arr = sitk.GetArrayFromImage(out)
            for k in range(arr.shape[0]):         # write_series fills slice i with the value i
                np.testing.assert_allclose(out.TransformIndexToPhysicalPoint((0, 0, k)),
                                           ipps[int(arr[k, 0, 0])], atol=1e-4)
            ref = io.read_image(src)              # and it is what `segment` reads
            np.testing.assert_allclose(out.GetOrigin(), ref.GetOrigin(), atol=1e-4)
            np.testing.assert_allclose(out.GetDirection(), ref.GetDirection(), atol=1e-6)
            np.testing.assert_allclose(out.GetSpacing(), ref.GetSpacing(), atol=1e-6)
            np.testing.assert_array_equal(arr, sitk.GetArrayFromImage(ref))

    def test_a_negative_sbs_series_converts_to_where_its_slices_are(self):
        self.converts_to_where_its_slices_are(ascending(n=6))

    def test_file_order_does_not_change_that(self):
        """The same series with its files written top-first: GDCM sorts by position, so the
        order they were stored in must not matter."""
        self.converts_to_where_its_slices_are(list(reversed(ascending(n=6))))


# -- the input copy and a multi-frame file meet the same bug (2026-09-26) --------------------

@pytest.mark.parametrize("form", ["uncompressed", "zstd"])
@pytest.mark.parametrize("stored", ["ascending", "descending"])
def test_an_input_copy_of_a_negative_sbs_series_is_where_its_slices_are(tmp_path, monkeypatch,
                                                                       form, stored):
    """The copy is written whole (uncompressed) or a slab at a time (zstd), each with its own
    file list: every voxel slice must sit where its file's IPP says, and the per-slice tags on
    sample k must be THAT file's - a copy is read in place of the series, so a copy that
    reversed the slice axis (GDCM's sign, from the negative 0018,0088) or paired slice k with
    another file's tags would be the 2026-08-24 bug again, or a header contradicting its data."""
    import SimpleITK as sitk

    from haversack import input_copy as ic
    monkeypatch.setenv(ic.COMPRESSION_ENV, form)
    ipps = ascending(n=6)
    if stored == "descending":
        ipps = list(reversed(ipps))
    src = tmp_path / "series"
    src.mkdir()
    write_series(src, ipps, sbs=-1.0)                  # slice file i: pixels = i, IPP ipps[i]
    by_index = {int(pydicom.dcmread(f).PixelData[0]): pydicom.dcmread(f) for f in src.iterdir()}
    copy = ic.transcode(src, tmp_path / "entry")
    assert copy is not None and ic.stored_compression(copy) == form
    image = io.read_image(copy)
    arr = sitk.GetArrayFromImage(image)
    sop = ic.slice_tags(copy, "SOPInstanceUID")
    number = ic.slice_tags(copy, "InstanceNumber")
    for k in range(arr.shape[0]):
        i = int(arr[k, 0, 0])                          # which file this voxel slice came from
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((0, 0, k)), ipps[i],
                                   atol=1e-4)
        assert sop[k] == by_index[i].SOPInstanceUID
        assert number[k] == by_index[i].InstanceNumber
    ref = io.read_image(src)                           # and exactly what `segment` reads
    np.testing.assert_allclose(image.GetOrigin(), ref.GetOrigin(), atol=1e-6)
    np.testing.assert_allclose(image.GetDirection(), ref.GetDirection(), atol=1e-9)
    np.testing.assert_array_equal(arr, sitk.GetArrayFromImage(ref))


@pytest.mark.parametrize("zs", [[10, 12, 14, 16], [16, 14, 12, 10]], ids=["up", "down"])
@pytest.mark.parametrize("sbs", [2.0, -2.0, None], ids=["sbs+", "sbs-", "no-sbs"])
def test_a_multi_frame_file_is_placed_by_its_frames_whatever_the_spacing_sign(tmp_path, zs,
                                                                             sbs):
    """An Enhanced file can carry the same signed Spacing Between Slices, in its Pixel Measures.
    Measured 2026-09-26: SimpleITK places the frames by their own positions, so the sign does
    not flip them - this holds that, frame by frame, and holds haversack's frame check to
    accepting it (a reader that DID follow the sign would now be refused, never believed)."""
    import SimpleITK as sitk
    from test_dicom_tag_review_0926 import _enhanced
    f = _enhanced(tmp_path / "enh.dcm", zs, rescale=False)
    if sbs is not None:
        ds = pydicom.dcmread(f)
        ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SpacingBetweenSlices = sbs
        ds.save_as(f, enforce_file_format=True)
    image = io.read_image(f)                           # runs _check_frame_positions
    arr = sitk.GetArrayFromImage(image)
    for k in range(len(zs)):
        frame = int((arr[k, 0, 0] - 100) // 12)        # _enhanced: frame j's pixels start 100+12j
        assert image.TransformIndexToPhysicalPoint((0, 0, k))[2] == pytest.approx(zs[frame])


def test_the_frame_check_refuses_a_reader_that_follows_the_sign(tmp_path, monkeypatch):
    """Should a SimpleITK ever place frames by the spacing's sign rather than by their
    positions, the frame check must refuse - never let the reversed grid through."""
    import SimpleITK as sitk
    from test_dicom_tag_review_0926 import _enhanced
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16], rescale=False)
    image = sitk.ReadImage(str(f))
    flipped = sitk.Image(image)
    d = list(image.GetDirection())
    d[2], d[5], d[8] = -d[2], -d[5], -d[8]             # the slice axis reversed about frame 0
    flipped.SetDirection(d)
    with pytest.raises(InputError, match="frames"):
        io._check_frame_positions(f, flipped)
