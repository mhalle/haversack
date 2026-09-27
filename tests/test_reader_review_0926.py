"""The 2026-09-26 review of the READER: where GDCM hands over other values or geometry than a
file means, haversack corrects them or refuses, and nothing it stores or exports contradicts
its voxels (the user's rule, "fix the reader").

- an Enhanced file's frames each read with their own rescale, not the one GDCM applies to all;
- a series whose slices differ in pixel type reads in a type that holds every value;
- MONOCHROME1 and a Modality LUT are asked of every slice, not the first alone;
- an Enhanced MONOCHROME1 file with a functional-group rescale reads as its values;
- a folder holding one multi-frame file reads as that file;
- a palette, in stored units, is not stated beside rescaled voxels;
- frames of different orientation or pixel spacing, and an RTDOSE's uneven Grid Frame Offset
  Vector, are refused;
- every DICOM read is held against pydicom's own decode (first and last slice or frame);
- a truncated or corrupt gzip file is refused, on the whole read and the slab read.

Ground truth is computed here with pydicom alone (``_meaning``): the stored values it decodes
through the modality mapping the file states - never through haversack's code. Every test
failed on 4b5ecb2 (checked against a ``git archive`` of it first on PYTHONPATH) unless it says
it pins behavior that must not move.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pydicom = pytest.importorskip("pydicom")
pytest.importorskip("duckn")
pytest.importorskip("zarr")

from pydicom.dataset import Dataset  # noqa: E402
from pydicom.sequence import Sequence  # noqa: E402

from haversack import input_copy as ic  # noqa: E402
from haversack import input_stream  # noqa: E402
from haversack import io as nio  # noqa: E402
from haversack.errors import InputError  # noqa: E402

from test_dicom_tag_review_0926 import (_attrs, _edit, _enhanced, _everything,  # noqa: E402
                                        _malformed_kvp, _modality_lut, _monochrome1, _rescale,
                                        _unsigned, form, write_series)


# -- ground truth, from pydicom alone ---------------------------------------------------------

def _frame_rescale(ds, k):
    """The rescale frame ``k`` means: its own functional group, else the shared one, else the
    top level, else the identity."""
    for name in ("PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence"):
        groups = ds.get(name)
        if groups:
            item = groups[k] if name.startswith("PerFrame") else groups[0]
            pv = item.get("PixelValueTransformationSequence")
            if pv:
                return float(pv[0].RescaleSlope), float(pv[0].RescaleIntercept)
    return float(ds.get("RescaleSlope", 1)), float(ds.get("RescaleIntercept", 0))


def _meaning(f) -> np.ndarray:
    """(frames, rows, columns) of the values the DICOM file ``f`` means, per pydicom."""
    from pydicom.pixels import apply_modality_lut
    ds = pydicom.dcmread(f)
    px = ds.pixel_array
    n = int(ds.get("NumberOfFrames", 1) or 1)
    px = px.reshape(n, ds.Rows, ds.Columns)
    if "ModalityLUTSequence" in ds:
        return np.stack([np.asarray(apply_modality_lut(px[k], ds), np.float64) for k in range(n)])
    return np.stack([px[k] * s + i for k in range(n) for s, i in [_frame_rescale(ds, k)]])


def _series_meaning(folder: Path) -> np.ndarray:
    """The series' values, slices in order of position along the normal."""
    files = sorted(folder.iterdir())
    heads = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    iop = np.asarray(heads[0].ImageOrientationPatient, np.float64)
    normal = np.cross(iop[:3], iop[3:])
    order = np.argsort([np.dot(np.asarray(h.ImagePositionPatient, np.float64), normal)
                        for h in heads])
    return np.concatenate([_meaning(files[k]) for k in order])


def _values(image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.float64)


# -- 1. an Enhanced file's per-frame rescales --------------------------------------------------

def _per_frame(f: Path, rescales, *, shared=True) -> Path:
    """Give each frame of ``f`` its own Pixel Value Transformation; keep (or drop) the shared."""
    ds = pydicom.dcmread(f)
    if not shared:
        del ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence
    for frame, (slope, intercept) in zip(ds.PerFrameFunctionalGroupsSequence, rescales):
        pv = Dataset()
        pv.RescaleSlope, pv.RescaleIntercept, pv.RescaleType = slope, intercept, "HU"
        frame.PixelValueTransformationSequence = Sequence([pv])
    ds.save_as(f, enforce_file_format=True)
    return f


@pytest.mark.parametrize("shared", [True, False], ids=["with-shared", "per-frame-only"])
def test_each_frame_reads_with_its_own_rescale(tmp_path, shared):
    """GDCM applied the shared rescale (or, without one, the first frame's) to every frame:
    errors of thousands. Frame 2's slope is 0.5, so the values are not integers."""
    f = _per_frame(_enhanced(tmp_path / "enh.dcm", [10, 12, 14]),
                   [(2, -500), (1, -1024), (0.5, 7)], shared=shared)
    raw = _values(sitk.ReadImage(str(f)))
    truth = _meaning(f)
    assert np.abs(raw - truth).max() > 100                            # the premise
    image = nio.read_image(f)
    np.testing.assert_array_equal(_values(image), truth)
    assert image.GetPixelID() == sitk.sitkFloat64


def test_a_per_frame_rescale_is_copied_as_read(tmp_path, form):
    f = _per_frame(_enhanced(tmp_path / "enh.dcm", [10, 12, 14]), [(1, -1024), (2, 0), (1, 5)])
    copy_ = ic.transcode(f, tmp_path / "e")
    np.testing.assert_array_equal(_values(ic.read_copy(copy_)), _meaning(f))
    assert _attrs(copy_)["extensions"]["dicom"]["stored_values"] is False


def test_a_per_frame_correction_nothing_can_check_is_refused(tmp_path, monkeypatch):
    """Where pydicom cannot decode the file the correction is not believed unchecked."""
    f = _per_frame(_enhanced(tmp_path / "enh.dcm", [10, 12, 14]), [(1, -1024), (2, 0), (1, 5)])
    monkeypatch.setattr(nio, "_pydicom_meaning", lambda *a, **k: None)
    with pytest.raises(InputError, match="different rescales"):
        nio.read_image(f)


# -- 2. slices of one rescale in other pixel types ---------------------------------------------

def _pixels(i, ds, *, bits_allocated, bits_stored, signed, value, rescale=True):
    if not rescale:
        for k in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
            delattr(ds, k)
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = bits_allocated, bits_stored, bits_stored - 1
    ds.PixelRepresentation = int(signed)
    dt = {(8, 0): np.uint8, (8, 1): np.int8, (16, 0): np.uint16, (16, 1): np.int16}
    ds.PixelData = np.full((ds.Rows, ds.Columns), value,
                           dt[(bits_allocated, int(signed))]).tobytes()


@pytest.mark.parametrize("first, later", [
    (dict(bits_allocated=16, bits_stored=16, signed=False, value=40000, rescale=False),
     dict(bits_allocated=16, bits_stored=16, signed=True, value=-3000, rescale=False)),
    (dict(bits_allocated=16, bits_stored=12, signed=False, value=4000),
     dict(bits_allocated=16, bits_stored=16, signed=False, value=60000)),
    (dict(bits_allocated=8, bits_stored=8, signed=False, value=200),
     dict(bits_allocated=16, bits_stored=16, signed=False, value=60000)),
], ids=["unsigned-then-signed", "12-then-16-bits", "8-then-16-bits"])
def test_slices_in_other_pixel_types_read_every_value(tmp_path, first, later, form):
    """One rescale on every slice (the identity, or intercept -1024), so the rescale alone said
    'one type': the series reader read every slice in the first file's type, and the later ones
    wrapped - on the whole read and the slab read alike."""
    series = _edit(write_series(tmp_path / "s", n=3),
                   lambda i, ds: _pixels(i, ds, **(first if i == 0 else later)))
    truth = _series_meaning(series)
    np.testing.assert_array_equal(_values(nio.read_image(series)), truth)
    assert input_stream.stream_of(series) is None                  # slabs would wrap the same
    np.testing.assert_array_equal(_values(ic.read_copy(ic.transcode(series, tmp_path / "e"))),
                                  truth)


def test_the_type_rule_reads_each_slices_width():
    """Ranges from Bits Stored and Pixel Representation, rescaled: a current type that holds
    them is kept (no second decode), one that does not is widened."""
    u16, i16, i32 = sitk.sitkUInt16, sitk.sitkInt16, sitk.sitkInt32
    same = [(1.0, 0.0), (1.0, 0.0)]
    assert nio._mixed_rescale_type(same, (16, 16), u16, bits_stored=(12, 16),
                                   representation=(0, 0)) is None
    assert nio._mixed_rescale_type(same, (16, 16), u16, bits_stored=(16, 16),
                                   representation=(0, 1)) == i32
    assert nio._mixed_rescale_type(same, (8, 16), sitk.sitkUInt8, bits_stored=(8, 16),
                                   representation=(0, 0)) == i32
    assert nio._mixed_rescale_type([(1.0, -1024.0)] * 2, (16, 16), i16, bits_stored=(12, 16),
                                   representation=(0, 0)) == i32
    assert nio._mixed_rescale_type([(1.0, -1024.0)] * 2, (16, 16), i32, bits_stored=(12, 16),
                                   representation=(0, 0)) is None


# -- 3. MONOCHROME1 and a Modality LUT on a later slice -----------------------------------------

def _mono1_on(series: Path, only) -> Path:
    def one(i, ds):
        ds.PhotometricInterpretation = "MONOCHROME1"
    return _edit(_unsigned(series), one, only=only)


def test_monochrome1_on_a_later_slice_only_is_refused(tmp_path):
    series = _mono1_on(write_series(tmp_path / "s", n=3), only={1})
    raw = sitk.ImageSeriesReader()
    raw.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series)))
    assert not np.array_equal(_values(raw.Execute()), _series_meaning(series))   # the premise
    with pytest.raises(InputError, match="mixes MONOCHROME1"):
        nio.read_image(series)


def test_a_modality_lut_on_a_later_slice_only_is_refused(tmp_path):
    from test_dicom_tag_review_0926 import _modality_lut as table
    series = write_series(tmp_path / "s", n=3)
    whole = table(write_series(tmp_path / "t", n=3))
    later = sorted(series.iterdir())[2]
    later.write_bytes(sorted(whole.iterdir())[2].read_bytes())      # slice 2 states a table
    ds = pydicom.dcmread(later)
    ds.SeriesInstanceUID = pydicom.dcmread(sorted(series.iterdir())[0]).SeriesInstanceUID
    ds.save_as(later, enforce_file_format=True)
    with pytest.raises(InputError, match="mixes MONOCHROME1 or a Modality LUT"):
        nio.read_image(series)


@pytest.mark.parametrize("only", [{1}, {1, 2}], ids=["middle", "middle-and-last"])
def test_the_slab_reader_asks_every_slice(tmp_path, only):
    """A middle slice: the check of the ends against pydicom never sees it, the headers do."""
    assert input_stream.stream_of(_mono1_on(write_series(tmp_path / "s", n=3), only)) is None


# -- 4. an Enhanced MONOCHROME1 file with a functional-group rescale ---------------------------

def test_an_enhanced_monochrome1_with_a_group_rescale_reads_as_its_values(tmp_path, form):
    """Refused 'could not be corrected': the undo read the top-level rescale, which is absent."""
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14])                # shared (1, -1024)
    ds = pydicom.dcmread(f)
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(f, enforce_file_format=True)
    truth = _meaning(f)
    np.testing.assert_array_equal(_values(nio.read_image(f)), truth)
    np.testing.assert_array_equal(_values(ic.read_copy(ic.transcode(f, tmp_path / "e"))), truth)


# -- 5. a folder of one multi-frame file -------------------------------------------------------

def test_a_folder_of_one_multi_frame_file_reads_as_the_file(tmp_path):
    folder = tmp_path / "one"
    folder.mkdir()
    f = _enhanced(folder / "enh.dcm", [10, 12, 14, 16])
    image, _, files = nio.read_image_and_tags(folder)
    direct = nio.read_image(f)
    np.testing.assert_array_equal(_values(image), _meaning(f))
    np.testing.assert_array_equal(_values(image), _values(direct))
    assert image.GetOrigin() == direct.GetOrigin() and image.GetSpacing() == direct.GetSpacing()
    assert files == [str(f)]


def test_a_folder_of_one_slice_is_still_no_volume(tmp_path):
    series = write_series(tmp_path / "s", n=2)
    sorted(series.iterdir())[1].unlink()
    with pytest.raises(InputError, match="fewer than 2 slices"):
        nio.read_image(series)


# -- 6. a palette is in stored units ------------------------------------------------------------

PALETTE_KEYWORDS = {"RedPaletteColorLookupTableDescriptor",
                    "GreenPaletteColorLookupTableDescriptor",
                    "BluePaletteColorLookupTableDescriptor", "RedPaletteColorLookupTableData",
                    "GreenPaletteColorLookupTableData", "BluePaletteColorLookupTableData",
                    "PixelPresentation"}


def _supplemental_palette(series: Path) -> Path:
    """A grayscale CT with a supplemental palette: descriptors whose first mapped value is a
    STORED value, beside a rescale (intercept -1024)."""
    def set_(i, ds):
        for k in (0x00281101, 0x00281102, 0x00281103):
            ds.add_new(k, "US", [256, 0, 16])
        lut = np.arange(256, dtype=np.uint16).tobytes()
        for k in (0x00281201, 0x00281202, 0x00281203):
            ds.add_new(k, "OW", lut)
        ds.PixelPresentation = "COLOR"
    return _edit(series, set_)


def test_a_palette_is_not_stated_beside_rescaled_voxels(tmp_path, form):
    series = _supplemental_palette(write_series(tmp_path / "s", n=3))
    copy_ = ic.transcode(series, tmp_path / "e")
    assert _attrs(copy_)["extensions"]["dicom"]["stored_values"] is False
    assert not PALETTE_KEYWORDS & _everything(copy_)
    one = sorted(series.iterdir())[0]
    header = nio.convert(one, tmp_path / "x.nrrd").read_bytes().split(b"\n\n")[0].decode()
    for key in ("0028|1101", "0028|1201", "0008|9205"):
        assert key not in header


def test_a_palette_stays_beside_stored_values(tmp_path):
    """With the stored values a palette maps them as it says, and stays (pinned: that must not
    move); an export's header judged not to hold them drops it."""
    series = _supplemental_palette(_unsigned(write_series(tmp_path / "s", n=3)))
    copy_ = ic.transcode(series, tmp_path / "e")
    assert _attrs(copy_)["extensions"]["dicom"]["stored_values"] is True
    assert {"RedPaletteColorLookupTableDescriptor", "PixelPresentation"} <= _everything(copy_)
    image = sitk.ReadImage(str(sorted(series.iterdir())[0]))
    ic.honest_metadata(image, False)
    assert not image.HasMetaDataKey("0028|1101")


# -- 7. frames of other orientations or spacings -----------------------------------------------

@pytest.mark.parametrize("what", ["orientation", "spacing"])
def test_frames_of_different_orientation_or_spacing_are_refused(tmp_path, what):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14], rescale=False)
    ds = pydicom.dcmread(f)
    frame = ds.PerFrameFunctionalGroupsSequence[2]
    item = Dataset()
    if what == "orientation":
        item.ImageOrientationPatient = [1, 0, 0, 0, 0.8, 0.6]
        frame.PlaneOrientationSequence = Sequence([item])
    else:
        item.PixelSpacing, item.SliceThickness = [1.0, 2.0], 2.0
        frame.PixelMeasuresSequence = Sequence([item])
    ds.save_as(f, enforce_file_format=True)
    assert sitk.ReadImage(str(f)).GetSize()[2] == 3                 # GDCM reads it, silently
    with pytest.raises(InputError, match="frames mix image orientations or pixel spacings"):
        nio.read_image(f)


# -- 8. every read held against pydicom's decode ------------------------------------------------

def _test_file(name):
    from pydicom.data import get_testdata_file
    path = get_testdata_file(name, download=False)
    if path is None:
        pytest.skip(f"pydicom's {name} is not installed")
    return Path(path)


@pytest.mark.parametrize("name", ["rtdose_expb.dcm", "SC_rgb_rle_16bit.dcm",
                                  "SC_rgb_rle_32bit.dcm"])
def test_a_file_gdcm_decodes_wrongly_is_refused(tmp_path, name):
    """Measured: an Explicit VR Big Endian 32-bit dose read 250085395 for 1249000; RLE RGB of
    16 and 32 bits read other values - each without a word."""
    f = _test_file(name)
    assert not np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(np.float64)
                              .reshape(-1), _pydicom_colors(f).reshape(-1))   # the premise
    with pytest.raises(InputError, match="other values than pydicom"):
        nio.read_image(f)
    with pytest.raises(InputError, match="other values than pydicom"):
        nio.convert(f, tmp_path / "x.nrrd")


def _pydicom_colors(f):
    return pydicom.dcmread(f).pixel_array.astype(np.float64)


def test_a_one_bit_file_read_as_0_and_255_is_refused(tmp_path):
    """GDCM reads 1-bit pixels (a SEG's) as 0/255; pydicom as 0/1."""
    from pydicom.pixels import pack_bits
    one = sorted(write_series(tmp_path / "s", n=2).iterdir())[0]
    ds = pydicom.dcmread(one)
    for k in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
        delattr(ds, k)
    ds.BitsAllocated = ds.BitsStored = 1
    ds.HighBit, ds.PixelRepresentation = 0, 0
    ds.Rows, ds.Columns = 8, 8
    bits = (np.arange(64) % 3 == 0).astype(np.uint8).reshape(8, 8)
    ds.PixelData = pack_bits(bits)
    f = tmp_path / "bit.dcm"
    ds.save_as(f, enforce_file_format=True)
    assert set(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(f))))) == {0, 255}  # premise
    with pytest.raises(InputError, match="other values than pydicom"):
        nio.read_image(f)


@pytest.mark.parametrize("where", ["series", "file"])
def test_any_disagreement_with_pydicoms_decode_is_refused(tmp_path, monkeypatch, where):
    """The policy, for whatever GDCM gets wrong next: its first or last slice one off."""
    series = write_series(tmp_path / "s", n=3)
    target = series if where == "series" else sorted(series.iterdir())[0]
    reader = sitk.ImageSeriesReader if where == "series" else sitk.ImageFileReader
    real = reader.Execute

    def off_by_one(self):
        image = real(self)
        return sitk.Cast(image, image.GetPixelID()) + 1
    nio.read_image(target)                                             # agrees: reads
    monkeypatch.setattr(reader, "Execute", off_by_one)
    monkeypatch.setattr(sitk, "ReadImage", lambda p, *a: off_by_one(_file_reader(p)))
    with pytest.raises(InputError, match="other values than pydicom"):
        nio.read_image(target)



def test_the_last_slice_is_held_against_pydicom_too(tmp_path, monkeypatch):
    """First AND last: a decode that goes wrong only at the end of a series is refused."""
    series = write_series(tmp_path / "s", n=4)
    real = sitk.ImageSeriesReader.Execute

    def last_off(self):
        a = sitk.GetArrayFromImage(real(self))
        a[-1] += 1
        return sitk.GetImageFromArray(a)
    monkeypatch.setattr(sitk.ImageSeriesReader, "Execute", last_off)
    with pytest.raises(InputError, match=r"other values than pydicom.*slice 3"):
        nio.read_image(series)

def _file_reader(p):
    r = sitk.ImageFileReader()
    r.SetFileName(str(p))
    return r


def test_the_slab_reader_holds_its_ends_against_pydicom(tmp_path, monkeypatch):
    series = write_series(tmp_path / "s", n=3)
    assert input_stream.stream_of(series) is not None
    real = sitk.ImageFileReader.Execute
    monkeypatch.setattr(sitk.ImageFileReader, "Execute", lambda self: real(self) + 1)
    assert input_stream.stream_of(series) is None


def _rtdose(tmp_path: Path, offsets) -> Path:
    f = _test_file("rtdose.dcm")
    ds = pydicom.dcmread(f)
    ds.GridFrameOffsetVector = list(offsets)
    out = tmp_path / "rd.dcm"
    ds.save_as(out)
    return out


def test_an_rtdose_with_uneven_frame_offsets_is_refused(tmp_path):
    """GDCM placed frames at offsets 0, 5, 10, 20, ... on a uniform 5 mm grid."""
    f = _rtdose(tmp_path, [0, 5, 10] + [20 + 5 * k for k in range(12)])
    image = sitk.ReadImage(str(f))
    assert image.GetSpacing()[2] == pytest.approx(5.0)               # the premise
    with pytest.raises(InputError, match="Grid Frame Offset Vector"):
        nio.read_image(f)


@pytest.mark.parametrize("offsets", [[5.0 * k for k in range(15)], [-5.0 * k for k in range(15)],
                                     [-761.87 + 5.0 * k for k in range(15)]],
                         ids=["ascending", "descending", "absolute"])
def test_an_rtdose_reads_where_its_offsets_place_it(tmp_path, offsets):
    """Pins what must not move: every placement GDCM gets right reads."""
    image = nio.read_image(_rtdose(tmp_path, offsets))
    ipp_z = -761.87
    zs = [image.TransformIndexToPhysicalPoint((0, 0, k))[2] for k in range(15)]
    want = [(o if offsets[0] != 0 else ipp_z + o) for o in offsets]
    np.testing.assert_allclose(zs, want, atol=1e-3)


def test_a_series_gdcm_cannot_read_is_a_refusal(tmp_path):
    """Signed 12-bit MONOCHROME1: GDCM raised a bare RuntimeError out of the series reader."""
    def s12(i, ds):
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 12, 11, 1
        ds.PixelData = np.full((ds.Rows, ds.Columns), -1000 + i, np.int16).tobytes()
    with pytest.raises(InputError, match="cannot read the DICOM series"):
        nio.read_image(_edit(write_series(tmp_path / "s", n=3), s12))


# -- 9. a truncated or corrupt gzip -------------------------------------------------------------

def _nii_gz(path: Path) -> Path:
    rng = np.random.default_rng(0)
    image = sitk.GetImageFromArray(rng.integers(-1000, 1000, (20, 30, 40)).astype(np.int16))
    sitk.WriteImage(image, str(path), True)
    return path


def _truncated(tmp_path) -> Path:
    f = _nii_gz(tmp_path / "v.nii.gz")
    data = f.read_bytes()
    f.write_bytes(data[:len(data) // 2])
    return f


def _bad_crc(tmp_path) -> Path:
    f = _nii_gz(tmp_path / "v.nii.gz")
    data = bytearray(f.read_bytes())
    data[-8] ^= 0xFF                                                   # the CRC32 of the trailer
    f.write_bytes(bytes(data))
    return f


def _no_trailer(tmp_path) -> Path:
    """Every data byte there, the gzip trailer (CRC and length) cut off: SimpleITK reads it."""
    f = _nii_gz(tmp_path / "v.nii.gz")
    f.write_bytes(f.read_bytes()[:-8])
    return f


@pytest.mark.parametrize("damage", [_truncated, _bad_crc, _no_trailer],
                         ids=["truncated", "bad-crc", "no-trailer"])
def test_a_damaged_gzip_is_refused(tmp_path, damage, form):
    f = damage(tmp_path)
    with pytest.raises((EOFError, gzip.BadGzipFile)):                 # Python's gzip refuses
        gzip.decompress(f.read_bytes())
    with pytest.raises(InputError, match="gzip"):
        nio.read_image(f)
    with pytest.raises(InputError, match="gzip"):
        nio.convert(f, tmp_path / "x.nrrd")
    assert ic.transcode(f, tmp_path / "e") is None                    # no copy, either path
    assert not ic.copy_path(tmp_path / "e").exists()


def test_the_slab_reader_reads_to_the_end_of_the_stream(tmp_path):
    """A stream whose data is whole but whose trailer is gone: only reading to the end sees it."""
    stream = input_stream.stream_of(_no_trailer(tmp_path))
    assert stream is not None
    with pytest.raises(EOFError):
        for _ in stream.slabs(8):
            pass


def test_a_whole_gzip_reads_as_before(tmp_path):
    """Pins what must not move: two members, and trailing zero padding, read as they did."""
    f = _nii_gz(tmp_path / "v.nii.gz")
    want = _values(sitk.ReadImage(str(f)))
    np.testing.assert_array_equal(_values(nio.read_image(f)), want)
    raw = gzip.decompress(f.read_bytes())
    half = len(raw) // 2
    g = tmp_path / "w.nii.gz"
    g.write_bytes(gzip.compress(raw[:half]) + gzip.compress(raw[half:]) + b"\0" * 16)
    np.testing.assert_array_equal(_values(nio.read_image(g)), want)


# -- 10. gaps a mutation run of the 2026-09-26 reader found (adopted from its probes) ----------

def _first_values(image):
    return list(sitk.GetArrayFromImage(image)[:, 0, 0])


def test_monochrome1_is_undone_with_each_slices_rescale(tmp_path):
    series = _rescale(_monochrome1(write_series(tmp_path / "s", n=3)), 1, -1024, only={1, 2})
    assert _first_values(nio.read_image(series)) == [100, 101 - 1024, 102 - 1024]


def test_a_series_partly_monochrome1_is_refused(tmp_path):
    series = _mono1_on(write_series(tmp_path / "s", n=3), only={0})
    with pytest.raises(InputError, match="mixes MONOCHROME1"):
        nio.read_image(series)


def test_monochrome1_with_a_slope_reads_as_its_real_values(tmp_path):
    series = _rescale(_monochrome1(write_series(tmp_path / "s", n=3)), 2, -1024)
    assert _first_values(nio.read_image(series)) == [2 * v - 1024 for v in (100, 101, 102)]


def test_every_frame_of_a_multi_frame_monochrome1_file_is_undone(tmp_path):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14], rescale=False)
    ds = pydicom.dcmread(f)
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.save_as(f, enforce_file_format=True)
    got = sitk.GetArrayFromImage(nio.read_image(f))
    assert np.array_equal(got.ravel(), np.arange(36) + 100)


def test_a_table_and_a_rescale_on_a_later_slice_is_refused(tmp_path):
    series = _rescale(_modality_lut(write_series(tmp_path / "s", n=3)), 1, -1024, only={1})
    with pytest.raises(InputError, match="Modality LUT Sequence and a rescale"):
        nio.read_image(series)


def test_a_file_stating_fewer_frame_positions_than_frames_is_refused(tmp_path):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16])
    ds = pydicom.dcmread(f)
    del ds.PerFrameFunctionalGroupsSequence[-1]
    ds.save_as(f, enforce_file_format=True)
    with pytest.raises(InputError, match="3 frame positions for 4 frames"):
        nio.read_image(f)


def test_a_frame_without_a_position_means_nothing_is_checked(tmp_path):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16])
    ds = pydicom.dcmread(f)
    del ds.PerFrameFunctionalGroupsSequence[2].PlanePositionSequence
    ds.save_as(f, enforce_file_format=True)
    assert nio.read_image(f).GetSize()[2] == 4


def test_a_two_frame_file_placed_off_its_positions_is_refused(tmp_path, monkeypatch):
    f = _enhanced(tmp_path / "enh.dcm", [10, 14], rescale=False)
    real = sitk.ReadImage

    def off(*a, **k):
        im = real(*a, **k)
        if im.GetDimension() == 3 and im.GetSize()[2] == 2:
            sx, sy, _ = im.GetSpacing()
            im.SetSpacing((sx, sy, 1.0))
        return im
    monkeypatch.setattr(sitk, "ReadImage", off)
    with pytest.raises(InputError, match="not acquired"):
        nio.read_image(f)


def test_an_empty_rescale_counts_as_the_identity(tmp_path):
    series = _rescale(_unsigned(write_series(tmp_path / "s", n=3), value=100), 1, -1024,
                      only={1, 2})

    def empty(i, ds):
        ds.add_new(0x00281053, "DS", None)
        ds.add_new(0x00281052, "DS", None)
    _edit(series, empty, only={0})
    assert _first_values(nio.read_image(series)) == [100, -924, -924]


def test_an_unparseable_rescale_is_a_refusal_never_a_crash(tmp_path):
    """A slope of "abc": what the values mean is unknown - refused, where it once risked a
    TypeError out of the type rule."""
    series = _unsigned(write_series(tmp_path / "s", n=3), value=100)
    f = sorted(series.iterdir())[1]
    ds = pydicom.dcmread(f)
    ds.RescaleSlope, ds.RescaleIntercept = "12345.6", 0
    ds.save_as(f, enforce_file_format=True)
    b = f.read_bytes()
    assert b.count(b"12345.6 ") == 1
    f.write_bytes(b.replace(b"12345.6 ", b"abc     "))
    with pytest.raises(InputError, match="not a number"):
        nio.read_image(series)
    assert input_stream.stream_of(series) is None


def test_a_per_frame_functional_group_rescale_is_not_stored_values(tmp_path):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14], rescale=False)
    ds = pydicom.dcmread(f)
    ds.PixelRepresentation = 1                          # signed: GDCM's output type is the stored
    ds.PixelData = (np.arange(36, dtype=np.int16) + 100).tobytes()
    for frame in ds.PerFrameFunctionalGroupsSequence:
        pv = Dataset()
        pv.RescaleIntercept, pv.RescaleSlope, pv.RescaleType = -1024, 1, "HU"
        frame.PixelValueTransformationSequence = Sequence([pv])
    ds.save_as(f, enforce_file_format=True)
    assert sitk.GetArrayFromImage(nio.read_image(f)).ravel()[0] == 100 - 1024   # the premise
    copy_ = ic.transcode(f, tmp_path / "e")
    assert _attrs(copy_)["extensions"]["dicom"]["stored_values"] is False
    assert not {"BitsStored", "PixelPaddingValue"} & _everything(copy_)


def _refuse_headers(monkeypatch):
    import duckn.dicom_tags as dtags

    def refuse(*a, **k):
        raise ValueError("could not convert string to float: 'abc'")
    monkeypatch.setattr(dtags, "tags_from_datasets", refuse)


def test_the_fallback_of_a_modality_lut_series_is_not_stored_values(tmp_path, monkeypatch):
    _refuse_headers(monkeypatch)
    copy_ = ic.transcode(_modality_lut(write_series(tmp_path / "s", n=3)), tmp_path / "e")
    d = _attrs(copy_)["extensions"]
    assert d["haversack"]["tags_version"] == 1
    assert d["dicom"]["stored_values"] is False
    assert "PixelPaddingValue" not in _everything(copy_)


def test_input_tags_report_a_fallback_copys_tags_version(tmp_path, monkeypatch):
    from haversack.inputs import Input
    _refuse_headers(monkeypatch)
    copy_ = ic.transcode(_malformed_kvp(write_series(tmp_path / "s")), tmp_path / "e")
    assert Input(None, copy_, None).tags()["tags_version"] == 1


def test_a_reader_version_3_copy_is_stale():
    """Copies made before these corrections hold the wrong values or tags for the inputs above:
    READER_VERSION moved past 3, so they are fetched again (an upload is gone)."""
    assert ic.READER_VERSION >= 4
