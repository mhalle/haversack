"""The 2026-09-26 review of the input copy's DICOM tags: what a file haversack stores or hands out
says must never contradict the data a reader gets from that file alone (the user's rule).
SimpleITK stays the converter; these hold that nothing it decodes is described wrongly.

- ``stored_values`` is judged from what the decode DID, not from the top-level rescale alone:
  MONOCHROME1 (GDCM inverts it), an Enhanced object's functional-group rescale, a pixel type
  other than the stored one, a rescale that does not parse, one on a later slice only;
- MONOCHROME1 carries no Photometric Interpretation and no window, in a copy or an export; a
  Modality LUT GDCM does not apply keeps the stored values but loses the window;
- an export of a color image carries no grayscale or palette description;
- a header that does not convert costs the tags their fullness (SimpleITK's instead), never the
  copy;
- a multi-frame file whose frames the reader would place where they were not acquired is
  refused;
- a series whose slices do not share one rescale reads in a type that holds every value;
- ``Input.tags()`` says what the copy holds;
- and the gaps a mutation run found in the tests of e4e0cab/0a34e0a (adopted from its probes).

Every test here failed on 6eb2a07 or pins behavior a mutant of it survived. Parametrized over
LITERAL tag lists: a set taken from the code under test (duckn's ``STORED_ENCODING``, say)
would shrink with the mutant and pass.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pydicom = pytest.importorskip("pydicom")
pytest.importorskip("duckn")
pytest.importorskip("zarr")

from haversack import input_copy as ic  # noqa: E402
from haversack import input_stream  # noqa: E402
from haversack import io as nio  # noqa: E402
from haversack.errors import InputError  # noqa: E402

from test_input_copy import _attrs, _enrich, _nrrd_header, _pad, write_series  # noqa: E402

CR_SOP = "1.2.840.10008.5.1.4.1.1.1"
ENHANCED_CT_SOP = "1.2.840.10008.5.1.4.1.1.2.1"
PALETTE_SOP = "1.2.840.10008.5.1.4.1.1.7"


def _edit(series: Path, fn, only=None) -> Path:
    for i, f in enumerate(sorted(series.iterdir())):
        if only is None or i in only:
            ds = pydicom.dcmread(f)
            fn(i, ds)
            ds.save_as(f, enforce_file_format=True)
    return series


def _rescale(series: Path, slope, intercept, only=None) -> Path:
    def set_(i, ds):
        ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    return _edit(series, set_, only)


def _unsigned(series: Path, *, rescale=False, value=None) -> Path:
    """Stored values an unsigned 16-bit type holds, 12 of them significant; no rescale unless
    asked (then intercept -1024, as a CT)."""
    def set_(i, ds):
        for k in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
            if k in ds and not rescale:
                delattr(ds, k)
        ds.PixelRepresentation, ds.BitsStored, ds.HighBit = 0, 12, 11
        v = 100 + i if value is None else value
        ds.PixelData = np.full((ds.Rows, ds.Columns), v, np.uint16).tobytes()
    return _edit(series, set_)


def _monochrome1(series: Path) -> Path:
    def set_(i, ds):
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.WindowCenter, ds.WindowWidth = 100, 50
        ds.WindowCenterWidthExplanation = "BONE"
        ds.VOILUTFunction = "LINEAR"
    return _edit(_unsigned(series), set_)


def _modality_lut(series: Path) -> Path:
    """A Modality LUT Sequence and no rescale: the window is in the LUT's output units."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    def set_(i, ds):
        item = Dataset()
        item.LUTDescriptor = [4096, 0, 16]
        item.ModalityLUTType = "US"
        item.add_new(0x00283006, "US", [min(65535, 2 * k + 7) for k in range(4096)])
        ds.ModalityLUTSequence = Sequence([item])
        ds.WindowCenter, ds.WindowWidth = 500, 100
        ds.add_new(0x00280120, "US", 0)                  # Pixel Padding Value, stored units
    return _edit(_unsigned(series), set_)


def _enhanced(path: Path, zs, *, rescale=True) -> Path:
    """An Enhanced CT file: one frame per z in ``zs``, its rescale (intercept -1024) only in the
    Shared Functional Groups, unsigned 12-bit stored values, a stored-unit padding value."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = ENHANCED_CT_SOP, uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = ENHANCED_CT_SOP, uid, "CT"
    ds.SeriesInstanceUID, ds.StudyInstanceUID = generate_uid(), generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Rows, ds.Columns, ds.NumberOfFrames = 4, 3, len(zs)
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelPaddingValue = 0
    shared = Dataset()
    measures = Dataset()
    measures.PixelSpacing, measures.SliceThickness = [1.0, 1.0], 2.0
    shared.PixelMeasuresSequence = Sequence([measures])
    orient = Dataset()
    orient.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = Sequence([orient])
    if rescale:
        pv = Dataset()
        pv.RescaleIntercept, pv.RescaleSlope, pv.RescaleType = -1024, 1, "HU"
        shared.PixelValueTransformationSequence = Sequence([pv])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    frames = []
    for z in zs:
        frame, position = Dataset(), Dataset()
        position.ImagePositionPatient = [0.0, 0.0, float(z)]
        frame.PlanePositionSequence = Sequence([position])
        frames.append(frame)
    ds.PerFrameFunctionalGroupsSequence = Sequence(frames)
    ds.PixelData = (np.arange(len(zs) * 12, dtype=np.uint16) + 100).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def _palette_file(tmp_path: Path) -> Path:
    def set_(i, ds):
        ds.PhotometricInterpretation = "PALETTE COLOR"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        for k in (0x00281101, 0x00281102, 0x00281103):
            ds.add_new(k, "US", [256, 0, 16])
        lut = np.array([k * 257 for k in range(256)], np.uint16)
        ds.add_new(0x00281201, "OW", lut.tobytes())
        ds.add_new(0x00281202, "OW", lut[::-1].tobytes())
        ds.add_new(0x00281203, "OW", lut.tobytes())
        for k in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
            delattr(ds, k)
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = PALETTE_SOP
        ds.PixelData = np.full((ds.Rows, ds.Columns), 10 + i, np.uint8).tobytes()
    series = _edit(write_series(tmp_path / "pal", n=2), set_)
    return sorted(series.iterdir())[0]


def _everything(copy: Path) -> set:
    """Every keyword the copy states, series-level and per slice."""
    d = _attrs(copy)
    per = {k for s in d["axes"][0].get("samples") or []
           for k in ((s.get("metadata") or {}).get("dicom") or {})}
    return set(d["extensions"]["dicom"]["tags"]) | per


@pytest.fixture(params=["uncompressed", "zstd"])
def form(request, monkeypatch):
    monkeypatch.setenv(ic.COMPRESSION_ENV, request.param)
    return request.param


# -- 1. stored_values is what the decode did -------------------------------------------------

def test_the_judgment_asks_the_pixel_type_the_decode_produced():
    """Stored values only in the type Bits Allocated and Pixel Representation imply: a widened
    or converted type holds other values. Unknown is never stored."""
    d = {"0028|0100": "16", "0028|0103": "0", "0028|0004": "MONOCHROME2 "}
    assert ic._holds_stored_values([d], sitk.sitkUInt16) is True
    assert ic._holds_stored_values([d], sitk.sitkInt32) is False
    assert ic._holds_stored_values([d], sitk.sitkFloat64) is False
    assert ic._holds_stored_values([{**d, "0028|0103": "1"}], sitk.sitkInt16) is True
    assert ic._holds_stored_values([{**d, "0028|0100": "8"}], sitk.sitkUInt8) is True
    assert ic._holds_stored_values([d], None) is False
    assert ic._holds_stored_values([{"0028|0103": "0"}], sitk.sitkUInt16) is False   # no bits
    assert ic._holds_stored_values([d], sitk.sitkUInt16, value_transform=True) is False
    assert ic._holds_stored_values([{**d, "0028|0004": "MONOCHROME1"}], sitk.sitkUInt16) is False
    # RGB is its stored values; a palette GDCM expanded, or YBR it converted, is not
    rgb = {"0028|0100": "8", "0028|0103": "0", "0028|0002": "3", "0028|0004": "RGB"}
    assert ic._holds_stored_values([rgb], sitk.sitkVectorUInt8) is True
    assert ic._holds_stored_values([{**rgb, "0028|0004": "YBR_FULL"}],
                                   sitk.sitkVectorUInt8) is False
    assert ic._holds_stored_values([{**rgb, "0028|0002": "1", "0028|0004": "PALETTE COLOR"}],
                                   sitk.sitkVectorUInt8) is False
    # slices that disagree about their type: no one type holds both as stored
    assert ic._holds_stored_values([d, {**d, "0028|0103": "1"}], sitk.sitkUInt16) is False


def test_a_monochrome1_copy_is_not_stored_values(tmp_path, form):
    """GDCM inverts MONOCHROME1 (stored 100 is 3995 in a 12-bit copy): the type is the stored
    one and there is no rescale, and still the values are not the stored ones."""
    series = _monochrome1(write_series(tmp_path / "s", n=3))
    copy = ic.transcode(series, tmp_path / "e")
    assert sitk.GetArrayFromImage(ic.read_copy(copy))[0, 0, 0] == 4095 - 100
    d = _attrs(copy)["extensions"]["dicom"]
    assert d["stored_values"] is False
    assert not {"BitsStored", "HighBit"} & _everything(copy)


def test_an_enhanced_objects_functional_group_rescale_is_not_stored_values(tmp_path, form):
    """An Enhanced CT states its rescale only in a functional group, which GDCM applies (stored
    100 is -924) into int16 where the stored type is uint16. The copy said its values were the
    stored ones and kept Bits Stored 12 and Pixel Padding Value 0 beside HU."""
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16])
    copy = ic.transcode(f, tmp_path / "e")
    assert sitk.GetArrayFromImage(ic.read_copy(copy))[0, 0, 0] == 100 - 1024
    d = _attrs(copy)["extensions"]["dicom"]
    assert d["stored_values"] is False
    assert not {"BitsStored", "HighBit", "PixelPaddingValue"} & d["tags"].keys()
    # duckn >= 0.5.4 applies the exclusions inside the Shared Functional Groups: the rescale
    # (Pixel Value Transformation) and the geometry macros are gone, and this file's groups
    # held nothing else - so a reader cannot apply the rescale a second time
    shared = d["tags"].get("SharedFunctionalGroupsSequence") or [{}]
    assert all("PixelValueTransformationSequence" not in item for item in shared)
    assert "PerFrameFunctionalGroupsSequence" not in d["tags"]


def test_an_enhanced_object_without_a_value_transform_is_its_stored_values(tmp_path):
    """The control: the same file with no functional-group rescale holds its stored values."""
    copy = ic.transcode(_enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16], rescale=False),
                        tmp_path / "e")
    d = _attrs(copy)["extensions"]["dicom"]
    assert d["stored_values"] is True and d["tags"]["BitsStored"] == 12


def test_an_export_of_an_enhanced_file_keeps_no_stored_unit_tag(tmp_path):
    h = _nrrd_header(nio.convert(_enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16]),
                                 tmp_path / "x.nrrd"))
    for key in ("0028|0101", "0028|0102", "0028|0120"):
        assert key not in h, key


# -- 2. MONOCHROME1 and a Modality LUT: the window is not about these values -----------------

def test_a_monochrome1_copy_states_no_photometric_interpretation_and_no_window(tmp_path, form):
    copy = ic.transcode(_monochrome1(write_series(tmp_path / "s", n=3)), tmp_path / "e")
    stated = _everything(copy)
    for name in ("PhotometricInterpretation", "WindowCenter", "WindowWidth",
                 "WindowCenterWidthExplanation", "VOILUTFunction"):
        assert name not in stated, name
    assert "Modality" in stated                                  # the rest stays


@pytest.mark.parametrize("what", ["copy", "file"])
def test_an_export_of_monochrome1_states_no_photometric_interpretation_and_no_window(
        tmp_path, what):
    series = _monochrome1(write_series(tmp_path / "s", n=3))
    src = ic.transcode(series, tmp_path / "e") if what == "copy" else sorted(series.iterdir())[0]
    h = _nrrd_header(nio.convert(src, tmp_path / "x.nrrd"))
    for key in ("0028|0004", "0028|1050", "0028|1051", "0028|1055", "0028|1056", "0028|0101"):
        assert key not in h, key
    assert "0008|0060:=CT" in h


def test_a_modality_lut_gdcm_does_not_apply_leaves_stored_values_and_no_window(tmp_path, form):
    series = _modality_lut(write_series(tmp_path / "s", n=3))
    # the premise, held here so that a SimpleITK that starts applying the LUT fails loudly
    # rather than leaving a rule that drops a window then true: stored 100 would be 207
    assert list(sitk.GetArrayFromImage(nio.read_image(series))[:, 0, 0]) == [100, 101, 102]
    copy = ic.transcode(series, tmp_path / "e")
    d = _attrs(copy)["extensions"]["dicom"]
    assert d["stored_values"] is True                           # the voxels ARE stored values
    assert d["tags"]["BitsStored"] == 12 and d["tags"]["PixelPaddingValue"] == 0
    assert not {"WindowCenter", "WindowWidth"} & _everything(copy)
    assert d["tags"]["PhotometricInterpretation"] == "MONOCHROME2"   # true of these values


def test_an_export_of_a_file_with_an_unapplied_modality_lut_states_no_window(tmp_path):
    f = sorted(_modality_lut(write_series(tmp_path / "s", n=2)).iterdir())[0]
    h = _nrrd_header(nio.convert(f, tmp_path / "x.nrrd"))
    assert "0028|1050" not in h and "0028|1051" not in h
    assert "0028|0101:=12" in h and "0028|0120:=0" in h         # stored values: still true


# -- 3. a color export describes no grayscale or palette pixel -------------------------------

def test_an_export_of_a_palette_file_describes_the_rgb_pixels_it_has(tmp_path):
    out = nio.convert(_palette_file(tmp_path), tmp_path / "x.nrrd")
    assert sitk.ReadImage(str(out)).GetNumberOfComponentsPerPixel() == 3
    h = _nrrd_header(out)
    for key in ("0028|0002", "0028|0004", "0028|0006", "0028|1101", "0028|1102", "0028|1103",
                "0028|1201", "0028|1202", "0028|1203", "0028|0101"):
        assert key not in h, key
    assert "0008|0060:=CT" in h


def test_honest_metadata_of_a_vector_image_drops_the_color_description():
    img = sitk.Image([2, 2, 2], sitk.sitkVectorUInt8, 3)
    for key in ("0028|0002", "0028|0004", "0028|0006", "0028|1111", "0028|1199", "0028|1221",
                "0028|1223", "0008|0060"):
        img.SetMetaData(key, "1")
    ic.honest_metadata(img, True)
    assert set(img.GetMetaDataKeys()) == {"0008|0060"}
    gray = sitk.Image([2, 2, 2], sitk.sitkUInt8)                  # the control: one component
    gray.SetMetaData("0028|0002", "1")
    gray.SetMetaData("0028|0004", "MONOCHROME2")
    ic.honest_metadata(gray, True)
    assert set(gray.GetMetaDataKeys()) == {"0028|0002", "0028|0004"}


# -- 4. a header that does not convert never costs the copy ----------------------------------

def _malformed_kvp(series: Path) -> Path:
    """KVP "abc" in one file: pydicom reads it and warns; converting its value raises. Written
    over the bytes of a unique valid value, since pydicom refuses to write it."""
    f = sorted(series.iterdir())[2]
    ds = pydicom.dcmread(f)
    ds.KVP = "12345.6"
    ds.save_as(f, enforce_file_format=True)
    b = f.read_bytes()
    assert b.count(b"12345.6 ") == 1
    f.write_bytes(b.replace(b"12345.6 ", b"abc     "))
    return series


def test_a_malformed_value_keeps_the_copy_and_itself_as_text(tmp_path, form):
    """duckn >= 0.5.4 keeps a value it cannot parse as its text: the copy keeps the files' own
    tags (tags_version 2), with that one value as written."""
    series = _malformed_kvp(write_series(tmp_path / "s"))
    copy = ic.transcode(series, tmp_path / "e", source="fixture:1")
    assert copy is not None
    d = _attrs(copy)["extensions"]
    assert d["haversack"]["tags_version"] == 2
    assert "abc" in str(d["dicom"]["tags"].get("KVP")) or any(
        "abc" in str(t) for t in ic.slice_tags(copy, "KVP"))


def test_a_tag_conversion_failure_keeps_the_copy_with_simpleitks_tags(tmp_path, form, capsys,
                                                                       monkeypatch):
    """Whatever the files' headers do to the tag conversion, haversack never loses the copy
    over tags: it falls back to SimpleITK's dictionaries (tags_version 1) and says so."""
    import duckn.dicom_tags as dtags

    def refuse(*a, **k):
        raise ValueError("could not convert string to float: 'abc'")
    monkeypatch.setattr(dtags, "tags_from_datasets", refuse)
    series = _malformed_kvp(write_series(tmp_path / "s"))
    ref = nio.read_image(series)
    copy = ic.transcode(series, tmp_path / "e", source="fixture:1")
    assert copy is not None
    assert np.array_equal(sitk.GetArrayFromImage(ic.read_copy(copy)),
                          sitk.GetArrayFromImage(ref))
    d = _attrs(copy)["extensions"]
    assert d["haversack"]["tags_version"] == 1
    assert d["dicom"]["tags"]["Modality"] == "CT" and d["dicom"]["stored_values"] is False
    assert not {"BitsStored", "PixelPaddingValue"} & _everything(copy)
    err = capsys.readouterr().err
    assert "fixture:1" in err and "did not convert" in err


# -- 5. frames are never placed where they were not acquired ---------------------------------

def test_a_multi_frame_file_with_gapped_frames_is_refused(tmp_path):
    """GDCM placed frames at z 10, 12, 16, 18 on a uniform grid at 10, 11, 12, 13."""
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 16, 18])
    with pytest.raises(InputError, match="frames' positions.*non-uniform"):
        nio.read_image(f)
    assert ic.transcode(f, tmp_path / "e") is None               # the original stays
    with pytest.raises(InputError, match="frames"):
        nio.convert(f, tmp_path / "x.nrrd")


@pytest.mark.parametrize("zs", [[10, 12, 14, 16], [19, 16, 13, 10]], ids=["up", "down"])
def test_a_multi_frame_file_with_uniform_frames_reads_where_they_were_acquired(tmp_path, zs):
    image = nio.read_image(_enhanced(tmp_path / "enh.dcm", zs))
    last = image.TransformIndexToPhysicalPoint((0, 0, len(zs) - 1))
    assert image.GetOrigin()[2] == pytest.approx(zs[0]) and last[2] == pytest.approx(zs[-1])


def test_the_frame_check_holds_the_readers_grid_against_the_positions(tmp_path):
    """The comparison half: uniform positions, but a grid that does not start or step where
    they do (GDCM places uniform frames right today; this is what catches it if it does not)."""
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 14, 16])
    image = sitk.ReadImage(str(f))
    nio._check_frame_positions(f, image)                          # agrees: no error
    for change in (lambda im: im.SetSpacing((1.0, 1.0, 1.0)),
                   lambda im: im.SetOrigin((0.0, 0.0, 11.0)),
                   lambda im: im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))):
        wrong = sitk.Image(image)
        change(wrong)
        with pytest.raises(InputError, match="not acquired"):
            nio._check_frame_positions(f, wrong)


def test_frames_without_positions_read_as_before(tmp_path):
    f = _enhanced(tmp_path / "enh.dcm", [10, 12, 16, 18])
    ds = pydicom.dcmread(f)
    del ds.PerFrameFunctionalGroupsSequence
    ds.save_as(f, enforce_file_format=True)
    assert nio.read_image(f).GetSize()[2] == 4


# -- 6. slices that do not share a rescale ---------------------------------------------------

def _mixed_rescale(tmp_path) -> Path:
    """The first slice unsigned and without rescale, the others at intercept -1024: the series
    reader took uint16 from the first file and wrapped -924 to 64612."""
    series = _unsigned(write_series(tmp_path / "s", n=3), value=100)
    return _rescale(series, 1, -1024, only={1, 2})


def test_a_series_of_mixed_rescales_reads_every_value(tmp_path):
    image = nio.read_image(_mixed_rescale(tmp_path))
    assert list(sitk.GetArrayFromImage(image)[:, 0, 0]) == [100, -924, -924]
    assert image.GetPixelID() == sitk.sitkInt32


def test_a_series_of_mixed_rescales_is_copied_whole_and_right(tmp_path, form):
    series = _mixed_rescale(tmp_path)
    assert input_stream.stream_of(series) is None                 # slabs would read uint16
    copy = ic.transcode(series, tmp_path / "e")
    assert list(sitk.GetArrayFromImage(ic.read_copy(copy))[:, 0, 0]) == [100, -924, -924]
    assert _attrs(copy)["extensions"]["dicom"]["stored_values"] is False


def test_a_series_of_one_rescale_is_read_once(tmp_path, monkeypatch):
    """The fix costs a second decode only where it is needed: never for one rescale, nor where
    the first file's type holds them all (PET's per-slice slopes read as float64)."""
    import SimpleITK
    executed = []
    real = SimpleITK.ImageSeriesReader.Execute
    monkeypatch.setattr(SimpleITK.ImageSeriesReader, "Execute",
                        lambda self: executed.append(1) or real(self))
    nio.read_image(write_series(tmp_path / "one"))
    assert len(executed) == 1
    assert nio._mixed_rescale_type([(0.5, 0.0), (0.25, 3.0)], (), sitk.sitkFloat64) is None
    assert nio._mixed_rescale_type([(1.0, 0.0), (1.0, -1024.0)], (), sitk.sitkUInt16) \
        == sitk.sitkInt32
    assert nio._mixed_rescale_type([(1.0, 0.0), (0.5, 0.0)], (), sitk.sitkInt32) \
        == sitk.sitkFloat64
    assert nio._mixed_rescale_type([(1.0, 0.0), (1.0, -1024.0)], ("32", "32"),
                                   sitk.sitkUInt32) == sitk.sitkFloat64


# -- 7. Input.tags() says what the copy holds ------------------------------------------------

def test_input_tags_say_whether_the_values_are_stored_and_which_tags(tmp_path):
    from haversack.inputs import Input
    rescaled = Input(None, ic.transcode(_pad(write_series(tmp_path / "a")), tmp_path / "ea"),
                     None).tags()
    assert rescaled["stored_values"] is False and rescaled["tags_version"] == 2
    assert rescaled["series"]["Modality"] == "CT" and len(rescaled["slices"]) == 6
    stored = Input(None, ic.transcode(_pad(_enrich(write_series(tmp_path / "b"), rescale=False)),
                                      tmp_path / "eb"), None).tags()
    assert stored["stored_values"] is True
    assert stored["series"]["PixelPaddingValue"] == 0


# -- 8. the mutation run's gaps (adopted from its probes, 2026-09-26) ------------------------

@pytest.mark.parametrize("slope,intercept", [(0.5, 0), (1, 100)])
def test_any_non_identity_rescale_is_not_stored_values(tmp_path, slope, intercept):
    s = _rescale(_pad(write_series(tmp_path / "s")), slope, intercept)
    d = _attrs(ic.transcode(s, tmp_path / "e"))["extensions"]["dicom"]
    assert d["stored_values"] is False and "PixelPaddingValue" not in d["tags"]


def test_a_rescale_on_a_later_slice_only_is_not_stored_values(tmp_path):
    s = _rescale(_rescale(_pad(write_series(tmp_path / "s")), 1, 0), 2, 0, only={3})
    d = _attrs(ic.transcode(s, tmp_path / "e"))["extensions"]["dicom"]
    assert d["stored_values"] is False


def test_an_unreadable_rescale_is_not_stored_values():
    ok = {"0028|1053": "1", "0028|1052": "0", "0028|0100": "16", "0028|0103": "0"}
    assert ic._holds_stored_values([ok], sitk.sitkUInt16) is True          # the control
    assert ic._holds_stored_values([{**ok, "0028|1053": "abc"}], sitk.sitkUInt16) is False
    assert ic._holds_stored_values([{**ok, "0028|1052": "abc"}], sitk.sitkUInt16) is False


def test_the_simpleitk_path_states_nothing_in_stored_units(tmp_path):
    image, per_slice, _ = nio.read_image_and_tags(_pad(write_series(tmp_path / "s")))
    vol = ic._metadata(image, per_slice, source=None, source_digest=None)
    d = vol.metadata.extensions["dicom"]
    assert not {"BitsStored", "HighBit", "PixelPaddingValue"} & d["tags"].keys()
    assert vol.metadata.extensions["haversack"]["tags_version"] == 1


def test_the_simpleitk_path_of_stored_values_says_so(tmp_path):
    image, per_slice, _ = nio.read_image_and_tags(
        _pad(_enrich(write_series(tmp_path / "s"), rescale=False)))
    vol = ic._metadata(image, per_slice, source=None, source_digest=None)
    d = vol.metadata.extensions["dicom"]
    assert d["stored_values"] is True and d["tags"]["PixelPaddingValue"] == 0


def test_a_single_dicom_file_copy_takes_the_files_header(tmp_path):
    f = sorted(_enrich(write_series(tmp_path / "s", n=2)).iterdir())[0]
    copy = ic.transcode(f, tmp_path / "e")
    assert copy is not None
    d = _attrs(copy)["extensions"]
    assert "AnatomicRegionSequence" in d["dicom"]["tags"] and d["haversack"]["tags_version"] == 2


def test_an_export_drops_every_overlay_group(tmp_path):
    f = sorted(write_series(tmp_path / "s", n=2).iterdir())[0]
    ds = pydicom.dcmread(f)
    ds.add_new(0x60020010, "US", 6)
    ds.add_new(0x60020011, "US", 5)
    ds.add_new(0x60020102, "US", 12)
    ds.save_as(f, enforce_file_format=True)
    h = _nrrd_header(nio.convert(f, tmp_path / "x.nrrd"))
    assert "6002|" not in h and "0008|0060:=CT" in h


def test_an_export_of_a_stored_values_copy_keeps_what_is_true(tmp_path):
    copy = ic.transcode(_pad(_enrich(write_series(tmp_path / "s"), rescale=False)),
                        tmp_path / "e")
    h = _nrrd_header(nio.convert(copy, tmp_path / "x.nrrd"))
    assert "0028|0101:=16" in h and "0028|0120:=0" in h


def test_honest_metadata_leaves_non_dicom_keys():
    img = sitk.Image([2, 2, 2], sitk.sitkInt16)
    img.SetMetaData("ITK_original_spacing", "x")
    img.SetMetaData("descrip", "y")
    ic.honest_metadata(img, False)
    assert {"ITK_original_spacing", "descrip"} <= set(img.GetMetaDataKeys())


def test_honest_metadata_without_duckn_keeps_no_dicom_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckn.dicom_tags", None)
    img = sitk.Image([2, 2, 2], sitk.sitkInt16)
    img.SetMetaData("0008|0060", "CT")
    img.SetMetaData("0028|0120", "-2000")
    img.SetMetaData("ITK_original_spacing", "x")
    ic.honest_metadata(img, True)
    assert set(img.GetMetaDataKeys()) == {"ITK_original_spacing"}


@pytest.mark.parametrize("key", ["0008|0000", "0002|0010", "5000|0010", "6000|3000", "0019|1001",
                                 "0028|1052", "0028|0030"])
def test_honest_metadata_drops_what_a_header_must_not_restate(key):
    img = sitk.Image([2, 2, 2], sitk.sitkInt16)
    img.SetMetaData(key, "1")
    img.SetMetaData("0008|0060", "CT")
    ic.honest_metadata(img, True)
    assert set(img.GetMetaDataKeys()) == {"0008|0060"}


@pytest.mark.parametrize("key", ["0028|0101", "0028|0102", "0028|0106", "0028|0107", "0028|0108",
                                 "0028|0109", "0028|0110", "0028|0111", "0028|0120", "0028|0121",
                                 "0040|9096"])
def test_honest_metadata_keeps_stored_units_only_beside_stored_values(key):
    """A literal list (dicom-spec §5.10): parametrizing over duckn's own STORED_ENCODING would
    shrink with a mutant of it and pass."""
    for stored in (True, False):
        img = sitk.Image([2, 2, 2], sitk.sitkInt16)
        img.SetMetaData(key, "1")
        ic.honest_metadata(img, stored)
        assert (key in img.GetMetaDataKeys()) is stored


def test_a_reader_version_2_copy_is_stale():
    """Copies made under 2 carry stored-unit tags beside rescaled voxels (0a34e0a)."""
    assert ic.READER_VERSION >= 3


def test_a_non_dicom_file_with_private_keys_copies_none(tmp_path):
    img = sitk.Image([5, 6, 4], sitk.sitkInt16)
    img.SetMetaData("0008|0060", "CT")
    img.SetMetaData("0019|1001", "vendor")
    src = tmp_path / "x.nrrd"
    sitk.WriteImage(img, str(src))
    image, per_slice, files = nio.read_image_and_tags(src)
    assert files == []
    ext = ic._metadata(image, per_slice, source=None, source_digest=None).metadata.extensions
    tags = ext["dicom"]["tags"]
    assert tags.get("Modality") == "CT" and "00191001" not in tags
    assert ext["dicom"]["stored_values"] is False       # nothing says what these values are


def test_a_deflated_header_is_what_the_copy_is_written_with(tmp_path):
    copy = ic.transcode(write_series(tmp_path / "s"), tmp_path / "e")
    with zipfile.ZipFile(copy) as z:
        assert z.getinfo("zarr.json").compress_type == zipfile.ZIP_DEFLATED
        assert all(i.compress_type == zipfile.ZIP_STORED
                   for i in z.infolist() if i.filename.startswith("c/"))


def test_a_copy_states_the_convention_version(tmp_path):
    """duckn-spec §3.1: `version` should always be present - a copy wrote none."""
    copy = ic.transcode(write_series(tmp_path / "s"), tmp_path / "e")
    assert _attrs(copy)["version"] == "1.0"
