"""The input copy (docs/input-copy.md): a cached input decoded once and kept INSTEAD of the
original - one uncompressed zarr chunk in a zip, duckn's geometry, the DICOM tags SimpleITK
reports - read back by mapping that chunk.

What these hold, against the ways it could go wrong:

- the copy reads back as exactly the image the reader produced - voxels, pixel type, geometry -
  on a series, a slightly TILTED series (a sheared direction), a gzip NIfTI and a single file;
- an entry holds ONE form: the copy and no original; a refused input (uneven spacing), a failed
  copy, a label map, a disabled switch keep the original;
- a copy another reader version wrote is stale: a fetched entry is fetched again, an upload is
  gone; a crash leftover is never read;
- the mapped reader refuses every other layout (and the generic duckn reader takes over), and
  reads without duckn exactly as duckn does;
- the tags follow duckn's dicom-spec: keywords, JSON-native values, VM arrays, series vs per
  slice, the convention-captured tags left to the convention fields;
- stored compressed (``HAVERSACK_INPUT_COPY_COMPRESSION=zstd``) the copy is the same image with
  the same tags, in its own layout and format version; either form reads whatever the setting
  says now, an unknown setting keeps the original, and a file whose version and layout disagree
  is stale.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pydicom = pytest.importorskip("pydicom")
pytest.importorskip("duckn")
pytest.importorskip("zarr")

from duckn import dicom_tags as dt  # noqa: E402
from haversack import input_copy as ic  # noqa: E402
from haversack import io as nio  # noqa: E402
from haversack.content import ContentStore  # noqa: E402
from haversack.serve import SeriesCache  # noqa: E402

CT_SOP = "1.2.840.10008.5.1.4.1.1.2"


def write_series(folder: Path, n: int = 6, *, dz: float = 2.0, tilt_mm: float = 0.0,
                 steps=None) -> Path:
    """A small CT series with the tags a real one carries: a per-slice tube current, a
    rescale type, an empty numeric field, a multi-valued window. ``tilt_mm`` shifts each slice
    in y by that much per slice (a gantry tilt the reader accepts as a sheared direction);
    ``steps`` gives uneven slice positions (which the reader refuses)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    series, study, frame = generate_uid(), generate_uid(), generate_uid()
    zs = list(steps) if steps is not None else [31.0 + i * dz for i in range(n)]
    rng = np.random.default_rng(len(zs))
    for i, z in enumerate(zs):
        sop = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_SOP
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CT_SOP, sop
        ds.Modality, ds.Manufacturer, ds.KVP = "CT", "Fixture Medical", 120
        ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.FrameOfReferenceUID = series, study, frame
        ds.PatientID, ds.PatientName = "fixture-id", "Fixture^Patient"
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [-10.0, -12.0 + i * tilt_mm, float(z)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [0.75, 0.5]
        ds.SliceThickness = 2.0
        ds.XRayTubeCurrent = 100 + 10 * i
        ds.WindowCenter, ds.WindowWidth = [40, 400], [400, 1500]
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
        ds.ContrastBolusVolume = None                      # present, empty
        ds.Rows, ds.Columns = 6, 5
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 15, 1, 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = rng.integers(-100, 1200, (6, 5), dtype=np.int16).tobytes()
        ds.save_as(folder / f"IM{i:04d}.dcm", enforce_file_format=True)
    return folder


def _same(a, b, exact=True):
    """Identical voxels and pixel type; geometry identical, or - a tilted series - within
    input_copy.GEOMETRY_TOLERANCE (its sheared direction round-trips to one unit in the last place)."""
    if not (np.array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))
            and a.GetPixelID() == b.GetPixelID()):
        return False
    geo = ("GetOrigin", "GetSpacing", "GetDirection")
    if exact:
        return all(getattr(a, g)() == getattr(b, g)() for g in geo)
    return all(np.allclose(getattr(a, g)(), getattr(b, g)(), rtol=0, atol=ic.GEOMETRY_TOLERANCE)
               for g in geo)


@pytest.fixture
def mapped(monkeypatch):
    """The uncompressed, mapped form - not the default since 2026-09-25, so a test of its layout
    or of its duckn-free read asks for it."""
    monkeypatch.setenv(ic.COMPRESSION_ENV, "uncompressed")


def _nifti(path: Path, gz=True) -> Path:
    img = sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6))
    img.SetSpacing((0.8, 0.9, 2.5))
    img.SetOrigin((1.0, -2.0, 3.0))
    sitk.WriteImage(img, str(path), gz)
    return path


# -- the copy reads back as the image -----------------------------------------------------

@pytest.mark.parametrize("tilt", [0.0, 0.03], ids=["straight", "tilted"])
def test_a_series_copy_is_exactly_the_image_the_reader_produced(tmp_path, tilt):
    series = write_series(tmp_path / "s", tilt_mm=tilt)
    ref = nio.read_image(series)
    copy = ic.transcode(series, tmp_path / "entry", source="fixture:1", source_digest="sha256:x")
    assert copy == ic.copy_path(tmp_path / "entry") and copy.is_file()
    got = nio.read_image(copy)                     # through the one door every engine uses
    assert _same(got, ref, exact=not tilt)         # an axis-aligned grid: bit for bit
    if tilt:                                       # the sheared direction survived
        d = np.asarray(ref.GetDirection()).reshape(3, 3)
        assert abs(d[1, 2]) > 1e-6


def test_a_compressed_single_file_is_copied_and_a_raw_one_is_not(tmp_path):
    gz = _nifti(tmp_path / "a.nii.gz")
    copy = ic.transcode(gz, tmp_path / "e1")
    assert copy is not None and _same(nio.read_image(copy), nio.read_image(gz))
    raw = _nifti(tmp_path / "b.nrrd", gz=False)
    assert ic.transcode(raw, tmp_path / "e2") is None          # already its efficient form


def test_the_file_is_one_stored_chunk_with_the_stated_metadata(tmp_path, mapped):
    series = write_series(tmp_path / "s")
    copy = ic.transcode(series, tmp_path / "entry", source="fixture:1", source_digest="sha256:x")
    with zipfile.ZipFile(copy) as z:
        assert [i.filename for i in z.infolist()] == ["zarr.json", "c/0/0/0"]
        assert z.getinfo("c/0/0/0").compress_type == zipfile.ZIP_STORED     # mapped at its offset
        assert z.getinfo("zarr.json").compress_type == zipfile.ZIP_DEFLATED  # read whole
        meta = json.loads(z.read("zarr.json"))
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == meta["shape"]
    assert meta["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}}]
    d = meta["attributes"]["duckn"]
    assert d["space"] == "left-posterior-superior" and not d.get("value_transforms")
    assert d["axes"][0]["thickness"] == 2.0 and d["sample_units"] == "HU"
    h = d["extensions"]["haversack"]
    assert (h["kind"], h["version"], h["reader_version"]) == (ic.KIND, ic.FORMAT_VERSION,
                                                             ic.READER_VERSION)
    assert (h["source"], h["source_digest"]) == ("fixture:1", "sha256:x")


# -- the tags ------------------------------------------------------------------------------

def test_the_tags_are_duckns_encoding_split_by_series_and_slice(tmp_path):
    series = write_series(tmp_path / "s")
    copy = ic.transcode(series, tmp_path / "entry")
    with zipfile.ZipFile(copy) as z:
        d = json.loads(z.read("zarr.json"))["attributes"]["duckn"]
    tags = d["extensions"]["dicom"]["tags"]
    assert d["extensions"]["dicom"]["version"] == "1.0"
    assert tags["Modality"] == "CT" and tags["KVP"] == 120 and tags["PatientName"] == "Fixture^Patient"
    assert tags["WindowCenter"] == [40, 400]                  # VM 1-n: an array
    for captured in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
                     "SliceThickness", "RescaleSlope", "RescaleIntercept", "RescaleType", "Rows",
                     "Columns", "BitsAllocated", "PixelData"):
        assert captured not in tags, captured
    assert "ContrastBolusVolume" not in tags                  # empty number: absent, not null
    assert "XRayTubeCurrent" not in tags and "InstanceNumber" not in tags
    assert ic.slice_tags(copy, "XRayTubeCurrent") == [100 + 10 * i for i in range(6)]
    assert ic.slice_tags(copy, "InstanceNumber") == list(range(1, 7))
    img = nio.read_image(copy)                                # series tags back on the image
    assert img.GetMetaData("0008|0060") == "CT" and img.GetMetaData("0018|0060") == "120"


def test_the_encoder_follows_the_spec_rules():
    series, slices = dt.tags_from_sitk([
        {"0008|0060": "CT", "0018|0050": "2.0", "0028|1050": "40\\400", "0011|1001": "vendor ",
         "0020|0013": "1", "0008|0000": "1234", "ITK_original_spacing": "x", "0018|1151": "100"},
        {"0008|0060": "CT", "0018|0050": "2.0", "0028|1050": "40\\400", "0011|1001": "vendor ",
         "0020|0013": "2", "0008|0000": "1234", "ITK_original_spacing": "x", "0018|1151": "110"},
    ])
    assert series == {"Modality": "CT", "WindowCenter": [40, 400], "00111001": "vendor"}
    assert slices == [{"InstanceNumber": 1, "XRayTubeCurrent": 100},
                      {"InstanceNumber": 2, "XRayTubeCurrent": 110}]
    assert dt.to_sitk_strings({"Modality": "CT", "KVP": 120.0, "WindowCenter": [40, 400]}) == {
        "0008|0060": "CT", "0018|0060": "120", "0028|1050": "40\\400"}



def _enrich(series: Path, *, rescale: bool = True) -> Path:
    """What SimpleITK's dictionaries do not hold, added to every file: a sequence, a binary value
    (an ICC profile), a private block with its creator, and an empty text value; ``rescale=False``
    takes the rescale out, so the copy holds the stored values themselves."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    for i, f in enumerate(sorted(series.iterdir())):
        ds = pydicom.dcmread(f)
        code = Dataset()
        code.CodeValue, code.CodingSchemeDesignator, code.CodeMeaning = "T-D3000", "SRT", "Chest"
        ds.AnatomicRegionSequence = Sequence([code])
        ds.add_new(0x00282000, "OB", b"\x01\x02\x03\x04")
        ds.add_new(0x00190010, "LO", "ACME 1.0")
        ds.add_new(0x00191001, "DS", str(1000 + i))           # a private per-slice number
        ds.StudyDescription = ""
        if not rescale:
            for k in ("RescaleIntercept", "RescaleSlope", "RescaleType"):
                delattr(ds, k)
            ds.PixelRepresentation = 0
            ds.PixelData = np.arange(30, dtype=np.uint16).reshape(6, 5).tobytes()
        ds.save_as(f, enforce_file_format=True)
    return series


def _attrs(copy: Path) -> dict:
    with zipfile.ZipFile(copy) as z:
        return json.loads(z.read("zarr.json"))["attributes"]["duckn"]


@pytest.mark.parametrize("form", ["uncompressed", "zstd"])
def test_the_tags_are_the_files_own_headers(tmp_path, monkeypatch, form):
    """TAGS_VERSION 2 (2026-09-26): sequences, binary values, private tags - through pydicom,
    whether the copy is written whole (uncompressed) or a slab at a time (zstd)."""
    monkeypatch.setenv(ic.COMPRESSION_ENV, form)
    copy = ic.transcode(_enrich(write_series(tmp_path / "s")), tmp_path / "entry")
    d = _attrs(copy)
    ext = d["extensions"]
    tags = ext["dicom"]["tags"]
    assert tags["AnatomicRegionSequence"] == [
        {"CodeValue": "T-D3000", "CodingSchemeDesignator": "SRT", "CodeMeaning": "Chest"}]
    assert tags["ICCProfile"] == "AQIDBA=="
    assert tags["StudyDescription"] == ""                     # empty text: "", never null
    # no private element: haversack cannot vouch for a vendor's data; the source keeps it
    assert "00190010" not in tags and not any(ic.slice_tags(copy, "00191001"))
    assert ic.slice_tags(copy, "SOPInstanceUID")[0]           # per-slice identity kept
    assert ext["dicom"]["source_transfer_syntax"] == "1.2.840.10008.1.2.1"
    assert not {"TransferSyntaxUID", "MediaStorageSOPInstanceUID"} & tags.keys()
    assert ext["haversack"]["tags_version"] == ic.TAGS_VERSION == 2
    # rescaled to HU (and widened): the stored-value bits are no longer true of the copy
    assert not {"BitsStored", "HighBit", "BitsAllocated", "PixelRepresentation"} & tags.keys()


def test_both_forms_record_the_same_tags(tmp_path, monkeypatch):
    series = _enrich(write_series(tmp_path / "s"))
    got = []
    for form in ("uncompressed", "zstd"):
        monkeypatch.setenv(ic.COMPRESSION_ENV, form)
        d = _attrs(ic.transcode(series, tmp_path / form))
        got.append((d["extensions"]["dicom"], [s.get("metadata") for s in d["axes"][0]["samples"]]))
    assert got[0] == got[1]


def test_a_copy_of_stored_values_keeps_bits_stored(tmp_path):
    copy = ic.transcode(_enrich(write_series(tmp_path / "s"), rescale=False), tmp_path / "entry")
    tags = _attrs(copy)["extensions"]["dicom"]["tags"]
    assert (tags["BitsStored"], tags["HighBit"]) == (16, 15)
    assert "BitsAllocated" not in tags


@pytest.mark.parametrize("form", ["uncompressed", "zstd"])
def test_a_copy_written_with_a_stored_header_still_reads_and_is_not_stale(tmp_path, monkeypatch,
                                                                          form):
    """Copies from before 2026-09-26 have a STORED zarr.json: the same file to every reader."""
    monkeypatch.setenv(ic.COMPRESSION_ENV, form)
    series = write_series(tmp_path / "s")
    copy = ic.transcode(series, tmp_path / "entry")
    old = tmp_path / "old" / ic.COPY_DIR / ic.COPY_NAME
    old.parent.mkdir(parents=True)
    with zipfile.ZipFile(copy) as src, zipfile.ZipFile(old, "w", zipfile.ZIP_STORED) as dst:
        for i in src.infolist():
            dst.writestr(i.filename, src.read(i.filename))
    assert not ic.stale(old) and ic.stored_compression(old) == form
    assert _same(nio.read_image(old), nio.read_image(copy))


def test_the_header_is_deflated_where_the_tags_are_large(tmp_path):
    """The reason for the deflated header: tags repeat. (Measured on Siemens' private per-slice
    blocks, which copies no longer carry; a large public binary value stands in.)"""
    series = write_series(tmp_path / "s", n=20)
    for f in sorted(series.iterdir()):
        ds = pydicom.dcmread(f)
        ds.add_new(0x00282000, "OB", bytes(range(256)) * 200)    # ICC Profile, 51 KB
        ds.save_as(f, enforce_file_format=True)
    copy = ic.transcode(series, tmp_path / "entry")
    with zipfile.ZipFile(copy) as z:
        h = z.getinfo("zarr.json")
    assert h.file_size > 60_000 and h.compress_size < h.file_size / 10


def _pad(series: Path) -> Path:
    for f in sorted(series.iterdir()):
        ds = pydicom.dcmread(f)
        ds.add_new(0x00280120, "SS" if ds.PixelRepresentation else "US",
                   -2000 if ds.PixelRepresentation else 0)       # Pixel Padding Value
        ds.add_new(0x00280107, "SS" if ds.PixelRepresentation else "US", 1199)  # Largest
        ds.save_as(f, enforce_file_format=True)
    return series


@pytest.mark.parametrize("form", ["uncompressed", "zstd"])
def test_nothing_the_file_says_contradicts_its_voxels(tmp_path, monkeypatch, form):
    """The user's rule (2026-09-26): the header must not contradict the data a reader gets from
    this file alone. A CT's padding -2000 is in STORED units; the copy holds HU (intercept
    -1024), where it is -3024 - so the stored-unit attributes are not written, and the file
    says its values are not the stored ones."""
    monkeypatch.setenv(ic.COMPRESSION_ENV, form)
    copy = ic.transcode(_pad(write_series(tmp_path / "s")), tmp_path / "entry")
    d = _attrs(copy)["extensions"]["dicom"]
    per_slice = {k for s in _attrs(copy)["axes"][0].get("samples") or []
                 for k in ((s.get("metadata") or {}).get("dicom") or {})}
    everything = d["tags"].keys() | per_slice
    assert not {"PixelPaddingValue", "LargestImagePixelValue", "BitsStored"} & everything
    assert d["stored_values"] is False


@pytest.mark.parametrize("form", ["uncompressed", "zstd"])
def test_the_padding_is_recorded_in_the_copys_own_units(tmp_path, monkeypatch, form):
    """2026-09-30: with the stored-unit Pixel Padding Value rightly gone, a materialized copy
    lost which voxels were padding. It records them in its own units: -2000 stored, under an
    intercept of -1024, is -3024 in the copy's HU; and Input.missing() says so."""
    from haversack.inputs import Input
    monkeypatch.setenv(ic.COMPRESSION_ENV, form)
    series = _pad(write_series(tmp_path / "s"))
    f = sorted(series.iterdir())[2]
    ds = pydicom.dcmread(f)
    px = ds.pixel_array.copy()
    px[0, :3] = -2000
    ds.PixelData = px.tobytes()
    ds.save_as(f, enforce_file_format=True)
    copy = ic.transcode(series, tmp_path / "entry")
    assert ic.info(copy)["missing"] == [-3024]
    inp = Input(None, copy, None)
    assert inp.missing() == [-3024]
    marked = np.isin(inp.array(), inp.missing())
    assert marked.sum() == 3 and marked[2, 0, :3].all()      # exactly the padded voxels


def test_a_copy_of_stored_values_records_the_padding_as_it_is(tmp_path):
    copy = ic.transcode(_pad(_enrich(write_series(tmp_path / "s"), rescale=False)),
                        tmp_path / "entry")
    assert ic.info(copy)["missing"] == [0]                    # identity: the stored value


@pytest.mark.parametrize("edit,why", [
    (lambda i, ds: setattr(ds, "RescaleSlope", 1 + (i == 3)), "a rescale that varies by slice"),
    (lambda i, ds: ds.add_new(0x00280121, "SS", -1990), "a padding range"),
    (lambda i, ds: None, "no padding"),
])
def test_no_padding_is_recorded_where_no_single_value_can_carry_it(tmp_path, edit, why):
    series = write_series(tmp_path / "s")
    if why != "no padding":
        _pad(series)
    for i, f in enumerate(sorted(series.iterdir())):
        ds = pydicom.dcmread(f)
        edit(i, ds)
        ds.save_as(f, enforce_file_format=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        copy = ic.transcode(series, tmp_path / "entry")
    assert "missing" not in ic.info(copy), why


def test_a_us_coded_padding_value_of_a_signed_image_is_read_as_signed():
    per = [{"0028|0120": "63536", "0028|1053": "1", "0028|1052": "-1024", "0028|0100": "16",
            "0028|0103": "1", "0028|0004": "MONOCHROME2"}]
    assert ic._missing(per) == [-3024]
    assert ic._missing(per, modality_lut=True) is None
    assert ic._missing(per, value_transform=True) is None
    assert ic._missing([dict(per[0], **{"0028|0004": "MONOCHROME1"})]) is None
    assert ic._missing([dict(per[0], **{"0028|1053": "0.5"})]) == [-2024]  # 63536 -> -2000


def test_a_copy_of_stored_values_states_them_and_says_so(tmp_path):
    copy = ic.transcode(_pad(_enrich(write_series(tmp_path / "s"), rescale=False)),
                        tmp_path / "entry")
    d = _attrs(copy)["extensions"]["dicom"]
    assert d["stored_values"] is True
    assert d["tags"]["PixelPaddingValue"] == 0 and d["tags"]["LargestImagePixelValue"] == 1199


def test_the_simpleitk_path_says_which_values_it_holds(tmp_path):
    """Tags from SimpleITK's dictionaries (no file list: a caller's own image) state it too."""
    image, per_slice, _ = nio.read_image_and_tags(write_series(tmp_path / "s"))
    vol = ic._metadata(image, per_slice, source=None, source_digest=None)
    assert vol.metadata.extensions["dicom"]["stored_values"] is False


def _one_file(tmp_path, rescale=True) -> Path:
    series = _pad(_enrich(write_series(tmp_path / "s", n=2), rescale=rescale))
    return sorted(series.iterdir())[0]


def _nrrd_header(path: Path) -> str:
    return path.read_bytes().split(b"\n\n")[0].decode(errors="replace")


def test_an_exported_file_says_nothing_its_pixels_contradict(tmp_path):
    """`get -o x.nrrd` of a single DICOM file: SimpleITK rescales the pixels and keeps the
    file's Rescale Intercept -1024 and Pixel Padding Value -2000, and writes both into the NRRD
    header - a reader would rescale twice and mask the wrong value. They are gone."""
    out = nio.convert(_one_file(tmp_path), tmp_path / "x.nrrd")
    h = _nrrd_header(out)
    for key in ("0028|1052", "0028|1053", "0028|0120", "0028|0107", "0028|0101", "0019|"):
        assert key not in h, key
    assert "0008|0060:=CT" in h                               # the rest of the header stays


def test_an_exported_file_of_stored_values_keeps_what_is_true(tmp_path):
    out = nio.convert(_one_file(tmp_path, rescale=False), tmp_path / "x.nrrd")
    h = _nrrd_header(out)
    assert "0028|0120:=0" in h and "0028|0101:=16" in h       # true of these pixels


def test_a_copy_that_does_not_say_what_it_holds_is_read_as_rescaled(tmp_path):
    """A copy from before the field: its restored tags are judged as beside rescaled voxels."""
    copy = ic.transcode(_pad(write_series(tmp_path / "s")), tmp_path / "entry")
    tags = {"PixelPaddingValue": -2000, "BitsStored": 16, "Modality": "CT", "00191001": "x"}
    image = ic._with_tags(sitk.Image([2, 2, 2], sitk.sitkInt32),
                          {"extensions": {"dicom": {"tags": tags}}})
    assert set(image.GetMetaDataKeys()) == {"0008|0060"}
    assert ic.stored_values_of(copy) is False


def test_only_public_text_tags_go_back_onto_the_image(tmp_path):
    """A SimpleITK read of the files shows no private tags; a header written from the image
    must never carry a base64 profile or a private block."""
    img = nio.read_image(ic.transcode(_enrich(write_series(tmp_path / "s")), tmp_path / "entry"))
    keys = set(img.GetMetaDataKeys())
    assert "0008|0060" in keys
    assert not {"0028|2000", "0019|0010"} & keys
    assert "0008|2218" not in keys                            # a sequence has no string form

# -- one form per entry ----------------------------------------------------------------------

def _fetching_cache(tmp_path, make):
    def fetch(key, entry):
        return make(Path(entry) / "series")
    return SeriesCache(tmp_path / "cache", fetch)


def test_a_fetched_entry_keeps_the_copy_and_not_the_original(tmp_path):
    cache = _fetching_cache(tmp_path, lambda d: write_series(d))
    got = cache.get_or_fetch("fixture:series-1")
    entry = cache.entry("fixture:series-1")
    assert got == ic.copy_path(entry) and cache.path("fixture:series-1") == got
    assert not (entry / "series").exists()
    committed = int((entry / cache.MARKER).read_text())
    assert committed == got.stat().st_size                    # the budget counts the copy
    assert cache.get_or_fetch("fixture:series-1") == got      # a hit returns it again


def test_what_cannot_be_copied_keeps_its_original(tmp_path, monkeypatch):
    # uneven spacing: the reader refuses it, so the original stays and fails at read as before
    uneven = _fetching_cache(tmp_path / "u", lambda d: write_series(d, steps=[0, 2, 4, 7, 9]))
    got = uneven.get_or_fetch("fixture:uneven")
    assert got.name == "series" and got.is_dir()
    assert not ic.copy_path(uneven.entry("fixture:uneven")).exists()
    # a result: reference is a label map and is never transcoded
    labels = _fetching_cache(tmp_path / "r", lambda d: (d.mkdir(parents=True), _nifti(d / "l.nii.gz"), d)[2])
    assert labels.get_or_fetch("result:" + "a" * 64).name == "series"
    # a failed copy (its read-back disagrees) keeps the original, and leaves no partial file
    monkeypatch.setattr(ic, "read_copy", lambda *a, **k: sitk.Image(1, 1, 1, sitk.sitkInt16))

    def disagree(*a):                        # the slab path's check (2026-09-25)
        raise ValueError("chunk 0 does not read back as the slab written")
    monkeypatch.setattr(ic, "_check_streamed", disagree)
    broken = _fetching_cache(tmp_path / "b", lambda d: write_series(d))
    got = broken.get_or_fetch("fixture:b")
    assert got.name == "series" and not list(ic.copy_path(broken.entry("fixture:b")).parent.glob("*"))


def test_the_operator_switch_keeps_originals(tmp_path, monkeypatch):
    monkeypatch.setenv(ic.ENV, "0")
    cache = _fetching_cache(tmp_path, lambda d: write_series(d))
    assert cache.get_or_fetch("fixture:off").name == "series"


def test_an_upload_is_stored_as_its_copy(tmp_path):
    store = ContentStore(SeriesCache(tmp_path / "cache", lambda k, e: None))
    gz = _nifti(tmp_path / "up.nii.gz")
    ref = nio.read_image(gz)
    d = store.put_file(gz)
    got = store.resolve(d)
    assert ic.is_copy(got) and store.fast_path(d) == got and _same(nio.read_image(got), ref)
    assert not (store.cache.entry(d) / "series").exists()
    assert store.put_file(gz) == d and store.resolve(d) == got     # the same bytes: one entry


# -- staleness, crashes -------------------------------------------------------------------------

def test_a_copy_another_reader_version_wrote_is_stale(tmp_path, monkeypatch):
    calls = []

    def make(d):
        calls.append(1)
        return write_series(d)
    cache = _fetching_cache(tmp_path, make)
    old = cache.get_or_fetch("fixture:v")
    monkeypatch.setattr(ic, "READER_VERSION", ic.READER_VERSION + 1)
    assert not cache.has("fixture:v")                          # the version is in the name
    got = cache.get_or_fetch("fixture:v")                     # a fetched input: fetched again
    assert len(calls) == 2 and not ic.stale(got) and got != old
    # an upload has nothing to fetch it from: it is gone (the server answers input_gone) -
    # and the same bytes uploaded again replace the dead entry
    store = ContentStore(SeriesCache(tmp_path / "up", lambda k, e: None))
    src = _nifti(tmp_path / "u.nii.gz")
    d = store.put_file(src)
    monkeypatch.setattr(ic, "READER_VERSION", ic.READER_VERSION + 2)
    assert not store.has(d)
    assert store.put_file(src) == d and store.has(d) and not ic.stale(store.resolve(d))


def test_a_crash_leftover_is_never_read(tmp_path):
    entry = tmp_path / "entry"
    leftover = ic.copy_path(entry).with_name("." + ic.COPY_NAME + ".partial")
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"PK\x03\x04TRUNCATED")
    assert not ic.copy_path(entry).exists() and not ic.is_copy(leftover)
    cache = SeriesCache(tmp_path / "c", lambda k, e: write_series(Path(e) / "series"))
    got = cache.get_or_fetch("fixture:crash")
    assert ic.is_copy(got) and not list(got.parent.glob(".*.partial"))


# -- the reader ---------------------------------------------------------------------------------

def _rewrite(copy: Path, edit) -> Path:
    """The same file with zarr.json edited (and the chunk carried over)."""
    with zipfile.ZipFile(copy) as z:
        meta = json.loads(z.read("zarr.json"))
        chunk = z.read("c/0/0/0")
    edit(meta)
    out = copy.with_name(ic.COPY_NAME)
    tmp = copy.with_name("rewrite.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("zarr.json", json.dumps(meta))
        z.writestr("c/0/0/0", chunk)
    tmp.replace(out)
    return out


@pytest.mark.parametrize("edit", [
    lambda m: m["codecs"].append({"name": "zstd", "configuration": {"level": 1}}),
    lambda m: m["codecs"].append({"name": "blosc", "configuration": {
        "cname": "lz4", "clevel": 5, "shuffle": "shuffle", "typesize": 2, "blocksize": 0}}),
    lambda m: m["chunk_grid"]["configuration"].update(chunk_shape=[1] + m["shape"][1:]),
    lambda m: m["attributes"]["duckn"].update(value_transforms=[{"type": "linear", "slope": 2.0, "intercept": 0}]),
    lambda m: m["attributes"]["duckn"]["axes"][0].pop("space_direction"),
], ids=["zstd-codec", "blosc-over-raw-bytes", "chunked", "rescale", "no-geometry"])
def test_the_mapped_reader_refuses_any_other_layout(tmp_path, edit, mapped):
    copy = ic.transcode(write_series(tmp_path / "s"), tmp_path / "entry")
    bad = _rewrite(copy, edit)
    with pytest.raises(ic.NotACopy):
        ic.read_copy(bad)


@pytest.mark.parametrize("compressed", [False, True], ids=["mapped", "zstd"])
def test_a_deflated_zip_is_refused(tmp_path, monkeypatch, compressed):
    """The mapped reader takes the member's bytes at its offset as the voxels: a DEFLATED member
    would hand it deflate output under the array's size. Either layout must be a stored zip."""
    if compressed:
        monkeypatch.setenv(ic.COMPRESSION_ENV, "zstd")
    copy = ic.transcode(write_series(tmp_path / "s"), tmp_path / "entry")
    out = tmp_path / "x" / ic.COPY_DIR / ic.COPY_NAME
    out.parent.mkdir(parents=True)
    with zipfile.ZipFile(copy) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for i in src.infolist():
            dst.writestr(i.filename, src.read(i.filename))
    with pytest.raises(ic.NotACopy):
        ic.read_copy(out)


def test_a_file_that_only_looks_like_a_copy_is_read_by_duckn(tmp_path):
    """Fallback, never an error: a well-formed duckn store under the copy's name whose layout
    the mapped reader does not take (two chunks) is read by duckn's own reader, correctly."""
    series = write_series(tmp_path / "s")
    ref = nio.read_image(series)
    copy = ic.transcode(series, tmp_path / "entry")
    import zarr
    from zarr.storage import ZipStore
    with zipfile.ZipFile(copy) as z:
        attrs = json.loads(z.read("zarr.json"))["attributes"]
    arr = sitk.GetArrayFromImage(ref)
    copy.unlink()
    st = ZipStore(str(copy), mode="w")
    z = zarr.create_array(st, shape=arr.shape, dtype=arr.dtype, chunks=(1,) + arr.shape[1:],
                          compressors=None, attributes=attrs, fill_value=0)
    z[:] = arr
    st.close()
    with pytest.raises(ic.NotACopy):
        ic.read_copy(copy)
    assert _same(nio.read_image(copy), ref)


def test_without_duckn_the_copy_reads_exactly_as_duckn_reads_it(tmp_path, monkeypatch, mapped):
    """An engine environment that cannot install duckn (VoxTell's, MONAI's) still reads a copy
    another container wrote - with the geometry duckn's to_sitk gives, and no other. It reads no
    DICOM tags back: the converter is duckn's (``duckn.dicom_tags``, since duckn 0.5.2), and the
    tags are provenance, never needed to compute."""
    import builtins
    copy = ic.transcode(write_series(tmp_path / "s", tilt_mm=0.03), tmp_path / "entry")
    with_duckn = ic.read_copy(copy)
    real = builtins.__import__

    def no_duckn(name, *a, **k):
        if name == "duckn" or name.startswith("duckn."):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_duckn)
    without = ic.read_copy(copy)
    assert _same(without, with_duckn)             # bit for bit: the same arithmetic as duckn's
    assert with_duckn.GetMetaData("0008|0060") == "CT"
    assert not without.HasMetaDataKey("0008|0060")


def test_a_label_map_kept_as_a_copy_still_reads_as_one(tmp_path):
    from haversack.labelmap import read_label_map
    lm = sitk.GetImageFromArray(np.array([[[0, 1], [2, 1]]] * 3, dtype=np.uint8))
    src = tmp_path / "lm.nii.gz"
    sitk.WriteImage(lm, str(src), True)
    copy = ic.transcode(src, tmp_path / "entry")
    got = read_label_map(copy, require_names=False)
    assert np.array_equal(got.array, sitk.GetArrayFromImage(lm)) and got.names == {}


def test_the_real_series_when_it_is_here():
    """The 709-slice CT the measurements were taken on, when this machine has it."""
    series = Path.home() / "tmp/data/idc-torso1_dicom"
    if not series.is_dir():
        pytest.skip("the local CT is not here")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ref = nio.read_image(series)
        copy = ic.transcode(series, Path(d) / "entry")
        assert _same(nio.read_image(copy), ref)
        assert ic.slice_tags(copy, "XRayTubeCurrent")[:1] == [148]



def test_a_job_on_a_stored_copy_keys_and_computes_as_on_the_original(tmp_path, monkeypatch):
    """The copy never reaches a result's identity: the same upload keys the same result with
    the copy and without it (the operator switch off), and the job reads the same image."""
    import hashlib
    from fastapi.testclient import TestClient
    from test_serve import FakeSegmenter, wait_state

    from haversack.serve import LocalExecutor, create_app
    raw = _nifti(tmp_path / "v.nii.gz").read_bytes()
    seen = {}
    for label, on in (("copy", "1"), ("original", "0")):
        monkeypatch.setenv(ic.ENV, on)
        seg = FakeSegmenter(steps=1)
        ex = LocalExecutor(seg, workdir=tmp_path / label, cache_dir=tmp_path / (label + "-c"))
        client = TestClient(create_app(ex))
        d = "sha256:" + hashlib.sha256(raw).hexdigest()
        assert client.put(f"/v1/inputs/{d}", content=raw).status_code == 200
        r = client.post("/v1/jobs", data={"task": "total_fast",
                                          "source": json.dumps([{"kind": "input", "sha256": d}])})
        s = wait_state(client, r.json()["id"], ("done", "failed"))
        assert s["state"] == "done", s
        stored = ex.content.resolve(d)
        assert ic.is_copy(stored) == (on == "1")
        seen[label] = (s["key"], nio.read_image(seg.inputs[0]) if not hasattr(seg.inputs[0], "GetSize")
                       else seg.inputs[0])
        ex.close()
    assert seen["copy"][0] == seen["original"][0]
    assert _same(seen["copy"][1], seen["original"][1])


def test_a_label_map_with_its_names_is_kept_as_it_was(tmp_path):
    """A .seg.nrrd carries its segment names and codes in its file - and any NRRD carrying
    segment fields, whatever its name, is one. The copy would lose them: it is never made."""
    lm = sitk.GetImageFromArray(np.array([[[0, 1], [2, 1]]] * 3, dtype=np.uint8))
    lm.SetMetaData("Segment0_Name", "liver")
    lm.SetMetaData("Segment0_LabelValue", "1")
    for name in ("lm.seg.nrrd", "renamed.nrrd"):
        p = tmp_path / name
        sitk.WriteImage(lm, str(p), True)                  # compressed: would otherwise qualify
        assert ic.transcode(p, tmp_path / ("e-" + name)) is None, name
    store = ContentStore(SeriesCache(tmp_path / "cache", lambda k, e: None))
    d = store.put_file(tmp_path / "lm.seg.nrrd")
    kept = store.resolve(d)
    assert not ic.is_copy(kept) and kept.parent.name == "series"
    assert sitk.ReadImage(str(kept)).GetMetaData("Segment0_Name") == "liver"


def test_the_prefetcher_stores_the_copy_too(tmp_path):
    cache = _fetching_cache(tmp_path, lambda d: write_series(d))
    assert cache.prefetch("fixture:pre")
    entry = cache.entry("fixture:pre")
    assert ic.is_copy(cache.path("fixture:pre")) and not (entry / "series").exists()


# -- compressed copies ---------------------------------------------------------------------------

@pytest.fixture
def zstd(monkeypatch):
    monkeypatch.setenv(ic.COMPRESSION_ENV, "zstd")
    monkeypatch.setattr(ic, "CHUNK_SLICES", 4)          # 6 slices: a full chunk and a short one


@pytest.mark.parametrize("tilt", [0.0, 0.03], ids=["straight", "tilted"])
def test_a_compressed_copy_is_the_same_image_with_the_same_tags(tmp_path, zstd, tilt):
    series = write_series(tmp_path / "s", tilt_mm=tilt)
    ref = nio.read_image(series)
    copy = ic.transcode(series, tmp_path / "entry")
    got = nio.read_image(copy)
    assert _same(got, ref, exact=not tilt)
    assert got.GetMetaData("0008|0060") == "CT" and got.GetMetaData("0018|0060") == "120"
    assert ic.slice_tags(copy, "XRayTubeCurrent") == [100 + 10 * i for i in range(6)]
    assert ic.stored_compression(copy) == "zstd" and not ic.stale(copy)


def test_a_compressed_copy_has_its_own_layout_and_version(tmp_path, zstd):
    copy = ic.transcode(write_series(tmp_path / "s"), tmp_path / "entry")
    with zipfile.ZipFile(copy) as z:
        names = sorted(i.filename for i in z.infolist())
        assert all(i.compress_type == zipfile.ZIP_STORED
                   for i in z.infolist() if i.filename.startswith("c/"))
        assert z.getinfo("zarr.json").compress_type == zipfile.ZIP_DEFLATED
        meta = json.loads(z.read("zarr.json"))
    assert names == ["c/0/0/0", "c/1/0/0", "zarr.json"]
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [4, 6, 5]
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "blosc"]
    blosc = meta["codecs"][1]["configuration"]
    assert (blosc["cname"], blosc["clevel"], blosc["shuffle"]) == ("zstd", ic.ZSTD_LEVEL, "bitshuffle")
    assert meta["attributes"]["duckn"]["extensions"]["haversack"]["version"] == ic.FORMATS["zstd"] == 2


def test_either_form_reads_whatever_the_setting_says_now(tmp_path, monkeypatch):
    series = write_series(tmp_path / "s")
    ref = nio.read_image(series)
    monkeypatch.setenv(ic.COMPRESSION_ENV, "uncompressed")
    plain = ic.transcode(series, tmp_path / "a")
    monkeypatch.setenv(ic.COMPRESSION_ENV, "zstd")
    packed = ic.transcode(series, tmp_path / "b")
    assert (ic.stored_compression(plain), ic.stored_compression(packed)) == ("uncompressed", "zstd")
    for setting in ("zstd", "uncompressed"):
        monkeypatch.setenv(ic.COMPRESSION_ENV, setting)
        for copy in (plain, packed):
            assert not ic.stale(copy) and _same(nio.read_image(copy), ref)


def test_an_unknown_setting_keeps_the_original(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ic.COMPRESSION_ENV, "lz77")
    cache = _fetching_cache(tmp_path, lambda d: write_series(d))
    assert cache.get_or_fetch("fixture:odd").name == "series"
    assert ic.COMPRESSION_ENV in capsys.readouterr().err


def test_a_version_that_disagrees_with_the_layout_is_stale(tmp_path, zstd, monkeypatch):
    packed = ic.transcode(write_series(tmp_path / "s"), tmp_path / "a")
    monkeypatch.setattr(ic, "FORMATS", {"uncompressed": 1, "zstd": 1})     # as a version-1 reader sees it
    assert ic.stale(packed)
    monkeypatch.setattr(ic, "FORMATS", {"uncompressed": 2, "zstd": 2})
    monkeypatch.setenv(ic.COMPRESSION_ENV, "uncompressed")
    plain = ic.transcode(write_series(tmp_path / "t"), tmp_path / "b")
    monkeypatch.setattr(ic, "FORMATS", {"uncompressed": 1, "zstd": 2})
    assert ic.stale(plain)                                          # says 2, is a mapped chunk


def test_a_compressed_copy_missing_a_chunk_is_refused(tmp_path, zstd):
    packed = ic.transcode(write_series(tmp_path / "s"), tmp_path / "entry")
    out = tmp_path / "x" / ic.COPY_DIR / ic.COPY_NAME
    out.parent.mkdir(parents=True)
    with zipfile.ZipFile(packed) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as dst:
        for i in src.infolist():
            if i.filename != "c/1/0/0":
                dst.writestr(i.filename, src.read(i.filename))
    with pytest.raises(ic.NotACopy):
        ic.read_copy(out)


def test_the_default_is_the_compressed_form(tmp_path, monkeypatch):
    """Compressed by default (2026-09-25): with the setting unset, a copy is the blosc-zstd
    layout, and `uncompressed` still gives the mapped one."""
    monkeypatch.delenv(ic.COMPRESSION_ENV, raising=False)
    series = write_series(tmp_path / "s")
    assert ic.compression() == ic.DEFAULT_COMPRESSION == "zstd"
    assert ic.stored_compression(ic.transcode(series, tmp_path / "a")) == "zstd"
    monkeypatch.setenv(ic.COMPRESSION_ENV, "uncompressed")
    assert ic.stored_compression(ic.transcode(series, tmp_path / "b")) == "uncompressed"
