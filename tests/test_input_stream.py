"""The input copy written slab by slab (2026-09-25, :mod:`haversack.input_stream`).

What these hold, against the ways a streamed copy could differ from the whole-volume one:

- it IS the whole-volume copy: the same zarr.json (bar the reader's own version stamp) and every
  compressed chunk byte for byte - on a series, a tilted one, one stored in descending order
  under shuffled file names, one whose slices carry different rescales, and gzipped NIfTIs of
  both byte orders;
- the header-only listing gives GDCM's order;
- what only the whole read can promise falls back to it: two series, a scaled or 4-D NIfTI,
  the uncompressed form; what the reader refuses is refused;
- the check catches a chunk that is not its slab, and a source that ends early keeps the original;
- memory is bounded by a slab: a streamed transcode's peak is a fraction of the whole one's.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import struct
import subprocess
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

from test_input_copy import write_series  # noqa: E402


@pytest.fixture(autouse=True)
def zstd(monkeypatch):
    monkeypatch.setenv(ic.COMPRESSION_ENV, "zstd")
    monkeypatch.setattr(ic, "CHUNK_SLICES", 4)          # several slabs from a small fixture


def _both(src, tmp_path, monkeypatch):
    """(whole-volume copy, streamed copy) of ``src``."""
    streamed = ic.transcode(src, tmp_path / "streamed")
    with monkeypatch.context() as m:
        m.setattr(input_stream, "stream_of", lambda content: None)
        whole = ic.transcode(src, tmp_path / "whole")
    assert whole is not None and streamed is not None
    return whole, streamed


def _same_file(a: Path, b: Path) -> None:
    za, zb = zipfile.ZipFile(a), zipfile.ZipFile(b)
    ma, mb = (json.loads(z.read("zarr.json")) for z in (za, zb))
    for m in (ma, mb):
        m["attributes"]["duckn"]["extensions"]["haversack"].pop("reader")
    assert ma == mb
    assert sorted(za.namelist()) == sorted(zb.namelist())
    for n in za.namelist():
        if n != "zarr.json":
            assert za.read(n) == zb.read(n), n


def _shuffle_names(folder: Path) -> Path:
    files = sorted(folder.iterdir())
    names = [f"x{random.Random(7).random():.6f}_{i}.dcm" for i in range(len(files))]
    random.Random(3).shuffle(names)
    for f, n in zip(files, names):
        f.rename(folder / n)
    return folder


def _vary_rescale(folder: Path, fractional_from=None) -> Path:
    """Different rescales per slice; from ``fractional_from`` on, a fractional slope, so a slab
    starting there would read as a float on its own - the whole read takes the FIRST file's type."""
    for i, f in enumerate(sorted(folder.glob("*.dcm"))):
        ds = pydicom.dcmread(f)
        ds.RescaleIntercept = -1024 + 7 * i
        frac = fractional_from is not None and i >= fractional_from
        ds.RescaleSlope = 0.5 if frac else 1 + (i % 3)
        ds.save_as(f)
    return folder


@pytest.mark.parametrize("make", [
    lambda d: write_series(d, n=10),
    lambda d: write_series(d, n=10, tilt_mm=0.03),
    lambda d: _shuffle_names(write_series(d, steps=[31.0 - 2 * i for i in range(10)])),
    lambda d: _vary_rescale(write_series(d, n=10)),
    lambda d: _vary_rescale(write_series(d, n=10), fractional_from=4),
], ids=["series", "tilted", "descending-shuffled", "per-slice-rescale", "later-slabs-fractional"])
def test_a_streamed_series_copy_is_the_whole_copy(tmp_path, monkeypatch, make):
    src = make(tmp_path / "s")
    assert input_stream.stream_of(src) is not None
    _same_file(*_both(src, tmp_path, monkeypatch))


def _nifti_gz(path: Path, a: np.ndarray, *, big_endian=False, slope=1.0) -> Path:
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((0.8, 0.9, 2.5))
    img.SetOrigin((1.0, -2.0, 3.0))
    raw = path.with_suffix("")                          # .nii
    sitk.WriteImage(img, str(raw), False)
    b = bytearray(raw.read_bytes())
    if slope != 1.0:
        b[112:116] = struct.pack("<f", slope)
    if big_endian:
        hdr = bytearray(b[:352])
        fmt = {"i": [0], "h": list(range(40, 56, 2)) + [68, 70, 72, 74],
               "f": list(range(76, 112, 4)) + [112, 116] + list(range(124, 132, 4))
               + list(range(252, 344, 4))}
        for code, offs in fmt.items():
            size = struct.calcsize(code)
            for o in offs:
                v = struct.unpack("<" + code, bytes(hdr[o:o + size]))[0]
                hdr[o:o + size] = struct.pack(">" + code, v)
        data = np.frombuffer(bytes(b[352:]), dtype=a.dtype.newbyteorder("<"))
        b = hdr + data.astype(a.dtype.newbyteorder(">")).tobytes()
    with gzip.open(path, "wb") as f:
        f.write(bytes(b))
    raw.unlink()
    return path


@pytest.mark.parametrize("dtype,big", [("int16", False), ("float32", False), ("int16", True)],
                         ids=["int16", "float32", "int16-big-endian"])
def test_a_streamed_nifti_copy_is_the_whole_copy(tmp_path, monkeypatch, dtype, big):
    a = (np.arange(10 * 6 * 5) % 997).astype(dtype).reshape(10, 6, 5)
    src = _nifti_gz(tmp_path / "v.nii.gz", a, big_endian=big)
    assert np.array_equal(sitk.GetArrayViewFromImage(sitk.ReadImage(str(src))), a)
    assert input_stream.stream_of(src) is not None
    _same_file(*_both(src, tmp_path, monkeypatch))


def test_the_header_listing_is_gdcms_order(tmp_path):
    for i, steps in enumerate(([31.0 + 2 * k for k in range(9)], [31.0 - 2 * k for k in range(9)])):
        d = _shuffle_names(write_series(tmp_path / f"s{i}", steps=steps))
        assert input_stream._series_files(d) == list(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d)))


def test_what_only_the_whole_read_promises_is_left_to_it(tmp_path):
    two = tmp_path / "two"
    write_series(two / "a")
    write_series(two / "b")
    for f in (two / "b").iterdir():                    # a second series, side by side
        f.rename(two / f"b_{f.name}")
    (two / "a").rename(tmp_path / "a_only")
    for f in (tmp_path / "a_only").iterdir():
        f.rename(two / f.name)
    assert input_stream.stream_of(two) is None
    a = np.ones((4, 5, 6), np.float32)       # a float stays float: only the slope says "whole"
    assert input_stream.stream_of(_nifti_gz(tmp_path / "scaled.nii.gz", a, slope=2.0)) is None
    assert sitk.GetArrayViewFromImage(sitk.ReadImage(str(tmp_path / "scaled.nii.gz"))).max() == 2
    four = sitk.Image([6, 5, 4, 2], sitk.sitkInt16)
    sitk.WriteImage(four, str(tmp_path / "four.nii.gz"))
    assert input_stream.stream_of(tmp_path / "four.nii.gz") is None


def test_the_uncompressed_form_is_written_whole(tmp_path, monkeypatch):
    monkeypatch.setenv(ic.COMPRESSION_ENV, "uncompressed")
    called = []
    monkeypatch.setattr(input_stream, "stream_of", lambda c: called.append(c))
    assert ic.transcode(write_series(tmp_path / "s"), tmp_path / "e") is not None
    assert called == []


def test_what_the_reader_refuses_is_refused(tmp_path):
    uneven = write_series(tmp_path / "s", steps=[0, 2, 4, 7, 9])
    with pytest.raises(Exception):
        input_stream.stream_of(uneven)
    assert ic.transcode(uneven, tmp_path / "e") is None


def test_the_check_catches_a_chunk_that_is_not_its_slab(tmp_path, monkeypatch):
    seen = []
    real = ic._check_streamed
    monkeypatch.setattr(ic, "_check_streamed", lambda *a: (seen.append(a), real(*a)))
    assert ic.transcode(write_series(tmp_path / "s", n=10), tmp_path / "e") is not None
    _, stream, digests = seen[0]
    placed = ic.copy_path(tmp_path / "e")
    real(placed, stream, digests)                                          # as written: fine
    with pytest.raises(ValueError, match="chunk 1"):
        real(placed, stream, digests[:1] + ["0" * 64] + digests[2:])


def test_a_source_that_ends_early_keeps_the_original(tmp_path, monkeypatch, capsys):
    src = write_series(tmp_path / "s", n=10)
    real = input_stream.stream_of

    def short(content):
        st = real(content)
        full = st.slabs
        st.slabs = lambda n: (x for i, x in enumerate(full(n)) if i < 1)
        return st
    monkeypatch.setattr(input_stream, "stream_of", short)
    assert ic.transcode(src, tmp_path / "e") is None
    assert "not 10" in capsys.readouterr().err
    assert not list((tmp_path / "e").rglob("*.partial")) and not list((tmp_path / "e").rglob(".stream-*"))


_PEAK = r'''
import os, resource, sys
os.environ["HAVERSACK_INPUT_COPY_COMPRESSION"] = "zstd"
from haversack import input_copy as ic, input_stream
if sys.argv[1] == "whole":
    input_stream.stream_of = lambda content: None
assert ic.transcode(sys.argv[2], sys.argv[3]) is not None
try:        # Linux: ru_maxrss survives exec, so it would report the parent pytest's peak;
            # VmHWM belongs to this process's own address space (kB)
    hwm = [l for l in open("/proc/self/status") if l.startswith("VmHWM:")][0]
    print(int(hwm.split()[1]) * 1024)
except OSError:                                        # macOS: ru_maxrss, in bytes
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
'''


def test_a_streamed_transcode_peaks_at_a_fraction_of_the_whole(tmp_path):
    """Peak memory in a fresh process each, on a 120-slice 512x512 series (60 MB of voxels):
    the whole-volume path holds the volume several times over, the streamed one a slab."""
    src = write_series(tmp_path / "s", n=2)
    first = sorted(src.glob("*.dcm"))
    big = tmp_path / "big"
    big.mkdir()
    rng = np.random.default_rng(1)
    for i in range(120):
        ds = pydicom.dcmread(first[0])
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [-10.0, -12.0, 31.0 + 2.0 * i]
        ds.Rows = ds.Columns = 512
        ds.PixelData = rng.integers(-100, 1200, (512, 512), dtype=np.int16).tobytes()
        ds.save_as(big / f"IM{i:04d}.dcm")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(ic.__file__).parents[1])] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    peaks = {}
    for mode in ("whole", "stream"):
        r = subprocess.run([sys.executable, "-c", _PEAK, mode, str(big), str(tmp_path / mode)],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        peaks[mode] = int(r.stdout.strip().splitlines()[-1])
    whole, stream = peaks["whole"], peaks["stream"]                        # bytes, both
    assert stream < 0.6 * whole, (stream / 1e6, whole / 1e6)


# -- the review of 2026-09-25: a DICOM file the listing cannot place hands over to the whole read

def _last(d: Path) -> Path:
    return sorted(d.glob("*.dcm"))[-1]


def _no_preamble(d: Path) -> Path:
    """The last slice rewritten as a bare implicit-VR dataset, no preamble or file meta: pydicom
    reads it only by force, GDCM reads it as a slice."""
    f = _last(d)
    ds = pydicom.dcmread(f)
    del ds.file_meta
    ds.preamble = None
    pydicom.dcmwrite(f, ds, implicit_vr=True, little_endian=True, enforce_file_format=False)
    with pytest.raises(Exception):
        pydicom.dcmread(f)
    return d


def _drop(tag):
    def make(d: Path) -> Path:
        f = _last(d)
        ds = pydicom.dcmread(f)
        delattr(ds, tag)
        ds.save_as(f, enforce_file_format=True)
        return d
    return make


def _with_secondary_capture(d: Path) -> Path:
    """A secondary capture of another series beside the CT: pixel data, no position."""
    from pydicom.uid import generate_uid
    ds = pydicom.dcmread(_last(d))
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    for tag in ("ImagePositionPatient", "ImageOrientationPatient"):
        delattr(ds, tag)
    ds.save_as(d / "SC0001.dcm", enforce_file_format=True)
    return d


@pytest.mark.parametrize("make", [_no_preamble, _drop("ImagePositionPatient"),
                                  _drop("ImageOrientationPatient"), _with_secondary_capture],
                         ids=["no-preamble-end-slice", "end-slice-without-position",
                              "end-slice-without-orientation", "secondary-capture-beside"])
def test_a_folder_the_listing_cannot_place_takes_the_whole_read(tmp_path, make):
    """Each of these was streamed: the first read with one slice fewer than the reader keeps,
    the others copied although the reader refuses them - and the copy replaced the original."""
    from haversack import io as nio
    src = make(write_series(tmp_path / "s", n=10))
    assert input_stream.stream_of(src) is None
    try:
        expected = nio.read_image(src).GetSize()
    except Exception:                                   # noqa: BLE001 - the reader refuses it
        expected = None
    got = ic.transcode(src, tmp_path / "e")
    if expected is None:
        assert got is None, "a copy of a folder the reader refuses"
    else:
        assert got is not None and nio.read_image(got).GetSize() == expected


def test_a_nifti_whose_data_offset_is_below_the_header_is_read_whole(tmp_path):
    """niftilib reads a single file from its 348-byte header's end when vox_offset says less;
    seeking to vox_offset read header bytes as voxels (0: a copy shifted by 174 values)."""
    from haversack import io as nio
    a = (np.arange(10 * 6 * 5) % 997).astype("int16").reshape(10, 6, 5) + 3
    src = _nifti_gz(tmp_path / "v.nii.gz", a)
    raw = bytearray(gzip.decompress(src.read_bytes()))
    raw[108:112] = struct.pack("<f", 0.0)
    src.write_bytes(gzip.compress(bytes(raw)))
    ref = sitk.GetArrayFromImage(nio.read_image(src))
    assert input_stream.stream_of(src) is None
    got = ic.transcode(src, tmp_path / "e")
    assert got is not None
    np.testing.assert_array_equal(sitk.GetArrayFromImage(nio.read_image(got)), ref)


def test_a_duplicated_end_slice_is_refused_not_read_with_nan_geometry(tmp_path):
    """GDCM sorts a series whose first and last positions tie by file name; the zero span made
    every geometry check compare against NaN and pass, and the volume read with NaN spacing."""
    from haversack import io as nio
    from haversack.errors import InputError
    from pydicom.uid import generate_uid
    d = write_series(tmp_path / "s", n=10)
    ds = pydicom.dcmread(_last(d))
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    # named to sort FIRST, so the by-name order puts the two tied positions first and last
    ds.save_as(d / "AA0000.dcm", enforce_file_format=True)
    with pytest.raises(InputError, match="duplicate"):
        nio.read_image(d)
