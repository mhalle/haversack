"""An input read a slab of slices at a time, for the input copy (2026-09-25).

:func:`haversack.input_copy.transcode` used to read the whole volume, write it, and read the whole
copy back to compare - several copies of the volume at once, ~3 GB at peak for a 709-slice CT, in
an api container that has 2 GB. The compressed copy is stored in whole-slice chunks anyway, so a
source that can hand over a slab at a time is transcoded chunk by chunk with memory bounded by a
slab: a DICOM series (one file per slice), and a gzipped NIfTI (one sequential decompression).

Everything here must produce exactly what :func:`haversack.io.read_image_and_tags` produces; a
source this cannot promise that for returns None, and the whole-volume path runs as before:

- **geometry** is the reader's own, from headers alone - a series' from ``io._series_geometry``
  (IPP/IOP, which also refuses what the reader refuses), a file's from SimpleITK's
  ``ReadImageInformation``;
- **pixel type**: a series read whole takes the first file's type, so every slab is read into it
  (``SetOutputPixelType``), never into a type its own first file would pick;
- **tags**: each slab's series reader reports its slices' dictionaries, the same dictionaries a
  whole read reports; a single file's are its header's.

Measured on four DICOM series and three gzipped NIfTIs (2026-09-25): voxels, pixel type and
per-slice tags identical to the whole read, at the same speed.
"""
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class Stream:
    """A source read a slab at a time: its geometry (SimpleITK's x, y, z order), pixel type, the
    array shape (z, y, x), and ``slabs(n)`` yielding ``(array (k, y, x), per-slice tag dicts)``
    in z order, ``k <= n``. ``tags`` is a single file's header dictionary, for a source whose
    slabs carry none."""
    origin: tuple
    spacing: tuple
    direction: tuple
    pixel_id: int
    dtype: np.dtype
    shape: tuple
    slabs: Callable[[int], Iterator[tuple]]
    tags: list


def stream_of(content) -> Stream | None:
    """The slab source for ``content``, or None when only the whole-volume read can promise the
    same image: not one DICOM series, a scaled or 4-D NIfTI, a format with no slab reader here.
    Raises :class:`haversack.errors.InputError` when the reader would refuse the input (the
    series' geometry), as the whole read would."""
    p = Path(content)
    if p.is_dir():
        return _dicom_series(p)
    if p.name.lower().endswith(".nii.gz"):
        return _nifti_gz(p)
    return None


def _series_files(directory: Path) -> list | None:
    """The one series' image files in ``directory``, in the order GDCM's
    ``GetGDCMSeriesFileNames`` gives them - ascending position along the slice normal - read
    from headers alone; None when the directory is not exactly one series of image slices.

    Not GDCM's own listing, which is what the whole read uses: it reads the files to list them,
    and on a 709-slice CT peaked at 920 MB and kept ~500 MB resident afterwards (2026-09-25) -
    the memory the slab path exists to save. pydicom stops before the pixel data (~50 MB
    peak). The order is unique here because :func:`haversack.io._series_geometry` refuses
    duplicate and non-monotonic positions; that it is GDCM's was checked on four real series
    and is held by a test against GDCM on descending and shuffled ones."""
    import numpy as np
    import pydicom
    slices, series = [], set()
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:                      # noqa: BLE001 - not DICOM: GDCM skips it too
            continue
        if not all(k in ds for k in ("Rows", "Columns", "ImagePositionPatient",
                                     "ImageOrientationPatient")):
            continue                           # not an image slice (RTSTRUCT, SR, ...)
        series.add(str(ds.get("SeriesInstanceUID", "")))
        slices.append((f, [float(v) for v in ds.ImagePositionPatient],
                       [float(v) for v in ds.ImageOrientationPatient]))
    if len(series) != 1 or len(slices) < 2:
        return None                            # none, or several: the whole read decides
    iop = np.asarray(slices[0][2])
    normal = np.cross(iop[:3], iop[3:])
    return [str(f) for f, ipp, _ in sorted(slices, key=lambda s: float(np.dot(s[1], normal)))]


def _dicom_series(directory: Path) -> Stream | None:
    import SimpleITK as sitk

    from .io import _series_geometry
    files = _series_files(directory)
    if files is None:
        return None
    first = sitk.ImageFileReader()
    first.SetFileName(files[0])
    first.ReadImageInformation()
    size = first.GetSize()
    if first.GetNumberOfComponents() != 1 or len(size) < 2 or (len(size) == 3 and size[2] != 1):
        return None                # colour, or a multi-frame file: not a slice per file
    origin, direction, spacing = _series_geometry(files)     # refuses what the reader refuses
    pixel_id = first.GetPixelID()
    dtype = sitk.GetArrayViewFromImage(sitk.Image([1, 1, 1], pixel_id)).dtype

    def slabs(n: int):
        for k in range(0, len(files), n):
            r = sitk.ImageSeriesReader()
            r.SetFileNames(files[k:k + n])
            r.SetOutputPixelType(pixel_id)
            r.MetaDataDictionaryArrayUpdateOn()
            image = r.Execute()
            a = sitk.GetArrayFromImage(image)
            if a.ndim == 2:
                a = a[None]
            if a.shape[1:] != (size[1], size[0]):
                raise ValueError(f"slices {k}-{k + len(a) - 1} are not {size[0]}x{size[1]}")
            tags = [{key: r.GetMetaData(i, key) for key in r.GetMetaDataKeys(i)}
                    for i in range(len(a))]
            yield a, tags
    return Stream(origin=origin, spacing=spacing, direction=direction, pixel_id=pixel_id,
                  dtype=dtype, shape=(len(files), size[1], size[0]), slabs=slabs, tags=[])


#: NIfTI-1 datatype codes a slab is read as, all single-component
_NIFTI_DTYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4",
                 1024: "i8", 1280: "u8"}


def _nifti_gz(path: Path) -> Stream | None:
    import SimpleITK as sitk
    try:
        info = sitk.ImageFileReader()
        info.SetFileName(str(path))
        info.ReadImageInformation()
    except RuntimeError:
        return None                # e.g. a non-orthonormal affine: the whole read snaps it
    with gzip.open(path, "rb") as f:
        h = f.read(348)
    if len(h) < 348:
        return None
    endian = "<" if struct.unpack("<i", h[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", h[:4])[0] != 348:
        return None                # NIfTI-2, or not a NIfTI-1 header
    dim = struct.unpack(endian + "8h", h[40:56])
    datatype = struct.unpack(endian + "h", h[70:72])[0]
    vox_offset = struct.unpack(endian + "f", h[108:112])[0]
    slope, inter = struct.unpack(endian + "ff", h[112:120])
    if datatype not in _NIFTI_DTYPES:
        return None                # colour, complex (a 4-D file fails the size check below)
    if not (slope in (0.0, 1.0) and inter == 0.0):
        return None                # scaled: SimpleITK's own arithmetic, whole read only
    nx, ny, nz = (int(d) for d in dim[1:4])
    if tuple(info.GetSize()) != (nx, ny, nz) or info.GetNumberOfComponents() != 1:
        return None
    file_dtype = np.dtype(_NIFTI_DTYPES[datatype]).newbyteorder(endian)
    pixel_id = info.GetPixelID()
    dtype = sitk.GetArrayViewFromImage(sitk.Image([1, 1, 1], pixel_id)).dtype
    if dtype != file_dtype.newbyteorder("="):
        return None                # the reader converts: whole read only
    tags = [{k: info.GetMetaData(k) for k in info.GetMetaDataKeys()}]

    def slabs(n: int):
        per = nx * ny * file_dtype.itemsize
        with gzip.open(path, "rb") as f:
            f.read(int(vox_offset))
            for k in range(0, nz, n):
                m = min(n, nz - k)
                buf = f.read(per * m)
                if len(buf) != per * m:
                    raise ValueError(f"{path}: the data ends at slice {k}")
                yield np.frombuffer(buf, dtype=file_dtype).reshape(m, ny, nx).astype(dtype), []
    return Stream(origin=tuple(info.GetOrigin()), spacing=tuple(info.GetSpacing()),
                  direction=tuple(info.GetDirection()), pixel_id=pixel_id, dtype=dtype,
                  shape=(nz, ny, nx), slabs=slabs, tags=tags)
