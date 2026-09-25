"""The input copy: a cached input decoded once, kept INSTEAD of the original (docs/input-copy.md).

A job reads its input every time it runs, and for a DICOM series that is a full decode: 13 s
for a 709-slice CT on a Modal worker, 45 s cold from a volume another container wrote. The copy
is the image ``io.read_image`` produced, written once when the cache stores the input - one
uncompressed zarr chunk in a zip, with duckn's geometry and the DICOM tags SimpleITK reported -
and read back by mapping that chunk (0.25 s for the same CT), with voxels and geometry identical.

An operator may store it compressed instead (``HAVERSACK_INPUT_COPY_COMPRESSION=zstd``): zstd in
chunks of :data:`CHUNK_SLICES` slices, decoded in parallel by zarr - 2.6x smaller, read in ~0.6-1 s
for that CT (measured 2026-09-25, docs/input-copy.md §13). It is the choice for a cache whose room
is the constraint: a Modal worker's series cache is RAM.

An entry holds one form: the copy, or - when the reader refuses the input, or anything about
the copy fails - the original, which is then read (and refused) as it always was. The original
exists only inside :func:`transcode`, as its input; nothing downstream is handed it.

The file (``<entry>/decoded/input.duckn.zip``):

    zarr.json   the array's metadata: shape, dtype, ONE chunk equal to the shape, the ``bytes``
                codec alone; ``attributes.duckn`` - LPS space, origin, z/y/x axes with
                direction x spacing, slice thickness, sample units, the series-level DICOM tags
                (``extensions.dicom``), per-slice tags (``axes[0].samples[i].metadata.dicom``),
                and what this file is (``extensions.haversack``)
    c/0/0/0     the voxels, C order, little-endian, stored (not deflated)

or, compressed (format version 2): chunks of ``CHUNK_SLICES`` whole slices, codecs ``bytes`` then
``zstd``, one zip member per chunk (``c/<k>/0/0``), the zip itself still stored.

Imports nothing heavy at module level: ``io`` asks :func:`is_copy` on every read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: Where the copy lives in a cache entry, and its name - looked up BY NAME, never by scanning.
COPY_DIR = "decoded"
COPY_NAME = "input.duckn.zip"
#: What ``extensions.haversack.kind`` says, and this file's format version: bump when the FILE's
#: meaning changes (a field moves, the layout changes).
KIND = "input_copy"
FORMAT_VERSION = 1
#: The file's format version by its compression: a compressed copy is a different layout, and a
#: reader that knows only version 1 must see it as stale (refetch), never try to map it.
FORMATS = {"none": 1, "zstd": 2}
#: The reader's version, the input side's ``CACHE_EPOCH``: bump whenever ``io.read_image`` would
#: produce different voxels, geometry or tags from the same original bytes. A copy written under
#: another version is STALE - its original is gone, so it cannot be redone: a fetched input is
#: fetched again, an upload counts as evicted (410 input_gone).
READER_VERSION = 1
#: The operator's switch: ``HAVERSACK_INPUT_COPY=0`` keeps originals, as before this existed.
ENV = "HAVERSACK_INPUT_COPY"
#: How new copies are stored: ``none`` (the default: one mapped chunk, the fastest read) or
#: ``zstd``. A cache may hold both; the reader reads each by its own layout, so changing this
#: rewrites nothing and invalidates nothing.
COMPRESSION_ENV = "HAVERSACK_INPUT_COPY_COMPRESSION"
#: zstd's level and the slices per chunk (measured on a 709-slice CT: level 3 in 32-slice chunks
#: was the fastest read at 2.6x; a single compressed chunk decodes on one thread, 2.4 s).
ZSTD_LEVEL = 3
CHUNK_SLICES = 32
#: How far the copy's geometry may differ from the reader's: duckn stores each axis as direction
#: x spacing and a reader takes it apart again, which is exact for an axis-aligned grid and off by
#: one unit in the last place for a tilted series' sheared direction (1.1e-16 measured - a sample
#: moved by ~1e-13 mm). Voxels and pixel type must be identical; this bound is for geometry only.
GEOMETRY_TOLERANCE = 1e-12
#: The DICOM extension's version, as duckn's dicom-spec numbers it.
DICOM_EXTENSION_VERSION = "1.0"


class NotACopy(Exception):
    """The file is not in the one layout the mapped reader reads; read it another way."""


def enabled() -> bool:
    """Whether inputs are transcoded here: the operator has not said no, and the packages a copy
    is WRITTEN with are installed (the duckn extra). An environment without them keeps
    originals, as before this existed; reading a copy needs neither (:func:`read_copy`)."""
    if os.environ.get(ENV, "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("duckn", "zarr", "pydicom"))
    except (ImportError, ValueError):
        return False


def compression() -> str:
    """How a new copy is stored, from :data:`COMPRESSION_ENV`; a value it does not know raises,
    naming the ones it does (a copy is then not written, and the warning says why)."""
    value = os.environ.get(COMPRESSION_ENV, "none").strip().lower() or "none"
    if value not in FORMATS:
        raise ValueError(f"{COMPRESSION_ENV}={value!r}: expected one of {', '.join(FORMATS)}")
    return value


def copy_path(entry) -> Path:
    return Path(entry) / COPY_DIR / COPY_NAME


def is_copy(path) -> bool:
    """Whether ``path`` names an input copy - by its name, which only this module writes."""
    p = Path(path)
    return p.name == COPY_NAME and p.parent.name == COPY_DIR and p.is_file()


# -- which inputs ----------------------------------------------------------------------

def _nrrd_header(path: Path) -> dict:
    head = _head(path)
    end = head.find(b"\n\n")
    text = head[: end if end >= 0 else len(head)].decode("latin-1", "replace")
    out = {}
    for line in text.splitlines()[1:]:
        if ":=" in line:
            k, v = line.split(":=", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        out[k.strip().lower()] = v.strip()
    return out


def _head(path: Path, n: int = 65536) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def wanted(content) -> bool:
    """Whether a cached input is transcoded: an image whose read DECODES - a DICOM series, or a
    compressed single file - and never a label map, whose names and codes live in its file (a
    ``.seg.nrrd``, any NRRD carrying segment fields). An input already raw and single (raw NRRD,
    uncompressed MHA/NIfTI, a duckn store) is its own efficient form and is kept as it is."""
    if not enabled():
        return False
    p = Path(content)
    if p.is_dir():
        from .duckn_io import is_duckn_store
        if is_duckn_store(p):
            return False
        files = [q for q in sorted(p.iterdir()) if q.is_file() and not q.name.startswith(".")]
        if len(files) == 1:
            return wanted(files[0])
        return len(files) > 1              # a series: the reader decides whether it is one
    name = p.name.lower()
    if name.endswith((".seg.nrrd", ".zarr.zip", ".duckn.zip", ".zarr", ".duckn")):
        return False
    if name.endswith((".nii.gz", ".gz")):
        return True
    if name.endswith((".nrrd", ".nhdr")):
        h = _nrrd_header(p)
        if any(k.startswith("segment") for k in h):
            return False                   # a label map with its names
        return h.get("encoding", "raw").lower() not in ("raw",)
    if name.endswith((".mha", ".mhd")):
        head = _head(p, 4096).decode("latin-1", "replace").lower()
        return "compresseddata = true" in head
    if name.endswith(".dcm") or _is_dicom(p):
        return True                        # one DICOM file: still a decode
    return False


def _is_dicom(p: Path) -> bool:
    head = _head(p, 132)
    return len(head) == 132 and head[128:132] == b"DICM"


# -- writing -----------------------------------------------------------------------------

def _thickness(per_slice):
    """(axis thickness, per-sample thicknesses or None) from Slice Thickness (spec §2)."""
    from .dicom_tags import SLICE_THICKNESS
    values = [d.get(SLICE_THICKNESS) for d in per_slice]
    try:
        nums = [float(v) for v in values if v not in (None, "")]
    except ValueError:
        return None, None
    if not nums or len(nums) != len(values):
        return None, None
    if all(n == nums[0] for n in nums):
        return nums[0], None
    return None, nums


def _sample_units(per_slice):
    from .dicom_tags import RESCALE_TYPE
    values = {d.get(RESCALE_TYPE, "").strip() for d in per_slice}
    return values.pop() if len(values) == 1 and "" not in values else None


def _source_size(content) -> tuple[int, int]:
    """(files, bytes) of the original as stored - what its digest names, recorded because the
    original will not be kept (``GET /v1/inputs/{digest}`` reports them)."""
    p = Path(content)
    files = ([q for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")]
             if p.is_dir() else [p])
    return len(files), sum(q.stat().st_size for q in files)


def _metadata(image, per_slice, *, source, source_digest, source_size=(None, None), how="none"):
    """The duckn metadata of the copy, through duckn's own models: the geometry is
    ``from_sitk``'s (LPS - no flip either way), the rest is filled into its fields."""
    import haversack
    import SimpleITK as sitk
    from duckn.models import SampleMetadata
    from duckn.sitk_adapter import from_sitk

    from .dicom_tags import tags_from_sitk
    vol = from_sitk(image)
    meta = vol.metadata
    series, slices = tags_from_sitk(per_slice) if per_slice else ({}, [])
    thick, thick_each = _thickness(per_slice) if per_slice else (None, None)
    z = meta.axes[0]
    if thick is not None:
        z.thickness = thick
    n = int(vol.raw.shape[0])
    if (any(slices) or thick_each) and len(per_slice) == n:
        z.samples = [SampleMetadata(thickness=(thick_each[i] if thick_each else None),
                                    metadata=({"dicom": slices[i]} if slices and slices[i] else None))
                     for i in range(n)]
    units = _sample_units(per_slice) if per_slice else None
    if units:
        meta.sample_units = units
    ext = dict(meta.extensions or {})
    if series or slices:
        ext["dicom"] = {"version": DICOM_EXTENSION_VERSION, "tags": series}
    ext["haversack"] = {"kind": KIND, "version": FORMATS[how], "reader_version": READER_VERSION,
                        "source": source, "source_digest": source_digest,
                        "source_files": source_size[0], "source_bytes": source_size[1],
                        "reader": {"haversack": haversack.__version__,
                                   "SimpleITK": sitk.Version.VersionString()}}
    meta.extensions = ext
    return vol


def _write(vol, out: Path, how: str = "none") -> None:
    import zarr
    from duckn.models import duckn_attrs
    from zarr.storage import ZipStore
    shape = tuple(int(n) for n in vol.raw.shape)
    if how == "zstd":
        from zarr.codecs import ZstdCodec
        chunks, compressors = (min(CHUNK_SLICES, shape[0]),) + shape[1:], [ZstdCodec(level=ZSTD_LEVEL)]
    else:
        chunks, compressors = shape, None
    store = ZipStore(str(out), mode="w")
    try:
        arr = zarr.create_array(store, shape=shape, dtype=vol.raw.dtype, chunks=chunks,
                                compressors=compressors, attributes=duckn_attrs(vol.metadata),
                                fill_value=0, config={"write_empty_chunks": True})
        arr[:] = vol.raw
    finally:
        store.close()


def transcode(content, entry, *, source=None, source_digest=None) -> Path | None:
    """Write ``entry``'s copy of ``content`` and return its path - or None, when the input is
    not one to transcode (:func:`wanted`), the reader refuses it, or the copy does not read back
    as exactly what the reader produced; the caller then keeps the original. Verified before it
    is placed: voxels, geometry and the series tags, read back through :func:`read_copy`."""
    import numpy as np
    import SimpleITK as sitk

    from . import io as nio
    from .errors import InputError
    if not wanted(content):
        return None
    try:
        image, per_slice = nio.read_image_and_tags(content)
    except InputError:
        return None                          # refused: the original stays, and fails at read
    final = copy_path(entry)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name("." + COPY_NAME + ".partial")
    try:
        how = compression()
        vol = _metadata(image, per_slice, source=source, source_digest=source_digest,
                        source_size=_source_size(content), how=how)
        _write(vol, partial, how)
        back = read_copy(partial, check_name=False)
        same = (np.array_equal(sitk.GetArrayViewFromImage(back), sitk.GetArrayViewFromImage(image))
                and back.GetPixelID() == image.GetPixelID()
                and all(np.allclose(getattr(back, g)(), getattr(image, g)(), rtol=0,
                                    atol=GEOMETRY_TOLERANCE)
                        for g in ("GetOrigin", "GetSpacing", "GetDirection")))
        if not same:
            raise ValueError("the copy did not read back as the image it was written from")
        os.replace(partial, final)
        return final
    except Exception as e:                 # noqa: BLE001 - any failure keeps the original
        import sys
        print(f"warning: no input copy for {source or content}: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        partial.unlink(missing_ok=True)
        return None


# -- reading -----------------------------------------------------------------------------

def _layout(path: Path):
    """``(zarr.json dict, how, chunk data offset, dtype, shape)`` for one of the two layouts this
    module writes - ``how`` "none" (one stored chunk, mapped at the offset) or "zstd" (slabs of
    whole slices, decoded by zarr; offset None) - or NotACopy for any other."""
    import math
    import struct
    import zipfile

    import numpy as np
    try:
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("zarr.json"))
            members = {i.filename: i for i in z.infolist()}
    except (zipfile.BadZipFile, ValueError, KeyError) as e:
        raise NotACopy(f"{path}: {e}") from None
    shape = tuple(int(n) for n in meta.get("shape") or ())
    codecs = meta.get("codecs") or []
    grid = tuple(((meta.get("chunk_grid") or {}).get("configuration") or {}).get("chunk_shape") or ())
    if meta.get("node_type") != "array" or len(shape) != 3 or not codecs \
            or codecs[0].get("name") != "bytes":
        raise NotACopy(f"{path}: not an input copy's array")
    endian = (codecs[0].get("configuration") or {}).get("endian", "little")
    dt = np.dtype(meta["data_type"]).newbyteorder("<" if endian == "little" else ">")
    stored = all(i.compress_type == zipfile.ZIP_STORED for i in members.values())
    names = {n for n in members if n.startswith("c/")}
    if len(codecs) == 2 and codecs[1].get("name") == "zstd":
        k = grid[0] if len(grid) == 3 else 0
        want = {f"c/{i}/0/0" for i in range(math.ceil(shape[0] / k))} if k else None
        if not stored or not k or grid[1:] != shape[1:] or names != want:
            raise NotACopy(f"{path}: not whole-slice zstd chunks, one member each")
        return meta, "zstd", None, dt, shape
    info = members.get("c/0/0/0")
    if len(codecs) != 1 or grid != shape or info is None or names != {"c/0/0/0"} or not stored:
        raise NotACopy(f"{path}: not one stored, uncompressed chunk")
    if info.file_size != int(np.prod(shape)) * dt.itemsize:
        raise NotACopy(f"{path}: the chunk holds {info.file_size} bytes, not the array's")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
    if local[:4] != b"PK\x03\x04":
        raise NotACopy(f"{path}: no local header where the index points")
    name_len, extra_len = struct.unpack("<HH", local[26:30])
    return meta, "none", info.header_offset + 30 + name_len + extra_len, dt, shape


def stored_compression(path) -> str:
    """How a copy is stored - "none" or "zstd" - read from its layout, not from what it says."""
    return _layout(Path(path))[1]


def _duckn(meta: dict):
    attrs = (meta.get("attributes") or {}).get("duckn")
    if not isinstance(attrs, dict):
        raise NotACopy("no duckn metadata")
    return attrs


def info(path) -> dict:
    """``extensions.haversack`` of a copy ({} when it has none)."""
    meta, *_ = _layout(Path(path))
    return (_duckn(meta).get("extensions") or {}).get("haversack") or {}


def stale(path) -> bool:
    """Whether a copy was written by another reader version, or another format: its original
    is gone, so it cannot be redone - the cache treats the entry as absent."""
    try:
        h = info(path)
        how = stored_compression(path)
    except NotACopy:
        return True
    return h.get("kind") != KIND or h.get("version") != FORMATS[how] \
        or h.get("reader_version") != READER_VERSION


def read_copy(path, *, check_name: bool = True):
    """The copy as a SimpleITK image: the chunk mapped at its offset (no copy, no CRC until
    SimpleITK takes the buffer) - or, compressed, every chunk decoded by zarr - the geometry
    through duckn's ``to_sitk``, the series-level DICOM tags restored as ``gggg|eeee`` strings.
    NotACopy for any file this does not fully understand - never a guess."""
    import mmap

    import numpy as np
    p = Path(path)
    if check_name and not is_copy(p):
        raise NotACopy(f"{p} is not named as an input copy")
    meta, how, start, dt, shape = _layout(p)
    attrs = _duckn(meta)
    if attrs.get("value_transforms"):
        raise NotACopy(f"{p}: a stored rescale - not a calibrated copy")
    axes = attrs.get("axes") or []
    if (attrs.get("space") is None or attrs.get("space_origin") is None or len(axes) != 3
            or any(a.get("space_direction") is None for a in axes)):
        raise NotACopy(f"{p}: the geometry is not stated in full")
    if how == "zstd":
        image = _to_sitk(attrs, _decoded(p, dt, shape))
        return _with_tags(image, attrs)
    n = int(np.prod(shape))
    with open(p, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            raw = np.frombuffer(mm, dtype=dt, count=n, offset=start).reshape(shape)
            image = _to_sitk(attrs, raw)                 # GetImageFromArray: the one copy
            del raw
        finally:
            mm.close()
    return _with_tags(image, attrs)


def _decoded(p: Path, dt, shape):
    """A compressed copy's voxels, through zarr (its codec pipeline decodes the chunks in
    parallel). An environment without zarr cannot read one: NotACopy, and ``io`` then tries
    duckn's reader, which says what to install."""
    try:
        import zarr
        from zarr.storage import ZipStore
    except ImportError as e:
        raise NotACopy(f"{p}: a compressed copy needs zarr to read ({e})") from None
    store = ZipStore(str(p), mode="r")
    try:
        raw = zarr.open_array(store, mode="r")[...]
    except Exception as e:                 # noqa: BLE001 - a chunk that does not decode
        raise NotACopy(f"{p}: {type(e).__name__}: {e}") from None
    finally:
        store.close()
    if raw.shape != shape or raw.dtype.newbyteorder("=") != dt.newbyteorder("="):
        raise NotACopy(f"{p}: decoded {raw.shape} {raw.dtype}, not the stated array")
    return raw


def _with_tags(image, attrs: dict):
    tags = ((attrs.get("extensions") or {}).get("dicom") or {}).get("tags") or {}
    if tags:
        from .dicom_tags import to_sitk_strings
        try:
            restored = to_sitk_strings(tags)
        except ImportError:
            # an environment without pydicom's data dictionary (an engine image that cannot take
            # the duckn extra): the tags are provenance, never needed to compute - the image
            # is read without them rather than not at all
            restored = {}
        for key, value in restored.items():
            image.SetMetaData(key, value)
    return image


def _to_sitk(attrs: dict, raw):
    """The image, through duckn's own ``to_sitk`` where duckn is installed. Where it is not - an
    engine environment that cannot take the duckn extra (SynthStrip's numpy<2) reading a copy
    another container wrote - the one layout this module writes is converted directly: LPS
    space (SimpleITK's own: no flip), axes z, y, x, each ``space_direction`` the direction cosine
    times the spacing. A test holds the two equal on the same file; anything else is refused."""
    try:
        from duckn.models import DucknMetadata
        from duckn.sitk_adapter import to_sitk
        from duckn.volume import Volume
    except ImportError:
        pass
    else:
        return to_sitk(Volume(raw=raw, metadata=DucknMetadata(**attrs)))
    import numpy as np
    import SimpleITK as sitk
    if attrs.get("space") not in ("left-posterior-superior", "LPS"):
        raise NotACopy("without duckn only an LPS copy can be read")
    vecs = [np.asarray(a["space_direction"], dtype=np.float64) for a in attrs["axes"]][::-1]
    spacing = [float(np.linalg.norm(v)) for v in vecs]      # x, y, z
    cols = np.stack([v / s for v, s in zip(vecs, spacing)], axis=1)
    image = sitk.GetImageFromArray(raw)
    image.SetSpacing(spacing)
    image.SetOrigin([float(v) for v in attrs["space_origin"]])
    image.SetDirection([float(v) for v in cols.ravel()])
    return image


def slice_tags(path, keyword: str) -> list:
    """One per-slice DICOM tag, as a list in the volume's z order (None where a slice lacks it)."""
    meta, *_ = _layout(Path(path))
    z = (_duckn(meta).get("axes") or [{}])[0]
    return [((s.get("metadata") or {}).get("dicom") or {}).get(keyword)
            for s in (z.get("samples") or [])]
