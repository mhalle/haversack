"""The input copy: a cached input decoded once, kept INSTEAD of the original (docs/input-copy.md).

A job reads its input every time it runs, and for a DICOM series that is a full decode: 13 s
for a 709-slice CT on a Modal worker, 45 s cold from a volume another container wrote. The copy
is the image ``io.read_image`` produced, written once when the cache stores the input - zstd
through blosc with bit shuffling, in chunks of :data:`CHUNK_SLICES` whole slices in a zip, with
duckn's geometry and the DICOM tags SimpleITK reported - and read back through zarr, which decodes
the chunks in parallel, with voxels and geometry identical. Compressed by default (2026-09-25):
3.1x smaller for that CT and 3.4-5.8x for six other datasets, at a read ~2-3x the uncompressed
form's (0.3-0.9 s for that CT; docs/input-copy.md §13) - against the 13 s DICOM decode it
replaces either way, and cache room is what a Modal worker's RAM series cache runs out of.

``HAVERSACK_INPUT_COPY_COMPRESSION=uncompressed`` stores one uncompressed chunk instead, read by mapping
it (0.15-0.25 s for that CT) - the fastest read, and one that needs neither duckn nor zarr.

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
``blosc`` (cname zstd, bitshuffle), one zip member per chunk (``c/<k>/0/0``), the chunk members
still stored.

In both, ``zarr.json`` is DEFLATED and every chunk member STORED (2026-09-26). The chunks must be
stored - the mapped reader takes a member's bytes at its offset, and duckn's zip guide asks it of
anything referenced by byte range - but the header is only ever read whole, and with the files'
own DICOM tags it is large: a 92-slice Siemens MR carries 1.8 MB of them (Siemens' private
per-slice parameter blocks, as base64), 43 % of its compressed copy, and 81 KB deflated. A copy
written before this has a stored header and reads the same; code from before it calls a
deflated header another layout (stale), as it does any layout it does not know.

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
FORMATS = {"uncompressed": 1, "zstd": 2}
#: The reader's version, the input side's ``CACHE_EPOCH``: bump whenever ``io.read_image`` would
#: produce different voxels, geometry or tags from the same original bytes. A copy written under
#: another version is STALE - its original is gone, so it cannot be redone: a fetched input is
#: fetched again, an upload counts as evicted (410 input_gone). 2 (2026-09-25): the tags come from
#: ``duckn.dicom_tags``, which leaves binary-VR values out where haversack's own converter kept
#: SimpleITK's strings of them - bumped before any deployment held a copy, so it cost nothing.
READER_VERSION = 3
#: 3 (2026-09-26): copies made under 2 carry, from SimpleITK's dictionaries, tags stated in
#: STORED-value units beside rescaled voxels - a CT's Pixel Padding Value -2000 where the copy's
#: padding is -3024 - and hand them to every header exported from them. The user's rule: nothing
#: haversack stores may let its bytes be misread. So they are stale: a fetched input is fetched
#: again, an upload is gone. (Tags that are merely THINNER are not a reader change - see
#: TAGS_VERSION; tags that are WRONG are.)
#: What the copy's DICOM tags were made by - NOT part of :data:`READER_VERSION`, though the
#: tags once were (2026-09-26): they are provenance, never what an engine reads, so a copy with
#: an older kind of tags is still exact and is kept; ``Input.tags()`` reports which kind it
#: holds. 1 (absent): SimpleITK's dictionaries. 2: the files' own headers through pydicom
#: (``duckn.dicom_tags.tags_from_files``) - sequences and binary values, the dicom-spec rules
#: settled that day (nothing in stored-value units beside rescaled voxels; ``stored_values``
#: says which the copy holds), and NO private elements: haversack cannot vouch for a vendor's
#: private data against the copy's voxels, and the source keeps the originals for anyone who
#: wants them (the user's call, 2026-09-26).
TAGS_VERSION = 2
#: The operator's switch: ``HAVERSACK_INPUT_COPY=0`` keeps originals, as before this existed.
ENV = "HAVERSACK_INPUT_COPY"
#: How new copies are stored: ``zstd`` (the default since 2026-09-25: blosc-zstd, the smallest
#: at the same read time of everything measured) or ``uncompressed`` (one mapped chunk: the fastest read,
#: and one that needs neither duckn nor zarr). A cache may hold both; the reader reads each by its
#: own layout, so changing this rewrites nothing and invalidates nothing.
COMPRESSION_ENV = "HAVERSACK_INPUT_COPY_COMPRESSION"
DEFAULT_COMPRESSION = "zstd"
#: The compressed form's codec and chunking (2026-09-25, docs/input-copy.md §13, seven datasets):
#: blosc's zstd with bit shuffling was 1.2-1.4x smaller than plain zstd on every one, at the same
#: read and write time; level 3, as levels 1 and 6 moved size by 5-8 % and write time by 2x; 32
#: slices a chunk, as 16-64 read alike and one whole-volume chunk decodes on one thread (2.4 s).
ZSTD_LEVEL = 3
BLOSC_SHUFFLE = "bitshuffle"
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
    is WRITTEN with are installed (core since 2026-09-25; absent only in a lean install). An environment without them keeps
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
    value = os.environ.get(COMPRESSION_ENV, DEFAULT_COMPRESSION).strip().lower() or DEFAULT_COMPRESSION
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
    from duckn.dicom_tags import SLICE_THICKNESS
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


#: What describes a grayscale or palette pixel beside a vector image's pixels: Samples per
#: Pixel, Photometric Interpretation, Planar Configuration, and the palette's descriptors,
#: data and segmented data (2026-09-26: an export of a PALETTE COLOR file wrote RGB pixels under
#: "Samples per Pixel 1, PALETTE COLOR" and the palette itself).
_COLOR_TAGS = frozenset({0x00280002, 0x00280004, 0x00280006,
                         0x00281101, 0x00281102, 0x00281103, 0x00281111, 0x00281112, 0x00281113,
                         0x00281199, 0x00281201, 0x00281202, 0x00281203,
                         0x00281221, 0x00281222, 0x00281223})


def _implied_pixel_id(bits_allocated: int, representation: int, samples: int, photometric: str):
    """The SimpleITK pixel type that holds a file's STORED values as they are: what Bits
    Allocated and Pixel Representation imply, a vector of them for RGB; None for anything whose
    stored values no pixel type holds unchanged (a palette, YBR, 1-bit, float pixel data)."""
    import SimpleITK as sitk
    scalar = {(8, 0): sitk.sitkUInt8, (8, 1): sitk.sitkInt8, (16, 0): sitk.sitkUInt16,
              (16, 1): sitk.sitkInt16, (32, 0): sitk.sitkUInt32, (32, 1): sitk.sitkInt32}
    if samples == 1:
        return scalar.get((bits_allocated, representation))
    if samples == 3 and photometric == "RGB" and representation == 0:
        return {8: sitk.sitkVectorUInt8, 16: sitk.sitkVectorUInt16}.get(bits_allocated)
    return None


def _holds_stored_values(per_slice, pixel_id=None, *, value_transform: bool = False) -> bool:
    """Whether the voxels SimpleITK decoded are the source's STORED values - judged from what
    the decode did, not from the top-level rescale alone (2026-09-26, review: that let a
    MONOCHROME1 file, which GDCM inverts, and an Enhanced CT, whose rescale sits in a functional
    group, say they held stored values). All must hold: every slice's top-level rescale is the
    identity or absent (one that does not parse is not); no Pixel Value Transformation Sequence
    anywhere (``value_transform``, from the headers: SimpleITK's dictionaries show no
    sequences); and ``pixel_id``, SimpleITK's output type, is the type Bits
    Allocated and Pixel Representation imply - a widened or converted type holds other values.
    Unknown (``pixel_id`` None, no slices) is False.

    Load-bearing: it decides whether anything stated in stored-value units is written at all
    (dicom-spec §5.10) - a CT's Pixel Padding Value -2000 is -3024 in the copy's HU, and the file
    must never say otherwise to a reader of it alone."""
    # (MONOCHROME1 is not a condition since io._true_values undoes GDCM's complement: its
    # voxels then ARE the stored values wherever the rescale is the identity)
    if not per_slice or pixel_id is None or value_transform:
        return False
    implied = set()
    for d in per_slice:
        try:
            if float(d.get("0028|1053", "1") or 1) != 1 or float(d.get("0028|1052", "0") or 0) != 0:
                return False
            implied.add(_implied_pixel_id(int(str(d["0028|0100"]).strip()),
                                          int(str(d.get("0028|0103", "0")).strip() or 0),
                                          int(str(d.get("0028|0002", "1")).strip() or 1),
                                          str(d.get("0028|0004", "")).strip().upper()))
        except (KeyError, ValueError):
            return False
    return len(implied) == 1 and pixel_id in implied


class _Headers:
    """The files' headers, read once through pydicom for duckn's conversion
    (:func:`duckn.dicom_tags.tags_from_datasets`), noting on the way what SimpleITK's
    dictionaries cannot show because it lives in a sequence: a Pixel Value Transformation
    Sequence (an Enhanced object's rescale, which GDCM applies), and a Modality LUT Sequence
    (which GDCM does NOT apply - a test holds that - so the voxels stay stored values while a
    window is stated after the LUT)."""

    def __init__(self, files):
        self.files = [str(f) for f in files]
        self.read = 0
        self.value_transform = False
        self.modality_lut = False

    def __iter__(self):
        import pydicom
        while self.read < len(self.files):
            ds = pydicom.dcmread(self.files[self.read], stop_before_pixels=True, force=True)
            self.read += 1
            self._note(ds)
            yield ds

    def _note(self, ds) -> None:
        if 0x00283000 in ds:
            self.modality_lut = True
        if 0x00289145 in ds:
            self.value_transform = True
        for groups in (0x52009229, 0x52009230):   # Shared / Per-frame Functional Groups
            if groups in ds and any(0x00289145 in item for item in (ds[groups].value or [])):
                self.value_transform = True

    def finish(self) -> "_Headers":
        """Note what the headers not yet read state - after a conversion that failed midway, or
        for a judgment without one. A header that does not read states nothing."""
        while self.read < len(self.files):
            try:
                next(iter(self))
            except StopIteration:
                break
            except Exception:                  # noqa: BLE001 - that file states nothing
                self.read += 1
        return self


def stored_values_of(path) -> bool:
    """What a copy says of its voxels: True only when it states they are the source's stored
    values. A copy that does not say (one written before the field) counts as rescaled - the
    safe reading, which never lets a stored-unit tag stand beside its voxels."""
    try:
        meta, *_ = _layout(Path(path))
    except NotACopy:
        return False
    return (((_duckn(meta).get("extensions") or {}).get("dicom") or {})
            .get("stored_values") is True)


def honest_metadata(image, stored_values: bool, *, modality_lut: bool = False):
    """Strip from ``image``'s metadata dictionary every DICOM key that could contradict its
    pixels in a header written from it (dicom-spec §5.10 / §9, the user's rule of 2026-09-26):
    what duckn's fields state instead (geometry, rescale, bits allocated), anything in
    stored-value units unless the pixels ARE the stored values, the file meta group, overlay and
    curve groups, group lengths, and every private element. SimpleITK keeps all of these from the
    file it read - a CT read by ``sitk.ReadImage`` has HU pixels and still says Rescale
    Intercept -1024 and Pixel Padding Value -2000, and writes both into an NRRD header. Keys that
    are not DICOM tags (``ITK_...``, a NIfTI's own) are left alone. Returns ``image``.

    One more, found by review the same day (MONOCHROME1 and an unapplied Modality LUT, which
    once dropped the window here, are now corrected by the reader, io._true_values). A vector
    image (RGB pixels, from RGB or a
    palette GDCM expanded): Samples per Pixel, Photometric Interpretation, Planar Configuration
    and the palette describe a pixel it does not have."""
    try:
        from duckn.dicom_tags import EXCLUDED, STORED_ENCODING
    except ImportError:                      # no rules to judge by: keep no DICOM key at all
        EXCLUDED, STORED_ENCODING, judge = frozenset(), frozenset(), False
    else:
        judge = True
    also: set = set()
    if image.GetNumberOfComponentsPerPixel() > 1:
        also |= _COLOR_TAGS
    for key in list(image.GetMetaDataKeys()):
        group, _, element = key.partition("|")
        try:
            g, e = int(group, 16), int(element, 16)
        except ValueError:
            continue
        tag = (g << 16) | e
        drop = (not judge or tag in EXCLUDED or e == 0 or g == 0x0002 or g % 2 == 1
                or 0x5000 <= g <= 0x50FF or 0x6000 <= g <= 0x60FF
                or (not stored_values and tag in STORED_ENCODING) or tag in also)
        if drop:
            image.EraseMetaData(key)
    return image


def judge_file(image, path):
    """``(stored_values, modality_lut)`` for ``image`` as SimpleITK read it from the one file
    ``path`` - :func:`_holds_stored_values` on its dictionary and pixel type, with what the
    file's header says in sequences when it is DICOM. What ``get -o`` of a plain file hands
    :func:`honest_metadata` (2026-09-26)."""
    from .io import _is_dicom_file
    facts = _Headers([path] if _is_dicom_file(Path(path)) else []).finish()
    per = [{k: image.GetMetaData(k) for k in image.GetMetaDataKeys()}]
    return (_holds_stored_values(per, image.GetPixelID(),
                                 value_transform=facts.value_transform or facts.modality_lut),
            facts.modality_lut)


def _sample_units(per_slice):
    from duckn.dicom_tags import RESCALE_TYPE
    values = {d.get(RESCALE_TYPE, "").strip() for d in per_slice}
    return values.pop() if len(values) == 1 and "" not in values else None


def _source_size(content) -> tuple[int, int]:
    """(files, bytes) of the original as stored - what its digest names, recorded because the
    original will not be kept (``GET /v1/inputs/{digest}`` reports them)."""
    p = Path(content)
    files = ([q for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")]
             if p.is_dir() else [p])
    return len(files), sum(q.stat().st_size for q in files)


def _metadata(image, per_slice, *, files=(), source, source_digest, source_size=(None, None), how="uncompressed",
              n: int | None = None):
    """The duckn metadata of the copy, through duckn's own models: the geometry is
    ``from_sitk``'s (LPS - no flip either way), the rest is filled into its fields. ``n`` is the
    number of slices when ``image`` is only the geometry (a streamed copy's one-slice stand-in:
    duckn states an axis by its direction and spacing, never its length)."""
    import haversack
    import SimpleITK as sitk
    from duckn.models import SampleMetadata
    from duckn.sitk_adapter import from_sitk

    from duckn.dicom_tags import tags_from_datasets, tags_from_files, tags_from_sitk
    vol = from_sitk(image)
    meta = vol.metadata
    # the convention's version, which duckn-spec says should always be present: duckn's
    # from_sitk wrote none before 0.5.4 (found checking copies against the spec, 2026-09-26).
    # 1.0: a copy uses no 1.1 field (lut transforms, structured units, space_transforms).
    meta.version = meta.version or "1.0"
    # judged from what the decode did: SimpleITK's dictionaries and its output type (a streamed
    # copy's stand-in carries the stream's), and the headers' sequences as they are read
    held = _holds_stored_values(per_slice, image.GetPixelID())
    heads = _Headers(files)
    fields: dict = {}
    tags_version = 1
    series = slices = None
    if files:
        # the files' own headers (TAGS_VERSION 2): what SimpleITK's dictionaries cannot hold
        try:
            series, slices, fields = tags_from_datasets(heads, stored_values=held, private=False)
            if held and (heads.value_transform or heads.modality_lut):
                # an Enhanced object's rescale, seen only in its functional groups: the values
                # are not the stored ones, so convert again without the stored-unit attributes
                # (one header - an Enhanced object is one file - so this costs nothing)
                held = False
                series, slices, fields = tags_from_files(files, stored_values=False,
                                                         private=False)
            tags_version = TAGS_VERSION
        except Exception as e:             # noqa: BLE001 - the tags are provenance (2026-09-26)
            # A malformed value in any file (a KVP of "abc") made the conversion raise, and the
            # whole copy was lost over it - of a series SimpleITK reads, which it copied before
            # tags came from the headers. Never lose a copy over tags: SimpleITK's dictionaries
            # (tags_version 1) instead, as the other failures here warn.
            import sys
            print(f"warning: the DICOM headers of {source or files[0]} did not convert "
                  f"({type(e).__name__}: {e}); the copy carries SimpleITK's tags instead",
                  file=sys.stderr, flush=True)
            facts = heads.finish()
            held = held and not (facts.value_transform or facts.modality_lut)
            series = None
    if series is None:
        series, slices = (tags_from_sitk(per_slice, stored_values=held, private=False)
                          if per_slice else ({}, []))
        if series or any(slices):
            fields = {"stored_values": held}   # what tags_from_datasets states itself
    # (MONOCHROME1 and a Modality LUT once dropped the window and Photometric Interpretation
    # here, beside GDCM's inverted or un-looked-up values; io._true_values now reads the values
    # the files mean, so both are true of the copy again - 2026-09-26)
    thick, thick_each = _thickness(per_slice) if per_slice else (None, None)
    z = meta.axes[0]
    if thick is not None:
        z.thickness = thick
    n = int(vol.raw.shape[0]) if n is None else int(n)
    if (any(slices) or thick_each) and len(per_slice) == n:
        z.samples = [SampleMetadata(thickness=(thick_each[i] if thick_each else None),
                                    metadata=({"dicom": slices[i]} if slices and slices[i] else None))
                     for i in range(n)]
    units = _sample_units(per_slice) if per_slice else None
    if units:
        meta.sample_units = units
    ext = dict(meta.extensions or {})
    if series or slices:
        ext["dicom"] = {"version": DICOM_EXTENSION_VERSION, **fields, "tags": series}
    ext["haversack"] = {"kind": KIND, "version": FORMATS[how], "reader_version": READER_VERSION,
                        "tags_version": tags_version,
                        "source": source, "source_digest": source_digest,
                        "source_files": source_size[0], "source_bytes": source_size[1],
                        "reader": {"haversack": haversack.__version__,
                                   "SimpleITK": sitk.Version.VersionString()}}
    meta.extensions = ext
    return vol


def _pack(work: Path, out: Path, raw=None) -> None:
    """The zip the readers take, from a zarr directory ``work``: ``zarr.json`` deflated, then
    each chunk member stored, in chunk order. ``raw``: the one uncompressed chunk's array, written
    from memory as ``c/0/0/0`` rather than from a file (never in ``work`` - it would be a second
    copy of the volume on disk)."""
    import zipfile

    def member(name, how):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = how
        info.external_attr = 0o644 << 16
        return info
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        zf.writestr(member("zarr.json", zipfile.ZIP_DEFLATED), (work / "zarr.json").read_bytes(),
                    compresslevel=6)
        if raw is not None:
            import numpy as np
            le = np.ascontiguousarray(raw, dtype=raw.dtype.newbyteorder("<"))
            with zf.open(member("c/0/0/0", zipfile.ZIP_STORED), "w", force_zip64=True) as f:
                f.write(memoryview(le).cast("B"))
            return
        chunks = sorted((work / "c").iterdir(), key=lambda d: int(d.name))
        for d in chunks:
            with open(d / "0" / "0", "rb") as src, \
                    zf.open(member(f"c/{d.name}/0/0", zipfile.ZIP_STORED), "w",
                            force_zip64=True) as dst:
                import shutil
                shutil.copyfileobj(src, dst, 8 << 20)


def _write(vol, out: Path, how: str = "uncompressed") -> None:
    import shutil
    import tempfile

    import zarr
    from duckn.models import duckn_attrs
    from zarr.storage import LocalStore
    shape = tuple(int(n) for n in vol.raw.shape)
    if how == "zstd":
        from zarr.codecs import BloscCodec
        chunks = (min(CHUNK_SLICES, shape[0]),) + shape[1:]
        compressors = [BloscCodec(cname="zstd", clevel=ZSTD_LEVEL, shuffle=BLOSC_SHUFFLE)]
    else:
        chunks, compressors = shape, None
    # zarr writes the metadata (and the compressed chunks) into a directory; _pack makes the
    # zip, header deflated. The uncompressed chunk goes from memory straight into the zip.
    work = Path(tempfile.mkdtemp(prefix=".write-", dir=out.parent))
    try:
        arr = zarr.create_array(LocalStore(str(work)), shape=shape, dtype=vol.raw.dtype,
                                chunks=chunks, compressors=compressors,
                                attributes=duckn_attrs(vol.metadata), fill_value=0,
                                config={"write_empty_chunks": True})
        if how == "zstd":
            arr[:] = vol.raw
            _pack(work, out)
        else:
            _pack(work, out, raw=vol.raw)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _geometry_image(stream):
    """A one-slice image carrying the stream's geometry and pixel type - what ``from_sitk``
    needs to state the copy's geometry, without the volume."""
    import SimpleITK as sitk
    image = sitk.Image([int(stream.shape[2]), int(stream.shape[1]), 1], stream.pixel_id)
    image.SetOrigin(stream.origin)
    image.SetSpacing(stream.spacing)
    image.SetDirection(stream.direction)
    return image


def _write_streamed(stream, out: Path, attributes) -> list:
    """Write ``stream`` as the compressed copy at ``out``, a chunk per slab, and return each
    slab's sha256 for the check. The chunks go to a zarr directory beside ``out`` first, because
    the attributes (the per-slice tags) are known only once the last slab is read; the directory
    is then packed into the zip the reader takes (:func:`_pack`), and removed."""
    import hashlib

    import numpy as np
    import shutil
    import tempfile
    import zipfile

    import zarr
    from zarr.codecs import BloscCodec
    from zarr.storage import LocalStore
    z, y, x = (int(v) for v in stream.shape)
    k_slices = min(CHUNK_SLICES, z)
    work = Path(tempfile.mkdtemp(prefix=".stream-", dir=out.parent))
    try:
        arr = zarr.create_array(
            LocalStore(str(work)), shape=(z, y, x), dtype=stream.dtype, chunks=(k_slices, y, x),
            compressors=[BloscCodec(cname="zstd", clevel=ZSTD_LEVEL, shuffle=BLOSC_SHUFFLE)],
            fill_value=0, config={"write_empty_chunks": True})
        digests, per_slice, at = [], [], 0
        for slab, tags in stream.slabs(k_slices):
            if slab.dtype != stream.dtype or slab.shape[1:] != (y, x):
                raise ValueError(f"slab at {at}: {slab.dtype} {slab.shape}, not the stream's")
            arr[at:at + len(slab)] = slab
            digests.append(hashlib.sha256(np.ascontiguousarray(slab).tobytes()).hexdigest())
            per_slice.extend(tags)
            at += len(slab)
        if at != z:
            raise ValueError(f"the source gave {at} slices, not {z}")
        arr.update_attributes(attributes(per_slice or stream.tags))
        _pack(work, out)
        return digests
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _check_streamed(path: Path, stream, digests: list) -> None:
    """The streamed copy is the stream, checked a chunk at a time (never the whole volume):
    its layout, dtype and shape; each chunk's sha256 against its slab's; its geometry, read the
    way the reader states it, against the source's within :data:`GEOMETRY_TOLERANCE`."""
    import hashlib

    import numpy as np

    import zarr
    from zarr.storage import ZipStore
    meta, how, _, dt, shape = _layout(path)
    if how != "zstd" or tuple(shape) != tuple(stream.shape) \
            or dt.newbyteorder("=") != np.dtype(stream.dtype).newbyteorder("="):
        raise ValueError("the copy's layout is not the stream's")
    store = ZipStore(str(path), mode="r")
    try:
        arr = zarr.open_array(store, mode="r")
        k_slices = arr.chunks[0]
        for k, want in enumerate(digests):
            got = np.ascontiguousarray(arr[k * k_slices:(k + 1) * k_slices])
            if hashlib.sha256(got.tobytes()).hexdigest() != want:
                raise ValueError(f"chunk {k} does not read back as the slab written")
    finally:
        store.close()
    origin, spacing, direction = _geometry_of(_duckn(meta))
    for got, want in ((origin, stream.origin), (spacing, stream.spacing),
                      (direction, stream.direction)):
        if not np.allclose(got, want, rtol=0, atol=GEOMETRY_TOLERANCE):
            raise ValueError("the copy's geometry is not the source's")


def transcode(content, entry, *, source=None, source_digest=None) -> Path | None:
    """Write ``entry``'s copy of ``content`` and return its path - or None, when the input is
    not one to transcode (:func:`wanted`), the reader refuses it, or the copy does not read back
    as exactly what the reader produced; the caller then keeps the original. Verified before it
    is placed: voxels, pixel type and geometry, read back through :func:`read_copy`."""
    import numpy as np
    import SimpleITK as sitk

    from . import io as nio
    from .errors import InputError
    if not wanted(content):
        return None
    # Slab by slab where the source allows it (2026-09-25): memory bounded by a slab, where the
    # whole-volume path below holds the volume several times over (~3 GB at peak for a
    # 709-slice CT, in an api container with 2 GB). The compressed form only - it is stored in
    # whole-slice chunks already; the uncompressed form is one chunk.
    try:
        if compression() == "zstd":
            from .input_stream import stream_of
            stream = stream_of(content)
            if stream is not None:
                return _transcode_streamed(stream, content, entry, source=source,
                                           source_digest=source_digest)
    except InputError:
        return None                          # refused: the original stays, and fails at read
    except Exception as e:                   # noqa: BLE001 - the whole path below decides
        import sys
        print(f"warning: {source or content} is copied whole: the slab reader could not plan it "
              f"({type(e).__name__}: {e})", file=sys.stderr, flush=True)
    try:
        image, per_slice, files = nio.read_image_and_tags(content)
    except InputError:
        return None                          # refused: the original stays, and fails at read
    except Exception as e:                   # noqa: BLE001 - review, 2026-09-25
        # Not only a refusal: SimpleITK raises a bare RuntimeError on, e.g., slices of
        # different sizes. That escaped this function, the cache tore the entry down, and a
        # fetched input was downloaded again on every job (an upload was a 500) - where
        # before the copy it was kept and failed at read. Keep the original, and say why.
        import sys
        print(f"warning: no input copy for {source or content}: the reader failed "
              f"({type(e).__name__}: {e}); the original is kept", file=sys.stderr, flush=True)
        return None
    final = copy_path(entry)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name("." + COPY_NAME + ".partial")
    try:
        how = compression()
        vol = _metadata(image, per_slice, files=files, source=source, source_digest=source_digest,
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


def _transcode_streamed(stream, content, entry, *, source, source_digest) -> Path | None:
    """:func:`transcode` for a slab source: written a chunk per slab, checked a chunk at a time,
    placed as the whole path places. Any failure keeps the original, as there."""
    from duckn.models import duckn_attrs
    final = copy_path(entry)
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name("." + COPY_NAME + ".partial")
    geometry = _geometry_image(stream)
    size = _source_size(content)
    n = int(stream.shape[0])

    def attributes(per_slice):
        vol = _metadata(geometry, per_slice, files=stream.files, source=source, source_digest=source_digest,
                        source_size=size, how="zstd", n=n)
        return duckn_attrs(vol.metadata)
    try:
        digests = _write_streamed(stream, partial, attributes)
        _check_streamed(partial, stream, digests)
        os.replace(partial, final)
        return final
    except Exception as e:                 # noqa: BLE001 - any failure keeps the original
        import sys
        print(f"warning: no input copy for {source or content}: {type(e).__name__}: {e}; "
              "the original is kept", file=sys.stderr, flush=True)
        partial.unlink(missing_ok=True)
        return None


# -- reading -----------------------------------------------------------------------------

def _layout(path: Path):
    """``(zarr.json dict, how, chunk data offset, dtype, shape)`` for one of the two layouts this
    module writes - ``how`` "uncompressed" (one stored chunk, mapped at the offset) or "zstd" (blosc slabs of
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
    # the chunks must be stored (mapped at their offset, or referenced by range); the header
    # may be deflated - it is, since 2026-09-26 - as it is only ever read whole
    stored = all(i.compress_type == zipfile.ZIP_STORED
                 for n, i in members.items() if n.startswith("c/"))
    names = {n for n in members if n.startswith("c/")}
    # any blosc configuration: zarr decodes it exactly or fails, and a failure is NotACopy
    if len(codecs) == 2 and codecs[1].get("name") == "blosc":
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
    return meta, "uncompressed", info.header_offset + 30 + name_len + extra_len, dt, shape


def stored_compression(path) -> str:
    """How a copy is stored - "uncompressed" or "zstd" - read from its layout, not from what it says."""
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
        try:
            from duckn.dicom_tags import to_sitk_strings
            restored = to_sitk_strings(tags)
        except ImportError:
            # an environment without duckn or pydicom's data dictionary (a lean install, or an
            # image built before they were core): the tags are provenance, never needed - the image is
            # read without them rather than not at all
            restored = {}
        for key, value in restored.items():
            image.SetMetaData(key, value)
        # whatever the copy holds, nothing restored may contradict the voxels: a copy that does
        # not say its voxels are the stored values is judged as rescaled
        stored = (((attrs.get("extensions") or {}).get("dicom") or {})
                  .get("stored_values") is True)
        honest_metadata(image, stored)
    return image


def _to_sitk(attrs: dict, raw):
    """The image, through duckn's own ``to_sitk`` where duckn is installed. Where it is not - a
    lean environment without duckn reading a copy
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
    import SimpleITK as sitk
    origin, spacing, direction = _geometry_of(attrs)
    image = sitk.GetImageFromArray(raw)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def _geometry_of(attrs: dict):
    """``(origin, spacing, direction)`` as SimpleITK states them, from the one layout this module
    writes: LPS space, axes z, y, x, each ``space_direction`` the direction cosine times the
    spacing. A test holds this equal to duckn's ``to_sitk`` on the same file."""
    import numpy as np
    if attrs.get("space") not in ("left-posterior-superior", "LPS"):
        raise NotACopy("without duckn only an LPS copy can be read")
    vecs = [np.asarray(a["space_direction"], dtype=np.float64) for a in attrs["axes"]][::-1]
    spacing = [float(np.linalg.norm(v)) for v in vecs]      # x, y, z
    cols = np.stack([v / s for v, s in zip(vecs, spacing)], axis=1)
    return ([float(v) for v in attrs["space_origin"]], spacing,
            [float(v) for v in cols.ravel()])


def slice_tags(path, keyword: str) -> list:
    """One per-slice DICOM tag, as a list in the volume's z order (None where a slice lacks it)."""
    meta, *_ = _layout(Path(path))
    z = (_duckn(meta).get("axes") or [{}])[0]
    return [((s.get("metadata") or {}).get("dicom") or {}).get(keyword)
            for s in (z.get("samples") or [])]
