"""Image IO through SimpleITK - the same reader nnU-Net itself defaults to.

TotalSegmentator is ``Nifti1Image`` all the way down; nnU-Net has a reader registry whose
default is ``SimpleITKIO``. haversack follows
nnU-Net: SimpleITK reads NIfTI, NRRD, MetaImage, DICOM series and more, carries direction
cosines (so oblique acquisitions survive the round trip), and hands back arrays already in
(Z, Y, X) order - no transposes.

Orientation is **not** uniform across the ecosystem: TotalSegmentator canonicalizes to RAS,
while nnU-Net's default readers keep the stored axis order and only the opt-in ``*WithReorient``
variants canonicalize. :func:`read` takes ``reorient`` and :func:`reader_reorients` reads the
model's declared choice out of its plans.

Geometry is the toolkit's :class:`Geometry` value (spacing/shape in Z, Y, X; origin and
direction in SimpleITK's X, Y, Z), so the neutral core is shared rather than duplicated.
"""
from __future__ import annotations

from pathlib import Path

from .errors import InputError

import numpy as np

CANONICAL = "RAS"


def _sitk():
    import SimpleITK as sitk
    return sitk


def geometry_of(image) -> "Geometry":
    from .values import Geometry
    return Geometry(
        spacing_zyx=tuple(float(s) for s in reversed(image.GetSpacing())),
        shape_zyx=tuple(int(s) for s in reversed(image.GetSize())),
        origin_xyz=tuple(float(o) for o in image.GetOrigin()),
        direction_xyz=tuple(float(d) for d in image.GetDirection()),
    )


def orientation_of(image) -> str:
    sitk = _sitk()
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())


def _series_tags(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(IPP, IOP, PixelSpacing) of one slice, from the geometric tags."""
    sitk = _sitk()
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()

    def tag(key, n):
        if not r.HasMetaDataKey(key):
            raise InputError(f"DICOM slice {path} lacks tag {key}; cannot establish geometry")
        v = np.array([float(x) for x in r.GetMetaData(key).split("\\")], dtype=np.float64)
        if v.size != n:
            raise InputError(f"DICOM tag {key} in {path} has {v.size} values, expected {n}")
        return v

    return tag("0020|0032", 3), tag("0020|0037", 6), tag("0028|0030", 2)


def _series_geometry(files) -> tuple[tuple, tuple, tuple]:
    """Origin/direction/spacing of a sorted slice stack, from IOP + IPP only.

    The reader cannot be trusted for this: ITK's GDCM layer takes the slice axis's
    *sign* from SpacingBetweenSlices (0018,0088) - a vendor acquisition convention
    that Philips writes negative on head-to-foot scans - while origin and stacking
    follow the spatially sorted file list. On such a series the two disagree and the
    assembled volume claims a physical span the scan never occupies; canonicalization
    then hands the network a head-down body (found 2026-08-24 on CPTAC-CCRCC: 28 of
    111 structures lost, including a kidney). The IPP sequence *is* the geometry, so
    it is constructed here from the tags and (0018,0088) is never consulted.

    Raises :class:`InputError` when the positions do not describe one uniform
    orthogonal grid - a tilted gantry (slice positions not advancing along the image
    normal), inconsistent orientations, duplicate slices, or gaps.
    """
    if len(files) < 2:
        raise InputError("DICOM series has fewer than 2 slices; not a 3D volume")
    ipps, iops, spacings = zip(*(_series_tags(f) for f in files))
    ipps = np.stack(ipps)
    if (np.abs(np.stack(iops) - iops[0]).max() > 1e-4
            or np.abs(np.stack(spacings) - spacings[0]).max() > 1e-4):
        raise InputError("DICOM series mixes image orientations or pixel spacings; "
                         "not a single uniform volume")
    row, col = iops[0][:3], iops[0][3:]
    n_hat, dz = _slice_axis(ipps, row, col)
    direction = np.stack([row, col, n_hat], axis=1)    # columns = x, y, z axes
    ps = spacings[0]                                    # (row spacing, column spacing)
    return (tuple(float(v) for v in ipps[0]),
            tuple(float(v) for v in direction.ravel()),
            (float(ps[1]), float(ps[0]), dz))


def _slice_axis(ipps, row, col) -> tuple[np.ndarray, float]:
    """``(unit slice normal, step in mm)`` of positions ``ipps`` (n x 3, in slice order) on
    planes spanned by ``row`` and ``col`` - or :class:`InputError` when they are not one uniform
    orthogonal grid. The rule :func:`_series_geometry` applies to a series and
    :func:`_check_frame_positions` to a multi-frame file's frames (2026-09-26)."""
    span = ipps[-1] - ipps[0]
    extent = float(np.linalg.norm(span))
    if not extent > 0:
        # the first and last slices at one position (a duplicated end slice, which GDCM then
        # sorts by name): dividing by the zero span made every check below compare against
        # NaN and pass, and the volume read with NaN spacing (review, 2026-09-25)
        raise InputError("duplicate slice positions in the series: its first and last slices "
                         "sit at one position")
    n_hat = span / extent
    if abs(float(np.dot(n_hat, np.cross(row, col)))) < 0.999:
        raise InputError("non-orthogonal acquisition (gantry tilt or shear): slice "
                         "positions do not advance along the image normal")
    s = (ipps - ipps[0]) @ n_hat                       # position along the normal, mm
    resid = np.linalg.norm((ipps - ipps[0]) - s[:, None] * n_hat[None, :], axis=1)
    if resid.max() > 0.05:
        raise InputError("slice positions drift off the image normal "
                         f"(max {resid.max():.3f} mm): tilted or sheared acquisition")
    steps = np.diff(s)
    if steps.min() <= 0:
        raise InputError("duplicate or non-monotonic slice positions in the series")
    dz = float(steps.mean())
    if np.abs(steps - dz).max() > max(0.01, 1e-3 * dz):
        raise InputError("non-uniform slice spacing "
                         f"(steps {steps.min():.3f}-{steps.max():.3f} mm): "
                         "missing or duplicate slices")
    return n_hat, dz


def _header(f):
    """pydicom's reading of a DICOM file's header, never its pixels (by force: GDCM reads a file
    without its preamble too) - what the file says its values mean. Asked of every file the
    reader decodes (2026-09-26): the corrections and the check below need each slice's own."""
    import pydicom
    try:
        return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
    except Exception as e:                     # noqa: BLE001 - whatever pydicom raises
        raise InputError(f"{f}: its DICOM header does not read ({type(e).__name__}), so what "
                         "its values mean is unknown") from None


def _text(ds, tag: int):
    """A header value as its text, never converted: pydicom's DS conversion raises on a malformed
    value, and what to make of one is the caller's to decide."""
    if tag not in ds:
        return None
    value = ds.get_item(tag).value
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return None if value is None else str(value)


#: A rescale nobody states (as opposed to one stated as the identity, or one that does not parse)
_UNSTATED = object()


def _stated_rescale(item):
    """(slope, intercept) of the rescale ``item`` (a dataset or a Pixel Value Transformation
    item) states - None when it does not parse, :data:`_UNSTATED` when it states none."""
    if item is None or (0x00281053 not in item and 0x00281052 not in item):
        return _UNSTATED
    return _rescale_pair(_text(item, 0x00281053), _text(item, 0x00281052))


def _group_rescale(groups, k: int = 0):
    """The rescale functional-group item ``k`` of ``groups`` states, or :data:`_UNSTATED`."""
    if not groups or k >= len(groups):
        return _UNSTATED
    seq = groups[k].get("PixelValueTransformationSequence")
    return _stated_rescale(seq[0]) if seq else _UNSTATED


def _states_rescale(ds) -> bool:
    """Whether ``ds`` states a rescale anywhere: top level, shared or per-frame groups."""
    per = ds.get("PerFrameFunctionalGroupsSequence") or []
    return (_stated_rescale(ds) is not _UNSTATED
            or _group_rescale(ds.get("SharedFunctionalGroupsSequence")) is not _UNSTATED
            or any(_group_rescale(per, k) is not _UNSTATED for k in range(len(per))))


def _frame_rescales(ds, n: int):
    """``(applied, meant)`` for the file ``ds`` read as ``n`` frames: the rescale GDCM applies to
    EVERY frame, and the one each frame means. GDCM takes the Shared Functional Groups' rescale,
    else the first frame's own, else the top-level one - measured 2026-09-26: an Enhanced file
    with a shared (1, -1024) and per-frame (2, -500), (1, -1024), (0.5, 7) read every frame at
    (1, -1024); without the shared one, at (2, -500): up to 3513 off. A frame means its own
    (Per-frame Functional Groups), else the shared, else the top-level one. The identity where
    nothing is stated; None where the statement does not parse."""
    shared = _group_rescale(ds.get("SharedFunctionalGroupsSequence"))
    per = ds.get("PerFrameFunctionalGroupsSequence")
    top = _stated_rescale(ds)
    own = [_group_rescale(per, k) for k in range(n)]

    def first(*stated):
        return next((s for s in stated if s is not _UNSTATED), (1.0, 0.0))
    return first(shared, own[0] if own else _UNSTATED, top), [first(o, shared, top) for o in own]


def _pydicom_meaning(file, head, index, rescale, lut: bool, scalar: bool, depth: int = 0):
    """The values slice/frame ``index`` of ``file`` means, decoded by pydicom's own decoders -
    its stored values through the file's modality mapping (the Modality LUT, else ``rescale``,
    the frame's own, functional groups included, which pydicom's ``apply_modality_lut`` does not
    read) - or None where pydicom cannot decode the file (a decoder it lacks: nothing to check).
    A vector image's are its color values: a palette's through the palette, and - where the
    image holds ``depth`` = 8 bits a component and the palette's entries are 16 - their high
    bytes, as GDCM expands a palette into 8-bit RGB (measured 2026-09-26: 9472 -> 37 on
    pydicom's OBXXXX1A, and on a synthetic palette): a coarser form of the same colors."""
    import pydicom
    from pydicom.pixels import apply_color_lut, apply_modality_lut, pixel_array
    try:
        px = pixel_array(str(file), index=index)
    except Exception:                          # noqa: BLE001 - e.g. no preamble: read by force
        try:
            px = pydicom.dcmread(str(file), force=True).pixel_array
            px = px[index] if index is not None else px
        except Exception:                      # noqa: BLE001 - no decoder here
            return None
    px = np.asarray(px)
    if not scalar:
        if str(head.get("PhotometricInterpretation", "")).strip().upper() == "PALETTE COLOR":
            px = apply_color_lut(px, head)
            if depth == 8 and px.dtype.itemsize == 2:
                px = px // 256
        return px.astype(np.float64)
    if lut:
        return np.asarray(apply_modality_lut(px, head), dtype=np.float64)
    return px.astype(np.float64) * rescale[0] + rescale[1]


def _true_values(image, files, heads=None):
    """``image`` with the values its DICOM files mean, where GDCM hands over others - and checked
    against pydicom's own decode (2026-09-26, the user's call: fix the reader, as for the
    slice-axis sign). What GDCM does, measured:

    - **MONOCHROME1.** GDCM complements each stored value within Bits Stored (unsigned: 2^b-1-v;
      signed: -v-1) and then applies the rescale: an unsigned 12-bit 100 read as 3995, a signed
      -1000 as 999. Photometric Interpretation is a DISPLAY rule - the values mean what the
      Modality LUT stage makes of them, as for any other image - so the complement is undone.
    - **A Modality LUT Sequence.** GDCM applies a rescale but never a lookup table. The table is
      applied (pydicom's ``apply_modality_lut``, which clamps as DICOM says). A file stating a
      table AND a rescale is refused: DICOM says the table wins, GDCM applies the rescale.
    - **An Enhanced file's per-frame rescales.** GDCM applies one rescale to every frame
      (:func:`_frame_rescales`); a frame that states another is rescaled from its stored values
      with its own.

    Every slice (a series' headers, read once: ~0.3 s for 709 slices) is asked, not the first
    alone: a MONOCHROME1 or a table on slice 2 of 3 went through uncorrected (review,
    2026-09-26). A series mixing them, or a rescale that does not parse, is refused.

    **The check.** Wherever pydicom can decode the file, the first and the last slice (frame)
    are held against the values pydicom decodes and maps itself, corrected or not, and a
    disagreement is refused - never passed on. GDCM decoded an RLE 16-bit RGB, an Explicit VR
    Big Endian 32-bit dose (250085395 for 1249000) and a 1-bit SEG (0/255 for 0/1) wrongly
    without a word (review, 2026-09-26). A decoder pydicom lacks means no check, except for a
    per-frame rescale correction, which is refused unchecked.

    ``files``: the DICOM files in the image's slice order, or one file for all of a multi-frame
    file's frames; ``heads``: their headers, when the caller has read them."""
    import SimpleITK as sitk
    files = [str(f) for f in files]
    heads = list(heads) if heads is not None else [_header(f) for f in files]
    n = int(image.GetSize()[2]) if image.GetDimension() == 3 else 1
    if len(files) == n:
        slots = []
        for f, h in zip(files, heads):
            applied, (meant,) = _frame_rescales(h, 1)
            slots.append((f, h, None, applied, meant))
    elif len(files) == 1:
        applied, meants = _frame_rescales(heads[0], n)
        slots = [(files[0], heads[0], k if n > 1 else None, applied, m)
                 for k, m in enumerate(meants)]
    else:
        raise InputError(f"{files[0]}: {len(files)} files for {n} slices")
    scalar = image.GetNumberOfComponentsPerPixel() == 1
    mono1 = [str(h.get("PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1"
             for _, h, *_ in slots]
    lut = ["ModalityLUTSequence" in h for _, h, *_ in slots]
    fix = frames_differ = False
    if scalar:
        if len(set(zip(mono1, lut))) > 1:
            raise InputError(f"{files[0]}: the series mixes MONOCHROME1 or a Modality LUT across "
                             "its slices, which one volume cannot hold as one kind of value")
        if lut[0] and any(_states_rescale(h) for h in heads):
            raise InputError(f"{files[0]}: a Modality LUT Sequence and a rescale in one file - "
                             "DICOM says the table wins, GDCM applies the rescale; refusing to "
                             "choose")
        if any(a is None or m is None for *_, a, m in slots):
            raise InputError(f"{files[0]}: a Rescale Slope or Intercept that is not a number: "
                             "what the values mean is unknown")
        frames_differ = any(m != a for *_, a, m in slots) and not lut[0]
        fix = mono1[0] or lut[0] or frames_differ
    out = sitk.GetArrayViewFromImage(image)
    if image.GetDimension() == 2:
        out = out[None]
    if fix:
        from pydicom.pixels import apply_modality_lut
        real = np.empty(out.shape, dtype=np.float64)
        for k, (f, h, _, (sa, ia), (sm, im)) in enumerate(slots):
            if sa == 0:
                raise InputError(f"{f}: a Rescale Slope of 0: its stored values cannot be "
                                 "recovered")
            v = (out[k].astype(np.float64) - ia) / sa          # what GDCM rescaled
            r = np.round(v)
            if np.abs(v - r).max() < 1e-3:
                v = r                                           # stored values are integers
            if mono1[k]:                                        # GDCM's complement, undone
                signed = int(h.get("PixelRepresentation", 0) or 0) == 1
                v = -v - 1 if signed else float(2 ** int(h.BitsStored) - 1) - v
            real[k] = apply_modality_lut(v.astype(np.int64), h) if lut[k] else sm * v + im
    got = real if fix else out
    what = ("MONOCHROME1" if mono1[0] else "Modality LUT" if lut[0] else "per-frame rescale")
    checked = False
    loose = 1e-6 if image.GetPixelID() in (sitk.sitkFloat32, sitk.sitkVectorFloat32) else 0.0
    for k in sorted({0, n - 1}):
        f, h, index, _, meant = slots[k]
        want = _pydicom_meaning(f, h, index, meant, lut[k], scalar, 8 * out.dtype.itemsize)
        if want is None:
            continue
        have = np.asarray(got[k], dtype=np.float64)
        if have.shape != want.shape or not np.allclose(have, want, rtol=loose, atol=1e-6):
            if fix:
                raise InputError(f"{f}: GDCM's values could not be corrected to the file's own "
                                 f"({what}): refusing it")
            diff = (float(np.abs(have - want).max()) if have.shape == want.shape
                    else f"shape {have.shape} for {want.shape}")
            raise InputError(f"{f}: GDCM decodes other values than pydicom does from the same "
                             f"file (slice {k}, max difference {diff}): refusing to guess which "
                             "are right")
        checked = True
    if frames_differ and not checked:
        raise InputError(f"{files[0]}: its frames state different rescales, which GDCM reads as "
                         "one, and pydicom cannot decode it to check the correction: refusing it")
    if not fix:
        return image
    if np.all(real == np.round(real)):
        lo, hi = real.min(), real.max()
        for dt in (out.dtype, np.int16, np.uint16, np.int32):
            if np.issubdtype(dt, np.integer) and np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
                real = real.astype(dt)
                break
    if real.dtype == np.float64 and np.issubdtype(out.dtype, np.floating):
        real = real.astype(out.dtype)
    fixed = sitk.GetImageFromArray(real if image.GetDimension() == 3 else real[0])
    fixed.CopyInformation(image)
    for key in image.GetMetaDataKeys():
        fixed.SetMetaData(key, image.GetMetaData(key))
    return fixed


def _frame_planes(ds):
    """``(positions, orientations, pixel spacings)`` of a multi-frame file's frames, from its
    header: the Per-frame Functional Groups' own, the orientation and spacing else the Shared
    ones' (a frame stating none has None there) - or, for a file without per-frame groups, from
    the Grid Frame Offset Vector (RTDOSE: relative to the position along the normal when its
    first offset is 0, else the frames' z coordinates, DICOM C.8.8.3.2). None when the header
    places no frame at all."""
    def attr(item, seq, key):
        s = item.get(seq) if item is not None else None
        if not s or key not in s[0]:
            return None
        return [float(v) for v in s[0].get(key)]
    shared = (ds.get("SharedFunctionalGroupsSequence") or [None])[0]
    per = ds.get("PerFrameFunctionalGroupsSequence")
    if per:
        return ([attr(it, "PlanePositionSequence", "ImagePositionPatient") for it in per],
                [attr(it, "PlaneOrientationSequence", "ImageOrientationPatient")
                 or attr(shared, "PlaneOrientationSequence", "ImageOrientationPatient")
                 for it in per],
                [attr(it, "PixelMeasuresSequence", "PixelSpacing")
                 or attr(shared, "PixelMeasuresSequence", "PixelSpacing") for it in per])
    offsets = ds.get("GridFrameOffsetVector")
    if offsets is None or "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
        return None
    off = [float(v) for v in (offsets if hasattr(offsets, "__len__") else [offsets])]
    ipp = np.array([float(v) for v in ds.ImagePositionPatient])
    iop = [float(v) for v in ds.ImageOrientationPatient]
    normal = np.cross(iop[:3], iop[3:])
    pos = ([list(ipp + o * normal) for o in off] if off and off[0] == 0
           else [[ipp[0], ipp[1], o] for o in off])
    return pos, [iop] * len(off), [None] * len(off)


def _check_frame_positions(p, image) -> None:
    """Refuse a multi-frame DICOM file whose frames SimpleITK places where they were not
    acquired (2026-09-26). GDCM places an Enhanced file's frames on a uniform grid from the
    first frame's position and one spacing, whatever each frame's own position says: frames at
    z 10, 12, 16, 18 were placed at 10, 11, 12, 13, and the copy then carried the true per-frame
    positions beside that grid. The frames' positions are checked the way
    :func:`_series_geometry` checks a series' slices - one uniform step along the image normal -
    and against the grid the reader built: its origin, slice spacing and slice direction. A file
    whose frames state no positions is read as before.

    As a series is (review, 2026-09-26): frames whose orientation or pixel spacing differ from
    each other are refused - GDCM took the first frame's for all, silently - and so is an RTDOSE
    whose Grid Frame Offset Vector is not the uniform grid GDCM placed its frames on."""
    n = int(image.GetSize()[2])
    if n < 2:
        return
    try:
        import pydicom
        planes = _frame_planes(pydicom.dcmread(str(p), stop_before_pixels=True, force=True))
    except Exception:                          # noqa: BLE001 - an unreadable header (or no
        return                                 # pydicom: a lean install) states nothing
    if planes is None:
        return
    positions, orientations, spacings = planes
    if len(positions) != n:
        raise InputError(f"{p}: {len(positions)} frame positions for {n} frames: the frames "
                         "cannot be placed where they were acquired")
    for stated in (orientations, spacings):
        known = [np.asarray(v, dtype=np.float64) for v in stated if v is not None]
        if known and any(v.shape != known[0].shape or np.abs(v - known[0]).max() > 1e-4
                         for v in known):
            raise InputError(f"{p}: its frames mix image orientations or pixel spacings "
                             "(Per-frame Functional Groups): not a single uniform volume")
    if any(v is None for v in positions):
        return
    ipps = np.asarray(positions, dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    try:
        n_hat, dz = _slice_axis(ipps, direction[:, 0], direction[:, 1])
    except InputError as e:
        raise InputError(f"{p}: the frames' positions (Per-frame Functional Groups or Grid "
                         f"Frame Offset Vector) are not one uniform grid - {e}") from None
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = float(image.GetSpacing()[2])
    tol = max(0.01, 1e-3 * dz)
    if (np.linalg.norm(ipps[0] - origin) > tol or abs(spacing - dz) > tol
            or np.abs(direction[:, 2] - n_hat).max() > 1e-3):
        raise InputError(
            f"{p}: the reader places the frames {spacing:.3f} mm apart from "
            f"{tuple(round(float(v), 3) for v in origin)}, but the file places them "
            f"{dz:.3f} mm apart from {tuple(round(float(v), 3) for v in ipps[0])}: refusing "
            "to place frames where they were not acquired")


def _check_gzip(p) -> None:
    """Refuse a gzip file whose stream does not end as gzip says it must (2026-09-26, found by
    a black-box review of the server): SimpleITK read a ``.nii.gz`` cut to half its length
    without an error, as 47 % zeros, and it was copied and segmented. Python's gzip refuses it.
    Decompressed once to the end - every member, the output thrown away - so a truncated stream
    and one whose CRC or length does not match are both caught: ~0.05 s for a 23 MB NIfTI.
    A file that is not gzip is left alone."""
    import zlib
    try:
        f = open(p, "rb")
    except OSError:
        return                                 # absent or unreadable: the reader says which
    with f:
        if f.read(2) != b"\x1f\x8b":
            return
        f.seek(0)
        d, ended = zlib.decompressobj(31), False
        try:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                while chunk:
                    if ended:                  # another member, or trailing bytes
                        if chunk[:2] != b"\x1f\x8b":
                            return             # gzip's own readers ignore what follows
                        d, ended = zlib.decompressobj(31), False
                    d.decompress(chunk)
                    chunk, ended = (d.unused_data, True) if d.eof else (b"", False)
        except zlib.error as e:
            raise InputError(f"{p}: a corrupt gzip stream ({e}): refusing to read what "
                             "survives of it") from None
    if not ended:
        raise InputError(f"{p}: the gzip stream ends early - the file is truncated; refusing to "
                         "read what survives of it")


def _sitk_reason(e: Exception) -> str:
    """The last line of a SimpleITK error: the reason, without the C++ source location and
    without the quoted path (the caller names the file)."""
    import re
    lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
    reason = lines[-1] if lines else str(e)
    reason = reason.removeprefix("sitk::ERROR:").strip()
    return re.sub(r'"[^"]*"', "the file", reason)


def _readable(p) -> None:
    """Absent and unreadable are different mistakes; SimpleITK reports both as 'does not
    exist', so they are told apart before it is asked."""
    import os
    p = Path(p)
    if not p.exists():
        raise InputError(f"input not found: {p}")
    if not os.access(p, os.R_OK):
        raise InputError(f"cannot read {p}: permission denied")


def _refuse_several_series(directory, ids) -> None:
    """One series per input: a directory holding several is refused, never read as one.

    Asked with GDCM's series ids for the directory, which is one SeriesInstanceUID per
    series. GDCM answers ``GetGDCMSeriesFileNames(directory)`` with the first series it
    finds and says nothing of the rest, so until 2026-09-11 a folder of two series (3 and
    5 slices) read as the 3-slice one without a word: ``segment ./study/`` segmented one
    series of several and ``get`` converted one. That is the plausible, wrong result
    :func:`dicom_series_ids` exists to prevent, and ``serve`` already refused such a
    folder at upload. This is the same refusal for every other way a folder arrives: a
    local path, and a fetch that flattens a prefix into one directory - ``!<folder>/``
    inside an archive, ``<bucket>/<prefix>/`` - which lands a study's series side by
    side. ``idc:`` and ``tcia:`` fetch one series by construction.

    An object without pixel data is not a series to GDCM, so a CT beside its RTSTRUCT
    still reads (probed 2026-09-11). Which series was wanted is not a question a reader
    can answer, so there is no flag to pick one: the fix is a narrower folder.
    """
    if len(ids) > 1:
        shown = ", ".join(ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
        raise InputError(f"{directory} holds {len(ids)} DICOM series ({shown}), and an input "
                         "is one series: give the folder of one (fetching, end the source "
                         "at one series' folder, as in ...zip!<folder>/ or <bucket>/<folder>/)")


def read_image(path):
    """Read any SimpleITK-supported image (or a DICOM series directory) into a
    3D SimpleITK image **in its stored orientation**, with the IPP-derived
    geometry override applied for series directories.

    This is the task-independent half of :func:`read` - reorientation is the
    task's decision (see :func:`reader_reorients`), so a pre-reader staging
    inputs ahead of the pipeline uses this and lets ``pipeline.segment`` apply
    orientation itself, exactly as it does for any caller-held image.

    An input copy (:mod:`haversack.input_copy`, a cached input decoded once) is read by its
    own mapped reader, and falls back to the generic duckn path should its layout be any
    other than the one that reader was written for."""
    return _read_image(path, tags=False)[0]


def read_image_and_tags(path):
    """:func:`read_image`, plus the DICOM tags SimpleITK reports for it: ``(image,
    per_slice, files)``, ``per_slice`` a list of ``{"gggg|eeee": value}`` dicts - one per slice,
    in the volume's z order, for a DICOM series (the series reader's own per-slice dictionaries,
    from the SAME decode); the file's one dictionary for a single file; ``[]`` for a volume
    that carries none (a duckn store). ``files``: the DICOM files the image was read from, in
    that same z order (one for a single DICOM file, none for anything else) - the input copy
    reads their headers for its tags (:func:`duckn.dicom_tags.tags_from_files`: sequences,
    binary values and private tags, which SimpleITK's dictionaries do not hold)."""
    return _read_image(path, tags=True)


def _read_image(path, *, tags: bool):
    sitk = _sitk()
    p = Path(path)
    from .duckn_io import is_duckn_store, read_duckn_image
    from .input_copy import NotACopy, is_copy, read_copy
    per_slice: list = []
    files: list = []
    if is_copy(p):
        try:
            image = read_copy(p)
        except NotACopy:                  # another layout: duckn's own reader
            image = read_duckn_image(p)
    elif is_duckn_store(p):               # a duckn/zarr volume (directory or zarr zip)
        image = read_duckn_image(p)
    elif p.is_dir():
        reader = sitk.ImageSeriesReader()
        # a folder with nothing DICOM in it (a staged single-file upload, every job of one)
        # is not handed to GDCM, which says so on stderr (_may_hold_dicom, 2026-09-26)
        files = []
        if _may_hold_dicom(p):
            _refuse_several_series(p, reader.GetGDCMSeriesIDs(str(p)))
            files = reader.GetGDCMSeriesFileNames(str(p))
        if not files:
            # not a DICOM series - but a directory holding exactly one image
            # file reads as that file (how staged single-file sources arrive)
            loose = [q for q in sorted(p.iterdir())
                     if q.is_file() and not q.name.startswith(".")]
            if len(loose) == 1:
                return _read_image(loose[0], tags=tags)
            raise InputError(f"no DICOM series found in {p}"
                             + (f" ({len(loose)} non-DICOM files)" if loose else ""))
        if len(files) == 1:
            # One file: a multi-frame object (an Enhanced CT, an RTDOSE) is a volume on its own
            # and reads as that file, as it does given directly - a folder of it was refused
            # "fewer than 2 slices" (review, 2026-09-26). One slice is still no volume.
            image, per_slice, files = _read_image(Path(files[0]), tags=tags)
            if image.GetSize()[2] < 2:
                raise InputError("DICOM series has fewer than 2 slices; not a 3D volume")
            return image, per_slice, files
        heads = [_header(f) for f in files]     # every slice's, for the type and the values
        reader.SetFileNames(files)
        # the per-slice dictionaries come from the same decode (measured: no cost)
        reader.MetaDataDictionaryArrayUpdateOn()
        image = _execute_series(reader, p)
        # The series reader takes its output pixel type from the FIRST file and converts every
        # slice to it: a series whose first slice states no rescale (unsigned stored values)
        # and whose later slices say intercept -1024 read as uint16, the negative values
        # wrapped to 64612 (2026-09-26) - and so did slices of one rescale in other pixel
        # types (signed after unsigned, 16 bits after 8: review, 2026-09-26). Read again in a
        # type every slice's values fit.
        wide = _series_type(heads, image.GetPixelID())
        if wide is not None:
            reader.SetOutputPixelType(wide)
            image = _execute_series(reader, p)
        if tags:
            per_slice = [{k: reader.GetMetaData(i, k) for k in reader.GetMetaDataKeys(i)}
                         for i in range(len(files))]
        # The reader decodes and stacks; the geometry comes from the tags. See
        # _series_geometry for why its own claim cannot be trusted.
        origin, direction, spacing = _series_geometry(files)
        image.SetOrigin(origin)
        image.SetDirection(direction)
        image.SetSpacing(spacing)
        # the values the files mean, checked against pydicom's decode (2026-09-26)
        image = _true_values(image, files, heads)
    else:
        _readable(p)
        _check_gzip(p)                          # a truncated .nii.gz reads as zeros otherwise
        try:
            image = sitk.ReadImage(str(p))
        except RuntimeError as e:
            if "orthonormal" not in str(e):
                raise InputError(f"cannot read {p} as an image: {_sitk_reason(e)}") from None
            image = _read_with_snapped_affine(p, e)
        else:
            image = _nifti_placement(image, p)
        # a multi-frame DICOM file's frames are placed by the reader on a uniform grid: held
        # against the frames' own positions (2026-09-26); asked only of a file with frames
        several = image.GetDimension() == 3 and image.GetSize()[2] > 1
        dicom = _is_dicom_file(p)
        if several and dicom:
            _check_frame_positions(p, image)
        if dicom:
            image = _true_values(image, [p])    # the values it means, checked (2026-09-26)
        if tags:
            per_slice = [{k: image.GetMetaData(k) for k in image.GetMetaDataKeys()}]
            if dicom:
                files = [str(p)]
    if image.GetDimension() != 3:
        raise InputError(f"expected a 3D image; {p} has {image.GetDimension()} dimensions")
    return image, per_slice, files


def _rescale_pair(slope, intercept) -> tuple[float, float] | None:
    """A slice's (Rescale Slope, Rescale Intercept) as numbers - absent as the identity (1, 0) -
    or None when either does not parse. Takes SimpleITK's strings or pydicom's values."""
    def number(v, default):
        if v is None:
            return default
        text = str(v).strip().split("\\")[0].strip()
        return float(text) if text else default
    try:
        return number(slope, 1.0), number(intercept, 0.0)
    except (TypeError, ValueError):
        return None


def _execute_series(reader, directory):
    """``reader.Execute()``, a refusal where GDCM raises: a signed 12-bit MONOCHROME1 series
    made it raise a bare RuntimeError (review, 2026-09-26)."""
    try:
        return reader.Execute()
    except RuntimeError as e:
        raise InputError(f"cannot read the DICOM series in {directory}: {_sitk_reason(e)}") \
            from None


def _series_type(heads, current) -> int | None:
    """:func:`_mixed_rescale_type` of a series from its slices' headers (pydicom datasets)."""
    return _mixed_rescale_type(
        [_rescale_pair(_text(h, 0x00281053), _text(h, 0x00281052)) for h in heads],
        [_number(h, "BitsAllocated") for h in heads], current,
        bits_stored=[_number(h, "BitsStored") for h in heads],
        representation=[_number(h, "PixelRepresentation") for h in heads])


def _number(ds, keyword):
    """An integer (US) attribute's value, or its text where it does not convert - which
    :func:`_mixed_rescale_type` reads as a width it cannot know."""
    try:
        return ds.get(keyword)
    except Exception:                          # noqa: BLE001 - a malformed value
        return "?"


def _mixed_rescale_type(pairs, bits_allocated=(), current=None, *, bits_stored=(),
                        representation=()) -> int | None:
    """The SimpleITK pixel type a series must be read in when its slices do not share one
    rescale and pixel type and ``current`` - the type the reader took from the FIRST file, and
    converts every slice to - cannot hold them all; None when it can, when they share one, or
    when a rescale does not parse (which no type fixes). Found 2026-09-26: an unsigned first
    slice without rescale and later slices at intercept -1024 read as uint16, -924 wrapped to
    64612; a first slice at slope 1 and later ones at 0.5 read as int32, the halves truncated;
    and (review, the same day) slices of ONE rescale in other pixel types - signed after
    unsigned, Bits Stored 16 after 12, Bits Allocated 16 after 8 - wrapped the same way, as the
    rescale alone was asked. Each slice's range is what its Bits Stored (else Bits Allocated)
    and Pixel Representation allow, rescaled; integer rescales of values up to 16 bits fit
    int32 - what GDCM itself gives a rescaled integer series; anything else needs float64, as
    GDCM gives a fractional slope. A series whose first file already reads in such a type (PET's
    per-slice slopes: float64) is read once, as before."""
    import SimpleITK as sitk
    if not pairs or None in pairs or current == sitk.sitkFloat64:
        return None

    def column(values):
        values = list(values)
        return [None if v in (None, "") else str(v).strip() for v in values] \
            + [None] * (len(pairs) - len(values))
    kinds = list(zip(pairs, column(bits_allocated), column(bits_stored),
                     column(representation)))
    if all(k == kinds[0] for k in kinds):
        return None
    try:
        allocated = [None if a is None else int(a) for _, a, _, _ in kinds]
        stored = [None if b is None else int(b) for _, _, b, _ in kinds]
        signed = [r is not None and int(r) == 1 for _, _, _, r in kinds]
    except ValueError:
        return sitk.sitkFloat64
    wide = any(a is not None and a > 16 for a in allocated)
    integral = all(float(v).is_integer() for p in pairs for v in p)
    lo = hi = None
    for (slope, intercept), a, b, sg in zip(pairs, allocated, stored, signed):
        bits = b or a
        if bits is None:
            lo = hi = None                      # a slice of unknown width: no range to judge by
            break
        top = 2 ** bits
        ends = [slope * v + intercept for v in ((-(top // 2), top // 2 - 1) if sg else (0, top - 1))]
        lo = min(ends) if lo is None else min(lo, *ends)
        hi = max(ends) if hi is None else max(hi, *ends)

    def holds(pixel_id):
        dt = sitk.GetArrayViewFromImage(sitk.Image([1, 1, 1], pixel_id)).dtype
        return (lo is not None and np.issubdtype(dt, np.integer)
                and np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max)
    if integral and current is not None and holds(current):
        return None
    need = (sitk.sitkInt32 if integral and not wide and (lo is None or holds(sitk.sitkInt32))
            else sitk.sitkFloat64)
    return None if current == need else need


def _is_dicom_file(p: Path) -> bool:
    """A single file read as an image that is DICOM: the Part 10 preamble, or DICOM without it
    (:func:`_dicom_by_force`). Asked when tags are wanted, and of a file with several frames
    (whose positions are checked, 2026-09-26) - so on every volume read, where a lean install
    has no pydicom to judge a preamble-less file by: that is not DICOM to it."""
    try:
        with open(p, "rb") as f:
            head = f.read(132)
    except OSError:
        return False
    if len(head) == 132 and head[128:132] == b"DICM":
        return True
    if p.name.lower().endswith((".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd")):
        return False
    try:
        return _dicom_by_force(p)
    except ImportError:
        return False


def _read_with_snapped_affine(p, itk_error, tol: float = 1e-3):
    """ITK refuses NIfTIs whose direction cosines are not exactly orthonormal;
    several published datasets (the TotalSegmentator training data among them)
    carry affines off by ~1e-4. When the deviation is tiny, snap to the
    CLOSEST rotation via SVD polar decomposition and proceed - never QR,
    whose sign indeterminacy can silently mirror the volume. Genuinely
    sheared or oblique affines (deviation beyond ``tol``) still refuse: that
    is real geometry, not float noise."""
    import numpy as np
    sitk = _sitk()
    try:
        import nibabel as nib
    except ImportError:
        raise InputError(f"{p}: non-orthonormal direction cosines and nibabel "
                         f"is unavailable to snap them ({itk_error})") from itk_error
    ni = nib.load(str(p))
    if ni.ndim < 3:
        raise InputError(f"expected a 3D image; {p} has {ni.ndim} dimensions")
    origin, spacing, direction = _itk_geometry(ni.affine, p, tol, cause=itk_error)
    a = np.asanyarray(ni.dataobj)
    if a.ndim == 4 and a.shape[3] == 1:
        a = a[..., 0]
    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(a, (2, 1, 0))))
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def _itk_geometry(aff, p, tol: float = 1e-3, cause=None):
    """``(origin, spacing, direction)`` in ITK's LPS for a NIfTI affine (RAS), its direction
    cosines snapped to the closest rotation (SVD polar decomposition) when they are off by
    float noise, and a refusal when they are off by more than ``tol``: a shear SimpleITK's image
    cannot hold, real geometry and not noise."""
    import numpy as np
    aff = np.asarray(aff, dtype=float)
    R = aff[:3, :3]
    spacing = np.linalg.norm(R, axis=0)
    if not np.all(spacing > 0):
        raise InputError(f"{p}: degenerate affine (zero-length axis)")
    U, _S, Vt = np.linalg.svd(R / spacing)
    Rn = U @ Vt
    deviation = float(np.abs(Rn - R / spacing).max())
    if deviation > tol:
        raise InputError(
            f"{p}: direction cosines deviate from orthonormal by {deviation:.2e} "
            f"(tolerance {tol:.0e}) - this looks like genuinely sheared/oblique "
            "geometry, not float noise; refusing to guess") from cause
    flip = np.diag([-1.0, -1.0, 1.0])          # nifti RAS -> ITK LPS
    return (tuple((flip @ aff[:3, 3]).tolist()), tuple(float(x) for x in spacing),
            tuple((flip @ Rn).flatten().tolist()))


#: The names a NIfTI-1 file goes by (single file, or a pair named by either half).
_NIFTI_NAMES = (".nii", ".nii.gz", ".hdr", ".hdr.gz", ".img", ".img.gz")


def _nifti1_transforms(p):
    """``(sform_code, qform_code, srow)`` from a NIfTI-1 header's own bytes (a pair's ``.hdr``
    when named by its ``.img``; gzip judged by its magic number), ``srow`` the sform's three rows
    as float64 from the header's float32; None for anything that is not a NIfTI-1 header.
    SimpleITK reports these fields, but prints ``srow_x`` to six significant digits - 0.1 mm off
    at 118.40123 - so the header is read here. NIfTI-2 needs nothing: SimpleITK 2.5 cannot read
    it at all."""
    import gzip
    import struct

    import numpy as np
    name = p.name.lower()
    if not name.endswith(_NIFTI_NAMES):
        return None
    header = p
    if name.endswith((".img", ".img.gz")):
        stem = p.name[:len(p.name) - (7 if name.endswith(".gz") else 4)]
        header = next((q for q in (p.with_name(stem + e) for e in
                                   (".hdr", ".HDR", ".hdr.gz", ".HDR.GZ")) if q.is_file()), None)
        if header is None:
            return None
    try:
        with open(header, "rb") as fh:
            raw = fh.read(2)
            fh.seek(0)
            raw = (gzip.open(fh).read(348) if raw == b"\x1f\x8b" else fh.read(348))
    except OSError:
        return None
    if len(raw) < 348 or raw[344:347] not in (b"n+1", b"ni1"):
        return None                       # Analyze 7.5, NIfTI-2, or not a header at all
    order = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    qform_code, sform_code = struct.unpack(order + "hh", raw[252:256])
    srow = np.array(struct.unpack(order + "12f", raw[280:328]), dtype=np.float64).reshape(3, 4)
    return int(sform_code), int(qform_code), srow


def _nifti_placement(image, p):
    """The image placed by the sform when its code is not 0, as nibabel, FSL and SPM place it
    (``get_best_affine``) and duckn's converter does - not by SimpleITK's own rule, which takes
    the qform beside an MNI (4) or aligned (2) sform: the same file was placed 5 mm apart by the
    two (2026-09-30, SimpleITK 2.5.6). nifti1.h leaves the choice to the reader; TotalSegmentator
    reads with nibabel, and this module's own fallback (:func:`_read_with_snapped_affine`)
    already placed by nibabel's affine, so a file was placed by one rule or the other depending
    on whether ITK accepted its cosines (the user's call, 2026-09-30: one rule, duckn's).

    Only a file with BOTH transforms can differ (with one, SimpleITK takes it; with neither, its
    method-1 placement is nibabel's). A singular or non-finite sform places nothing and is
    passed over, as duckn passes it over. The image is touched only when SimpleITK's placement
    differs beyond float32 noise, so a scanner converter's file, whose qform is its sform's
    float32 quaternion, reads exactly as before. A sform SimpleITK's image cannot hold (a
    shear beyond float noise) is refused, as the fallback refuses it."""
    import numpy as np
    found = _nifti1_transforms(p)
    if found is None or image.GetDimension() != 3:
        return image
    sform_code, qform_code, srow = found
    if sform_code <= 0 or qform_code <= 0:
        return image
    if not np.all(np.isfinite(srow)) or abs(np.linalg.det(srow[:, :3])) < 1e-12:
        return image
    aff = np.vstack([srow, [0.0, 0.0, 0.0, 1.0]])
    origin, spacing, direction = _itk_geometry(aff, p)
    scale = max(1.0, float(np.max(np.abs(aff))))
    if (np.allclose(image.GetOrigin(), origin, rtol=0, atol=1e-4 * scale)
            and np.allclose(image.GetSpacing(), spacing, rtol=1e-5, atol=0)
            and np.allclose(image.GetDirection(), direction, rtol=0, atol=1e-5)):
        return image
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def read(path, *, reorient: bool = True, target: str = CANONICAL) -> tuple[np.ndarray, "Geometry", str]:
    """Read any SimpleITK-supported image (or a DICOM series directory).

    Returns ``(array (Z, Y, X), geometry, original orientation code)``.

    With ``reorient=True`` (the default, and what TotalSegmentator does) the array comes back
    in ``target`` - RAS unless a model's own packaging says otherwise (MRSegmentator's reader
    forces LPS); feeding a model the mirror of what it saw in training mirrors left and right
    silently. But **nnU-Net's default reader does not reorient**: ``SimpleITKIO`` hands the
    array over in its stored axis order, and only the opt-in ``SimpleITKIOWithReorient`` /
    ``NibabelIOWithReorient`` canonicalize. A model trained through the plain reader therefore
    expects its own acquisition orientation, so pass ``reorient=False`` for those - see
    :func:`reader_reorients`.
    """
    sitk = _sitk()
    image = read_image(path)
    original = orientation_of(image)
    if reorient:
        image = sitk.DICOMOrient(image, target)
    return sitk.GetArrayFromImage(image), geometry_of(image), original


def reader_reorients(model_folder) -> bool:
    """Does this nnU-Net model's declared reader reorient to RAS?

    The plans name an ``image_reader_writer`` (``dataset.json``'s ``overwrite_image_reader_writer``
    wins when present). Only the ``*WithReorient`` variants canonicalize; ``SimpleITKIO`` and
    ``NibabelIO`` - the defaults, and what almost every trained model declares - do not. Getting
    this wrong mirrors left and right, which shows up as paired structures scoring Dice 0 while
    unpaired ones merely degrade.
    """
    import json
    f = Path(model_folder)
    name = None
    ds = f / "dataset.json"
    if ds.exists():
        name = json.loads(ds.read_text(encoding="utf-8")).get("overwrite_image_reader_writer")
    if not name:
        pl = f / "plans.json"
        if pl.exists():
            name = json.loads(pl.read_text(encoding="utf-8")).get("image_reader_writer")
    return bool(name) and "reorient" in str(name).lower()


def to_image(array_zyx: np.ndarray, geometry: "Geometry"):
    """(Z, Y, X) array + geometry -> a SimpleITK image in the canonical frame."""
    sitk = _sitk()
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_zyx))
    image.SetSpacing(tuple(float(s) for s in reversed(geometry.spacing_zyx)))
    image.SetOrigin(tuple(float(o) for o in geometry.origin_xyz))
    image.SetDirection(tuple(float(d) for d in geometry.direction_xyz))
    return image


def restore_orientation(image, original: str):
    """Undo the canonical reorientation, so the output sits in the input's own frame."""
    return _sitk().DICOMOrient(image, original)


def orientation_transform(geometry: "Geometry", target: str) -> tuple[tuple[int, int, int], tuple[bool, bool, bool], tuple[float, ...], tuple[float, float, float]]:
    """What ``DICOMOrient`` would do to an array with this geometry, as a torch-applicable recipe.

    Returns ``(perm, flips, direction_xyz, spacing_xyz)``: the new (Z, Y, X) axis ``k`` is the
    old axis ``perm[k]``, reversed if ``flips[k]``. Derived by running ``DICOMOrient`` itself on
    a 3x3x3 probe whose voxel values encode their own index, so nothing of SimpleITK's
    orientation logic is re-implemented - the probe *is* the answer. Direction and spacing of
    the result are size-independent, so the probe's are the real ones; the origin is not, and
    :func:`reorient` computes it from the real size.
    """
    sitk = _sitk()
    probe = sitk.GetImageFromArray(np.arange(27, dtype=np.int32).reshape(3, 3, 3))
    probe.SetSpacing(tuple(float(s) for s in reversed(geometry.spacing_zyx)))
    probe.SetOrigin(tuple(float(o) for o in geometry.origin_xyz))
    probe.SetDirection(tuple(float(d) for d in geometry.direction_xyz))
    out = sitk.DICOMOrient(probe, target)
    arr = sitk.GetArrayFromImage(out)                               # (3, 3, 3), values = old flat index
    old_idx = np.stack(np.unravel_index(arr, (3, 3, 3)), axis=-1)  # (3, 3, 3, 3): old (z, y, x) per new voxel
    perm, flips = [], []
    for k in range(3):
        first = old_idx[tuple(0 if a != k else 0 for a in range(3))]
        last = old_idx[tuple(0 if a != k else 2 for a in range(3))]
        delta = last - first
        (axis,) = np.nonzero(delta)[0]
        perm.append(int(axis))
        flips.append(bool(delta[axis] < 0))
    return tuple(perm), tuple(flips), tuple(float(d) for d in out.GetDirection()), tuple(float(s) for s in out.GetSpacing())


def reorient(array_zyx, geometry: "Geometry", target: str):
    """Reorient a (Z, Y, X) array - numpy or torch, on any device - to ``target`` exactly as
    ``DICOMOrient`` would, but as a permute + flip on the tensor where it lives.

    On a 418 M-voxel label volume ``DICOMOrient`` is ~4 s of single-threaded CPU; the same
    permutation on the GPU is milliseconds. Returns ``(array, Geometry)`` with the array on
    the host (numpy), ready for :func:`to_image`.
    """
    from .values import Geometry
    perm, flips, direction, spacing_xyz = orientation_transform(geometry, target)
    is_torch = hasattr(array_zyx, "permute")
    shape_old = tuple(int(n) for n in array_zyx.shape)
    if is_torch:
        t = array_zyx.permute(*perm)
        dims = [k for k, f in enumerate(flips) if f]
        if dims:
            t = t.flip(dims)
        out = t.contiguous().cpu().numpy()
    else:
        out = np.transpose(np.asarray(array_zyx), perm)
        for k, f in enumerate(flips):
            if f:
                out = np.flip(out, axis=k)
        out = np.ascontiguousarray(out)
    # origin: world position of the new voxel (0, 0, 0) = the old voxel at index 0, or n-1 on a
    # flipped axis, along each old axis
    old_idx_zyx = np.zeros(3)
    for k, (p_, f) in enumerate(zip(perm, flips)):
        old_idx_zyx[p_] = (shape_old[p_] - 1) if f else 0
    d = np.asarray(geometry.direction_xyz, dtype=np.float64).reshape(3, 3)
    sp_xyz = np.asarray(geometry.spacing_zyx, dtype=np.float64)[::-1]
    origin = np.asarray(geometry.origin_xyz, dtype=np.float64) + d @ (old_idx_zyx[::-1] * sp_xyz)
    geo = Geometry(spacing_zyx=tuple(reversed(spacing_xyz)), shape_zyx=tuple(int(n) for n in out.shape),
                   origin_xyz=tuple(float(o) for o in origin), direction_xyz=direction)
    return out, geo


IMAGE_SUFFIXES = (".nii.gz", ".nii", ".seg.nrrd", ".nrrd", ".mha", ".mhd", ".nia", ".img")
_FORMAT_EXT = {"nifti": ".nii.gz", "nii": ".nii.gz", "niigz": ".nii.gz", "nrrd": ".nrrd",
               "seg.nrrd": ".seg.nrrd", "segnrrd": ".seg.nrrd", "mha": ".mha", "mhd": ".mhd"}


def format_extension(fmt: str) -> str:
    """The file extension for a ``--format`` name (e.g. ``nrrd`` -> ``.nrrd``)."""
    key = fmt.lower().lstrip(".")
    if key in _FORMAT_EXT:
        return _FORMAT_EXT[key]
    from .errors import InputError
    raise InputError(f"unknown format {fmt!r}; known: {', '.join(sorted(set(_FORMAT_EXT)))}")


def image_suffix(name) -> str | None:
    """The medical-image extension of ``name`` (``.nii.gz`` before ``.nii``), or None."""
    n = str(name).lower()
    for suf in IMAGE_SUFFIXES:                        # longest compound suffixes first
        if n.endswith(suf):
            return suf
    return None


def convert(src, dst, *, compress: bool = True) -> Path:
    """Read ``src`` (an image file or a DICOM series directory) and write it to ``dst`` as one
    volume, its format taken from ``dst``'s extension. Geometry (spacing, direction, origin) is
    preserved: nothing is reoriented or resampled. Returns ``dst``.

    A directory is read by :func:`read_image`, exactly as ``segment`` reads it: the geometry
    comes from the slice positions, and a series that is not one uniform grid (a missing or
    duplicate slice, a tilted gantry) raises the same :class:`InputError` before anything is
    written. Until 2026-09-11 this was a bare ``ImageSeriesReader``, which regrids a gapped
    series onto its mean step with only a stderr warning (IDC series
    33754f28-f0fe-4bbe-bba9-a98e4a4926ef, eay131: 3 mm slices with four 6 mm gaps, written at
    3.0577 mm with slices up to ~2.9 mm from where they were acquired) and takes the slice
    axis's sign from a negative (0018,0088) - so ``get -o scan.nii.gz`` wrote series that
    ``segment`` refuses, or places differently, as clean uniform grids on which ``segment``
    could no longer see anything wrong.

    A file is read as it stands. :func:`read_image` would also refuse one that is not 3D (a 4D
    NIfTI converts to NRRD fine) and snap a near-orthonormal affine, which is a geometry change.
    The exception is haversack's own form - an input copy, or any duckn store - which SimpleITK
    cannot open at all: it is read by :func:`read_image`, whose read of a copy is verified exact
    when the copy is written. Until 2026-09-26 `get -o scan.nii.gz` of a copied input failed
    "Unable to determine ImageIO reader" (found as `get` became the copy's door)."""
    sitk = _sitk()
    src = Path(src)
    from .duckn_io import is_duckn_store
    from .input_copy import is_copy
    if src.is_dir() or is_copy(src) or is_duckn_store(src):
        img = read_image(src)
    else:
        _readable(src)
        _check_gzip(src)                        # never convert what survives of a truncation
        try:
            img = sitk.ReadImage(str(src))
        except RuntimeError as e:
            raise InputError(f"cannot read {src} as an image: {_sitk_reason(e)}") from None
        # as it stands, but never with frames placed where they were not acquired: the gapped
        # series lesson above, for a multi-frame file's frames (2026-09-26)
        if _is_dicom_file(src):
            if img.GetDimension() == 3 and img.GetSize()[2] > 1:
                _check_frame_positions(src, img)
            img = _true_values(img, [src])      # the values the file means, checked, as segment
                                                # reads them
    # A header written from the image must not contradict its pixels (2026-09-26): SimpleITK
    # keeps the file's rescale, padding and bit tags beside pixels it has already rescaled, and
    # writes them all into an NRRD header. A copy says what its voxels are; a file read here is
    # judged as the copy judges its source - by what the decode did (its rescale, MONOCHROME1,
    # its pixel type against the stored one, the header's sequences), review of 2026-09-26.
    from .input_copy import honest_metadata, judge_file, stored_values_of
    if is_copy(src):
        stored, modality_lut = stored_values_of(src), False
    elif src.is_dir() or is_duckn_store(src):
        stored, modality_lut = False, False  # SimpleITK's series reader keeps no dictionary
    else:
        stored, modality_lut = judge_file(img, src)
    honest_metadata(img, stored, modality_lut=modality_lut)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(dst), compress)
    return Path(dst)


def write(image, path, *, compress: bool = True) -> None:
    _sitk().WriteImage(image, str(path), compress)


def dicom_series_ids(directory) -> list:
    """Every DICOM SeriesInstanceUID present in ``directory``.

    Empty when the directory holds no DICOM at all - a NIfTI, an Analyze
    hdr/img pair - which is a normal answer, not a failure.

    Exists so that a folder someone dropped in can be checked BEFORE it becomes
    an input. A mixed folder (two series, or a series plus a derived object) is
    the case where reading "the" series means picking one, and picking silently
    is how a plausible, wrong segmentation gets produced.
    """
    if not _may_hold_dicom(Path(directory)):
        return []
    sitk = _sitk()
    try:
        return list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)))
    except Exception:
        return []


def _may_hold_dicom(directory: Path) -> bool:
    """Whether any file in ``directory`` could be DICOM, asked BEFORE GDCM is: GDCM answers
    a folder with no DICOM in it correctly (no series) but prints two ITK warnings to stderr
    on the way, and the server asks this of every folder an upload is staged in - so a NIfTI
    upload put "No Series were found" into the log of every job (seen 2026-09-26 running the
    object store). The first file that could be DICOM ends the check, so a real series costs
    one header read; only a folder with none reads each file's header once. Without pydicom
    (a lean install) GDCM is asked as before."""
    try:
        import pydicom
    except ImportError:
        return True
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            pydicom.dcmread(f, stop_before_pixels=True)
            return True
        except Exception:                          # noqa: BLE001 - not plainly DICOM
            if _dicom_by_force(f):
                return True
    return False


def _dicom_by_force(f: Path) -> bool:
    """Whether a file pydicom refused to read plainly is DICOM after all - a dataset written
    without the preamble and file meta, which GDCM reads. Judged by elements a DICOM object
    carries (its SOP class, its image size, its modality), not by pydicom merely not raising:
    ``force`` makes something of almost any bytes."""
    import pydicom
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        return any(k in ds for k in ("SOPClassUID", "Rows", "Modality"))
    except Exception:                          # noqa: BLE001 - not DICOM even by force
        return False


# A ranked store output is named `.duckn` (a directory) or `.duckn.zip` (a standard zarr zip).
# The check lives HERE, dependency-free: the CLI asks it on every `segment`, before the
# inference stack, and the store modules it used to live in import rankfield and torch.
STORE_OUTPUT_SUFFIXES = (".duckn.zip", ".duckn")


def is_store_output(path) -> bool:
    """Whether an output path asks for a ranked store rather than labels. A bare ``.zip`` is
    not enough - that would silently turn a typo into a different kind of output."""
    return str(path).lower().endswith(STORE_OUTPUT_SUFFIXES)
