"""SimpleITK's DICOM tags in duckn's DICOM extension encoding (duckn ``docs/dicom-spec.md``).

A LOCAL converter, written to move into duckn beside its pydicom converter: the rules below are
the spec's, and haversack keeps them only until duckn has ``tags_from_sitk``. The source is what
SimpleITK hands over - the per-slice dictionaries ``ImageSeriesReader`` reports with
``MetaDataDictionaryArrayUpdateOn()`` (keys ``gggg|eeee``, values strings) - never a second
DICOM reader: pydicom is used as a DATA DICTIONARY only (keyword, VR and VM of a tag), and no file
is opened through it.

The encoding (spec §4): keys are PS3.6 keywords (a private or unknown tag keeps its uppercase hex
code); DS, IS and the binary integer and float VRs become JSON numbers; an attribute whose VM can
exceed 1 is always an array, one whose VM is exactly 1 a bare value; strings are stripped of the
padding DICOM adds. Tags the convention fields capture are left out (spec §9), and so are group
lengths. A tag with the same value on every slice is SERIES-level; one that differs is per slice
(spec §6.1) and comes back as one dict per slice, in the order given.

The inverse, :func:`to_sitk_strings`, gives the series-level tags back in SimpleITK's own form,
for a reader that restores them onto an image as a single-file read would show them. It is not a
byte-exact round trip: ``"120.000 "`` comes back ``"120.0"``, as the spec accepts (§4.2).
"""
from __future__ import annotations

#: Attributes the duckn convention fields capture, or that are no header at all (spec §9), by
#: tag. Their values would be a second statement of the array's geometry, shape, type or value
#: mapping - and a stale one wherever the array's encoding differs from the source's.
EXCLUDED = frozenset({
    0x7FE00010,                              # Pixel Data
    0x00200032, 0x00200037,                  # Image Position / Orientation (Patient)
    0x00280010, 0x00280011, 0x00280008,      # Rows, Columns, Number of Frames
    0x00280100, 0x00280101, 0x00280102, 0x00280103,   # Bits Allocated/Stored, High Bit, Pixel Representation
    0x00281052, 0x00281053, 0x00281054,      # Rescale Intercept, Slope, Type
    0x00283000, 0x00283002, 0x00283006,      # Modality LUT Sequence, LUT Descriptor, LUT Data
    0x00283004,                              # Modality LUT Type (-> sample_units, §2)
    0x00280030, 0x00180088,                  # Pixel Spacing, Spacing Between Slices (§2)
    0x00180050,                              # Slice Thickness (-> axes[i].thickness, §2)
})
#: The convention fields a caller must fill from the tags this module drops (spec §2):
#: Slice Thickness -> the slice axis' ``thickness`` (or per sample), Rescale Type -> the array's
#: ``sample_units``. Given by SimpleITK key, for the caller that reads them before conversion.
SLICE_THICKNESS = "0018|0050"
RESCALE_TYPE = "0028|1054"

_NUMERIC_INT = {"IS", "US", "SS", "UL", "SL", "UV", "SV"}
_NUMERIC_FLOAT = {"DS", "FL", "FD"}


def _tag(key: str) -> int | None:
    """``"0018|0060"`` -> 0x00180060; None for a key that is not a DICOM tag (``ITK_...``)."""
    if "|" not in key:
        return None
    group, elem = key.split("|", 1)
    try:
        return (int(group, 16) << 16) | int(elem, 16)
    except ValueError:
        return None


def _dictionary():
    import pydicom.datadict as dd
    return dd


def keyword_of(tag: int) -> str:
    """The PS3.6 keyword, or the uppercase hex code for a private or unknown tag (spec §4.1)."""
    if (tag >> 16) % 2 == 1:
        return f"{tag:08X}"
    return _dictionary().keyword_for_tag(tag) or f"{tag:08X}"


def _vr_vm(tag: int) -> tuple[str | None, str | None]:
    dd = _dictionary()
    try:
        return dd.dictionary_VR(tag), dd.dictionary_VM(tag)
    except KeyError:                          # private or unknown: a plain string
        return None, None


def _is_multi(vm: str | None) -> bool:
    """Whether the spec encodes this attribute as an array: any VM that can exceed 1 (§4.6)."""
    return vm is not None and vm != "1"


def _number(text: str, integer: bool):
    v = float(text)
    return int(v) if integer and v == int(v) else v


def encode(tag: int, text: str):
    """One SimpleITK value (a string) in the spec's JSON-native form (§4.2, §4.6)."""
    vr, vm = _vr_vm(tag)
    if vr is not None and "or" in vr:         # "US or SS": both integers
        vr = vr.split(" or ")[0]
    parts = [p.strip() for p in str(text).split("\\")]
    multi = _is_multi(vm)
    if vr in _NUMERIC_INT or vr in _NUMERIC_FLOAT:
        try:
            values = [_number(p, vr in _NUMERIC_INT) for p in parts if p != ""]
        except ValueError:                    # a malformed number: keep the text, never guess
            values = [p for p in parts]
            return values if multi or len(values) != 1 else values[0]
        if multi:
            return values
        return values[0] if values else None
    stripped = [p.rstrip("\x00 ").lstrip() for p in parts]
    if multi:
        return stripped
    return str(text).rstrip("\x00 ").lstrip()


def tags_from_sitk(per_slice: list[dict]) -> tuple[dict, list[dict]]:
    """``(series_tags, per_slice_tags)`` from SimpleITK's per-slice dictionaries, in the order
    given. A tag whose value is the same on every slice is series-level; one that differs, or
    that some slices lack, is per slice (each slice's dict holds the ones it has). Keys that are
    not DICOM tags (SimpleITK's own ``ITK_...``), excluded tags and group lengths are dropped."""
    if not per_slice:
        return {}, []
    keys = set()
    for d in per_slice:
        keys.update(d)
    series, varying = {}, []
    for key in sorted(keys):
        tag = _tag(key)
        if tag is None or tag in EXCLUDED or (tag & 0xFFFF) == 0:
            continue
        values = [d.get(key) for d in per_slice]
        if all(v == values[0] for v in values) and values[0] is not None:
            value = encode(tag, values[0])
            if value is not None:             # an empty number: absent, never null (§4.3)
                series[keyword_of(tag)] = value
        else:
            varying.append((tag, key))
    slices = []
    for d in per_slice:
        one = {}
        for tag, key in varying:
            if key in d and (value := encode(tag, d[key])) is not None:
                one[keyword_of(tag)] = value
        slices.append(one)
    return series, slices


def _tag_of_keyword(keyword: str) -> int | None:
    if len(keyword) == 8:
        try:
            return int(keyword, 16)
        except ValueError:
            pass
    return _dictionary().tag_for_keyword(keyword)


def _text(value) -> str:
    def one(v):
        if isinstance(v, float) and v == int(v):
            return str(int(v)) if abs(v) < 1e15 else repr(v)
        return str(v)
    if isinstance(value, list):
        return "\\".join(one(v) for v in value)
    return one(value)


def to_sitk_strings(tags: dict) -> dict:
    """The series-level tags as SimpleITK shows them: ``gggg|eeee`` -> string."""
    out = {}
    for keyword, value in tags.items():
        tag = _tag_of_keyword(keyword)
        if tag is None or value is None:
            continue
        out[f"{tag >> 16:04x}|{tag & 0xFFFF:04x}"] = _text(value)
    return out
