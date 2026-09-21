"""A label map as an INPUT: the labels together with what they mean.

Until 2026-09-20 every input role was an image channel, read by :func:`haversack.io.read_image`
into voxels and geometry and nothing else. A task that consumes (image, mask) - statistics or
radiomics under a segmentation, a cascade gated on an organ another model found, pooling a
token field under a structure - needs three more things from its mask, and this is the reader
that hands them over:

* **the segment names.** A consumer gates by structure NAME, never by label value: ``liver``
  is 5 in one catalog, 1 in another and absent from a third, and a consumer that hardcodes a
  value is silently wrong for every task but the one it was written against. The names travel
  in the ``.seg.nrrd`` header, where :meth:`haversack.result.Segmentation.save` puts them, in
  3D Slicer's keys - so a segmentation exported from Slicer reads the same way.
* **the geometry**, so the mask can be checked against, or resampled onto, the image's grid.
* **the task that made it**, so a consumer can choose its convention (which structure feeds
  which of its own queries differs per catalog).

Its own module rather than a function in :mod:`haversack.io`: that module is the IMAGE
reader, its callers all expect a bare SimpleITK image back, and the default path ``segment
IN -o labels.nii.gz`` has no use for any of this. Nothing here imports torch, pydantic,
duckn, zarr or rankfield; SimpleITK and numpy load at call time.

A NIfTI label map reads too, and has no names: NIfTI has nowhere to put them, which is why
it is this project's lossy format. ``read_label_map`` refuses one by default and says what
to send instead, because a consumer that asked for names and got none would have to guess.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputError
from .result import PROVENANCE_KEY, SEGMENT_KEY

#: The ``kind`` of an input role that takes a label map, beside ``image`` - what a task's
#: ``inputs[].kind`` says, and what a result's output is checked against before a
#: ``result:`` reference is bound to a role. Written here, once, because both the wire
#: (:mod:`haversack.schemas`) and the source that resolves references
#: (:class:`haversack.sources.ResultSource`) need it and neither may import the other's
#: dependencies.
LABELS_KIND = "labels"

_SEGMENT_FIELD = re.compile(
    "^" + SEGMENT_KEY.format(index=r"(\d+)", field=r"([A-Za-z]+)") + "$")


@dataclass(frozen=True)
class LabelMap:
    """A label volume and its meaning.

    ``labels`` is the SimpleITK image in its STORED orientation (nothing is reoriented or
    resampled: what a consumer does about the grid is its decision, as it is for an image).
    ``names`` maps a label value to its structure name for every segment the file declares -
    for a haversack result, the structures present in that volume. ``task`` is the catalog
    task that produced it, or None when nothing says.
    """

    labels: object
    names: dict
    task: str | None = None
    provenance: dict = field(default_factory=dict)
    path: str = ""

    @property
    def array(self):
        """The labels as numpy (Z, Y, X). A fresh array each call."""
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(self.labels)

    @property
    def geometry(self):
        from . import io as nio
        return nio.geometry_of(self.labels)

    def label_of(self, name: str) -> int:
        """The label value of a structure, by name."""
        for value, n in self.names.items():
            if n == name:
                return int(value)
        raise KeyError(f"no structure named {name!r} in {self.path or 'this label map'}"
                       f"{f' ({self.task})' if self.task else ''}; it has "
                       f"{', '.join(sorted(self.names.values())[:8])}"
                       f"{'...' if len(self.names) > 8 else ''}")

    def mask(self, name: str):
        """A boolean (Z, Y, X) mask of one structure, BY NAME. Deliberately takes no label
        value: that is the mistake this type exists to make hard."""
        return self.array == self.label_of(name)


def _the_file(path) -> Path:
    """A staged input arrives as a directory holding one file (every fetch produces
    ``<entry>/series/``); read that file. Several is not a label map."""
    p = Path(path)
    if not p.is_dir():
        if not p.exists():
            raise InputError(f"label map not found: {p}")
        return p
    files = [q for q in sorted(p.iterdir()) if q.is_file() and not q.name.startswith(".")]
    if len(files) != 1:
        raise InputError(f"{p} holds {len(files)} files, and a label map is one "
                         "(.seg.nrrd); give the file")
    return files[0]


def _staged_task(file: Path) -> str | None:
    """The task a STAGED label map was made by, from the record its fetch left beside it.

    A ``result:`` reference is fetched into ``<entry>/series/`` and recorded in
    ``<entry>/.input.json`` (``sources.fetch_recording_origin``), whose origin names the
    upstream task from the result cache's own ``meta.json`` - the name the server KEYED
    the result on, present for every engine. The file's header says it only when the
    engine wrote it (the nnU-Net pipeline does; FastSurfer, SynthStrip, VoxTell and MONAI
    bundles do not), so the record is asked first.
    """
    from .content import digest_file
    from .sources import ResultSource, read_input_record
    if file.parent.name != "series":       # not a staged input at all: ask nothing
        return None
    rec = read_input_record(file.parent.parent)
    if not rec or rec.get("kind") != ResultSource.prefix:
        return None
    # ...and the record has to be about THESE bytes. A `.input.json` two levels above a
    # user's own file is somebody else's record, and believing it names a task that never
    # made this label map - the false claim `sources._dicom_facts` was rewritten to stop
    # making about DICOM series (review, 2026-09-20). Only asked of a file that already
    # looks staged, so a user's own label map is never hashed for this.
    if (rec.get("content") or {}).get("digest") != digest_file(file):
        return None
    return (rec.get("origin") or {}).get("task") or None


def read_label_map(path, *, require_names: bool = True) -> LabelMap:
    """Read a label map - a ``.seg.nrrd`` file, or the directory a fetch staged one in -
    with its segment names, geometry and upstream task.

    ``require_names=False`` lets a nameless label map through (a NIfTI, a bare NRRD) with
    ``names == {}``, for a consumer that really does mean label values.

    Refused whatever is asked, because each would make a lookup BY NAME a silent choice:
    two segments on one label value, two label values under one name, overlapping
    (layered or vector) segments, and voxels that are not integers. ``task`` comes from the
    record a ``result:`` fetch left beside the bytes when there is one about THESE bytes,
    else from the file's own header, else it is None.
    """
    import SimpleITK as sitk
    file = _the_file(path)
    try:
        image = sitk.ReadImage(str(file))
    except RuntimeError as e:
        from .io import _sitk_reason
        raise InputError(f"cannot read {file} as a label map: {_sitk_reason(e)}") from None
    integer = {sitk.sitkUInt8, sitk.sitkInt8, sitk.sitkUInt16, sitk.sitkInt16,
               sitk.sitkUInt32, sitk.sitkInt32, sitk.sitkUInt64, sitk.sitkInt64}
    if image.GetNumberOfComponentsPerPixel() == 1 and image.GetPixelID() not in integer:
        raise InputError(f"{file} holds {image.GetPixelIDTypeAsString()} voxels, and a label "
                         "map is integers: this looks like an image, not a segmentation")
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        # Slicer writes overlapping segments as a 4-D (layered) seg.nrrd; haversack writes
        # one layer. Reading layer 0 and dropping the rest would be choosing silently.
        raise InputError(f"{file} is not a single-layer 3D label map "
                         f"({image.GetDimension()}D, {image.GetNumberOfComponentsPerPixel()} "
                         "component(s)); overlapping (layered) segmentations are not read yet")
    fields: dict = {}
    for key in image.GetMetaDataKeys():
        m = _SEGMENT_FIELD.match(key)
        if m:
            fields.setdefault(int(m.group(1)), {})[m.group(2)] = image.GetMetaData(key)
    names, unvalued = {}, []
    for index in sorted(fields):
        seg = fields[index]
        if str(seg.get("Layer", "0")).strip() not in ("", "0"):
            raise InputError(f"{file}: segment {seg.get('Name')!r} is on layer "
                             f"{seg.get('Layer')}; layered segmentations are not read yet")
        try:
            value = int(str(seg.get("LabelValue", "")).strip())
        except ValueError:
            unvalued.append(seg.get("Name"))   # a segment with no label value names no voxels
            continue
        if not seg.get("Name"):
            continue
        # Two segments on one value, or two values under one name, and a lookup BY NAME
        # would have to pick: `names` kept the last of the first kind without a word, and
        # `mask("kidney")` covered one kidney of two (review, 2026-09-20). Refused, like a
        # layered file: choosing silently is how a plausible, wrong number gets reported.
        if value in names:
            raise InputError(f"{file}: segments {names[value]!r} and {seg['Name']!r} share label "
                             f"value {value}; a label map names each value once")
        if seg["Name"] in names.values():
            raise InputError(f"{file}: two segments are named {seg['Name']!r}; structures are "
                             "selected by name, so each name has to be one segment")
        names[value] = seg["Name"]
    prov = {}
    if image.HasMetaDataKey(PROVENANCE_KEY):
        try:
            prov = json.loads(image.GetMetaData(PROVENANCE_KEY))
        except ValueError:
            prov = {}
    prov = prov if isinstance(prov, dict) else {}
    if require_names and not names and unvalued:
        raise InputError(f"{file} names {len(unvalued)} segment(s) and gives none a "
                         "LabelValue, so no name can be matched to voxels")
    if require_names and not names:
        raise InputError(f"{file} names no segments, and a label map is read by structure "
                         "name: give a .seg.nrrd (haversack's results are), or a result: "
                         "reference")
    return LabelMap(labels=image, names=names,
                    task=_staged_task(file) or prov.get("task") or None,
                    provenance=prov, path=str(file))
