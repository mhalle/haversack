"""A folder holding several DICOM series is refused - never read as one of them.

Found 2026-09-11. `io.read_image` read a directory through GDCM's
`GetGDCMSeriesFileNames(dir)`, which answers with the first series it finds and says
nothing of the rest: a folder holding two series, 3 and 5 slices under different
SeriesInstanceUIDs, read as the 3-slice one with no message, so `segment ./study/`
segmented one series of several. `io.convert`, which `get -o scan.nii.gz` goes through,
made the same pick with `ids[0]`. The server has refused such a folder at upload since
`dicom_series_ids` was written; these pin the refusal at both readers, and on the fetch
that can deliver such a folder - an archive's `!<folder>/`, flattened into one directory.

The slices are SimpleITK's own (ImageFileWriter with KeepOriginalImageUIDOn, which is how
the defect was reproduced), so only the RTSTRUCT case needs pydicom.
"""
import io as bytes_io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from haversack import io, sources
from haversack.errors import InputError

# distinctive UIDs, so an assertion that one is named cannot pass on a constant
THREE = "1.2.826.0.1.3680043.2.1125.7.3"
FIVE = "1.2.826.0.1.3680043.2.1125.7.5"
STUDY = "1.2.826.0.1.3680043.2.1125.7"


def write_series(folder, n, uid, *, value, z0=0.0, stem="IM"):
    """``n`` 4x4 CT slices of one series, 1 mm apart from ``z0``, every voxel ``value`` so
    that a read says which series it got. Two series written with one ``stem`` share file
    names, as the folders of a PACS export do."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    vol = sitk.GetImageFromArray(np.full((n, 4, 4), value, np.int16))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for k in range(n):
        sl = vol[:, :, k]
        for tag, v in {"0020|000e": uid, "0020|000d": STUDY, "0008|0060": "CT",
                       "0020|0013": str(k + 1), "0020|0032": f"0\\0\\{z0 + k}",
                       "0020|0037": "1\\0\\0\\0\\1\\0", "0028|0030": "0.5\\0.5"}.items():
            sl.SetMetaData(tag, v)
        writer.SetFileName(str(folder / f"{stem}{k:03d}.dcm"))
        writer.Execute(sl)
    return folder


def voxels(image) -> set:
    return set(np.unique(sitk.GetArrayFromImage(image)).tolist())


class SeveralSeriesInOneFolder(unittest.TestCase):
    """The reproduction: 3 and 5 slices under two SeriesInstanceUIDs, in one folder."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        self.study = self.root / "study"
        write_series(self.study, 3, THREE, value=3, stem="a")
        write_series(self.study, 5, FIVE, value=5, z0=10.0, stem="b")

    def test_read_image_refuses_naming_both_series_and_the_fix(self):
        with self.assertRaises(InputError) as cm:
            io.read_image(self.study)
        msg = str(cm.exception)
        self.assertIn("2 DICOM series", msg)
        self.assertIn(THREE, msg)
        self.assertIn(FIVE, msg)
        self.assertIn("give the folder of one", msg)
        self.assertEqual(len(msg.splitlines()), 1, msg)

    def test_the_pipelines_read_refuses_too(self):
        """`pipeline.segment` reads through `io.read`; the engines through `read_image`."""
        with self.assertRaises(InputError):
            io.read(self.study, reorient=True)

    def test_convert_refuses_before_it_writes_anything(self):
        """`get -o scan.nii.gz` converts through here, and it wrote the 3-slice series."""
        out = self.root / "scan.nii.gz"
        with self.assertRaises(InputError) as cm:
            io.convert(self.study, out)
        self.assertIn(THREE, str(cm.exception))
        self.assertIn(FIVE, str(cm.exception))
        self.assertFalse(out.exists())

    def test_each_series_in_a_folder_of_its_own_reads_as_itself(self):
        """The fix the message names works, and a folder of one series reads as it did -
        through both readers."""
        for n, uid, value in ((3, THREE, 3), (5, FIVE, 5)):
            folder = write_series(self.root / f"only-{n}", n, uid, value=value)
            img = io.read_image(folder)
            self.assertEqual(img.GetSize(), (4, 4, n))
            self.assertEqual(voxels(img), {value})
            out = io.convert(folder, self.root / f"only-{n}.nrrd")
            self.assertEqual(sitk.ReadImage(str(out)).GetSize(), (4, 4, n))

    def test_many_series_are_counted_and_named_up_to_three(self):
        for i in range(2):
            write_series(self.study, 2, f"{STUDY}.{9 + i}", value=9, z0=20.0 + 5 * i,
                         stem=f"c{i}")
        with self.assertRaises(InputError) as cm:
            io.read_image(self.study)
        msg = str(cm.exception)
        self.assertIn("4 DICOM series", msg)
        self.assertIn("and 1 more", msg)
        named = [u for u in (THREE, FIVE, f"{STUDY}.9", f"{STUDY}.10") if u in msg]
        self.assertEqual(len(named), 3, msg)         # a study of forty stays one line
        self.assertEqual(len(msg.splitlines()), 1, msg)


class ASeriesBesideAnObjectWithoutPixels(unittest.TestCase):
    """GDCM does not count an object without pixel data as a series, so a CT exported
    beside its RTSTRUCT still reads. Pinned because the refusal leans on it: were a
    SimpleITK upgrade to count one, every such folder would start being refused."""

    def test_a_ct_beside_its_rtstruct_still_reads(self):
        try:
            from pydicom.dataset import FileDataset, FileMetaDataset
            from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        except ImportError:
            self.skipTest("pydicom writes the RTSTRUCT")
        with tempfile.TemporaryDirectory() as td:
            folder = write_series(Path(td), 4, THREE, value=3)
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"   # RT Structure Set
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Modality = "RTSTRUCT"
            ds.SeriesInstanceUID = FIVE
            ds.StudyInstanceUID = STUDY
            ds.save_as(folder / "rtstruct.dcm", enforce_file_format=True)
            img = io.read_image(folder)
            self.assertEqual(img.GetSize(), (4, 4, 4))


class _Archive(sources.ArchiveReadingSource):
    """A zip held in memory, read through the real `ArchiveReadingSource.fetch` - member
    selection and the one flattening rule - with no network."""

    prefix = "tst"
    id_pattern = r"study\.zip![a-z/]+"
    description = "a test archive"

    def __init__(self, data):
        self._data = data

    def _zip(self, outer, credentials=None):
        return zipfile.ZipFile(bytes_io.BytesIO(self._data))


class AFetchedFolderOfSeveralSeries(unittest.TestCase):
    """What a remote source can deliver: `!<folder>/` extracts every member under a prefix
    and flattens it into one directory, so a study's series folders arrive side by side -
    here under the same file names, which the flattening indexes. The refusal is the same
    one, and the fix it names - end the source at one series' folder - reads that series."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        write_series(root / "study" / "three", 3, THREE, value=3)
        write_series(root / "study" / "five", 5, FIVE, value=5, z0=10.0)
        buf = bytes_io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in sorted((root / "study").rglob("*.dcm")):
                z.write(f, f.relative_to(root).as_posix())
        self.src = _Archive(buf.getvalue())
        self.cache = root / "inputs"

    def fetch(self, ident):
        return sources.materialize(f"tst:{ident}", cache_dir=self.cache, sources=[self.src])

    def test_the_study_folder_is_refused_and_one_series_folder_reads(self):
        got = self.fetch("study.zip!study/")
        self.assertEqual(len(list(got.iterdir())), 8)       # every slice of both arrived
        with self.assertRaises(InputError) as cm:
            io.read_image(got)
        self.assertIn(THREE, str(cm.exception))
        self.assertIn(FIVE, str(cm.exception))
        img = io.read_image(self.fetch("study.zip!study/five/"))
        self.assertEqual(img.GetSize(), (4, 4, 5))
        self.assertEqual(voxels(img), {5})

    def test_the_fetch_record_names_every_series_and_never_a_pick(self):
        """Provenance never leaned on the reader's pick: it asks GDCM for every series id
        itself, so the record beside such a fetch lists both, and a one-series fetch its one."""
        both = sources.read_input_record(self.fetch("study.zip!study/").parent)
        self.assertEqual(sorted(both["content"]["dicom"]["series_instance_uid"]),
                         sorted([THREE, FIVE]))
        one = sources.read_input_record(self.fetch("study.zip!study/three/").parent)
        self.assertEqual(one["content"]["dicom"]["series_instance_uid"], THREE)
