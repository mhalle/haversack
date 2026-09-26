"""An embedding field states the digest of its SOURCE's bytes, not of the input copy the cache
keeps instead of them (found mapping the input caches, 2026-09-26): a fetched series reached
the encoder as `decoded/input.duckn.zip`, and the field recorded that file's hash."""
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from haversack.encoders.pipeline import _identity
from haversack.input_copy import transcode


class TheFieldNamesTheSourcesDigest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "series" / "ct.nii.gz"
        self.src.parent.mkdir()
        sitk.WriteImage(sitk.GetImageFromArray(np.full((5, 4, 3), 9, np.int16)), str(self.src))

    def test_a_copy_names_the_digest_it_was_made_from(self):
        source = "sha256:" + "ab" * 32
        copy = transcode(self.src, self.tmp, source="idc:x", source_digest=source)
        self.assertIsNotNone(copy)
        self.assertEqual(_identity("idc:x", copy)["digest"], source)

    def test_an_original_is_hashed_as_before(self):
        from haversack.content import digest_file
        self.assertEqual(_identity("idc:x", self.src)["digest"], digest_file(self.src))

    def test_a_copy_that_records_no_source_digest_is_hashed(self):
        from haversack.content import digest_file
        copy = transcode(self.src, self.tmp, source="idc:x", source_digest=None)
        self.assertEqual(_identity("idc:x", copy)["digest"], digest_file(copy))
