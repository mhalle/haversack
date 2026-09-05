"""Defects the 0.6.0 release reviews reproduced (2026-09-05), pinned."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from haversack.errors import InputError


class TestStoreGeometryFollowsTheCrop(unittest.TestCase):
    """The array's placement is derived from the grid the resampler RAN ON - the crop-to-nonzero
    sub-grid when there was one - not from the full canonical grid. The review's exact case:
    canonical 100^3 at 1 mm, cropped to 60^3 at 20 mm, model 30^3."""

    def _meta(self, convention):
        return {"frame": {"canonical": {"shape_zyx": [100, 100, 100], "spacing_zyx": [1.0, 1.0, 1.0],
                                        "origin_xyz": [0.0, 0.0, 0.0],
                                        "direction_xyz": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                          "source": {"shape": [100, 100, 100], "spacing": [1.0, 1.0, 1.0], "origin": [0, 0, 0]},
                          "model_source": {"shape": [60, 60, 60], "spacing": [1.0, 1.0, 1.0],
                                           "origin": [20.0, 20.0, 20.0]},
                          "convention": convention},
                "model_grid": [30, 30, 30], "envelope": {"start": [0, 0, 0]}, "convention": convention}

    def test_center_rule(self):
        from haversack.ranked_output import model_grid_geometry
        eff, origin, _, centering = model_grid_geometry(self._meta("center"))
        np.testing.assert_allclose(eff, [2.0, 2.0, 2.0])
        np.testing.assert_allclose(origin, [20.5, 20.5, 20.5])
        self.assertEqual(centering, "cell")

    def test_corner_rule(self):
        from haversack.ranked_output import model_grid_geometry
        eff, origin, _, centering = model_grid_geometry(self._meta("corner"))
        np.testing.assert_allclose(eff, [59 / 29] * 3)
        np.testing.assert_allclose(origin, [20.0, 20.0, 20.0])
        self.assertEqual(centering, "node")

    def test_the_builder_and_the_emit_share_the_derivation(self):
        import importlib.util
        if importlib.util.find_spec("rankfield") is None:
            self.skipTest("no rankfield")
        from haversack.ranked_build import geometry
        from haversack.ranked_output import model_grid_geometry
        m = self._meta("center")
        self.assertEqual(geometry(m)[:2], model_grid_geometry(m)[:2])


class TestStoreTarget(unittest.TestCase):
    def test_an_occupied_or_unwritable_target_is_refused_up_front(self):
        import importlib.util
        if importlib.util.find_spec("zarr") is None:
            self.skipTest("no zarr")
        import tempfile
        from haversack.ranked_store import check_target
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "taken.duckn.zip").write_text("not a store")
            with self.assertRaisesRegex(InputError, "not a ranked store"):
                check_target(Path(d) / "taken.duckn.zip")
            check_target(Path(d) / "new" / "deeper" / "x.duckn.zip")      # a path to create: fine
            if os.geteuid() != 0:
                ro = Path(d) / "ro"
                ro.mkdir()
                ro.chmod(0o500)
                try:
                    with self.assertRaisesRegex(InputError, "not writable"):
                        check_target(ro / "x.duckn")
                finally:
                    ro.chmod(0o700)


class TestProvenanceRecordsWhatWasBuilt(unittest.TestCase):
    def test_the_layout_step_takes_its_parameters(self):
        import importlib.util
        if importlib.util.find_spec("rankfield") is None:
            self.skipTest("no rankfield")
        from haversack.ranked_build import generator_steps
        steps = generator_steps({"task": "t", "depth": 6, "clip": 8.0}, [("p", {"haversack": "0.6.1"})], "nnunetv2",
                                parts_kept="last", layers=["occupancy"], distance_voxels=0)
        p = steps[-1]["parameters"]
        self.assertEqual((p["parts_kept"], p["derived_layers"], p["distance_voxels"]), ("last", ["occupancy"], 0))

    def test_the_input_becomes_a_source(self):
        from haversack.ranked_output import input_source
        self.assertEqual(input_source("idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c")["identifier"],
                         "idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c")
        s = input_source("/x/y/chest.nii.gz")
        self.assertEqual((s["path"], s["format"]), ("chest.nii.gz", "nii.gz"))


class TestStatisticsSayWhyFieldColumnsAreMissing(unittest.TestCase):
    def test_a_missing_library_is_reported_not_swallowed(self):
        import tempfile
        from haversack import statistics
        class Code:                                   # enough of a ranked code to be asked
            meta = {"labels": [0, 1], "spacing_zyx": [1.0, 1.0, 1.0], "clip": 8.0}
        import haversack
        saved = sys.modules.get("haversack.ranked")
        attr = getattr(haversack, "ranked", None)
        sys.modules["haversack.ranked"] = None        # import raises ImportError ...
        if attr is not None:
            delattr(haversack, "ranked")              # ... once the package attribute is gone too
        try:
            got, why = statistics._field_measurements(Code())
        finally:
            if saved is None:
                del sys.modules["haversack.ranked"]
            else:
                sys.modules["haversack.ranked"] = saved
            if attr is not None:
                haversack.ranked = attr
        self.assertEqual(got, {})
        self.assertIn("ranked", why)
        del tempfile


class TestExports(unittest.TestCase):
    def test_the_remote_client_is_importable_from_the_package(self):
        import importlib.util
        if importlib.util.find_spec("httpx") is None:
            self.skipTest("no httpx")
        import haversack
        from haversack.client import RemoteClient
        self.assertIs(haversack.RemoteClient, RemoteClient)
