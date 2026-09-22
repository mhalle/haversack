"""describe() of a task whose model is not installed answers its structures from the segments
index (2026-09-22).

The index (``data/segments.json``) was mined from each checkpoint at the version its catalog
pins, and since 2026-09-13 ``haversack tasks TASK`` and ``GET /v1/segments`` read it - while
``GET /v1/tasks/{task}`` (``Segmenter.describe``) still said "structures are read from the
checkpoint once installed". A user browsing a server saw ``moose:clin_ct_muscles`` with no
structure list that its own segment search could list.

What must hold, beyond the list appearing: describe is also the door every result key's weights
versions pass (``serve.weights_versions_of``), so the key is the same with the index, without
it, and with a broken one; a stale record, or a caller's version pin, gives no list rather than
another version's; and the CLI and the server give one answer.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from haversack import segments
from haversack.segmenter import Segmenter
from haversack.weights import WeightsStore

TASK = "moose:clin_ct_muscles"          # the task the user reported, never installed in a test


def _packaged_record(task=TASK) -> dict:
    return json.loads(segments.PACKAGED.read_text(encoding="utf-8"))["tasks"][task]


class _EmptyRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "weights"
        self.root.mkdir()
        self.user = self.tmp / "user-segments.json"   # absent unless a test writes it
        env = mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": str(self.user),
                                           "HAVERSACK_CACHE_DIR": str(self.tmp / "cache")})
        env.start()
        self.addCleanup(env.stop)
        self.seg = Segmenter(device="cpu", weights=WeightsStore(self.root, fetch=False))

    def write_user_index(self, tasks: dict):
        self.user.write_text(json.dumps({"_meta": {"schema_version": segments.SCHEMA_VERSION},
                                         "tasks": tasks}), encoding="utf-8")


class DescribeReadsTheIndex(_EmptyRoot):
    def test_an_uninstalled_task_lists_what_its_checkpoint_states(self):
        d = self.seg.describe(TASK)
        self.assertFalse(d["materialized"])
        rec = _packaged_record()
        self.assertEqual(d["structures"], [s["id"] for s in rec["segments"]])
        self.assertEqual(d["n_structures"], len(rec["segments"]))
        self.assertEqual(d["segments"], rec["segments"])
        self.assertEqual(d["structures_from"]["source"], "segments index")
        self.assertEqual(d["structures_from"]["version"], rec["version"])
        self.assertIn("segments index", d["hint"])

    def test_every_catalog_task_the_index_holds_is_described_with_its_list(self):
        """Not just the one reported: every uninstalled nnU-Net-catalog task on an empty root."""
        held = json.loads(segments.PACKAGED.read_text(encoding="utf-8"))["tasks"]
        checked = 0
        for task in self.seg.tasks():
            info = self.seg.catalog.info(task)
            rec = held.get(task)
            if info.get("materialized", True) or info.get("structures") or not rec \
                    or rec.get("kind") != "segments":
                continue
            checked += 1
            self.assertEqual(self.seg.describe(task).get("structures"),
                             [s["id"] for s in rec["segments"]], task)
        self.assertGreater(checked, 10)

    def test_a_record_note_travels_with_the_list(self):
        rec = {**_packaged_record(), "note": "values are channels, one layer each"}
        self.write_user_index({TASK: rec})
        self.assertEqual(self.seg.describe(TASK)["structures_from"]["note"],
                         "values are channels, one layer each")


class TheIndexNeverSaysWhatIsNotCurrent(_EmptyRoot):
    def test_a_stale_record_gives_no_list(self):
        rec = _packaged_record()
        stale = {**rec, "version": {**rec["version"], "tag": "01011999"},
                 "segments": [{"id": "not_what_the_model_says", "value": 1}]}
        self.write_user_index({TASK: stale})
        d = self.seg.describe(TASK)
        self.assertNotIn("structures", d)
        self.assertNotIn("segments", d)
        self.assertIn("read from the checkpoint once installed", d["hint"])

    def test_a_version_pin_gives_no_list(self):
        """The record is of the version the catalog pins, not of the one a caller names."""
        self.assertIsNone(segments.before_install(self.seg.catalog, TASK + "@21052025"))
        self.assertIsNotNone(segments.before_install(self.seg.catalog, TASK))

    def test_an_open_record_gives_no_list(self):
        """``open`` is "no fixed label set" (VoxTell): whatever it lists is no structure list.
        The record keeps its segments here, so the kind alone must refuse it."""
        rec = {**_packaged_record(), "kind": "open"}
        self.write_user_index({TASK: rec})
        self.assertNotIn("structures", self.seg.describe(TASK))

    def test_a_broken_user_index_costs_the_list_and_nothing_else(self):
        self.user.write_text("{ not json", encoding="utf-8")
        d = self.seg.describe(TASK)
        self.assertNotIn("structures", d)
        self.assertFalse(d["materialized"])


class TheKeyDoesNotMove(_EmptyRoot):
    """weights_versions_of reads describe; the index must change nothing it reads."""

    def key_parts(self, task):
        from haversack.serve import installed_versions, weights_versions_of
        return weights_versions_of(self.seg, task), installed_versions(self.seg, task)

    def test_the_key_is_the_same_with_without_and_with_a_broken_index(self):
        tasks = [TASK, "cads:organs", "ts.v2:total", "ts.v3:total"]
        with_index = {t: self.key_parts(t) for t in tasks}
        self.assertIn("structures", self.seg.describe(TASK))     # the index WAS read
        with mock.patch.object(segments, "before_install", return_value=None):
            without = {t: self.key_parts(t) for t in tasks}
        self.user.write_text("{ not json", encoding="utf-8")
        broken = {t: self.key_parts(t) for t in tasks}
        self.assertEqual(with_index, without)
        self.assertEqual(with_index, broken)

    def test_describe_differs_only_by_the_index_fields(self):
        d = self.seg.describe(TASK)
        with mock.patch.object(segments, "before_install", return_value=None):
            before = self.seg.describe(TASK)
        added = {"structures", "n_structures", "segments", "structures_from", "hint"}
        self.assertEqual({k: v for k, v in d.items() if k not in added},
                         {k: v for k, v in before.items() if k not in added})
        self.assertEqual(set(d) - set(before), added - {"hint"})

    def test_before_install_never_raises(self):
        class Exploding:
            def resolve(self, name):
                raise RuntimeError("catalog down")
        self.assertIsNone(segments.before_install(Exploding(), TASK))
        with mock.patch.object(segments, "current_version", side_effect=RuntimeError("x")):
            self.assertIsNone(segments.before_install(self.seg.catalog, TASK))


class OneAnswerAtEveryDoor(_EmptyRoot):
    def test_the_route_and_the_cli_agree(self):
        import pytest
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from haversack.serve import LocalExecutor, create_app
        client = TestClient(create_app(LocalExecutor(self.seg, workdir=self.tmp / "work")))
        r = client.get(f"/v1/tasks/{TASK}")
        self.assertEqual(r.status_code, 200, r.text)
        served = r.json()["structures"]

        import contextlib
        import io
        from haversack.cli import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["tasks", TASK, "--json", "--model-root", str(self.root)])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertEqual(json.loads(out.getvalue())["structures"], served)

    def test_the_cli_passes_over_a_stale_record_as_describe_does(self):
        rec = _packaged_record()
        self.write_user_index({TASK: {**rec, "version": {**rec["version"], "tag": "01011999"}}})
        import contextlib
        import io
        from haversack.cli import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["tasks", TASK, "--model-root", str(self.root)])
        self.assertNotEqual(rc, 0)
        self.assertIn("catalog check", err.getvalue())
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
