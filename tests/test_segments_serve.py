"""GET /v1/segments: the segments search over the wire - a read anyone may make, words and glob
only, answered from the tasks the deployment serves, and present on the anonymous twin."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient                          # noqa: E402

from haversack.client import RemoteClient, RemoteError             # noqa: E402
from haversack.serve import LocalExecutor, create_app, create_public_app   # noqa: E402

from test_serve import FakeSegmenter                               # noqa: E402

SERVED = ["ts.v2:total", "fastsurfer:asegdkt"]


class ServedSegmenter(FakeSegmenter):
    """Serves qualified names, as a real deployment's catalog does."""

    def tasks(self):
        return list(SERVED)


class TheSegmentsRoute(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # the packaged index only: a user index on the machine running the suite must not
        # change what these tests see
        env = mock.patch.dict(os.environ, {"HAVERSACK_SEGMENTS": str(self.tmp / "none.json")})
        env.start()
        self.addCleanup(env.stop)
        self.ex = LocalExecutor(ServedSegmenter(), workdir=self.tmp)
        self.client = TestClient(create_app(self.ex))

    def get(self, **params):
        return self.client.get("/v1/segments", params=params)

    @staticmethod
    def tasks(r) -> set:
        return {s["task"] for g in r.json()["results"] for s in g["segments"]}

    def test_a_word_search_answers_from_the_served_tasks_only(self):
        r = self.get(q="kidney left")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("ts.v2:total", self.tasks(r))
        self.assertLessEqual(self.tasks(r), set(SERVED))

    def test_a_glob_is_offered(self):
        r = self.get(q="vertebrae_[ct]*", mode="glob")
        self.assertEqual(r.status_code, 200, r.text)
        keys = [g["key"] for g in r.json()["results"]]
        self.assertTrue(keys)
        self.assertTrue(all(k.startswith(("vertebrae_c", "vertebrae_t")) for k in keys), keys)

    def test_regex_is_refused_over_the_wire(self):
        r = self.get(q="^liver$", mode="regex")
        self.assertEqual(r.status_code, 422)
        self.assertIn("regex", r.json()["detail"])

    def test_bad_requests_are_422(self):
        for params in ({"q": ""}, {"q": "liver", "limit": 0}, {"q": "liver", "limit": 5000},
                       {"q": "liver", "catalog": "nosuch"}, {"q": "liver", "mode": "fuzzy"}, {}):
            with self.subTest(params=params):
                self.assertEqual(self.get(**params).status_code, 422)

    def test_no_token_is_needed(self):
        client = TestClient(create_app(self.ex, token="s3cret"))
        r = client.get("/v1/segments", params={"q": "pancreas"})
        self.assertEqual(r.status_code, 200, r.text)

    def test_the_anonymous_twin_serves_it_from_its_own_tasks(self):
        app = create_public_app(lambda *a: None, lambda *a: None, lambda: ["ts.v2:total"])
        r = TestClient(app).get("/v1/segments", params={"q": "pancreas"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.tasks(r), {"ts.v2:total"})

    def test_a_server_that_cannot_list_its_tasks_says_so(self):
        """An empty 200 read as "nothing produces this"; it is a fault, and says so."""
        with mock.patch.object(ServedSegmenter, "tasks", side_effect=RuntimeError("boom")):
            r = self.get(q="liver")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("boom", r.text)

    def test_an_unserved_catalog_is_not_named_back(self):
        r = self.get(q="left", catalog="moose")
        self.assertEqual(r.status_code, 422)
        self.assertIn("has no tasks here", r.json()["detail"])
        self.assertNotIn("cads", r.json()["detail"])

    def test_the_openapi_says_what_the_parameters_and_errors_are(self):
        op = self.client.app.openapi()["paths"]["/v1/segments"]["get"]
        params = {p["name"]: p for p in op["parameters"]}
        self.assertIn("glob", params["mode"]["description"])
        self.assertIn("own spelling", params["field"]["description"])
        self.assertIn("503", op["responses"])

    def test_pages_and_counts_go_over_the_wire(self):
        first = self.get(q="vertebrae", limit=3).json()
        self.assertTrue(first["truncated"])
        second = self.get(q="vertebrae", limit=3, offset=first["next_offset"]).json()
        self.assertEqual(second["offset"], 3)
        self.assertFalse({g["key"] for g in first["results"]} & {g["key"] for g in second["results"]})
        self.assertEqual(list(second)[-1], "end")
        counted = self.get(q="vertebrae", count_only="true").json()
        self.assertNotIn("results", counted)
        self.assertEqual(counted["key_count"], first["key_count"])
        self.assertEqual(self.get(q="vertebrae", offset=-1).status_code, 422)

    def test_the_task_listing_leaves_structures_and_credit_to_describe(self):
        """GET /v1/tasks is one line per task; a task's structures, label map and attribution
        are GET /v1/tasks/{task}'s. The attribution alone was most of the listing's weight."""
        from haversack.ecosystems import EcosystemCatalog, TSEcosystem

        catalog = EcosystemCatalog([TSEcosystem()], root=self.tmp)

        class Cataloged(FakeSegmenter):
            def tasks(self):
                return catalog.names()
        seg = Cataloged()
        seg.catalog = catalog
        client = TestClient(create_app(LocalExecutor(seg, workdir=self.tmp / "cat")))
        detail = client.get("/v1/tasks").json()["detail"]
        self.assertIn("ts.v2:total", detail)
        self.assertFalse([t for t, d in detail.items()
                          if {"structures", "label_map", "attribution"} & set(d)])
        self.assertIn("attribution", catalog.info("ts.v2:total"))     # still there per task

    def test_the_remote_client_speaks_it(self):
        rc = RemoteClient("http://testserver")
        rc._http = self.client                   # starlette's TestClient is an httpx.Client
        res = rc.segments("pancreas", limit=5)
        self.assertIn("ts.v2:total", {s["task"] for g in res["results"] for s in g["segments"]})
        page = rc.segments("vertebrae", limit=2, offset=2)
        self.assertEqual(page["offset"], 2)
        self.assertNotIn("results", rc.segments("vertebrae", count_only=True))
        with self.assertRaisesRegex(RemoteError, "422"):
            rc.segments("^x$", mode="regex")
