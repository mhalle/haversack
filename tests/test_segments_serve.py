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

    def test_the_remote_client_speaks_it(self):
        rc = RemoteClient("http://testserver")
        rc._http = self.client                   # starlette's TestClient is an httpx.Client
        res = rc.segments("pancreas", limit=5)
        self.assertIn("ts.v2:total", {s["task"] for g in res["results"] for s in g["segments"]})
        with self.assertRaisesRegex(RemoteError, "422"):
            rc.segments("^x$", mode="regex")
