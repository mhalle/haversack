"""One door for every HTTP request, and the two properties that live behind it.

Both are checked against a real server on loopback rather than a mock, because
both are about what goes on the wire: the agent the peer sees, and whether a
credential follows a redirect off the origin it was meant for.
"""
import ast
import http.server
import pathlib
import threading
import unittest
import urllib.error

from haversack import __version__, fetchlib

SRC = pathlib.Path(fetchlib.__file__).parent
TOOLS = SRC.parents[1] / "tools"


class _Server:
    """A loopback HTTP server that records what it receives and answers as told."""

    def __init__(self):
        seen = self.seen = []
        rules = self.rules = {}

        class H(http.server.BaseHTTPRequestHandler):
            def _serve(self):
                seen.append((self.command, self.path, dict(self.headers)))
                status, headers, body = rules.get(self.path, (200, {}, b"ok"))
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_GET = do_HEAD = _serve

            def log_message(self, *a):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class OnTheWire(unittest.TestCase):

    def setUp(self):
        self.a, self.b = _Server(), _Server()       # two ports: two origins

    def tearDown(self):
        self.a.close()
        self.b.close()

    def test_every_request_names_the_client(self):
        """Python's default agent is refused by at least one weights host."""
        with fetchlib.open(self.a.url + "/x", timeout=5) as r:
            self.assertEqual(r.read(), b"ok")
        _, _, headers = self.a.seen[0]
        self.assertEqual(headers.get("User-Agent"), f"haversack/{__version__}")
        self.assertNotIn("Python-urllib", headers.get("User-Agent", ""))

    def test_a_caller_header_wins_over_the_default(self):
        with fetchlib.open(self.a.url + "/x", timeout=5, headers={"User-Agent": "other"}):
            pass
        self.assertEqual(self.a.seen[0][2].get("User-Agent"), "other")

    def test_a_token_does_not_follow_a_redirect_to_another_origin(self):
        """Zenodo and Hugging Face redirect downloads to a CDN; the bearer token
        for the first host must not be replayed to the second."""
        self.a.rules["/go"] = (302, {"Location": self.b.url + "/cdn"}, b"")
        with fetchlib.open(self.a.url + "/go", timeout=5,
                           headers={"Authorization": "Bearer T"}) as r:
            self.assertEqual(r.read(), b"ok")
        self.assertEqual(self.a.seen[0][2].get("Authorization"), "Bearer T")
        self.assertEqual(self.b.seen[0][1], "/cdn")
        self.assertNotIn("Authorization", self.b.seen[0][2])

    def test_a_token_does_follow_a_redirect_on_the_same_origin(self):
        self.a.rules["/go"] = (302, {"Location": self.a.url + "/there"}, b"")
        with fetchlib.open(self.a.url + "/go", timeout=5,
                           headers={"Authorization": "Bearer T"}):
            pass
        self.assertEqual(self.a.seen[1][1], "/there")
        self.assertEqual(self.a.seen[1][2].get("Authorization"), "Bearer T")

    def test_head_status_returns_rather_than_raises(self):
        self.a.rules["/gone"] = (404, {}, b"")
        self.assertEqual(fetchlib.head_status(self.a.url + "/gone", timeout=5), 404)
        self.assertEqual(fetchlib.head_status(self.a.url + "/x", timeout=5), 200)
        self.assertEqual(self.a.seen[0][0], "HEAD")
        with self.assertRaises(urllib.error.URLError):     # below HTTP: still raised
            fetchlib.head_status("http://127.0.0.1:1/x", timeout=2)

    def test_get_json_and_content_length(self):
        self.a.rules["/j"] = (200, {"Content-Type": "application/json"}, b'{"a": 1}')
        self.assertEqual(fetchlib.get_json(self.a.url + "/j", timeout=5), {"a": 1})
        with fetchlib.open(self.a.url + "/j", timeout=5) as r:
            self.assertEqual(fetchlib.content_length(r), 8)
        self.assertEqual(fetchlib.content_length(None), 0)


class CopyCap(unittest.TestCase):

    def test_a_body_over_the_cap_is_refused_mid_stream(self):
        import io
        from haversack.errors import InputError
        out = io.BytesIO()
        with self.assertRaises(InputError):
            fetchlib.copy_capped(io.BytesIO(b"x" * (3 << 20)), out, 2 << 20, "t")
        self.assertLessEqual(len(out.getvalue()), 2 << 20)
        out = io.BytesIO()
        self.assertEqual(fetchlib.copy_capped(io.BytesIO(b"x" * 10), out, 2 << 20, "t"), 10)


class OneDoor(unittest.TestCase):
    """The structural guard: no module opens a URL on its own. A second opener
    is a second place the agent and the redirect rule have to be remembered -
    which is how the User-Agent fix reached the weights path and missed every
    hosted source."""

    HTTP_NAMES = {"urlopen", "build_opener", "install_opener", "OpenerDirector"}

    def _offenders(self, files, allowed):
        out = []
        for path in files:
            if path.name in allowed:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and node.attr in self.HTTP_NAMES:
                    out.append(f"{path.name}:{node.lineno} {ast.unparse(node)}")
                if (isinstance(node, ast.Attribute) and node.attr == "Request"
                        and ast.unparse(node.value) == "urllib.request"):
                    out.append(f"{path.name}:{node.lineno} {ast.unparse(node)}")
        return out

    def test_the_package_opens_urls_only_through_fetchlib(self):
        files = list(SRC.glob("*.py")) + list((SRC / "engines").glob("*.py"))
        self.assertEqual(self._offenders(files, {"fetchlib.py"}), [])

    def test_the_generators_open_urls_only_through_zippeek(self):
        files = [p for p in TOOLS.glob("*.py")]
        self.assertEqual(self._offenders(files, {"zippeek.py"}), [])

    def test_the_guard_can_see_a_bare_urlopen(self):
        """Pin the guard to the code it exists to reject: the last serve-era
        sources.py, which built its own opener and used it eight times."""
        import subprocess
        old = subprocess.run(["git", "show", "ab41860:src/haversack/sources.py"],
                             capture_output=True, text=True, cwd=SRC.parents[1]).stdout
        if not old:
            self.skipTest("revision ab41860 is not in this clone (a shallow checkout)")
        hits = [n for n in ast.walk(ast.parse(old))
                if isinstance(n, ast.Attribute) and n.attr in self.HTTP_NAMES]
        self.assertTrue(hits, "the guard cannot see the bug it was written for")


if __name__ == "__main__":
    unittest.main()
