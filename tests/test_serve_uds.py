"""`haversack serve --uds PATH` and `--server unix:PATH` (2026-10-25): the same HTTP API on a
Unix socket, the local transport of record (AGENTS.md, "Local transport").

What these pin, each against a way it could go wrong silently:
- the socket is this user's only (uvicorn's own bind makes it 0666) from before it listens;
- a live server's socket is never taken over, and a non-socket file is never deleted, while
  a dead server's socket is replaced;
- auth is UNCHANGED: the token is still required over the socket, and a generated one is
  found by the bundled client through a file keyed by the socket's path - never handed to a
  socket the file does not name;
- a real server started by `main_serve` answers the jobs API, its event stream included,
  through `RemoteClient` - and leaves no socket behind.

Socket paths live under /tmp: AF_UNIX paths are capped (104 bytes on macOS), which pytest's
tmp_path and a harness's TMPDIR can both exceed.
"""
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest

if not hasattr(socket, "AF_UNIX"):
    pytest.skip("no Unix sockets on this platform", allow_module_level=True)

from haversack import cache_admin
from haversack.errors import InputError


def _private_dir():
    d = tempfile.mkdtemp(prefix="hv-", dir="/tmp")
    os.chmod(d, 0o700)
    return d


def _rmtree(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TheSpelling(unittest.TestCase):

    def test_unix_paths_in_both_spellings_and_nothing_else(self):
        self.assertEqual(cache_admin.unix_socket("unix:/tmp/a/x.sock"), "/tmp/a/x.sock")
        self.assertEqual(cache_admin.unix_socket("unix:///tmp/a/x.sock"), "/tmp/a/x.sock")
        self.assertEqual(cache_admin.unix_socket("unix:rel.sock"), "rel.sock")
        self.assertIsNone(cache_admin.unix_socket("unix:"))
        self.assertIsNone(cache_admin.unix_socket("http://127.0.0.1:8790"))
        self.assertIsNone(cache_admin.unix_socket("gpu-box:8790"))


class BindingTheSocket(unittest.TestCase):

    def setUp(self):
        from haversack.serve import bind_unix_socket
        self.bind = bind_unix_socket
        self.dir = _private_dir()
        self.addCleanup(_rmtree, self.dir)
        self.path = os.path.join(self.dir, "s.sock")

    def test_the_socket_is_this_users_only_before_it_listens(self):
        sock = self.bind(self.path)
        self.addCleanup(sock.close)
        mode = os.stat(self.path).st_mode
        self.assertTrue(stat.S_ISSOCK(mode))
        self.assertEqual(stat.S_IMODE(mode), 0o600)
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(probe.close)
        with self.assertRaises(ConnectionRefusedError):   # bound, not listening: nobody in
            probe.connect(self.path)

    def test_a_live_servers_socket_is_refused_by_name(self):
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(live.close)
        live.bind(self.path)
        live.listen(1)
        with self.assertRaisesRegex(InputError, "another server is listening"):
            self.bind(self.path)
        self.assertTrue(stat.S_ISSOCK(os.stat(self.path).st_mode))

    def test_a_dead_servers_socket_is_replaced(self):
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(self.path)
        dead.close()                                       # the file stays, nobody listens
        before = os.stat(self.path).st_ino
        sock = self.bind(self.path)
        self.addCleanup(sock.close)
        self.assertNotEqual(os.stat(self.path).st_ino, before)

    def test_a_file_that_is_not_a_socket_is_never_deleted(self):
        Path(self.path).write_text("mine")
        with self.assertRaisesRegex(InputError, "not a socket"):
            self.bind(self.path)
        self.assertEqual(Path(self.path).read_text(), "mine")


class TheLocalToken(unittest.TestCase):
    """The bundled client reads a socket server's generated token from a file keyed by the
    socket's resolved path, and only while the file names that path and its server lives."""

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.addCleanup(_rmtree, self.cache)
        env = mock.patch.dict(os.environ, {"HAVERSACK_CACHE_DIR": self.cache})
        env.start()
        self.addCleanup(env.stop)
        self.dir = _private_dir()
        self.addCleanup(_rmtree, self.dir)
        self.sock = str(Path(self.dir, "s.sock").resolve())

    def _write(self, **over):
        f = cache_admin.serve_token_path_for_socket(self.sock)
        f.parent.mkdir(parents=True, exist_ok=True)
        rec = {"token": "tok", "pid": os.getpid(), "host": "127.0.0.1", "port": 8790,
               "uds": self.sock}
        rec.update(over)
        f.write_text(json.dumps(rec))
        return f

    def test_found_for_the_socket_the_file_names(self):
        self._write()
        self.assertEqual(cache_admin.local_token_for(f"unix:{self.sock}"), "tok")
        self.assertEqual(cache_admin.local_token_file(f"unix:{self.sock}"),
                         cache_admin.serve_token_path_for_socket(self.sock))

    def test_a_symlinked_spelling_of_the_same_socket_finds_it(self):
        self._write()
        link = os.path.join(self.dir, "alias")
        os.symlink(self.dir, link)
        self.assertEqual(cache_admin.local_token_for(f"unix:{link}/s.sock"), "tok")

    def test_never_for_another_socket_or_a_dead_server(self):
        self._write(uds="/somewhere/else.sock")
        self.assertIsNone(cache_admin.local_token_for(f"unix:{self.sock}"))
        self._write(pid=2 ** 22 + 12345)                       # no such process
        self.assertIsNone(cache_admin.local_token_for(f"unix:{self.sock}"))

    def test_a_port_servers_file_is_not_a_sockets(self):
        f = cache_admin.serve_token_path(8790)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"token": "tcp", "pid": os.getpid(), "host": "127.0.0.1",
                                 "port": 8790}))
        self.assertIsNone(cache_admin.local_token_for(f"unix:{self.sock}"))
        self.assertEqual(cache_admin.local_token_for("http://127.0.0.1:8790"), "tcp")


class MainServeOnASocket(unittest.TestCase):
    """`main_serve --uds` with uvicorn stood in: what it binds, writes and says. Runs without
    uvicorn installed, as CI is."""

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.addCleanup(_rmtree, self.cache)
        self.dir = _private_dir()
        self.addCleanup(_rmtree, self.dir)
        self.sock = str(Path(self.dir, "s.sock").resolve())

    def _serve(self, **over):
        from haversack import serve
        seen = {}

        class Config:
            def __init__(self, app, **kw):
                seen["config"] = kw

        class Server:
            def __init__(self, config):
                pass

            def run(self, sockets):
                s = sockets[0]
                seen["sock"] = s
                seen["family"] = s.family
                seen["mode"] = stat.S_IMODE(os.stat(self_path).st_mode)
                tf = cache_admin.serve_token_path_for_socket(self_path)
                seen["token_file"] = json.loads(tf.read_text()) if tf.exists() else None

        self_path = self.sock
        fake = types.ModuleType("uvicorn")
        fake.Config, fake.Server = Config, Server
        args = types.SimpleNamespace(
            token=None, no_token=False, host="127.0.0.1", port=8790, uds=self.sock,
            device="cpu", dtype="fp16", model_root=None, cache_models=1,
            workdir=os.path.join(self.dir, "work"), no_result_cache=True, max_pending=4,
            keep_finished=4)
        for k, v in over.items():
            setattr(args, k, v)
        env = {k: v for k, v in os.environ.items()
               if k not in ("HAVERSACK_SERVER_TOKEN", "HAVERSACK_TOKEN")}
        env["HAVERSACK_CACHE_DIR"] = self.cache
        with mock.patch.dict(sys.modules, {"uvicorn": fake}), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(serve, "LocalExecutor"), \
                mock.patch.object(serve, "create_app", lambda ex, token=None, **_: (
                    seen.setdefault("token", token))), \
                mock.patch("haversack.segmenter.Segmenter"), \
                mock.patch("atexit.register"), \
                mock.patch("sys.stdout", new_callable=lambda: _Tee()) as out:
            self.assertEqual(serve.main_serve(args), 0)
        seen["out"] = out.getvalue()
        seen["sock"].close()
        return seen

    def test_it_serves_the_socket_private_with_a_token_the_client_can_find(self):
        seen = self._serve()
        self.assertEqual(seen["family"], socket.AF_UNIX)
        self.assertEqual(seen["mode"], 0o600)
        self.assertEqual(seen["config"].get("uds"), self.sock)
        self.assertTrue(seen["token"])                      # auth unchanged: a token, generated
        self.assertEqual(seen["token_file"]["token"], seen["token"])
        self.assertEqual(seen["token_file"]["uds"], self.sock)
        self.assertIn(f"serving on unix:{self.sock}", seen["out"])
        self.assertNotIn("http://127.0.0.1:8790", seen["out"])

    def test_the_host_does_not_decide_the_token_file_on_a_socket(self):
        seen = self._serve(host="10.1.2.3")                 # ignored beside --uds
        self.assertEqual(seen["token_file"]["token"], seen["token"])

    def test_no_token_still_says_so_naming_the_socket(self):
        err = _Tee()
        with mock.patch("sys.stderr", err):
            seen = self._serve(no_token=True)
        self.assertIsNone(seen["token"])
        self.assertIsNone(seen["token_file"])
        self.assertIn(f"unix:{self.sock} WITHOUT a token", err.getvalue())


class _Tee:
    def __init__(self):
        self.parts = []

    def write(self, s):
        self.parts.append(s)
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


class ARealServerOnASocket(unittest.TestCase):
    """`main_serve` itself, real uvicorn, a fake segmenter: the jobs API through
    `RemoteClient("unix:...")` - submit, events, status, result, cancel - and a 401 without
    the token. Skipped where uvicorn is not installed (CI installs none)."""

    def setUp(self):
        pytest.importorskip("uvicorn")
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        sys.path.insert(0, str(Path(__file__).parent))
        self.addCleanup(sys.path.remove, str(Path(__file__).parent))
        from test_serve import FakeSegmenter, volume_bytes
        self.volume_bytes = volume_bytes
        self.cache = tempfile.mkdtemp()
        self.addCleanup(_rmtree, self.cache)
        self.dir = _private_dir()
        self.addCleanup(_rmtree, self.dir)
        self.sock = os.path.join(self.dir, "s.sock")
        env = {k: v for k, v in os.environ.items()
               if k not in ("HAVERSACK_SERVER_TOKEN", "HAVERSACK_TOKEN")}
        env["HAVERSACK_CACHE_DIR"] = self.cache
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)

        import uvicorn

        from haversack import serve
        servers = []
        real_server = uvicorn.Server

        class Kept(real_server):
            def __init__(self, config):
                super().__init__(config)
                servers.append(self)

        self.seg = FakeSegmenter(steps=3)
        self.at_exit = []
        args = types.SimpleNamespace(
            token=None, no_token=False, host="127.0.0.1", port=8790, uds=self.sock,
            device="cpu", dtype="fp16", model_root=None, cache_models=1,
            workdir=os.path.join(self.dir, "work"), no_result_cache=True, max_pending=4,
            keep_finished=8)
        patches = [mock.patch.object(uvicorn, "Server", Kept),
                   mock.patch("haversack.segmenter.Segmenter", lambda **kw: self.seg),
                   # what main_serve leaves for exit runs when the test says, not at pytest's
                   mock.patch("atexit.register", self.at_exit.append)]
        for q in patches:
            q.start()
            self.addCleanup(q.stop)
        self.status = {}
        self.thread = threading.Thread(
            target=lambda: self.status.setdefault("rc", serve.main_serve(args)), daemon=True)
        self.thread.start()
        t0 = time.time()
        while not (servers and servers[0].started):
            if not self.thread.is_alive() or time.time() - t0 > 20:
                self.fail(f"the server did not start: {self.status}")
            time.sleep(0.02)
        self.server = servers[0]
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=20)
        try:
            os.unlink(self.sock)
        except FileNotFoundError:
            pass

    def _client(self, token="auto"):
        from haversack.client import RemoteClient
        url = f"unix:{self.sock}"
        tok = cache_admin.local_token_for(url) if token == "auto" else token
        return RemoteClient(url, token=tok, timeout=10)

    def test_the_jobs_api_answers_through_the_socket(self):
        self.assertEqual(stat.S_IMODE(os.stat(self.sock).st_mode), 0o600)
        scan = Path(self.dir, "scan.nii.gz")
        scan.write_bytes(self.volume_bytes(4242))
        with self._client() as c:
            self.assertIn("total_fast", c.tasks())
            jid = c.submit(scan, "total_fast")
            snaps = list(c.events(jid))                    # SSE over the socket
            self.assertEqual(snaps[-1]["state"], "done")
            final = c.status(jid)
            self.assertEqual(final["state"], "done")
            self.assertEqual(final["result"]["volumes_ml"], {"spleen": 210.0})
            out = c.fetch(jid, Path(self.dir, "labels.seg.nrrd"))
            self.assertTrue(Path(out).read_bytes().startswith(self.volume_bytes()))
        self.assertEqual(self.seg.calls[0][1], "total_fast")

    def test_a_cancel_reaches_a_running_job(self):
        gate = threading.Event()
        self.seg.gate = gate
        scan = Path(self.dir, "scan.nii.gz")
        scan.write_bytes(self.volume_bytes(4343))
        with self._client() as c:
            jid = c.submit(scan, "total_fast")
            t0 = time.time()
            while c.status(jid)["state"] != "running" and time.time() - t0 < 10:
                time.sleep(0.02)
            c.cancel(jid)
            gate.set()
            self.assertEqual(c.wait(jid, poll_interval=0.05)["state"], "cancelled")

    def test_the_token_is_still_required_on_the_socket(self):
        from haversack.client import RemoteError
        scan = Path(self.dir, "scan.nii.gz")
        scan.write_bytes(self.volume_bytes(4444))
        with self._client(token=None) as c:
            with self.assertRaisesRegex(RemoteError, "401"):
                c.submit(scan, "total_fast")
        self.assertEqual(self.seg.calls, [])

    def test_an_unreachable_socket_is_named(self):
        from haversack.client import RemoteClient, RemoteError
        gone = os.path.join(self.dir, "nobody.sock")
        with RemoteClient(f"unix:{gone}", timeout=2) as c:
            with self.assertRaisesRegex(RemoteError, f"cannot reach unix:{gone}"):
                c.health()

    def test_the_socket_and_its_token_file_go_with_the_server(self):
        token_file = cache_admin.serve_token_path_for_socket(self.sock)
        self.assertTrue(token_file.exists())
        self.server.should_exit = True
        self.thread.join(timeout=20)
        self.assertEqual(self.status.get("rc"), 0)
        for fn in self.at_exit:
            fn()
        self.assertFalse(os.path.lexists(self.sock))
        self.assertFalse(token_file.exists())


if __name__ == "__main__":
    unittest.main()
