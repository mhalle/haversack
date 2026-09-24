"""A server's bearer token (2026-09-24): `--token`, then HAVERSACK_SERVER_TOKEN, for both
`haversack serve` and `haversack modal deploy`, and on Modal a Secret the api function alone
mounts, in place of proxy auth.

What these pin, each against a way it could go wrong silently:
- the CLIENT's variable, HAVERSACK_TOKEN, never becomes a server's token - exported for
  `remote`, it would otherwise switch a deployment's auth mode;
- the token never reaches the deploy's command line or its environment, nor any image;
- a deployment that asked for a token and whose container has none refuses to start
  rather than serve every route open;
- the Secret is attached to the api and to nothing else (the twin, the workers).
"""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from haversack import cache_admin, cli

_TOKEN_VARS = ("HAVERSACK_SERVER_TOKEN", "HAVERSACK_TOKEN", "HAVERSACK_TOKEN_SECRET",
               "HAVERSACK_APP_NAME")


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _TOKEN_VARS}
    env.update(extra)
    return env


class ServerTokenResolution(unittest.TestCase):

    def test_the_flag_wins_then_the_server_variable(self):
        with mock.patch.dict(os.environ, _clean_env(HAVERSACK_SERVER_TOKEN="from-env"),
                             clear=True):
            self.assertEqual(cache_admin.server_token("from-flag"), ("from-flag", "--token"))
            self.assertEqual(cache_admin.server_token(None),
                             ("from-env", "HAVERSACK_SERVER_TOKEN"))

    def test_the_clients_variable_is_never_a_servers(self):
        with mock.patch.dict(os.environ, _clean_env(HAVERSACK_TOKEN="client-only"),
                             clear=True):
            self.assertEqual(cache_admin.server_token(None), (None, None))

    def test_whitespace_is_dropped_and_empty_means_unset(self):
        with mock.patch.dict(os.environ, _clean_env(HAVERSACK_SERVER_TOKEN=" tok\n"),
                             clear=True):
            self.assertEqual(cache_admin.server_token(None)[0], "tok")
        with mock.patch.dict(os.environ, _clean_env(HAVERSACK_SERVER_TOKEN="  \n"),
                             clear=True):
            self.assertEqual(cache_admin.server_token(None), (None, None))


class ServeTakesTheServerVariable(unittest.TestCase):
    """`main_serve` hands create_app the token it resolved; uvicorn is never started."""

    def _serve(self, env, token=None, no_token=False):
        import tempfile
        import types

        from haversack import serve
        seen = {}

        def fake_create_app(ex, token=None, **_):
            seen["token"] = token
            raise SystemExit(0)                    # stop before binding a socket

        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                token=token, no_token=no_token, host="127.0.0.1", port=0, device="cpu",
                dtype="auto", model_root=None, cache_models=1, workdir=tmp,
                no_result_cache=True, max_pending=4, keep_finished=4)
            # main_serve imports uvicorn first, and CI installs none; create_app stops the
            # run before uvicorn is touched, so an empty stand-in is enough
            with mock.patch.dict(sys.modules, {"uvicorn": types.ModuleType("uvicorn")}), \
                    mock.patch.dict(os.environ, _clean_env(**env), clear=True), \
                    mock.patch.object(serve, "create_app", fake_create_app), \
                    mock.patch.object(serve, "LocalExecutor"), \
                    mock.patch("haversack.segmenter.Segmenter"), \
                    mock.patch.object(cache_admin, "check_cache_root"), \
                    self.assertRaises(SystemExit):
                serve.main_serve(args)
        self.assertIn("token", seen, "create_app was never reached")
        return seen["token"]

    def test_serve_reads_the_server_variable(self):
        self.assertEqual(self._serve({"HAVERSACK_SERVER_TOKEN": "srv"}), "srv")

    def test_the_flag_beats_the_variable(self):
        self.assertEqual(self._serve({"HAVERSACK_SERVER_TOKEN": "srv"}, token="flag"), "flag")

    def test_no_token_beats_both(self):
        self.assertIsNone(self._serve({"HAVERSACK_SERVER_TOKEN": "srv"}, no_token=True))

    def test_the_clients_variable_does_not_reach_serve(self):
        got = self._serve({"HAVERSACK_TOKEN": "client-only"})
        self.assertNotEqual(got, "client-only")
        self.assertTrue(got)                       # a generated one instead


class ModalDeployAuth(unittest.TestCase):

    def _deploy(self, argv, env):
        seen = {"secrets": []}

        def fake_call(cmd, env=None):
            seen["cmd"], seen["env"] = cmd, env
            return 0

        with mock.patch.dict(os.environ, _clean_env(**env), clear=True), \
                mock.patch.object(subprocess, "call", fake_call), \
                mock.patch.object(cli, "_put_token_secret",
                                  lambda name, tok: seen["secrets"].append((name, tok))), \
                mock.patch("sys.stderr") as err:
            self.assertEqual(cli.main(["modal", "deploy", *argv]), 0)
        seen["stderr"] = "".join(c.args[0] for c in err.write.call_args_list if c.args)
        return seen

    def test_the_server_variable_puts_the_token_in_a_secret_named_for_the_app(self):
        s = self._deploy(["--app-name", "unit-app"], {"HAVERSACK_SERVER_TOKEN": "tok-123"})
        self.assertEqual(s["secrets"], [("unit-app-token", "tok-123")])
        self.assertEqual(s["env"]["HAVERSACK_TOKEN_SECRET"], "unit-app-token")
        self.assertIn("bearer token from HAVERSACK_SERVER_TOKEN", s["stderr"])
        self.assertNotIn("process list", s["stderr"])

    def test_the_token_never_rides_the_deploys_command_line_or_environment(self):
        for argv, env in ((["--token", "tok-flag"], {}),
                          ([], {"HAVERSACK_SERVER_TOKEN": "tok-flag"})):
            s = self._deploy(argv, env)
            self.assertFalse(any("tok-flag" in a for a in s["cmd"]), s["cmd"])
            self.assertFalse(any("tok-flag" in v for v in s["env"].values()),
                             [k for k, v in s["env"].items() if "tok-flag" in v])

    def test_the_flag_wins_and_says_it_is_visible(self):
        s = self._deploy(["--token", "flag"], {"HAVERSACK_SERVER_TOKEN": "env"})
        self.assertEqual(s["secrets"], [("haversack-serve-token", "flag")])
        self.assertIn("process list", s["stderr"])

    def test_no_token_keeps_proxy_auth_and_an_inherited_secret_name_is_dropped(self):
        s = self._deploy([], {"HAVERSACK_TOKEN": "client-only",
                              "HAVERSACK_TOKEN_SECRET": "left-over"})
        self.assertEqual(s["secrets"], [])
        self.assertNotIn("HAVERSACK_TOKEN_SECRET", s["env"])
        self.assertNotIn("HAVERSACK_TOKEN", s["env"])
        self.assertIn("Modal proxy auth", s["stderr"])

    def test_no_proxy_auth_without_a_token_says_it_is_open(self):
        s = self._deploy(["--no-proxy-auth"], {})
        self.assertEqual(s["env"]["HAVERSACK_PROXY_AUTH"], "0")
        self.assertIn("NONE", s["stderr"])


# The api function, as a deploy and a container see it: in a subprocess (importing modal_app
# registers a Modal app), with the decorators spied on so the raw function can be called.
_PROBE = r"""
import json, os, modal
applied, fns, secrets_named = [], [], []
real_asgi, real_fn, real_from_name = modal.asgi_app, modal.App.function, modal.Secret.from_name
def asgi_spy(**k):
    dec = real_asgi(**k)
    def w(f):
        applied.append((f, k.get("requires_proxy_auth")))
        return dec(f)
    return w
def fn_spy(self, *a, **k):
    dec = real_fn(self, *a, **k)
    def w(obj):
        fns.append(len(k.get("secrets") or []))
        return dec(obj)
    return w
def from_name_spy(name, *a, **k):
    secrets_named.append([name, list(k.get("required_keys") or [])])
    return real_from_name(name, *a, **k)
modal.asgi_app, modal.App.function = asgi_spy, fn_spy
modal.Secret.from_name = staticmethod(from_name_spy)
import haversack.modal_app as M
import haversack, haversack.serve as S
got = {}
S.create_app = lambda ex, token=None, **_: got.setdefault("token", token)
M.ModalExecutor = lambda: type("Ex", (), {})()
haversack.Segmenter = lambda **_: None
api_raw = applied[0][0]
os.environ.pop("HAVERSACK_TOKEN", None)
if os.environ.get("PROBE_CONTAINER_TOKEN"):
    os.environ["HAVERSACK_TOKEN"] = os.environ["PROBE_CONTAINER_TOKEN"]
try:
    api_raw()
    outcome = got.get("token")
except RuntimeError as e:
    outcome = "refused: " + str(e)
print(json.dumps({"proxy": [p for _, p in applied], "secrets_per_function": fns,
                  "secrets_named": secrets_named, "api_token": outcome}))
"""


def _probe(**env):
    base = {k: v for k, v in os.environ.items()
            if k not in _TOKEN_VARS + ("HAVERSACK_PUBLIC", "HAVERSACK_PROXY_AUTH")}
    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                       env={**base, "HAVERSACK_PUBLIC": "1", **env})
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class TheModalApi(unittest.TestCase):

    def test_without_a_token_the_api_is_behind_the_proxy_and_mounts_no_secret(self):
        p = _probe()
        self.assertEqual(p["proxy"], [True, False])            # api, then the public twin
        self.assertEqual(p["secrets_per_function"], [0, 0])
        self.assertEqual(p["secrets_named"], [])
        self.assertIsNone(p["api_token"])

    def test_a_token_secret_replaces_the_proxy_on_the_api_alone(self):
        p = _probe(HAVERSACK_TOKEN_SECRET="unit-token", PROBE_CONTAINER_TOKEN="tok\n")
        self.assertEqual(p["proxy"], [False, False])
        self.assertEqual(p["secrets_per_function"], [1, 0])    # the twin holds no token
        self.assertEqual(p["secrets_named"], [["unit-token", ["HAVERSACK_TOKEN"]]])
        self.assertEqual(p["api_token"], "tok")                # stripped, then enforced

    def test_a_container_without_its_token_refuses_to_serve_open(self):
        p = _probe(HAVERSACK_TOKEN_SECRET="unit-token")
        self.assertTrue(str(p["api_token"]).startswith("refused: "), p["api_token"])
        self.assertIn("unit-token", p["api_token"])

    def test_a_token_present_is_enforced_even_if_the_secret_name_did_not_arrive(self):
        p = _probe(PROBE_CONTAINER_TOKEN="tok")
        self.assertEqual(p["api_token"], "tok")


if __name__ == "__main__":
    unittest.main()
