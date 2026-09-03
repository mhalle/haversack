"""The CLI hands pipeline.segment the kwargs it actually accepts.

Regression for 2026-08-24: the WeightsStore work renamed segment()'s weights-location
parameter to ``weights=``, but the CLI kept passing ``model_root=`` - unconditionally,
so every ``haversack segment`` invocation raised TypeError. Nothing exercised the handler,
which is how it survived. This pins the wiring with a captured fake, no real model.
"""
import inspect
import subprocess
import sys

import pytest

from haversack import cli, pipeline


class FakeResult:
    def save(self, path):
        return path


def test_segment_cli_kwargs_are_accepted_by_pipeline(monkeypatch, tmp_path):
    real_params = set(inspect.signature(pipeline.segment).parameters)
    captured = {}

    def fake_segment(image, task, **kw):
        captured.update(kw, image=image, task=task)
        return FakeResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz"),
                   "--model-root", str(tmp_path / "weights"), "--quiet"])
    assert rc == 0
    assert captured["task"] == "total_fast"
    assert captured["weights"] == str(tmp_path / "weights")
    assert "model_root" not in captured

    # every kwarg the CLI passes must exist in the real signature, so a future
    # rename cannot silently break the handler again
    unknown = set(captured) - real_params - {"image", "task"}
    assert not unknown, f"CLI passes kwargs segment() does not accept: {sorted(unknown)}"


def test_errors_are_one_line_not_a_traceback(monkeypatch, tmp_path, capsys):
    """An outsider running `haversack serve` without the serve extra got a raw ModuleNotFoundError
    traceback (2026-09-02) although main_serve raises a worded InputError - nothing caught it.
    Every HaversackError now ends as `haversack: <message>` on stderr with status 2."""
    from haversack.errors import InputError

    def fake_segment(image, task, **kw):
        raise InputError("the server needs the serve extra")

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz"), "--quiet"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == "haversack: the server needs the serve extra"
    assert "Traceback" not in err


def test_tasks_lists_the_catalog_without_torch(tmp_path):
    """`haversack tasks` is the local answer to `haversack remote tasks`: every ecosystem's tasks,
    from the catalog alone - no weights, no torch (the describe-only front end stays light)."""
    code = (
        "import sys, haversack.cli as c\n"
        f"rc = c.main(['tasks', '--model-root', {str(tmp_path)!r}])\n"
        "assert rc == 0, rc\n"
        "assert 'torch' not in sys.modules, 'haversack tasks imported torch'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    names = [line.split()[0] for line in r.stdout.splitlines() if line.strip()]
    assert "ts:total_fast" in names and "mrsegmentator:base" in names and "moose:clin_ct_body" in names
    # an empty weights root: no nnU-Net task is installed, whatever "materialized" says about
    # its spec (TS specs ship in the catalog, so materialized is always True there)
    assert not [line for line in r.stdout.splitlines()
                if line.startswith(("ts:", "moose:", "mrsegmentator:")) and line.endswith("installed")]
    r2 = subprocess.run([sys.executable, "-c", code.replace("'tasks', ", "'tasks', '--installed', ")],
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert not [line for line in r2.stdout.splitlines() if line.startswith("ts:")]


def test_lean_install_says_what_it_lacks(monkeypatch, tmp_path, capsys):
    """`uvx git+https://github.com/mhalle/haversack segment ...` on 2026-09-03 died with a raw
    `ModuleNotFoundError: torch` while torch was an extra. The stack is core now; a lean
    (--no-deps) install still gets one line naming what to install, never a traceback."""
    import importlib.util
    real = importlib.util.find_spec

    def no_torch(name, *a, **k):
        return None if name == "torch" else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", no_torch)
    (tmp_path / "in.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "lean install" in err and "needs torch" in err and "Traceback" not in err


def test_missing_input_is_one_line(tmp_path, capsys):
    """A missing input file ended in a SimpleITK traceback from inside the reader
    (2026-09-03); the CLI now says so before touching any stack."""
    from haversack import cli
    rc = cli.main(["segment", str(tmp_path / "nope.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.nii.gz")])
    err = capsys.readouterr().err
    assert rc == 2 and err.strip() == f"haversack: input not found: {tmp_path / 'nope.nii.gz'}"


def test_tasks_with_a_name_prints_its_structures(tmp_path):
    """A blind user could not find the structure list (2026-09-03): `tasks` printed only
    name/engine/modality and nothing said --json carried `structures`. Now `tasks <name>`."""
    code = f"import haversack.cli as c; raise SystemExit(c.main(['tasks', 'total_fast', '--model-root', {str(tmp_path)!r}]))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    names = r.stdout.split()
    assert len(names) == 117 and "liver" in names and names[0] != "ts:total_fast"


def test_serve_refuses_contradictory_token_flags(capsys):
    assert cli.main(["serve", "--token", "x", "--no-token"]) == 2
    assert "contradict" in capsys.readouterr().err


def test_the_client_finds_a_local_servers_generated_token(tmp_path, monkeypatch):
    """A server without --token leaves its token in a file only this user can read; the
    client on the same machine reads it for a loopback URL on that port, and for nothing
    else - another host, or a port with no server file, gets no token."""
    import json
    import os
    from haversack.cache_admin import local_token_for, serve_token_path, write_serve_token
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path))
    p = serve_token_path(8790)
    assert p == tmp_path / "serve" / "8790.token"
    assert local_token_for("http://127.0.0.1:8790") is None            # no server yet
    write_serve_token(p, "s3cret", host="127.0.0.1", port=8790)
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    assert json.loads(p.read_text())["pid"] == os.getpid()             # this process "serves"
    assert local_token_for("http://127.0.0.1:8790") == "s3cret"
    assert local_token_for("http://localhost:8790") == "s3cret"
    assert local_token_for("127.0.0.1:8790") == "s3cret"
    assert local_token_for("http://[::1]:8790") == "s3cret"
    assert local_token_for("http://127.0.0.1:9000") is None            # another port
    assert local_token_for("http://gpu-box:8790") is None              # another machine
    assert local_token_for("http://127.0.0.1.evil.example:8790") is None   # a NAME, not local
    from haversack.cache_admin import server_address
    assert server_address("https://127.0.0.1")[1] == 443 and server_address("127.0.0.1:8790")[1] == 8790
    p.unlink()                                                          # a server bound elsewhere
    write_serve_token(p, "lan", host="192.168.1.5", port=8790)
    assert local_token_for("http://127.0.0.1:8790") is None            # is not the one on loopback
    p.unlink()
    write_serve_token(p, "any", host="0.0.0.0", port=8790)
    assert local_token_for("http://127.0.0.1:8790") == "any"
    p.unlink()
    write_serve_token(p, "s3cret", host="127.0.0.1", port=8790)
    assert local_token_for("http://127.0.0.1.evil.example.:8790") is None
    # a file left by a dead server is ignored: the token is never handed to whatever
    # answers on that port next
    p.write_text(json.dumps({"token": "stale", "pid": 2 ** 22 + 12345}))
    assert local_token_for("http://127.0.0.1:8790") is None
    p.write_text("old-plain-text-token\n")                             # the pre-review format
    assert local_token_for("http://127.0.0.1:8790") is None
    with pytest.raises(FileExistsError):                                # never overwrite silently
        write_serve_token(p, "x", host="127.0.0.1", port=8790)


def test_the_cli_refuses_output_names_it_does_not_write_and_store_batches(tmp_path, capsys):
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "-o", str(tmp_path / "out.zarr")])
    assert rc == 2 and ".duckn" in capsys.readouterr().err
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                   "--format", "seg.nrrd", "-o", str(tmp_path / "out.duckn")])
    assert rc == 2 and "exactly one input" in capsys.readouterr().err
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), str(tmp_path / "in.nii.gz"),
                   "--task", "total_fast", "-o", str(tmp_path / "out.duckn.zip")])
    assert rc == 2 and "exactly one input" in capsys.readouterr().err


def test_an_unknown_task_an_unreadable_image_and_a_bad_cache_root_are_one_line_errors(tmp_path, capsys, monkeypatch):
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "no_such_task",
                   "-o", str(tmp_path / "o.seg.nrrd")])
    err = capsys.readouterr().err
    assert rc == 2 and "unknown task 'no_such_task'" in err and "Traceback" not in err
    from haversack import io
    from haversack.errors import InputError
    with pytest.raises(InputError, match="cannot read .* as an image"):
        io.read_image(tmp_path / "in.nii.gz")
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "in.nii.gz"))
    rc = cli.main(["cache", "list"])
    assert rc == 2 and "not a directory" in capsys.readouterr().err


def test_an_unreachable_server_is_one_line():
    from haversack.client import RemoteClient, RemoteError
    with pytest.raises(RemoteError, match="cannot reach http://127.0.0.1:1"):
        RemoteClient("http://127.0.0.1:1", timeout=2).tasks()


def test_bad_output_names_are_refused_before_anything_runs(tmp_path, capsys, monkeypatch):
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    from haversack import cli as cli_mod
    monkeypatch.setattr(cli_mod, "_need_inference_stack",
                        lambda task=None: (_ for _ in ()).throw(AssertionError("stack demanded first")))
    for out in ("outdir/", "out", "out.txt", "out.zarr"):
        rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "total_fast",
                       "-o", str(tmp_path / out)])
        err = capsys.readouterr().err
        assert rc == 2 and "not an output haversack writes" in err, (out, err)
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), str(tmp_path / "in.nii.gz"),
                   "--task", "total_fast", "--format", "seg.nrrd", "-o", str(tmp_path / "o.zarr")])
    assert rc == 2 and "directory of labels" in capsys.readouterr().err


def test_a_half_built_model_folder_is_one_line(tmp_path):
    from haversack.errors import ModelNotFound
    from haversack.tasks import TaskSpec
    mf = tmp_path / "mf" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    mf.mkdir(parents=True)
    with pytest.raises(ModelNotFound, match="no dataset.json"):
        TaskSpec.from_model_folder(tmp_path / "mf")


def test_the_cache_root_is_one_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", "~/hv-root-test")
    from haversack.cache_admin import cache_root, stores, trainer_shim_dir
    from haversack.sources import default_input_cache
    root = cache_root()
    assert "~" not in str(root)
    assert default_input_cache() == root / "inputs"
    assert trainer_shim_dir() == root / "trainer_shims"
    assert {s["name"] for s in stores()} >= {"inputs", "results", "checkpoints", "trainer_shims", "serve"}
    assert all(str(s["path"]).startswith(str(root)) for s in stores() if s["name"] != "weights")
