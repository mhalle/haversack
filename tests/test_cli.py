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
    # one `<label>\t<name>` per line, in LABEL order - the only way to read a
    # result whose labels are neither contiguous nor alphabetical
    rows = [ln.split("\t") for ln in r.stdout.splitlines() if ln.strip()]
    labels = [int(k) for k, _ in rows]
    names = [n for _, n in rows]
    assert len(rows) == 117 and "liver" in names and names[0] != "ts:total_fast"
    assert labels == sorted(labels) and labels[0] == 1


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
    assert json.loads(p.read_text(encoding="utf-8"))["pid"] == os.getpid()             # this process "serves"
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


# -- the weights commands, which had no tests and rewrite a destructive path ---

def _installed(root, bucket, dataset, config="nnUNetTrainer__nnUNetPlans__3d_fullres"):
    d = root / bucket / dataset / config if bucket else root / dataset / config
    (d / "fold_0").mkdir(parents=True)
    (d / "dataset.json").write_text('{"channel_names":{"0":"CT"},"labels":{"background":0,"a":1},'
                                    '"numTraining":1,"file_ending":".nii.gz"}')
    (d / "fold_0" / "checkpoint_final.pth").write_bytes(b"w" * 32)
    return d.parent


def test_weights_list_sees_the_ecosystem_catalogs_not_only_the_root(tmp_path, capsys):
    """A catalog installs under <root>/<ecosystem>/Dataset*; scanning only the top
    level reported "0 dataset(s)" for a root two models had just been used from,
    and left no way to remove them."""
    _installed(tmp_path, None, "Dataset297_total")               # a TotalSegmentator model
    _installed(tmp_path, "totalvibe", "Dataset278")              # a catalog's
    _installed(tmp_path, "dentalsegmentator", "Dataset112_Dental")
    assert cli.main(["weights", "list", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Dataset297_total" in out
    assert "totalvibe/Dataset278" in out                         # named by where it lives
    assert "dentalsegmentator/Dataset112_Dental" in out
    assert "3 dataset(s)" in out


def test_weights_remove_needs_confirmation_and_names_what_it_would_delete(tmp_path, capsys):
    folder = _installed(tmp_path, "totalvibe", "Dataset278")
    assert cli.main(["weights", "remove", "278", "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "totalvibe/Dataset278" in err and "--yes" in err
    assert folder.is_dir()                                       # nothing deleted without --yes
    assert cli.main(["weights", "remove", "278", "--root", str(tmp_path), "--yes"]) == 0
    assert not folder.exists()


def test_weights_remove_does_not_match_a_longer_dataset_id(tmp_path, capsys):
    """`remove 27` must not take Dataset278 with it."""
    keep = _installed(tmp_path, "totalvibe", "Dataset278")
    assert cli.main(["weights", "remove", "27", "--root", str(tmp_path), "--yes"]) != 0
    assert "no installed weights match" in capsys.readouterr().err
    assert keep.is_dir()


def test_weights_remove_says_where_to_look_when_nothing_matches(tmp_path, capsys):
    _installed(tmp_path, "totalvibe", "Dataset278")
    assert cli.main(["weights", "remove", "999", "--root", str(tmp_path), "--yes"]) != 0
    assert "weights list" in capsys.readouterr().err


def test_weights_remove_refuses_a_pattern_a_bucket_and_anything_outside_the_root(tmp_path, capsys):
    """The id reaches a glob and then an rmtree. `remove '*'` deleted every
    dataset; `remove moose` deleted a whole ecosystem bucket - which the new
    "listed as <ecosystem>/Dataset<id>" hint makes a natural thing to type; and a
    symlink under the root put real data elsewhere within reach of rmtree."""
    keep = _installed(tmp_path / "root", "moose", "Dataset001_a")
    outside = tmp_path / "outside" / "Dataset999_precious"
    outside.mkdir(parents=True)
    (outside / "weights.pth").write_bytes(b"someone else's data")
    (tmp_path / "root" / "mybackup").symlink_to(tmp_path / "outside")

    for bad in ("*", "Dataset[01]*", "/etc", "..", "moose"):
        assert cli.main(["weights", "remove", bad, "--root", str(tmp_path / "root"), "--yes"]) != 0
        capsys.readouterr()
    assert keep.is_dir()                                   # the bucket is untouched
    # and the symlinked tree is not even listed, let alone removable
    assert cli.main(["weights", "remove", "999", "--root", str(tmp_path / "root"), "--yes"]) != 0
    assert (outside / "weights.pth").read_bytes() == b"someone else's data"
    capsys.readouterr()
    # the real thing still works
    assert cli.main(["weights", "remove", "1", "--root", str(tmp_path / "root"), "--yes"]) == 0
    assert not keep.exists()


def test_the_dataset_scan_never_descends_a_symlink(tmp_path):
    """Pinned on its own, because the containment check downstream masks it: with
    both guards removed the destructive test fails, with either one removed it
    passes. `weights remove` rmtree's what this yields, and rmtree follows a
    symlinked final component through to real data."""
    from haversack.cli import _installed_datasets
    root = tmp_path / "root"
    (root / "moose").mkdir(parents=True)
    (root / "moose" / "Dataset001_a").mkdir()
    outside = tmp_path / "outside"
    (outside / "Dataset999_precious").mkdir(parents=True)
    (root / "mybackup").symlink_to(outside)
    (root / "Dataset500_link").symlink_to(outside / "Dataset999_precious")

    found = _installed_datasets(root)
    assert [d.name for d in found] == ["Dataset001_a"]
    for d in found:
        assert outside not in d.parents and d != outside


def test_remove_refuses_a_match_that_resolves_outside_the_root(tmp_path, capsys):
    """The containment check, pinned on its own path.

    The bucket scan skips symlinks, but the id glob that runs first does not - it
    globs `Dataset<id>_*` at the root and returns whatever matches, symlink or
    not. So a root-level link named like a dataset reaches the delete with no
    other guard in front of it, and containment is what refuses it."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside" / "Dataset999_precious"
    outside.mkdir(parents=True)
    (outside / "weights.pth").write_bytes(b"data")
    (root / "Dataset999_precious").symlink_to(outside)

    from haversack.tasks import _dataset_dirs
    assert [d.name for d in _dataset_dirs(root, "999")] == ["Dataset999_precious"], \
        "the glob branch no longer reaches this; re-point the test at what does"
    assert cli.main(["weights", "remove", "999", "--root", str(root), "--yes"]) != 0
    assert "resolves outside" in capsys.readouterr().err
    assert (outside / "weights.pth").read_bytes() == b"data"
