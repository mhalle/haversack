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
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz"),
                   "--model-root", str(tmp_path / "weights"), "--quiet"])
    assert rc == 0
    assert captured["task"] == "ts.v2:total_fast"
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
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
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
    assert "ts.v2:total_fast" in names and "mrsegmentator:base" in names and "moose:clin_ct_body" in names
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
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "lean install" in err and "needs torch" in err and "Traceback" not in err


def test_missing_input_is_one_line(tmp_path, capsys):
    """A missing input file ended in a SimpleITK traceback from inside the reader
    (2026-09-03); the CLI now says so before touching any stack."""
    from haversack import cli
    rc = cli.main(["segment", str(tmp_path / "nope.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz")])
    err = capsys.readouterr().err
    assert rc == 2 and err.strip() == f"haversack: input not found: {tmp_path / 'nope.nii.gz'}"


def test_tasks_with_a_name_prints_its_structures(tmp_path):
    """A blind user could not find the structure list (2026-09-03): `tasks` printed only
    name/engine/modality and nothing said --json carried `structures`. Now `tasks <name>`."""
    code = f"import haversack.cli as c; raise SystemExit(c.main(['tasks', 'ts.v2:total_fast', '--model-root', {str(tmp_path)!r}]))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    # one `<label>\t<name>` per line, in LABEL order - the only way to read a
    # result whose labels are neither contiguous nor alphabetical
    rows = [ln.split("\t") for ln in r.stdout.splitlines() if ln.strip()]
    labels = [int(k) for k, _ in rows]
    names = [n for _, n in rows]
    assert len(rows) == 117 and "liver" in names and names[0] != "ts.v2:total_fast"
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
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.zarr")])
    assert rc == 2 and ".duckn" in capsys.readouterr().err
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "--format", "seg.nrrd", "-o", str(tmp_path / "out.duckn")])
    assert rc == 2 and "exactly one input" in capsys.readouterr().err
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), str(tmp_path / "in.nii.gz"),
                   "--task", "ts.v2:total_fast", "-o", str(tmp_path / "out.duckn.zip")])
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
        rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                       "-o", str(tmp_path / out)])
        err = capsys.readouterr().err
        assert rc == 2 and "not an output haversack writes" in err, (out, err)
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), str(tmp_path / "in.nii.gz"),
                   "--task", "ts.v2:total_fast", "--format", "seg.nrrd", "-o", str(tmp_path / "o.zarr")])
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


def test_rights_given_a_task_name_points_at_cite(capsys):
    """`rights totalvibe:vibe` died of a raw KeyError (2026-09-12): a task name is shaped
    like a remote input, and `rights` looked its prefix up among the data sources. Every
    ecosystem the catalog serves - read from its registry, so a new catalog is covered the
    day it lands - is now a one-line refusal naming the `cite` command, and that command
    works. `ts` is in the loop on purpose: a two-letter prefix does not parse as a remote
    input at all, and was called a local file."""
    from haversack.ecosystems import registry
    for eco in registry():
        assert cli.main(["rights", f"{eco}:sometask"]) == 2, eco
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and "Traceback" not in err, err
        assert f"`haversack cite {eco}:sometask`" in err, err
    assert cli.main(["cite", "totalvibe:vibe"]) == 0
    assert "totalvibe" in capsys.readouterr().out


def test_rights_names_the_source_kinds_it_takes(capsys):
    """An unknown prefix that is not a model either is refused in one line listing the
    kinds `rights` does take, and says nothing of `cite`."""
    assert cli.main(["rights", "nosuchkind:x"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err, err
    assert "unknown source kind 'nosuchkind'" in err
    assert "idc:" in err and "zenodo:" in err and "http(s)://" in err
    assert "cite" not in err


# --- --folds and --events --------------------------------------------------------------------
# --folds: Steve Pieper, 2026-10-20 (#5). --events replaced #5's --json-progress on 2026-10-25: the
# run is written as the server's own job-status records, framed as /v1/jobs/{id}/events frames
# them, so a program reads a local run and a server with one parser (haversack.jobstatus.read).

@pytest.mark.parametrize("given, want", [(None, None), ("all", "all"), ("0,2", (0, 2)), ("3", (3,))])
def test_folds_reach_segment(monkeypatch, tmp_path, given, want):
    captured = {}

    def fake_segment(image, task, **kw):
        captured.update(kw)
        return FakeResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").touch()
    argv = ["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
            "-o", str(tmp_path / "out.nii.gz"), "--quiet"]
    rc = cli.main(argv + ([] if given is None else ["--folds", given]))
    assert rc == 0
    assert captured.get("folds") == want      # absent: segment()'s own default, as before


@pytest.mark.parametrize("bad", ["", "x", "1,a", "-1"])
def test_bad_folds_are_refused(monkeypatch, tmp_path, capsys, bad):
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: FakeResult())
    (tmp_path / "in.nii.gz").touch()
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz"), "--folds", bad])
    assert rc == 2
    assert "--folds" in capsys.readouterr().err


class EventResult(FakeResult):
    """A Segmentation as far as jobstatus.result_payload reads one."""

    class schema:
        names = {1: "liver", 2: "spleen"}

    timings = {"total": 1.5}
    provenance = {"deviations": []}

    def volumes_ml(self):
        return {"liver": 12.345}

    def present(self):
        return {1: "liver"}

    class grid:
        shape = (2, 3, 4)

    def save(self, path):
        from pathlib import Path
        Path(path).write_bytes(b"labels")
        return path


def _records(out):
    from haversack import jobstatus
    return list(jobstatus.read(out.splitlines()))


def _only_events(out):
    """Every stdout line is SSE framing: an event line, a data line, or the blank separator."""
    return all(line == "" or line.startswith(("event: status", "data: ")) for line in out.splitlines())


def test_events_are_the_servers_records_on_stdout(monkeypatch, tmp_path, capsys):
    from dataclasses import fields
    from haversack import jobstatus
    from haversack.content import digest_file
    from haversack.progress import Progress

    def fake_segment(image, task, progress=None, **kw):
        progress(Progress(stage="predict", step=1, n_steps=4, fraction=0.5))
        return EventResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    src = tmp_path / "in.nii.gz"
    src.write_bytes(b"an image")
    out = tmp_path / "out.seg.nrrd"
    rc = cli.main(["segment", str(src), "--task", "ts.v2:total_fast", "-o", str(out), "--events"])
    assert rc == 0
    std = capsys.readouterr().out
    assert _only_events(std)
    recs = _records(std)
    assert [r["state"] for r in recs] == ["running", "done"]
    run, done = recs
    assert set(run["progress"]) == {f.name for f in fields(Progress)}
    assert run["progress"]["stage"] == "predict" and run["progress"]["fraction"] == 0.5
    assert done["task"] == "ts.v2:total_fast"
    assert done["input_identity"] == [digest_file(src)]       # as the server names an upload
    assert done["progress"]["stage"] == "done" and done["progress"]["fraction"] == 1.0
    assert done["result"] == jsonish(jobstatus.result_payload(EventResult(), out))
    assert done["result"]["outputs"][0]["sha256"] == digest_file(out)
    assert done["links"] == {"result": out.resolve().as_uri()}
    assert done["created"] <= done["started"] <= done["finished"]


def jsonish(d):
    import json
    return json.loads(json.dumps(d, default=str))


def test_a_failure_is_a_failed_record_and_the_plain_line_stays(monkeypatch, tmp_path, capsys):
    from haversack.errors import UnsupportedModel

    def fake_segment(image, task, **kw):
        raise UnsupportedModel("not this one")

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz"), "--events"])
    cap = capsys.readouterr()
    assert rc == 2
    [rec] = _records(cap.out)                         # said once, not by every layer it crossed
    assert rec["state"] == "failed" and rec["error"] == "UnsupportedModel: not this one"
    assert rec["input_identity"]
    assert "haversack: not this one" in cap.err      # the line a person reads is still there


def test_a_missing_input_fails_by_name_without_an_identity(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: EventResult())
    rc = cli.main(["segment", str(tmp_path / "nope.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.nii.gz"), "--events"])
    [rec] = _records(capsys.readouterr().out)
    assert rc == 2 and rec["state"] == "failed" and "input not found" in rec["error"]
    assert "input_identity" not in rec


def test_a_batch_is_one_record_stream_per_input(monkeypatch, tmp_path, capsys):
    def fake_segment(image, task, **kw):
        if "bad" in str(image):
            raise RuntimeError("broken scan")
        return EventResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    good, bad = tmp_path / "good.nii.gz", tmp_path / "bad.nii.gz"
    good.write_bytes(b"g")
    bad.write_bytes(b"b")
    rc = cli.main(["segment", str(good), str(bad), "--task", "ts.v2:total_fast", "--format",
                   "seg.nrrd", "-o", str(tmp_path / "out"), "--events"])
    recs = _records(capsys.readouterr().out)
    assert rc == 1
    assert [r["state"] for r in recs] == ["done", "failed"]
    assert recs[0]["input_identity"] != recs[1]["input_identity"]
    assert recs[1]["error"] == "RuntimeError: broken scan"


def test_an_interrupted_run_says_cancelled():
    import io
    import contextlib
    buf = io.StringIO()
    ev = cli._Events("ts.v2:total_fast", out=buf)       # the stream the records go to, fixed at start
    ev.failed(KeyboardInterrupt())
    [rec] = _records(buf.getvalue())
    assert rec["state"] == "cancelled" and "error" not in rec


def test_the_record_is_the_servers_and_so_are_its_frame_and_reader():
    """One record, one framing, one parser: the server's events route frames with jobstatus.sse,
    its executors build results with jobstatus.result_payload, and RemoteClient reads the stream
    with jobstatus.read - the same three the command line uses. A record from a local run carries
    no field the server's status does not."""
    import ast
    import inspect
    import textwrap
    from haversack import client, jobstatus, serve
    assert serve.result_payload is jobstatus.result_payload
    assert serve.ranked_payload is jobstatus.ranked_payload
    # calls of a bare NAME only: `r.read()` on an HTTP error is an attribute call, and counting
    # it let a client with its own parser pass (a mutant did, 2026-10-25)
    calls = lambda f: {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(f))))
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "frame" in calls(serve.create_app)                 # `from .jobstatus import sse as frame`
    assert "read" in calls(client.RemoteClient.events)
    server_fields = {"id", "task", "state", "created", "started", "finished", "queue_position",
                     "progress", "version", "error", "input_refresh_skipped", "input_identity",
                     "key", "options", "deliverables", "deliverables_unavailable", "kind",
                     "cached", "result", "links"}
    rec = jobstatus.record("t", "done", progress={}, created=1, started=1, finished=2,
                           input_identity=["x"], error=None, result={}, links={})
    assert set(rec) <= server_fields


def test_read_takes_bytes_and_passes_over_keepalives_and_other_lines():
    from haversack import jobstatus
    lines = [b"event: status", b'data: {"state": "running"}', b"", b": keepalive", "",
             "event: status", 'data: {"state": "done"}', ""]
    assert [r["state"] for r in jobstatus.read(lines)] == ["running", "done"]


def test_a_download_message_is_a_fetch_stage_as_on_a_server(monkeypatch, tmp_path, capsys):
    def fake_segment(image, task, progress=None, **kw):
        progress("downloading series 3 of 9")          # what a source reports: text, not Progress
        return EventResult()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
              "-o", str(tmp_path / "out.seg.nrrd"), "--events"])
    run = _records(capsys.readouterr().out)[0]
    assert run["state"] == "running"
    assert run["progress"]["stage"] == "fetch" and run["progress"]["detail"] == "downloading series 3 of 9"


# --- the 2026-10-25 review of --events: each test holds one finding ---------------------------

def _fresh_result():
    r = EventResult()
    r.provenance = {"deviations": []}        # its own dict: run_one writes `inputs` into it
    return r


def test_a_folder_of_one_file_is_named_as_that_file_everywhere(monkeypatch, tmp_path, capsys):
    """A folder holding one file is that file (sources.sole_file) - in the record, in its own
    provenance and on a server. The record alone read sha256-tree: once."""
    from haversack.content import digest_file
    from haversack.sources import _content_facts
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: _fresh_result())
    folder = tmp_path / "one"
    folder.mkdir()
    (folder / "in.nii.gz").write_bytes(b"an image")
    cli.main(["segment", str(folder), "--task", "ts.v2:total_fast",
              "-o", str(tmp_path / "out.seg.nrrd"), "--events"])
    done = _records(capsys.readouterr().out)[-1]
    assert done["input_identity"] == [digest_file(folder / "in.nii.gz")]
    assert done["input_identity"] == [_content_facts(folder)["digest"]]


def _store_run(monkeypatch, tmp_path, said=None, extra=()):
    """The rank-field branch with the store writer faked: progress, a stray print, a file."""
    import haversack.ranked_output as ro
    import haversack.ranked_store as rs
    from haversack.progress import Progress

    def fake_store(image, task, out, *, progress=None, quiet=False, **kw):
        if said is not None:
            said["quiet"] = quiet
        progress(Progress(stage="predict", fraction=0.5))
        print("a library line on stdout")
        from pathlib import Path
        Path(out).write_bytes(b"store")
        return _fresh_result(), out

    monkeypatch.setattr(cli, "_need_store_extra", lambda: None)
    monkeypatch.setattr(rs, "check_target", lambda out: None)
    monkeypatch.setattr(ro, "supports_store_output", lambda task: True)
    monkeypatch.setattr(ro, "segment_to_store", fake_store)
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    return cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                     "-o", str(tmp_path / "out.duckn.zip"), "--events", *extra])


def test_a_rankfield_run_says_its_kind_on_every_record_and_its_inputs(monkeypatch, tmp_path, capsys):
    said = {}
    assert _store_run(monkeypatch, tmp_path, said) == 0
    recs = _records(capsys.readouterr().out)
    assert [r["state"] for r in recs] == ["running", "done"]
    assert all(r["kind"] == "rankfield" for r in recs)        # as a server's rankfield job says
    assert recs[-1]["result"]["outputs"][0]["kind"] == "rankfield"
    assert recs[-1]["result"]["provenance"]["inputs"][0]      # as a labels run's result has them
    assert said["quiet"] is False
    capsys.readouterr()
    (tmp_path / "out.duckn.zip").unlink()
    assert _store_run(monkeypatch, tmp_path, said, extra=["--quiet"]) == 0
    assert said["quiet"] is True                              # --quiet reaches the store writer


def test_stdout_holds_only_records_whatever_prints(monkeypatch, tmp_path, capsys):
    """An engine's or a library's print lands on stderr while --events owns stdout."""
    _store_run(monkeypatch, tmp_path)
    cap = capsys.readouterr()
    assert _only_events(cap.out) and "a library line" not in cap.out
    assert "a library line on stdout" in cap.err


def test_a_labels_run_records_no_kind(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: _fresh_result())
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
              "-o", str(tmp_path / "out.seg.nrrd"), "--events"])
    assert all("kind" not in r for r in _records(capsys.readouterr().out))   # as the server's


def test_a_failure_and_a_cancel_keep_where_the_run_stopped(monkeypatch, tmp_path, capsys):
    from haversack.errors import ResourceError
    from haversack.progress import Progress

    def fake_segment(image, task, progress=None, **kw):
        progress(Progress(stage="predict", fraction=0.25))
        raise ResourceError("not for Half")

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
              "-o", str(tmp_path / "out.nii.gz"), "--events"])
    failed = _records(capsys.readouterr().out)[-1]
    assert failed["error"] == "ResourceError: not for Half"           # serve's own wording
    assert failed["progress"]["stage"] == "predict" and failed["progress"]["fraction"] == 0.25

    import io
    buf = io.StringIO()
    ev = cli._Events("ts.v2:total_fast", out=buf)
    ev.progress(Progress(stage="restore", fraction=0.9))
    ev.failed(KeyboardInterrupt())
    cancelled = _records(buf.getvalue())[-1]
    assert cancelled["state"] == "cancelled" and "error" not in cancelled
    assert cancelled["progress"]["stage"] == "restore"


def test_the_timestamps_are_always_there_as_on_a_server():
    from haversack import jobstatus
    rec = jobstatus.record("t", "running", created=1.0, started=1.0, error=None)
    assert rec["finished"] is None and "finished" in rec         # null until known, as serve's
    assert "error" not in rec                                    # any other None is left out
    assert {"created", "started", "finished"} <= set(jobstatus.record("t", "failed"))


def test_a_pin_is_its_own_field_and_the_task_is_canonical():
    ev = cli._Events("ts.v2:total_fast@2.13.0")
    assert (ev.task, ev.version) == ("ts.v2:total_fast", "2.13.0")
    assert cli._Events("ts.v2:total_fast").version is None


def test_a_model_folder_is_named_by_its_absolute_path_and_batch_names_by_its_folder(
        monkeypatch, tmp_path, capsys):
    model = tmp_path / "Dataset990_Tiny" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    rel = "Dataset990_Tiny/nnUNetTrainer__nnUNetPlans__3d_fullres"
    assert cli._Events(rel).task == str(model.resolve())     # not the cwd-relative spelling
    monkeypatch.setattr(pipeline, "segment", lambda *a, **k: _fresh_result())
    for n in ("a", "b"):
        (tmp_path / f"{n}.nii.gz").write_bytes(n.encode())
    rc = cli.main(["segment", "a.nii.gz", "b.nii.gz", "--task", rel, "--format", "seg.nrrd",
                   "-o", "out", "--quiet"])
    assert rc == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "a_nnUNetTrainer__nnUNetPlans__3d_fullres.seg.nrrd",
        "b_nnUNetTrainer__nnUNetPlans__3d_fullres.seg.nrrd"]


class _ClosedPipe:
    def write(self, s):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_a_reader_that_goes_away_stops_the_records_not_the_run(monkeypatch, tmp_path, capsys):
    from haversack.progress import Progress

    def fake_segment(image, task, progress=None, **kw):
        progress(Progress(stage="predict", fraction=0.5))
        progress(Progress(stage="predict", fraction=0.9))
        return _fresh_result()

    monkeypatch.setattr(pipeline, "segment", fake_segment)
    (tmp_path / "in.nii.gz").write_bytes(b"x")
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    rc = cli.main(["segment", str(tmp_path / "in.nii.gz"), "--task", "ts.v2:total_fast",
                   "-o", str(tmp_path / "out.seg.nrrd"), "--events"])
    assert rc == 0 and (tmp_path / "out.seg.nrrd").read_bytes() == b"labels"
    err = capsys.readouterr().err
    assert err.count("closed its end") == 1                  # said once, not per record


_TERM_CHILD = r'''
import sys, time
from pathlib import Path
from haversack import cli, pipeline
from haversack.progress import Progress

def slow(image, task, progress=None, **kw):
    progress(Progress(stage="predict", fraction=0.1))
    time.sleep(60)

pipeline.segment = slow
src = Path(sys.argv[1]) / "in.nii.gz"
src.write_bytes(b"x")
sys.exit(cli.main(["segment", str(src), "--task", "ts.v2:total_fast",
                   "-o", str(Path(sys.argv[1]) / "out.nii.gz"), "--events"]))
'''


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_says_cancelled_and_still_dies_of_sigterm(tmp_path):
    import os
    import signal
    from pathlib import Path
    from haversack import jobstatus
    import haversack
    src = str(Path(haversack.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [src] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    errfile = tmp_path / "stderr.txt"
    with open(errfile, "wb") as errs:
        child = subprocess.Popen([sys.executable, "-c", _TERM_CHILD, str(tmp_path)], env=env,
                                 stdout=subprocess.PIPE, stderr=errs)
    lines = []
    try:
        for line in child.stdout:                      # until the run is under way
            lines.append(line)
            if line.startswith(b"data:"):
                break
        child.send_signal(signal.SIGTERM)
        lines += child.stdout.readlines()              # through the file object: it holds a buffer
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
    recs = list(jobstatus.read(lines))
    assert [r["state"] for r in recs] == ["running", "cancelled"], errfile.read_text()[-2000:]
    assert child.returncode == -signal.SIGTERM


def test_read_is_an_sse_parser_not_a_line_grep():
    from haversack import jobstatus
    lines = [
        "event: log", 'data: {"state": "not a status"}', "",       # another event: passed over
        'data: loaded 3 volumes', "",                              # a stray line: no event, no error
        b"\xff\xfe stray bytes", b"",                              # not UTF-8: no error
        "event: status\r\n", 'data: {"state":\r\n', 'data: "running"}\r\n', "\r\n",  # CRLF, two data lines
        "event:status", 'data:{"state": "done"}', "",              # no space after the colon
        'event: status\rdata: {"state": "cr"}\r\r',              # a lone CR ends a line too
        "event: status", 'data: {"state": "never dispatched"}',   # no blank line: not an event
    ]
    assert [r["state"] for r in jobstatus.read(lines)] == ["running", "done", "cr"]
    with pytest.raises(ValueError):                                # a broken producer is loud
        list(jobstatus.read(["event: status", "data: {not json", ""]))


_STRAY_CHILD = r"""
import os, sys
from pathlib import Path
from haversack import cli, pipeline
from haversack.progress import Progress

class R:
    class schema:
        names = {1: "liver"}
    timings = {}
    def __init__(self):
        self.provenance = {}
    def volumes_ml(self):
        return {}
    def present(self):
        return {}
    class grid:
        shape = (1, 1, 1)
    def save(self, path):
        Path(path).write_bytes(b"labels")

def run(image, task, progress=None, **kw):
    progress(Progress(stage="predict", fraction=0.5))
    os.write(1, b"data: a C library writing to fd 1\n")    # below Python's sys.stdout
    print("a print")
    progress(Progress(stage="restore", fraction=0.9))
    return R()

pipeline.segment = run
src = Path(sys.argv[1]) / "in.nii.gz"
src.write_bytes(b"x")
sys.exit(cli.main(["segment", str(src), "--task", "ts.v2:total_fast",
                   "-o", str(Path(sys.argv[1]) / "out.seg.nrrd"), "--events"]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptors")
def test_nothing_below_python_reaches_the_records_either(tmp_path):
    """fd 1 itself is stderr for the run: a write to the descriptor, which no sys.stdout
    redirect sees, would otherwise land inside the record stream."""
    import os
    from pathlib import Path
    import haversack
    from haversack import jobstatus
    src = str(Path(haversack.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [src] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    done = subprocess.run([sys.executable, "-c", _STRAY_CHILD, str(tmp_path)], env=env,
                          capture_output=True, timeout=120)
    out, err = done.stdout.decode(), done.stderr.decode()
    assert done.returncode == 0, err[-2000:]
    assert _only_events(out), out
    assert [r["state"] for r in jobstatus.read(out.splitlines())] == ["running", "running", "done"]
    assert "a C library writing to fd 1" in err and "a print" in err
