"""The command line explains itself (2026-09-03).

Every option has help text, every command a description; the guide ships in the wheel
and `haversack docs` prints it whole or by section. Since 2026-09-11 the command line is
click's, so these walk its command tree - the argparse internals the first version reached
into are gone with argparse.
"""
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import click
import pytest

from haversack import cli

#: Every command the argparse command line had, so the move to click provably dropped none.
#: A new command extends this; a missing one is the regression it exists for.
COMMANDS = {
    "haversack", "haversack segment", "haversack get", "haversack tasks", "haversack cite",
    "haversack rights", "haversack weights", "haversack weights fetch",
    "haversack weights coverage", "haversack weights list", "haversack weights remove",
    "haversack weights refresh", "haversack serve", "haversack modal",
    "haversack modal deploy", "haversack modal app-path", "haversack remote",
    "haversack remote submit", "haversack remote status", "haversack remote fetch",
    "haversack remote cancel", "haversack remote tasks", "haversack docs", "haversack cache",
    "haversack cache list", "haversack cache path", "haversack cache clean",
}


def _commands(cmd, path=("haversack",)):
    yield " ".join(path), cmd
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _commands(sub, path + (name,))


def test_no_command_was_lost_moving_to_click():
    assert COMMANDS <= {where for where, _ in _commands(cli.COMMAND_LINE)}


def test_every_option_has_help_and_every_command_a_description():
    walked, missing = 0, []
    for where, cmd in _commands(cli.COMMAND_LINE):
        walked += 1
        if not (cmd.help or "").strip():
            missing.append(f"{where}: no description")
        for p in cmd.params:
            if not (getattr(p, "help", None) or "").strip():
                missing.append(f"{where}: {'/'.join(getattr(p, 'opts', ())) or p.name}")
    assert walked >= len(COMMANDS), f"walked {walked} commands - the walk is not reaching them"
    assert not missing, missing


def test_top_level_help_is_for_humans(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "TotalSegmentator" in out and "examples:" in out and "haversack docs" in out
    assert "fused logit restore" not in out                      # jargon stays in the code


def test_the_commands_are_listed_in_the_order_a_user_needs_them(capsys):
    """Click lists subcommands alphabetically unless told otherwise; `segment` comes first."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.index("  segment ") < out.index("  get ") < out.index("  tasks ") < out.index("  cache ")


def test_segment_help_shows_defaults_and_examples(capsys):
    with pytest.raises(SystemExit):
        cli.main(["segment", "--help"])
    out = " ".join(capsys.readouterr().out.split())            # click wraps; compare unwrapped
    assert "[default: fp16]" in out and "[default: 20.0]" in out and "examples:" in out
    assert "--output" in out and "extension picks the format" in out
    assert out.count("one or more inputs; several = batch mode") == 1, (
        "the positional's help is printed more than once")


def test_version_names_the_release(capsys):
    """`haversack --version` failed under argparse, which had no such option - a `uvx` smoke
    run found it (2026-09-11)."""
    from haversack import __version__
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"haversack {__version__}"


#: The real entry point, for what only a process of its own shows.
ENTRY = [sys.executable, "-c", "from haversack.cli import main; raise SystemExit(main())"]


def _env(**extra):
    repo = Path(__file__).resolve().parents[1]
    inherited = os.environ.get("PYTHONPATH", "")
    # prepended, never replaced: CI finds haversack ONLY through PYTHONPATH=src
    return {**os.environ, **extra,
            "PYTHONPATH": os.pathsep.join(p for p in (str(repo / "src"), inherited) if p)}


def test_the_shell_completes_task_names_from_the_catalog():
    """Completion is click's own, driven by `_HAVERSACK_COMPLETE` through the real entry point;
    `--task` completes from the catalog, including the short spelling people type for a
    TotalSegmentator task. A subprocess, because completion ends by exiting."""
    env = _env(_HAVERSACK_COMPLETE="bash_complete",
               COMP_WORDS="haversack segment --task total_f", COMP_CWORD="3")
    r = subprocess.run(ENTRY, env=env, capture_output=True, text=True, timeout=120)
    items = [line.split(",", 1)[1] for line in r.stdout.splitlines() if "," in line]
    assert "total_fast" in items, (r.returncode, r.stdout[:300], r.stderr[-300:])
    assert all(i.startswith("total_f") for i in items), items


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX signal and exit status")
def test_ctrl_c_still_ends_the_process_by_sigint(tmp_path):
    """Click turns Ctrl-C into "Aborted!" and exit 1. Under argparse it reached the interpreter
    and the process died of SIGINT - the status on which a shell's `for` loop over a folder of
    scans stops, where exit 1 goes on to the next scan. The command waits on a server that
    accepts the connection and never answers; the signal goes only once it has connected, so
    it lands in the command rather than in the imports, where it would pass either way."""
    with socket.create_server(("127.0.0.1", 0)) as srv:
        srv.settimeout(60)
        url = f"http://127.0.0.1:{srv.getsockname()[1]}"
        p = subprocess.Popen([*ENTRY, "remote", "--server", url, "status", "abc"],
                             env=_env(HAVERSACK_CACHE_DIR=str(tmp_path)),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            p.kill()
            pytest.fail(f"the command never reached the server: {p.communicate()[1][-300:]}")
        with conn:
            p.send_signal(signal.SIGINT)
            err = p.communicate(timeout=60)[1]
    assert p.returncode == -signal.SIGINT, (p.returncode, err[-300:])


def test_docs_prints_the_guide_whole_and_by_section(capsys):
    assert cli.main(["docs"]) == 0
    whole = capsys.readouterr().out
    assert whole.startswith("# haversack") and "## Weights" in whole and "## Local server" in whole
    assert cli.main(["docs", "weights"]) == 0
    part = capsys.readouterr().out
    assert part.startswith("## Weights") and "## Local server" not in part
    assert cli.main(["docs", "--sections"]) == 0
    heads = capsys.readouterr().out.splitlines()
    assert "Weights" in heads and "Local server" in heads
    assert cli.main(["docs", "no-such-section"]) == 2
    assert "no guide section matches" in capsys.readouterr().err
