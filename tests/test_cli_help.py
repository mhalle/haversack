"""The command line explains itself (2026-09-03).

Every option has help text, every subcommand a description; the guide ships in the wheel
and `haversack docs` prints it whole or by section.
"""
import argparse

import pytest

from haversack import cli


def _parsers(parser):
    yield parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                yield from _parsers(sub)


def _build():
    """The parser, without running anything: parse `--help` is fatal, so build it by
    calling _run with a command that returns before dispatch would matter."""
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        captured["ap"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            cli._run(["tasks"])
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["ap"]


def test_every_option_has_help_and_every_command_a_description():
    ap = _build()
    missing = []
    for p in _parsers(ap):
        for a in p._actions:
            if a.help is None or a.help == argparse.SUPPRESS and a.dest != "help":
                if a.dest != "help":
                    missing.append(f"{p.prog}: {'/'.join(a.option_strings) or a.dest}")
        if p is not ap and p.description is None and "{" in p.prog:
            missing.append(f"{p.prog}: no description")
    assert not missing, missing


def test_top_level_help_is_for_humans(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "TotalSegmentator" in out and "examples:" in out and "haversack docs" in out
    assert "fused logit restore" not in out                      # jargon stays in the code


def test_segment_help_shows_defaults_and_examples(capsys):
    with pytest.raises(SystemExit):
        cli.main(["segment", "--help"])
    out = " ".join(capsys.readouterr().out.split())            # argparse wraps; compare unwrapped
    assert "(default: fp16)" in out and "(default: 20.0)" in out and "examples:" in out
    assert "--output" in out and "extension picks the format" in out


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
