"""`tools/` has no other coverage, so its rankfield calls are checked against the real package.

2026-09-08: the 0.3.0 migration moved `Geometry` to one order (`shape`/`directions`/`origin`)
and every call site was updated except `tools/ranked_restore_modal.py`, which kept calling
`rf.Geometry(spacing_zyx=..., shape_zyx=...)` and raised on its first line of real work. The
whole suite stayed green because nothing under `tools/` is imported by a test - these are
`--no-project` scripts run by hand, several of them only on Modal or only against a GPU.

This reads the scripts rather than running them: every name imported from rankfield must
still exist, and every call through one must bind against the installed signature. That is
enough to catch a rename, a moved member, or a changed keyword the day the pin moves, which
is the whole failure mode. It cannot catch a call whose meaning changed while its shape did
not - no static check can - and it deliberately says nothing about behaviour.

It checks against whatever rankfield is INSTALLED, not against the tag pyproject pins. In a
clone with `uv pip install -e ../rankfield` that is the sibling working tree, so drift shows
up here before the pin moves; in CI it is the pinned tag. Both are the right answer for the
environment asking the question.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import unittest
from pathlib import Path

import pytest

pytest.importorskip("rankfield")

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _resolve(dotted: str):
    """Resolve ``rankfield.store.KNOWN_VERSIONS`` the way the running script would.

    Returns ``(object, None)`` or ``(None, reason)``. The attribute is tried BEFORE the
    submodule of the same name, because that is what the caller gets: rankfield exports a
    `restore` function and also has a `restore` module, and importing the module first made
    this guard skip every `rf.restore(...)` call as an unsignaturable object - it survived
    a renamed-keyword mutant until that was fixed.
    """
    parts = dotted.split(".")
    try:
        obj = importlib.import_module(parts[0])
    except ImportError:
        return None, f"no module {parts[0]!r}"
    seen = parts[0]
    for attr in parts[1:]:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            try:
                obj = importlib.import_module(f"{seen}.{attr}")
            except ImportError:
                return None, f"{seen} has no {attr!r}"
        seen = f"{seen}.{attr}"
    return obj, None


def _rankfield_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> dotted rankfield path, for every form the scripts actually use:
    ``import rankfield as rf``, ``from rankfield import levels as rf_levels``,
    ``from rankfield.store import KNOWN_VERSIONS``."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name != "rankfield" and not a.name.startswith("rankfield."):
                    continue
                # Bare `import rankfield.store` binds the TOP name, not the submodule;
                # only `as` binds the dotted path to a name of its own.
                aliases[a.asname or "rankfield"] = a.name if a.asname else "rankfield"
        elif isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "rankfield" or node.module.startswith("rankfield.")):
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def _rooted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """``rf.Grid.isotropic`` -> ``rankfield.Grid.isotropic``, if rooted at a rankfield alias."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in aliases:
        return None
    return aliases[node.id] + "".join("." + p for p in reversed(parts))


class ToolsAgreeWithTheInstalledRankfield(unittest.TestCase):
    def test_every_rankfield_name_and_call_under_tools_still_exists_and_binds(self):
        scripts = sorted(TOOLS.glob("*.py"))
        self.assertTrue(scripts, f"no scripts found under {TOOLS}")

        problems: list[str] = []
        checked = signatures = 0
        for path in scripts:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            aliases = _rankfield_aliases(tree)
            if not aliases:
                continue

            for dotted in sorted(set(aliases.values())):
                _, err = _resolve(dotted)
                if err:
                    problems.append(f"{path.name}: imports {dotted} - {err}")

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                dotted = _rooted(node.func, aliases)
                if dotted is None:
                    continue
                checked += 1
                obj, err = _resolve(dotted)
                if err:
                    problems.append(f"{path.name}:{node.lineno}: {dotted} - {err}")
                    continue
                try:
                    sig = inspect.signature(obj)
                except (TypeError, ValueError):
                    continue                       # a builtin or C callable: nothing to bind
                else:
                    signatures += 1
                if any(k.arg is None for k in node.keywords) or any(
                        isinstance(a, ast.Starred) for a in node.args):
                    continue                       # *args / **kwargs: the shape is unknowable
                try:
                    sig.bind_partial(*[None] * len(node.args),
                                     **{k.arg: None for k in node.keywords})
                except TypeError as exc:
                    problems.append(f"{path.name}:{node.lineno}: {dotted}{sig} <- {exc}")

        self.assertEqual([], problems, "tools/ has drifted from the installed rankfield:\n  "
                                       + "\n  ".join(problems))
        self.assertGreater(checked, 0, "no rankfield calls found under tools/ - has the guard "
                                       "stopped seeing them? (an import form it cannot read)")
        # Finding calls is not the same as checking them: a resolver that hands back
        # unsignaturable objects finds everything and verifies nothing.
        self.assertGreater(signatures, checked // 2,
                           f"only {signatures} of {checked} rankfield calls resolved to "
                           "something with a signature - the resolver is returning modules "
                           "or builtins where the caller gets a function")


if __name__ == "__main__":
    unittest.main()
