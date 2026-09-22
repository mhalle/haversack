"""`tools/` scripts reach into haversack and into each other; every such name must exist.

2026-09-22: `tools/ranked_emit_modal.py` did `import ranked_emit` and called
`ranked_emit.main(...)`. Since 2026-09-03 `tools/ranked_emit.py` had been a command-line shim
over `haversack.ranked_output` with no `main`, so every nnU-Net emit through the Modal tool
died of an AttributeError on the worker - after a GPU was scheduled and the series fetched.
The suite stayed green because no test imports `tools/`; the ranked-store smoke after the
seg-0.8 merge found it. `test_tools_rankfield_api` guards the rankfield half of this; this
file guards the haversack half and the sibling-script half.

Everything is read from SOURCE, never imported: several scripts build a Modal app, its
images and volumes at import, and a sibling script is imported on the worker from a path
this checkout does not have. A module's names are what its top level binds (defs, classes,
assignments, imports), the package root adds `_LAZY` and `_LAZY_SUBMODULES`, and a package
adds its submodule files. A module that can make names up (`import *`, a module-level
`__getattr__` other than the root's) is not judged. Calls into a function defined at a
module's top level are bound against a signature built from its `def` - a dropped or
renamed parameter fails here the day it happens, not on a GPU.

Static only, as its rankfield sibling: it cannot see a call whose meaning changed while its
shape did not.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
PKG = ROOT / "src" / "haversack"


def _module_file(dotted: str) -> Path | None:
    """`haversack.ranked_output` -> its source; a sibling script's bare name -> tools/<name>.py."""
    parts = dotted.split(".")
    if parts[0] == "haversack":
        base = PKG.joinpath(*parts[1:])
        for p in (base.with_suffix(".py") if parts[1:] else None, base / "__init__.py"):
            if p is not None and p.is_file():
                return p
        return None
    if len(parts) == 1 and (TOOLS / f"{dotted}.py").is_file():
        return TOOLS / f"{dotted}.py"
    return None


def _ours(dotted: str) -> bool:
    return dotted == "haversack" or dotted.startswith("haversack.") or (
        "." not in dotted and (TOOLS / f"{dotted}.py").is_file())


def _top_level(tree: ast.Module):
    """The statements a module runs at import: its body, descending into if/try/with."""
    stack = list(tree.body)
    while stack:
        node = stack.pop(0)
        yield node
        if isinstance(node, (ast.If, ast.Try, ast.With)):
            for field in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, field, []) or [])
            for h in getattr(node, "handlers", []):
                stack.extend(h.body)


def _targets(t):
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)


@lru_cache(maxsize=None)
def _names(dotted: str):
    """`(names, defs)` a module binds at top level, or `None` when it cannot be judged.
    `defs` maps a function name to its `ast.FunctionDef`, for binding calls."""
    path = _module_file(dotted)
    if path is None:
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    defs: dict[str, ast.FunctionDef] = {}
    for node in _top_level(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(_targets(t))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_targets(node.target))
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                return None
            names.update(a.asname or a.name for a in node.names)
    if "__getattr__" in names:
        if dotted != "haversack":
            return None
        # The root's lazy exports: `_LAZY`'s keys and `_LAZY_SUBMODULES`, as literals.
        for node in _top_level(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in ("_LAZY", "_LAZY_SUBMODULES")
                    for t in node.targets):
                value = ast.literal_eval(node.value)
                names.update(value.keys() if isinstance(value, dict) else value)
    if path.name == "__init__.py":
        names.update(p.stem for p in path.parent.glob("*.py") if p.stem != "__init__")
        names.update(p.name for p in path.parent.iterdir() if (p / "__init__.py").is_file())
    return frozenset(names), defs


def _signature(fn: ast.FunctionDef) -> inspect.Signature:
    """An `inspect.Signature` with the def's parameter kinds and which ones have defaults."""
    P, a = inspect.Parameter, fn.args
    params = []
    positional = [*a.posonlyargs, *a.args]
    first_default = len(positional) - len(a.defaults)
    for i, arg in enumerate(positional):
        kind = P.POSITIONAL_ONLY if i < len(a.posonlyargs) else P.POSITIONAL_OR_KEYWORD
        params.append(P(arg.arg, kind, default=None if i >= first_default else P.empty))
    if a.vararg:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append(P(arg.arg, P.KEYWORD_ONLY, default=P.empty if d is None else None))
    if a.kwarg:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))
    return inspect.Signature(params)


def _aliases(tree: ast.Module) -> dict[str, tuple[str, str | None]]:
    """Local name -> (module, member or None), for every import of ours ANYWHERE in the file
    (the defect was a function-local import). A name bound to two different targets is
    dropped: which one a given call sees is not decidable from here."""
    seen: dict[str, set] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _ours(a.name):
                    continue
                if a.asname:
                    seen.setdefault(a.asname, set()).add((a.name, None))
                else:                                       # `import a.b` binds `a`
                    top = a.name.split(".")[0]
                    seen.setdefault(top, set()).add((top, None))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and _ours(node.module):
            for a in node.names:
                seen.setdefault(a.asname or a.name, set()).add((node.module, a.name))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def _check(owner: str, member: str):
    """(problem or None, the member's def or None)."""
    got = _names(owner)
    if got is None:
        return (None, None) if _module_file(owner) else (f"no module {owner!r}", None)
    names, defs = got
    if member in names:
        return None, defs.get(member)
    if _module_file(f"{owner}.{member}"):
        return None, None                                   # a submodule
    return f"{owner} has no {member!r}", None


def problems_in(path: Path) -> tuple[list[str], int, int]:
    """(problems, names checked, calls bound) for one script."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _aliases(tree)
    problems: list[str] = []
    checked = bound = 0

    for local, (module, member) in sorted(aliases.items()):
        if member is None:
            if _module_file(module) is None:
                problems.append(f"{path.name}: imports {module} - no such module")
            continue
        checked += 1
        err, _ = _check(module, member)
        if err:
            problems.append(f"{path.name}: from {module} import {member} - {err}")

    for node in ast.walk(tree):
        # `mod.attr`, rooted at an alias of one of our MODULES (not a member)
        target = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in aliases and aliases[node.value.id][1] is None:
            checked += 1
            err, _ = _check(aliases[node.value.id][0], node.attr)
            if err:
                problems.append(f"{path.name}:{node.lineno}: {node.value.id}.{node.attr}"
                                f" - {err}")
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in aliases and aliases[f.id][1] is not None:
            target = aliases[f.id]
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id in aliases and aliases[f.value.id][1] is None:
            target = (aliases[f.value.id][0], f.attr)
        if target is None:
            continue
        err, fn = _check(*target)
        if err or fn is None:
            continue                                        # reported above, or not a def
        if any(k.arg is None for k in node.keywords) or any(
                isinstance(a, ast.Starred) for a in node.args):
            continue                                        # *args / **kwargs: unknowable
        bound += 1
        try:
            _signature(fn).bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})
        except TypeError as exc:
            problems.append(f"{path.name}:{node.lineno}: {target[0]}.{target[1]}"
                            f"{_signature(fn)} <- {exc}")
    return problems, checked, bound


class ToolsNameOnlyWhatExists(unittest.TestCase):
    def test_every_haversack_and_sibling_name_under_tools_exists_and_binds(self):
        scripts = sorted(TOOLS.glob("*.py"))
        self.assertTrue(scripts, f"no scripts under {TOOLS}")
        problems: list[str] = []
        checked = bound = 0
        for path in scripts:
            p, c, b = problems_in(path)
            problems += p
            checked += c
            bound += b
        self.assertEqual([], problems, "tools/ names what does not exist:\n  "
                                       + "\n  ".join(problems))
        # A guard that sees nothing passes everything: hold it to having looked.
        self.assertGreater(checked, 20, "the guard found almost no haversack/sibling names "
                                        "under tools/ - an import form it cannot read?")
        self.assertGreater(bound, 5, "almost no calls were bound against a signature")

    def test_the_defect_it_was_written_for_is_caught(self):
        """The 2026-09-22 shape, verbatim: a function-local import of a sibling script and a
        call to a `main` it does not define."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "probe.py"
            p.write_text("def emit():\n    import ranked_emit\n"
                         "    ranked_emit.main('a', 'b', 'c', 6, 8.0, 'none')\n")
            problems, _, _ = problems_in(p)
        self.assertTrue(any("ranked_emit has no 'main'" in m for m in problems), problems)

    def test_a_call_that_no_longer_binds_is_caught(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "probe.py"
            p.write_text("from haversack.ranked_output import main as emit_one\n"
                         "emit_one('a', 'b', 'c', 6, 8.0, None, 'extra', 'more')\n"
                         "emit_one('a', depth=6)\n"             # task, outdir missing
                         "emit_one('a', 'b', 'c', anything=1)\n")  # **segment_kw: binds
            problems, _, bound = problems_in(p)
        self.assertEqual(3, bound)
        self.assertEqual(2, len(problems), problems)


if __name__ == "__main__":
    unittest.main()
