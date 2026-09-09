"""The kernel layer must stay a leaf.

``haversack`` holds two layers in one package. The kernel modules know nothing about tasks, plans,
weights or files - torch and numpy only, plus rankfield (the ranked encoding, already lifted out
as of 0.6.0; ``ranked`` is a shim over it) - so they could be lifted into their own package the
day something outside wants them (nnU-Net's export path, TotalSegmentator, MOOSE, a CUDA user
who only needs the fused restore). That property is easy to lose by accident and cheap to
check, so it is checked here rather than trusted.
"""
import ast
import fnmatch
import pathlib
import unittest

import pytest

def _package_dir() -> pathlib.Path:
    """Locate the package by import, not by repo layout - the tests also run against a copy
    shipped into a container (cuda/), where there is no src/ directory."""
    try:
        import haversack
        return pathlib.Path(haversack.__file__).resolve().parent
    except Exception:
        return pathlib.Path(__file__).resolve().parent.parent / "src" / "haversack"


SRC = _package_dir()

KERNEL = {"grid", "mapping", "tables", "restore", "resample", "reference", "shuffleup",
          "ranked", "phantoms", "measure", "backends", "backends.metal", "backends.torch_gather", "backends.triton_gpu"}
PIPELINE = {"io", "preprocess", "frame", "network", "pipeline", "cli", "tasks", "values", "envelope",
            "weights_fetch", "trainers", "result", "cache", "segmenter", "weights", "progress", "job",
            "serve", "client", "modal_app", "sources", "ecosystems", "preview", "statistics",
            "schemas", "content", "jobstore", "jobpolicy", "filelock", "fetchlib", "attribution", "cache_admin", "ranked_store", "ranked_build",
            "ranked_output", "view", "ranked_restore", "duckn_io"}
# errors.py is deliberately dependency-free (stdlib only) so either layer may raise from it.
SHARED = {"errors"}
FORBIDDEN_FOR_KERNEL = {"nnunetv2", "SimpleITK", "nibabel", "scipy", "mlx", "totalsegmentator",
                        "nnunet_inference_mlx", "acvl_utils", "batchgenerators"}
# haversack must import on a machine with no mlx - that is the whole point of the torch path, and
# depending on the MLX toolkit's value types once made it unimportable on Linux.
FORBIDDEN_EVERYWHERE = {"mlx", "nnunet_inference_mlx"}
# scipy is allowed at call time inside resample (the identity-probe operators) but must not be a
# module-level import, so the kernel layer stays importable without it.
SCIPY_OK_AT_CALL_TIME = {"resample"}


def _module_path(name: str) -> pathlib.Path:
    return SRC / (name.replace(".", "/") + ".py") if name != "backends" else SRC / "backends" / "__init__.py"


def _imports(path: pathlib.Path, top_level_only: bool):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if top_level_only and getattr(node, "col_offset", 0) != 0:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import within haversack
                yield "." * node.level + (node.module or ""), node.lineno
            elif node.module:
                yield node.module.split(".")[0], node.lineno


def _tests_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _test_sources() -> list[pathlib.Path]:
    """Every python file under tests/, whatever it is called. Recursive on purpose:
    a rule that stops at the top level stops applying the moment someone makes a
    subdirectory, which is an ordinary thing to do."""
    return sorted(p for p in _tests_dir().rglob("*.py")
                  if "__pycache__" not in p.parts)


def _declared_tests(path: pathlib.Path) -> set[str]:
    """The test functions a file DECLARES, from its AST: module-level `test*`
    functions and `test*` methods of module-level classes, which is what pytest's
    default `python_functions`/`python_classes` would run.

    Nested functions are excluded deliberately. Walking the whole tree also matched
    a helper called `test_double_for_a_source()` defined inside another function,
    and flagged a module pytest never looks at."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    fn = (ast.FunctionDef, ast.AsyncFunctionDef)
    out = {n.name for n in tree.body if isinstance(n, fn) and n.name.startswith("test")}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            out |= {n.name for n in cls.body
                    if isinstance(n, fn) and n.name.startswith("test")}
    return out


class TestLayering(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _grab_config(self, pytestconfig):
        """pytest's own live configuration, stashed where a TestCase method can read it.
        A unittest class takes no funcargs, but an autouse fixture on it still runs."""
        self._pytestconfig = pytestconfig

    def test_every_test_this_repo_DECLARES_is_one_PYTEST_ACTUALLY_COLLECTS(self):
        """Reconciles what the tests directory declares against what pytest really runs.

        Six files failed this for the whole life of the repo. `kernel_test_grid.py` and
        five siblings arrived already misnamed in f0d83dd, matching neither default
        pattern - `test_*.py` wants the prefix, `*_test.py` the suffix, and
        `kernel_test_grid.py` has neither - and `python_files` was never configured, so
        98 kernel tests were never collected once: the grid, mapping, resample, backend
        and MLX-oracle parity checks, all of them silently absent from every green run.
        They cost 1.3 s. Renaming them to `test_kernel_*.py` on 2026-09-08 was the fix.

        The FIRST version of this guard compared filenames against the configured
        `python_files` and was itself hollow, three ways: it globbed one directory so a
        `tests/kernel/` subdirectory reinstated the bug one level down; a
        `collect_ignore_glob` in conftest removed 99 tests with the guard still green;
        and `python_files` is only one of four gates, so dropping `unittest.TestCase`
        from a class silently deleted it - every class in `test_engine_completeness.py`
        is collected ONLY because it subclasses TestCase, none being `Test`-prefixed, so
        one edit there deletes a whole checklist section. The fix is to stop modelling
        pytest and ASK it: run a real collection and compare the node ids against the
        functions the files declare.
        """
        import json
        import subprocess
        import sys

        here = _tests_dir()
        declared = {p: _declared_tests(p) for p in _test_sources()
                    if p.name != "conftest.py" and _declared_tests(p)}
        self.assertTrue(declared, "found no test functions at all - this guard is blind")

        out = subprocess.run(
            [sys.executable, "-m", "pytest", str(here), "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=600, cwd=here.parent)
        if out.returncode not in (0, 5):
            self.skipTest(f"could not collect: {out.stdout[-400:]}{out.stderr[-400:]}")
        collected: dict = {}
        for line in out.stdout.splitlines():
            if "::" not in line:
                continue
            path, _, rest = line.partition("::")
            name = rest.split("::")[-1].partition("[")[0]     # drop parametrize ids
            collected.setdefault((here.parent / path.strip()).resolve(), set()).add(name)

        problems = []
        for path, names in sorted(declared.items()):
            got = collected.get(path.resolve())
            if got is None:
                # zero nodes: either a gate dropped the file, or the module skips itself
                # at import (`pytest.importorskip` for zarr/duckn/fastapi). Naming the
                # file explicitly bypasses `python_files`, so it tells the two apart.
                one = subprocess.run(
                    [sys.executable, "-m", "pytest", str(path), "--collect-only", "-q",
                     "-p", "no:cacheprovider"],
                    capture_output=True, text=True, timeout=300, cwd=here.parent)
                if "::" in one.stdout:
                    problems.append(
                        f"{path.relative_to(here)}: declares {len(names)} test(s) that a "
                        "normal run never collects, though naming the file directly does - "
                        "check python_files, testpaths, collect_ignore* and norecursedirs")
                continue
            missing = sorted(names - got)
            if missing:
                problems.append(
                    f"{path.relative_to(here)}: declares {missing} which the run does not "
                    "collect - a class that stopped subclassing unittest.TestCase without a "
                    "`Test` prefix is the usual cause (python_classes)")
        self.assertEqual([], problems, "\n  ".join(problems))

    def test_every_module_is_classified(self):
        found = {p.stem for p in SRC.glob("*.py") if p.stem not in ("__init__", "__main__")}
        found |= {f"backends.{p.stem}" for p in (SRC / "backends").glob("*.py") if p.stem != "__init__"}
        found |= {"backends"} if (SRC / "backends").is_dir() else set()
        self.assertEqual(found, KERNEL | PIPELINE | SHARED,
                         "a module was added without deciding which layer it belongs to")

    def test_shared_modules_import_nothing(self):
        """errors.py is classified SHARED because either layer may raise from it - which only
        holds while it stays dependency-free. Enforce that rather than trusting the comment."""
        for name in sorted(SHARED):
            for mod, line in _imports(_module_path(name), top_level_only=False):
                self.assertTrue(mod.startswith("__future__"),
                                f"{name}.py:{line} imports {mod!r}; SHARED modules must stay stdlib-only")

    def test_kernel_modules_do_not_import_the_pipeline_layer(self):
        for name in sorted(KERNEL):
            path = _module_path(name)
            for mod, line in _imports(path, top_level_only=False):
                if mod.startswith("."):
                    target = mod.lstrip(".")
                    if target in PIPELINE:
                        self.fail(f"{name}.py:{line} imports the pipeline module {target!r}")

    def test_kernel_modules_depend_only_on_torch_and_numpy(self):
        for name in sorted(KERNEL):
            path = _module_path(name)
            for mod, line in _imports(path, top_level_only=True):
                if mod in FORBIDDEN_FOR_KERNEL:
                    self.assertIn(name, SCIPY_OK_AT_CALL_TIME if mod == "scipy" else set(),
                                  f"{name}.py:{line} imports {mod!r} at module level; the kernel "
                                  f"layer must stay torch + numpy (+ rankfield) so it can be extracted")

    def test_every_core_dependency_is_actually_installed_here(self):
        """CI does not `uv sync` - it hand-lists what to install, so torch can come
        from the CPU index - and that list drifts from pyproject silently. `obstore`
        has been a core dependency since 2026-09-03 and was never added to it; it
        did not matter while only `idc:` used it and the idc tests skipped without
        it. When 0.7.0 moved `s3:` and `gs:` onto the same client, those sources
        began reporting themselves disabled in CI, and the allowlist tests got
        "missing dependency" where they assert a refusal by name.

        Names, not imports: the mapping from distribution to module is not
        mechanical (scikit-image -> skimage), and what drifted is the install list.
        """
        import re
        import tomllib
        from importlib.metadata import PackageNotFoundError, distribution

        pyproject = SRC.parents[1] / "pyproject.toml"
        if not pyproject.exists():
            self.skipTest("running against an installed copy, not the repository")
        core = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
        missing = []
        for spec in core:
            name = re.split(r"[<>=!~\[; ]", spec.strip(), 1)[0]
            try:
                distribution(name)
            except PackageNotFoundError:
                missing.append(name)
        self.assertEqual(missing, [], f"core dependencies not installed: {missing} - if this "
                                      "is CI, add them to the install list in "
                                      ".github/workflows/tests.yml")

    def test_ci_pins_the_same_git_refs_as_pyproject(self):
        """The git-sourced dependencies are pinned in TWO files and nothing made them agree.

        CI hand-lists its installs rather than `uv sync`-ing, so a tag bump has to be made in
        `pyproject.toml` `[tool.uv.sources]` AND in the workflow. Miss the workflow and CI
        keeps testing the old revision while the project claims the new one - a green run
        that proves something about code nobody ships. Miss pyproject and it is the reverse.
        Neither failure announces itself; both were a comment in the notes, not a check.

        Only CI's entries are checked against pyproject, not the reverse: the engine extras
        (fastsurfer-lean, voxtell, synthstrip-torch) are git sources CI deliberately does not
        install.
        """
        import re
        import tomllib

        root = SRC.parents[1]
        pyproject, workflow = root / "pyproject.toml", root / ".github/workflows/tests.yml"
        if not pyproject.exists() or not workflow.exists():
            self.skipTest("running against an installed copy, not the repository")

        sources = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["uv"]["sources"]
        ci = re.findall(r"'([A-Za-z0-9_.-]+) @ git\+([^@']+)@([^']+)'",
                        workflow.read_text(encoding="utf-8"))
        self.assertTrue(ci, "no git-pinned installs found in the workflow - has the install "
                            "list changed shape? This guard reads it as text.")

        problems = []
        for name, url, ref in ci:
            declared = sources.get(name)
            if declared is None:
                problems.append(f"{name}: CI installs it from git, pyproject declares no source")
                continue
            if declared.get("git") != url:
                problems.append(f"{name}: CI {url} != pyproject {declared.get('git')}")
            pinned = declared.get("tag") or declared.get("rev")
            if pinned != ref:
                problems.append(f"{name}: CI pins {ref}, pyproject pins {pinned}")
        self.assertEqual(problems, [], "CI and pyproject disagree about a git dependency:\n  "
                                       + "\n  ".join(problems))

    def test_text_is_read_and_written_as_utf8_not_as_the_locale(self):
        """`Path.read_text()` with no encoding uses the LOCALE's, which is ASCII
        under LANG=C - and CI runs that way. The shipped store README has had an
        em-dash and a section sign for months; the day the runner image stopped
        coercing the C locale, `haversack segment -o x.duckn` began failing there
        with UnicodeDecodeError, and 0.7.0 and 0.7.1 both went out on a red CI.

        Data files are UTF-8 because we write them; say so at every read rather
        than inherit whatever the machine is set to. Reproduce the failure with::

            LC_ALL=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0 uv run pytest -q
        """
        import re
        bare = []
        for path in list(SRC.rglob("*.py")) + list((SRC.parents[1] / "tools").glob("*.py")):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\.read_text\(\s*\)", line):
                    bare.append(f"{path.name}:{i}")
        self.assertEqual(bare, [], f"read_text() without an encoding: {bare}")

    def test_the_default_path_does_not_need_the_ranked_store_extra(self):
        """rankfield, duckn and zarr are the `duckn` extra: the undocumented ranked store. A
        plain install has none of them, and `haversack segment IN -o labels.nii.gz` must still
        run - which it did not for a moment in 0.6.0, when the store-output check the CLI
        makes on EVERY segment imported a module that imported rankfield at module level.
        So: block the three, import the default path, run the CLI's error path."""
        import subprocess
        import sys
        code = (
            "import sys, importlib.abc\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('rankfield', 'duckn', 'zarr'):\n"
            "            raise ModuleNotFoundError(name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import haversack, haversack.cli, haversack.pipeline, haversack.segmenter\n"
            "import haversack.statistics, haversack.preview, haversack.client\n"
            "try:\n    import fastapi\nexcept ImportError:\n    pass\nelse:\n    import haversack.serve\n"
            "rc = haversack.cli.main(['segment', 'x.nii.gz', '--task', 'total_fast', '-o', 'out.bogus'])\n"
            "assert rc == 2, rc\n"
            "rc = haversack.cli.main(['segment', 'missing.nii.gz', '--task', 'total_fast', '-o', 'out.nii.gz'])\n"
            "assert rc == 2, rc\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertNotIn("Traceback", r.stderr)

    def test_a_lean_install_answers_segment_in_one_line(self):
        """README "Lean install": no torch, no nnunetv2, no scipy, no skimage - and no store
        extra. `haversack segment` must still validate its arguments and say what it lacks in
        one line. 0.6.0 died at the store-output check, which imported a module that imported
        the pipeline, which imported torch; the store door must likewise refuse before the
        network, naming the extra."""
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            code = (
                "import sys, importlib.abc\n"
                "BLOCK = ('torch', 'nnunetv2', 'scipy', 'skimage', 'rankfield', 'duckn', 'zarr')\n"
                "class Block(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in BLOCK:\n"
                "            raise ModuleNotFoundError(name)\n"
                "sys.meta_path.insert(0, Block())\n"
                "import haversack, haversack.cli\n"
                "import numpy as np, SimpleITK as sitk\n"
                f"img = '{d}/in.nii.gz'\n"
                "sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16)), img)\n"
                "rc = haversack.cli.main(['segment', img, '--task', 'total_fast', '-o', 'out.bogus'])\n"
                "assert rc == 2, rc\n"
                f"rc = haversack.cli.main(['segment', img, '--task', 'total_fast', '-o', '{d}/out.duckn.zip'])\n"
                "assert rc == 2, rc\n"
                "rc = haversack.cli.main(['restore', 'x.duckn', '-o', 'y.nii.gz'])\n"
                "assert rc == 2, rc\n"
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("duckn extra", r.stderr)

    def test_no_module_depends_on_the_mlx_toolkit(self):
        for path in sorted(SRC.rglob("*.py")):
            for mod, line in _imports(path, top_level_only=False):
                self.assertNotIn(mod, FORBIDDEN_EVERYWHERE,
                                 f"{path.name}:{line} imports {mod!r}; haversack must run where mlx does not")

    def test_the_haversack_tests_do_not_depend_on_the_mlx_toolkit_either(self):
        """The rules above cover src/haversack. The tests need the same property or CI cannot run
        them on Linux - test_frame once imported nnunet_inference_mlx.values.Geometry and
        broke the build."""
        for path in _test_sources():
            for mod, line in _imports(path, top_level_only=False):
                self.assertNotIn(mod, FORBIDDEN_EVERYWHERE,
                                 f"{path.name}:{line} imports {mod!r}; the haversack tests must run "
                                 f"where mlx does not")

    def test_scipy_is_only_a_call_time_dependency(self):
        """resample builds its operators with scipy, but importing haversack must not need it."""
        for mod, line in _imports(_module_path("resample"), top_level_only=True):
            self.assertNotEqual(mod, "scipy", f"resample.py:{line} imports scipy at module level")

    def test_importing_haversack_is_torch_free(self):
        """`import haversack` and the front-end/describe surface (Segmenter, ModelCache,
        TaskCatalog) must not pull torch - the serve front-end and describe-only callers
        should not pay for a multi-GB CUDA torch. Torch loads lazily on first use of an
        inference symbol (haversack.segment). A transitive leak (e.g. an eager subpackage that
        imports torch) is only caught at runtime, so use a fresh subprocess."""
        import subprocess
        import sys
        code = (
            "import sys, haversack\n"
            "assert 'torch' not in sys.modules, 'import haversack pulled torch'\n"
            "_ = (haversack.Segmenter, haversack.ModelCache, haversack.TaskCatalog, haversack.io)\n"
            "assert 'torch' not in sys.modules, 'the front-end API surface pulled torch'\n"
            # the serve front-end path: build a Segmenter + list + describe a task, all torch-free\n"
            "s = haversack.Segmenter(device='cpu'); ts = s.tasks(); assert ts\n"
            "s.describe(ts[0])\n"
            "assert 'torch' not in sys.modules, 'the describe/front-end path pulled torch'\n"
            "_ = haversack.segment\n"
            "assert 'torch' in sys.modules, 'lazy inference symbol failed to load torch'\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
