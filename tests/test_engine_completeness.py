"""Every engine must reach every place that has to know about it.

The registry is deliberately a static, closed set rather than a plugin system - see
:mod:`haversack.engines.registry` for the three properties of this system that rule
discovered plugins out. The cost of that choice is a checklist, and a checklist in prose
goes stale, so it lives here: one obligation per test, each failing with the step it found
missing.

WHY THIS FILE IS SHAPED THE WAY IT IS. Its first version was written on 2026-09-08 and was
substantially hollow. Three adversarial reviews built a plausible sixth engine, broke it in
five ways at once, and watched the whole suite pass. What they found is worth stating,
because every rule below exists to kill one of them:

* An obligation compared a derived list against the dict it was derived FROM, so it was
  empty for any registry, while the thing it claimed to check - that the enable flags reach
  the container - could be deleted outright with the suite still green.
* The Modal obligation checked a dictionary KEY, so naming a class that does not exist
  passed. Two silent, deploy-fatal omissions lived in that gap.
* Routing was checked against the registry's own dict, proving only that the registry
  agreed with itself; an engine routed from an ecosystem nothing constructs passed.
* The distribution anchor accepted any package the engine merely pulls in. Every engine
  brings torch, so a typo in the only name that mattered passed with torch beside it.
* Four facts had no test at all, and an attribution entry of ``{}`` satisfied the rest.

Two lessons are baked in. First, a check must reconcile TWO INDEPENDENT SOURCES; comparing
a thing to itself always passes. Second, the Modal obligations are STATIC (they read
modal_app as text) rather than importing it - an ``importorskip`` deleted them silently in
exactly the per-engine environments where an author is most likely to be working.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import pytest

from haversack.engines import registry as R

ROOT = Path(__file__).resolve().parent.parent
PKG = Path(R.__file__).resolve().parent.parent
MODAL_APP = PKG / "modal_app.py"
PYPROJECT = ROOT / "pyproject.toml"

#: Engines whose runtime is core rather than an extra. The default engine is the only one:
#: its packages ship with haversack, so the rules about extras cannot apply to it.
DEFAULT_ONLY = {R.NNUNETV2}


def _optional_engines() -> dict:
    return {n: e for n, e in R.ENGINES.items() if n not in DEFAULT_ONLY}


def _requirement_names(specs) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", str(s).strip())[0] for s in specs}


def _pyproject() -> dict:
    if not PYPROJECT.exists():
        pytest.skip("running against an installed copy, not the repository")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _modal_tree() -> ast.Module:
    if not MODAL_APP.exists():
        pytest.skip("modal_app.py is not present in this install")
    return ast.parse(MODAL_APP.read_text(encoding="utf-8"))


class TheRegistryItself(unittest.TestCase):
    def test_the_registry_is_not_empty_and_names_its_own_keys(self):
        """The floor under everything else: a guard that walks an empty dict passes
        vacuously, and one whose keys and rows disagree checks the wrong engine."""
        self.assertGreaterEqual(len(R.ENGINES), 2, "the registry has no engines to check")
        self.assertEqual([], [k for k, e in R.ENGINES.items() if e.name != k],
                         "Engine.name must equal its key in ENGINES")

    def test_every_optional_engine_can_be_switched_off(self):
        """`enabled_env=None` means ALWAYS ON, which is the default engine's semantics.
        A second always-on engine cannot be turned off and deploys unconditionally."""
        missing = sorted(n for n, e in _optional_engines().items() if not e.enabled_env)
        self.assertEqual([], missing,
                         f"engines with no enable flag: {missing} - set enabled_env, or "
                         "this engine can never be switched off")

    def test_every_optional_engine_can_tell_whether_its_runtime_is_installed(self):
        """`available()` returns True unconditionally when `runtime_module` is None, so an
        engine without one reports itself installed in an environment holding nothing."""
        missing = sorted(n for n, e in _optional_engines().items() if not e.runtime_module)
        self.assertEqual([], missing,
                         f"engines with no runtime_module: {missing} - name the module that "
                         "proves the runtime is present, or `available()` always says yes")


class EveryEngineDeclaresWhatInstallsIt(unittest.TestCase):
    def test_every_engine_declares_the_distributions_it_installs(self):
        missing = [n for n, e in R.ENGINES.items() if not e.dist]
        self.assertEqual([], missing,
                         f"engines with no `dist`: {missing} - /v1/version cannot report "
                         "which build produced a result without it")

    def test_every_optional_engine_names_a_distribution_ONLY_ITS_EXTRA_BRINGS(self):
        """The anchor has to be a package specific to this engine.

        Accepting any installed name is what made the first version useless: every engine
        also brings torch, so `dist=("swin-unetr", "torch")` - a typo in the one name that
        identifies the engine - anchored on torch and passed, while /v1/version reported
        nothing new. That is the exact defect `dist` was added to prevent. So the anchor
        must be a NON-CORE package declared by the engine's OWN extra.
        """
        data = _pyproject()
        core = _requirement_names(data["project"]["dependencies"])
        extras = {k: _requirement_names(v)
                  for k, v in data["project"]["optional-dependencies"].items()}
        problems = []
        for name, e in _optional_engines().items():
            own = extras.get(e.extra or "", set()) - core
            if not (set(e.dist) & own):
                problems.append(
                    f"{name}: dist={e.dist} names nothing that only extra {e.extra!r} "
                    f"brings (its own packages: {sorted(own) or 'none'}) - anchoring on a "
                    "core package means a typo in the engine's real package ships silently")
        self.assertEqual([], problems, "\n  ".join(problems))

    def test_every_optional_engine_extra_exists_and_installs_something(self):
        """`torch = []` is a real extra that installs nothing, kept as a compatibility
        alias. Copied onto a new row it satisfies mere existence and installs no runtime."""
        data = _pyproject()
        extras = data["project"]["optional-dependencies"]
        problems = []
        for name, e in _optional_engines().items():
            if e.extra is None:
                problems.append(f"{name}: no extra - nothing installs this engine's runtime")
            elif e.extra not in extras:
                problems.append(f"{name}: extra {e.extra!r} is not defined in pyproject")
            elif not extras[e.extra]:
                problems.append(f"{name}: extra {e.extra!r} declares no packages")
        self.assertEqual([], problems, "\n  ".join(problems))


class EveryGitSourceIsPinnedToSomethingReal(unittest.TestCase):
    """A git dependency has to be pinned, and the pin has to look like a pin.

    Nothing validated these. An adversarial review shipped
    ``rev = "0000000000000000000000000000000000000000"`` for a new engine and the whole
    suite stayed green - CI installs no engine extra, so no engine source is ever resolved
    there. This cannot prove a revision EXISTS without the network, but it does refuse a
    floating branch and the placeholder shape, which is what a hand-edited pin looks like.
    """

    def test_every_git_source_is_pinned_by_tag_or_revision(self):
        sources = _pyproject()["tool"]["uv"]["sources"]
        unpinned = sorted(n for n, spec in sources.items()
                          if "git" in spec and not ({"tag", "rev"} & set(spec)))
        self.assertEqual([], unpinned,
                         f"git sources with no tag or rev: {unpinned} - a floating branch "
                         "makes a sync unreproducible")

    def test_no_pinned_revision_is_a_placeholder(self):
        sources = _pyproject()["tool"]["uv"]["sources"]
        problems = []
        for name, spec in sources.items():
            rev = spec.get("rev")
            if rev is None:
                continue
            if not re.fullmatch(r"[0-9a-f]{40}", rev):
                problems.append(f"{name}: rev {rev!r} is not a full 40-character sha")
            elif len(set(rev)) < 5:
                problems.append(f"{name}: rev {rev!r} looks like a placeholder, not a commit")
        self.assertEqual([], problems, "\n  ".join(problems))


    @pytest.mark.slow
    def test_every_pinned_revision_actually_exists_upstream(self):
        """The only check that can prove a pin resolves, so it needs the network.

        Marked slow, therefore out of the fast suite and out of CI, which installs no engine
        extra and so never resolves an engine source at all - a bad rev reaches a developer's
        `uv sync` and nothing earlier. Run it deliberately: `uv run pytest -m slow -k pinned`.

        A TAG is checked by listing: `git ls-remote URL <tag>` prints it or does not.

        A REVISION cannot be checked that way, and the first version of this test tried.
        `ls-remote`'s positional arguments are refname PATTERNS, so a sha matches nothing
        unless a ref is literally named that; a fabricated sha and a real one both exit 0
        with empty output, and this test then excused any 40-hex value outright. It proved
        reachability of the repository and nothing about the pin - while three of the four
        git sources here are pinned by `rev`, so almost every pin went unchecked. A review
        demonstrated it by passing a sha that provably did not exist (2026-09-08).

        So a revision is FETCHED. `git fetch --depth 1 URL <sha>` into a scratch repository
        either brings the object down or refuses, and refusing is the answer we want. It
        needs a positive control: a server that serves no object by hash at all would fail
        a good pin the same way it fails a bad one. So each repository is first asked for
        its own HEAD sha by hash, and one that cannot answer that is skipped rather than
        failed.
        """
        import subprocess
        import tempfile

        def git(*args, cwd=None):
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=120, cwd=cwd)

        def serves_by_hash(scratch, url, sha):
            """Whether ``url`` will hand over the commit ``sha`` when asked for it by hash."""
            return git("fetch", "--depth", "1", "--quiet", url, sha, cwd=scratch).returncode == 0

        sources = _pyproject()["tool"]["uv"]["sources"]
        problems = []
        with tempfile.TemporaryDirectory() as scratch:
            if git("init", "--quiet", scratch).returncode != 0:
                self.skipTest("cannot create a scratch git repository")
            for name, spec in sources.items():
                url, tag, rev = spec.get("git"), spec.get("tag"), spec.get("rev")
                if not url or not (tag or rev):
                    continue
                try:
                    if tag:
                        out = git("ls-remote", url, tag, f"{tag}^{{}}")
                        if out.returncode != 0:
                            self.skipTest(f"cannot reach {url}: {out.stderr.strip()[:120]}")
                        if not out.stdout.strip():
                            problems.append(f"{name}: tag {tag!r} is not published by {url}")
                        continue
                    # the positive control: this repository's own HEAD, asked for by hash
                    head = git("ls-remote", url, "HEAD")
                    if head.returncode != 0 or not head.stdout.strip():
                        self.skipTest(f"cannot reach {url}: {head.stderr.strip()[:120]}")
                    head_sha = head.stdout.split()[0]
                    if not serves_by_hash(scratch, url, head_sha):
                        self.skipTest(f"{url} serves no object by hash; {rev!r} unverifiable here")
                    if not serves_by_hash(scratch, url, rev):
                        problems.append(f"{name}: rev {rev!r} is not a commit {url} will serve")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.skipTest(f"cannot reach {url}: {exc}")
        self.assertEqual([], problems, "\n  ".join(problems))


class EveryEngineIsCreditedAndReachable(unittest.TestCase):
    def test_every_engine_is_credited_with_a_SUBSTANTIVE_attribution_record(self):
        """Key presence is not credit: `"swinunetr": {}` passed the first version. A record
        either carries the facts or points at the ecosystem entry that does."""
        data = json.loads((PKG / "data" / "attribution.json").read_text(encoding="utf-8"))
        engines, ecosystems = data.get("engines", {}), data.get("ecosystems", {})
        problems = []
        for name in R.ENGINES:
            rec = engines.get(name)
            if rec is None:
                problems.append(f"{name}: no attribution entry at all")
                continue
            alias = rec.get("same_as_ecosystem")
            if alias is not None:
                if alias not in ecosystems:
                    problems.append(f"{name}: points at ecosystem {alias!r}, which has no record")
            elif not ({"license", "cite"} <= set(rec)):
                problems.append(f"{name}: record has {sorted(rec) or 'nothing'} - needs at "
                                "least a licence and the citation its makers ask for, READ "
                                "from the project's own README/LICENSE")
        self.assertEqual([], problems, "\n  ".join(problems))

    def test_every_engine_is_reachable_from_an_ECOSYSTEM_THAT_ACTUALLY_SHIPS(self):
        """Reconciles two independent sources.

        The first version read `ECOSYSTEM_ENGINE`, the registry's own dict, so it proved
        only that the registry agreed with itself: an engine routed from an ecosystem name
        nothing ever constructs passed, advertising zero reachable tasks. The catalogs
        `default_ecosystems()` builds are the other source, and the two must agree.
        """
        ecosystems = pytest.importorskip("haversack.ecosystems")
        with mock.patch.dict("os.environ",
                             {e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}):
            built = ecosystems.default_ecosystems()
        by_engine = {}
        for eco in built:
            by_engine.setdefault(getattr(eco, "engine", R.NNUNETV2), []).append(eco.name)

        stranded = sorted(set(R.ENGINES) - set(by_engine))
        self.assertEqual([], stranded,
                         f"engines no shipped ecosystem runs: {stranded} - an ECOSYSTEM_ENGINE "
                         "entry alone is not enough; add the ecosystem class and list it in "
                         "ecosystems.default_ecosystems()")

        # and the registry's route must name one of the ecosystems that really exist
        wrong = [f"{eco}->{eng} (built ecosystems for {eng}: {by_engine.get(eng)})"
                 for eco, eng in R.ECOSYSTEM_ENGINE.items()
                 if eco not in {n for names in by_engine.values() for n in names}]
        self.assertEqual([], wrong,
                         "ECOSYSTEM_ENGINE routes from names no ecosystem provides: "
                         + "; ".join(wrong))


class EveryEngineCacheIsVisibleToCacheAdmin(unittest.TestCase):
    def test_every_declared_cache_store_is_reported_by_cache_admin(self):
        """`cache usage` and `cache clean` must see it, or the disk fills invisibly.

        The expectation is computed the way production computes it - override first, then
        the default under the cache root - rather than assumed to be the default. The
        first version unpacked the override variable and threw it away, so it asserted the
        default path unconditionally and FAILED for anyone who had the override set: a
        developer with pre-fetched checkpoints, or anyone whose shell carries what the
        Modal deploy path itself sets (`modal_app._fs_image`). `conftest`'s autouse fixture
        pins the engine ENABLE flags off and touches nothing else, so an ambient
        `HAVERSACK_*` store variable really does reach this test. It reported a cache-admin
        defect that did not exist (2026-09-08).
        """
        declaring = {n: e.cache_store for n, e in R.ENGINES.items() if e.cache_store}
        if not declaring:
            self.skipTest("no engine declares a cache store")
        from haversack.cache_admin import cache_root, stores
        for name, (sub, env_var) in declaring.items():
            for label, override in (("no override", None),
                                    ("absolute override", "/tmp/haversack-cache-store-probe"),
                                    ("tilde override", "~/haversack-cache-store-probe")):
                with self.subTest(engine=name, case=label):
                    with mock.patch.dict("os.environ", {}, clear=False):
                        if env_var and override is None:
                            os.environ.pop(env_var, None)
                        elif env_var:
                            os.environ[env_var] = override
                        expected = (Path(override).expanduser() if override and env_var
                                    else cache_root() / sub)
                        reported = {str(s["path"]) for s in stores()}
                        self.assertIn(str(expected), reported,
                                      f"{name}: {expected} is not among the stores cache admin "
                                      f"reports ({sorted(reported)})")

    def test_only_one_engine_declares_a_cache_store(self):
        """`cache_admin.clean` addresses ONE path per category, and `checkpoints` is a
        user-facing category name fixed by the CLI, so the single-path model holds only
        while a single engine declares a store. This is deliberately a tripwire rather
        than machinery for a case that does not exist: the day a second engine caches
        under the cache root, this fails and says what has to change."""
        declaring = sorted(n for n, e in R.ENGINES.items() if e.cache_store)
        self.assertLessEqual(len(declaring), 1,
                             f"{declaring} all declare a cache store, but "
                             "cache_admin.checkpoint_dir returns one path and clean() maps "
                             "one path per category - widen both to a store per engine, and "
                             "decide what `cache clean checkpoints` then means")


class TheModalDeploymentIsWiredForEveryEngine(unittest.TestCase):
    """Read as TEXT, never imported.

    The first version used `importorskip("modal")`, which deleted these obligations wherever
    modal is absent - the per-engine virtualenvs, and any sync without the modal extra, which
    is precisely where someone adding an engine works. Parsing costs nothing and always runs.
    """

    def setUp(self):
        self.tree = _modal_tree()
        self.classes = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef)}
        self.assigned = {t.id for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                         for t in n.targets if isinstance(t, ast.Name)}
        self.workers = self._worker_map()

    def _worker_map(self) -> dict:
        """`_WORKER_CLASSES` as {engine: class name}, read from the source."""
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "_WORKER_CLASSES" for t in node.targets)):
                continue
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    key = k.value
                elif isinstance(k, ast.Attribute):        # _engines.NNUNETV2
                    key = getattr(R, k.attr, k.attr)
                else:
                    continue
                out[key] = v.value if isinstance(v, ast.Constant) else None
            return out
        self.fail("no _WORKER_CLASSES assignment found in modal_app.py - this guard reads "
                  "it as text and can no longer see it")

    def test_every_engine_has_an_entry_in_the_worker_map(self):
        self.assertEqual([], sorted(set(R.ENGINES) - set(self.workers)),
                         f"engines missing from _WORKER_CLASSES: "
                         f"{sorted(set(R.ENGINES) - set(self.workers))}")

    def test_every_worker_map_entry_NAMES_A_CLASS_THAT_EXISTS(self):
        """The import-time assert compares keys only, so a typo in the VALUE dropped the
        engine out of dispatch in silence. `globals().get(name)` returned None and nothing
        was said; the engine was simply absent from every deploy."""
        bad = sorted(f"{eng} -> {cls!r}" for eng, cls in self.workers.items()
                     if cls not in self.classes)
        self.assertEqual([], bad,
                         f"_WORKER_CLASSES names classes modal_app does not define: {bad}")

    def test_every_engine_has_the_MODULE_FLAG_that_gates_its_worker(self):
        """`_worker_classes()` finds the flag by stripping HAVERSACK_ off `enabled_env` and
        reading module globals. A missing assignment - a line a thousand lines above the
        worker it gates - makes the lookup return its False default, so the engine is absent
        from every deploy while its variable is set to 1, and the error tells the caller to
        set the variable that is already set."""
        missing = []
        for name, e in R.ENGINES.items():
            if e.enabled_env is None:
                continue
            flag = e.enabled_env[len("HAVERSACK_"):]
            if flag not in self.assigned:
                missing.append(f"{name}: no module-level `{flag} = _engines.enabled({name!r})`")
        self.assertEqual([], missing, "\n  ".join(missing))

    def test_the_engine_enable_flags_are_FORWARDED_INTO_THE_CONTAINER(self):
        """Two independent sources again.

        The first version compared `engine_env_vars()` to the dict it is derived from, so it
        was true for any registry, and the claim in its name went untested: the spread could
        be deleted from `_RUNTIME_KNOBS` with the whole suite green. A knob that exists at
        deploy time but not inside the container is a bug this project has already hit.
        """
        spread = [n for n in ast.walk(self.tree) if isinstance(n, ast.Starred)
                  and isinstance(n.value, ast.Call)
                  and getattr(n.value.func, "attr", None) == "engine_env_vars"]
        self.assertTrue(spread,
                        "modal_app no longer spreads `engine_env_vars()` into _RUNTIME_KNOBS, "
                        "so an engine's enable flag never reaches the container: the engine "
                        "is listed and described, then refuses every job inside the worker")

    def test_the_import_time_wiring_check_agrees_when_modal_is_installed(self):
        pytest.importorskip("modal")
        from haversack import modal_app
        self.assertEqual([], modal_app._wiring_problems())


class TheCommandLineNamesEveryEngine(unittest.TestCase):
    def test_the_top_level_description_mentions_every_optional_engine(self):
        """`haversack --help` is where someone finds out what this can run.

        The sentence was hand-written and had gone two engines stale: VoxTell and the MONAI
        bundles had shipped for weeks without appearing, so the only place a new user looks
        said they did not exist. Prose is the right form here - the families are a mix of
        catalogs and engines and a generated list would read badly - but it has to be
        complete, so the check is that each optional engine's name appears somewhere in it.
        """
        import contextlib
        import io

        from haversack import cli
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
            cli.main(["--help"])               # argparse prints the description, then exits
        text = buf.getvalue()
        self.assertIn("haversack", text.lower(),
                      "`--help` produced nothing this guard can read")
        missing = [n for n in _optional_engines() if n.lower() not in text.lower()]
        self.assertEqual([], missing,
                         f"`haversack --help` never mentions {missing} - add them to the "
                         "top-level description, or nobody discovers the engine exists")


class TheVersionEndpointDerivesItsPackageList(unittest.TestCase):
    """The regression guard for the defect that motivated `Engine.dist`.

    Structural checks cannot catch this one: a hardcoded list naming the right packages
    today satisfies any assertion about the response. So a probe engine is added at runtime
    declaring a package that is certainly installed, and the endpoint is asked whether it
    reported it. Only a derived list can pass.
    """

    def _packages(self, client) -> dict:
        r = client.get("/v1/version")
        self.assertEqual(200, r.status_code, f"/v1/version answered {r.status_code}; this "
                                             "guard cannot say anything about a non-200")
        return r.json().get("packages", {})

    def test_an_engine_added_at_runtime_appears_in_the_package_report(self):
        pytest.importorskip("fastapi")
        import tempfile

        from test_serve import make

        probe = R.Engine(name="_probe", enabled_env="HAVERSACK_PROBE_ENGINE",
                         dist=("pytest",), description="probe")
        with tempfile.TemporaryDirectory() as td:
            _, ex, client = make(Path(td))
            try:
                before = self._packages(client)
                self.assertNotIn("pytest", before,
                                 "the probe package is already reported; pick one this "
                                 "project does not ship")
                with mock.patch.dict(R.ENGINES, {"_probe": probe}):
                    after = self._packages(client)
            finally:
                client.close()
                getattr(ex, "shutdown", lambda: None)()
        self.assertIn("pytest", after,
                      "adding an engine did not change /v1/version's package report, so the "
                      "list is not derived from Engine.dist - it is hardcoded again")

    def test_the_report_holds_NOTHING_BUT_the_registry_distributions(self):
        """Closes the hybrid: a stale hand-written list carried ALONGSIDE the derived one
        still satisfies the probe above, because the probe only proves the derived half
        runs. Every package reported must be traceable to some engine's `dist`."""
        pytest.importorskip("fastapi")
        import tempfile

        from test_serve import make

        known = {d for e in R.ENGINES.values() for d in e.dist}
        with tempfile.TemporaryDirectory() as td:
            _, ex, client = make(Path(td))
            try:
                reported = set(self._packages(client))
            finally:
                client.close()
                getattr(ex, "shutdown", lambda: None)()
        self.assertEqual(set(), reported - known,
                         f"/v1/version reports {sorted(reported - known)}, which no engine's "
                         "`dist` claims - a hand-written list is being merged in")


if __name__ == "__main__":
    unittest.main()
