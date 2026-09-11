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
import sys
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
        problems, unchecked = [], []
        with tempfile.TemporaryDirectory() as scratch:
            try:
                if git("init", "--quiet", scratch).returncode != 0:
                    self.skipTest("cannot create a scratch git repository")
            except OSError as exc:               # no git on PATH: skip, never error
                self.skipTest(f"git is not runnable here: {exc}")
            for name, spec in sources.items():
                url, tag, rev = spec.get("git"), spec.get("tag"), spec.get("rev")
                if not url or not (tag or rev):
                    continue
                # A source that cannot be reached is recorded and the loop CONTINUES.
                # These were `self.skipTest`, which aborts the whole test and discards
                # every problem found so far - so one deleted, renamed, private or
                # auth-gated repository turned the entire pin check into a no-op, and a
                # renamed upstream is one of the things a pin check is for. Reported at
                # the end, and only when nothing worse was found.
                try:
                    if tag:
                        out = git("ls-remote", url, tag, f"{tag}^{{}}")
                        if out.returncode != 0:
                            unchecked.append(f"{name}: cannot reach {url} "
                                             f"({out.stderr.strip()[:80]})")
                        elif not out.stdout.strip():
                            problems.append(f"{name}: tag {tag!r} is not published by {url}")
                    if not rev:
                        continue                 # a tag-only pin: checked above
                    # a pin carrying BOTH is checked twice; the rev is the one that
                    # actually resolves, and short-circuiting on the tag skipped it
                    head = git("ls-remote", url, "HEAD")
                    if head.returncode != 0 or not head.stdout.strip():
                        unchecked.append(f"{name}: cannot reach {url} "
                                         f"({head.stderr.strip()[:80]})")
                        continue
                    # the positive control: this repository's own HEAD, asked for by hash.
                    # A server that serves nothing by hash would fail a good pin the same
                    # way it fails a bad one.
                    if not serves_by_hash(scratch, url, head.stdout.split()[0]):
                        unchecked.append(f"{name}: {url} serves no object by hash")
                        continue
                    if not serves_by_hash(scratch, url, rev):
                        problems.append(f"{name}: rev {rev!r} is not a commit {url} will serve")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    unchecked.append(f"{name}: cannot reach {url} ({exc})")
        self.assertEqual([], problems, "\n  ".join(problems))
        if unchecked:
            self.skipTest("could not verify: " + "; ".join(unchecked))


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


    def test_THE_TWO_ROUTING_DECLARATIONS_AGREE(self):
        """The same fact is written in two places, and only one of them dispatches.

        `ECOSYSTEM_ENGINE` is what `engine_for`/`engine_for_task` read, so it decides
        which runtime a task actually runs on - in serve, cli, modal_app and segmenter.
        `ModelEcosystem.engine` is what `describe()` publishes to clients, and it defaults
        to the nnU-Net engine. An ecosystem that declares `engine="monai"` with no
        `ECOSYSTEM_ENGINE` entry passes every other rule in this file: it is reachable
        (the reachability test reads the class attribute), and every route names a real
        ecosystem (that test iterates the dict's own keys). Meanwhile `/v1/tasks/{task}`
        would answer monai while the job ran on nnU-Net.

        Deriving one from the other is not on offer: `ecosystems` imports `registry`, so
        the reverse import would be circular. Reconciling them is.
        """
        ecosystems = pytest.importorskip("haversack.ecosystems")
        with mock.patch.dict("os.environ",
                             {e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}):
            built = ecosystems.default_ecosystems()
        wrong = []
        for eco in built:
            declared = getattr(eco, "engine", R.NNUNETV2)
            routed = R.ECOSYSTEM_ENGINE.get(eco.name, R.NNUNETV2)
            if declared != routed:
                wrong.append(f"{eco.name}: describe() says {declared!r}, "
                             f"engine_for() routes to {routed!r}")
        self.assertEqual([], wrong,
                         "an ecosystem's declared engine and its dispatch route disagree:\n  "
                         + "\n  ".join(wrong)
                         + "\n  - add or correct the ECOSYSTEM_ENGINE entry in "
                           "engines/registry.py")


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

    def test_the_engine_ITSELF_reads_the_registry_for_its_store(self):
        """Not just cache admin: the engine's own downloads have to land where
        `cache usage` and `cache clean` look.

        The registry became authoritative for one side and not the other - the engine
        restated the subdirectory and the environment variable in its own module - so a
        change made through `Engine.cache_store` would have redirected the sweeping
        without redirecting what was swept, and the equality test below would still have
        passed because both literals still agreed. Moving the registry value here is the
        only way to tell those apart: it must move BOTH.
        """
        pytest.importorskip("haversack.engines.fastsurfer")
        from dataclasses import replace

        from haversack.cache_admin import stores
        from haversack.engines import fastsurfer

        declaring = sorted(n for n, e in R.ENGINES.items() if e.cache_store)
        self.assertEqual(["fastsurfer"], declaring,
                         "this test moves fastsurfer's store; the registry no longer says "
                         f"fastsurfer is the engine that declares one ({declaring})")

        # BOTH halves of `cache_store` are moved. Pinning the environment variable to its
        # real name made it a third hand-written copy of the fact and left the override
        # half untested, so restating `HAVERSACK_FASTSURFER_CHECKPOINTS` inside the engine
        # passed - and with that variable set, which is exactly what `modal_app._fs_image`
        # does, the engine downloaded to one directory while `cache clean` swept another.
        moved_sub, moved_env = "review-checkpoints-probe", "HAVERSACK_REVIEW_CKPT_PROBE"
        override = "/tmp/haversack-review-override-probe"
        with mock.patch.dict(R.ENGINES, {
                "fastsurfer": replace(R.ENGINES["fastsurfer"],
                                      cache_store=(moved_sub, moved_env))}):
            for label, env in (("registry default", None), ("registry override", override)):
                with self.subTest(case=label):
                    with mock.patch.dict("os.environ", {}, clear=False):
                        # the OLD variable stays set throughout: an engine still reading it
                        # must be caught, not accommodated
                        os.environ["HAVERSACK_FASTSURFER_CHECKPOINTS"] = "/tmp/the-old-name"
                        os.environ.pop(moved_env, None)
                        if env:
                            os.environ[moved_env] = env
                        engine_says = str(fastsurfer.checkpoint_dir())
                        admin_says = {s["name"]: str(s["path"]) for s in stores()}["checkpoints"]
                    self.assertEqual(engine_says, admin_says,
                                     "the engine and cache admin disagree about where "
                                     "checkpoints live")
                    self.assertNotIn("the-old-name", engine_says,
                                     "the engine is still reading the environment variable it "
                                     "used to restate, not the one the registry names")
                    if env:
                        self.assertEqual(env, engine_says,
                                         "the engine ignored the registry's override variable")
                    else:
                        self.assertTrue(engine_says.endswith(moved_sub),
                                        f"the engine answered {engine_says} - it is restating "
                                        "the subdirectory instead of reading `cache_store`")

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

    Reshaped on 2026-09-09, when each engine's image and worker moved into its own
    `engines/modal_<engine>.py` beside the runtime it deploys. Two of the four obligations
    here existed to catch failure modes that shape removes rather than guards: a missing
    module-level flag, and an entry naming a class that does not exist. `modal_app` composed
    its workers from a map of name STRINGS looked up in `globals()`, so either mistake
    dropped an engine from every deploy in silence. It now imports the adapter and takes its
    class object, so there is no name to mistype and no flag to forget.
    """

    def setUp(self):
        self.tree = _modal_tree()
        # AnnAssign as well as Assign: `ENGINE_WORKERS: dict = {...}` is annotated, and a
        # walk that sees only Assign reported the composer missing when it was right there
        self.assigned = {t.id for n in ast.walk(self.tree) if isinstance(n, ast.Assign)
                         for t in n.targets if isinstance(t, ast.Name)}
        self.assigned |= {n.target.id for n in ast.walk(self.tree)
                          if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}

    def _adapter(self, engine: str) -> Path:
        return PKG / "engines" / f"modal_{engine}.py"

    def test_every_engine_has_an_adapter_module_or_is_the_default(self):
        """The default engine's worker is `modal_app.Worker` - its image IS the base image
        and it is never optional. Every other engine needs a module of its own."""
        missing = sorted(n for n in _optional_engines() if not self._adapter(n).exists())
        self.assertEqual([], missing,
                         f"engines with no Modal adapter: {missing} - add "
                         f"src/haversack/engines/modal_<engine>.py, holding that engine's "
                         "image builder and its @app.cls worker")

    def test_every_adapter_DEFINES_A_WORKER_AND_SAYS_WHICH_ENGINE_IT_IS(self):
        """Three sources reconciled: the filename, the module's own `ENGINE`, and the
        registry key. The composer imports by filename and then checks `ENGINE`, so a
        module copied from a sibling and half-renamed fails at import rather than
        deploying the wrong engine's image under the right engine's name.

        `WORKER` must be a NAME BOUND TO A CLASS defined in the module, and that class must
        carry an `@app.cls` decoration - a module that defines a plain class and exports it
        registers nothing with Modal and would deploy as nothing at all.
        """
        problems = []
        for engine in sorted(_optional_engines()):
            path = self._adapter(engine)
            if not path.exists():
                continue                     # the test above owns that
            tree = ast.parse(path.read_text(encoding="utf-8"))
            declared = [n.value.value for n in tree.body
                        if isinstance(n, ast.Assign)
                        and any(getattr(t, "id", None) == "ENGINE" for t in n.targets)
                        and isinstance(n.value, ast.Constant)]
            if declared != [engine]:
                problems.append(f"{path.name}: ENGINE is {declared or None}, not {engine!r}")
            exported = [ast.unparse(n.value) for n in tree.body
                        if isinstance(n, ast.Assign)
                        and any(getattr(t, "id", None) == "WORKER" for t in n.targets)]
            if len(exported) != 1:
                problems.append(f"{path.name}: defines {len(exported)} WORKER assignments; "
                                "the composer takes exactly one")
                continue
            classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
            cls = classes.get(exported[0])
            if cls is None:
                problems.append(f"{path.name}: WORKER = {exported[0]}, which is not a class "
                                "defined in this module")
            elif not any("app.cls" in ast.unparse(d) for d in cls.decorator_list):
                problems.append(f"{path.name}: {exported[0]} carries no @app.cls decoration, "
                                "so Modal registers nothing for this engine")
        self.assertEqual([], problems, "\n  ".join(problems))

    def test_modal_app_COMPOSES_the_adapters_rather_than_naming_them(self):
        """The composer must import by the registry's own names.

        What this replaced was a literal map from engine to class-name string. Reading the
        registry means a new engine needs no edit here at all; hardcoding four imports would
        pass every other rule in this file and go stale exactly the way the old map did.
        """
        src = MODAL_APP.read_text(encoding="utf-8")
        self.assertIn("ENGINE_WORKERS", self.assigned,
                      "modal_app no longer builds ENGINE_WORKERS - the composer is gone")
        self.assertIn("haversack.engines.modal_", src,
                      "modal_app imports no adapter modules")
        self.assertNotIn("_WORKER_CLASSES", src,
                         "the old name-string worker map is back; the composer takes class "
                         "objects so that a typo cannot silently drop an engine")
        loops = [n for n in ast.walk(self.tree) if isinstance(n, ast.For)
                 and "ENGINES" in ast.unparse(n.iter)]
        self.assertTrue(loops,
                        "the composer does not iterate the registry, so adding an engine "
                        "needs an edit in modal_app after all")

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

    def _compose_after_importing(self, first: str) -> set:
        """Import `first` in a clean subprocess with every engine on, then report what
        `haversack.modal_app.ENGINE_WORKERS` ended up holding. A subprocess because the
        thing under test IS module import order, which cannot be undone in-process."""
        import json
        import subprocess
        import sys as _sys

        code = ("import importlib, json, sys; importlib.import_module(%r); "
                "print(json.dumps(sorted(sys.modules['haversack.modal_app'].ENGINE_WORKERS)))"
                % first)
        env = {**os.environ,
               **{e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}}
        root = Path(R.__file__).resolve().parent.parent.parent
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
        out = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=300, env=env)
        self.assertEqual(0, out.returncode,
                         f"importing {first} first failed:\n{out.stderr[-1200:]}")
        return set(json.loads(out.stdout.strip().splitlines()[-1]))

    def test_the_composer_works_WHICHEVER_MODULE_IS_ENTERED_FIRST(self):
        """Both entry points must end with the same complete map, and neither may raise.

        Modal uses BOTH. The api container imports `modal_app`; a WORKER container imports
        the module its class LIVES IN, which is now the adapter. The first colocation
        handled only the first order, so every engine worker crashed on start with
        `defines no WORKER` while the api container was perfectly healthy - the job simply
        stayed queued forever, with the failure visible only in the worker's own log. The
        composer skips an adapter that is mid-import (it is the importer, and defines its
        own class when it resumes) and each adapter registers itself on the way out.

        Found by deploying. No static check and no in-process import reached it, which is
        why this one spends a subprocess.
        """
        pytest.importorskip("modal")
        engines = set(R.ENGINES)
        for first in ("haversack.modal_app",
                      *(f"haversack.engines.modal_{n}" for n in sorted(_optional_engines()))):
            with self.subTest(first=first):
                self.assertEqual(engines, self._compose_after_importing(first))

    def test_the_composer_survives_being_loaded_BY_PATH(self):
        """`modal deploy src/haversack/modal_app.py` imports this file by PATH.

        That lands it in sys.modules under a synthetic name, leaving
        `haversack.modal_app` unclaimed - so each adapter's import of the canonical name
        executed the module a SECOND time, as a different object, re-entering the
        composer while the adapter was still half-initialized. Every static check passed
        and every ordinary import worked; the deploy failed on the first engine
        (2026-09-09). This reproduces the deploy's import mode in-process, because
        nothing else in the suite does.
        """
        pytest.importorskip("modal")
        import importlib.util

        with mock.patch.dict("os.environ",
                             {e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}):
            spec = importlib.util.spec_from_file_location("modal_app_by_path", MODAL_APP)
            mod = importlib.util.module_from_spec(spec)
            saved = {k: sys.modules[k] for k in list(sys.modules)
                     if k == "haversack.modal_app" or k.startswith("haversack.engines.modal_")}
            for k in saved:
                del sys.modules[k]
            sys.modules[spec.name] = mod
            try:
                spec.loader.exec_module(mod)
                composed = set(mod.ENGINE_WORKERS)
                same = sys.modules.get("haversack.modal_app") is mod
            finally:
                sys.modules.pop(spec.name, None)
                for k in list(sys.modules):
                    if k == "haversack.modal_app" or k.startswith("haversack.engines.modal_"):
                        del sys.modules[k]
                sys.modules.update(saved)
        self.assertEqual(set(R.ENGINES), composed,
                         f"loaded by path, the composer misses "
                         f"{sorted(set(R.ENGINES) - composed)}")
        self.assertTrue(same,
                        "`haversack.modal_app` is a SECOND module object here - the "
                        "adapters would import a fresh copy and re-enter the composer")

    def test_the_composed_deployment_covers_every_engine_when_modal_is_installed(self):
        """The one obligation that has to import: static text cannot prove that Modal
        accepted the decorators. Skips where modal is absent, which is why every rule
        above is written to run without it."""
        pytest.importorskip("modal")
        with mock.patch.dict("os.environ",
                             {e.enabled_env: "1" for e in R.ENGINES.values() if e.enabled_env}):
            import importlib

            from haversack import modal_app
            importlib.reload(modal_app)
            composed = set(modal_app.ENGINE_WORKERS)
        self.assertEqual(set(R.ENGINES), composed,
                         f"engines the deployment composes no worker for: "
                         f"{sorted(set(R.ENGINES) - composed)}")


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
            cli.main(["--help"])               # prints the description, then exits
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

    def test_an_ENABLED_engine_is_never_SILENTLY_absent_from_the_report(self):
        """An engine whose packages this process cannot see must say so, not vanish.

        `_pkg_info` reads the distributions installed in the API PROCESS, and on Modal the
        API runs in `api_image` while each optional engine's dependencies are installed only
        in that engine's own worker image - so for four of the five engines those versions
        can never appear there, however the deployment is configured. Dropping them made an
        enabled engine's absence look exactly like an engine nobody asked for, in the
        endpoint whose whole job is to say which build produced a result. This asserts the
        report ACCOUNTS for every enabled engine's distributions, with a version or with a
        reason.
        """
        pytest.importorskip("fastapi")
        import tempfile

        from test_serve import make

        # EVERY optional engine, each with its own uninstallable package name. Testing
        # one - whichever came first in registry order - let the branch be narrowed to
        # that engine (`elif remote_only[name] == "fastsurfer" and ...`) with the guard
        # still green and the other three back to vanishing silently; and it let the note
        # hardcode that engine's name, so the report could tell an operator that fastsurfer
        # holds monai's package. Distinct names per engine make both impossible.
        from dataclasses import replace
        optional = _optional_engines()
        self.assertGreater(len(optional), 1, "only one optional engine: this cannot bite")
        fake = {n: f"a-package-nobody-installed-{n}" for n in optional}
        patched = {n: replace(R.ENGINES[n], dist=(fake[n],)) for n in optional}
        with mock.patch.dict(R.ENGINES, patched):
            with mock.patch.dict("os.environ",
                                 {R.ENGINES[n].enabled_env: "1" for n in optional}):
                with tempfile.TemporaryDirectory() as td:
                    _, ex, client = make(Path(td))
                    try:
                        body = client.get("/v1/version").json()
                    finally:
                        client.close()
                        getattr(ex, "shutdown", lambda: None)()
        pkgs = body.get("packages") or {}
        for eng, name in sorted(fake.items()):
            with self.subTest(engine=eng):
                self.assertIn(name, pkgs,
                              f"{eng} is enabled and its distribution is missing from "
                              f"/v1/version entirely: {sorted(pkgs)} - report it as unknown "
                              "rather than dropping it")
                self.assertIsNone(pkgs[name]["version"])
                self.assertIn(eng, pkgs[name]["note"],
                              f"the note for {name} does not name {eng}, the engine whose "
                              f"environment holds it: {pkgs[name]['note']!r}")

    def test_a_DISABLED_engine_is_not_padded_into_the_report(self):
        """The inverse of the rule above: reporting an unknown regardless of enablement
        satisfies it just as well, and fills the report with engines nobody asked for."""
        pytest.importorskip("fastapi")
        import tempfile

        from dataclasses import replace

        from test_serve import make

        optional = _optional_engines()
        fake = {n: f"a-package-nobody-installed-{n}" for n in optional}
        patched = {n: replace(R.ENGINES[n], dist=(fake[n],)) for n in optional}
        with mock.patch.dict(R.ENGINES, patched):
            with mock.patch.dict("os.environ",
                                 {R.ENGINES[n].enabled_env: "0" for n in optional}):
                with tempfile.TemporaryDirectory() as td:
                    _, ex, client = make(Path(td))
                    try:
                        body = client.get("/v1/version").json()
                    finally:
                        client.close()
                        getattr(ex, "shutdown", lambda: None)()
        pkgs = body.get("packages") or {}
        padded = sorted(n for n in fake.values() if n in pkgs)
        self.assertEqual([], padded,
                         f"/v1/version lists {padded} for engines this deployment has "
                         "switched off - an unknown is only worth saying about an engine "
                         "that is meant to be running")


if __name__ == "__main__":
    unittest.main()
