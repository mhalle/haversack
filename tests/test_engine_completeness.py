"""Every engine must reach every place that has to know about it.

The registry is deliberately a static, closed set rather than a plugin system - see
:mod:`haversack.engines.registry` for the three properties of this system that make
discovered plugins impossible to do honestly. The cost of that choice is a checklist,
and a checklist in prose goes stale: the working notes claimed adding an engine was
"one row plus a Modal worker class" while the measured surface was four times that,
and `/v1/version` had hand-listed engine packages since before voxtell and monai
existed, so the endpoint whose job is to say WHICH REVISION is running said nothing
about the one engine pinned to a git rev for exactly that reason.

So the checklist lives here instead, one obligation per test, each failing with the
step it found missing. Add an engine, run this, and it names what is left to do.

These are structural, not behavioural: they prove a fact was declared, never that the
engine segments anything. The exception is the version endpoint, which is checked by
adding a probe engine at runtime and looking for its package in the response - a test
that a hardcoded list cannot pass.
"""
from __future__ import annotations

import json
import re
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import pytest

from haversack.engines import registry as R

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _requirement_names(specs) -> set[str]:
    """``"monai>=1.4"`` -> ``"monai"``, for a list of PEP 508 strings."""
    return {re.split(r"[<>=!~;\[ ]", str(s).strip())[0] for s in specs}


def _pyproject() -> dict:
    if not PYPROJECT.exists():
        pytest.skip("running against an installed copy, not the repository")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


class EveryEngineReachesEveryPlaceThatMustKnowIt(unittest.TestCase):
    def test_the_registry_is_not_empty_and_names_its_own_keys(self):
        """The floor under every other test here: a guard that walks an empty dict
        passes vacuously, and one that walks a dict whose keys and rows disagree
        checks the wrong engine."""
        self.assertGreaterEqual(len(R.ENGINES), 2, "the registry has no engines to check")
        mismatched = [k for k, e in R.ENGINES.items() if e.name != k]
        self.assertEqual([], mismatched, f"key != Engine.name for {mismatched}")

    def test_every_engine_declares_the_distributions_it_installs(self):
        """`Engine.dist` feeds `/v1/version`. Empty means that deployment cannot say
        which build of the engine produced a result."""
        missing = [n for n, e in R.ENGINES.items() if not e.dist]
        self.assertEqual([], missing,
                         f"engines with no `dist`: {missing} - add the distribution "
                         "name(s) to the row so /v1/version can report the rev")

    def test_every_engine_distribution_is_declared_in_pyproject(self):
        """At least one of an engine's distributions must be a package this project
        actually installs, in its extra or in core. Transitives are allowed to appear
        in `dist` without being declared (surfa arrives through synthstrip-torch and
        is reported because a change in it moves the output grid), so this asks for
        one anchor rather than for every name - enough to catch a typo, which would
        otherwise show up only as a silently absent package in a live deployment."""
        data = _pyproject()
        core = _requirement_names(data["project"]["dependencies"])
        extras = {k: _requirement_names(v)
                  for k, v in data["project"]["optional-dependencies"].items()}
        unanchored = []
        for name, e in R.ENGINES.items():
            declared = core | extras.get(e.extra or "", set())
            if not (set(e.dist) & declared):
                unanchored.append(f"{name}: dist={e.dist} matches nothing in "
                                  f"core or extra {e.extra!r}")
        self.assertEqual([], unanchored, "\n  ".join(unanchored))

    def test_every_engine_extra_is_a_real_extra(self):
        data = _pyproject()
        extras = set(data["project"]["optional-dependencies"])
        bad = [f"{n}: extra {e.extra!r}" for n, e in R.ENGINES.items()
               if e.extra is not None and e.extra not in extras]
        self.assertEqual([], bad, f"engines naming an extra pyproject does not define: {bad}")

    def test_every_engine_is_credited_in_attribution(self):
        """`data/attribution.json` carries the licence and the papers each engine's
        makers ask for. An engine missing from it ships uncredited."""
        path = Path(R.__file__).resolve().parent.parent / "data" / "attribution.json"
        engines = json.loads(path.read_text(encoding="utf-8")).get("engines", {})
        missing = sorted(set(R.ENGINES) - set(engines))
        self.assertEqual([], missing,
                         f"engines with no attribution entry: {missing} - add licence, "
                         "group and citations, READ from the project's own README/LICENSE")

    def test_every_engine_enable_flag_is_forwarded_into_containers(self):
        """A knob that exists at deploy time but not inside the container is a bug this
        project has already hit; `engine_env_vars` is derived so it cannot be missed.
        This proves the derivation still covers every engine that has a flag."""
        forwarded = set(R.engine_env_vars())
        missing = sorted({e.enabled_env for e in R.ENGINES.values()
                          if e.enabled_env and e.enabled_env not in forwarded})
        self.assertEqual([], missing, f"enable flags not forwarded: {missing}")

    def test_every_non_default_engine_is_reachable_from_an_ecosystem(self):
        """An engine no ecosystem routes to cannot be asked for: `engine_for_task`
        reads the ecosystem prefix. The default engine needs no entry - every
        ecosystem without one falls through to it."""
        routed = set(R.ECOSYSTEM_ENGINE.values()) | {R.NNUNETV2}
        stranded = sorted(set(R.ENGINES) - routed)
        self.assertEqual([], stranded,
                         f"engines no ecosystem routes to: {stranded} - add an "
                         "ECOSYSTEM_ENGINE entry, or the tasks are unreachable")

    def test_every_engine_has_a_modal_worker_class(self):
        """modal_app asserts this at import; restated here so the failure names the
        engine while the suite runs, rather than only when a deploy imports the app."""
        pytest.importorskip("modal")
        from haversack import modal_app
        missing = sorted(set(R.ENGINES) - set(modal_app._WORKER_CLASSES))
        self.assertEqual([], missing,
                         f"engines with no Modal worker class: {missing}")


class TheVersionEndpointDerivesItsPackageList(unittest.TestCase):
    """The regression guard for the defect that motivated `Engine.dist`.

    Structural checks cannot catch this one: a hardcoded list that happens to name
    the right packages today satisfies any assertion about the response's contents.
    So a probe engine is added to the registry at runtime, declaring a distribution
    that is certainly installed, and the endpoint is asked whether it reported it.
    Only a derived list can pass.
    """

    def test_an_engine_added_at_runtime_appears_in_the_package_report(self):
        pytest.importorskip("fastapi")
        from test_serve import make  # the fake-segmenter client used by the serve suite

        probe = R.Engine(name="_probe", enabled_env="HAVERSACK_PROBE_ENGINE",
                         dist=("pytest",), description="probe")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            _, _, client = make(Path(td))
            before = client.get("/v1/version").json().get("packages", {})
            self.assertNotIn("pytest", before, "the probe package is already reported; "
                                               "pick one this project does not ship")
            with mock.patch.dict(R.ENGINES, {"_probe": probe}):
                after = client.get("/v1/version").json().get("packages", {})
        self.assertIn("pytest", after,
                      "adding an engine did not change /v1/version's package report, so "
                      "the list is not derived from Engine.dist - it is hardcoded again")


if __name__ == "__main__":
    unittest.main()
