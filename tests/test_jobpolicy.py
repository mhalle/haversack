"""The shared no-cache policy, and the guard that keeps it shared.

``LocalExecutor._refresh_input`` and the Modal worker's ``_refresh_series`` were
the same rule written twice in different vocabulary. A shingle comparison of
serve.py and modal_app.py finds exactly ONE shared five-line window between
them, so the duplication was invisible to search - the two stayed in step only
because someone remembered to change both. They did not always.

These tests cover the rule once, and then assert it still has only one home.
"""
import ast
import inspect
import pathlib
import unittest

from haversack.jobpolicy import refresh_cached_input

SRC = pathlib.Path(inspect.getsourcefile(__import__("haversack"))).parent


class _Cache:
    def __init__(self, present=True, discards=True):
        self._present, self._discards = present, discards
        self.discard_calls = []

    def has(self, key):
        return self._present

    def discard(self, key):
        self.discard_calls.append(key)
        return self._discards


class _ReadAhead:
    def __init__(self):
        self.popped = []

    def pop(self, key):
        self.popped.append(key)


class _Reporter:
    def __init__(self):
        self.stages = []

    def stage(self, name, text):
        self.stages.append((name, text))


def _run(**kw):
    kw.setdefault("wanted", True)
    kw.setdefault("cache", _Cache())
    kw.setdefault("read_ahead", _ReadAhead())
    kw.setdefault("reporter", _Reporter())
    skipped = []
    kw.setdefault("on_skipped", lambda: skipped.append(True))
    fresh = refresh_cached_input("s1", **kw)
    return fresh, kw["cache"], kw["read_ahead"], kw["reporter"], skipped


class RefreshPolicy(unittest.TestCase):

    def test_a_job_that_did_not_ask_is_left_alone(self):
        fresh, cache, ra, rep, skipped = _run(wanted=False)
        self.assertTrue(fresh)
        self.assertEqual((cache.discard_calls, ra.popped, rep.stages, skipped),
                         ([], [], [], []))

    def test_nothing_cached_means_the_fetch_itself_is_the_refresh(self):
        fresh, cache, ra, rep, skipped = _run(cache=_Cache(present=False))
        self.assertTrue(fresh)
        self.assertEqual(cache.discard_calls, [])
        self.assertEqual(ra.popped, ["s1"])   # a pre-read image would still be stale
        self.assertEqual(skipped, [])

    def test_a_successful_discard_drops_the_pre_read_image_too(self):
        """The half that is easy to forget: dropping the bytes alone re-downloads
        the series and then segments the pre-read image anyway, because the
        read-ahead is keyed by series and nothing else invalidates it."""
        fresh, cache, ra, rep, skipped = _run()
        self.assertTrue(fresh)
        self.assertEqual((cache.discard_calls, ra.popped), (["s1"], ["s1"]))
        self.assertEqual(rep.stages, [("fetch", "refetching (no-cache)")])
        self.assertEqual(skipped, [])

    def test_a_refused_discard_is_reported_and_leaves_the_read_ahead_alone(self):
        """The refusal means another job is still using those bytes. Its pre-read
        image belongs to that job, and the caller must be able to see that the
        refresh it asked for did not happen."""
        fresh, cache, ra, rep, skipped = _run(cache=_Cache(discards=False))
        self.assertFalse(fresh)
        self.assertEqual(ra.popped, [])
        self.assertEqual(skipped, [True])
        self.assertIn("could not refresh", rep.stages[0][1])

    def test_one_identifier_bound_to_two_roles_refreshes_once(self):
        """A multi-input task may bind two roles to ONE identifier. Without the
        dedup the second pass sees the first pass's fresh bytes cached under this
        job's own pin, and reports a skip for a key this very job just
        refreshed."""
        cache, ra, rep, seen, skipped = _Cache(), _ReadAhead(), _Reporter(), set(), []
        for _ in range(2):
            refresh_cached_input("s1", wanted=True, cache=cache, read_ahead=ra,
                                 reporter=rep, already=seen,
                                 on_skipped=lambda: skipped.append(True))
        self.assertEqual(cache.discard_calls, ["s1"])
        self.assertEqual(skipped, [])

    def test_a_worker_without_a_read_ahead_is_supported(self):
        """The Modal worker may have none; the policy must not assume one."""
        cache = _Cache()
        self.assertTrue(refresh_cached_input(
            "s1", wanted=True, cache=cache, read_ahead=None,
            reporter=_Reporter(), on_skipped=lambda: None))
        self.assertEqual(cache.discard_calls, ["s1"])


class PolicyHasOneHome(unittest.TestCase):
    """The drift guard. Text search could not see the old duplication, so these
    assert the structural property instead: the decision has exactly one call
    site, and both deployments reach it."""

    def test_only_jobpolicy_decides_to_discard_a_cached_input(self):
        callers = []
        for path in sorted(SRC.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "discard"):
                    callers.append(f"{path.name}:{node.lineno}")
        self.assertEqual([c.split(":")[0] for c in callers], ["jobpolicy.py"],
                         f"a second place decides to discard a cached input: {callers}")

    def test_both_deployments_delegate_rather_than_reimplement(self):
        for module in ("serve.py", "modal_app.py"):
            src = (SRC / module).read_text()
            self.assertIn("refresh_cached_input", src,
                          f"{module} no longer reaches the shared policy")


if __name__ == "__main__":
    unittest.main()
