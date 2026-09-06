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
import os
import pathlib
import tempfile
import threading
import time
import unittest

from haversack.jobpolicy import fill_read_ahead, refresh_cached_input

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


class PreReadPin(unittest.TestCase):
    """A committed entry is unpinned between the writer releasing its claim and
    the job that wants it taking one - and the read-ahead reads in exactly that
    window. Both deployments used to read there without a pin."""

    def _cache(self, root):
        from haversack.serve import SeriesCache

        def fetch(series, dest):
            # the cache addresses an entry's `series/` subdirectory, and a real
            # series is a DIRECTORY of DICOM files that the reader walks one by one
            (dest / "series").mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (dest / "series" / f"{i:04d}.dcm").write_bytes(b"x" * 64)
        return SeriesCache(root, fetch)

    def test_a_concurrent_discard_refuses_while_the_pre_read_is_running(self):
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            self.assertTrue(cache.prefetch("idc:abc"))
            seen = {}

            class _ReadingReadAhead:
                def fill(self, key, path):
                    # what a no-cache job on the same series does, mid-read
                    seen["discard_won"] = cache.discard(key)
                    seen["files"] = len(list(pathlib.Path(path).iterdir()))
                    return True

            fill_read_ahead("idc:abc", cache=cache, read_ahead=_ReadingReadAhead())
            self.assertFalse(seen["discard_won"],
                             "discard deleted the series while it was being read")
            self.assertEqual(seen["files"], 3, "the series vanished mid-read")

    def test_the_pin_is_released_afterwards(self):
        """Held forever it would be a leak that makes every later no-cache fail."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            cache.prefetch("idc:abc")

            class _RA:
                def fill(self, key, path):
                    return True

            fill_read_ahead("idc:abc", cache=cache, read_ahead=_RA())
            self.assertTrue(cache.discard("idc:abc"))

    def test_the_pin_survives_a_failing_read(self):
        """A read that raises must still unpin, or one bad input wedges the key."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            cache.prefetch("idc:abc")

            class _Boom:
                def fill(self, key, path):
                    raise RuntimeError("unreadable")

            with self.assertRaises(RuntimeError):
                fill_read_ahead("idc:abc", cache=cache, read_ahead=_Boom())
            self.assertTrue(cache.discard("idc:abc"), "the pin leaked")

    def test_local_bytes_need_no_pin(self):
        """An upload already sits on local disk; nothing can evict it."""
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "up.nii.gz"
            f.write_bytes(b"z")
            got = {}

            class _RA:
                def fill(self, key, path):
                    got["path"] = path
                    return True

            self.assertTrue(fill_read_ahead("j1", read_ahead=_RA(), path=f))
            self.assertEqual(got["path"], f)


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

    def test_only_jobpolicy_fills_the_read_ahead(self):
        """The pin that makes a pre-read safe lives in fill_read_ahead. A call
        straight to read_ahead.fill is the unpinned window coming back.

        Matched on the RECEIVER'S SOURCE, not on a bare Name: the regression this
        guards was spelled `self.read_ahead.fill(...)`, and an earlier version of
        this test - which only matched `ast.Name` receivers - passed cleanly
        against the very code containing it.
        """
        callers = []
        for path in sorted(SRC.glob("*.py")):
            if path.name == "jobpolicy.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "fill"
                        and ast.unparse(node.func.value).endswith("read_ahead")):
                    callers.append(f"{path.name}:{node.lineno}")
        self.assertEqual(callers, [], f"an unpinned pre-read came back: {callers}")

    def test_that_guard_catches_the_regression_it_was_written_for(self):
        """Pin the guard to the code it exists to reject: main's serve.py, which
        contains two unpinned `self.read_ahead.fill(...)` calls."""
        import subprocess
        old = subprocess.run(["git", "show", "main:src/haversack/serve.py"],
                             capture_output=True, text=True,
                             cwd=SRC.parents[1]).stdout
        if not old:
            self.skipTest("no `main` to compare against")
        found = [n.lineno for n in ast.walk(ast.parse(old))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "fill"
                 and ast.unparse(n.func.value).endswith("read_ahead")]
        self.assertTrue(found, "the guard cannot see the bug it was written for")

    def test_both_deployments_delegate_rather_than_reimplement(self):
        """Structural, not a substring: the wrapper's BODY must call the shared
        function. `assertIn("refresh_cached_input", src)` was satisfied by the
        docstrings that merely cite it."""
        for module, fname in (("serve.py", "_refresh_input"),
                              ("modal_app.py", "_refresh_series")):
            tree = ast.parse((SRC / module).read_text())
            fn = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == fname), None)
            self.assertIsNotNone(fn, f"{module}: {fname} is gone")
            calls = {ast.unparse(n.func) for n in ast.walk(fn)
                     if isinstance(n, ast.Call)}
            self.assertIn("refresh_cached_input", calls,
                          f"{module}:{fname} stopped delegating")

class RetentionAgreesAcrossSubstrates(unittest.TestCase):
    """One retention rule, two substrates that cannot share code.

    The Modal deployment evaluates `purgeable` over records loaded from its job
    dict; the local server decides in SQL over rows it never loads. `reap`'s
    docstring used to simply assert the two agreed - "the same policy Modal's
    jobs store already runs" - and nothing checked it. This drives both over the
    same cases instead.
    """

    CASES = [
        ("done, long past the ttl",        {"state": "done", "finished": -7200}, True),
        ("failed, long past the ttl",      {"state": "failed", "finished": -7200}, True),
        ("cancelled, long past the ttl",   {"state": "cancelled", "finished": -7200}, True),
        ("done, still inside the ttl",     {"state": "done", "finished": -60}, False),
        ("done, exactly at the boundary",  {"state": "done", "finished": -3600}, False),
        ("queued since forever",           {"state": "queued", "created": -10 ** 5}, False),
        ("running since forever",          {"state": "running", "created": -10 ** 5,
                                    "started": -10 ** 5}, False),
        ("terminal with no finished stamp", {"state": "done", "created": -7200}, True),
    ]
    TTL = 3600.0

    def test_sql_and_python_reach_the_same_verdict(self):
        from haversack.jobpolicy import purgeable
        from haversack.jobstore import JobStore

        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as td:
            store = JobStore(pathlib.Path(td) / "jobs.db")
            expected = {}
            for i, (label, fields, want) in enumerate(self.CASES):
                jid = f"j{i}"
                rec = {"id": jid, "task": "ts:total", "kind": "segment",
                       "cache_key": None, "state": fields["state"],
                       "created": now + fields.get("created", -10),
                       "started": None,
                       "finished": (now + fields["finished"]
                                    if "finished" in fields else None)}
                store.put(rec)
                meta = {k: v for k, v in rec.items() if v is not None}
                self.assertEqual(purgeable(meta, now, self.TTL), want,
                                 f"python verdict wrong for {label}")
                expected[jid] = want

            reaped = set(store.reap(self.TTL, now=now))
            for jid, want in expected.items():
                label = self.CASES[int(jid[1:])][0]
                self.assertEqual(jid in reaped, want,
                                 f"sql and python disagree on {label}")

    def test_garbage_that_is_not_a_record_is_purgeable(self):
        from haversack.jobpolicy import purgeable
        self.assertTrue(purgeable(None, 1000.0, 60.0))
        self.assertTrue(purgeable("not a dict", 1000.0, 60.0))


class TerminalStatesHaveOneDefinition(unittest.TestCase):
    """They had five: serve.TERMINAL, jobstore.TERMINAL, modal_app._TERMINAL, an
    inline literal inside _purgeable, and a local in client.wait. A sixth state
    would have needed five edits, and a missed one fails silently and
    asymmetrically - a client polling forever, a purge that never collects."""

    def test_nothing_else_spells_the_set_out(self):
        spellings = []
        for path in sorted(SRC.glob("*.py")):
            if path.name == "jobpolicy.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                    vals = [e.value for e in node.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    if set(vals) == {"done", "failed", "cancelled"}:
                        spellings.append(f"{path.name}:{node.lineno}")
        self.assertEqual(spellings, [],
                         f"the terminal set is written again at {spellings}")

    def test_every_module_that_needs_them_gets_the_same_object(self):
        from haversack import jobpolicy, jobstore, serve
        from haversack.modal_app import _TERMINAL
        self.assertIs(serve.TERMINAL, jobpolicy.TERMINAL)
        self.assertIs(jobstore.TERMINAL, jobpolicy.TERMINAL)
        self.assertIs(_TERMINAL, jobpolicy.TERMINAL)


class SourceKeysAreDerivedOnce(unittest.TestCase):
    """Six places built the series-cache key by hand - three in serve, three in
    modal_app - and the prefetcher had to agree with the dispatcher on every one
    or it warmed a slot nothing looked up. That miss is silent: no error, just a
    download paid for twice."""

    def test_the_shapes_a_source_can_take(self):
        from haversack.jobpolicy import source_cache_key as k
        self.assertIsNone(k({"kind": "upload"}))
        self.assertIsNone(k({"kind": "input", "id": "sha256:ab"}))
        self.assertIsNone(k(None))                   # missing source == upload
        self.assertIsNone(k({}))

        sk = k({"kind": "idc", "crdc_series_uuid": "abc-123"})
        self.assertEqual((sk.kind, sk.ident, sk.key), ("idc", "abc-123", "idc:abc-123"))
        sk = k({"kind": "s3", "id": "fcp-indi/x.nii.gz"})
        self.assertEqual(sk.key, "s3:fcp-indi/x.nii.gz")

    def test_a_source_with_no_identifier_does_not_become_the_string_None(self):
        """The single-input paths used to build "idc:None" from an absent id
        while the multi-input paths built "idc:" for the same source. Two
        spellings of one derivation, disagreeing on the case that matters."""
        from haversack.jobpolicy import source_cache_key
        sk = source_cache_key({"kind": "idc"})
        self.assertEqual(sk.ident, "")
        self.assertNotIn("None", sk.key)

    def test_every_substrate_derives_its_keys_through_the_helper(self):
        """The literal check below only catches a copy-paste of the deleted line;
        `kind + ":" + ident` sails through it. This counts the call sites
        instead, so a hand-rolled derivation shows up as a MISSING delegation."""
        for module, least in (("serve.py", 3), ("modal_app.py", 3)):
            tree = ast.parse((SRC / module).read_text())
            n = sum(1 for x in ast.walk(tree)
                    if isinstance(x, ast.Call)
                    and ast.unparse(x.func).endswith("source_cache_key"))
            self.assertGreaterEqual(n, least,
                                    f"{module} derives only {n} keys through the "
                                    f"helper; expected at least {least}")

    def test_nothing_builds_the_key_by_hand_any_more(self):
        # Only the two job substrates: sources.py builds a string of the same
        # shape for a progress message and a .done marker, which is a different
        # thing that happens to read alike.
        pattern = '{kind}:{ident}'
        offenders = []
        for path in (SRC / "serve.py", SRC / "modal_app.py"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern in line:
                    offenders.append(f"{path.name}:{i}")
        self.assertEqual(offenders, [],
                         f"a hand-built cache key came back at {offenders}")


class GraveyardSweep(unittest.TestCase):
    """Debris is crash-only - a process dying between the rename into the
    graveyard and the delete. Sweeping only at construction therefore cleaned up
    after the LAST crash and never again, so a server that stays up for weeks
    accumulated its own."""

    def _cache(self, root):
        from haversack.serve import SeriesCache
        return SeriesCache(root, lambda series, dest: None)

    def test_eviction_re_sweeps_the_graveyard_after_the_interval(self):
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            debris = cache.graveyard / "idc%3Aabc-999-123"
            debris.mkdir(parents=True)
            (debris / "0000.dcm").write_bytes(b"x")

            cache._evict(keep=set())
            self.assertTrue(debris.exists(), "swept before the interval elapsed")

            cache._last_sweep -= cache.SWEEP_INTERVAL + 1
            cache._evict(keep=set())
            self.assertFalse(debris.exists(), "a long-lived server never re-sweeps")

    def test_the_sweep_never_reaches_outside_the_graveyard(self):
        """Entry names come from cache keys, so no pattern in that namespace can
        be trusted to mean 'discarded' - `s3:b/x.stale1` is a legal object. Two
        bugs came from matching names there."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            # sweep_stale returns immediately when there is no graveyard, so
            # without this the body never ran - and a sweep that also globbed
            # `*.stale*` in the LIVE root passed this test untouched.
            cache.graveyard.mkdir(parents=True, exist_ok=True)
            (cache.graveyard / "debris").mkdir()
            live = cache.root / "s3%3Ab%2Fx.stale1"
            live.mkdir(parents=True)
            (live / ".done").write_text("")
            cache._last_sweep -= cache.SWEEP_INTERVAL + 1
            cache._evict(keep=set())
            self.assertTrue(live.exists(), "the sweep deleted a live entry")
            self.assertFalse((cache.graveyard / "debris").exists(),
                             "the sweep never ran, so this proved nothing")


class ClaimIsAtomicWithItsIdentity(unittest.TestCase):
    """The property the linked claim buys, under contention rather than by
    inspection: many threads racing one key, exactly one writer, and never a
    moment where a claim exists without naming someone."""

    def test_one_writer_wins_and_the_claim_is_never_anonymous(self):
        import threading
        from haversack.serve import SeriesCache

        with tempfile.TemporaryDirectory() as td:
            cache = SeriesCache(pathlib.Path(td), lambda s, e: e / "series")
            entry = cache._entry("s3:b/x")
            tokens, anonymous, stop = [], [], threading.Event()

            def watcher():
                """Poll the claim throughout the race. If claiming were still two
                operations this is what would catch the window."""
                while not stop.is_set():
                    # `_owner_of` reports an existing-but-empty claim as
                    # "<establishing>", never None, so THAT is what an anonymous
                    # claim looks like. Testing for None could only fire on a read
                    # racing an unlink - an assertion that could not fail.
                    if (entry / cache.CLAIM).exists():
                        if cache._owner_of(entry) in (None, "<establishing>"):
                            anonymous.append(True)

            w = threading.Thread(target=watcher, daemon=True)
            w.start()
            barrier = threading.Barrier(8)

            def claimer():
                barrier.wait()
                tokens.append(cache._claim(entry))

            threads = [threading.Thread(target=claimer) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            stop.set()
            w.join(timeout=2)

            won = [t for t in tokens if t is not None]
            self.assertEqual(len(won), 1, f"{len(won)} writers claimed one series")
            self.assertEqual(cache._owner_of(entry), won[0])
            self.assertEqual(anonymous, [], "a claim existed without an owner")

    def test_a_committed_entry_is_never_re_claimed(self):
        """A writer racing a commit must not re-fetch over finished bytes."""
        from haversack.serve import SeriesCache

        with tempfile.TemporaryDirectory() as td:
            def fetch(series, entry):
                out = entry / "series"
                out.mkdir(parents=True, exist_ok=True)
                (out / "a.bin").write_bytes(b"ok")
                return out

            cache = SeriesCache(pathlib.Path(td), fetch)
            cache.get_or_fetch("s3:b/x")
            entry = cache._entry("s3:b/x")
            (entry / cache.CLAIM).unlink()      # an entry committed with no claim left
            self.assertIsNone(cache._claim(entry),
                              "claimed an already-committed entry")



class ClaimStatesThatWereUntested(unittest.TestCase):
    """The paths an adversarial review found uncovered: a claim that exists but
    cannot be read, a genuinely crashed claim, and the commit-time ownership
    check this branch changed from `pass` to `raise`."""

    def _cache(self, root, **kw):
        from haversack.serve import SeriesCache

        def fetch(series, entry):
            out = entry / "series"
            out.mkdir(parents=True, exist_ok=True)
            (out / "a.bin").write_bytes(b"fresh")
            return out
        return SeriesCache(root, fetch, **kw)

    def test_an_unreadable_claim_is_not_reported_as_unclaimed(self):
        """`_owner_of` returned None on ANY OSError, so a claim that exists but
        cannot be read read as "no claim". The waiter's `break` then skipped its
        sleep and retook the claim immediately: 6399 attempts and 95% of one
        core, on the single dispatcher thread, indefinitely.

        The trigger is ordinary. `_claim` writes the token with Path.write_text,
        so under umask 077 the claim is mode 0600 and os.link preserves it - any
        second uid on a shared cache root hits this.
        """
        if os.geteuid() == 0:
            self.skipTest("root can read a 0000 file")
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            entry = cache._entry("s3:b/x")
            token = cache._claim(entry)
            self.assertEqual(cache._owner_of(entry), token)
            claim = entry / cache.CLAIM
            claim.chmod(0o000)
            try:
                self.assertIsNotNone(cache._owner_of(entry),
                                     "an unreadable claim read as unclaimed")
            finally:
                claim.chmod(0o600)

    def test_waiting_on_an_unreadable_claim_does_not_spin(self):
        """The same defect, end to end: the wait loop must not retake the claim
        in a tight loop when it cannot read the owner."""
        if os.geteuid() == 0:
            self.skipTest("root can read a 0000 file")
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td), claim_timeout=30.0)
            entry = cache._entry("s3:b/x")
            cache._claim(entry)                       # a live writer holds it
            (entry / cache.CLAIM).chmod(0o000)
            attempts, real = [], cache._claim

            def counting(e, **kw):
                attempts.append(1)
                return real(e, **kw)

            cache._claim = counting
            stop = threading.Event()
            t = threading.Thread(
                target=lambda: cache.get_or_fetch("s3:b/x", check=lambda: (
                    stop.is_set() and (_ for _ in ()).throw(RuntimeError("stop")))),
                daemon=True)
            t.start()
            time.sleep(1.0)
            stop.set()
            n = len(attempts)
            (entry / cache.CLAIM).chmod(0o600)
            self.assertLess(n, 50, f"{n} claim attempts in one second: spinning")

    def test_a_crashed_claim_is_reclaimed_and_the_graveyard_left_empty(self):
        """A REAL crashed claim - a .owner with no .done and no live writer.
        Nothing in the suite constructed one, so both "delete in place instead of
        renaming aside" and "never reclaim at all" survived mutation."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td), claim_timeout=0.3)
            entry = cache._entry("s3:b/x")
            entry.mkdir(parents=True)
            (entry / cache.CLAIM).write_text("ghost-writer-token")
            (entry / "half.bin").write_bytes(b"partial")
            time.sleep(0.6)                            # older than claim_timeout

            out = cache.get_or_fetch("s3:b/x")
            self.assertEqual((out / "a.bin").read_bytes(), b"fresh")
            self.assertFalse((entry / "half.bin").exists(),
                             "the dead writer's bytes survived into the new entry")
            self.assertEqual(list(cache.graveyard.iterdir()), [],
                             "the graveyard was not emptied")

    def test_a_reclaim_never_deletes_the_live_entry_name_in_place(self):
        """The reclaim renames the dead entry aside and deletes the RENAMED copy.

        Deleting `entry` in place instead looks identical afterwards - which is
        why an earlier version of the crashed-claim test above could not tell the
        two apart - but the delete is not instant. A successor that claims the
        freed name while an in-place rmtree is still walking it has its fresh
        files deleted underneath it by that walk. The rename releases the name
        atomically, so this asserts on WHAT gets deleted, not on the aftermath.
        """
        import shutil as _sh

        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td), claim_timeout=0.3)
            entry = cache._entry("s3:b/x")
            entry.mkdir(parents=True)
            (entry / cache.CLAIM).write_text("ghost-writer-token")
            time.sleep(0.6)

            deleted, real_rmtree = [], _sh.rmtree

            def recording(path, *a, **kw):
                deleted.append(pathlib.Path(path))
                return real_rmtree(path, *a, **kw)

            _sh.rmtree = recording
            try:
                cache.get_or_fetch("s3:b/x")
            finally:
                _sh.rmtree = real_rmtree

            self.assertTrue(deleted, "nothing was reclaimed at all")
            for path in deleted:
                self.assertNotEqual(path, entry,
                                    "the live entry name was rmtree'd in place")
                self.assertEqual(path.parent, cache.graveyard,
                                 f"reclaim deleted {path}, outside the graveyard")

    def test_a_claim_taken_over_a_dirty_directory_starts_clean(self):
        """Bytes in an entry with no marker are an attempt that never finished.
        Adopting them silently produced a MIXED series - two fetches in one
        entry, committed as complete."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            entry = cache._entry("s3:b/x")
            entry.mkdir(parents=True)
            (entry / "series").mkdir()
            (entry / "series" / "stale.bin").write_bytes(b"another writer's slice")

            token = cache._claim(entry)
            self.assertIsNotNone(token)
            self.assertEqual([p.name for p in entry.iterdir()], [cache.CLAIM],
                             "untrusted content survived the claim")

    def test_commit_refuses_when_the_claim_is_gone_or_changed(self):
        """This branch changed _commit from `pass` to `raise` on an unreadable
        claim, and nothing anywhere tested either branch."""
        from haversack.errors import ResourceError

        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            entry = cache._entry("s3:b/x")
            token = cache._claim(entry)
            (entry / "series").mkdir()
            with self.assertRaises(ResourceError):      # reclaimed by a successor
                (entry / cache.CLAIM).write_text("someone-else")
                cache._commit(entry, key="s3:b/x", token=token)
            with self.assertRaises(ResourceError):      # claim gone entirely
                (entry / cache.CLAIM).unlink()
                cache._commit(entry, key="s3:b/x", token=token)
            self.assertFalse((entry / cache.MARKER).exists(),
                             "committed a series whose claim it did not hold")

    def test_the_no_hard_link_fallback_still_admits_exactly_one_writer(self):
        """os.link never fails on APFS, so this branch runs only on FAT32 or a
        network mount - that is, only in production."""
        with tempfile.TemporaryDirectory() as td:
            cache = self._cache(pathlib.Path(td))
            entry = cache._entry("s3:b/x")
            real_link = os.link

            def no_links(src, dst):
                raise OSError(95, "Operation not supported")

            os.link = no_links
            try:
                tokens = []
                barrier = threading.Barrier(6)

                def claimer():
                    barrier.wait()
                    tokens.append(cache._claim(entry))

                ts = [threading.Thread(target=claimer) for _ in range(6)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            finally:
                os.link = real_link
            won = [t for t in tokens if t is not None]
            self.assertEqual(len(won), 1, f"{len(won)} writers won via the fallback")
            self.assertEqual(cache._owner_of(entry), won[0])
            self.assertEqual(list(cache.claims.iterdir()), [],
                             "the fallback leaked a staged token")


if __name__ == "__main__":
    unittest.main()
