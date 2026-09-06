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
import tempfile
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
        straight to read_ahead.fill is the unpinned window coming back."""
        callers = []
        for path in sorted(SRC.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "fill"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id in ("read_ahead", "_read_ahead")):
                    callers.append(f"{path.name}:{node.lineno}")
        self.assertEqual(sorted({c.split(":")[0] for c in callers}), ["jobpolicy.py"],
                         f"an unpinned pre-read came back: {callers}")

    def test_both_deployments_delegate_rather_than_reimplement(self):
        for module in ("serve.py", "modal_app.py"):
            src = (SRC / module).read_text()
            self.assertIn("refresh_cached_input", src,
                          f"{module} no longer reaches the shared policy")


if __name__ == "__main__":
    unittest.main()


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
            live = cache.root / "s3%3Ab%2Fx.stale1"
            live.mkdir(parents=True)
            (live / ".done").write_text("")
            cache._last_sweep -= cache.SWEEP_INTERVAL + 1
            cache._evict(keep=set())
            self.assertTrue(live.exists(), "the sweep deleted a live entry")


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
                    if (entry / cache.CLAIM).exists() and cache._owner_of(entry) is None:
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
        ("queued since forever",           {"state": "queued", "created": -10 ** 6}, False),
        ("running since forever",          {"state": "running", "started": -10 ** 6}, False),
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
