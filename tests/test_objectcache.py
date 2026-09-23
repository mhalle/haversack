"""The shared result cache on an object store (``haversack.objectcache``, 2026-09-19).

Two "hosts" here are two local ``ResultCache`` directories in front of ONE in-memory
store - obstore's ``MemoryStore`` honors both conditional writes, which is the property
the protocol stands on. What these tests hold it to:

- a store that does not refuse a stale conditional write is refused at startup - asked of
  the store, not assumed (obstore's own local-disk store fails it, measured);
- a publication on one host is served on another, and a republication replaces it there;
- an artifact is never recorded beside another publication's labels, whatever interleaves;
- anything missing or wrong in the store is a MISS, never an error, and the next
  publication repairs it;
- the sweep takes only blobs no pointer references, and only once they are old.
"""
from __future__ import annotations

import hashlib
import os
import json
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import pytest

obstore = pytest.importorskip("obstore")
from obstore.store import LocalStore, MemoryStore  # noqa: E402

from haversack import objectcache  # noqa: E402
from provender import DiskStore, ops  # noqa: E402
from provender import Blobs as BlobStore  # noqa: E402  (the store moved to provender)
from haversack.objectcache import (ObjectStoreUnsuitable,  # noqa: E402
                                   SharedResultCache, check_conditional_writes, open_store)
from haversack.serve import CURRENT_NAME, RESULT_NAME, ResultCache  # noqa: E402

KEY = "ab" * 32


class _Hosts(unittest.TestCase):
    def make_store(self):
        """The ONE store both hosts share. Every class built on this runs twice: here in
        memory, and again on a provender DiskStore (the ``...OnDisk`` classes at the end of
        this module) - the protocol is meant to be one protocol whatever holds it."""
        return MemoryStore()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = self.make_store()
        self.a = self.host("a")
        self.b = self.host("b")

    def tearDown(self):
        import shutil
        # not TemporaryDirectory.cleanup(): a test that leaves a staging directory behind
        # (a publication interrupted on purpose) made cleanup raise ENOTEMPTY and failed a
        # test whose body had passed - measured flaky 1 run in 6 (review, 2026-09-20)
        shutil.rmtree(self.tmp, ignore_errors=True)
        self._tmp.cleanup()

    def host(self, name, prefix="pre/"):
        return SharedResultCache(self.store, ResultCache(self.tmp / f"local-{name}"),
                                 prefix=prefix)

    def file(self, name, data: bytes) -> Path:
        p = self.tmp / "src" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        return p

    def publish(self, cache, labels=b"labels-1", *, preview=None, key=KEY):
        return cache.put(key, self.file(f"labels-{labels.hex()}", labels),
                         {"outputs": [labels.decode()]}, {"task": "t", "computed": 1.0},
                         preview_path=self.file("preview-src.png", preview) if preview else None)

    def pointer(self, key=KEY):
        """The entry as a reader sees it: the ref resolved to its manifest (format 2), in
        the shape format 1's pointer had - ``generation``, ``files``, ``result``, ``meta`` -
        plus ``_manifest``, ``replaces`` and the manifest document itself (``_doc``)."""
        kind, entry, _ = self.a._read_ref(key)
        assert kind in ("live", "tombstone"), kind
        return entry

    def data_blobs(self) -> set:
        """Digests of the stored blobs that are RESULT bytes, not manifests - format 2 keeps
        its manifests in the same content-addressed store."""
        out = set()
        for o in ops.list(self.store, "pre/blobs/").collect():
            data = bytes(ops.get(self.store, o["path"]).bytes())
            try:
                doc = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                doc = None
            if not (isinstance(doc, dict) and doc.get("format") == objectcache.POINTER_FORMAT
                    and "key" in doc):
                out.add("sha256:" + o["path"].rsplit("/", 1)[1])
        return out

    def ref(self, key=KEY):
        """The ref object exactly as stored."""
        return json.loads(bytes(ops.get(self.store, f"pre/results/{key}.json").bytes()))

    def write_legacy(self, doc, key=KEY):
        """Put a FORMAT 1 pointer in place, as a host before 2026-09-23 wrote it."""
        ops.put(self.store, f"pre/results/{key}.json", json.dumps(doc).encode())

    def write_manifest(self, key=KEY, **changes) -> str:
        """Point ``key``'s ref at a manifest that is the current one with ``changes`` -
        written as another host would, digest and inline copy consistent. Its digest."""
        doc = {**self.pointer(key)["_doc"], **changes}
        data = objectcache._canonical(doc)
        self.a.blobs.put_bytes(data)
        digest = objectcache._digest_of(data)
        ops.put(self.store, f"pre/results/{key}.json", objectcache._canonical(
            {"format": objectcache.POINTER_FORMAT, "manifest": digest,
             "body": data.decode()}))
        return digest

    def as_legacy(self, key=KEY, **changes) -> dict:
        """The current entry rewritten as the format 1 pointer it would have been, with
        ``changes`` applied - how the format 1 tests damage a pointer."""
        p = self.pointer(key)
        doc = {"format": 1, "generation": p["generation"], "published": p["published"],
               "files": p["files"], "result": p["result"], "meta": p["meta"], "history": []}
        doc.update(changes)
        self.write_legacy(doc, key)
        return doc


class TestConditionalWriteProbe(unittest.TestCase):
    def test_memory_store_passes(self):
        check_conditional_writes(MemoryStore(), "x/")

    def test_local_store_is_refused_naming_the_missing_write(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ObjectStoreUnsuitable) as cm:
                check_conditional_writes(LocalStore(d))
            self.assertIn("replace-if-unchanged", str(cm.exception))
            self.assertIn("LocalStore", str(cm.exception))
            self.assertEqual([], [p for p in Path(d).rglob("*") if p.is_file()],
                             "the probe object is removed even when the store is refused")

    def test_a_store_that_ignores_the_etag_is_refused(self):
        """Every conditional replace succeeds: two writers would both think they won."""
        real_put = ops.put
        store = MemoryStore()

        def put(s, path, data, *, mode=None, **kw):
            return real_put(s, path, data, mode=None if isinstance(mode, dict) else mode, **kw)
        with unittest.mock.patch.object(ops, "put", put):
            with self.assertRaises(ObjectStoreUnsuitable) as cm:
                check_conditional_writes(store)
        self.assertIn("stale etag", str(cm.exception))

    def test_a_store_that_overwrites_on_create_is_refused(self):
        real_put = ops.put

        def put(s, path, data, *, mode=None, **kw):
            return real_put(s, path, data, mode=None if mode == "create" else mode, **kw)
        with unittest.mock.patch.object(ops, "put", put):
            with self.assertRaises(ObjectStoreUnsuitable) as cm:
                check_conditional_writes(MemoryStore())
        self.assertIn("create-if-absent", str(cm.exception))

    def test_open_store_urls(self):
        store, prefix = open_store("memory://bucket/some/prefix/")
        self.assertIsInstance(store, MemoryStore)
        self.assertEqual("bucket/some/prefix/", prefix)
        with self.assertRaises(Exception):
            open_store("http://example.org/x")


class TestSharing(_Hosts):
    def test_a_publication_is_served_on_another_host(self):
        gen = self.publish(self.a, b"one")
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit)
        self.assertEqual(b"one", Path(hit[0]).read_bytes())
        self.assertEqual({"outputs": ["one"]}, hit[1])
        self.assertEqual(gen, self.b.local.generation(KEY),
                         "the local copy keeps the store's generation token")

    def test_a_republication_replaces_the_other_hosts_copy(self):
        self.publish(self.a, b"one")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        gen2 = self.publish(self.a, b"two")
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(gen2, self.b.local.generation(KEY))

    def test_a_current_local_copy_downloads_nothing(self):
        self.publish(self.a, b"one")
        self.b.get(KEY)
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_deleted_entry_is_a_miss_on_every_host(self):
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertTrue(self.a.delete(KEY))
        self.assertIsNone(self.b.get(KEY), "B's local copy must not outlive the pointer")
        self.assertFalse(self.a.delete(KEY))

    def test_list_reads_pointers(self):
        self.publish(self.a, b"one", preview=b"png")
        rows, position = self.b.list()
        self.assertEqual([KEY], [e["key"] for e in rows])
        self.assertEqual(3, rows[0]["bytes"])
        self.assertEqual("t", rows[0]["task"])
        self.assertIsNone(position, "one page held everything")
        self.assertEqual(1, len(self.b.list(limit=1)[0]))

    def test_identical_bytes_are_one_blob(self):
        self.publish(self.a, b"same", key="cd" * 32)
        self.publish(self.b, b"same", key="ef" * 32)
        self.assertEqual({f"sha256:{hashlib.sha256(b'same').hexdigest()}"}, self.data_blobs())

    def test_published_result_needs_no_bytes(self):
        self.publish(self.a, b"one")
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual({"outputs": ["one"]}, self.b.published_result(KEY))


class TestArtifacts(_Hosts):
    def test_an_artifact_reaches_the_other_host(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertTrue(self.a.add_artifact(KEY, "preview.png",
                                            self.file("p.png", b"png"), generation=gen))
        path, _ = self.b.get(KEY)
        self.assertEqual(b"png", (Path(path).parent / "preview.png").read_bytes())

    def test_an_artifact_for_a_superseded_publication_is_refused(self):
        old = self.publish(self.a, b"one")
        self.publish(self.b, b"two")
        self.assertFalse(self.a.add_artifact(KEY, "preview.png",
                                             self.file("p.png", b"png"), generation=old))
        self.assertNotIn("preview.png", self.pointer()["files"])

    def test_a_publication_landing_mid_swap_keeps_the_artifact_off_it(self):
        """The interleaving the conditional write exists for: the artifact's writer reads
        the OLD pointer, another host publishes, and only then does the artifact's write
        go out. It must be refused by the store, reread, and abandoned."""
        old = self.publish(self.a, b"one")
        real = SharedResultCache._read_ref
        state = {"raced": False}

        def read(cache, key, **kw):
            got = real(cache, key, **kw)
            if cache is self.a and not state["raced"]:
                state["raced"] = True
                self.publish(self.b, b"two")
            return got
        with unittest.mock.patch.object(SharedResultCache, "_read_ref", read):
            placed = self.a.add_artifact(KEY, "preview.png", self.file("p.png", b"png"),
                                         generation=old)
        self.assertTrue(state["raced"])
        self.assertFalse(placed)
        ptr = self.pointer()
        self.assertNotIn("preview.png", ptr["files"])
        self.assertEqual(hashlib.sha256(b"two").hexdigest(),
                         ptr["files"][RESULT_NAME]["digest"].split(":")[1])

    def test_a_racing_publication_is_retried_not_lost(self):
        """Last writer wins, as the rename did: the lost race rereads and writes again."""
        real = SharedResultCache._read_ref
        state = {"raced": False}

        def read(cache, key, **kw):
            got = real(cache, key, **kw)
            if cache is self.a and not state["raced"]:
                state["raced"] = True
                self.publish(self.b, b"two")
            return got
        with unittest.mock.patch.object(SharedResultCache, "_read_ref", read):
            gen = self.publish(self.a, b"one")
        self.assertEqual(gen, self.pointer()["generation"])
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())


class TestMisses(_Hosts):
    def _blob_path(self, data: bytes) -> str:
        return f"pre/blobs/sha256/{hashlib.sha256(data).hexdigest()}"

    def test_a_swept_blob_is_a_miss_and_a_republication_heals_it(self):
        self.publish(self.a, b"one")
        ops.delete(self.store, self._blob_path(b"one"))
        self.assertIsNone(self.b.get(KEY))
        self.publish(self.a, b"one")               # same bytes: the blob must be re-uploaded
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_corrupt_blob_is_a_miss_and_is_replaced_not_deleted(self):
        """A reader cannot tell a corrupt object from a stream that ended early, and a
        blob is shared by every result with identical output - so deleting it on one
        client's reading took unrelated entries with it (review, 2026-09-19). The reader
        stops deduplicating onto it instead, and the next publication replaces it."""
        self.publish(self.a, b"one")
        digest = f"sha256:{hashlib.sha256(b'one').hexdigest()}"
        ops.put(self.store, self._blob_path(b"one"), b"not one")
        self.assertIsNone(self.b.get(KEY))
        self.assertTrue(self.b.blobs.has(digest), "not deleted: other keys may share it")
        self.assertIn(digest, self.b.blobs.suspect)
        self.publish(self.b, b"one")           # the host that saw it wrong republishes
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertNotIn(digest, self.b.blobs.suspect)

    def test_an_unrelated_key_survives_a_corrupt_blob(self):
        self.publish(self.a, b"same", key="cd" * 32)
        self.publish(self.a, b"same")          # one blob, two keys
        ops.put(self.store, self._blob_path(b"same"), b"wrong")
        self.assertIsNone(self.b.get(KEY))
        self.assertTrue(self.b.blobs.has(f"sha256:{hashlib.sha256(b'same').hexdigest()}"),
                        "deleting it would take the other key down too")

    def test_an_unreadable_pointer_is_a_miss(self):
        self.publish(self.a, b"one")
        ops.put(self.store, f"pre/results/{KEY}.json", b"{not json")
        self.assertIsNone(self.b.get(KEY))
        self.publish(self.a, b"two")               # garbage is nobody's: publish over it
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

    def test_an_entry_from_a_NEWER_haversack_is_a_miss_and_is_not_overwritten(self):
        """Refusing is the difference between a miss and taking another host's current
        result and its whole history out of the index in one write (review, 2026-09-20).
        Garbage gets published over; something that says it came from a later version does
        not."""
        self.publish(self.a, b"one")
        newer = {**self.ref(), "format": objectcache.POINTER_FORMAT + 1}
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(newer).encode())
        self.assertIsNone(self.b.get(KEY), "cannot read it")
        with self.assertRaises(ObjectStoreUnsuitable) as cm:
            self.publish(self.a, b"two")
        self.assertIn("newer haversack", str(cm.exception))
        self.assertEqual(newer, self.ref(), "left exactly as it was")

    def test_a_blob_swept_before_the_pointer_is_put_back(self):
        """put skipped the upload because the blob existed; a sweep then took it before
        the pointer was written. The check after the pointer re-uploads it."""
        self.publish(self.a, b"one", key="cd" * 32)
        self.a.delete("cd" * 32)
        real = BlobStore.put_file
        swept = {"done": False}

        def put_file(blobs, src):
            got = real(blobs, src)
            if not swept["done"]:
                swept["done"] = True
                ops.delete(self.store, self._blob_path(b"one"))
            return got
        with unittest.mock.patch.object(BlobStore, "put_file", put_file):
            self.publish(self.a, b"one")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())


class TestSweep(_Hosts):
    def test_only_old_unreferenced_blobs_go(self):
        """A superseded generation is KEPT (bounded history), so its blobs stay referenced
        until it falls off the end - then, and only then, the sweep may take them."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        self.assertEqual(0, self.a.sweep()["deleted_blobs"], "inside the grace")
        self.assertEqual(0, self.a.sweep(now=later)["deleted_blobs"], "still in history")
        self.assertTrue(self.a.blobs.has(f"sha256:{hashlib.sha256(b'one').hexdigest()}"))
        for i in range(objectcache.HISTORY_KEEP + 1):   # push "one" and "two" off the end
            self.publish(self.a, f"more-{i}".encode())
        # their bytes AND their two manifests: the chain's tail goes with them
        self.assertEqual(4, self.a.sweep(now=later)["deleted_blobs"])
        for gone in (b"one", b"two"):
            self.assertFalse(self.a.blobs.has(f"sha256:{hashlib.sha256(gone).hexdigest()}"))
        self.assertEqual(objectcache.HISTORY_KEEP + 1, len(self.a.history(KEY)),
                         "what the sweep spared is exactly what history keeps - every link "
                         "of the chain it walks included")
        self.assertEqual(b"more-4", Path(self.b.get(KEY)[0]).read_bytes())

    def test_max_age_expires_pointers_and_then_their_blobs(self):
        self.publish(self.a, b"one")
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        got = self.a.sweep(now=later, max_age_s=3600)
        self.assertEqual({"expired_pointers": 1, "deleted_blobs": 2, "already_gone": 0,
                          "unreadable_pointers": 0}, got, "the labels and their manifest")
        self.assertIsNone(self.b.get(KEY))
        self.assertEqual(set(), self.data_blobs())

    def test_another_prefix_is_not_touched(self):
        other = self.host("c", prefix="other/")
        self.publish(other, b"one")
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, max_age_s=0)
        self.assertEqual(b"one", Path(self.host("d", prefix="other/").get(KEY)[0]).read_bytes())


class TestLocalCopyGeneration(unittest.TestCase):
    """``ResultCache.put(generation=...)``: the local copy's side of the protocol."""

    def test_an_existing_generation_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            cache = ResultCache(Path(d) / "c")
            src = Path(d) / "l"
            src.write_bytes(b"x")
            cache.put(KEY, src, {}, {}, generation="g1")
            with self.assertRaises(FileExistsError):
                cache.put(KEY, src, {"other": 1}, {}, generation="g1")
            self.assertEqual("g1", cache.generation(KEY))
            self.assertEqual({}, cache.get(KEY)[1])

    def test_a_copy_that_lands_mid_put_is_not_removed_by_the_loser(self):
        """Two processes copying one generation: the other lands after this one's check
        and before its rename. The rename fails - and the failure path must not delete
        a directory this call never placed."""
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "l"
            src.write_bytes(b"x")

            class Racing(ResultCache):
                def _claim(self, entry, gen):
                    other = self._generation_dir(KEY, gen)
                    other.mkdir(parents=True)
                    (other / RESULT_NAME).write_bytes(b"the other copy")
                    return super()._claim(entry, gen)
            cache = Racing(Path(d) / "c")
            with self.assertRaises(OSError):
                cache.put(KEY, src, {}, {}, generation="g1")
            self.assertEqual(b"the other copy",
                             (cache._generation_dir(KEY, "g1") / RESULT_NAME).read_bytes())


# -- through the server -----------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
sitk = pytest.importorskip("SimpleITK")


def test_two_servers_share_one_store(tmp_path):
    """A job computed on one server is a cache hit on another that never saw it."""
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_job_result_cache import _Segmenter
    from test_serve import submit, wait_state

    store = MemoryStore()
    ex_a = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "wa",
                         cache_dir=tmp_path / "ca", result_store=store)
    ex_b = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "wb",
                         cache_dir=tmp_path / "cb", result_store=store)
    try:
        client = TestClient(create_app(ex_a))
        jid = submit(client)
        s = wait_state(client, jid, ("done",))
        hit = ex_b.cache_get(s["key"])
        assert hit is not None
        assert Path(hit[0]).read_bytes() == Path(ex_a.cache_get(s["key"])[0]).read_bytes()
        assert [e["key"] for e in ex_b.cache_list()[0]] == [s["key"]]
    finally:
        ex_a.close()
        ex_b.close()


def test_a_result_store_without_a_local_cache_is_refused(tmp_path):
    from haversack.errors import InputError
    from haversack.serve import LocalExecutor
    from test_job_result_cache import _Segmenter
    with pytest.raises(InputError, match="--no-result-cache"):
        LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", result_store=MemoryStore())


def test_an_unsuitable_store_stops_the_server_at_startup(tmp_path):
    """obstore's LocalStore cannot replace conditionally. (This test used to name the store
    by `file://`, which opened one; since provender 0.1.6 `file://` opens a DiskStore, which
    can - see the next test.)"""
    from haversack.serve import LocalExecutor
    from test_job_result_cache import _Segmenter
    (tmp_path / "bucket").mkdir()
    with pytest.raises(ObjectStoreUnsuitable):
        LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                      result_store=LocalStore(tmp_path / "bucket"))


def test_two_servers_share_a_file_url_store(tmp_path):
    """`--result-store file:///path` (provender 0.1.6): the same protocol on a directory, no
    network - a result computed by one server is a hit on another that never saw it."""
    from haversack.serve import LocalExecutor, create_app
    from fastapi.testclient import TestClient
    from test_job_result_cache import _Segmenter
    from test_serve import submit, wait_state

    url = f"file://{tmp_path / 'store'}"
    ex_a = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "wa",
                         cache_dir=tmp_path / "ca", result_store=url)
    ex_b = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "wb",
                         cache_dir=tmp_path / "cb", result_store=url)
    try:
        assert isinstance(ex_a.cache.store, DiskStore)
        s = wait_state(TestClient(create_app(ex_a)), submit(TestClient(create_app(ex_a))),
                       ("done",))
        hit = ex_b.cache_get(s["key"])
        assert hit is not None
        assert Path(hit[0]).read_bytes() == Path(ex_a.cache_get(s["key"])[0]).read_bytes()
        assert [e["key"] for e in ex_b.cache_list()[0]] == [s["key"]]
    finally:
        ex_a.close()
        ex_b.close()


# -- what the 2026-09-19 review round found ---------------------------------------------
#
# Three agents reviewed the module, its tests and its wiring. Every class below pins one
# finding: the first version of each behaved as the test's own docstring describes.

POINTER_FORMAT_NEXT = objectcache.POINTER_FORMAT + 1


class TestStoreFaultsAreMisses(_Hosts):
    """A store fault on a READ is a miss; on a WRITE it raises.

    ``obstore``'s errors do not subclass OSError, so only FileNotFoundError was caught and
    everything else - expired credentials, DNS, a 503 - left ``cache_get`` as a bare 500 on
    routes SERVER.md promises 404/410 for, anonymous ones included. That is the class of
    defect 0.12.3 fixed for the scratch read, arriving through a different door.
    """

    def faulty(self, which="get"):
        from obstore.exceptions import PermissionDeniedError
        real = getattr(obstore, which)

        def boom(store, path, *a, **kw):
            if "results/" in str(path) or "blobs/" in str(path):
                raise PermissionDeniedError("403 from the bucket")
            return real(store, path, *a, **kw)
        return unittest.mock.patch.object(ops, which, boom)

    def test_a_read_of_an_unreachable_store_is_a_miss(self):
        self.publish(self.a, b"one")
        with self.faulty("get"):
            self.assertIsNone(self.b.get(KEY))
            self.assertIsNone(self.b.published_result(KEY))
            self.assertIsNone(self.b.generation(KEY))

    def test_a_write_to_an_unreachable_store_raises(self):
        from obstore.exceptions import PermissionDeniedError
        with self.faulty("get"):
            with self.assertRaises(PermissionDeniedError):
                self.publish(self.a, b"one")

    def test_a_blob_fault_mid_fill_is_a_miss(self):
        self.publish(self.a, b"one")

        def boom(blobs, digest, dest):
            raise OSError("connection reset mid-stream")
        with unittest.mock.patch.object(BlobStore, "fetch", boom):
            self.assertIsNone(self.b.get(KEY))
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_local_copy_that_cannot_be_written_does_not_fail_the_publication(self):
        """The publication HAPPENED - every host can read it - so a full disk here must
        not fail the job that produced it."""
        with unittest.mock.patch.object(ResultCache, "put",
                                        side_effect=OSError(28, "No space left on device")):
            gen = self.publish(self.a, b"one")
        self.assertEqual(gen, self.b.generation(KEY))
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_add_artifact_never_raises_on_the_overlap_thread(self):
        gen = self.publish(self.a, b"one")
        with self.faulty("put"):
            self.assertFalse(self.a.add_artifact(KEY, "preview.png",
                                                 self.file("p.png", b"png"), generation=gen))


class TestDamagedPointers(_Hosts):
    """A pointer is written by another host: every field is data, not a promise."""

    def damaged(self, ptr):
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())

    def test_a_pointer_missing_its_fields_is_a_miss(self):
        for ptr in ({"format": 1},                                     # no generation
                    {"format": 1, "generation": "g", "files": []},     # files not a dict
                    {"format": 1, "generation": "g",
                     "files": {"labels.seg.nrrd": {"digest": "sha256:zz", "size": 1}}},
                    {"format": 1, "generation": "g",
                     "files": {"labels.seg.nrrd": {"digest": "sha256:" + "a" * 64}}}):
            with self.subTest(ptr=ptr):
                self.damaged(ptr)
                self.assertIsNone(self.b.get(KEY))
                self.assertIsNone(self.b.generation(KEY))
                self.assertEqual([], self.b.list()[0])
                self.b.sweep()                 # must not raise either

    def test_a_pointer_may_not_choose_where_bytes_land(self):
        blob = self.b.blobs.put_file(self.file("evil", b"evil"))
        escape = str(self.tmp / "escaped")
        self.damaged({"format": 1, "generation": "g",
                      "files": {"labels.seg.nrrd": blob, f"../../{escape}": blob}})
        self.b.get(KEY)
        self.assertFalse(Path(escape).exists(), "a foreign name decided a local path")

    def test_a_stray_object_does_not_abort_list_or_sweep(self):
        """One `.tmp` file from someone's sync used to raise out of both - and a sweep
        that never runs is a bucket that never stops growing."""
        self.publish(self.a, b"one")
        ops.put(self.store, "pre/results/.tmp-upload.json", b"{}")
        ops.put(self.store, "pre/results/notes.json", b"not a pointer")
        self.assertEqual([KEY], [e["key"] for e in self.b.list()[0]])
        got = self.b.sweep(grace_s=0)
        self.assertEqual(2, got["unreadable_pointers"])
        self.assertEqual(0, got["deleted_blobs"],
                         "blobs it cannot account for are not deleted")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_newer_pointer_format_is_never_garbage_collected(self):
        """An old host's sweeper meeting a new writer's pointer: it cannot read which
        blobs are live, so it deletes none."""
        self.publish(self.a, b"one")
        labels = self.pointer()["files"][RESULT_NAME]["digest"]
        self.damaged({**self.ref(), "format": POINTER_FORMAT_NEXT})
        got = self.a.sweep(grace_s=0)
        self.assertEqual(0, got["deleted_blobs"])
        self.assertEqual(1, got["unreadable_pointers"])
        self.assertTrue(self.a.blobs.has(labels))


class TestArtifactBlobs(_Hosts):
    def test_a_missing_artifact_blob_does_not_lose_the_labels(self):
        """A lost thumbnail is not worth a GPU recompute."""
        gen = self.publish(self.a, b"one")
        self.a.add_artifact(KEY, "preview.png", self.file("p.png", b"png"), generation=gen)
        ops.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'png').hexdigest()}")
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit)
        self.assertEqual(b"one", Path(hit[0]).read_bytes())
        self.assertFalse((Path(hit[0]).parent / "preview.png").exists())

    def test_an_artifacts_blob_is_re_checked_after_the_pointer(self):
        """The same sweep window ``put`` closes: the upload was skipped because the blob
        was there, and it was swept before the pointer named it."""
        gen = self.publish(self.a, b"one")
        png = self.file("p.png", b"png")
        self.a.blobs.put_file(png)             # the blob exists, so the next upload skips
        real = SharedResultCache._swap

        def swap(cache, key, update):
            written = real(cache, key, update)
            ops.delete(self.store,
                           f"pre/blobs/sha256/{hashlib.sha256(b'png').hexdigest()}")
            return written
        with unittest.mock.patch.object(SharedResultCache, "_swap", swap):
            self.assertTrue(self.a.add_artifact(KEY, "preview.png", png, generation=gen))
        hit = self.b.get(KEY)
        self.assertEqual(b"png", (Path(hit[0]).parent / "preview.png").read_bytes())

    def test_an_artifact_name_must_be_an_artifact(self):
        self.publish(self.a, b"one")
        with self.assertRaises(ValueError):
            self.a.add_artifact(KEY, RESULT_NAME, self.file("p.png", b"png"))

    def test_an_artifact_on_a_key_that_was_never_published_is_refused(self):
        self.assertFalse(self.a.add_artifact("cd" * 32, "preview.png",
                                             self.file("p.png", b"png")))


class TestGapsFromMutation(_Hosts):
    """Facts the first round of tests left unpinned (mutation run, 2026-09-19)."""

    def test_the_publishing_host_keeps_its_own_copy(self):
        gen = self.publish(self.a, b"one")
        self.assertEqual(gen, self.a.local.generation(KEY))
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded its own")):
            self.assertEqual(b"one", Path(self.a.get(KEY)[0]).read_bytes())

    def test_delete_removes_the_local_copy_too(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        self.assertIsNone(self.a.local.get(KEY))

    def test_list_is_newest_first_and_honors_its_limit(self):
        for i, key in enumerate(("aa" * 32, "bb" * 32, "cc" * 32)):
            self.a.put(key, self.file(f"l{i}", f"l{i}".encode()), {}, {"computed": float(i)})
        self.assertEqual(3, len(self.b.list()[0]))
        self.assertEqual(2, len(self.b.list(limit=2)[0]))

    def test_list_reads_only_as_many_pointers_as_asked_for(self):
        for i, key in enumerate(("aa" * 32, "bb" * 32, "cc" * 32)):
            self.a.put(key, self.file(f"l{i}", f"l{i}".encode()), {}, {"computed": float(i)})
        real, seen = SharedResultCache._read_pointer, []

        def read(cache, key, **kw):
            seen.append(key)
            return real(cache, key, **kw)
        with unittest.mock.patch.object(SharedResultCache, "_read_pointer", read):
            self.b.list(limit=1)
        self.assertEqual(1, len(seen), "a bucket of 40,000 entries is 40,000 round trips")

    def test_evict_bounds_the_local_copy(self):
        with unittest.mock.patch.object(ResultCache, "evict") as evict:
            self.a.evict()
        evict.assert_called_once()

    def test_a_swept_result_blob_is_a_miss_on_the_publishing_host_too(self):
        self.publish(self.a, b"one")
        ops.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        self.a.local.delete(KEY)               # its copy is gone; the store must answer
        self.assertIsNone(self.a.get(KEY))

    def test_a_publication_storm_raises_rather_than_losing_the_pointer(self):
        """Every conditional write is refused: the publication must give up loudly, after
        exactly SWAP_ATTEMPTS tries, rather than return a generation it never wrote.

        The first version drove this with a mock that republished inside `_read_pointer`,
        which recursed and passed on the RecursionError - a RuntimeError subclass - without
        ever reaching the guard. Mutation testing caught it: raising ValueError instead,
        and multiplying SWAP_ATTEMPTS by 1000, both survived (2026-09-20).
        """
        from obstore.exceptions import PreconditionError
        self.publish(self.a, b"first")         # so the write is a replace, not a create
        reads = []
        real = SharedResultCache._read_ref

        def counting_read(cache, key, **kw):
            reads.append(key)
            return real(cache, key, **kw)
        real_put = ops.put

        def refuse_pointer_writes(store, path, *a, **kw):
            if "results/" in str(path):        # the blobs still upload: only the pointer
                raise PreconditionError("etag moved")
            return real_put(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "put", refuse_pointer_writes), \
                unittest.mock.patch.object(SharedResultCache, "_read_ref",
                                           counting_read), \
                unittest.mock.patch.object(objectcache, "SWAP_ATTEMPTS", 3):
            with self.assertRaises(RuntimeError) as cm:
                self.publish(self.a, b"one")
        self.assertIn("publications raced this one", str(cm.exception))
        self.assertEqual(3, len(reads), "it tried exactly SWAP_ATTEMPTS times")

    def test_two_threads_filling_one_key_download_once(self):
        import threading
        self.publish(self.a, b"one")
        real, calls = BlobStore.fetch, []

        def fetch(blobs, digest, dest):
            calls.append(digest)
            time.sleep(0.05)
            return real(blobs, digest, dest)
        with unittest.mock.patch.object(BlobStore, "fetch", fetch):
            threads = [threading.Thread(target=self.b.get, args=(KEY,)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(1, len(calls), f"downloaded {len(calls)} times")

    def test_a_dead_fills_work_directory_is_swept_by_the_next_fill(self):
        stale = self.b.local.root / f"{objectcache.WORK_PREFIX}dead"
        stale.mkdir(parents=True)
        os.utime(stale, (0, 0))                # older than WORK_GRACE_S
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertFalse(stale.exists(), "nothing else reclaims a dotted directory")

    def test_the_bucket_prefix_is_parsed(self):
        self.assertEqual("p/q/", open_store("s3://bucket/p/q")[1])
        self.assertEqual("", open_store("s3://bucket")[1])

    def test_a_key_may_not_escape_its_prefix(self):
        for bad in ("../../escape", ".hidden", "", "a/b"):
            with self.subTest(key=bad), self.assertRaises(ValueError):
                self.a._pointer_path(bad)
        with self.assertRaises(ValueError):
            self.a.blobs.path("sha256:../../results/" + "a" * 44)

    def test_max_age_spares_a_fresh_pointer(self):
        self.publish(self.a, b"one")
        got = self.a.sweep(now=time.time() + 60, max_age_s=3600)
        self.assertEqual(0, got["expired_pointers"])
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())


def test_a_broken_store_is_a_miss_on_the_wire_not_a_500(tmp_path):
    """SERVER.md promises 404 for a result that is not there and 410 for bytes that are
    gone. A store fault used to answer 500 on every one of these routes, to anonymous
    callers included (review, 2026-09-19)."""
    from fastapi.testclient import TestClient
    from obstore.exceptions import PermissionDeniedError

    from haversack.serve import LocalExecutor, create_app
    from test_job_result_cache import _Segmenter
    from test_serve import submit, wait_state

    store = MemoryStore()
    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       result_store=store)
    try:
        client = TestClient(create_app(ex))
        jid = submit(client)
        s = wait_state(client, jid, ("done",))
        (ex.get(jid).dir / RESULT_NAME).unlink()        # only the entry is left
        real = ops.get

        def boom(st, path, *a, **kw):
            if "results/" in str(path) or "blobs/" in str(path):
                raise PermissionDeniedError("403 from the bucket")
            return real(st, path, *a, **kw)
        with unittest.mock.patch.object(ops, "get", boom):
            # this host holds the result it just computed, so the outage is not a miss:
            # it serves its own copy rather than throwing away warm work
            assert client.get(f"/v1/jobs/{jid}/result").status_code == 200
            assert client.get("/v1/segmentations").status_code == 200
            assert ex.cache_get(s["key"]) is not None
        ex.cache.local.delete(s["key"])        # now nothing here holds it
        with unittest.mock.patch.object(ops, "get", boom):
            assert client.get(f"/v1/jobs/{jid}/result").status_code == 410
            assert ex.cache_get(s["key"]) is None
    finally:
        ex.close()


def test_no_async_route_blocks_the_event_loop_on_the_cache():
    """A cache lookup is a network round trip once a store is configured, and a hit can
    download the labels. Inside an async handler that stalls every other request, so these
    calls go through the threadpool - checked as PARSED CALLS, because a comment or a
    docstring satisfies a grep (AGENTS.md, 2026-09-12)."""
    import ast
    import inspect

    from haversack import serve as serve_mod
    tree = ast.parse(inspect.getsource(serve_mod))
    blocking = {"cache_get", "status_of", "submit"}
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        # a sync helper defined inside is not ours, and neither is a lambda: both are HANDED
        # to something (main's `_await_artifact` runs its `look_again` through
        # `asyncio.to_thread`), never run inline by the handler
        inner = {n for f in ast.walk(node)
                 if isinstance(f, (ast.FunctionDef, ast.Lambda))
                 for n in ast.walk(f)}
        for call in ast.walk(node):
            if call in inner or not isinstance(call, ast.Call):
                continue
            fn = call.func
            if (isinstance(fn, ast.Attribute) and fn.attr in blocking
                    and isinstance(fn.value, ast.Name) and fn.value.id == "executor"):
                problems.append(f"{node.name} (line {call.lineno}) calls executor."
                                f"{fn.attr} directly; use `await asyncio.to_thread(...)`")
    assert not problems, "\n  ".join(problems)


class TestBoundedHistory(_Hosts):
    """A republication keeps its predecessors, bounded by count and age (decided
    2026-09-19). The local copy keeps none, no ordinary read is ever served a superseded
    result, and `delete` takes the history with it."""

    def test_history_is_newest_first_with_the_current_publication_at_its_head(self):
        self.publish(self.a, b"one")
        self.publish(self.b, b"two")
        got = self.a.history(KEY)
        self.assertEqual([True, False], [h["current"] for h in got])
        self.assertEqual([{"outputs": ["two"]}, {"outputs": ["one"]}],
                         [h["result"] for h in got])

    def test_history_is_bounded_by_count(self):
        for i in range(objectcache.HISTORY_KEEP + 4):
            self.publish(self.a, f"v{i}".encode())
        self.assertEqual(objectcache.HISTORY_KEEP + 1, len(self.a.history(KEY)))

    def test_history_is_bounded_by_age(self):
        long_ago = time.time() - objectcache.HISTORY_MAX_AGE_S - 60
        with unittest.mock.patch.object(objectcache.time, "time", return_value=long_ago):
            self.publish(self.a, b"one")       # a manifest is immutable: it was written old
        self.publish(self.a, b"two")
        self.assertEqual([True], [h["current"] for h in self.a.history(KEY)])

    def test_an_ordinary_read_is_never_served_a_superseded_result(self):
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(b"two", Path(self.a.get(KEY)[0]).read_bytes())

    def test_a_superseded_generation_can_be_fetched_deliberately(self):
        gen = self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        dest = self.tmp / "out"
        dest.mkdir()
        entry = self.b.fetch_generation(KEY, gen, dest)
        self.assertEqual({"outputs": ["one"]}, entry["result"])
        self.assertEqual(b"one", (dest / RESULT_NAME).read_bytes())
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes(),
                         "fetching history must not change what this host serves")

    def test_a_generation_that_is_not_kept_is_not_fetchable(self):
        dest = self.tmp / "out2"
        dest.mkdir()
        self.publish(self.a, b"one")
        self.assertIsNone(self.b.fetch_generation(KEY, "nosuchgeneration", dest))

    def test_the_local_copy_keeps_no_history_of_its_own(self):
        """History lives in the store. The local copy holds the current generation and
        whatever its OWN rules still protect - a superseded generation survives only while
        a reader may hold it (`GENERATION_GRACE_S`), and goes after that."""
        with unittest.mock.patch.object(ResultCache, "GENERATION_GRACE_S", 0):
            self.publish(self.a, b"one")
            gen2 = self.publish(self.a, b"two")
            gens = [p.name for p in (self.a.local.root / KEY).iterdir()
                    if p.name.startswith("g-")]
        self.assertEqual([f"g-{gen2}"], gens)
        self.assertEqual(2, len(self.a.history(KEY)), "the store still has both")

    def test_delete_takes_the_history_with_it(self):
        """For anything near patient data, deletion means gone."""
        gen = self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        self.a.delete(KEY)
        dest = self.tmp / "out3"
        dest.mkdir()
        self.assertEqual([], self.b.history(KEY))
        self.assertIsNone(self.b.fetch_generation(KEY, gen, dest))
        # the bytes of both generations and both manifests are now unreferenced - the
        # tombstone keeps only itself - and a sweep takes them
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        self.assertEqual(4, self.a.sweep(now=later, grace_s=0,
                                         allow_empty=True)["deleted_blobs"])
        self.assertEqual(set(), self.data_blobs())

    def test_identical_bytes_make_history_nearly_free(self):
        self.publish(self.a, b"same")
        self.publish(self.a, b"same")          # a recompute that changed nothing
        self.assertEqual({f"sha256:{hashlib.sha256(b'same').hexdigest()}"}, self.data_blobs())
        self.assertEqual(2, len(self.a.history(KEY)))

    def test_a_damaged_past_does_not_make_the_present_unreadable(self):
        """Format 1, where history was inline and could be junk."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        ptr = self.pointer()
        self.write_legacy({"format": 1, "generation": ptr["generation"],
                           "published": ptr["published"], "files": ptr["files"],
                           "result": ptr["result"], "meta": ptr["meta"],
                           "history": [{"generation": 7}, "rubbish",
                                       {"generation": "g", "files": []}]})
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual([True], [h["current"] for h in self.b.history(KEY)])
        self.b.sweep(grace_s=0)                # must not raise

    def test_a_broken_chain_does_not_make_the_present_unreadable(self):
        """Format 2: a predecessor manifest that is gone or wrong ends the history there."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        prev = self.pointer()["replaces"][0]
        ops.put(self.store, self.a.blobs.path(prev), b"not the manifest it was")
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual([True], [h["current"] for h in self.b.history(KEY)])
        self.b.sweep(grace_s=0)                # must not raise


# -- migration: cache push / cache pull -------------------------------------------------


class TestPush(_Hosts):
    """A cache that has been filling for months is worth GPU-hours, and nothing else
    recovers it once the local protocol goes (docs/cache-consolidation.md, step 2)."""

    def local_entry(self, cache, key, labels=b"local", *, computed=1.0, preview=None):
        """Publish straight into a host's LOCAL cache, as a server would have."""
        return cache.local.put(key, self.file(f"l-{key[:4]}-{labels.hex()}", labels),
                               {"outputs": [labels.decode()]},
                               {"task": "t", "computed": computed},
                               preview_path=self.file("p.png", preview) if preview else None)

    def test_a_local_entry_reaches_the_store_and_another_host(self):
        gen = self.local_entry(self.a, KEY, b"months of work", preview=b"png")
        got = self.a.push()
        self.assertEqual(1, got["pushed"])
        self.assertEqual(gen, self.b.generation(KEY), "the generation token is preserved")
        hit = self.b.get(KEY)
        self.assertEqual(b"months of work", Path(hit[0]).read_bytes())
        self.assertEqual(b"png", (Path(hit[0]).parent / "preview.png").read_bytes())

    def test_the_pushing_host_then_downloads_nothing(self):
        """The point of keeping the token: the local copy IS the store's copy."""
        self.local_entry(self.a, KEY, b"warm")
        self.a.push()
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual(b"warm", Path(self.a.get(KEY)[0]).read_bytes())

    def test_a_second_push_uploads_nothing_and_replaces_nothing(self):
        self.local_entry(self.a, KEY, b"once")
        self.a.push()
        with unittest.mock.patch.object(BlobStore, "put_file",
                                        side_effect=AssertionError("re-uploaded")):
            got = self.a.push()
        self.assertEqual({"pushed": 0, "skipped": 1, "replaced": 0, "failed": 0,
                          "unreadable": 0}, got)

    def test_an_interrupted_push_is_simply_rerun(self):
        for i, key in enumerate(("aa" * 32, "bb" * 32, "cc" * 32)):
            self.local_entry(self.a, key, f"v{i}".encode())
        real, n = SharedResultCache._swap, []

        def dying_swap(cache, key, update):
            if len(n) >= 2:
                raise OSError("connection reset")
            n.append(key)
            return real(cache, key, update)
        with unittest.mock.patch.object(SharedResultCache, "_swap", dying_swap):
            first = self.a.push()
        self.assertEqual(2, first["pushed"])
        self.assertEqual(1, first["failed"])
        again = self.a.push()
        self.assertEqual(1, again["pushed"])
        self.assertEqual(2, again["skipped"])
        self.assertEqual(3, len(self.b.list()[0]))

    def test_a_key_the_store_already_has_is_kept_by_default(self):
        """It may be newer than ours: another host computed it after this cache went cold."""
        self.publish(self.b, b"theirs")
        self.local_entry(self.a, KEY, b"ours")
        self.assertEqual(1, self.a.push()["skipped"])
        self.assertEqual({"outputs": ["theirs"]}, self.b.published_result(KEY))

    def test_conflict_newer_compares_the_timestamps(self):
        self.publish(self.b, b"theirs")        # computed 1.0
        self.local_entry(self.a, KEY, b"ours", computed=2.0)
        self.assertEqual(1, self.a.push(conflict="newer")["replaced"])
        self.assertEqual({"outputs": ["ours"]}, self.b.published_result(KEY))

    def test_conflict_newer_keeps_theirs_when_ours_is_older(self):
        self.publish(self.b, b"theirs")        # computed 1.0
        self.local_entry(self.a, KEY, b"ours", computed=0.5)
        self.assertEqual(1, self.a.push(conflict="newer")["skipped"])
        self.assertEqual({"outputs": ["theirs"]}, self.b.published_result(KEY))

    def test_conflict_force_takes_ours_and_keeps_theirs_in_history(self):
        self.publish(self.b, b"theirs")
        self.local_entry(self.a, KEY, b"ours", computed=0.5)
        self.assertEqual(1, self.a.push(conflict="force")["replaced"])
        self.assertEqual([{"outputs": ["ours"]}, {"outputs": ["theirs"]}],
                         [h["result"] for h in self.b.history(KEY)])

    def test_an_unknown_conflict_policy_is_refused(self):
        from haversack.errors import InputError
        with self.assertRaises(InputError):
            self.a.push(conflict="clobber")

    def test_a_half_written_entry_is_counted_not_pushed(self):
        self.local_entry(self.a, KEY, b"fine")
        (self.a.local._resolve(KEY, lease=False) / "meta.json").unlink()
        got = self.a.push()
        self.assertEqual(1, got["unreadable"])
        self.assertEqual([], self.b.list()[0])

    def test_a_legacy_flat_entry_is_given_a_generation(self):
        """An entry from before generations existed: still migratable, at the cost of one
        download the first time it is read."""
        d = self.a.local.root / KEY
        d.mkdir(parents=True)
        (d / RESULT_NAME).write_bytes(b"from the old protocol")
        (d / "result.json").write_text(json.dumps({"outputs": ["old"]}))
        (d / "meta.json").write_text(json.dumps({"task": "t", "computed": 1.0}))
        self.assertEqual(1, self.a.push()["pushed"])
        self.assertEqual(b"from the old protocol", Path(self.b.get(KEY)[0]).read_bytes())

    def test_dotted_directories_are_not_entries(self):
        """A staging directory, a tomb mid-reclamation, a fill's work directory: pushing
        one would publish an unfinished result."""
        (self.a.local.root / ".fill-abcd").mkdir(parents=True)
        (self.a.local.root / ".reclaim-1").mkdir(parents=True)
        self.local_entry(self.a, KEY, b"real")
        self.assertEqual(1, self.a.push()["pushed"])
        self.assertEqual([KEY], [e["key"] for e in self.b.list()[0]])

    def test_limit_takes_the_newest(self):
        for i, key in enumerate(("aa" * 32, "bb" * 32, "cc" * 32)):
            self.local_entry(self.a, key, f"v{i}".encode())
            os.utime(self.a.local.root / key, (1000 + i, 1000 + i))
        self.assertEqual(1, self.a.push(limit=1)["pushed"])
        self.assertEqual(["cc" * 32], [e["key"] for e in self.b.list()[0]])


class TestPull(_Hosts):
    def test_pull_makes_a_cold_host_warm(self):
        self.publish(self.a, b"one", preview=b"png")
        got = self.b.pull()
        self.assertEqual(1, got["pulled"])
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            hit = self.b.get(KEY)
        self.assertEqual(b"one", Path(hit[0]).read_bytes())
        self.assertEqual(b"png", (Path(hit[0]).parent / "preview.png").read_bytes())

    def test_a_pulled_cache_answers_without_the_store(self):
        """The way out: after a pull the local cache stands on its own."""
        self.publish(self.a, b"one")
        self.b.pull()
        self.assertEqual(b"one", Path(self.b.local.get(KEY)[0]).read_bytes())

    def test_pull_is_idempotent_and_free_when_current(self):
        self.publish(self.a, b"one")
        self.b.pull()
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            got = self.b.pull()
        self.assertEqual({"pulled": 0, "current": 1, "failed": 0, "unreadable": 0,
                          "evicted": 0}, got)

    def test_a_swept_blob_makes_one_entry_fail_and_not_the_rest(self):
        self.publish(self.a, b"one", key="aa" * 32)
        self.publish(self.a, b"two", key="bb" * 32)
        ops.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        got = self.b.pull()
        self.assertEqual({"pulled": 1, "current": 0, "failed": 1, "unreadable": 0,
                          "evicted": 0}, got)
        self.assertEqual(b"two", Path(self.b.get("bb" * 32)[0]).read_bytes())

    def test_pull_reports_pointers_it_cannot_read(self):
        self.publish(self.a, b"one")
        ops.put(self.store, "pre/results/.junk.json", b"{}")
        got = self.b.pull()
        self.assertEqual(1, got["pulled"])
        self.assertEqual(1, got["unreadable"])


class TestMigrationCli(_Hosts):
    """The command line, driven end to end against the in-memory store."""

    def run_cli(self, *argv):
        from haversack import cli
        return cli.main(list(argv))

    def test_push_then_pull_between_two_cache_dirs(self):
        src = self.tmp / "cli-a"
        dst = self.tmp / "cli-b"
        ResultCache(src).put(KEY, self.file("l", b"by the cli"), {"outputs": ["cli"]},
                             {"task": "t", "computed": 1.0})
        url = "memory://cli"                   # one process: the store persists in it
        with unittest.mock.patch.object(objectcache, "open_store",
                                        lambda _u: (self.store, "cli/")):
            self.assertEqual(0, self.run_cli("cache", "push", url, "--cache-dir", str(src)))
            self.assertEqual(0, self.run_cli("cache", "pull", url, "--cache-dir", str(dst),
                                             "--quiet"))
        self.assertEqual(b"by the cli", Path(ResultCache(dst).get(KEY)[0]).read_bytes())

    def test_a_store_must_be_named(self):
        """cli.main turns an InputError into a status and one line, not a traceback."""
        self.assertEqual(2, self.run_cli("cache", "push"))


# -- the 2026-09-20 review round: push/pull, history, deletion ---------------------------


class TestPushBindsBytesToTheirOwnToken(_Hosts):
    """The severe one: `push` read the directory and the generation token SEPARATELY, so a
    server publishing the same key in between put one generation's bytes into the store
    under another generation's token - and the pushing host then believed its local copy
    current for ever."""

    def test_a_publication_during_a_push_cannot_mix_the_two(self):
        self.a.local.put(KEY, self.file("old", b"OLD"), {"v": 1}, {"task": "t",
                                                                   "computed": 1.0})
        real = ResultCache._resolve
        raced = []

        def resolve(local, key, *, lease=True):
            where = real(local, key, lease=lease)
            if not raced:                      # the window: between the resolve and the
                raced.append(key)              # generation the old code read separately
                local.put(KEY, self.file("new", b"NEW"), {"v": 2},
                          {"task": "t", "computed": 2.0})
            return where
        with unittest.mock.patch.object(ResultCache, "_resolve", resolve):
            self.a.push()
        self.assertTrue(raced)
        stored = self.pointer()
        gen, published = stored["generation"], bytes(
            ops.get(self.store,
                        f"pre/blobs/sha256/{stored['files'][RESULT_NAME]['digest'][7:]}"
                        ).bytes())
        served = Path(self.b.get(KEY)[0]).read_bytes()
        self.assertEqual(published, served, "every host reads what the pointer names")
        # whichever generation was pushed, its OWN bytes and its OWN result went with it
        expected = {b"OLD": {"v": 1}, b"NEW": {"v": 2}}[published]
        self.assertEqual(expected, stored["result"])
        self.assertEqual(expected, self.b.get(KEY)[1])
        local_dir = self.a.local._generation_dir(KEY, gen)
        if local_dir.exists():
            self.assertEqual(published, (local_dir / RESULT_NAME).read_bytes(),
                             "the local copy under that token holds those same bytes")


class TestPushRerunsMakeProgress(_Hosts):
    def entries(self, cache, n):
        for i in range(n):
            cache.local.put(f"{i:02d}" * 32, self.file(f"l{i}", f"v{i}".encode()),
                            {"outputs": [f"v{i}"]}, {"task": "t", "computed": float(i)})

    def test_limit_bounds_the_work_not_the_entries_examined(self):
        """`--limit 2` three times over six entries used to migrate two and then keep
        re-skipping the same two for ever."""
        self.entries(self.a, 6)
        for _ in range(3):
            self.a.push(limit=2)
        self.assertEqual(6, len(self.b.list()[0]))

    def test_a_skipped_key_does_not_use_up_a_slot(self):
        self.entries(self.a, 3)
        self.a.push(limit=1)
        got = self.a.push(limit=1)
        self.assertEqual(1, got["pushed"])
        self.assertEqual(2, len(self.b.list()))


class TestPushReportsHonestly(_Hosts):
    def test_an_unreadable_cache_root_is_a_failure_not_silence(self):
        import os
        root = self.a.local.root
        os.chmod(root, 0o000)
        try:
            got = self.a.push()
        finally:
            os.chmod(root, 0o755)
        self.assertEqual(1, got["failed"], "every entry was dropped without being counted")

    def test_a_result_document_that_is_not_a_document_is_refused(self):
        self.a.local.put(KEY, self.file("l", b"x"), {}, {"task": "t", "computed": 1.0})
        where = self.a.local._resolve(KEY, lease=False)
        (where / "result.json").write_text(json.dumps(["not", "a", "document"]))
        self.assertEqual(1, self.a.push()["unreadable"])
        self.assertEqual([], self.b.list()[0])

    def test_replaced_is_counted_when_the_store_had_the_key(self):
        self.publish(self.b, b"theirs")
        self.a.local.put(KEY, self.file("l", b"ours"), {"outputs": ["ours"]},
                         {"task": "t", "computed": 9.0})
        said = []
        got = self.a.push(conflict="newer", report=lambda k, w: said.append(w))
        self.assertEqual(1, got["replaced"])
        self.assertEqual(0, got["pushed"])
        self.assertEqual(["replaced"], said)

    def test_a_refused_key_uploads_nothing_under_any_policy(self):
        self.publish(self.b, b"theirs", key=KEY)
        self.a.local.put(KEY, self.file("l", b"older"), {"outputs": ["older"]},
                         {"task": "t", "computed": 0.5})
        before = len(self.b.blobs.entries())
        with unittest.mock.patch.object(BlobStore, "put_file",
                                        side_effect=AssertionError("hashed and uploaded")):
            self.assertEqual(1, self.a.push(conflict="newer")["skipped"])
        self.assertEqual(before, len(self.b.blobs.entries()))


class TestPullRepairsAndFits(_Hosts):
    def test_a_local_copy_missing_its_files_is_repaired_not_called_current(self):
        self.publish(self.a, b"one")
        self.b.pull()
        (Path(self.b.local.get(KEY)[0])).unlink()
        got = self.b.pull()
        self.assertEqual(1, got["pulled"], "a re-pull is the obvious repair")
        self.assertEqual(b"one", Path(self.b.local.get(KEY)[0]).read_bytes())

    def test_more_entries_than_the_local_bound_is_reported_not_claimed(self):
        small = SharedResultCache(self.store, ResultCache(self.tmp / "small", keep=3),
                                  prefix="pre/", check=False)
        for i in range(6):
            self.publish(self.a, f"v{i}".encode(), key=f"{i:02d}" * 32)
        got = small.pull()
        self.assertEqual(6, got["pulled"] + got["evicted"])
        self.assertGreater(got["evicted"], 0, "a cache that keeps 3 cannot hold 6")
        servable = sum(1 for i in range(6) if small.local.get(f"{i:02d}" * 32))
        self.assertEqual(got["pulled"], servable, "the count is what is actually servable")


class TestDamagedHistoryIsNotLoadBearing(_Hosts):
    def test_a_history_that_is_not_a_list_loses_nothing(self):
        """`"history": null` is what another language emits for "no history". It used to
        lose the current result AND freeze blob deletion for the whole store."""
        for junk in (None, "a string", 7, {"not": "a list"}):
            with self.subTest(history=junk):
                self.publish(self.a, b"one")
                self.as_legacy(history=junk)   # format 1: history was inline
                self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
                self.assertEqual(0, self.b.sweep(grace_s=0)["unreadable_pointers"])
                self.assertEqual(0, self.b.sweep(grace_s=0)["deleted_blobs"])

    def test_a_junk_history_entry_never_occupies_a_slot(self):
        """The writer used a looser rule than the readers, so an entry nothing could read
        rode forward for ever and pushed real predecessors off the end."""
        self.publish(self.a, b"one")
        ptr = self.as_legacy(
            history=[{"generation": "junk", "files": "not a map", "result": {}}] * 3)
        new = self.publish(self.a, b"two")     # converts the format 1 entry to a chain
        self.assertEqual([new, ptr["generation"]],
                         [h["generation"] for h in self.a.history(KEY)],
                         "only the real predecessor survives the conversion")

    def test_an_undatable_generation_is_not_kept_for_ever(self):
        self.publish(self.a, b"one")
        self.as_legacy(published="2026-01-01")  # not a time this code can compare
        self.publish(self.a, b"two")
        self.assertEqual([True], [h["current"] for h in self.a.history(KEY)])

    def test_a_pointer_nothing_can_date_is_never_expired(self):
        """The other direction: cleanup refuses what it cannot account for."""
        self.publish(self.a, b"one")
        self.as_legacy(published="2026-01-01")
        got = self.a.sweep(now=time.time() + 10 ** 9, max_age_s=1)
        self.assertEqual(0, got["expired_pointers"])
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_string_published_does_not_stop_every_sweep(self):
        self.publish(self.a, b"one")
        self.as_legacy(published="2026-09-01T00:00:00")
        self.a.sweep(max_age_s=3600)           # must not raise for any host, ever again


class TestHistoryAgesOutWhenQuiet(_Hosts):
    def test_a_key_left_alone_still_loses_its_old_generations(self):
        """The bound used to be applied only when a key was republished - useless for the
        quiescent keys where a 30-day limit is the point."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        self.assertEqual(2, len(self.a.history(KEY)))
        later = time.time() + objectcache.HISTORY_MAX_AGE_S + 60
        with unittest.mock.patch.object(objectcache.time, "time", lambda: later):
            self.assertEqual([True], [h["current"] for h in self.a.history(KEY)])
        self.assertEqual(0, self.a.sweep(now=later, grace_s=0)["deleted_blobs"],
                         "the bytes wait out the clock-skew margin first")
        past_margin = later + objectcache.HISTORY_GC_MARGIN_S + 60
        self.assertEqual(2, self.a.sweep(now=past_margin, grace_s=0)["deleted_blobs"],
                         "the old labels, and the manifest that was their link")
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual([True], [h["current"] for h in self.b.history(KEY)])

    def test_a_fast_clock_does_not_collect_what_other_hosts_still_list(self):
        """Hosts do not share a clock. One running a day fast would otherwise sweep the
        rollback data every other host can still see (review, 2026-09-20)."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        fast = time.time() + objectcache.HISTORY_MAX_AGE_S + 3600   # a bit past the bound
        self.assertEqual(0, self.a.sweep(now=fast, grace_s=0)["deleted_blobs"])
        self.assertEqual(2, len(self.b.history(KEY)), "still listed on a normal clock")
        old_gen = self.b.history(KEY)[1]["generation"]
        dest = self.tmp / "still-there"
        self.assertIsNotNone(self.b.fetch_generation(KEY, old_gen, dest))

    def test_a_re_push_of_the_same_generation_is_not_its_own_predecessor(self):
        self.a.local.put(KEY, self.file("l", b"same"), {"outputs": ["same"]},
                         {"task": "t", "computed": 1.0})
        self.a.push()
        self.a.push(conflict="force")
        self.a.push(conflict="force")
        got = self.a.history(KEY)
        self.assertEqual(1, len(got), got)
        self.assertEqual(1, len({h["generation"] for h in got}))

    def test_a_read_of_the_present_carries_no_history(self):
        """Format 1 copied every predecessor into the pointer, and every read paid for it
        (so history entries were stripped of meta). A format 2 ref holds the current
        manifest only: its size does not grow with the history behind it."""
        self.publish(self.a, b"v0")
        one = len(json.dumps(self.ref()))
        for i in range(1, objectcache.HISTORY_KEEP + 2):
            self.publish(self.a, f"v{i}".encode())
        self.assertEqual(objectcache.HISTORY_KEEP + 1, len(self.a.history(KEY)))
        self.assertLess(abs(len(json.dumps(self.ref())) - one), 120,
                        "one digest of `replaces`, not a copy of each predecessor")
        self.assertNotIn("v0", self.ref()["body"])


class TestDeleteRemovesTheEntry(_Hosts):
    """`delete` removes the entry - the pointer and this host's copy. The BYTES are
    reclaimed by `haversack cache sweep`, not here (decided 2026-09-20): deciding at delete
    time whether a blob was one a live publication had just deduplicated onto took four
    attempts and reviewers were still finding holes, and a sweep answers the same question
    with nothing else moving."""

    def test_the_entry_goes_everywhere_at_once(self):
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertTrue(self.a.delete(KEY))
        self.assertIsNone(self.b.get(KEY))
        self.assertIsNone(self.a.local.get(KEY), "and this host's own copy with it")

    def test_the_bytes_are_reclaimed_by_a_sweep(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        digest = f"sha256:{hashlib.sha256(b'one').hexdigest()}"
        self.assertTrue(self.a.blobs.has(digest), "still there: a sweep decides, not this")
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.assertEqual(2, self.a.sweep(now=later, grace_s=0,
                                         allow_empty=True)["deleted_blobs"],
                         "the labels and the manifest that named them")
        self.assertFalse(self.a.blobs.has(digest))

    def test_a_sweep_after_a_delete_spares_what_another_entry_shares(self):
        self.publish(self.a, b"shared", key="cd" * 32)
        self.publish(self.a, b"shared")
        self.a.delete(KEY)
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, grace_s=0)
        self.assertEqual(b"shared", Path(self.b.get("cd" * 32)[0]).read_bytes())

    def test_deleting_an_entry_this_version_cannot_PARSE_still_says_it_deleted_it(self):
        """Garbage under a key - a truncated write, a stray object - is nobody's data."""
        self.publish(self.a, b"one")
        ops.put(self.store, f"pre/results/{KEY}.json", b"{not a pointer")
        cold = SharedResultCache(self.store, ResultCache(self.tmp / "cold"), prefix="pre/",
                                 check=False)
        self.assertTrue(cold.delete(KEY), "the operator removed something, and is told so")
        self.assertEqual([], [o for b in ops.list(self.store, f"pre/results/{KEY}")
                              for o in b])

class TestFetchGenerationReportsWhatItWrote(_Hosts):
    def test_a_generation_whose_files_cannot_be_placed_is_not_success(self):
        gen = self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["files"] = {"labels.other.nrrd": ptr["files"][RESULT_NAME]}
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        dest = self.tmp / "out-empty"
        self.assertIsNone(self.b.fetch_generation(KEY, gen, dest))

    def test_it_says_which_files_it_wrote(self):
        gen = self.publish(self.a, b"one", preview=b"png")
        dest = self.tmp / "out-written"
        got = self.b.fetch_generation(KEY, gen, dest)
        self.assertEqual(sorted([RESULT_NAME, "preview.png"]), sorted(got["written"]))
        self.assertTrue((dest / RESULT_NAME).exists())


class TestGapsFromTheSecondMutationRun(_Hosts):
    """Facts the push/pull and history tests left unpinned (mutation run, 2026-09-20)."""

    def local_entry(self, cache, key=KEY, labels=b"ours", **meta):
        return cache.local.put(key, self.file(f"m-{labels.hex()}", labels),
                               {"outputs": [labels.decode()]}, {"task": "t", **meta})

    def test_a_tie_on_computed_keeps_the_store_copy(self):
        self.publish(self.b, b"theirs")        # computed 1.0
        self.local_entry(self.a, computed=1.0)
        self.assertEqual(1, self.a.push(conflict="newer")["skipped"])
        self.assertEqual({"outputs": ["theirs"]}, self.b.published_result(KEY))

    def test_a_missing_timestamp_on_either_side_keeps_the_store_copy(self):
        for theirs, ours in (({"task": "t"}, {"computed": 5.0}),
                             ({"computed": 5.0}, {"task": "t"}),
                             ({"task": "t"}, {"task": "t"})):
            with self.subTest(theirs=theirs, ours=ours):
                self.a.delete(KEY)
                self.b.put(KEY, self.file("t", b"theirs"), {"outputs": ["theirs"]}, theirs)
                self.local_entry(self.a, **ours)
                self.assertEqual(1, self.a.push(conflict="newer")["skipped"],
                                 "unknown is not old")

    def test_a_pushed_pointer_is_published_now_and_keeps_the_compute_time_in_meta(self):
        """`published` is when the pointer was written. Taking the compute time made a
        migrated entry's own history instantly older than the age bound."""
        old = time.time() - 40 * 24 * 3600
        self.local_entry(self.a, computed=old)
        before = time.time()
        self.a.push()
        ptr = self.pointer()
        self.assertGreaterEqual(ptr["published"], before)
        self.assertEqual(old, ptr["meta"]["computed"])

    def test_a_months_old_cache_pushed_over_a_key_keeps_what_it_replaced(self):
        self.publish(self.b, b"theirs")
        self.local_entry(self.a, labels=b"ours", computed=time.time())
        self.assertEqual(1, self.a.push(conflict="force")["replaced"])
        self.assertEqual([{"outputs": ["ours"]}, {"outputs": ["theirs"]}],
                         [h["result"] for h in self.b.history(KEY)])

    def test_an_entry_whose_labels_are_gone_is_unreadable(self):
        self.local_entry(self.a)
        (self.a.local._resolve(KEY, lease=False) / RESULT_NAME).unlink()
        self.assertEqual(1, self.a.push()["unreadable"])
        self.assertIsNone(self.b.get(KEY))

    def test_dotted_directories_are_not_even_examined(self):
        (self.a.local.root / ".fill-abcd").mkdir(parents=True)
        (self.a.local.root / ".reclaim-1").mkdir(parents=True)
        self.local_entry(self.a)
        self.assertEqual({"pushed": 1, "skipped": 0, "replaced": 0, "failed": 0,
                          "unreadable": 0}, self.a.push())

    def test_history_reports_sizes_and_file_names(self):
        self.publish(self.a, b"one", preview=b"png")
        got = self.a.history(KEY)[0]
        self.assertEqual(len(b"one") + len(b"png"), got["bytes"])
        self.assertEqual(sorted([RESULT_NAME, "preview.png"]), got["files"])

    def test_fetch_generation_honors_the_allowlist(self):
        for fmt in ("format 1", "format 2"):
            with self.subTest(fmt=fmt):
                gen = self.publish(self.a, b"one")
                ptr = self.pointer()
                files = {**ptr["files"], "../escaped-near": ptr["files"][RESULT_NAME]}
                if fmt == "format 1":
                    self.as_legacy(files=files)
                else:
                    self.write_manifest(files=files)   # a manifest is another host's too
                dest = self.tmp / f"dest-allow-{fmt[-1]}"
                got = self.b.fetch_generation(KEY, gen, dest)
                self.assertFalse((dest.parent / "escaped-near").exists(), "wrote outside dest")
                self.assertEqual([RESULT_NAME], got["written"], "and does not claim it")

    def test_fetch_generation_of_a_swept_generation_is_none(self):
        gen = self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        ops.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        dest = self.tmp / "dest-swept"
        dest.mkdir()
        self.assertIsNone(self.b.fetch_generation(KEY, gen, dest))

    def test_fetch_generation_leaves_the_local_cache_untouched(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        before = sorted(str(p.relative_to(self.b.local.root))
                        for p in self.b.local.root.rglob("*"))
        self.b.fetch_generation(KEY, gen, self.tmp / "dest-untouched")
        after = sorted(str(p.relative_to(self.b.local.root))
                       for p in self.b.local.root.rglob("*"))
        self.assertEqual([p for p in before if ".lease" not in p],
                         [p for p in after if ".lease" not in p])


class TestSweepOrdering(_Hosts):
    def test_a_publication_during_a_sweep_keeps_its_blobs(self):
        """Blobs are listed BEFORE the pointers are read, so a publication landing after
        that listing was never a candidate - whatever the two clocks say. Letting the
        sweep list for itself lost exactly this blob at grace_s=0 (review, 2026-09-20)."""
        self.publish(self.a, b"already here")
        real = SharedResultCache._scan_pointers
        raced = []

        def scan(cache, **kw):
            got = real(cache, **kw)
            if not raced:                      # a second host publishes mid-sweep
                raced.append(True)
                self.publish(self.b, b"mid-sweep", key="cd" * 32)
            return got
        with unittest.mock.patch.object(SharedResultCache, "_scan_pointers", scan):
            self.a.sweep(grace_s=0)
        self.assertTrue(raced)
        # asked of the STORE, and of a host with no local copy: the publishing host serves
        # from its own disk and would pass this test with the blob deleted (mutation run,
        # 2026-09-20)
        self.assertTrue(self.a.blobs.has(
            f"sha256:{hashlib.sha256(b'mid-sweep').hexdigest()}"))
        self.assertEqual(b"mid-sweep", Path(self.host("cold").get("cd" * 32)[0]).read_bytes())

    def test_a_deduplicated_publication_is_not_swept_from_under_itself(self):
        """The candidate list cannot save bytes that were ALREADY stored: a recomputation
        producing identical output uploads nothing, so the blob is old while the pointer
        naming it is new. provender refreshes the timestamp on dedup; without that this
        loses a result that was just computed (review, 2026-09-20)."""
        self.publish(self.a, b"unrelated", key="ee" * 32)   # a live key, so the sweep runs
        self.publish(self.a, b"identical", key="cd" * 32)
        self.a.delete("cd" * 32)                       # its blob is now an orphan
        real = SharedResultCache._scan_pointers
        raced = []

        def scan(cache, **kw):
            got = real(cache, **kw)
            if not raced:
                raced.append(True)
                self.publish(self.b, b"identical")     # dedupes onto the orphan
            return got
        with unittest.mock.patch.object(SharedResultCache, "_scan_pointers", scan):
            got = self.a.sweep(grace_s=0)
        self.assertTrue(raced)
        self.assertEqual(0, got["unreadable_pointers"], "the sweep really ran")
        self.assertEqual(b"identical", Path(self.host("cold2").get(KEY)[0]).read_bytes())

    def test_an_index_that_is_not_where_we_looked_sweeps_nothing(self):
        """No pointers FOUND is not evidence that nothing is live - a drifted prefix, a
        layout change. provender refuses; haversack used to switch that refusal off."""
        self.publish(self.a, b"precious")
        for batch in ops.list(self.store, "pre/results/"):
            for obj in batch:
                body = bytes(ops.get(self.store, obj["path"]).bytes())
                ops.put(self.store, obj["path"].replace("results/", "entries/"), body)
                ops.delete(self.store, obj["path"])
        got = self.a.sweep(grace_s=0)
        self.assertEqual(0, got["deleted_blobs"])
        self.assertTrue(self.a.blobs.has(
            f"sha256:{hashlib.sha256(b'precious').hexdigest()}"))

    def test_an_empty_store_is_not_an_error(self):
        self.assertEqual({"expired_pointers": 0, "deleted_blobs": 0, "already_gone": 0,
                          "unreadable_pointers": 0}, self.a.sweep(grace_s=0))


class TestCorruptBlobIsReported(_Hosts):
    def test_a_blob_that_does_not_hash_to_its_name_says_so(self):
        """A corrupt blob reads exactly like a cold cache unless someone says otherwise;
        the message lived in the blob code and was lost when it moved to provender."""
        import contextlib
        import io
        self.publish(self.a, b"one")
        ops.put(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}",
                    b"tampered")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertIsNone(self.b.get(KEY))
        self.assertIn("does not hash to its name", err.getvalue())


class TestStoreUrlErrors(unittest.TestCase):
    def test_a_malformed_url_names_the_forms_it_accepts(self):
        from haversack.errors import InputError
        with self.assertRaises(InputError) as cm:
            open_store("https://example.org/x")
        self.assertIn("s3://bucket", str(cm.exception))
        self.assertNotIn("credential", str(cm.exception).lower())


class TestGapsFromTheThirdMutationRun(_Hosts):
    def test_a_dated_history_entry_with_damaged_files_is_still_refused(self):
        """Every junk entry in the earlier tests was also undated, so the date rule alone
        dropped it and the files rule was never exercised."""
        self.publish(self.a, b"one")
        self.as_legacy(history=[{"generation": "junk", "files": "not a map",
                                 "published": time.time()},
                                {"generation": 7, "files": {}, "published": time.time()}])
        self.assertEqual(1, len(self.a.history(KEY)))

    def test_pull_keeps_the_newest_when_the_local_bound_is_reached(self):
        small = SharedResultCache(self.store, ResultCache(self.tmp / "small3", keep=3),
                                  prefix="pre/", check=False)
        for i in range(6):
            self.a.put(f"{i:02d}" * 32, self.file(f"l{i}", f"v{i}".encode()), {},
                       {"computed": float(i)})
        small.pull()
        kept = {i for i in range(6) if small.local.get(f"{i:02d}" * 32)}
        self.assertEqual({3, 4, 5}, kept, "oldest first, so the newest survive eviction")

    def test_the_refusal_to_sweep_an_absent_index_says_so(self):
        import contextlib
        import io
        self.publish(self.a, b"precious")
        for batch in ops.list(self.store, "pre/results/"):
            for obj in batch:
                ops.delete(self.store, obj["path"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.a.sweep(grace_s=0)
        self.assertIn("no entries under", err.getvalue())
        self.assertIn("--empty-index-ok", err.getvalue(),
                      "and names the flag that would mean it on purpose")

    def test_pull_names_the_key_that_failed(self):
        self.publish(self.a, b"one")
        ops.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        said = []
        got = self.b.pull(report=lambda k, w: said.append((k, w)))
        self.assertEqual(1, got["failed"])
        self.assertEqual([(KEY, "failed")], said, "an operator needs the key to act on")

    def test_a_repaired_copy_gets_its_documents_back(self):
        """A repair that placed only files left a hit whose result.json was missing, which
        `ResultCache.get` serves as `{}` - a 200 with no outputs."""
        self.publish(self.a, b"one")
        self.b.pull()
        where = Path(self.b.local.get(KEY)[0]).parent
        (where / "result.json").unlink()
        got = self.b.pull()
        self.assertEqual(1, got["pulled"], "incomplete is not current")
        self.assertEqual({"outputs": ["one"]}, self.b.local.get(KEY)[1])


class TestAnOrphanedGenerationIsAdopted(_Hosts):
    """A generation directory that is here but not current: a crash between its rename and
    the pointer write, a second process placing it, or a pointer that moved back to it.
    Returning a miss left the key unreadable on that host FOR EVER - every read
    re-downloaded, repaired, and still answered None - and on a compute server that miss
    became a recompute that overwrote the generation an operator had just rolled back to."""

    def orphan(self, cache, gen):
        """The state a crash between the two steps leaves."""
        (cache.local.root / KEY / CURRENT_NAME).unlink()

    def test_a_crash_between_the_rename_and_the_pointer_is_repaired_on_the_next_read(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        self.orphan(self.b, gen)
        self.assertIsNone(self.b.local.get(KEY), "the local copy is unreachable")
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit, "and the next read fixes it")
        self.assertEqual(b"one", Path(hit[0]).read_bytes())
        self.assertEqual(gen, self.b.local.generation(KEY))

    def test_it_is_adopted_without_downloading_anything(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        self.orphan(self.b, gen)
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_an_incomplete_directory_is_not_adopted(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        self.orphan(self.b, gen)
        (self.b.local._generation_dir(KEY, gen) / "result.json").unlink()
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual({"outputs": ["one"]}, self.b.get(KEY)[1],
                         "it is filled properly rather than published half-formed")

    def test_a_rollback_survives_a_host_that_still_holds_the_newer_generation(self):
        """The workflow the history decision exists for: host x served A then B; an
        operator rolls the store back to A; x must serve A, not recompute over it."""
        gen_a = self.publish(self.a, b"version A")
        self.b.get(KEY)                        # x holds A
        self.publish(self.a, b"version B")
        self.b.get(KEY)                        # x holds B, and still has A on disk
        self.a._swap(KEY, lambda cur: {"publication": gen_a, "meta": cur["meta"],
                                       "files": {RESULT_NAME: self.a.blobs.put_file(
                                           self.file("a-again", b"version A"))},
                                       "result": {"outputs": ["version A"]},
                                       "published": time.time()})
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit, "the rollback is readable on the host that had B")
        self.assertEqual(b"version A", Path(hit[0]).read_bytes())


class TestTheStoreIsNotInTheAvailabilityPath(_Hosts):
    def faulty(self):
        from obstore.exceptions import GenericError
        real = ops.get

        def boom(store, path, *a, **kw):
            if "results/" in str(path):
                raise GenericError("the bucket is unreachable")
            return real(store, path, *a, **kw)
        return unittest.mock.patch.object(ops, "get", boom)

    def test_a_host_serves_its_own_complete_copy_while_the_store_is_down(self):
        self.publish(self.a, b"one")
        self.b.get(KEY)                        # b now holds it
        with self.faulty():
            hit = self.b.get(KEY)
        self.assertIsNotNone(hit, "a warm cache is not thrown away by an outage")
        self.assertEqual(b"one", Path(hit[0]).read_bytes())

    def test_a_host_that_holds_nothing_still_answers_a_miss(self):
        self.publish(self.a, b"one")
        with self.faulty():
            self.assertIsNone(self.b.get(KEY))

    def test_a_publication_the_store_refused_keeps_its_bytes_here(self):
        """The segmentation is finished and on this disk: failing the job is right, losing
        the work is not."""
        from obstore.exceptions import GenericError
        with unittest.mock.patch.object(ops, "put",
                                        side_effect=GenericError("bucket down")):
            with self.assertRaises(GenericError):
                self.publish(self.a, b"expensive")
        self.assertEqual(b"expensive", Path(self.a.local.get(KEY)[0]).read_bytes())


class TestLocalWriteFailuresAreMisses(_Hosts):
    def test_a_full_disk_during_a_fill_is_a_miss_not_an_exception(self):
        self.publish(self.a, b"one")
        with unittest.mock.patch.object(ResultCache, "put",
                                        side_effect=OSError(28, "No space left on device")):
            self.assertIsNone(self.b.get(KEY))

    def test_a_cache_directory_that_cannot_be_written_is_a_miss(self):
        self.publish(self.a, b"one")
        with unittest.mock.patch.object(SharedResultCache, "_fresh_work_dir",
                                        side_effect=OSError(30, "Read-only file system")):
            self.assertIsNone(self.b.get(KEY))


class TestASlowFillIsNotReaped(_Hosts):
    def test_a_dead_process_work_directory_goes_at_once(self):
        """A reader killed mid-fill left its download for an hour, where `cache usage` and
        `cache clean` cannot see it (configuration sweep, 2026-09-20)."""
        import os
        dead = self.b.local.root / f"{objectcache.WORK_PREFIX}999999-abcd"
        dead.mkdir(parents=True)
        (dead / "half-a-download").write_bytes(b"x" * 1000)
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertFalse(dead.exists(), "its process is gone: no need to wait out an hour")

    def test_a_work_directory_belonging_to_a_live_process_survives(self):
        """Age is not liveness - this repo's own rule. A fill slower than the grace had
        its work deleted under it and reported the blob gone."""
        import os
        live = self.b.local.root / f"{objectcache.WORK_PREFIX}{os.getpid()}-abcd"
        live.mkdir(parents=True)               # running now: newer than WORK_GRACE_S
        dead = self.b.local.root / f"{objectcache.WORK_PREFIX}999999-dead"
        dead.mkdir(parents=True)
        os.utime(dead, (0, 0))
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertTrue(live.exists(), "a live fill's work is not reaped by age")
        self.assertFalse(dead.exists(), "a dead one's is")

    def test_a_pid_that_looks_alive_but_is_ancient_is_reaped_anyway(self):
        """On a directory two HOSTS share, a pid means nothing across the boundary - and a
        number gets reused. Past the grace, age is the only thing left to judge by, and the
        comment used to promise that while the code did not do it (review, 2026-09-20)."""
        import os
        ancient = self.b.local.root / f"{objectcache.WORK_PREFIX}{os.getpid()}-ancient"
        ancient.mkdir(parents=True)
        os.utime(ancient, (0, 0))
        self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertFalse(ancient.exists())


class TestFillsHandOverRatherThanCopy(_Hosts):
    """A fill downloaded into its work directory and then COPIED into place, so it needed
    twice the result's size free and read every byte twice. Measured on a 12 MB disk image
    with a 6 MB result: it failed with 11 MB free (configuration sweep, 2026-09-20)."""

    def test_the_downloaded_file_is_moved_not_copied(self):
        self.publish(self.a, b"a result")
        moved = []
        real = ResultCache.put

        def watch(local, key, labels_path, *a, **kw):
            moved.append(kw.get("move"))
            return real(local, key, labels_path, *a, **kw)
        with unittest.mock.patch.object(ResultCache, "put", watch):
            self.b.get(KEY)
        self.assertEqual([True], moved)

    def test_the_work_directory_is_empty_afterwards(self):
        self.publish(self.a, b"a result", preview=b"png")
        self.b.get(KEY)
        self.assertEqual([], sorted(self.b.local.root.glob(f"{objectcache.WORK_PREFIX}*")))

    def test_a_publication_that_fails_leaves_no_empty_entry(self):
        with unittest.mock.patch("shutil.copy2", side_effect=OSError(28, "No space")):
            with self.assertRaises(OSError):
                self.a.local.put(KEY, self.file("l", b"x"), {}, {})
        self.assertFalse((self.a.local.root / KEY).exists(),
                         "an empty entry directory counts against the cache's bound and "
                         "resolves to nothing")


# -- the fifth review round (2026-09-20): what the previous round's fixes broke -----------


class TestAStaleClaimDoesNotStrandAKey(_Hosts):
    """A filler SIGKILLed between its claim and its release left `<key>/.writer-<gen>`
    behind. Every later fill on that host then read "someone is placing it" and missed -
    for ever on a host that never computes, because only a successful publication of that
    key prunes the claim."""

    def stale_claim(self, cache, gen, *, host=None, pid=999999, state="locked"):
        import socket
        d = cache.local.root / KEY
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ResultCache.CLAIM_PREFIX}{gen}").write_text(
            f"{host or socket.gethostname()}\n{pid}\n{state}\n")

    def test_a_dead_writers_claim_is_cleared_and_the_key_is_served(self):
        gen = self.publish(self.a, b"one")
        self.stale_claim(self.b, gen)
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(gen, self.b.local.generation(KEY))

    def test_a_LIVE_writers_claim_is_still_respected(self):
        """A live writer HOLDS the lock, which is the whole proof: it is not takeable, so
        death cannot be concluded and the claim stands."""
        import os

        from haversack import filelock
        gen = self.publish(self.a, b"one")
        self.stale_claim(self.b, gen, pid=os.getpid())
        held = os.open(self.b.local.root / KEY / f"{ResultCache.CLAIM_PREFIX}{gen}",
                       os.O_RDWR)
        try:
            filelock.lock(held, blocking=False)
            self.assertIsNone(self.b.get(KEY), "another process really is placing it")
        finally:
            os.close(held)

    def test_a_claim_from_another_host_is_left_alone(self):
        """An advisory lock need not reach across machines, so this host cannot judge it -
        and must not delete another host's work to find out."""
        gen = self.publish(self.a, b"one")
        self.stale_claim(self.b, gen, host="some-other-machine")
        self.assertIsNone(self.b.get(KEY))
        self.assertTrue((self.b.local.root / KEY
                         / f"{ResultCache.CLAIM_PREFIX}{gen}").exists())


class TestAdoptChecksSizes(_Hosts):
    def test_a_truncated_generation_is_not_adopted(self):
        """A power loss behind a rename leaves the directory with zero-length files, which
        is exactly the case adoption was written for - it published one as a 200."""
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        where = self.b.local._generation_dir(KEY, gen)
        (self.b.local.root / KEY / CURRENT_NAME).unlink()
        (where / RESULT_NAME).write_bytes(b"")          # truncated by the power loss
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit)
        self.assertEqual(b"one", Path(hit[0]).read_bytes(),
                         "it is refilled rather than published empty")

    def test_a_complete_generation_is_still_adopted_without_downloading(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        (self.b.local.root / KEY / CURRENT_NAME).unlink()
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(gen, self.b.local.generation(KEY))


class TestTheOutageFallbackIsBounded(_Hosts):
    def faulty(self):
        from obstore.exceptions import GenericError
        real = ops.get

        def boom(store, path, *a, **kw):
            if "results/" in str(path):
                raise GenericError("unreachable")
            return real(store, path, *a, **kw)
        return unittest.mock.patch.object(ops, "get", boom)

    def test_a_recently_confirmed_copy_is_served(self):
        self.publish(self.a, b"one")
        self.b.get(KEY)
        with self.faulty():
            self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_copy_nobody_has_confirmed_lately_is_a_miss(self):
        """A host in an outage cannot know an entry was deleted elsewhere, and deletion is
        the one thing this design promises means gone - so the fallback is bounded by when
        this host last saw the entry alive."""
        import os
        self.publish(self.a, b"one")
        hit = self.b.get(KEY)
        stale = time.time() - objectcache.OUTAGE_GRACE_S - 60
        os.utime(Path(hit[0]).parent / objectcache.CONFIRMED_NAME, (stale, stale))
        with self.faulty():
            self.assertIsNone(self.b.get(KEY))

    def test_the_publishing_host_counts_as_having_confirmed_it(self):
        self.publish(self.a, b"one")
        with self.faulty():
            self.assertEqual(b"one", Path(self.a.get(KEY)[0]).read_bytes())


class TestWorkKeptOnlyForAnOutage(_Hosts):
    def test_a_refused_publication_keeps_nothing(self):
        """A refusal is deliberate and repeatable; keeping a copy for it only fills the
        disk with generations no read will ever reach."""
        self.publish(self.a, b"one")
        newer = {**self.pointer(), "format": objectcache.POINTER_FORMAT + 1}
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(newer).encode())
        before = self.b.local.generation(KEY)
        with self.assertRaises(ObjectStoreUnsuitable):
            self.publish(self.b, b"two")
        self.assertEqual(before, self.b.local.generation(KEY), "no generation was kept")


class TestADeleteRefusalCancelsNothing(_Hosts):
    def test_the_delete_is_decided_before_anything_is_cancelled(self):
        """A 409 that has already killed a running compute is work thrown away for a
        deletion that did not happen."""
        import inspect

        from haversack import serve as serve_mod
        src = inspect.getsource(serve_mod)
        for chunk in src.split("cache_delete(key)")[:-1]:
            tail = chunk[-600:]
            self.assertNotIn("executor.cancel(", tail,
                             "the cancel must follow the delete, not precede it")


class TestTheServerSweepsItsStore(_Hosts):
    """`delete` leaves bytes for a sweep and a republication leaves its predecessor's, so a
    store nothing sweeps only grows - and an operator who has to remember a cron line will
    not. The server that writes to a store tidies it (2026-09-20)."""

    def executor(self, **kw):
        from haversack.serve import LocalExecutor
        from test_job_result_cache import _Segmenter
        return LocalExecutor(_Segmenter(steps=1), workdir=self.tmp / f"w{len(kw)}",
                             cache_dir=self.tmp / f"c{len(kw)}", result_store=self.store,
                             **kw)

    def test_it_sweeps_on_its_timer(self):
        import threading
        swept = threading.Event()
        with unittest.mock.patch.object(
                SharedResultCache, "sweep",
                side_effect=lambda **kw: swept.set() or {"deleted_blobs": 0,
                                                         "expired_pointers": 0,
                                                         "already_gone": 0,
                                                         "unreadable_pointers": 0}):
            ex = self.executor(sweep_interval_h=0.0004)          # ~1.4 s
            try:
                self.assertTrue(swept.wait(20), "the scheduled sweep never ran")
            finally:
                ex.close()

    def test_a_failing_sweep_does_not_take_the_server_with_it(self):
        import threading
        tries = []
        done = threading.Event()

        def boom(**kw):
            tries.append(1)
            if len(tries) >= 2:
                done.set()
            raise RuntimeError("the store is unreachable")
        with unittest.mock.patch.object(SharedResultCache, "sweep", side_effect=boom):
            ex = self.executor(sweep_interval_h=0.0004)
            try:
                self.assertTrue(done.wait(20), "it stopped after the first failure")
                self.assertTrue(ex._sweeper.is_alive())
            finally:
                ex.close()

    def test_it_is_off_without_a_store_and_when_asked_for_zero(self):
        from haversack.serve import LocalExecutor
        from test_job_result_cache import _Segmenter
        plain = LocalExecutor(_Segmenter(steps=1), workdir=self.tmp / "wp",
                              cache_dir=self.tmp / "cp")
        try:
            self.assertIsNone(getattr(plain, "_sweeper", None))
        finally:
            plain.close()
        off = self.executor(sweep_interval_h=0)
        try:
            self.assertIsNone(off._sweeper)
        finally:
            off.close()

    def test_closing_the_server_stops_it_promptly(self):
        ex = self.executor(sweep_interval_h=24)
        sweeper = ex._sweeper
        self.assertTrue(sweeper.is_alive())
        ex.close()
        sweeper.join(10)
        self.assertFalse(sweeper.is_alive(), "it waits on the server's own condition")


class TestFindingAGenerationByItsBytes(_Hosts):
    """`result:<key>` pins the referenced output's digest at submit and resolves again in
    the worker, refusing other bytes (main, 2026-09-20). On one machine a lease keeps the
    pinned generation; across machines it cannot, so the question becomes "which generation
    has these bytes" - which bounded history can answer."""

    def digest(self, labels: bytes) -> str:
        return f"sha256:{hashlib.sha256(labels).hexdigest()}"

    def test_the_pinned_generation_survives_another_hosts_republication(self):
        gen1 = self.publish(self.a, b"the referenced result")
        self.publish(self.b, b"something else")          # another host moves the key on
        found = self.a.find_generation(KEY, self.digest(b"the referenced result"))
        self.assertEqual(gen1, found)
        dest = self.tmp / "pinned"
        got = self.a.fetch_generation(KEY, found, dest)
        self.assertIsNotNone(got)
        self.assertEqual(b"the referenced result", (dest / RESULT_NAME).read_bytes())

    def test_a_sweep_spares_it_while_history_lists_it(self):
        self.publish(self.a, b"the referenced result")
        self.publish(self.b, b"something else")
        self.a.sweep(grace_s=0)
        self.assertIsNotNone(
            self.a.find_generation(KEY, self.digest(b"the referenced result")))

    def test_bytes_no_kept_generation_published_are_not_found(self):
        self.publish(self.a, b"one")
        self.assertIsNone(self.a.find_generation(KEY, self.digest(b"never published")))
        for i in range(objectcache.HISTORY_KEEP + 2):    # push the first off the end
            self.publish(self.a, f"v{i}".encode())
        self.assertIsNone(self.a.find_generation(KEY, self.digest(b"one")))

    def test_the_current_publication_answers_for_its_own_digest(self):
        gen = self.publish(self.a, b"current")
        self.assertEqual(gen, self.a.find_generation(KEY, self.digest(b"current")))

    def test_a_deleted_entry_answers_nothing(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        self.assertIsNone(self.a.find_generation(KEY, self.digest(b"one")))


class TestTheListingContract(_Hosts):
    """`ResultCache.list`'s contract (main, 2026-09-21), answered from the store. Its three
    mechanisms carry over, and two get cheaper: a bucket listing hands out names AND times,
    so paging costs no stats; and the pointer IS the content, so a row is one request where
    a filesystem pays a read and three stats."""

    def some(self, n, task="t"):
        keys = []
        for i in range(n):
            key = f"{i:02x}" * 32
            keys.append(key)
            self.a.put(key, self.file(f"l{i}", f"v{i}".encode()), {"outputs": [f"v{i}"]},
                       {"task": task, "identity": [f"upload:{i}"], "options": {},
                        "computed": float(i)})
        return keys

    def test_rows_carry_what_the_local_cache_carries(self):
        self.some(1)
        rows, _ = self.b.list()
        row = rows[0]
        self.assertEqual({"key", "task", "identity", "options", "computed", "published",
                          "bytes"}, set(row) - {"links"})
        self.assertEqual("t", row["task"])
        self.assertEqual(2, row["bytes"])
        self.assertLessEqual(row["published"], time.time() + 1)

    def test_newest_published_first(self):
        keys = self.some(4)
        rows, _ = self.b.list()
        self.assertEqual(len(keys), len(rows))
        stamps = [r["published"] for r in rows]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_paging_resumes_after_the_position_it_returned(self):
        self.some(5)
        seen, after, pages = [], None, 0
        while True:
            rows, after = self.b.list(limit=2, after=after)
            seen += [r["key"] for r in rows]
            pages += 1
            if after is None:
                break
        self.assertEqual(5, len(seen), f"{pages} pages")
        self.assertEqual(len(set(seen)), len(seen), "no entry appears twice")

    def test_a_publication_during_paging_does_not_shift_a_page(self):
        """A position is (stamp, key), not an offset: a new entry sorts ahead of every
        position already handed out."""
        self.some(4)
        first, after = self.b.list(limit=2)
        self.publish(self.a, b"brand new", key="ff" * 32)
        rest, _ = self.b.list(limit=10, after=after)
        self.assertNotIn("ff" * 32, [r["key"] for r in rest],
                         "it belongs ahead of this page, not inside it")
        self.assertEqual(4, len({*[r["key"] for r in first], *[r["key"] for r in rest]}))

    def test_keys_reads_only_the_names_it_was_given(self):
        keys = self.some(4)
        asked = []
        real = SharedResultCache._stamp

        def stamp(cache, key):
            asked.append(key)
            return real(cache, key)
        with unittest.mock.patch.object(SharedResultCache, "_stamp", stamp), \
                unittest.mock.patch.object(SharedResultCache, "_stamps",
                                           side_effect=AssertionError("listed the bucket")):
            rows, _ = self.b.list(keys=[keys[1], "ab" * 32])
        self.assertEqual([keys[1]], [r["key"] for r in rows])
        self.assertEqual({keys[1], "ab" * 32}, set(asked), "a miss is one HEAD, not a search")

    def test_keys_refuses_a_name_that_is_not_a_key(self):
        self.some(1)
        rows, _ = self.b.list(keys=["../escape", ".hidden", ""])
        self.assertEqual([], rows)

    def test_match_is_asked_of_meta_before_a_row_is_built(self):
        self.some(3, task="ts.v2:total")
        self.publish(self.a, b"other", key="ee" * 32)
        rows, _ = self.b.list(match=lambda f: f.get("task") == "ts.v2:total")
        self.assertEqual(3, len(rows))
        self.assertTrue(all(r["task"] == "ts.v2:total" for r in rows))

    def test_accept_filters_the_finished_row(self):
        self.some(4)
        rows, _ = self.b.list(accept=lambda r: r["bytes"] == 2)
        self.assertEqual(4, len(rows))
        rows, _ = self.b.list(accept=lambda r: False)
        self.assertEqual([], rows)

    def test_a_memo_spares_the_re_read(self):
        from haversack.serve import ListingMemo
        self.some(3)
        memo = ListingMemo()
        self.b.list(memo=memo)
        reads = []
        real = SharedResultCache._read_pointer

        def read(cache, key, **kw):
            reads.append(key)
            return real(cache, key, **kw)
        with unittest.mock.patch.object(SharedResultCache, "_read_pointer", read):
            rows, _ = self.b.list(memo=memo)
        self.assertEqual(3, len(rows))
        self.assertEqual([], reads, "the publications are unchanged: nothing to re-read")

    def test_a_republication_is_read_again_despite_the_memo(self):
        from haversack.serve import ListingMemo
        keys = self.some(2)
        memo = ListingMemo()
        self.b.list(memo=memo)
        time.sleep(0.01)
        self.a.put(keys[0], self.file("new", b"republished"), {"outputs": ["new"]},
                   {"task": "changed", "identity": ["upload:0"], "options": {},
                    "computed": 9.0})
        rows, _ = self.b.list(memo=memo)
        task = next(r["task"] for r in rows if r["key"] == keys[0])
        self.assertEqual("changed", task, "a new stamp is a new entry to the memo")

    def test_the_hold_is_held_around_the_reads(self):
        import contextlib
        self.some(3)
        held = []

        @contextlib.contextmanager
        def hold():
            held.append("in")
            try:
                yield
            finally:
                held.append("out")
        self.b.list(hold=hold)
        self.assertTrue(held and held[0] == "in" and held[-1] == "out")

    def test_an_entry_whose_pointer_is_damaged_is_skipped_not_raised(self):
        keys = self.some(3)
        ops.put(self.store, f"pre/results/{keys[1]}.json", b"{not a pointer")
        rows, _ = self.b.list()
        self.assertEqual(2, len(rows))
        self.assertNotIn(keys[1], [r["key"] for r in rows])


class TestRefsAndManifests(_Hosts):
    """Format 2 (step 3 of the from-scratch design, 2026-09-23): one ref per key naming an
    immutable manifest; history is the manifests' `replaces` chain; a late artifact is an
    AMENDING manifest of the same publication; a deletion is a tombstone."""

    def test_a_ref_names_its_manifest_and_carries_its_exact_bytes(self):
        self.publish(self.a, b"one")
        ref = self.ref()
        self.assertEqual({"format", "manifest", "body"}, set(ref))
        self.assertEqual(ref["manifest"], objectcache._digest_of(ref["body"].encode()))
        self.assertTrue(self.a.blobs.has(ref["manifest"]), "the object is the authority")

    def test_a_read_of_the_present_is_one_request(self):
        self.publish(self.a, b"one")
        paths = []
        real = ops.get

        def counting(store, path, *a, **kw):
            paths.append(path)
            return real(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "get", counting):
            self.assertEqual({"outputs": ["one"]}, self.b.published_result(KEY))
        self.assertEqual([f"pre/results/{KEY}.json"], paths)

    def test_a_wrong_inline_copy_is_not_believed(self):
        """The inline copy is a convenience; the digest decides. A copy that does not hash
        to the name is ignored and the object read instead."""
        self.publish(self.a, b"one")
        ref = self.ref()
        body = json.loads(ref["body"])
        body["result"] = {"outputs": ["forged"]}
        ops.put(self.store, f"pre/results/{KEY}.json",
                json.dumps({**ref, "body": json.dumps(body)}).encode())
        self.assertEqual({"outputs": ["one"]}, self.b.published_result(KEY))

    def test_a_ref_may_not_borrow_another_keys_manifest(self):
        other = "cd" * 32
        self.publish(self.a, b"theirs", key=other)
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(self.ref(other)).encode())
        self.assertIsNone(self.b.get(KEY), "a manifest is bound to its key")

    def test_a_token_may_not_choose_a_local_path(self):
        """A generation token becomes `g-<token>` on this disk. Found while writing step 3:
        format 1 checked only that it was a non-empty string."""
        def everything():
            return {p for p in self.tmp.rglob("*")
                    if not p.is_relative_to(self.tmp / "local-b" / KEY)
                    and not p.is_relative_to(self.tmp / "disk-store")}
        # `g-<token>` sits in the key's directory: three `..` climb out of the local root's
        # key directory and name a sibling of it - still inside this test's tmp
        bad = "../../../x-escaped"
        for fmt in ("format 1", "format 2"):
            with self.subTest(fmt=fmt):
                self.publish(self.a, b"one")   # a sound entry to damage, each time
                if fmt == "format 1":
                    self.as_legacy(generation=bad)
                else:
                    self.write_manifest(publication=bad)
                self.b.local.delete(KEY)
                self.assertEqual("unreadable", self.b._read_ref(KEY)[0],
                                 "refused as data, before any path is built from it")
                before = everything()
                self.assertIsNone(self.b.get(KEY))
                self.assertEqual(set(), everything() - before,
                                 "a token from the store created a path of its choosing")

    def test_a_manifest_swept_before_its_ref_is_written_is_put_back(self):
        """The manifest was already stored (a republication of identical content), and a
        sweep took it in the window before the ref named it. The ref's inline copy keeps
        the PRESENT readable regardless - but the next publication's history walk needs
        the object, so the check after the write puts it back."""
        self.publish(self.a, b"one")
        real = ops.put

        def sweep_in_the_window(store, path, *a, **kw):
            if "results/" in str(path):
                body = json.loads(bytes(a[0]) if a else kw["file"])
                ops.delete(store, self.a.blobs.path(body["manifest"]))
            return real(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "put", sweep_in_the_window):
            self.publish(self.a, b"two")
        self.assertTrue(self.a.blobs.has(self.ref()["manifest"]))
        self.publish(self.a, b"three")
        self.assertEqual(3, len(self.b.history(KEY)), "the chain is whole")

    def test_late_artifacts_amend_the_publication_they_were_rendered_for(self):
        gen = self.publish(self.a, b"one")
        self.b.get(KEY)
        self.assertTrue(self.a.add_artifact(KEY, "preview.png", self.file("p", b"png"),
                                            generation=gen))
        self.assertTrue(self.a.add_artifact(KEY, "statistics.json", self.file("s", b"{}"),
                                            generation=gen),
                        "the second artifact names the SAME publication and must land")
        ptr = self.pointer()
        self.assertEqual(gen, ptr["generation"])
        self.assertTrue(ptr["amends"])
        self.assertEqual({RESULT_NAME, "preview.png", "statistics.json"}, set(ptr["files"]))
        self.assertEqual([gen], [h["generation"] for h in self.b.history(KEY)],
                         "one publication, however many manifests it took")
        real, fetched = BlobStore.fetch, []

        def fetch(blobs, digest, dest):
            fetched.append(digest)
            return real(blobs, digest, dest)
        with unittest.mock.patch.object(BlobStore, "fetch", fetch):
            hit = self.b.get(KEY)
        self.assertNotIn(ptr["files"][RESULT_NAME]["digest"], fetched,
                         "an amendment keeps the token, so the labels stay current")
        self.assertEqual(2, len(fetched), "the two artifacts, and nothing else")
        self.assertTrue((Path(hit[0]).parent / "preview.png").exists())

    def test_an_artifact_for_a_superseded_publication_is_refused(self):
        old = self.publish(self.a, b"one")
        self.publish(self.b, b"two")
        self.assertFalse(self.a.add_artifact(KEY, "preview.png", self.file("p", b"png"),
                                             generation=old))
        self.assertNotIn("preview.png", self.pointer()["files"])

    def test_history_counts_publications_not_amendments(self):
        for i in range(objectcache.HISTORY_KEEP + 2):
            gen = self.publish(self.a, f"v{i}".encode())
            self.a.add_artifact(KEY, "preview.png", self.file(f"p{i}", f"png{i}".encode()),
                                generation=gen)
        got = self.b.history(KEY)
        self.assertEqual(objectcache.HISTORY_KEEP + 1, len(got))
        self.assertTrue(all("preview.png" in h["files"] for h in got),
                        "each publication as its NEWEST manifest has it")

    def test_a_sweep_keeps_every_link_history_walks(self):
        """Amending manifests are links too: sweeping one would end the walk there and
        orphan everything older, a sweep later."""
        gen = self.publish(self.a, b"one")
        self.a.add_artifact(KEY, "preview.png", self.file("p", b"png"), generation=gen)
        self.publish(self.a, b"two")
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, grace_s=0)
        self.assertEqual(2, len(self.b.history(KEY)))
        self.assertIsNotNone(self.b.fetch_generation(KEY, gen, self.tmp / "after-sweep"))

    def test_a_walk_that_faults_deletes_nothing(self):
        """A history walk that could not finish marked LESS than is live."""
        from obstore.exceptions import GenericError
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        stray = self.a.blobs.put_bytes(b"unreferenced")["digest"]
        real = ops.get
        prev = self.pointer()["replaces"][0]

        def boom(store, path, *a, **kw):
            if path.endswith(prev.split(":")[1]):
                raise GenericError("the bucket hiccuped")
            return real(store, path, *a, **kw)
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        with unittest.mock.patch.object(ops, "get", boom):
            got = self.a.sweep(now=later, grace_s=0)
        self.assertEqual(0, got["deleted_blobs"])
        self.assertEqual(1, got["unreadable_pointers"])
        self.assertTrue(self.a.blobs.has(stray))

    def test_a_deletion_is_a_tombstone_that_keeps_nothing(self):
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        self.assertTrue(self.a.delete(KEY))
        self.assertEqual("tombstone", self.a._read_ref(KEY)[0])
        self.assertIsNone(self.b.get(KEY))
        self.assertEqual([], self.b.list()[0])
        self.assertEqual([], self.b.history(KEY))
        self.assertFalse(self.b.delete(KEY), "already deleted: nothing went")
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        got = self.a.sweep(now=later, grace_s=0)
        self.assertEqual(0, got["unreadable_pointers"], "a tombstone is read, not unreadable")
        self.assertEqual(set(), self.data_blobs())

    def test_a_publication_after_a_deletion_starts_a_new_history(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        gen = self.publish(self.a, b"two")
        self.assertEqual([gen], [h["generation"] for h in self.b.history(KEY)])
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

    def test_delete_removes_garbage_outright(self):
        ops.put(self.store, f"pre/results/{KEY}.json", b"not a ref")
        self.assertTrue(self.a.delete(KEY))
        with self.assertRaises(FileNotFoundError):
            ops.head(self.store, f"pre/results/{KEY}.json")


class TestSync(_Hosts):
    """Step 4: `sync` makes one store hold what another holds, decided by ancestry. The
    other store is the same backend as ``self.store``, so the class runs memory-to-memory
    here and directory-to-directory in its OnDisk twin."""

    def setUp(self):
        super().setUp()
        self.other = (DiskStore(self.tmp / "other-store") if isinstance(self.store, DiskStore)
                      else MemoryStore())
        self.c = SharedResultCache(self.other, ResultCache(self.tmp / "local-c"), prefix="pre/")
        self.A = SharedResultCache.index(self.store, "pre/", check=False)
        self.B = SharedResultCache.index(self.other, "pre/", check=False)

    def sync(self, src=None, dst=None, **kw):
        return objectcache.sync(src or self.A, dst or self.B, **kw)

    def only(self, got):
        """The one outcome a single-key sync had."""
        (what,) = [k for k, v in got.items() if v]
        return what

    def on_b(self, key=KEY):
        hit = self.c.get(key)
        return Path(hit[0]).read_bytes() if hit else None

    def test_a_key_the_destination_lacks_is_copied_with_its_history(self):
        gens = [self.publish(self.a, f"v{i}".encode()) for i in range(3)]
        self.assertEqual("copied", self.only(self.sync()))
        self.assertEqual(b"v2", self.on_b())
        self.assertEqual(gens[::-1], [h["generation"] for h in self.c.history(KEY)])
        dest = self.tmp / "v0-on-b"
        self.assertIsNotNone(self.c.fetch_generation(KEY, gens[0], dest))
        self.assertEqual(b"v0", (dest / RESULT_NAME).read_bytes())

    def test_a_second_sync_copies_nothing(self):
        self.publish(self.a, b"one")
        self.sync()
        before = {o["path"] for o in ops.list(self.other).collect()}
        self.assertEqual("current", self.only(self.sync()))
        self.assertEqual(before, {o["path"] for o in ops.list(self.other).collect()})

    def test_a_newer_source_fast_forwards(self):
        self.publish(self.a, b"one")
        self.sync()
        self.publish(self.a, b"two")
        self.assertEqual("fast_forwarded", self.only(self.sync()))
        self.assertEqual(b"two", self.on_b())
        self.assertEqual(2, len(self.c.history(KEY)))

    def test_a_newer_destination_is_left_alone(self):
        self.publish(self.a, b"one")
        self.sync()
        self.publish(self.c, b"newer there")
        self.assertEqual("newer_there", self.only(self.sync()))
        self.assertEqual(b"newer there", self.on_b())

    def test_independent_computations_merge_and_the_later_wins(self):
        b_gen = self.publish(self.c, b"computed at B")
        time.sleep(0.01)
        a_gen = self.publish(self.a, b"computed at A, later")
        self.assertEqual("merged", self.only(self.sync()))
        self.assertEqual(b"computed at A, later", self.on_b())
        self.assertEqual([a_gen, b_gen], [h["generation"] for h in self.c.history(KEY)],
                         "the winner is current and the loser stays in history")

    def test_syncing_both_ways_converges(self):
        self.publish(self.c, b"at B")
        self.publish(self.a, b"at A")
        self.sync()                                            # merge at B
        self.assertEqual("fast_forwarded", self.only(self.sync(self.B, self.A)))
        self.assertEqual("current", self.only(self.sync()))
        self.assertEqual(self.A._read_ref(KEY)[1]["_manifest"],
                         self.B._read_ref(KEY)[1]["_manifest"])

    def test_converges_when_the_destination_won_the_merge(self):
        """The loser is a merge's SECOND parent; ancestry must follow every parent, or the
        sync back sees no relation and merges again instead of fast-forwarding."""
        self.publish(self.a, b"at A")
        time.sleep(0.01)
        self.publish(self.c, b"at B, later")
        self.assertEqual("merged", self.only(self.sync()))
        self.publish(self.c, b"at B, after the merge")   # A's manifest is now TWO levels down
        self.assertEqual("fast_forwarded", self.only(self.sync(self.B, self.A)))
        self.assertEqual(b"at B, after the merge", Path(self.b.get(KEY)[0]).read_bytes())

    def test_history_through_a_merge_keeps_both_branches(self):
        """A publication on top of a merge reaches the loser only through the merge's
        second parent."""
        b_gen = self.publish(self.c, b"at B")
        time.sleep(0.01)
        a_gen = self.publish(self.a, b"at A, later")
        self.sync()
        new = self.publish(self.c, b"after the merge")
        self.assertEqual([new, a_gen, b_gen], [h["generation"] for h in self.c.history(KEY)])

    def make_a_gap(self):
        """What the step 5 soak did: the source republishes past its history bound between
        two syncs, and its sweep takes the links, so the next sync cannot see that the
        destination's version is an ancestor - and merges (2026-09-23)."""
        self.publish(self.a, b"v0")
        self.sync()
        for i in range(1, objectcache.HISTORY_KEEP + 3):
            self.publish(self.a, f"v{i}".encode())
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, grace_s=0)
        return self.only(self.sync())

    def test_a_merge_is_not_repeated_by_every_later_sync(self):
        """A merge manifest exists only at the destination, so the source's history can
        never hold it. Taken as a new version, it made every later sync of the key merge
        AGAIN - 1, 2, 5, 9 of 16 keys in successive syncs of the soak, each one slower."""
        self.assertEqual("merged", self.make_a_gap(), "the gap itself is a merge")
        self.assertEqual("current", self.only(self.sync()),
                         "the source has not changed: the merge carries its content")
        self.publish(self.a, b"after the merge")
        self.assertEqual("fast_forwarded", self.only(self.sync()))
        self.assertEqual("current", self.only(self.sync()))
        self.assertEqual(b"after the merge", self.on_b())

    def test_a_merge_the_destination_won_still_reads_as_newer_there(self):
        self.publish(self.a, b"at A")
        time.sleep(0.01)
        self.publish(self.c, b"at B, later")
        self.assertEqual("merged", self.only(self.sync()))
        self.assertEqual("newer_there", self.only(self.sync()))

    def test_a_sync_asks_nothing_about_what_the_destination_already_references(self):
        """The soak's syncs took 6 s a key on R2 and grew: every object of every key's kept
        history was refreshed (a server-side copy each) on every run, although the
        destination's own ref already kept each one alive."""
        for i in range(3):
            gen = self.publish(self.a, f"v{i}".encode())
            self.a.add_artifact(KEY, "preview.png", self.file(f"p{i}", f"png{i}".encode()),
                                generation=gen)
        self.sync()
        self.publish(self.a, b"one more")
        touched = []
        real = BlobStore.touch

        def counting(blobs, digest):
            touched.append(digest)
            return real(blobs, digest)
        with unittest.mock.patch.object(BlobStore, "touch", counting):
            self.assertEqual("fast_forwarded", self.only(self.sync()))
        new = self.pointer()
        self.assertEqual({new["_manifest"], new["files"][RESULT_NAME]["digest"]}, set(touched),
                         "the new publication's labels and manifest, and nothing older")
        self.assertEqual(b"one more", self.on_b())
        self.assertEqual(objectcache.HISTORY_KEEP, len(self.c.history(KEY)),
                         "and the history is whole at the destination")

    def test_a_fast_forward_reads_nothing_at_the_destination_but_its_ref(self):
        """Profiled on R2 (step 5): a one-publication fast-forward cost 33 requests a key -
        13 reads walking the destination's history, 12 server-side copies (each new object
        refreshed twice), 7 writes. What the destination holds is learned from the source's
        copy of the same history, and a new object is ONE create."""
        for i in range(3):
            gen = self.publish(self.a, f"v{i}".encode())
            self.a.add_artifact(KEY, "preview.png", self.file(f"p{i}", f"png{i}".encode()),
                                generation=gen)
        self.sync()
        self.publish(self.a, b"one more")
        calls = []
        real = {name: getattr(ops, name) for name in ("get", "put", "head", "copy")}

        def watch(name):
            def f(store, *a, **kw):
                if store is self.other:
                    calls.append(name)
                return real[name](store, *a, **kw)
            return f
        with unittest.mock.patch.multiple(ops, **{n: watch(n) for n in real}):
            self.assertEqual("fast_forwarded", self.only(self.sync()))
        self.assertEqual(1, calls.count("get"), calls)
        self.assertEqual(2, calls.count("copy"), "one refresh-or-404 per new object")
        self.assertEqual(3, calls.count("put"), "the two new objects and the ref")

    def test_many_keys_sync_in_parallel_and_all_land(self):
        keys = [f"{i:02x}" * 32 for i in range(20)]
        for k in keys:
            self.publish(self.a, k.encode(), key=k)
        got = self.sync(workers=6)
        self.assertEqual(20, got["copied"])
        for k in keys:
            self.assertEqual(k.encode(), self.on_b(k))

    def test_a_deletion_travels(self):
        self.publish(self.a, b"one")
        self.sync()
        self.assertEqual(b"one", self.on_b())
        self.a.delete(KEY)
        self.assertEqual("fast_forwarded", self.only(self.sync()))
        self.c.local.delete(KEY)
        self.assertIsNone(self.c.get(KEY))
        self.assertEqual("tombstone", self.B._read_ref(KEY)[0])
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.c.sweep(now=later, grace_s=0)
        self.assertEqual([], [o for o in ops.list(self.other, "pre/blobs/").collect()
                              if not self._is_manifest(self.other, o["path"])])

    def _is_manifest(self, store, path):
        try:
            doc = json.loads(bytes(ops.get(store, path).bytes()))
        except (ValueError, UnicodeDecodeError):
            return False
        return isinstance(doc, dict) and "key" in doc

    def test_objects_are_there_before_the_ref_names_them(self):
        for i in range(2):
            self.publish(self.a, f"v{i}".encode())
        self.a.add_artifact(KEY, "preview.png", self.file("p", b"png"),
                            generation=self.pointer()["generation"])
        real, seen = ops.put, []

        def watching(store, path, *a, **kw):
            if store is self.other and "results/" in str(path):
                body = json.loads(bytes(a[0]) if a else kw["file"])
                m = json.loads(body["body"])
                seen.append(all(self.B.blobs.has(b["digest"]) for b in m["files"].values())
                            and all(self.B.blobs.has(d) for d in m["replaces"]))
            return real(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "put", watching):
            self.sync()
        self.assertEqual([True], seen)

    def test_a_ref_write_that_fails_leaves_the_destination_as_it_was(self):
        from obstore.exceptions import GenericError
        self.publish(self.a, b"one")
        real = ops.put

        def refuse(store, path, *a, **kw):
            if store is self.other and "results/" in str(path):
                raise GenericError("the destination hiccuped")
            return real(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "put", refuse):
            self.assertEqual("failed", self.only(self.sync()))
        self.assertEqual("absent", self.B._read_ref(KEY)[0])
        self.assertEqual("copied", self.only(self.sync()), "and a rerun finishes it")

    def test_only_the_keys_asked_for(self):
        self.publish(self.a, b"one")
        self.publish(self.a, b"two", key="cd" * 32)
        self.assertEqual(1, self.sync(keys=["cd" * 32])["copied"])
        self.assertEqual("absent", self.B._read_ref(KEY)[0])

    def test_a_format_1_source_is_left_for_its_first_write(self):
        self.publish(self.a, b"one")
        self.as_legacy()
        self.assertEqual("legacy", self.only(self.sync()))

    def test_a_format_1_destination_is_converted_then_decided(self):
        self.publish(self.c, b"old at B")
        p = self.c._read_ref(KEY)[1]           # rewritten as format 1, AT THE DESTINATION
        ops.put(self.other, f"pre/results/{KEY}.json", json.dumps(
            {"format": 1, "generation": p["generation"], "published": p["published"],
             "files": p["files"], "result": p["result"], "meta": p["meta"],
             "history": []}).encode())
        self.publish(self.a, b"new at A")
        self.assertEqual("merged", self.only(self.sync()))
        self.assertEqual(b"new at A", self.on_b())
        self.assertIn(p["generation"], [h["generation"] for h in self.c.history(KEY)])

    def test_a_newer_format_at_the_destination_stops_the_sync(self):
        self.publish(self.a, b"one")
        ops.put(self.other, f"pre/results/{KEY}.json",
                json.dumps({"format": objectcache.POINTER_FORMAT + 1}).encode())
        with self.assertRaises(ObjectStoreUnsuitable):
            self.sync()

    def test_an_old_tombstone_is_removed_by_the_sweep(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        later = time.time() + objectcache.TOMBSTONE_KEEP_S + 60
        got = self.a.sweep(now=later, grace_s=0)
        self.assertEqual(1, got["expired_pointers"])
        self.assertEqual("absent", self.A._read_ref(KEY)[0])
        self.assertEqual([], ops.list(self.store, "pre/blobs/").collect())

    def test_a_young_tombstone_stays(self):
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        self.a.sweep(grace_s=0)
        self.assertEqual("tombstone", self.A._read_ref(KEY)[0])

    def test_a_tombstone_replaced_since_it_was_read_is_not_removed(self):
        """No conditional delete: the sweep rereads the ref and removes only the SAME
        tombstone it judged."""
        self.publish(self.a, b"one")
        self.a.delete(KEY)
        tomb = self.A._read_ref(KEY)[1]
        self.publish(self.a, b"republished")
        self.assertFalse(self.A._expire_tombstone(tomb))
        self.assertEqual(b"republished", Path(self.b.get(KEY)[0]).read_bytes())

    def test_the_cli_syncs_one_store_into_another(self):
        from haversack import cli
        self.publish(self.a, b"one")
        stores = {"memory://a": (self.store, "pre/"), "memory://b": (self.other, "pre/")}
        with unittest.mock.patch.object(objectcache, "open_store", lambda u: stores[u]):
            self.assertEqual(0, cli.main(["cache", "sync", "memory://a", "memory://b",
                                          "--quiet"]))
        self.assertEqual(b"one", self.on_b())


class _TaskNames:
    """What a reader's segmenter is asked: names and aliases, no weights."""
    TASK = "ts.v2:total_fast"

    def tasks(self):
        return [self.TASK]

    def resolve_task(self, t):
        if t in (self.TASK, "total_fast"):
            return self.TASK
        raise LookupError(f"unknown task {t!r}")


class TestReadOnlyServer(_Hosts):
    """Step 4b: every read route of the server over a result store, and not one write to
    it; keys from the versions writers record."""

    U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    VERSIONS = ["297=v2.0.0"]

    def key(self, versions=None):
        from haversack.serve import result_key
        return result_key((f"idc:{self.U}",), _TaskNames.TASK, {}, versions or self.VERSIONS)

    def publish_labels(self, *, note=True):
        from test_serve import volume_bytes
        src = self.file("labels.seg.nrrd", volume_bytes())
        self.a.put(self.key(), src, {"volumes_ml": {"spleen": 1.0}},
                   {"task": _TaskNames.TASK, "identity": [f"idc:{self.U}"], "options": {},
                    "computed": 1.0})
        if note:
            self.a.note_task(_TaskNames.TASK, self.VERSIONS, ["v2.0.0"])
        return src.read_bytes()

    def reader(self, **kw):
        from fastapi.testclient import TestClient
        app = objectcache.read_only_app(self.store, local_dir=self.tmp / "reader",
                                        prefix="pre/", segmenter=_TaskNames(), **kw)
        return TestClient(app)

    def labels_url(self, task="total_fast"):
        return f"/v1/idc/{self.U}/{task}/labels.seg.nrrd"

    def test_a_result_a_writer_published_is_served(self):
        want = self.publish_labels()
        client = self.reader()
        self.assertEqual("public-cache", client.get("/v1/health").json()["mode"])
        got = client.get(self.labels_url())
        self.assertEqual(200, got.status_code, got.text)
        self.assertEqual(want, got.content)
        self.assertEqual(200, client.get(self.labels_url(_TaskNames.TASK)).status_code)

    def test_a_task_no_writer_recorded_is_a_miss(self):
        self.publish_labels(note=False)
        self.assertEqual(404, self.reader().get(self.labels_url()).status_code)

    def test_versions_from_another_cache_epoch_are_a_miss(self):
        self.publish_labels()
        path = self.a._task_path(_TaskNames.TASK)
        doc = json.loads(bytes(ops.get(self.store, path).bytes()))
        ops.put(self.store, path, json.dumps({**doc, "epoch": "another"}).encode())
        self.assertEqual(404, self.reader().get(self.labels_url()).status_code)

    def test_new_versions_are_picked_up_within_the_ttl(self):
        self.publish_labels(note=False)
        with unittest.mock.patch.object(objectcache, "TASK_DOC_TTL_S", 0.0):
            client = self.reader()
            self.assertEqual(404, client.get(self.labels_url()).status_code)
            self.a.note_task(_TaskNames.TASK, self.VERSIONS)
            self.assertEqual(200, client.get(self.labels_url()).status_code)

    def test_it_cannot_compute_and_never_writes_the_store(self):
        self.publish_labels()
        writes = []
        real = {name: getattr(ops, name) for name in ("put", "delete", "copy")}

        def watch(name):
            def f(store, *a, **kw):
                if store is self.store:
                    writes.append((name, a[0] if a else None))
                return real[name](store, *a, **kw)
            return f
        with unittest.mock.patch.multiple(ops, **{n: watch(n) for n in real}):
            client = self.reader()
            # no route writes today; the view refusing is what keeps it so if one ever did
            self.assertTrue(client.app.state.result_store.read_only)
            self.assertEqual(200, client.get(self.labels_url()).status_code)
            client.get("/v1/segmentations")
            client.get(f"/v1/idc/{self.U}/total_fast/meta.json")
            self.assertIn(client.post("/v1/jobs").status_code, (404, 405))
            self.assertIn(client.delete(self.labels_url()).status_code, (404, 405))
        self.assertEqual([], writes, "a read-only server wrote to the store")

    def test_the_listing_can_be_kept_private(self):
        self.publish_labels()
        rows = self.reader().get("/v1/segmentations").json()["segmentations"]
        self.assertEqual([self.key()], [r["key"] for r in rows])
        self.assertIn(self.reader(listing=False).get("/v1/segmentations").status_code,
                      (403, 404, 405, 501))

    def test_a_read_only_view_refuses_every_write_before_writing_anything(self):
        """Refused BEFORE the first byte: `put` uploads blobs before its conditional write,
        so a refusal only at the write would still have written - and constructing one,
        which runs the write probe on an ordinary view, must not probe either."""
        from haversack.errors import InputError
        self.publish_labels()
        self.a.local.put(KEY, self.file("l", b"local"), {}, {"computed": 1.0})
        writes = []
        real = {name: getattr(ops, name) for name in ("put", "delete", "copy")}

        def watch(name):
            def f(store, *a, **kw):
                writes.append((name, a[0] if a else None))
                return real[name](store, *a, **kw)
            return f
        with unittest.mock.patch.multiple(ops, **{n: watch(n) for n in real}):
            ro = SharedResultCache(self.store, ResultCache(self.tmp / "ro"), prefix="pre/",
                                   read_only=True)
            ro_push = SharedResultCache(self.store, self.a.local, prefix="pre/",
                                        read_only=True)
            for what, call in (
                    ("put", lambda: ro.put("ab" * 32, self.file("x", b"x"), {}, {})),
                    ("delete", lambda: ro.delete(self.key())),
                    ("sweep", lambda: ro.sweep(grace_s=0)),
                    ("push", lambda: ro_push.push()),
                    ("note_task", lambda: ro.note_task("t", ["v"])),
                    ("sync into it", lambda: objectcache.sync(self.a, ro))):
                with self.subTest(what=what), self.assertRaises(InputError):
                    call()
            self.assertFalse(ro.add_artifact(self.key(), "preview.png",
                                             self.file("p", b"png")))
            self.assertIsNotNone(ro.get(self.key()), "and reads")
        self.assertEqual([], writes)

    def test_a_task_name_is_safe_as_a_file_name(self):
        """':' is not allowed in a file name on exFAT or FAT32, both supported roots."""
        self.assertNotIn(":", self.a._task_path(_TaskNames.TASK).split("/")[-1])

    def test_the_versions_are_written_only_when_they_change(self):
        writes = []
        real = ops.put

        def counting(store, path, *a, **kw):
            if "tasks/" in str(path):
                writes.append(path)
            return real(store, path, *a, **kw)
        with unittest.mock.patch.object(ops, "put", counting):
            for _ in range(3):
                self.a.note_task(_TaskNames.TASK, self.VERSIONS, ["v2.0.0"])
            self.a.note_task(_TaskNames.TASK, ["297=v2.1.0"], ["v2.1.0"])
        self.assertEqual(2, len(writes))

    def test_the_cli_needs_a_store(self):
        from haversack import cli
        self.assertEqual(2, cli.main(["serve-store"]))


def test_a_writer_server_records_its_task_versions(tmp_path):
    """A segmentation published through a store records the versions its key was made
    from - the very list `versions_for` gives - so a reader derives the same key."""
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app, versions_for
    from test_job_result_cache import _Segmenter
    from test_serve import submit, wait_state

    store = MemoryStore()
    ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                       result_store=store)
    try:
        s = wait_state(TestClient(create_app(ex)), submit(TestClient(create_app(ex))),
                       ("done",))
        doc = SharedResultCache.index(store, check=False).task_versions(s["task"])
        assert doc is not None
        assert doc["weights"] == versions_for(ex.segmenter, s["task"])
        from haversack.serve import CACHE_EPOCH
        assert doc["epoch"] == CACHE_EPOCH
    finally:
        ex.close()


class TestFormat1IsReadAndConverted(_Hosts):
    """The time-limited shim of decision 1: every entry written before 2026-09-23 is a
    format 1 pointer with its history inline. It is read as it is, and the first write to
    its key turns it into a manifest chain - tokens kept."""

    def legacy_with_history(self):
        gens = [self.publish(self.a, f"v{i}".encode()) for i in range(3)]
        history = [{"generation": h["generation"], "published": h["published"],
                    "files": h["files"], "result": h["result"]}
                   for h in self.a._chain(self.pointer())[1:]]
        self.as_legacy(history=history)
        return gens

    def test_a_format_1_entry_is_read_whole(self):
        gens = self.legacy_with_history()
        self.b.local.delete(KEY)
        self.assertEqual(b"v2", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(gens[::-1], [h["generation"] for h in self.b.history(KEY)])
        self.assertEqual([KEY], [r["key"] for r in self.b.list()[0]])
        digest = f"sha256:{hashlib.sha256(b'v0').hexdigest()}"
        self.assertEqual(gens[0], self.b.find_generation(KEY, digest))

    def test_the_first_write_converts_it_and_keeps_every_token(self):
        gens = self.legacy_with_history()
        new = self.publish(self.a, b"v3")
        self.assertEqual(objectcache.POINTER_FORMAT, self.ref()["format"])
        self.assertEqual([new, *gens[::-1]], [h["generation"] for h in self.b.history(KEY)])
        dest = self.tmp / "v0"
        self.assertIsNotNone(self.b.fetch_generation(KEY, gens[0], dest))
        self.assertEqual(b"v0", (dest / RESULT_NAME).read_bytes())

    def test_an_artifact_on_a_format_1_entry_keeps_the_local_copy_current(self):
        gen = self.legacy_with_history()[-1]
        self.b.get(KEY)
        self.assertTrue(self.a.add_artifact(KEY, "preview.png", self.file("p", b"png"),
                                            generation=gen))
        self.assertEqual(gen, self.pointer()["generation"])
        self.assertEqual(gen, self.b.local.generation(KEY))

    def test_a_sweep_keeps_a_format_1_entrys_history(self):
        gens = self.legacy_with_history()
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, grace_s=0)
        self.assertIsNotNone(self.b.fetch_generation(KEY, gens[0], self.tmp / "kept"))

    def test_deleting_a_format_1_entry_leaves_a_tombstone(self):
        self.legacy_with_history()
        self.assertTrue(self.a.delete(KEY))
        self.assertEqual("tombstone", self.a._read_ref(KEY)[0])
        later = time.time() + objectcache.BLOB_GRACE_S + 60
        self.a.sweep(now=later, grace_s=0)
        self.assertEqual(set(), self.data_blobs())


class TestLateDeliverablesIntoAPublishedGeneration(_Hosts):
    """Per-request deliverables (main, 2026-09-21): a cache HIT renders what is missing,
    into the SAME generation. The publication model allows it because artifacts are exactly
    what may arrive after the labels - the pointer names them, and adding one is a
    conditional write that cannot land beside another publication's labels."""

    def test_a_filled_copy_reports_what_the_publisher_declined(self):
        from haversack.jobpolicy import missing_deliverables
        gen = self.publish(self.a, b"one")               # published with no preview
        labels, _ = self.b.get(KEY)
        self.assertEqual(("preview",),
                         missing_deliverables(("preview",), Path(labels).parent))
        self.assertTrue(self.b.add_artifact(KEY, "preview.png",
                                            self.file("p.png", b"png"), generation=gen))
        labels, _ = self.b.get(KEY)
        self.assertEqual((), missing_deliverables(("preview",), Path(labels).parent),
                         "rendered into the generation that was already published")

    def test_it_reaches_every_other_host_without_a_republication(self):
        gen = self.publish(self.a, b"one")
        before = self.a.generation(KEY)
        self.b.get(KEY)
        self.b.add_artifact(KEY, "preview.png", self.file("p.png", b"png"), generation=gen)
        cold = self.host("cold-deliverable")
        labels, _ = cold.get(KEY)
        self.assertEqual(b"png", (Path(labels).parent / "preview.png").read_bytes())
        self.assertEqual(before, cold.generation(KEY),
                         "the same generation: an artifact is not a publication")

    def test_a_render_for_a_generation_that_moved_on_is_refused(self):
        gen = self.publish(self.a, b"one")
        self.publish(self.a, b"two")                     # the key moves on mid-render
        self.assertFalse(self.b.add_artifact(KEY, "preview.png",
                                             self.file("p.png", b"png"), generation=gen))
        rows, _ = self.b.list()
        self.assertEqual(1, len(rows))


class TestAnEncodeJobsField(_Hosts):
    """An encode job publishes an embedding field where a segmentation publishes labels
    (main, 2026-09-23): ``put(output_name=)``, one primary output per generation. The
    PROTOCOL half answers for a field as for labels; the local-tier fill does not yet, so a
    field is a miss on `get` until the one-layer decision is made (2026-09-23)."""

    FIELD_META = {"task": "ts.v2:total_fast", "identity": ["upload:x"], "options": {},
                  "computed": 1.0, "kind": "encode"}

    def publish_field(self, cache, data=b"a field", key=KEY):
        from haversack.serve import FIELD_NAME
        return cache.put(key, self.file(f"field-{data.hex()}", data), {"outputs": []},
                         dict(self.FIELD_META), output_name=FIELD_NAME)

    def test_the_pointer_names_the_field_and_no_labels(self):
        from haversack.serve import FIELD_NAME
        self.publish_field(self.a)
        files = self.pointer()["files"]
        self.assertIn(FIELD_NAME, files)
        self.assertNotIn(RESULT_NAME, files)
        self.assertEqual(len(b"a field"), files[FIELD_NAME]["size"])

    def test_the_local_copy_holds_the_field_under_its_own_name(self):
        from haversack.serve import FIELD_NAME
        gen = self.publish_field(self.a)
        where = self.a.local._generation_dir(KEY, gen)
        self.assertEqual(b"a field", (where / FIELD_NAME).read_bytes())
        self.assertFalse((where / RESULT_NAME).exists())

    def test_a_name_that_is_not_a_primary_output_uploads_nothing(self):
        with self.assertRaises(ValueError):
            self.a.put(KEY, self.file("x", b"bytes"), {}, {}, output_name="preview.png")
        self.assertEqual([], list(ops.list(self.store, "pre/").collect()))

    def test_find_and_fetch_a_kept_generation_by_the_fields_digest(self):
        from haversack.serve import FIELD_NAME
        gen = self.publish_field(self.a, b"the pinned field")
        self.publish_field(self.b, b"a later field")
        digest = f"sha256:{hashlib.sha256(b'the pinned field').hexdigest()}"
        self.assertEqual(gen, self.b.find_generation(KEY, digest))
        dest = self.tmp / "pinned"
        got = self.b.fetch_generation(KEY, gen, dest)
        self.assertIsNotNone(got)
        self.assertEqual([FIELD_NAME], got["written"])
        self.assertEqual(b"the pinned field", (dest / FIELD_NAME).read_bytes())

    def test_the_listing_row_is_the_local_caches_row(self):
        """Said as a field and never linked as labels - the listing route refuses a row by
        its ``kind``, so a store row without it would come back as a segmentation."""
        self.publish_field(self.a)
        local_rows, _ = self.a.local.list()
        store_rows, _ = self.b.list()
        self.assertEqual(1, len(store_rows))
        self.assertEqual("encode", store_rows[0]["kind"])
        self.assertNotIn("links", store_rows[0])
        drop = {"published"}
        self.assertEqual({k: v for k, v in local_rows[0].items() if k not in drop},
                         {k: v for k, v in store_rows[0].items() if k not in drop})

    def test_match_sees_the_kind_as_the_local_cache_does(self):
        self.publish_field(self.a)
        seen = []
        self.b.list(match=lambda f: seen.append(f) or True)
        self.assertEqual("encode", seen[0].get("kind"))

    def test_a_pointer_naming_two_primary_outputs_is_served_as_neither(self):
        from haversack.serve import FIELD_NAME
        self.publish(self.a, b"labels")
        ptr = self.pointer()
        ptr["files"][FIELD_NAME] = dict(ptr["files"][RESULT_NAME])
        ops.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        rows, _ = self.b.list()
        self.assertEqual([], rows)
        digest = ptr["files"][RESULT_NAME]["digest"]
        self.assertIsNone(self.b.find_generation(KEY, digest))

    def test_another_host_is_served_the_field(self):
        """It was published and then read back as a miss - the fill knew labels only - so
        with a store on, every repeat of an encode job re-ran the encoder (2026-09-23)."""
        from haversack.serve import FIELD_NAME
        self.publish_field(self.a)
        hit = self.b.get(KEY)
        self.assertIsNotNone(hit)
        self.assertEqual(FIELD_NAME, Path(hit[0]).name)
        self.assertEqual(b"a field", Path(hit[0]).read_bytes())
        self.assertFalse((Path(hit[0]).parent / RESULT_NAME).exists())

    def test_a_republished_field_replaces_the_copy_on_another_host(self):
        self.publish_field(self.a, b"first")
        self.b.get(KEY)
        self.publish_field(self.a, b"second")
        self.assertEqual(b"second", Path(self.b.get(KEY)[0]).read_bytes())

    def test_pull_places_a_field_and_then_calls_it_current(self):
        from haversack.serve import FIELD_NAME
        self.publish_field(self.a)
        self.assertEqual(1, self.b.pull()["pulled"])
        self.assertEqual(FIELD_NAME, Path(self.b.local.get(KEY)[0]).name)
        self.assertEqual({"pulled": 0, "current": 1}, {k: v for k, v in self.b.pull().items()
                                                        if k in ("pulled", "current")})

    def test_an_orphaned_field_generation_is_adopted_without_downloading(self):
        """`ResultCache.adopt` always demanded the LABELS, so the repair a crash between
        the rename and the pointer write needs could never finish for a field."""
        gen = self.publish_field(self.a)
        self.b.get(KEY)
        (self.b.local.root / KEY / CURRENT_NAME).unlink()
        with unittest.mock.patch.object(BlobStore, "fetch",
                                        side_effect=AssertionError("downloaded")):
            self.assertEqual(b"a field", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertEqual(gen, self.b.local.generation(KEY))

    def test_a_copy_that_lost_its_field_is_not_current(self):
        """``_holds`` is the one definition of "this host has it whole": a copy asked about
        its artifacts and documents but not its field would be reported current by `pull`
        and never repaired."""
        self.publish_field(self.a)
        self.b.pull()
        field = Path(self.b.local.get(KEY)[0])
        field.unlink()
        self.assertEqual(0, self.b.pull()["current"])
        self.assertEqual(b"a field", field.read_bytes())

    def test_push_carries_a_local_field_into_the_store(self):
        from haversack.serve import FIELD_NAME
        self.a.local.put(KEY, self.file("f", b"local field"), {"outputs": []},
                         dict(self.FIELD_META), output_name=FIELD_NAME)
        self.assertEqual(1, self.a.push()["pushed"])
        self.assertIn(FIELD_NAME, self.pointer()["files"])
        self.assertEqual(b"local field", Path(self.b.get(KEY)[0]).read_bytes())


@pytest.mark.parametrize("backend", ["memory", "disk"])
def test_an_encode_job_on_one_server_is_a_hit_on_another(tmp_path, backend):
    """The whole path: a field computed on one server is served by another sharing the
    store, without its encoder running, and byte for byte."""
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_encode_jobs import FakeEncoder, post
    from test_serve import FakeSegmenter, wait_state

    store = MemoryStore() if backend == "memory" else DiskStore(tmp_path / "store")
    enc_a, enc_b = FakeEncoder(), FakeEncoder()
    ex_a = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "wa", cache_dir=tmp_path / "ca",
                         encode_fn=enc_a, result_store=store)
    ex_b = LocalExecutor(FakeSegmenter(steps=1), workdir=tmp_path / "wb", cache_dir=tmp_path / "cb",
                         encode_fn=enc_b, result_store=store)
    try:
        client_a, client_b = TestClient(create_app(ex_a)), TestClient(create_app(ex_b))
        a = wait_state(client_a, post(client_a).json()["id"], ("done",))
        b = wait_state(client_b, post(client_b).json()["id"], ("done", "failed"))
        assert b["state"] == "done" and b["key"] == a["key"], b
        assert b.get("cached") is True and enc_b.calls == []
        assert client_b.get(b["links"]["result"]).content == \
            client_a.get(a["links"]["result"]).content
        rows = client_b.get("/v1/segmentations").json()
        assert rows.get("results", rows.get("rows", [])) == [], "a field is not a segmentation"
    finally:
        ex_a.close()
        ex_b.close()


# -- every class above, again on a directory ------------------------------------------------
#
# Step 2 of the consolidation (2026-09-23): the store protocol, unchanged, on provender's
# DiskStore - a directory that honors both conditional writes - instead of a bucket. Faults
# are injected through `provender.ops`, which both backends answer through, so the fault
# tests run here too rather than being skipped.

def _on_disk(base):
    def make_store(self):
        return DiskStore(self.tmp / "disk-store")
    return type(f"{base.__name__}OnDisk", (base,), {"make_store": make_store})


for _name, _base in list(globals().items()):
    if (isinstance(_base, type) and issubclass(_base, _Hosts) and _base is not _Hosts
            and not _name.endswith("OnDisk")):
        globals()[f"{_name}OnDisk"] = _on_disk(_base)
del _name, _base


def test_the_protocol_suite_really_runs_on_disk():
    """A rerun that silently stayed in memory would prove nothing."""
    runs = [v for k, v in globals().items() if k.endswith("OnDisk")]
    assert len(runs) >= 30, len(runs)
    case = runs[0]("setUp")
    case.setUp()
    try:
        assert isinstance(case.store, DiskStore)
    finally:
        case.tearDown()
