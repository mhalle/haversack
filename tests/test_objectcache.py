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
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import pytest

obstore = pytest.importorskip("obstore")
from obstore.store import LocalStore, MemoryStore  # noqa: E402

from haversack import objectcache  # noqa: E402
from haversack.objectcache import (BlobStore, ObjectStoreUnsuitable,  # noqa: E402
                                   SharedResultCache, check_conditional_writes, open_store)
from haversack.serve import RESULT_NAME, ResultCache  # noqa: E402

KEY = "ab" * 32


class _Hosts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = MemoryStore()
        self.a = self.host("a")
        self.b = self.host("b")

    def tearDown(self):
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
        return json.loads(bytes(obstore.get(self.store, f"pre/results/{key}.json").bytes()))


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
        real_put = obstore.put
        store = MemoryStore()

        def put(s, path, data, *, mode=None, **kw):
            return real_put(s, path, data, mode=None if isinstance(mode, dict) else mode, **kw)
        with unittest.mock.patch.object(obstore, "put", put):
            with self.assertRaises(ObjectStoreUnsuitable) as cm:
                check_conditional_writes(store)
        self.assertIn("stale etag", str(cm.exception))

    def test_a_store_that_overwrites_on_create_is_refused(self):
        real_put = obstore.put

        def put(s, path, data, *, mode=None, **kw):
            return real_put(s, path, data, mode=None if mode == "create" else mode, **kw)
        with unittest.mock.patch.object(obstore, "put", put):
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
        entries = self.b.list()
        self.assertEqual([KEY], [e["key"] for e in entries])
        self.assertEqual(3, entries[0]["bytes"])
        self.assertEqual("t", entries[0]["task"])

    def test_identical_bytes_are_one_blob(self):
        self.publish(self.a, b"same", key="cd" * 32)
        self.publish(self.b, b"same", key="ef" * 32)
        blobs = [o for batch in obstore.list(self.store, "pre/blobs/") for o in batch]
        self.assertEqual(1, len(blobs))

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
        real = SharedResultCache._read_pointer
        state = {"raced": False}

        def read(cache, key):
            got = real(cache, key)
            if cache is self.a and not state["raced"]:
                state["raced"] = True
                self.publish(self.b, b"two")
            return got
        with unittest.mock.patch.object(SharedResultCache, "_read_pointer", read):
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
        real = SharedResultCache._read_pointer
        state = {"raced": False}

        def read(cache, key):
            got = real(cache, key)
            if cache is self.a and not state["raced"]:
                state["raced"] = True
                self.publish(self.b, b"two")
            return got
        with unittest.mock.patch.object(SharedResultCache, "_read_pointer", read):
            gen = self.publish(self.a, b"one")
        self.assertEqual(gen, self.pointer()["generation"])
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())


class TestMisses(_Hosts):
    def _blob_path(self, data: bytes) -> str:
        return f"pre/blobs/sha256/{hashlib.sha256(data).hexdigest()}"

    def test_a_swept_blob_is_a_miss_and_a_republication_heals_it(self):
        self.publish(self.a, b"one")
        obstore.delete(self.store, self._blob_path(b"one"))
        self.assertIsNone(self.b.get(KEY))
        self.publish(self.a, b"one")               # same bytes: the blob must be re-uploaded
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_corrupt_blob_is_a_miss_and_is_removed(self):
        self.publish(self.a, b"one")
        obstore.put(self.store, self._blob_path(b"one"), b"not one")
        self.assertIsNone(self.b.get(KEY))
        self.assertFalse(self.b.blobs.has(f"sha256:{hashlib.sha256(b'one').hexdigest()}"),
                         "left in place, put_file would skip it and it would never heal")
        self.publish(self.a, b"one")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_an_unreadable_pointer_is_a_miss(self):
        self.publish(self.a, b"one")
        obstore.put(self.store, f"pre/results/{KEY}.json", b"{not json")
        self.assertIsNone(self.b.get(KEY))
        obstore.put(self.store, f"pre/results/{KEY}.json",
                    json.dumps({"format": 999}).encode())
        self.assertIsNone(self.b.get(KEY))
        self.publish(self.a, b"two")               # and a publication replaces it
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

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
                obstore.delete(self.store, self._blob_path(b"one"))
            return got
        with unittest.mock.patch.object(BlobStore, "put_file", put_file):
            self.publish(self.a, b"one")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())


class TestSweep(_Hosts):
    def test_only_old_unreferenced_blobs_go(self):
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")               # "one" is now unreferenced
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        young = self.a.sweep()
        self.assertEqual(0, young["deleted_blobs"], "unreferenced but inside the grace")
        old = self.a.sweep(now=later)
        self.assertEqual(1, old["deleted_blobs"])
        self.assertFalse(self.a.blobs.has(f"sha256:{hashlib.sha256(b'one').hexdigest()}"))
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

    def test_max_age_expires_pointers_and_then_their_blobs(self):
        self.publish(self.a, b"one")
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        got = self.a.sweep(now=later, max_age_s=3600)
        self.assertEqual({"expired_pointers": 1, "deleted_blobs": 1}, got)
        self.assertIsNone(self.b.get(KEY))

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
        assert [e["key"] for e in ex_b.cache_list()] == [s["key"]]
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
    from haversack.serve import LocalExecutor
    from test_job_result_cache import _Segmenter
    with pytest.raises(ObjectStoreUnsuitable):
        LocalExecutor(_Segmenter(steps=1), workdir=tmp_path / "w", cache_dir=tmp_path / "c",
                      result_store=f"file://{tmp_path / 'bucket'}")
