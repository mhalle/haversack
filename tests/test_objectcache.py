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
from provender import Blobs as BlobStore  # noqa: E402  (the store moved to provender)
from haversack.objectcache import (ObjectStoreUnsuitable,  # noqa: E402
                                   SharedResultCache, check_conditional_writes, open_store)
from haversack.serve import CURRENT_NAME, RESULT_NAME, ResultCache  # noqa: E402

KEY = "ab" * 32


class _Hosts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.store = MemoryStore()
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

        def read(cache, key, **kw):
            got = real(cache, key, **kw)
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

        def read(cache, key, **kw):
            got = real(cache, key, **kw)
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

    def test_a_corrupt_blob_is_a_miss_and_is_replaced_not_deleted(self):
        """A reader cannot tell a corrupt object from a stream that ended early, and a
        blob is shared by every result with identical output - so deleting it on one
        client's reading took unrelated entries with it (review, 2026-09-19). The reader
        stops deduplicating onto it instead, and the next publication replaces it."""
        self.publish(self.a, b"one")
        digest = f"sha256:{hashlib.sha256(b'one').hexdigest()}"
        obstore.put(self.store, self._blob_path(b"one"), b"not one")
        self.assertIsNone(self.b.get(KEY))
        self.assertTrue(self.b.blobs.has(digest), "not deleted: other keys may share it")
        self.assertIn(digest, self.b.blobs.suspect)
        self.publish(self.b, b"one")           # the host that saw it wrong republishes
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
        self.assertNotIn(digest, self.b.blobs.suspect)

    def test_an_unrelated_key_survives_a_corrupt_blob(self):
        self.publish(self.a, b"same", key="cd" * 32)
        self.publish(self.a, b"same")          # one blob, two keys
        obstore.put(self.store, self._blob_path(b"same"), b"wrong")
        self.assertIsNone(self.b.get(KEY))
        self.assertTrue(self.b.blobs.has(f"sha256:{hashlib.sha256(b'same').hexdigest()}"),
                        "deleting it would take the other key down too")

    def test_an_unreadable_pointer_is_a_miss(self):
        self.publish(self.a, b"one")
        obstore.put(self.store, f"pre/results/{KEY}.json", b"{not json")
        self.assertIsNone(self.b.get(KEY))
        self.publish(self.a, b"two")               # garbage is nobody's: publish over it
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

    def test_an_entry_from_a_NEWER_haversack_is_a_miss_and_is_not_overwritten(self):
        """Refusing is the difference between a miss and taking another host's current
        result and its whole history out of the index in one write (review, 2026-09-20).
        Garbage gets published over; something that says it came from a later version does
        not."""
        self.publish(self.a, b"one")
        newer = {**self.pointer(), "format": objectcache.POINTER_FORMAT + 1}
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(newer).encode())
        self.assertIsNone(self.b.get(KEY), "cannot read it")
        with self.assertRaises(ObjectStoreUnsuitable) as cm:
            self.publish(self.a, b"two")
        self.assertIn("newer haversack", str(cm.exception))
        self.assertEqual(newer, self.pointer(), "left exactly as it was")

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
        self.assertEqual(2, self.a.sweep(now=later)["deleted_blobs"])
        for gone in (b"one", b"two"):
            self.assertFalse(self.a.blobs.has(f"sha256:{hashlib.sha256(gone).hexdigest()}"))
        self.assertEqual(objectcache.HISTORY_KEEP + 1, len(self.a.history(KEY)),
                         "what the sweep spared is exactly what history keeps")
        self.assertEqual(b"more-4", Path(self.b.get(KEY)[0]).read_bytes())

    def test_max_age_expires_pointers_and_then_their_blobs(self):
        self.publish(self.a, b"one")
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        got = self.a.sweep(now=later, max_age_s=3600)
        self.assertEqual({"expired_pointers": 1, "deleted_blobs": 1, "already_gone": 0,
                          "unreadable_pointers": 0}, got)
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
        return unittest.mock.patch.object(obstore, which, boom)

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
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())

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
                self.assertEqual([], self.b.list())
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
        obstore.put(self.store, "pre/results/.tmp-upload.json", b"{}")
        obstore.put(self.store, "pre/results/notes.json", b"not a pointer")
        self.assertEqual([KEY], [e["key"] for e in self.b.list()])
        got = self.b.sweep(grace_s=0)
        self.assertEqual(2, got["unreadable_pointers"])
        self.assertEqual(0, got["deleted_blobs"],
                         "blobs it cannot account for are not deleted")
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_newer_pointer_format_is_never_garbage_collected(self):
        """An old host's sweeper meeting a new writer's pointer: it cannot read which
        blobs are live, so it deletes none."""
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["format"] = POINTER_FORMAT_NEXT
        self.damaged(ptr)
        got = self.a.sweep(grace_s=0)
        self.assertEqual(0, got["deleted_blobs"])
        self.assertEqual(1, got["unreadable_pointers"])
        self.assertTrue(self.a.blobs.has(ptr["files"][RESULT_NAME]["digest"]))


class TestArtifactBlobs(_Hosts):
    def test_a_missing_artifact_blob_does_not_lose_the_labels(self):
        """A lost thumbnail is not worth a GPU recompute."""
        gen = self.publish(self.a, b"one")
        self.a.add_artifact(KEY, "preview.png", self.file("p.png", b"png"), generation=gen)
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'png').hexdigest()}")
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
            obstore.delete(self.store,
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
        self.assertEqual(["cc" * 32, "bb" * 32, "aa" * 32], [e["key"] for e in self.b.list()])
        self.assertEqual(2, len(self.b.list(limit=2)))

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
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
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
        real = SharedResultCache._read_pointer

        def counting_read(cache, key, **kw):
            reads.append(key)
            return real(cache, key, **kw)
        real_put = obstore.put

        def refuse_pointer_writes(store, path, *a, **kw):
            if "results/" in str(path):        # the blobs still upload: only the pointer
                raise PreconditionError("etag moved")
            return real_put(store, path, *a, **kw)
        with unittest.mock.patch.object(obstore, "put", refuse_pointer_writes), \
                unittest.mock.patch.object(SharedResultCache, "_read_pointer",
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
        real = obstore.get

        def boom(st, path, *a, **kw):
            if "results/" in str(path) or "blobs/" in str(path):
                raise PermissionDeniedError("403 from the bucket")
            return real(st, path, *a, **kw)
        with unittest.mock.patch.object(obstore, "get", boom):
            # this host holds the result it just computed, so the outage is not a miss:
            # it serves its own copy rather than throwing away warm work
            assert client.get(f"/v1/jobs/{jid}/result").status_code == 200
            assert client.get("/v1/segmentations").status_code == 200
            assert ex.cache_get(s["key"]) is not None
        ex.cache.local.delete(s["key"])        # now nothing here holds it
        with unittest.mock.patch.object(obstore, "get", boom):
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
        inner = {n for f in ast.walk(node)
                 if isinstance(f, ast.FunctionDef)
                 for n in ast.walk(f)}          # a sync helper defined inside is not ours
        for call in ast.walk(node):
            if call in inner or not isinstance(call, ast.Call):
                continue
            fn = call.func
            if (isinstance(fn, ast.Attribute) and fn.attr in blocking
                    and isinstance(fn.value, ast.Name) and fn.value.id == "executor"):
                problems.append(f"{node.name} (line {call.lineno}) calls executor."
                                f"{fn.attr} directly; use `await _offload(...)`")
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
        self.publish(self.a, b"one")
        stale = self.pointer()
        stale["published"] = time.time() - objectcache.HISTORY_MAX_AGE_S - 60
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(stale).encode())
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
        # the BYTES too, and without waiting for a sweep that nothing may ever run: a
        # delete that leaves them readable in the bucket does not mean "gone"
        for labels in (b"one", b"two"):
            self.assertFalse(self.b.blobs.has(
                f"sha256:{hashlib.sha256(labels).hexdigest()}"), labels)
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        self.assertEqual(0, self.a.sweep(now=later)["deleted_blobs"],
                         "nothing left for the sweep to find")

    def test_identical_bytes_make_history_nearly_free(self):
        self.publish(self.a, b"same")
        self.publish(self.a, b"same")          # a recompute that changed nothing
        blobs = [o for batch in obstore.list(self.store, "pre/blobs/") for o in batch]
        self.assertEqual(1, len(blobs))
        self.assertEqual(2, len(self.a.history(KEY)))

    def test_a_damaged_past_does_not_make_the_present_unreadable(self):
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        ptr = self.pointer()
        ptr["history"] = [{"generation": 7}, "rubbish", {"generation": "g", "files": []}]
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
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
        self.assertEqual(3, len(self.b.list()))

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
        self.assertEqual([], self.b.list())

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
        self.assertEqual([KEY], [e["key"] for e in self.b.list()])

    def test_limit_takes_the_newest(self):
        for i, key in enumerate(("aa" * 32, "bb" * 32, "cc" * 32)):
            self.local_entry(self.a, key, f"v{i}".encode())
            os.utime(self.a.local.root / key, (1000 + i, 1000 + i))
        self.assertEqual(1, self.a.push(limit=1)["pushed"])
        self.assertEqual(["cc" * 32], [e["key"] for e in self.b.list()])


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
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        got = self.b.pull()
        self.assertEqual({"pulled": 1, "current": 0, "failed": 1, "unreadable": 0,
                          "evicted": 0}, got)
        self.assertEqual(b"two", Path(self.b.get("bb" * 32)[0]).read_bytes())

    def test_pull_reports_pointers_it_cannot_read(self):
        self.publish(self.a, b"one")
        obstore.put(self.store, "pre/results/.junk.json", b"{}")
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
            obstore.get(self.store,
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
        self.assertEqual(6, len(self.b.list()))

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
        self.assertEqual([], self.b.list())

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
                ptr = self.pointer()
                ptr["history"] = junk
                obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
                self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())
                self.assertEqual(0, self.b.sweep(grace_s=0)["unreadable_pointers"])
                self.assertEqual(0, self.b.sweep(grace_s=0)["deleted_blobs"])

    def test_a_junk_history_entry_never_occupies_a_slot(self):
        """The writer used a looser rule than the readers, so an entry nothing could read
        rode forward for ever and pushed real predecessors off the end."""
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["history"] = [{"generation": "junk", "files": "not a map", "result": {}}] * 3
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        self.publish(self.a, b"two")
        raw = self.pointer()
        self.assertEqual([h["generation"] for h in raw["history"]],
                         [ptr["generation"]], "only the real predecessor survives")

    def test_an_undatable_generation_is_not_kept_for_ever(self):
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["published"] = "2026-01-01"        # not a time this code can compare
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        self.publish(self.a, b"two")
        self.assertEqual([True], [h["current"] for h in self.a.history(KEY)])

    def test_a_pointer_nothing_can_date_is_never_expired(self):
        """The other direction: cleanup refuses what it cannot account for."""
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["published"] = "2026-01-01"
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        got = self.a.sweep(now=time.time() + 10 ** 9, max_age_s=1)
        self.assertEqual(0, got["expired_pointers"])
        self.assertEqual(b"one", Path(self.b.get(KEY)[0]).read_bytes())

    def test_a_string_published_does_not_stop_every_sweep(self):
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["published"] = "2026-09-01T00:00:00"
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
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
        self.assertEqual(1, self.a.sweep(now=past_margin, grace_s=0)["deleted_blobs"])
        self.assertEqual(b"two", Path(self.b.get(KEY)[0]).read_bytes())

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

    def test_history_entries_do_not_carry_meta(self):
        """No reader ever looked at a history entry's meta, and every copy was paid for on
        every pointer read."""
        self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        self.assertNotIn("meta", self.pointer()["history"][0])


class TestDeleteMeansGone(_Hosts):
    def test_the_bytes_go_without_waiting_for_a_sweep(self):
        self.publish(self.a, b"patient one")
        self.publish(self.a, b"patient two")
        self.a.delete(KEY)
        for labels in (b"patient one", b"patient two"):
            self.assertFalse(self.a.blobs.has(f"sha256:{hashlib.sha256(labels).hexdigest()}"))

    def test_bytes_another_entry_shares_are_kept(self):
        self.publish(self.a, b"shared", key="cd" * 32)
        self.publish(self.a, b"shared")
        self.a.delete(KEY)
        self.assertEqual(b"shared", Path(self.b.get("cd" * 32)[0]).read_bytes())

    def test_bytes_another_entry_holds_only_in_HISTORY_are_kept(self):
        """A blob that is another key's superseded generation is still that key's to
        lose - and a keep set built from current generations alone took it."""
        self.publish(self.a, b"shared", key="cd" * 32)
        self.publish(self.a, b"newer", key="cd" * 32)     # "shared" is now history there
        self.publish(self.a, b"shared")
        self.a.delete(KEY)
        digest = f"sha256:{hashlib.sha256(b'shared').hexdigest()}"
        self.assertTrue(self.a.blobs.has(digest))
        old_gen = self.a.history("cd" * 32)[1]["generation"]
        dest = self.tmp / "hist-kept"
        self.assertIsNotNone(self.b.fetch_generation("cd" * 32, old_gen, dest))

    def test_an_aged_out_generations_bytes_go_too(self):
        """The age bound decides what history LISTS; it must not decide what a deletion
        leaves behind."""
        self.publish(self.a, b"ancient")
        ptr = self.pointer()
        self.publish(self.a, b"current")
        aged = self.pointer()
        aged["history"][0]["published"] = time.time() - objectcache.HISTORY_MAX_AGE_S - 60
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(aged).encode())
        self.assertEqual(1, len(self.a.history(KEY)), "no longer listed")
        self.a.delete(KEY)
        self.assertFalse(self.a.blobs.has(ptr["files"][RESULT_NAME]["digest"]),
                         "but still deleted")

    def test_a_huge_store_refuses_the_purge_and_says_what_to_run(self):
        import contextlib
        import io
        self.publish(self.a, b"patient")
        self.publish(self.a, b"another", key="cd" * 32)    # something left to scan
        err = io.StringIO()
        with unittest.mock.patch.object(objectcache, "PURGE_SCAN_LIMIT", 0), \
                contextlib.redirect_stderr(err):
            self.assertTrue(self.a.delete(KEY))
        self.assertIn("cache sweep", err.getvalue())
        self.assertTrue(self.a.blobs.has(
            f"sha256:{hashlib.sha256(b'patient').hexdigest()}"))

    def test_delete_reports_whether_the_bytes_went(self):
        """The return value says an entry was removed; only the report says the bytes
        were - and a caller that means "gone" has to look."""
        self.publish(self.a, b"patient")
        seen = []
        self.a.delete(KEY, report=seen.append)
        self.assertEqual(True, seen[0]["purged"])
        self.publish(self.a, b"patient two")
        self.publish(self.a, b"another", key="cd" * 32)
        seen.clear()
        with unittest.mock.patch.object(objectcache, "PURGE_SCAN_LIMIT", 0):
            self.a.delete(KEY, report=seen.append)
        self.assertEqual(False, seen[0]["purged"])
        self.assertTrue(seen[0]["existed"])

    def test_a_pointer_that_cannot_be_read_stops_the_purge_loudly(self):
        """"I could not tell" must not look like "done" for a deletion."""
        self.publish(self.a, b"patient")
        obstore.put(self.store, "pre/results/.stray.json", b"{}")
        self.assertTrue(self.a.delete(KEY))
        self.assertTrue(self.a.blobs.has(f"sha256:{hashlib.sha256(b'patient').hexdigest()}"),
                        "kept, because nothing could establish they were unreferenced")

    def test_deleting_an_entry_this_version_cannot_PARSE_still_says_it_deleted_it(self):
        """Garbage under a key - a truncated write, a stray object - is nobody's data, so
        an operator can still remove the entry. Something a NEWER haversack wrote is
        different, and is refused: see TestVersionSkewOnOneBucket."""
        self.publish(self.a, b"one")
        obstore.put(self.store, f"pre/results/{KEY}.json", b"{not a pointer")
        cold = SharedResultCache(self.store, ResultCache(self.tmp / "cold"), prefix="pre/",
                                 check=False)
        self.assertTrue(cold.delete(KEY), "the operator removed something, and is told so")
        self.assertEqual([], [o for b in obstore.list(self.store, f"pre/results/{KEY}")
                              for o in b])


class TestFetchGenerationReportsWhatItWrote(_Hosts):
    def test_a_generation_whose_files_cannot_be_placed_is_not_success(self):
        gen = self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["files"] = {"labels.other.nrrd": ptr["files"][RESULT_NAME]}
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
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
        gen = self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["files"] = {**ptr["files"], "../escaped-near": ptr["files"][RESULT_NAME]}
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
        dest = self.tmp / "dest-allow"
        got = self.b.fetch_generation(KEY, gen, dest)
        self.assertFalse((dest.parent / "escaped-near").exists(), "wrote outside dest")
        self.assertEqual([RESULT_NAME], got["written"], "and does not claim it")

    def test_fetch_generation_of_a_swept_generation_is_none(self):
        gen = self.publish(self.a, b"one")
        self.publish(self.a, b"two")
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
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
        self.a.delete("cd" * 32, purge=False)          # the blob is now an orphan
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
        for batch in obstore.list(self.store, "pre/results/"):
            for obj in batch:
                body = bytes(obstore.get(self.store, obj["path"]).bytes())
                obstore.put(self.store, obj["path"].replace("results/", "entries/"), body)
                obstore.delete(self.store, obj["path"])
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
        obstore.put(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}",
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


class TestPurgeWindow(_Hosts):
    """`delete` purges the entry's bytes by scanning the other pointers and then deleting
    what none of them names. Those are two steps, and this pins which way the gap falls."""

    def test_an_entry_published_before_the_scan_keeps_its_shared_bytes(self):
        self.publish(self.a, b"shared", key="cd" * 32)
        self.publish(self.a, b"shared")
        self.a.delete(KEY)
        self.assertEqual(b"shared", Path(self.b.get("cd" * 32)[0]).read_bytes())

    def test_one_published_inside_the_window_keeps_them(self):
        """The window is closed, and by the same pair of mechanisms the sweep uses: the
        candidates are listed before the entries are read, and a deduplicated write
        refreshes the blob, which the re-check before each delete sees. It cost three
        attempts to get here (reviews of 2026-09-20)."""
        self.publish(self.a, b"shared")
        real = SharedResultCache._scan_pointers
        raced = []

        def scan(cache, **kw):
            got = real(cache, **kw)
            if not raced:                      # lands after the scan, before the deletes
                raced.append(True)
                self.publish(self.b, b"shared", key="cd" * 32)
            return got
        seen = []
        with unittest.mock.patch.object(SharedResultCache, "_scan_pointers", scan):
            self.a.delete(KEY, report=seen.append)
        self.assertTrue(raced)
        # asked of a host that has to FETCH them: the publisher would serve from its own
        # disk and hide the loss
        cold = self.host("cold")
        self.assertEqual(b"shared", Path(cold.get("cd" * 32)[0]).read_bytes())
        self.assertFalse(seen[0]["purged"],
                         "and the deletion says it could not finish, rather than claiming "
                         "bytes are gone that are not")


class TestGapsFromTheThirdMutationRun(_Hosts):
    def test_a_dated_history_entry_with_damaged_files_is_still_refused(self):
        """Every junk entry in the earlier tests was also undated, so the date rule alone
        dropped it and the files rule was never exercised."""
        self.publish(self.a, b"one")
        ptr = self.pointer()
        ptr["history"] = [{"generation": "junk", "files": "not a map",
                           "published": time.time()},
                          {"generation": 7, "files": {}, "published": time.time()}]
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(ptr).encode())
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
        for batch in obstore.list(self.store, "pre/results/"):
            for obj in batch:
                obstore.delete(self.store, obj["path"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.a.sweep(grace_s=0)
        self.assertIn("no readable entries under", err.getvalue())

    def test_pull_names_the_key_that_failed(self):
        self.publish(self.a, b"one")
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
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


class TestPurgeReChecksToo(_Hosts):
    def test_a_publication_that_dedupes_during_a_purge_keeps_its_bytes(self):
        """`delete` decides "unreferenced" by scanning the other entries; a publication
        that lands after that scan and deduplicates onto these bytes refreshes them, and
        the re-check before each delete is what tells the two apart."""
        self.publish(self.a, b"shared")
        real = SharedResultCache._scan_pointers
        raced = []

        def scan(cache, **kw):
            got = real(cache, **kw)
            if not raced:
                raced.append(True)
                self.publish(self.b, b"shared", key="cd" * 32)
            return got
        with unittest.mock.patch.object(SharedResultCache, "_scan_pointers", scan):
            self.a.delete(KEY)
        self.assertTrue(raced)
        self.assertEqual(b"shared", Path(self.host("cold3").get("cd" * 32)[0]).read_bytes())

    def test_and_says_it_could_not_finish(self):
        self.publish(self.a, b"shared")
        real = SharedResultCache._scan_pointers
        raced, seen = [], []

        def scan(cache, **kw):
            got = real(cache, **kw)
            if not raced:
                raced.append(True)
                self.publish(self.b, b"shared", key="cd" * 32)
            return got
        with unittest.mock.patch.object(SharedResultCache, "_scan_pointers", scan):
            self.a.delete(KEY, report=seen.append)
        self.assertFalse(seen[0]["purged"], "a deletion that left bytes says so")


# -- the fourth review round (one reviewer, whole-branch, 2026-09-20) --------------------


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
        self.a._swap(KEY, lambda cur: {**cur, "generation": gen_a,
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
        real = obstore.get

        def boom(store, path, *a, **kw):
            if "results/" in str(path):
                raise GenericError("the bucket is unreachable")
            return real(store, path, *a, **kw)
        return unittest.mock.patch.object(obstore, "get", boom)

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
        with unittest.mock.patch.object(obstore, "put",
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
        real = obstore.get

        def boom(store, path, *a, **kw):
            if "results/" in str(path):
                raise GenericError("unreachable")
            return real(store, path, *a, **kw)
        return unittest.mock.patch.object(obstore, "get", boom)

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
        obstore.put(self.store, f"pre/results/{KEY}.json", json.dumps(newer).encode())
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
