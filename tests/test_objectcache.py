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
        self.assertEqual({"expired_pointers": 1, "deleted_blobs": 1,
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
        real = SharedResultCache._read_pointer

        def read(cache, key, **kw):
            got = real(cache, key, **kw)
            self.publish(self.b, b"racing")    # always lose the race
            return got
        with unittest.mock.patch.object(SharedResultCache, "_read_pointer", read):
            with self.assertRaises(RuntimeError):
                self.publish(self.a, b"one")

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
            assert client.get(f"/v1/jobs/{jid}/result").status_code == 410
            assert client.get("/v1/segmentations").status_code == 200
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
    blocking = {"cache_get", "status_of"}
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
        later = objectcache.time.time() + objectcache.BLOB_GRACE_S + 60
        self.assertEqual(2, self.a.sweep(now=later)["deleted_blobs"],
                         "and its bytes are collectable, not pinned by a kept generation")

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
        self.assertEqual({"pulled": 0, "current": 1, "failed": 0, "unreadable": 0}, got)

    def test_a_swept_blob_makes_one_entry_fail_and_not_the_rest(self):
        self.publish(self.a, b"one", key="aa" * 32)
        self.publish(self.a, b"two", key="bb" * 32)
        obstore.delete(self.store, f"pre/blobs/sha256/{hashlib.sha256(b'one').hexdigest()}")
        got = self.b.pull()
        self.assertEqual({"pulled": 1, "current": 0, "failed": 1, "unreadable": 0}, got)
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
