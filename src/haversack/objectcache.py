"""The result cache on an object store (S3, GCS, ...), shared by every host that names it.

Why this is not ``ResultCache`` pointed at a mounted bucket (2026-09-19). ``ResultCache``
is a POSIX protocol: one rename publishes, ``flock`` proves a writer dead, a lease file's
mtime says a reader holds a generation. An object store offers none of those, and a
filesystem that emulates them over one (ZeroFS, assessed the same day) funnels every host
through a single writer process whose locks vanish on restart - exactly when "I could take
its lock" stops proving death. What an object store DOES offer is enough for a different
protocol, the one build caches use (Bazel's ActionCache over a CAS):

- **Blobs** under ``blobs/sha256/<hex>``, named by their own bytes and written only if
  absent. Immutable, so a reader never holds anything and nothing needs a lease. That half
  is ``provender`` (2026-09-20), a package of its own because feldglas needed exactly it
  and two copies of one protocol is how this repo's defects have always started. What is
  haversack's here is the POINTER and the policy around it - including, for the sweep, the
  live set, which is the only part that knows what a result is.
- **One pointer per key**, ``results/<key>.json``: the files by digest plus the result and
  meta documents inline. Publication is uploading the blobs and then ONE conditional write
  of the pointer (create-if-absent, or replace-if-unchanged against the etag read), so the
  pointer IS the generation - no staging, no rename, no writer claim. A lost race rereads
  and writes again: last writer wins, as the rename did.
- **Cleanup is dumb on purpose.** ``sweep`` deletes blobs no pointer references once they
  are older than a grace period. Age is not liveness, and this does judge by age - but what
  it can get wrong is now a MISS (a pointer naming a swept blob reads as absent and the next
  compute republishes it), where the POSIX cache's worst case was deleting a path a reader
  had been handed. Everything below keeps it that way: a missing blob is never an error.

A LOCAL ``ResultCache`` stays in front as a read-through copy: requests are served from
files on this host's disk, under every guarantee the POSIX protocol already proved there,
and the object store is only asked which generation is current. The local copy of a
generation keeps the SAME generation token as the pointer, which is how a hit is known to
be current without downloading anything.

The store must honor conditional writes, and not every one does - obstore's own local
filesystem store refuses replace-if-unchanged, and S3-compatible servers vary. So
``check_conditional_writes`` ASKS the store at startup, and a store that does not refuse a
stale write is refused itself, naming the backend. Modelling the store from its docs is how
this repo has been wrong before.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path

from provender import Blobs, GRACE_S, check_store, open_store, update_mode
from provender import StoreUnsuitable as _StoreUnsuitable

from .errors import InputError

#: Bump if the pointer document changes shape incompatibly; a reader refuses (reads as a
#: miss) any pointer whose format it does not know, rather than guessing at its fields.
POINTER_FORMAT = 1
#: How old an unreferenced blob must be before ``sweep`` may delete it - provender's
#: default, and for the same reason: a blob is unreferenced for the few seconds between its
#: upload and its pointer's write, and a too-short grace costs a republication while a long
#: one costs only storage.
BLOB_GRACE_S = GRACE_S
#: Conditional-write attempts before a publication gives up. Each retry means another
#: writer published this key in between, so this many in a row is a storm, not a race.
SWAP_ATTEMPTS = 16
#: How many superseded generations a key keeps, and for how long (2026-09-19, by decision).
#: A republication - `Cache-Control: no-cache`, a weights upgrade, a changed epoch - used to
#: erase what it replaced, which left "what did this server answer in August?" unanswerable
#: and a bad upgrade unrollable. Bounded by BOTH, so a key's storage is bounded: history
#: costs one small pointer entry when the recomputation produced identical bytes (the blobs
#: dedupe) and one result when it did not. The local copy keeps NO history.
HISTORY_KEEP = 4
HISTORY_MAX_AGE_S = 30 * 24 * 3600
#: How often a store fault on a READ path may be reported. A store that is down faults on
#: every request, and a line per request is a denial of service on the operator's terminal.
WARN_INTERVAL_S = 60.0
#: Where a fill assembles its downloads, and how long a dead fill's directory survives.
WORK_PREFIX = ".fill-"
WORK_GRACE_S = 3600.0
_warned_at = 0.0


class ObjectStoreUnsuitable(InputError):
    """The store cannot carry the protocol - it does not honor conditional writes.

    provender raises its own ``StoreUnsuitable`` (a ValueError); this is the same fact as
    an ``InputError``, which is what `serve` turns into one line naming the fix instead of
    a traceback from inside a dependency.
    """


def check_conditional_writes(store, prefix: str = "") -> None:
    """Ask the store whether it can carry the protocol; ``ObjectStoreUnsuitable`` if not.

    The pointer needs BOTH conditional writes, so this never passes ``updates=False``.
    """
    try:
        check_store(store, prefix)
    except _StoreUnsuitable as e:
        raise ObjectStoreUnsuitable(
            f"{e}; use S3, GCS, Azure or R2, or drop --result-store for a local-only "
            "cache") from None


def _miss(where: str, exc: BaseException) -> None:
    """A store fault on a READ is a miss - but never a silent one.

    The local cache has always treated an unreadable entry as a miss (``ResultCache.get``
    swallows OSError), and the wire depends on it: SERVER.md promises 404 for a result that
    is not there and 410 for bytes that are gone, never 500. A shared store adds faults the
    local cache does not have - expired credentials, DNS, a 503 - and the first version let
    every one of them out of ``cache_get`` into a bare 500 on routes that are open to
    anonymous callers (found by review, 2026-09-19; the same class of defect as the scratch
    read that 0.12.3 fixed). So reads degrade and WRITES still raise: a publication that
    cannot reach the store must fail its job rather than pretend.
    """
    global _warned_at
    now = time.time()
    if now - _warned_at >= WARN_INTERVAL_S:
        _warned_at = now
        print(f"warning: result store unreachable ({where}): {type(exc).__name__}: {exc}; "
              "serving as a cache miss", file=sys.stderr, flush=True)


def _well_formed(ptr) -> bool:
    """Is this pointer one this code may act on?

    Checked FIELD BY FIELD, because a pointer is written by another host and read back
    here: the first version checked only that it was a dict with a known format, and then
    indexed ``generation``, ``files[...]["digest"]`` and ``["size"]`` blind - a truncated or
    hand-edited pointer raised KeyError/TypeError out of a read that had promised a miss
    (review, 2026-09-19). A digest is checked here too, where it is DATA; ``BlobStore.path``
    keeps its own check for the bytes this process supplies.
    """
    if not isinstance(ptr, dict) or ptr.get("format") != POINTER_FORMAT:
        return False
    if not isinstance(ptr.get("generation"), str) or not ptr["generation"]:
        return False
    if not _well_formed_files(ptr.get("files")):
        return False
    # history is optional and its ENTRIES are checked leniently by the readers that use
    # them (`_referenced`, `history`): a damaged past must not make the present unreadable
    return isinstance(ptr.get("history", []), list)


def _well_formed_files(files) -> bool:
    if not isinstance(files, dict):
        return False
    for name, blob in files.items():
        if not isinstance(name, str) or not isinstance(blob, dict):
            return False
        if not isinstance(blob.get("size"), int):
            return False
        digest = blob.get("digest")
        if (not isinstance(digest, str) or not digest.startswith("sha256:")
                or len(digest) != len("sha256:") + 64
                or any(c not in "0123456789abcdef" for c in digest[len("sha256:"):])):
            return False
    return True


def _generations(ptr) -> list:
    """The current publication and every kept predecessor, newest first, as
    ``{"generation", "published", "files"}`` - skipping any entry too damaged to use."""
    out = [{"generation": ptr["generation"], "published": ptr.get("published"),
            "files": ptr.get("files") or {}, "result": ptr.get("result"),
            "meta": ptr.get("meta"), "current": True}]
    for past in ptr.get("history") or []:
        if (isinstance(past, dict) and isinstance(past.get("generation"), str)
                and _well_formed_files(past.get("files"))):
            out.append({**past, "current": False})
    return out


class SharedResultCache:
    """``ResultCache``'s interface, with the object store as the authority and a local
    ``ResultCache`` as the copy requests are served from. See the module docstring."""

    def __init__(self, store, local, *, prefix: str = "", check: bool = True):
        if check:
            check_conditional_writes(store, prefix)
        self.store = store
        self.prefix = prefix
        self.local = local
        self.blobs = Blobs(store, prefix)
        # weak: a lock lives while a filler holds it and is forgotten after. A plain dict
        # keeps one entry per key forever, in a process that runs for weeks (review).
        self._fill_locks: "weakref.WeakValueDictionary[str, threading.Lock]" = (
            weakref.WeakValueDictionary())
        self._fill_guard = threading.Lock()

    @classmethod
    def open(cls, url: str, local) -> "SharedResultCache":
        store, prefix = open_store(url)
        return cls(store, local, prefix=prefix)

    @property
    def root(self) -> Path:
        return self.local.root

    # -- the pointer ---------------------------------------------------------------------

    def _pointer_path(self, key: str) -> str:
        if not key or "/" in key or key.startswith("."):
            raise ValueError(f"not a result key: {key!r}")
        return f"{self.prefix}results/{key}.json"

    def _read_pointer(self, key: str, *, for_write: bool = False):
        """``(pointer, update mode)`` or ``(None, None)``. A pointer this code cannot read
        - unknown format, not JSON - is reported as absent: a miss, never a guess.

        A store FAULT (credentials, network, a 503) is a miss too on a read, and is raised
        for a writer: ``for_write`` marks the caller as one. Reporting a fault as "absent"
        to a writer would have it publish over a pointer it could not read - the very race
        the conditional write exists to lose.
        """
        import obstore
        try:
            got = obstore.get(self.store, self._pointer_path(key))
        except FileNotFoundError:
            return None, None
        except Exception as e:                 # noqa: BLE001 - a read degrades, see _miss
            if for_write:
                raise
            _miss(f"reading the pointer for {key[:12]}", e)
            return None, None
        mode = update_mode(got.meta)
        try:
            ptr = json.loads(bytes(got.bytes()))
        except (ValueError, UnicodeDecodeError):
            return None, mode
        return (ptr if _well_formed(ptr) else None), mode

    def _swap(self, key: str, update):
        """Replace the pointer with ``update(current)`` by conditional write, rereading on
        every lost race; ``update`` returning None abandons the swap. Returns what was
        written, or None."""
        import obstore
        from obstore.exceptions import AlreadyExistsError, PreconditionError
        for _ in range(SWAP_ATTEMPTS):
            ptr, mode = self._read_pointer(key, for_write=True)
            new = update(ptr)
            if new is None:
                return None
            body = json.dumps(new, sort_keys=True).encode("utf-8")
            try:
                obstore.put(self.store, self._pointer_path(key), body,
                            mode=mode if mode is not None else "create")
            except (AlreadyExistsError, PreconditionError):
                continue                       # another writer moved it: read again
            return new
        raise RuntimeError(f"result {key}: {SWAP_ATTEMPTS} publications raced this one; "
                           "the pointer was not written")

    # -- ResultCache's interface ---------------------------------------------------------

    def generation(self, key: str) -> str | None:
        ptr, _ = self._read_pointer(key)
        return ptr["generation"] if ptr else None

    def published_result(self, key: str):
        """The current publication's result document, without downloading any bytes."""
        ptr, _ = self._read_pointer(key)
        return ptr.get("result") if ptr else None

    def history(self, key: str) -> list:
        """What this key has published, newest first and current first: one entry per kept
        generation with its token, publication time, result document and file sizes.

        Explicit on purpose. A superseded result is never served by an ordinary read - it
        is here to answer "what did this server answer in August?", to compare a result
        before and after a weights upgrade, and to roll one back deliberately.
        """
        ptr, _ = self._read_pointer(key)
        if ptr is None:
            return []
        return [{"generation": g["generation"], "published": g.get("published"),
                 "current": g["current"], "result": g.get("result"),
                 "bytes": sum(b["size"] for b in g["files"].values()),
                 "files": sorted(g["files"])}
                for g in _generations(ptr)]

    def fetch_generation(self, key: str, generation: str, dest) -> dict | None:
        """Materialize one kept generation into ``dest`` (which must exist); its entry, or
        None when that generation is not kept or its bytes have been swept.

        Into a directory the CALLER owns, never into the local cache: a historical read
        must not become what this host serves.
        """
        from .serve import ARTIFACT_NAMES, RESULT_NAME
        ptr, _ = self._read_pointer(key)
        if ptr is None:
            return None
        for g in _generations(ptr):
            if g["generation"] != generation:
                continue
            dest = Path(dest)
            for name, blob in g["files"].items():
                if name not in (RESULT_NAME, *ARTIFACT_NAMES):
                    continue                   # a foreign name may not decide a path
                try:
                    if not self.blobs.fetch(blob["digest"], dest / name):
                        return None
                except Exception as e:         # noqa: BLE001
                    _miss(f"fetching {name} of {key[:12]}@{generation[:8]}", e)
                    return None
            return g
        return None

    def get(self, key: str):
        """``(labels path, result)`` from the local copy of the CURRENT generation, filling
        it from the store first when the local copy is older or missing; None on a miss -
        including a pointer whose blobs have been swept."""
        ptr, _ = self._read_pointer(key)
        if ptr is None:
            return None
        if not self._fill(key, ptr):
            return None
        return self.local.get(key)

    def _fill(self, key: str, ptr) -> bool:
        """Make the local copy hold ``ptr``'s generation with all of its files."""
        import shutil
        import tempfile

        from .serve import ARTIFACT_NAMES, RESULT_NAME
        gen, files = ptr["generation"], ptr.get("files") or {}
        if RESULT_NAME not in files:
            return False
        with self._fill_lock(key):
            local_dir = self.local._generation_dir(key, gen)
            have_gen = self.local.generation(key) == gen and (local_dir / RESULT_NAME).exists()
            # ONLY these names, and they are spelled out here rather than taken from the
            # pointer: a pointer is written by another host, and a name of its choosing
            # ("../..", an absolute path) would decide where these bytes land.
            wanted = [n for n in (RESULT_NAME, *ARTIFACT_NAMES) if n in files]
            missing = [n for n in wanted if not (have_gen and (local_dir / n).exists())]
            if not missing:
                return True
            work = self._fresh_work_dir()
            try:
                got = []
                for name in missing:
                    try:
                        here = self.blobs.fetch(files[name]["digest"], work / name)
                    except Exception as e:     # noqa: BLE001 - a read degrades, see _miss
                        _miss(f"fetching {name} of {key[:12]}", e)
                        here = False
                    if here:
                        got.append(name)
                    elif name == RESULT_NAME:
                        return False           # swept: a miss, and the next compute heals it
                    # an artifact that is gone is not a reason to lose the labels: a missing
                    # preview used to make the whole entry a miss, and a lost thumbnail is
                    # not worth a GPU recompute (review, 2026-09-19)
                if not have_gen:
                    try:
                        self.local.put(key, work / RESULT_NAME, ptr.get("result") or {},
                                       ptr.get("meta") or {},
                                       preview_path=_present(work / "preview.png"),
                                       statistics_path=_present(work / "statistics.json"),
                                       generation=gen)
                    except FileExistsError:
                        # another process placed this generation; it is only usable if that
                        # process also made it current - otherwise this read is a miss
                        # rather than a different generation served as if it were current
                        return self.local.generation(key) == gen
                else:
                    for name in got:
                        self.local.add_artifact(key, name, work / name, generation=gen)
                return True
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _fill_lock(self, key: str) -> threading.Lock:
        with self._fill_guard:
            lock = self._fill_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._fill_locks[key] = lock
            return lock                        # the caller's reference keeps it alive

    def _fresh_work_dir(self) -> Path:
        """Where a fill assembles what it downloaded, with older leavings removed first.

        Dotted, so `cache_admin` and eviction skip it - which means a process killed
        mid-fill leaks a directory nothing reclaims (the 2026-09-09 finding, in a new
        place). Whoever creates it owns removing it, so every fill also clears the ones
        left by fills that died.
        """
        import shutil
        import time as _time
        root = self.local.root
        for stale in root.glob(f"{WORK_PREFIX}*"):
            try:
                if _time.time() - stale.stat().st_mtime > WORK_GRACE_S:
                    shutil.rmtree(stale, ignore_errors=True)
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix=WORK_PREFIX, dir=root))

    def put(self, key: str, labels_path, result: dict, meta: dict,
            preview_path=None, statistics_path=None) -> str:
        """Publish: blobs first, then the pointer, then the local copy. Returns the
        generation token, which the local copy shares."""
        from .serve import RESULT_NAME
        sources = {RESULT_NAME: labels_path, "preview.png": preview_path,
                   "statistics.json": statistics_path}
        files = {name: self.blobs.put_file(src) for name, src in sources.items()
                 if src and Path(src).exists()}
        gen, now = uuid.uuid4().hex, time.time()

        def publish(current):
            # what is being replaced joins the history, and the oldest falls off it
            return {"format": POINTER_FORMAT, "generation": gen, "published": now,
                    "files": files, "result": result, "meta": meta,
                    "history": _kept_history(current, now)}
        pointer = self._swap(key, publish)
        self._verify(pointer["files"], sources)
        try:
            self.local.put(key, labels_path, result, meta, preview_path=preview_path,
                           statistics_path=statistics_path, generation=gen)
        except Exception as e:                 # noqa: BLE001
            # The publication HAPPENED - every host can read it - so a local copy that
            # cannot be written must not fail the job that just produced it (a full disk
            # did exactly that: `put` raised after the result was visible cluster-wide,
            # review 2026-09-19). The next read on this host fills the copy again.
            _miss(f"keeping a local copy of {key[:12]}", e)
        return gen

    def _verify(self, files: dict, sources: dict) -> None:
        """Put back any blob that was not uploaded because it already existed and has since
        been swept - the window between that check and the pointer write. The pointer now
        references them, so this is the last moment anything may quietly remove them."""
        for name, blob in files.items():
            try:
                if not self.blobs.has(blob["digest"]):
                    self.blobs.put_file(sources[name])
            except Exception as e:             # noqa: BLE001 - the pointer is already out
                _miss(f"re-checking {name}", e)

    def add_artifact(self, key: str, name: str, src_path, generation=None) -> bool:
        """Add an artifact to the publication it was rendered for; False when that
        publication is no longer current (or the entry is gone). The pointer's conditional
        write is what makes it safe: an artifact can never be recorded beside another
        publication's labels - PROVIDED a generation is named, which is what
        ``serve.publish_completion`` does. Without one this grafts the artifact onto
        whatever publication is current, which for a late worker is not the one it rendered.

        Never raises: this runs on the overlap thread after "done" has been served, and a
        store fault there must not take the rest of that thread's work (the statistics) with
        it (review, 2026-09-19)."""
        from .serve import ARTIFACT_NAMES
        if name not in ARTIFACT_NAMES:
            raise ValueError(f"not an artifact: {name!r}")

        def update(ptr):
            if ptr is None or (generation and ptr["generation"] != generation):
                return None
            files = dict(ptr.get("files") or {})
            files[name] = blob
            return {**ptr, "files": files}
        try:
            blob = self.blobs.put_file(src_path)
            written = self._swap(key, update)
            if written is None:
                return False
            self._verify({name: blob}, {name: src_path})   # the sweep window, as in `put`
        except Exception as e:                 # noqa: BLE001
            _miss(f"placing {name} on {key[:12]}", e)
            return False
        self.local.add_artifact(key, name, src_path, generation=written["generation"])
        return True

    def delete(self, key: str) -> bool:
        """Remove the entry everywhere this host can reach: the pointer and the local copy.
        Other hosts' local copies stop being served at their next read of the pointer."""
        import obstore
        ptr, _ = self._read_pointer(key)
        try:
            obstore.delete(self.store, self._pointer_path(key))
        except FileNotFoundError:
            pass
        local = self.local.delete(key)
        return ptr is not None or local

    def list(self, limit: int = 500) -> list:
        """The newest published entries, read from the pointers alone.

        At most ``limit`` pointers are READ: they are ordered by the listing's own
        last-modified first. Reading every pointer in the bucket and then slicing was one
        request per entry - fine for a local directory, minutes and tens of thousands of
        requests on a bucket several servers share (review, 2026-09-19). The order within
        the answer is still the publication time each pointer records.
        """
        from .serve import RESULT_NAME, resource_links
        out = []
        for ptr in self._scan_pointers(newest_first=True, limit=limit)[0]:
            meta, files = ptr.get("meta") or {}, ptr.get("files") or {}
            if RESULT_NAME not in files:
                continue
            entry = {"key": ptr["_key"], "task": meta.get("task"),
                     "identity": meta.get("identity"), "options": meta.get("options"),
                     "computed": meta.get("computed"), "bytes": files[RESULT_NAME]["size"]}
            links = resource_links(meta.get("task"), meta.get("identity"),
                                   meta.get("options"), preview="preview.png" in files,
                                   statistics="statistics.json" in files)
            if links:
                entry["links"] = links
            out.append(entry)
        out.sort(key=lambda e: e.get("computed") or 0, reverse=True)
        return out[:limit]

    def evict(self) -> None:
        """Bounds the LOCAL copy only. The store is bounded by ``sweep``: it keeps no
        access times, and a count-bounded LRU over it would need an index this protocol
        deliberately does not have."""
        self.local.evict()

    # -- store maintenance ---------------------------------------------------------------

    def _scan_pointers(self, *, newest_first: bool = False, limit: int | None = None):
        """``(pointers this code can read, how many it could NOT)``.

        An object under ``results/`` that is not a
        pointer this version understands - a stray upload, a `.tmp` file from someone's
        sync, a FORMAT FROM A NEWER WRITER - is skipped and counted, never raised: one
        stray object used to abort both ``list`` and ``sweep`` permanently, and a sweep
        that never runs is a bucket that never stops growing (review, 2026-09-19).

        ``newest_first`` orders by the pointers' own last-modified before reading any of
        them, so ``limit`` costs that many reads instead of one per entry in the bucket.
        """
        import obstore
        base = f"{self.prefix}results/"
        entries = []
        for batch in obstore.list(self.store, base):
            for obj in batch:
                name = obj["path"][len(base):]
                if "/" in name or not name.endswith(".json"):
                    continue
                entries.append((obj.get("last_modified"), name[:-len(".json")]))
        if newest_first:
            entries.sort(key=lambda e: (e[0] is not None, e[0]), reverse=True)
        out, unreadable = [], 0
        for _, key in entries:
            if limit is not None and len(out) >= limit:
                break
            try:
                ptr, _mode = self._read_pointer(key)
            except ValueError:                 # not a key this code would ever write
                ptr = None
            if ptr is None:
                unreadable += 1
                continue
            out.append({**ptr, "_key": key})
        return out, unreadable

    def sweep(self, *, max_age_s: float | None = None, grace_s: float = BLOB_GRACE_S,
              now: float | None = None) -> dict:
        """Delete what no pointer needs: pointers published more than ``max_age_s`` ago
        (when given), then blobs no remaining pointer - current OR kept history - refers
        to, through provender's sweep.

        What is haversack's here is the LIVE SET, which is the only part that knows what a
        result is. The grace belongs to provender and is the same idea it always was: a
        blob is unreferenced for the moments between its upload and its pointer's write,
        and ``put`` re-checks its blobs afterwards for exactly that window. At the default
        grace nothing young is a candidate at all; at ``grace_s=0`` what slips through is
        a miss the next computation repairs.

        Expiring a pointer is an unconditional delete, so a republication that lands
        between reading the pointer and deleting it is expired with it: a miss.
        """
        import obstore
        now = time.time() if now is None else now
        referenced, expired = set(), 0
        pointers, unreadable = self._scan_pointers()
        for ptr in pointers:
            if max_age_s is not None and (ptr.get("published") or 0) < now - max_age_s:
                try:
                    obstore.delete(self.store, self._pointer_path(ptr["_key"]))
                except FileNotFoundError:
                    pass
                expired += 1
                continue
            for gen in _generations(ptr):      # the current publication AND its history
                for blob in gen["files"].values():
                    referenced.add(blob["digest"])
        if unreadable:
            # A pointer this version cannot read may still name live blobs - a writer on a
            # newer POINTER_FORMAT is the case that matters. Its blobs would look
            # unreferenced, and an old host's sweeper would collect a new host's results
            # (review, 2026-09-19). Expiry above is per-pointer and safe; deleting blobs is
            # not, so it waits until whatever is unreadable has been explained.
            print(f"warning: {unreadable} object(s) under results/ could not be read as "
                  "pointers; deleting no blobs this sweep", file=sys.stderr, flush=True)
            return {"expired_pointers": expired, "deleted_blobs": 0,
                    "unreadable_pointers": unreadable}
        # an empty live set is a refusal in provender unless it is meant - and here it IS
        # meant, because every pointer was read and none of them referenced anything
        got = self.blobs.sweep(keep=referenced, grace_s=grace_s, now=now,
                               allow_empty=not referenced)
        return {"expired_pointers": expired, "deleted_blobs": got["deleted"],
                "unreadable_pointers": 0}


    # -- migration -----------------------------------------------------------------------

    def push(self, *, conflict: str = "skip", limit: int | None = None,
             report=None) -> dict:
        """Publish this host's LOCAL cache into the store; counts by outcome.

        The point of the transition (`docs/cache-consolidation.md`): a cache that has been
        filling up for months is worth GPU-hours, and nothing else recovers it once the
        local protocol goes.

        Each entry keeps the generation token it already has, so the local copy is
        instantly the store's own copy of that publication and the first read after the
        switch downloads nothing. A legacy flat entry - from before generations - is given
        one, and costs one download the first time it is read.

        Idempotent by construction: blobs are create-if-absent and the pointer is written
        conditionally, so a push interrupted halfway is rerun, and two hosts pushing
        overlapping caches upload the shared bytes once.

        ``conflict`` decides what happens when the store already has the key:
        ``"skip"`` (the default: it may be newer than ours), ``"newer"`` (compare the
        ``computed`` timestamps and replace only when ours is newer), or ``"force"``.
        """
        from .serve import ARTIFACT_NAMES, RESULT_NAME
        if conflict not in ("skip", "newer", "force"):
            raise InputError(f"conflict {conflict!r}: expected skip, newer or force")
        out = {"pushed": 0, "skipped": 0, "replaced": 0, "failed": 0, "unreadable": 0}
        for key in self._local_keys(limit=limit):
            where = self.local._resolve(key)           # leases it: not swept mid-upload
            if where is None:
                out["unreadable"] += 1
                continue
            if conflict == "skip" and self._read_pointer(key)[0] is not None:
                # asked BEFORE the bytes are touched: a rerun over a cache of thousands
                # should cost one pointer read each, not a re-hash of every result. The
                # swap below asks again, so a key published in between is still not lost
                out["skipped"] += 1
                if report:
                    report(key, "skipped")
                continue
            gen = self.local.generation(key) or uuid.uuid4().hex
            try:
                result = json.loads((where / "result.json").read_text(encoding="utf-8"))
                meta = json.loads((where / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out["unreadable"] += 1                 # half an entry is not worth pushing
                continue
            sources = {n: where / n for n in (RESULT_NAME, *ARTIFACT_NAMES)
                       if (where / n).exists()}
            if RESULT_NAME not in sources:
                out["unreadable"] += 1
                continue
            try:
                files = {n: self.blobs.put_file(src) for n, src in sources.items()}
                outcome = self._swap(key, _migrating(conflict, gen, files, result, meta))
                self._verify(files, sources)
            except Exception as e:                     # noqa: BLE001 - one bad entry is
                out["failed"] += 1                     # not a reason to abandon the rest
                if report:
                    report(key, f"failed: {type(e).__name__}: {e}")
                continue
            if outcome is None:
                out["skipped"] += 1
            else:
                out["replaced" if outcome.get("history") else "pushed"] += 1
            if report:
                report(key, "skipped" if outcome is None else "pushed")
        return out

    def pull(self, *, limit: int | None = None, report=None) -> dict:
        """Materialize the store's entries into this host's local cache.

        For a host that wants to be warm before it serves, and the way out of the store:
        after a pull, the local cache answers on its own. Entries already current here cost
        one pointer read and no bytes.
        """
        out = {"pulled": 0, "current": 0, "failed": 0, "unreadable": 0}
        pointers, unreadable = self._scan_pointers(newest_first=True, limit=limit)
        out["unreadable"] = unreadable
        for ptr in pointers:
            key = ptr["_key"]
            if self.local.generation(key) == ptr["generation"]:
                out["current"] += 1
                if report:
                    report(key, "current")
                continue
            try:
                ok = self._fill(key, ptr)
            except Exception as e:                     # noqa: BLE001
                ok = False
                if report:
                    report(key, f"failed: {type(e).__name__}: {e}")
            out["pulled" if ok else "failed"] += 1
            if report and ok:
                report(key, "pulled")
        return out

    def _local_keys(self, *, limit: int | None = None) -> list:
        """Every entry in the local cache, newest first. Dotfiles are not entries - a
        staging directory, a tomb mid-reclamation and a fill's work directory all live
        there, and pushing one would publish an unfinished result."""
        try:
            dirs = [d for d in self.local.root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")]
        except OSError:
            return []

        def _mtime(d):
            try:
                return d.stat().st_mtime
            except OSError:
                return 0.0
        dirs.sort(key=_mtime, reverse=True)
        return [d.name for d in dirs[:limit]]


def _migrating(conflict: str, gen: str, files: dict, result: dict, meta: dict):
    """The pointer a migration writes, or None to leave the store's entry alone.

    A key already in the store may be NEWER than the copy being pushed - another host
    computed it after this cache went cold - so the default refuses to replace it. Under
    ``newer`` the two ``computed`` timestamps decide, and a store entry with no timestamp
    is treated as unknown rather than old: a migration should not overwrite what it cannot
    compare.
    """
    def update(current):
        if current is not None and conflict != "force":
            if conflict == "skip":
                return None
            theirs = (current.get("meta") or {}).get("computed")
            ours = meta.get("computed")
            if theirs is None or ours is None or theirs >= ours:
                return None
        now = time.time()
        return {"format": POINTER_FORMAT, "generation": gen,
                "published": meta.get("computed") or now, "files": files,
                "result": result, "meta": meta,
                "history": _kept_history(current, now)}
    return update


def _kept_history(current, now: float) -> list:
    """The history a publication replacing ``current`` should carry: what it replaces,
    then that pointer's own history, bounded by ``HISTORY_KEEP`` and ``HISTORY_MAX_AGE_S``.

    Bounded by both on purpose. Count alone lets a key that is republished constantly hold
    four copies of a large result forever; age alone lets a key republished hourly for a
    month hold seven hundred.
    """
    if current is None:
        return []
    older = [p for p in (current.get("history") or [])
             if isinstance(p, dict) and isinstance(p.get("generation"), str)]
    kept = [{"generation": current["generation"], "published": current.get("published"),
             "files": current.get("files") or {},
             "result": current.get("result"), "meta": current.get("meta")}, *older]
    fresh = [p for p in kept
             if not isinstance(p.get("published"), (int, float))
             or p["published"] > now - HISTORY_MAX_AGE_S]
    return fresh[:HISTORY_KEEP]


def _present(p: Path):
    return p if p.exists() else None
