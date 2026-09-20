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

from provender import Blobs, GRACE_S, check_store, update_mode
from provender import StoreUnsuitable as _StoreUnsuitable
from provender import open_store as _open_store

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


def open_store(url: str):
    """provender's, with its ValueError turned into an ``InputError``.

    The house rule is that an error names the fix in one line. A malformed URL used to say
    exactly what the forms are; after the extraction it reached the caller as a bare
    ValueError, which `serve` and the CLI then wrapped in advice about CREDENTIALS - for a
    URL that never got as far as using any (review, 2026-09-20).
    """
    try:
        return _open_store(url)
    except _StoreUnsuitable:
        raise
    except ValueError as e:
        raise InputError(str(e)) from None


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
    (review, 2026-09-19). A digest is checked here too, where it is DATA; ``provender.Blobs.path``
    keeps its own check for the bytes this process supplies.
    """
    if not isinstance(ptr, dict) or ptr.get("format") != POINTER_FORMAT:
        return False
    if not isinstance(ptr.get("generation"), str) or not ptr["generation"]:
        return False
    if not _well_formed_files(ptr.get("files")):
        return False
    # History is OPTIONAL and never load-bearing for the present. Requiring it to be a
    # list made `"history": null` - what another language emits for "none" - lose the
    # current result and, because the pointer then counted as unreadable, freeze blob
    # deletion for the whole store (review, 2026-09-20). A past this code cannot read is
    # simply no past: `_generations` takes the entries one at a time.
    return True


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


def _history_of(ptr) -> list:
    """A pointer's history entries, or none. The shape is checked HERE so that no caller
    has to: `"history": 7` iterated as an int and raised out of a read that had promised a
    miss - the same lesson as every other field in this document (review, 2026-09-20)."""
    hist = ptr.get("history")
    return hist if isinstance(hist, list) else []


def _usable(past, *, now: float, keep_undated: bool = False) -> bool:
    """Is this history entry one every reader AND the writer will honor?

    ONE rule, asked in one place. The writer used a looser test than the readers, so an
    entry with damaged ``files`` was invisible to ``history`` and to the sweep while still
    occupying one of ``HISTORY_KEEP`` slots - immortal, because it also had no date, and
    it pushed real predecessors off the end (review, 2026-09-20).

    An entry that cannot be DATED is not kept: an undatable or future-dated generation was
    immune to ``HISTORY_MAX_AGE_S`` forever, which is the opposite of a bounded history.
    """
    if not isinstance(past, dict) or not isinstance(past.get("generation"), str):
        return False
    if not past["generation"] or not _well_formed_files(past.get("files")):
        return False
    when = past.get("published")
    if not isinstance(when, (int, float)) or isinstance(when, bool):
        return keep_undated
    return when > now - HISTORY_MAX_AGE_S


def _generations(ptr, *, now: float | None = None) -> list:
    """The current publication and every kept predecessor, newest first.

    The age bound is applied HERE as well as at write time. It used to be applied only
    when a key was republished, so a key published three times and then left alone kept
    all three for ever and the sweep kept their blobs with them - the bound did nothing
    for exactly the quiescent keys where it matters (review, 2026-09-20).
    """
    now = time.time() if now is None else now
    out = [{"generation": ptr["generation"], "published": ptr.get("published"),
            "files": ptr.get("files") or {}, "result": ptr.get("result"),
            "meta": ptr.get("meta"), "current": True}]
    for past in _history_of(ptr):
        if _usable(past, now=now):
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
            dest.mkdir(parents=True, exist_ok=True)
            written = []
            for name, blob in g["files"].items():
                if name not in (RESULT_NAME, *ARTIFACT_NAMES):
                    continue                   # a foreign name may not decide a path
                try:
                    if not self.blobs.fetch(blob["digest"], dest / name):
                        return None
                except Exception as e:         # noqa: BLE001
                    _miss(f"fetching {name} of {key[:12]}@{generation[:8]}", e)
                    return None
                written.append(name)
            # nothing written is not success: a generation whose files are all named
            # something this code will not place left the caller an empty directory and a
            # non-None answer (review, 2026-09-20)
            return {**g, "written": written} if RESULT_NAME in written else None
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
                    if not here:
                        _warn_if_corrupt(self.blobs, files[name]["digest"])
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
                        # The generation directory is already here and INCOMPLETE - a file
                        # deleted under it, or another process mid-copy. Put back what is
                        # missing, atomically, rather than declaring the copy current
                        # while its labels are gone: a re-pull is the obvious repair and
                        # it has to actually repair (review, 2026-09-20).
                        for name in got:
                            _place(local_dir, name, work / name)
                        return (self.local.generation(key) == gen
                                and (local_dir / RESULT_NAME).exists())
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
                    "history": _kept_history(current, now, gen)}
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

    def delete(self, key: str, *, purge: bool = True) -> bool:
        """Remove the entry everywhere this host can reach: the pointer, its BYTES, and
        the local copy. Other hosts' local copies stop being served at their next read of
        the pointer.

        ``purge`` (the default) deletes the blobs this entry held - current generation and
        history - once no remaining pointer references them, without waiting for the
        sweep's grace. The decision behind history says deletion means gone, and a `delete`
        that left the bytes readable in the bucket for a day, or for ever if nothing runs a
        sweep, does not mean that (review, 2026-09-20). Bytes another entry shares are
        KEPT, and so are bytes whose fate cannot be established because some pointer here
        cannot be read - that case is reported, because for a deletion "I could not tell"
        must not look like "done".

        Returns whether anything was removed. Asked of the OBJECT, not of a parse: a
        pointer this version cannot read is still an entry, and answering False for it
        told an operator deleting a patient's result that there had been nothing there.
        """
        import obstore
        existed = False
        try:
            obstore.head(self.store, self._pointer_path(key))
            existed = True
        except FileNotFoundError:
            pass                               # anything else the store raises comes out:
                                               # a delete must not report success on doubt
        ptr, _ = self._read_pointer(key)
        mine = {b["digest"] for g in (_generations(ptr) if ptr else [])
                for b in g["files"].values()}
        try:
            obstore.delete(self.store, self._pointer_path(key))
        except FileNotFoundError:
            pass
        local = self.local.delete(key)
        if purge and mine:
            self._purge(key, mine)
        return existed or local

    def _purge(self, key: str, digests: set) -> None:
        """Delete the blobs a removed entry held, except those another entry still needs.

        Refuses - loudly - when a pointer cannot be read, because then "no remaining entry
        references these bytes" is not something this process knows.
        """
        import obstore
        pointers, unreadable = self._scan_pointers()
        if unreadable:
            print(f"warning: {key[:12]}... was deleted, but {unreadable} object(s) under "
                  "results/ could not be read, so its bytes were left in place. Remove or "
                  "repair them and run `haversack cache sweep` to finish the deletion.",
                  file=sys.stderr, flush=True)
            return
        now = time.time()
        keep = {b["digest"] for ptr in pointers for g in _generations(ptr, now=now)
                for b in g["files"].values()}
        for digest in digests - keep:
            try:
                obstore.delete(self.store, self.blobs.path(digest))
            except FileNotFoundError:
                pass

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
        result is.

        Blobs are LISTED BEFORE the pointers are read, and that list is what provender is
        asked to delete from. A publication landing after the listing uploaded its blobs
        after it too, so they are not candidates whatever the clocks say; one landing
        before the pointers are read is seen referencing them. Letting the sweep list for
        itself inverted that order and left only an age comparison between this host's
        clock and the store's - which loses a live blob at ``grace_s=0``, reproduced by
        review (2026-09-20).

        Expiring a pointer is an unconditional delete, so a republication that lands
        between reading the pointer and deleting it is expired with it: a miss.
        """
        import obstore
        from provender import EmptyKeepSet
        now = time.time() if now is None else now
        candidates = self.blobs.entries(older_than=now - grace_s)   # BEFORE the pointers
        referenced, expired = set(), 0
        pointers, unreadable = self._scan_pointers()
        for ptr in pointers:
            published = ptr.get("published")
            datable = isinstance(published, (int, float)) and not isinstance(published, bool)
            # an entry nothing can date is never expired: the one field cleanup judges by
            # must be one it can read, or it is not evidence (review, 2026-09-20)
            if max_age_s is not None and datable and published < now - max_age_s:
                try:
                    obstore.delete(self.store, self._pointer_path(ptr["_key"]))
                except FileNotFoundError:
                    pass
                expired += 1
                continue
            for gen in _generations(ptr, now=now):   # current AND its kept history
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
            return {"expired_pointers": expired, "deleted_blobs": 0, "already_gone": 0,
                    "unreadable_pointers": unreadable}
        # provender refuses an empty live set unless told it was meant, and passing
        # `allow_empty=not referenced` disarmed that guard on every call - which is exactly
        # the case it exists for: no pointers FOUND is not evidence that nothing is live.
        # A prefix that drifted, a layout change, a listing that returned nothing: the
        # answer is to refuse and say so, not to empty the bucket (review, 2026-09-20).
        # "the index is empty" and "the index is not where I looked" are the two cases,
        # and only the first may sweep: pointers that were FOUND and then expired are an
        # index that was read
        try:
            got = self.blobs.sweep(keep=referenced, candidates=candidates, grace_s=0,
                                   now=now, allow_empty=bool(pointers))
        except EmptyKeepSet:
            if candidates:
                print(f"warning: no readable entries under {self.prefix}results/, but "
                      f"{len(candidates)} blob(s) are stored there; deleting none. If the "
                      "entries really are gone, `haversack cache clean` the local copy and "
                      "remove the prefix by hand.", file=sys.stderr, flush=True)
            return {"expired_pointers": expired, "deleted_blobs": 0, "already_gone": 0,
                    "unreadable_pointers": 0}
        return {"expired_pointers": expired, "deleted_blobs": got["deleted"],
                "already_gone": got["already_gone"], "unreadable_pointers": 0}


    # -- migration -----------------------------------------------------------------------

    def push(self, *, conflict: str = "skip", limit: int | None = None,
             report=None) -> dict:
        """Publish this host's LOCAL cache into the store; counts by outcome.

        The point of the transition (`docs/cache-consolidation.md`): a cache that has been
        filling up for months is worth GPU-hours, and nothing else recovers it once the
        local protocol goes.

        Each entry keeps the generation token it already has, so the local copy is
        instantly the store's own copy of that publication and the first read after the
        switch downloads nothing. The token comes from the DIRECTORY that was resolved and
        leased, never from a second read of the pointer: a server publishing this key in
        between would otherwise bind one generation's bytes to another generation's token,
        and the pushing host would then believe its copy current forever (review,
        2026-09-20). A legacy flat entry - from before generations - is given a fresh
        token, and costs one download the first time it is read.

        Idempotent by construction: blobs are create-if-absent and the pointer is written
        conditionally, so a push interrupted halfway is rerun, and two hosts pushing
        overlapping caches upload the shared bytes once. ``limit`` bounds the WORK, not the
        entries examined - a skipped key does not use up a slot, or a rerun with the same
        limit would keep migrating the same few and never finish.

        ``conflict`` decides what happens when the store already has the key:
        ``"skip"`` (the default: it may be newer than ours), ``"newer"`` (compare the
        ``computed`` timestamps and replace only when ours is newer), or ``"force"``.
        """
        from .serve import ARTIFACT_NAMES, RESULT_NAME
        if conflict not in ("skip", "newer", "force"):
            raise InputError(f"conflict {conflict!r}: expected skip, newer or force")
        out = {"pushed": 0, "skipped": 0, "replaced": 0, "failed": 0, "unreadable": 0}
        keys, listed = self._local_keys()
        if not listed:
            out["failed"] += 1                         # a cache root nobody can read is
            if report:                                 # not "nothing to do"
                report(str(self.local.root), "failed: the local cache cannot be listed")
            return out
        for key in keys:
            if limit is not None and out["pushed"] + out["replaced"] >= limit:
                break
            where = self.local._resolve(key)           # leases it: not swept mid-upload
            if where is None:
                out["unreadable"] += 1
                continue
            # the token of the directory in hand, not whatever the pointer says NOW
            gen = where.name[2:] if where.name.startswith("g-") else uuid.uuid4().hex
            try:
                result = json.loads((where / "result.json").read_text(encoding="utf-8"))
                meta = json.loads((where / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out["unreadable"] += 1                 # half an entry is not worth pushing
                continue
            if not isinstance(result, dict) or not isinstance(meta, dict):
                out["unreadable"] += 1                 # a pointer carries documents, and
                continue                               # every reader indexes them
            sources = {n: where / n for n in (RESULT_NAME, *ARTIFACT_NAMES)
                       if (where / n).exists()}
            if RESULT_NAME not in sources:
                out["unreadable"] += 1
                continue
            # Asked BEFORE the bytes are touched, for every policy that can refuse: a rerun
            # over a cache of thousands should cost one pointer read each, not a re-hash
            # and re-upload of every result. `_swap` asks again under the conditional
            # write, so a key published in between is still not lost.
            try:
                current, _mode = self._read_pointer(key, for_write=True)
            except Exception as e:                     # noqa: BLE001
                out["failed"] += 1
                if report:
                    report(key, f"failed: {type(e).__name__}: {e}")
                continue
            if current is not None and _refuses(conflict, current, meta):
                out["skipped"] += 1
                if report:
                    report(key, "skipped (the store has it)")
                continue
            try:
                files = {n: self.blobs.put_file(src) for n, src in sources.items()}
                outcome = self._swap(key, _migrating(conflict, gen, files, result, meta))
                if outcome is not None:
                    self._verify(files, sources)
            except Exception as e:                     # noqa: BLE001 - one bad entry is
                out["failed"] += 1                     # not a reason to abandon the rest
                if report:
                    report(key, f"failed: {type(e).__name__}: {e}")
                continue
            if outcome is None:                        # it changed under us after all
                out["skipped"] += 1
            else:
                out["replaced" if current is not None else "pushed"] += 1
            if report:
                report(key, "skipped" if outcome is None
                       else ("replaced" if current is not None else "pushed"))
        return out

    def pull(self, *, limit: int | None = None, report=None) -> dict:
        """Materialize the store's entries into this host's local cache.

        For a host that wants to be warm before it serves, and the way out of the store:
        after a pull, the local cache answers on its own. Entries already complete here
        cost one pointer read and no bytes - "already current" is asked of the FILES, not
        only of the generation token, so a pull repairs a local copy whose labels went
        missing instead of reporting it current (review, 2026-09-20).

        Oldest first, because the local cache evicts least-recently-used: pulling newest
        first made each new entry evict the one before it. Even so a store with more
        entries than the local bound cannot fit, and what would not fit is REPORTED rather
        than counted as pulled.
        """
        from .serve import ARTIFACT_NAMES, RESULT_NAME
        out = {"pulled": 0, "current": 0, "failed": 0, "unreadable": 0, "evicted": 0}
        placed: list = []
        pointers, unreadable = self._scan_pointers(newest_first=True, limit=limit)
        out["unreadable"] = unreadable
        for ptr in reversed(pointers):                 # oldest first: see above
            key, gen = ptr["_key"], ptr["generation"]
            local_dir = self.local._generation_dir(key, gen)
            wanted = [n for n in (RESULT_NAME, *ARTIFACT_NAMES) if n in (ptr["files"] or {})]
            if (self.local.generation(key) == gen
                    and all((local_dir / n).exists() for n in wanted)):
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
            if ok:
                placed.append(key)
            if report and ok:
                report(key, "pulled")
        # the local cache is count-bounded, and a pull of more entries than it keeps cannot
        # leave them all servable. Asked only of what THIS pull placed - an entry that
        # failed was never there to evict - and reported rather than counted as pulled.
        gone = sum(1 for key in placed if self.local.generation(key) is None)
        if gone:
            out["evicted"] = gone
            out["pulled"] = max(0, out["pulled"] - gone)
            print(f"warning: {gone} pulled entr{'y' if gone == 1 else 'ies'} did not fit "
                  f"in the local cache (it keeps {self.local.keep}); they were evicted "
                  "again. Raise the bound or pull fewer.", file=sys.stderr, flush=True)
        return out

    def _local_keys(self, *, limit: int | None = None):
        """``(entries newest first, could the root be listed)``. Dotfiles are not entries - a
        staging directory, a tomb mid-reclamation and a fill's work directory all live
        there, and pushing one would publish an unfinished result."""
        try:
            dirs = [d for d in self.local.root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")]
        except OSError:
            return [], False                   # unreadable is not "nothing to do"

        def _mtime(d):
            try:
                return d.stat().st_mtime
            except OSError:
                return 0.0
        dirs.sort(key=_mtime, reverse=True)
        return [d.name for d in dirs[:limit]], True


def _migrating(conflict: str, gen: str, files: dict, result: dict, meta: dict):
    """The pointer a migration writes, or None to leave the store's entry alone.

    A key already in the store may be NEWER than the copy being pushed - another host
    computed it after this cache went cold - so the default refuses to replace it. Under
    ``newer`` the two ``computed`` timestamps decide, and a store entry with no timestamp
    is treated as unknown rather than old: a migration should not overwrite what it cannot
    compare.
    """
    def update(current):
        if current is not None and _refuses(conflict, current, meta):
            return None
        now = time.time()
        # `published` is when this pointer was written, NOT when the result was computed
        # (which lives in meta). Taking the compute time put a migrated entry's own
        # history instantly past HISTORY_MAX_AGE_S - so pushing a months-old cache over an
        # existing key discarded what it replaced - and let a non-numeric `computed` from
        # a local meta.json reach `sweep`, where it raised for every host (review,
        # 2026-09-20).
        return {"format": POINTER_FORMAT, "generation": gen, "published": now,
                "files": files, "result": result, "meta": meta,
                "history": _kept_history(current, now, gen)}
    return update


def _refuses(conflict: str, current, meta) -> bool:
    """Would this conflict policy leave the store's entry alone? Asked before any bytes
    are hashed, and again inside the conditional write, which is the authority."""
    if conflict == "force":
        return False
    if conflict == "skip":
        return True
    theirs = (current.get("meta") or {}).get("computed")
    ours = meta.get("computed")
    if not isinstance(theirs, (int, float)) or not isinstance(ours, (int, float)):
        return True                            # not comparable: do not overwrite
    return theirs >= ours


def _kept_history(current, now: float, gen: str | None = None) -> list:
    """The history a publication replacing ``current`` should carry: what it replaces,
    then that pointer's own history, bounded by ``HISTORY_KEEP`` and ``HISTORY_MAX_AGE_S``.

    Bounded by both on purpose. Count alone lets a key that is republished constantly hold
    four copies of a large result forever; age alone lets a key republished hourly for a
    month hold seven hundred.
    """
    if current is None:
        return []
    if current.get("generation") == gen:
        # a re-push of the same generation (`cache push --conflict force` over an entry
        # this host already published): it is not its own predecessor
        return [p for p in _history_of(current) if _usable(p, now=now)][:HISTORY_KEEP]
    older = [p for p in _history_of(current) if _usable(p, now=now)]
    kept = [{"generation": current["generation"], "published": current.get("published"),
             "files": current.get("files") or {},
             # `meta` is deliberately NOT carried: no reader has ever looked at a history
             # entry's meta, and each copy cost a full document in every pointer read
             "result": current.get("result")}, *older]
    return [p for p in kept if _usable(p, now=now)][:HISTORY_KEEP]


def _warn_if_corrupt(blobs, digest: str) -> None:
    """Say so when a blob was there but did not hash to its name.

    provender suspects such a blob and answers False, which otherwise looks exactly like a
    cold cache - and the two could not be more different. The message was in the blob code
    before it moved out and was lost at the seam (review, 2026-09-20). Not throttled:
    unlike a store being down, this should not happen.
    """
    if digest in getattr(blobs, "suspect", ()):
        print(f"warning: blob {digest} does not hash to its name; serving as a cache miss "
              "and replacing it on the next publication", file=sys.stderr, flush=True)


def _place(where: Path, name: str, src: Path) -> None:
    """Put ``src`` into ``where`` as ``name``, atomically: a reader sees the old file or
    the new one, never a partial copy."""
    import os
    import shutil
    tmp = where / f".{name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, where / name)
    except OSError:
        Path(tmp).unlink(missing_ok=True)


def _present(p: Path):
    return p if p.exists() else None
