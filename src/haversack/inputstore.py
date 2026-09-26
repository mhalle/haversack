"""The input cache on the content-addressed store (step 6 of docs/cache-consolidation.md).

Every byte under the digest of its own content, one small ref per input, and nothing else
mutable. It replaces what ``serve.SeriesCache`` + ``content.ContentStore`` do today - with
their ``.owner`` claims, heartbeats, handovers, pins and graveyard - by three observations:

- **A stored input never changes.** Its key names the fetch epoch and the reader version
  (``e<F>.r<R>!idc:<uuid>``; an upload ``r<R>!sha256:<hex>``), so a newer build looks up
  another key rather than rewriting this one. The ref is written create-if-absent, never
  replaced: there is nothing to compare-and-swap.
- **A job reads its own VIEW** (:meth:`InputStore.materialize`), built in the job's
  directory from the blobs - hard links where the filesystem has them, copies where it does
  not (exFAT, FAT). The view has the entry layout every reader already knows
  (``decoded/input.duckn.zip``, ``series/...``, ``.input.json`` beside them), so no reader
  changes. And since the job owns its view, eviction never needs to know who is reading:
  that is what pins, claims and the graveyard existed for.
- **One fetch per key on a host is an economy, not a guarantee.** A per-key ``flock`` stops
  two jobs on this host downloading one series twice; if it failed they would write the same
  blobs and the second ref write would be a no-op. Correctness never depends on it - which is
  why the claim protocol's four review rounds are not ported.

Eviction: refs least recently used first, down to a byte budget; then the blobs no remaining
ref names, older than a grace (a blob is written before its ref, so a young unreferenced
blob may belong to a store in progress). A job holding a view is untouched by either: a
hard link keeps its inode, a copy is its own. A fetched input whose ref is gone is fetched
again; an upload whose ref is gone answers ``input_gone``, as it always has.

Local only for now: the shared (remote) half, and which inputs it may carry
(``--share-inputs``), come next; see the design record.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

#: The ref document's format; a ref of another format reads as absent (and is refetched or
#: answers input_gone), never guessed at.
FORMAT = 1
#: Where refs live in the store; blobs are provender's ``blobs/sha256/``.
REF_DIR = "inputs/"
#: How long an unreferenced blob is kept: a store in progress writes its blobs before its ref.
GRACE_S = 3600.0
#: Characters a ref's file name keeps as they are. Everything else is escaped with LOWER-case
#: hex, and upper-case letters are escaped too, so two keys differing only in case - which a
#: case-insensitive filesystem (APFS, exFAT) would fold into one file - never share a ref.
_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_.")
#: How long a fetch waits for another fetch of the same key on this host before going ahead
#: anyway: the lock saves a download, and a lock that cannot be had must not stop the work.
ECONOMY_WAIT_S = 600.0
#: A ref name longer than this is hashed; the key itself is always inside the document.
_MAX_NAME = 180


class InputGone(FileNotFoundError):
    """The input's ref or one of its blobs is not in the store (evicted, or never stored)."""


def key_for(identity: str) -> str:
    """The store key of an input: what ``SeriesCache._entry`` named, and why. A fetched
    input carries the fetch epoch AND the reader version, so a build that fetches or reads
    differently never meets this one's entry; an upload (a digest) carries the reader version
    only - its bytes cannot change, but the copy made of them can."""
    from .content import is_digest
    from .input_copy import READER_VERSION
    from .sources import FETCH_EPOCH
    if is_digest(identity):
        return f"r{READER_VERSION}!{identity}"
    return f"e{FETCH_EPOCH}.r{READER_VERSION}!{identity}"


def ref_name(key: str) -> str:
    name = "".join(c if c in _KEEP else "".join(f"%{b:02x}" for b in c.encode("utf-8"))
                   for c in key)
    if len(name) > _MAX_NAME:
        name = "h_" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{REF_DIR}{name}.json"


def _relpath(name) -> str:
    """A path from a ref, as data written by another process: relative, no ``..``, no
    leading dot component except the one sidecar the view writes itself."""
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        raise ValueError(f"not a stored path: {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"not a stored path: {name!r}")
    return name


def reap_dead(base: Path) -> None:
    """Remove ``base/<pid>/`` for every pid that is no longer running: what a process killed
    mid-fetch (its staging) or mid-job (its views) left, which nothing else would ever take."""
    for d in base.iterdir() if base.is_dir() else ():
        try:
            pid = int(d.name)
            if pid != os.getpid():
                os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
        except (ValueError, PermissionError, OSError):
            continue


class InputStore:
    """Inputs as blobs + one ref each, under ``root`` (``root/store`` is the object store,
    ``root/staging`` where a fetch is assembled, ``root/locks`` the per-key economy locks)."""

    def __init__(self, root, fetch_fn=None, *, budget_bytes: int = 8 << 30,
                 grace_s: float = GRACE_S, transcode: bool = True):
        from provender import Blobs
        from provender.disk import DiskStore
        self.root = Path(root)
        self.store = DiskStore(self.root / "store")
        self.blobs = Blobs(self.store)
        self.staging_root = self.root / "staging"
        self.locks = self.root / "locks"
        self.fetch = fetch_fn
        self.budget = int(budget_bytes)
        self.grace_s = float(grace_s)
        #: whether an input is kept as its input copy (the server) or as fetched (the command
        #: line, whose `get` hands the user the original files - a copy no viewer opens)
        self.transcode = bool(transcode)
        import threading
        self._held = threading.local()          # stripes this thread holds (re-entrancy)
        reap_dead(self.staging_root)            # a fetch killed mid-way leaves its staging

    # -- refs ------------------------------------------------------------------------------

    def _ref_file(self, key: str) -> Path:
        return self.store.root / ref_name(key)

    def ref(self, identity: str) -> dict | None:
        """The stored ref for ``identity``, validated, or None. A ref that does not parse, is
        of another format, names another key, or names a path that could leave the view is
        treated as absent - it is data another process wrote."""
        from provender import ops
        key = key_for(identity)
        try:
            doc = json.loads(ops.get(self.store, ref_name(key)).bytes())
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            return None
        try:
            if doc.get("format") != FORMAT or doc.get("key") != key:
                return None
            files = doc["files"]
            if not isinstance(files, dict) or not files:
                return None
            for name, blob in files.items():
                _relpath(name)
                self.blobs.path(blob["digest"])          # validates the digest
                int(blob["size"])
            _relpath(doc["read"])
        except (KeyError, TypeError, ValueError):
            return None
        return doc

    def has(self, identity: str) -> bool:
        """Stored, with every blob present. A ref whose blob was swept is not an input."""
        doc = self.ref(identity)
        return doc is not None and all(
            (self.store.root / self.blobs.path(b["digest"])).is_file()
            for b in doc["files"].values())

    def record(self, identity: str) -> dict | None:
        """The source record (what ``.input.json`` held): origin, license, digest, UIDs."""
        doc = self.ref(identity)
        return None if doc is None else doc.get("record")

    def touch(self, identity: str) -> None:
        """Mark it used, for the LRU. The ref's mtime is the clock: a ref is never rewritten,
        so nothing else moves it."""
        with contextlib.suppress(OSError):
            os.utime(self._ref_file(key_for(identity)))

    def forget(self, identity: str) -> bool:
        """Forget ``identity``: the next use fetches it again. Its blobs are left to the sweep -
        they may be shared, and a job's view may be linked to them. Not called ``discard``:
        DECIDING to discard a cached input is jobpolicy's alone (a test holds every
        ``.discard(`` call to it), and this is the mechanism, which the store also uses for a
        ref whose blobs were swept."""
        from provender import ops
        existed = self.ref(identity) is not None
        ops.delete(self.store, ref_name(key_for(identity)))
        return existed

    # -- storing ---------------------------------------------------------------------------

    @contextlib.contextmanager
    def _economy_lock(self, key: str, check=None):
        """One fetch of ``key`` at a time on this host - to save a download, never for
        correctness. Released by the kernel when the holder exits, however it exits.

        Because nothing depends on it, it gives way rather than wait without end (2026-09-26):
        - RE-ENTRANT in a thread: a fetch that stores another input while it runs (a source
          whose fetch materializes something) takes the same stripe again - and two opens of
          one lock file in one process exclude each other, so a nested call waited forever
          (`test_an_unlocked_publish_CANNOT_REMOVE_a_completed_winner`, which nests exactly
          that, hung). A different key on the same stripe nests the same way.
        - BOUNDED: past :data:`ECONOMY_WAIT_S` a waiter goes ahead unlocked; the worst it can
          cost is the download it would have saved."""
        import fcntl
        import threading
        stripe = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") % 256
        held = self._held.__dict__.setdefault("stripes", {})
        if held.get(stripe):
            held[stripe] += 1
            try:
                yield
            finally:
                held[stripe] -= 1
            return
        self.locks.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.locks / f"{stripe:03d}", os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + ECONOMY_WAIT_S
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if check is not None:
                        check()
                    if time.monotonic() > deadline:
                        break                      # go ahead unlocked: a duplicate fetch at worst
                    time.sleep(0.1)
            held[stripe] = 1
            try:
                yield
            finally:
                held.pop(stripe, None)
        finally:
            os.close(fd)

    def ensure(self, identity: str, *, fetch=None, credentials=None, check=None) -> dict:
        """The ref for ``identity``, fetching and storing it first when absent.

        ``fetch(identity, entry[, credentials=])`` is the sources' contract: build
        ``<entry>/series`` (and ``<entry>/.input.json``) and return the content path. The
        entry is a private staging directory; what is kept is its input copy - or, when the
        reader refuses the input, its original files - as blobs, and one ref naming them."""
        key = key_for(identity)
        with self._economy_lock(key, check):
            if self.has(identity):
                self.touch(identity)
                return self.ref(identity)
            self.forget(identity)                  # a ref whose blobs were swept
            stage = self.staging_root / str(os.getpid()) / uuid.uuid4().hex
            stage.mkdir(parents=True)
            try:
                fn = fetch or self.fetch
                content = Path(fn(identity, stage, credentials=credentials)
                               if credentials is not None else fn(identity, stage))
                doc = self._store_staged(identity, key, stage, content)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        self.evict(keep={key})
        return doc

    def _store_staged(self, identity: str, key: str, stage: Path, content: Path) -> dict:
        from provender import ops
        from obstore.exceptions import AlreadyExistsError

        from .content import is_digest
        from .input_copy import transcode
        from .sources import ResultSource, read_input_record
        record = read_input_record(stage)
        read = content
        if self.transcode and not str(identity).startswith(ResultSource.prefix + ":"):
            # a result: reference is a label map, never transcoded (as SeriesCache)
            digest = identity if is_digest(identity) else ((record or {}).get("content") or {}).get("digest")
            copy = transcode(content, stage, source=None if is_digest(identity) else identity,
                             source_digest=digest)
            if copy is not None:
                shutil.rmtree(stage / "series", ignore_errors=True)
                read = copy
        stored = sorted(p for p in stage.rglob("*")
                        if p.is_file() and not p.name.startswith("."))
        if not stored:
            raise FileNotFoundError(f"{identity}: the fetch stored no files")
        files = {}
        for p in stored:
            files[p.relative_to(stage).as_posix()] = self.blobs.put_file(p)
        doc = {"format": FORMAT, "key": key, "identity": identity,
               "files": files, "read": read.relative_to(stage).as_posix(),
               "record": record, "stored": time.time(),
               "bytes": sum(b["size"] for b in files.values())}
        try:
            ops.put(self.store, ref_name(key), json.dumps(doc, sort_keys=True).encode(),
                    mode="create")
        except AlreadyExistsError:
            # another process stored the same key meanwhile (its lock was not ours - another
            # host, or a lock that failed): the same bytes by construction, so its ref stands
            got = self.ref(identity)
            if got is not None:
                return got
            ops.put(self.store, ref_name(key), json.dumps(doc, sort_keys=True).encode())
        return doc

    # -- uploads (ContentStore's half) ----------------------------------------------------

    def put_file(self, staged, *, expect: str | None = None, computed: str | None = None) -> str:
        """Store an uploaded file under the digest of its bytes (``computed`` when the caller
        hashed it while streaming; ``expect`` is checked, never trusted)."""
        from .content import DigestMismatch, _stored_name, digest_file
        digest = computed or digest_file(staged)
        if expect and expect != digest:
            raise DigestMismatch(expect, digest)
        self._adopt(digest, [(Path(staged), _stored_name(staged))])
        return digest

    def put_dir(self, staged, *, expect: str | None = None) -> str:
        """A directory of files (a DICOM series) as one tree - or as a blob, when it holds
        exactly one file (one file is a file however it arrived, as ContentStore says)."""
        from .content import DigestMismatch, digest_file, tree_digest
        members = sorted(p for p in Path(staged).rglob("*") if p.is_file())
        if not members:
            raise FileNotFoundError(f"{staged} holds no files")
        if len(members) == 1:
            return self.put_file(members[0], expect=expect)
        per_member = [(p, digest_file(p)) for p in members]
        digest = tree_digest(d for _, d in per_member)
        if expect and expect != digest:
            raise DigestMismatch(expect, digest)
        self._adopt(digest, [(p, d.split(":", 1)[1][:32]) for p, d in per_member])
        return digest

    def _adopt(self, digest: str, members) -> None:
        def write(_key, entry):
            content = Path(entry) / "series"
            content.mkdir(parents=True, exist_ok=True)
            for src, name in members:
                shutil.copyfile(src, content / Path(name).name)
            return content
        self.ensure(digest, fetch=write)

    # -- reading ---------------------------------------------------------------------------

    def materialize(self, identity: str, dest) -> Path:
        """Build ``identity``'s view in ``dest`` (a directory the CALLER owns - a job's) and
        return what to hand the reader: the input copy, the one uploaded file, or the DICOM
        folder, exactly as the old entry held them, with ``.input.json`` beside them.

        Hard links where the filesystem has them - the view then costs nothing and survives
        the blob's eviction - else copies. Raises :class:`InputGone` when the ref or a blob is
        not here; the caller fetches again (a hosted input) or answers input_gone (an upload)."""
        from .content import TREE, is_digest
        doc = self.ref(identity)
        if doc is None:
            raise InputGone(f"{identity} is not held by this store")
        dest = Path(dest)
        for name, blob in doc["files"].items():
            target = dest / _relpath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = self.store.root / self.blobs.path(blob["digest"])
            try:
                if target.exists():
                    target.unlink()
                try:
                    os.link(src, target)
                except OSError as e:
                    if isinstance(e, FileNotFoundError):
                        raise
                    shutil.copyfile(src, target)     # no hard links here (exFAT, FAT)
            except FileNotFoundError:
                raise InputGone(f"{identity}: blob {blob['digest'][:19]} is gone") from None
        if doc.get("record") is not None:
            (dest / ".input.json").write_text(json.dumps(doc["record"], indent=1),
                                              encoding="utf-8")
        self.touch(identity)
        read = dest / doc["read"]
        if read.is_dir() and is_digest(identity) and not str(identity).startswith(TREE):
            # an upload of one file: the FILE, as ContentStore.resolve hands it (a directory
            # would make SimpleITK read it as a DICOM series)
            files = [p for p in read.iterdir() if p.is_file()]
            if len(files) == 1:
                return files[0]
        return read

    def get_or_fetch(self, identity: str, dest, *, fetch=None, credentials=None,
                     check=None) -> Path:
        """Store ``identity`` if absent, then build its view in ``dest``. A blob swept between
        the two is fetched again, once."""
        for attempt in range(2):
            self.ensure(identity, fetch=fetch, credentials=credentials, check=check)
            try:
                return self.materialize(identity, dest)
            except InputGone:
                if attempt:
                    raise
                self.forget(identity)
        raise AssertionError("unreachable")

    # -- eviction --------------------------------------------------------------------------

    def _refs(self) -> list[tuple[float, str, dict]]:
        """Every readable ref as ``(last used, key, document)``, least recent first."""
        out = []
        base = self.store.root / REF_DIR
        for f in base.glob("*.json") if base.is_dir() else ():
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
                out.append((f.stat().st_mtime, doc["key"], doc))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        out.sort(key=lambda t: t[0])
        return out

    def usage(self) -> int:
        return sum(int(d.get("bytes") or 0) for _, _, d in self._refs())

    def evict(self, keep=frozenset()) -> dict:
        """Down to the byte budget, least recently used first, never a key in ``keep``; then
        the blobs no ref names and older than the grace. Refs that do not parse are left
        alone and their blobs kept: cleanup refuses what it cannot account for."""
        from provender import ops
        listed = self.blobs.entries(older_than=time.time() - self.grace_s)   # BEFORE the refs
        refs = self._refs()
        total, dropped = sum(int(d.get("bytes") or 0) for _, _, d in refs), 0
        live = []
        for used, key, doc in refs:
            if total > self.budget and key not in keep:
                ops.delete(self.store, ref_name(key))
                total -= int(doc.get("bytes") or 0)
                dropped += 1
            else:
                live.append(doc)
        base = self.store.root / REF_DIR
        unreadable = base.is_dir() and len(list(base.glob("*.json"))) != len(refs) - dropped
        keep_blobs = {b["digest"] for d in live for b in d.get("files", {}).values()}
        swept = {"deleted": 0}
        if not unreadable and (keep_blobs or not live):
            swept = self.blobs.sweep(keep=keep_blobs, grace_s=self.grace_s, candidates=listed,
                                     allow_empty=True)
        return {"refs_dropped": dropped, "blobs": swept}


class ServerInputs:
    """``SeriesCache`` + ``ContentStore``'s interface over an :class:`InputStore`, so the
    local server's call sites run unchanged behind ``HAVERSACK_INPUT_STORE=blobs``.

    Those call sites already follow the lifecycle a view needs - pin, look, use, unpin - so a
    PIN holds a view of the input in this process (``<root>/views/<pid>/``), ``path`` and
    ``get_or_fetch`` answer inside it, and the last unpin deletes it. The store's eviction
    never touches a view: it is hard links or copies. A view built without a pin (a status
    route's ``resolve``) is reaped once it is older than :attr:`LOOSE_VIEW_S`."""

    LOOSE_VIEW_S = 120.0

    def __init__(self, root, fetch_fn, *, budget_bytes: int = 8 << 30, grace_s: float = GRACE_S,
                 legacy_root=None):
        self.store = InputStore(root, fetch_fn, budget_bytes=budget_bytes, grace_s=grace_s)
        #: the legacy SeriesCache root whose UPLOADS are adopted on first use (the migration
        #: shim; see :meth:`_adopt_legacy_upload`)
        self.legacy_root = Path(legacy_root) if legacy_root else None
        self.root = self.store.root
        self._views_root = self.root / "views" / str(os.getpid())
        shutil.rmtree(self._views_root, ignore_errors=True)   # a previous process's, same pid
        self._guard = __import__("threading").RLock()
        self._pins: dict[str, int] = {}
        self._views: dict[str, tuple[Path, float]] = {}
        self._fetching: set[str] = set()
        self._reap_dead_processes()

    def _reap_dead_processes(self) -> None:
        reap_dead(self.root / "views")

    # -- views -------------------------------------------------------------------------------

    def _view(self, key: str, *, fetch=None, credentials=None, check=None) -> Path:
        with self._guard:
            held = self._views.get(key)
            if held is not None and held[0].exists():
                return held[0]
        dest = self._views_root / uuid.uuid4().hex
        dest.mkdir(parents=True)
        try:
            if fetch is False:
                path = self.store.materialize(key, dest)
            else:
                path = self.store.get_or_fetch(key, dest, fetch=fetch,
                                               credentials=credentials, check=check)
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        with self._guard:
            self._views[key] = (path, time.time())
            self._reap_loose()
        return path

    def _drop_view(self, key: str) -> None:
        held = self._views.pop(key, None)
        if held is not None:
            view = held[0]
            while view.parent != self._views_root and view.parent != view:
                view = view.parent
            shutil.rmtree(view, ignore_errors=True)

    def _reap_loose(self) -> None:
        now = time.time()
        for key, (path, made) in list(self._views.items()):
            if not self._pins.get(key) and now - made > self.LOOSE_VIEW_S:
                self._drop_view(key)

    # -- SeriesCache's interface -----------------------------------------------------------------

    def pin(self, key: str) -> None:
        with self._guard:
            self._pins[key] = self._pins.get(key, 0) + 1

    def unpin(self, key: str) -> None:
        with self._guard:
            n = self._pins.get(key, 0) - 1
            if n > 0:
                self._pins[key] = n
                return
            self._pins.pop(key, None)
            self._drop_view(key)

    def has(self, key: str) -> bool:
        return self.store.has(key) or self._adopt_legacy_upload(key)

    def _adopt_legacy_upload(self, key: str) -> bool:
        """MIGRATION SHIM (decision 1 of docs/cache-consolidation.md: time-limited - removed two
        minor releases after the release that makes this store the default, or 90 days,
        whichever is later; the CHANGELOG says when). An UPLOAD the legacy SeriesCache holds
        cannot be fetched again, so on first use it is stored here from the legacy entry - its
        input copy as it is (never re-encoded), or its original files - and the legacy entry
        is left for the legacy cache's own eviction. A FETCHED input is not adopted: it is
        public and simply fetched again under the new key. Only a committed, current entry
        is adopted: the legacy cache's own `has` decides (`.done`, and a copy of this reader
        version), so an entry mid-write or stale is not."""
        from .content import is_digest
        if self.legacy_root is None or not is_digest(key) or not self.legacy_root.is_dir():
            return False
        try:
            if getattr(self, "_legacy", None) is None:
                from .serve import SeriesCache
                self._legacy = SeriesCache(self.legacy_root, None)
            legacy = self._legacy
            if not legacy.has(key):
                return False
            entry = legacy.entry(key)
            content = legacy.path(key)
        except Exception:                          # noqa: BLE001 - an unreadable legacy entry
            return False                           # is simply not adopted: input_gone, as before

        def carry(_identity, stage):
            stage = Path(stage)
            rel = content.relative_to(entry)
            if content.is_dir():
                shutil.copytree(content, stage / rel)
            else:
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(content, stage / rel)
            side = entry / ".input.json"
            if side.is_file():
                shutil.copyfile(side, stage / ".input.json")
            return stage / rel
        try:
            self.store.ensure(key, fetch=carry)
        except Exception:                          # noqa: BLE001 - as unreadable, above
            return False
        return self.store.has(key)

    def path(self, key: str) -> Path:
        return self._view(key, fetch=False)

    def entry(self, key: str) -> Path:
        """Where ``.input.json`` is: the view's root (``input_records`` reads it there)."""
        view = self._view(key, fetch=False)
        while view.parent != self._views_root and view.parent != view:
            view = view.parent
        return view

    def staging(self, key: str) -> bool:
        with self._guard:
            return key in self._fetching

    def get_or_fetch(self, key: str, *, check=None, credentials=None, fetch=None) -> Path:
        with self._guard:
            self._fetching.add(key)
        try:
            return self._view(key, fetch=fetch, credentials=credentials, check=check)
        finally:
            with self._guard:
                self._fetching -= {key}

    def prefetch(self, key: str) -> bool:
        """Store without blocking and without a view: False when it is here already, is
        being fetched here, or the fetch failed."""
        if self.store.has(key) or self.staging(key):
            return False
        with self._guard:
            self._fetching.add(key)
        try:
            self.store.ensure(key)
            return True
        except Exception:                      # noqa: BLE001 - a prefetch never fails a job
            return False
        finally:
            with self._guard:
                self._fetching -= {key}

    def discard(self, key: str) -> bool:
        """Forget ``key`` so the next use fetches it again - refused while this process holds
        it pinned, as SeriesCache refuses (jobpolicy says "input in use")."""
        with self._guard:
            if self._pins.get(key):
                return False
        return self.store.forget(key)

    # -- ContentStore's interface --------------------------------------------------------------

    def resolve(self, digest: str) -> Path:
        if not self.has(digest):
            raise FileNotFoundError(f"{digest} is not held by this store")
        return self._view(digest, fetch=False)

    def fast_path(self, digest: str) -> Path:
        return self.resolve(digest)

    def put_file(self, staged, *, expect=None, computed=None) -> str:
        return self.store.put_file(staged, expect=expect, computed=computed)

    def put_dir(self, staged, *, expect=None) -> str:
        return self.store.put_dir(staged, expect=expect)


def input_store_enabled() -> bool:
    """``HAVERSACK_INPUT_STORE=blobs``: inputs on this store (step 6), until it is the default."""
    return os.environ.get("HAVERSACK_INPUT_STORE", "").strip().lower() == "blobs"


_COMMAND: dict = {}


class CommandInputs:
    """The command line's inputs: an :class:`InputStore` whose views are persistent EXPORTS,
    one per input (``<root>/exports/<ref name>/``), because the command line hands out paths
    for later - `haversack get` prints one for the user to open, a batch materializes before
    it segments - which a per-process view would take away at exit. An export is built from
    the blobs once (hard links, else copies), in a temporary directory renamed into place, so
    two processes never half-build one; it is reused by every later command, and `haversack
    cache clean inputs` removes it with its ref. No byte budget: the command line's cache
    never had one."""

    def __init__(self, root):
        self.store = InputStore(root, None, budget_bytes=1 << 62, transcode=False)
        self.root = self.store.root
        self.exports = self.root / "exports"

    def export_dir(self, identity: str) -> Path:
        return self.exports / Path(ref_name(key_for(identity))).stem

    def has(self, identity: str) -> bool:
        return self.store.has(identity)

    def record(self, identity: str) -> dict | None:
        return self.store.record(identity)

    def get_or_fetch(self, identity: str, *, fetch) -> Path:
        """The input's export, storing it first if absent; what to hand the reader."""
        doc = self.store.ensure(identity, fetch=fetch)
        dest = self.export_dir(identity)
        if not dest.is_dir() or self.store.ref(identity) is None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex[:8]}"
            try:
                self.store.materialize(identity, tmp)
                shutil.rmtree(dest, ignore_errors=True)   # a stale export of a forgotten ref
                try:
                    os.rename(tmp, dest)
                except OSError:                    # another process placed it meanwhile
                    pass
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        self.store.touch(identity)
        read = dest / doc["read"]
        if read.is_dir():
            from .content import TREE, is_digest
            if is_digest(identity) and not str(identity).startswith(TREE):
                files = [p for p in read.iterdir() if p.is_file()]
                if len(files) == 1:
                    return files[0]
        return read

    def forget(self, identity: str) -> None:
        self.store.forget(identity)
        shutil.rmtree(self.export_dir(identity), ignore_errors=True)


def command_inputs(root) -> CommandInputs:
    """The command line's inputs under ``root``, one per root per process."""
    key = str(Path(root).resolve())
    got = _COMMAND.get(key)
    if got is None:
        got = _COMMAND[key] = CommandInputs(root)
    return got
