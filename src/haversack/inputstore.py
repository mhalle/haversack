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


class InputStore:
    """Inputs as blobs + one ref each, under ``root`` (``root/store`` is the object store,
    ``root/staging`` where a fetch is assembled, ``root/locks`` the per-key economy locks)."""

    def __init__(self, root, fetch_fn=None, *, budget_bytes: int = 8 << 30,
                 grace_s: float = GRACE_S):
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

    def discard(self, identity: str) -> bool:
        """Forget ``identity`` (a refresh: the next use fetches it again). Its blobs are left
        to the sweep - they may be shared, and a job's view may be linked to them."""
        from provender import ops
        existed = self.ref(identity) is not None
        ops.delete(self.store, ref_name(key_for(identity)))
        return existed

    # -- storing ---------------------------------------------------------------------------

    @contextlib.contextmanager
    def _economy_lock(self, key: str, check=None):
        """One fetch of ``key`` at a time on this host - to save a download, never for
        correctness. Released by the kernel when the holder exits, however it exits."""
        import fcntl
        self.locks.mkdir(parents=True, exist_ok=True)
        stripe = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") % 256
        fd = os.open(self.locks / f"{stripe:03d}", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if check is not None:
                        check()
                    time.sleep(0.1)
            yield
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
            self.discard(identity)                 # a ref whose blobs were swept
            stage = self.staging_root / uuid.uuid4().hex
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
        if not str(identity).startswith(ResultSource.prefix + ":"):
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
                self.discard(identity)
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
