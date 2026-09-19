"""The result cache on an object store (S3, GCS, ...), shared by every host that names it.

Why this is not ``ResultCache`` pointed at a mounted bucket (2026-09-19). ``ResultCache``
is a POSIX protocol: one rename publishes, ``flock`` proves a writer dead, a lease file's
mtime says a reader holds a generation. An object store offers none of those, and a
filesystem that emulates them over one (ZeroFS, assessed the same day) funnels every host
through a single writer process whose locks vanish on restart - exactly when "I could take
its lock" stops proving death. What an object store DOES offer is enough for a different
protocol, the one build caches use (Bazel's ActionCache over a CAS):

- **Blobs** under ``blobs/sha256/<hex>``, named by their own bytes and written only if
  absent. Immutable, so a reader never holds anything and nothing needs a lease.
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

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from .errors import InputError

#: Bump if the pointer document changes shape incompatibly; a reader refuses (reads as a
#: miss) any pointer whose format it does not know, rather than guessing at its fields.
POINTER_FORMAT = 1
#: How old an unreferenced blob must be before ``sweep`` may delete it. Generous: a blob is
#: unreferenced for the few seconds between its upload and its pointer's write, and what a
#: too-short grace costs is a republication, while a long one costs only storage.
BLOB_GRACE_S = 24 * 3600
#: Conditional-write attempts before a publication gives up. Each retry means another
#: writer published this key in between, so this many in a row is a storm, not a race.
SWAP_ATTEMPTS = 16
_CHUNK = 1 << 20


class ObjectStoreUnsuitable(InputError):
    """The store cannot carry the protocol - it does not honor conditional writes."""


def open_store(url: str):
    """``(store, prefix)`` for ``s3://bucket/prefix``, ``gs://...``, ``az://...``,
    ``file:///path`` or ``memory://``. Credentials come from the environment, as obstore
    reads them (``AWS_*``, ``GOOGLE_*``, ``AZURE_*``); nothing here takes a secret.

    ``memory://`` is process-local and exists for tests. ``file://`` opens, and is then
    refused by ``check_conditional_writes``: obstore's local store cannot replace an object
    conditionally (measured 2026-09-19, obstore 0.11.1) - the plain ``ResultCache`` is the
    right cache for a local disk anyway.
    """
    from urllib.parse import urlparse
    u = urlparse(url)
    scheme = u.scheme.lower()
    if scheme == "memory":
        from obstore.store import MemoryStore
        return MemoryStore(), _prefix(u.netloc + u.path)
    if scheme == "file":
        from obstore.store import LocalStore
        root = Path(u.path)
        root.mkdir(parents=True, exist_ok=True)
        return LocalStore(root), ""
    if scheme in ("s3", "s3a", "gs", "az", "abfs", "abfss") and u.netloc:
        from obstore.store import from_url
        return from_url(f"{scheme}://{u.netloc}"), _prefix(u.path)
    raise InputError(f"result store {url!r}: expected s3://bucket[/prefix], gs://..., "
                     "az://..., file:///path or memory://")


def _prefix(path: str) -> str:
    p = path.strip("/")
    return f"{p}/" if p else ""


def check_conditional_writes(store, prefix: str = "") -> None:
    """Refuse a store on which the pointer protocol would silently lose publications.

    Four questions, each answered by the store rather than assumed: a create-if-absent over
    an existing object must fail; a replace with the current etag must succeed; a replace
    with a STALE etag must fail. A store that answers any of them wrong - or that does not
    implement the question - would let two writers both believe they published.
    """
    import obstore
    from obstore.exceptions import AlreadyExistsError, PreconditionError
    path = f"{prefix}.probe/{uuid.uuid4().hex}"
    name = type(store).__name__

    def refuse(why: str) -> ObjectStoreUnsuitable:
        return ObjectStoreUnsuitable(
            f"result store ({name}) {why}: the shared result cache needs create-if-absent "
            "and replace-if-unchanged writes; use S3, GCS or Azure, or drop --result-store "
            "for a local-only cache")
    try:
        try:
            obstore.put(store, path, b"0", mode="create")
        except (NotImplementedError, TypeError) as e:
            raise refuse(f"cannot create-if-absent ({e})") from None
        try:
            obstore.put(store, path, b"1", mode="create")
        except AlreadyExistsError:
            pass
        except (NotImplementedError, TypeError) as e:
            raise refuse(f"cannot create-if-absent ({e})") from None
        else:
            raise refuse("overwrote an existing object on create-if-absent")
        stale = _update_mode(obstore.head(store, path))
        try:
            obstore.put(store, path, b"2", mode=stale)
        except (NotImplementedError, TypeError) as e:
            raise refuse(f"cannot replace-if-unchanged ({e})") from None
        except PreconditionError:
            raise refuse("refused a replace carrying the current etag") from None
        try:
            obstore.put(store, path, b"3", mode=stale)
        except PreconditionError:
            pass
        else:
            raise refuse("accepted a replace carrying a stale etag")
    finally:
        try:
            obstore.delete(store, path)
        except Exception:                      # noqa: BLE001 - a probe left behind is litter
            pass


def _update_mode(meta) -> dict:
    """The replace-if-unchanged mode for an object whose metadata is ``meta``. ``version``
    only when the store reports one: passing None is a TypeError in obstore."""
    mode = {"e_tag": meta["e_tag"]}
    if meta.get("version") is not None:
        mode["version"] = meta["version"]
    return mode


class BlobStore:
    """Bytes named by their SHA-256, under ``<prefix>blobs/sha256/<hex>``."""

    def __init__(self, store, prefix: str = ""):
        self.store = store
        self.prefix = prefix

    def path(self, digest: str) -> str:
        algo, _, hexd = digest.partition(":")
        if algo != "sha256" or len(hexd) != 64 or not all(c in "0123456789abcdef" for c in hexd):
            raise ValueError(f"not a sha256 digest: {digest!r}")
        return f"{self.prefix}blobs/sha256/{hexd}"

    def put_file(self, src) -> dict:
        """Upload ``src`` unless its bytes are already stored; ``{"digest", "size"}``.

        Create-if-absent, so a concurrent upload of the same bytes is not a conflict - it
        is the same object by definition, and whichever landed first is kept.
        """
        import obstore
        from obstore.exceptions import AlreadyExistsError
        src = Path(src)
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        blob = {"digest": f"sha256:{h.hexdigest()}", "size": src.stat().st_size}
        if not self.has(blob["digest"]):
            try:
                obstore.put(self.store, self.path(blob["digest"]), src, mode="create")
            except AlreadyExistsError:
                pass
        return blob

    def has(self, digest: str) -> bool:
        import obstore
        try:
            obstore.head(self.store, self.path(digest))
        except FileNotFoundError:
            return False
        return True

    def fetch(self, digest: str, dest) -> bool:
        """Write the blob to ``dest``, verified against its name; False when it is gone -
        or was wrong, in which case it is deleted.

        Verified because the name is a promise the store does not check: a truncated or
        corrupted object would otherwise be published into the local cache as a result.
        Deleted because ``put_file`` skips an upload whose name already exists, so a bad
        blob left in place would survive every recompute that should replace it; bytes
        that do not hash to their name are provably nobody's.
        """
        import obstore
        dest = Path(dest)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
        h = hashlib.sha256()
        try:
            with open(tmp, "wb") as f:
                for chunk in obstore.get(self.store, self.path(digest)).stream(
                        min_chunk_size=_CHUNK):
                    h.update(chunk)
                    f.write(chunk)
            if f"sha256:{h.hexdigest()}" != digest:
                try:
                    obstore.delete(self.store, self.path(digest))
                except FileNotFoundError:
                    pass
                return False
            tmp.replace(dest)
            return True
        except FileNotFoundError:
            return False
        finally:
            tmp.unlink(missing_ok=True)


class SharedResultCache:
    """``ResultCache``'s interface, with the object store as the authority and a local
    ``ResultCache`` as the copy requests are served from. See the module docstring."""

    def __init__(self, store, local, *, prefix: str = "", check: bool = True):
        if check:
            check_conditional_writes(store, prefix)
        self.store = store
        self.prefix = prefix
        self.local = local
        self.blobs = BlobStore(store, prefix)
        self._fill_locks: dict[str, threading.Lock] = {}
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

    def _read_pointer(self, key: str):
        """``(pointer, update mode)`` or ``(None, None)``. A pointer this code cannot read
        - unknown format, not JSON - is reported as absent: a miss, never a guess."""
        import obstore
        try:
            got = obstore.get(self.store, self._pointer_path(key))
        except FileNotFoundError:
            return None, None
        mode = _update_mode(got.meta)
        try:
            ptr = json.loads(bytes(got.bytes()))
        except (ValueError, UnicodeDecodeError):
            return None, mode
        if not isinstance(ptr, dict) or ptr.get("format") != POINTER_FORMAT:
            return None, mode
        return ptr, mode

    def _swap(self, key: str, update):
        """Replace the pointer with ``update(current)`` by conditional write, rereading on
        every lost race; ``update`` returning None abandons the swap. Returns what was
        written, or None."""
        import obstore
        from obstore.exceptions import AlreadyExistsError, PreconditionError
        for _ in range(SWAP_ATTEMPTS):
            ptr, mode = self._read_pointer(key)
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
            wanted = [n for n in files if n in ARTIFACT_NAMES or n == RESULT_NAME]
            missing = [n for n in wanted if not (have_gen and (local_dir / n).exists())]
            if not missing:
                return True
            work = Path(tempfile.mkdtemp(prefix=".fill-", dir=self.local.root))
            try:
                for name in missing:
                    if not self.blobs.fetch(files[name]["digest"], work / name):
                        return False           # swept: a miss, and the next compute heals it
                if not have_gen:
                    try:
                        self.local.put(key, work / RESULT_NAME, ptr.get("result") or {},
                                       ptr.get("meta") or {},
                                       preview_path=_present(work / "preview.png"),
                                       statistics_path=_present(work / "statistics.json"),
                                       generation=gen)
                    except FileExistsError:
                        pass                   # another process filled it first
                else:
                    for name in missing:
                        self.local.add_artifact(key, name, work / name, generation=gen)
                return True
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _fill_lock(self, key: str) -> threading.Lock:
        with self._fill_guard:
            return self._fill_locks.setdefault(key, threading.Lock())

    def put(self, key: str, labels_path, result: dict, meta: dict,
            preview_path=None, statistics_path=None) -> str:
        """Publish: blobs first, then the pointer, then the local copy. Returns the
        generation token, which the local copy shares."""
        from .serve import RESULT_NAME
        sources = {RESULT_NAME: labels_path, "preview.png": preview_path,
                   "statistics.json": statistics_path}
        files = {name: self.blobs.put_file(src) for name, src in sources.items()
                 if src and Path(src).exists()}
        gen = uuid.uuid4().hex
        pointer = {"format": POINTER_FORMAT, "generation": gen, "published": time.time(),
                   "files": files, "result": result, "meta": meta}
        self._swap(key, lambda _current: pointer)
        # A blob that already existed was not uploaded - and a sweep may have taken it
        # between that check and the pointer. Put it back: the pointer now protects it.
        for name, blob in files.items():
            if not self.blobs.has(blob["digest"]):
                self.blobs.put_file(sources[name])
        self.local.put(key, labels_path, result, meta, preview_path=preview_path,
                       statistics_path=statistics_path, generation=gen)
        return gen

    def add_artifact(self, key: str, name: str, src_path, generation=None) -> bool:
        """Add an artifact to the publication it was rendered for; False when that
        publication is no longer current (or the entry is gone). The pointer's conditional
        write is what makes it safe: an artifact can never be recorded beside another
        publication's labels, whatever the ordering."""
        from .serve import ARTIFACT_NAMES
        if name not in ARTIFACT_NAMES:
            raise ValueError(f"not an artifact: {name!r}")
        blob = self.blobs.put_file(src_path)

        def update(ptr):
            if ptr is None or (generation and ptr["generation"] != generation):
                return None
            files = dict(ptr.get("files") or {})
            files[name] = blob
            return {**ptr, "files": files}
        written = self._swap(key, update)
        if written is None:
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
        """Every published entry, newest first, read from the pointers alone."""
        from .serve import RESULT_NAME, resource_links
        out = []
        for ptr in self._pointers():
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

    def _pointers(self):
        import obstore
        base = f"{self.prefix}results/"
        for batch in obstore.list(self.store, base):
            for obj in batch:
                name = obj["path"][len(base):]
                if "/" in name or not name.endswith(".json"):
                    continue
                key = name[:-len(".json")]
                ptr, _ = self._read_pointer(key)
                if ptr is not None:
                    yield {**ptr, "_key": key}

    def sweep(self, *, max_age_s: float | None = None, grace_s: float = BLOB_GRACE_S,
              now: float | None = None) -> dict:
        """Delete what no pointer needs: pointers published more than ``max_age_s`` ago
        (when given), then blobs no remaining pointer references that are older than
        ``grace_s``.

        Blobs are LISTED before pointers are read. A publication that lands after the
        listing uploaded its blobs after it too, so they are not candidates; one that lands
        before the pointers are read is seen referencing them. What slips between is a
        blob that already existed and was not re-uploaded - ``put`` re-checks its blobs
        after writing the pointer for exactly that, and a read that still finds one missing
        is a miss the next compute repairs.

        Expiring a pointer is an unconditional delete, so a republication that lands
        between reading the pointer and deleting it is expired with it: a miss.
        """
        import datetime as _dt

        import obstore
        now = time.time() if now is None else now
        blob_base = f"{self.prefix}blobs/sha256/"
        candidates = []
        for batch in obstore.list(self.store, blob_base):
            for obj in batch:
                modified = obj["last_modified"]
                if isinstance(modified, _dt.datetime):
                    modified = modified.timestamp()
                if modified < now - grace_s:
                    candidates.append(obj["path"])
        referenced, expired = set(), 0
        for ptr in self._pointers():
            if max_age_s is not None and (ptr.get("published") or 0) < now - max_age_s:
                try:
                    obstore.delete(self.store, self._pointer_path(ptr["_key"]))
                except FileNotFoundError:
                    pass
                expired += 1
                continue
            for blob in (ptr.get("files") or {}).values():
                referenced.add(self.blobs.path(blob["digest"]))
        deleted = 0
        for path in candidates:
            if path in referenced:
                continue
            try:
                obstore.delete(self.store, path)
            except FileNotFoundError:
                pass
            deleted += 1
        return {"expired_pointers": expired, "deleted_blobs": deleted}


def _present(p: Path):
    return p if p.exists() else None
