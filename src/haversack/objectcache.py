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
- **One ref per key**, ``results/<key>.json``, naming an immutable MANIFEST (format 2,
  2026-09-23): the files by digest, the result and meta documents, the publication's token,
  and ``replaces`` - the manifest it superseded. Manifests are stored as blobs, so they are
  content-addressed like everything else; the ref also carries its manifest's exact bytes,
  checked against the digest, so reading the present is still one request. Publication is
  uploading the blobs and the manifest and then ONE conditional write of the ref
  (create-if-absent, or replace-if-unchanged against the etag read) - no staging, no rename,
  no writer claim. A lost race rereads and writes again: last writer wins, as the rename did.
  History is the ``replaces`` chain; a late artifact is an AMENDING manifest of the same
  publication (its token kept); a deletion is a TOMBSTONE manifest. Format 1 - one pointer
  with its history inline - is still read, and converted by the first write to its key.
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
stale write is refused itself, naming the backend. Modeling the store from its docs is how
this repo has been wrong before.
"""
from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import threading
import time
import uuid
import weakref
from pathlib import Path

from provender import Blobs, GRACE_S, check_store, ops, update_mode
from provender import StoreUnsuitable as _StoreUnsuitable
from provender import open_store as _open_store

from .errors import InputError

#: Bump if the pointer document changes shape incompatibly; a reader refuses (reads as a
#: miss) any pointer whose format it does not know, rather than guessing at its fields.
#: 2 (2026-09-23): a REF naming an immutable MANIFEST, history as the manifests' `replaces`
#: chain, deletion as a tombstone manifest (docs/cache-consolidation.md, "The design from
#: scratch", step 3).
POINTER_FORMAT = 2
#: The inline-history pointer every host wrote before format 2. Still READ - a time-limited
#: shim (decision 1 of the design) - and converted to a manifest chain, old generation
#: tokens kept, by the first write to its key. Never written.
LEGACY_FORMAT = 1
#: How many manifests a history walk reads before it stops, whatever it has found. Each
#: publication is one manifest plus one per late artifact (a preview, statistics), so this
#: is generous for HISTORY_KEEP publications and bounds what a damaged or malicious chain
#: can cost a reader.
CHAIN_STEPS_MAX = 64
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
#: Extra age a generation's BLOBS keep past the point history stops listing it. Hosts do
#: not share a clock, and one running fast would otherwise collect what every other host
#: still lists - the rollback data the history decision exists to provide (review,
#: 2026-09-20). Listing is cheap to get wrong; deleting is not.
HISTORY_GC_MARGIN_S = 24 * 3600
#: How often a store fault on a READ path may be reported. A store that is down faults on
#: every request, and a line per request is a denial of service on the operator's terminal.
WARN_INTERVAL_S = 60.0
#: How long a host may go on serving its own copy while the store is unreachable, measured
#: from the last time it saw that entry in the store. A host in an outage cannot know that
#: an entry was deleted elsewhere, and "deletion means gone" is this branch's own rule - so
#: the fallback that keeps a warm cache usable is bounded rather than open-ended (review,
#: 2026-09-20). Long enough to ride out a blip, short enough that a deletion is not
#: outlived by a cache nobody is watching.
OUTAGE_GRACE_S = 15 * 60
#: Touched in the generation directory whenever the store confirmed this copy is current.
CONFIRMED_NAME = ".confirmed"
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
            f"{e}; use S3, GCS, Azure, R2 or a directory (file:///path), or drop "
            "--result-store for a local-only cache") from None


def _warn_once(message: str) -> None:
    """Report a deviation at most once a minute, like ``_miss``."""
    global _warned_at
    now = time.time()
    if now - _warned_at >= WARN_INTERVAL_S:
        _warned_at = now
        print(f"warning: {message}", file=sys.stderr, flush=True)


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
    if not isinstance(ptr, dict) or ptr.get("format") != LEGACY_FORMAT:
        return False
    if not _token_ok(ptr.get("generation")):
        return False
    if not _well_formed_files(ptr.get("files")):
        return False
    # History is OPTIONAL and never load-bearing for the present. Requiring it to be a
    # list made `"history": null` - what another language emits for "none" - lose the
    # current result and, because the pointer then counted as unreadable, freeze blob
    # deletion for the whole store (review, 2026-09-20). A past this code cannot read is
    # simply no past: `_generations` takes the entries one at a time.
    return True


def _token_ok(token) -> bool:
    """A generation token this host may use as part of a directory name (``g-<token>``).

    It comes from a document another host wrote, and it becomes part of a PATH. Format 1
    checked only that it was a non-empty string. A ``../`` in it was not exploitable in
    practice - every path built from a token starts with a prefixed name (``g-..``,
    ``.staging-..``) that does not exist, so resolution fails before anything is written
    (measured, 2026-09-23) - but that is luck, not a rule. The rule is this module's: no name
    read from the store decides a local path. Only ``[0-9A-Za-z_-]``; every token haversack
    has written is a uuid4's hex."""
    return (isinstance(token, str) and 0 < len(token) <= 128
            and all(c.isascii() and (c.isalnum() or c in "_-") for c in token))


def _digest_ok(digest) -> bool:
    return (isinstance(digest, str) and digest.startswith("sha256:")
            and len(digest) == len("sha256:") + 64
            and all(c in "0123456789abcdef" for c in digest[len("sha256:"):]))


def _well_formed_files(files) -> bool:
    if not isinstance(files, dict):
        return False
    for name, blob in files.items():
        if not isinstance(name, str) or not isinstance(blob, dict):
            return False
        if not isinstance(blob.get("size"), int):
            return False
        if not _digest_ok(blob.get("digest")):
            return False
    return True


def _canonical(doc) -> bytes:
    """The one serialization of a document whose bytes are hashed: a manifest's digest
    must not depend on which host wrote it."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_of(data: bytes) -> str:
    import hashlib
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _dated(when) -> bool:
    return isinstance(when, (int, float)) and not isinstance(when, bool)


def _manifest_ok(m, key: str) -> bool:
    """Is this manifest one this code may act on - checked field by field, as a pointer is,
    and bound to its KEY: a ref naming another key's manifest would otherwise serve that
    key's result under this one."""
    if not isinstance(m, dict) or m.get("format") != POINTER_FORMAT or m.get("key") != key:
        return False
    if not _dated(m.get("published")):
        return False
    replaces = m.get("replaces")
    if not isinstance(replaces, list) or not all(_digest_ok(d) for d in replaces):
        return False
    if m.get("deleted") is True:
        return True                            # a tombstone names nothing else
    return _token_ok(m.get("publication")) and _well_formed_files(m.get("files"))


def _view(m: dict, digest: str) -> dict:
    """A manifest as every reader in this module sees an entry: the fields format 1's
    pointer had (``generation`` is the publication's token), plus where it came from."""
    return {"format": POINTER_FORMAT, "key": m["key"], "generation": m.get("publication"),
            "published": m["published"], "files": m.get("files") or {},
            "result": m.get("result"), "meta": m.get("meta"),
            "replaces": list(m.get("replaces") or []), "amends": m.get("amends") is True,
            "deleted": m.get("deleted") is True, "_manifest": digest, "_doc": m}


def _manifest_of(entry: dict) -> dict:
    """The manifest document an entry was read from - kept whole, because rebuilding it
    from the view could serialize differently and so name a different digest."""
    return entry["_doc"]


def _primary_name(files) -> str | None:
    """Which primary output a publication's files hold - labels or an embedding field
    (main, 2026-09-23) - or None. A generation holds exactly ONE (``serve.PRIMARY_NAMES``),
    so a pointer naming two is not one this code serves: which it answered with would
    depend on the order it asked in, the ambiguity the local cache refuses at write time."""
    from .serve import PRIMARY_NAMES
    held = [n for n in PRIMARY_NAMES if n in (files or {})]
    return held[0] if len(held) == 1 else None


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
    if not isinstance(past, dict) or not _token_ok(past.get("generation")):
        return False
    if not _well_formed_files(past.get("files")):
        return False
    when = past.get("published")
    if not isinstance(when, (int, float)) or isinstance(when, bool):
        return keep_undated
    return when > now - HISTORY_MAX_AGE_S


def _generations(ptr, *, now: float | None = None, keep_undated: bool = False) -> list:
    """The current publication and every kept predecessor, newest first - of a FORMAT 1
    pointer, whose history is inline. Format 2's is a chain: ``SharedResultCache._chain``.

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
        if _usable(past, now=now, keep_undated=keep_undated):
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

    def _is_newer_format(self, key: str) -> bool:
        """Does an entry this version cannot read say it came from a LATER one?

        Only that is worth refusing to overwrite: it is another haversack's current result
        and its whole history. Garbage under the same name - a truncated write, a stray
        object - is nobody's data, and refusing to publish over it would strand the key
        for ever.
        """
        try:
            raw = json.loads(bytes(ops.get(self.store, self._pointer_path(key)).bytes()))
        except Exception:                      # noqa: BLE001 - unreadable is not "newer"
            return False
        fmt = raw.get("format") if isinstance(raw, dict) else None
        return isinstance(fmt, int) and not isinstance(fmt, bool) and fmt > POINTER_FORMAT

    def _read_pointer(self, key: str, *, raise_faults: bool = False):
        """``(entry, update mode)``, or ``(None, mode)`` when there is no LIVE entry - absent,
        deleted (a tombstone), or unreadable: a miss, never a guess. The entry is a
        :func:`_view`, or a format 1 pointer as it was written.

        A store FAULT (credentials, network, a 503) is a miss here, and is RAISED for a
        caller that passes ``raise_faults`` - every writer, because reporting a fault as
        "absent" would have it publish over a pointer it could not read, and `get`, which
        falls back to this host's own copy rather than losing it.
        """
        kind, entry, mode = self._read_ref(key, raise_faults=raise_faults)
        return (entry if kind == "live" else None), mode

    def _read_ref(self, key: str, *, raise_faults: bool = False):
        """``(kind, entry, update mode)``: kind is ``"absent"``, ``"live"``, ``"tombstone"``
        or ``"unreadable"``. What callers that must tell those apart use - the sweep, which
        may not count a deletion as an unreadable entry, and `delete`, which may not count
        garbage as nothing.

        A format 2 ref carries its manifest's exact bytes beside the digest: checked against
        the digest, they make a current read ONE request, as format 1's was. The manifest
        OBJECT is the authority, and is read when the copy is missing or wrong.
        """
        try:
            got = ops.get(self.store, self._pointer_path(key))
        except FileNotFoundError:
            return "absent", None, None
        except Exception as e:                 # noqa: BLE001 - a read degrades, see _miss
            if raise_faults:
                raise
            _miss(f"reading the pointer for {key[:12]}", e)
            return "absent", None, None
        mode = update_mode(got.meta)
        try:
            doc = json.loads(bytes(got.bytes()))
        except (ValueError, UnicodeDecodeError):
            return "unreadable", None, mode
        if isinstance(doc, dict) and doc.get("format") == LEGACY_FORMAT:
            return ("live", doc, mode) if _well_formed(doc) else ("unreadable", None, mode)
        if (not isinstance(doc, dict) or doc.get("format") != POINTER_FORMAT
                or not _digest_ok(doc.get("manifest"))):
            return "unreadable", None, mode
        m = self._manifest(key, doc["manifest"], inline=doc.get("body"),
                           raise_faults=raise_faults)
        if m is None:
            return "unreadable", None, mode
        return ("tombstone" if m.get("deleted") is True else "live"), \
            _view(m, doc["manifest"]), mode

    def _manifest(self, key: str, digest: str, *, inline=None, raise_faults: bool = False):
        """The manifest ``digest`` names, verified against its name and bound to ``key``;
        None when it is gone, wrong, or not one of this key's."""
        data = inline.encode("utf-8") if isinstance(inline, str) else None
        if data is None or _digest_of(data) != digest:
            try:
                data = bytes(ops.get(self.store, self.blobs.path(digest)).bytes())
            except FileNotFoundError:
                return None
            except Exception as e:             # noqa: BLE001 - a read degrades, see _miss
                if raise_faults:
                    raise
                _miss(f"reading a manifest of {key[:12]}", e)
                return None
            if _digest_of(data) != digest:
                return None
        try:
            m = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return None
        return m if _manifest_ok(m, key) else None

    def _chain(self, entry, *, now: float | None = None, raise_faults: bool = False,
               manifests: set | None = None) -> list:
        """The current publication and every kept predecessor, newest first - one element
        per PUBLICATION, in the shape `_generations` gives a format 1 pointer.

        Walks ``replaces`` from the current manifest. A late artifact is an AMENDING
        manifest of the same publication, so a walk meets a publication's newest (most
        complete) manifest first and skips its earlier ones. It stops at HISTORY_KEEP
        predecessors, at the first one older than HISTORY_MAX_AGE_S, at a deletion (history
        does not survive one), at a manifest that is gone - the sweep takes a chain's tail
        once it falls out of these bounds, so a gone manifest is simply where history ends -
        and after CHAIN_STEPS_MAX reads whatever it found.

        ``manifests`` collects every manifest digest the walk READ, amending ones included:
        the sweep must keep the links, or the next walk ends early.
        """
        now = time.time() if now is None else now
        if entry.get("format") == LEGACY_FORMAT:
            return _generations(entry, now=now)
        key = entry["key"]
        if manifests is not None:
            manifests.add(entry["_manifest"])
        out = [{"generation": entry["generation"], "published": entry["published"],
                "files": entry["files"], "result": entry["result"], "meta": entry["meta"],
                "current": True}]
        seen = {entry["generation"]}
        nxt = entry["replaces"][0] if entry["replaces"] else None
        for _ in range(CHAIN_STEPS_MAX):
            if nxt is None or len(out) > HISTORY_KEEP:
                break
            m = self._manifest(key, nxt, raise_faults=raise_faults)
            if m is None or m.get("deleted") is True:
                break
            if m["publication"] not in seen:
                if m["published"] <= now - HISTORY_MAX_AGE_S:
                    break                      # and its manifest is not kept: the tail goes
                seen.add(m["publication"])
                out.append({"generation": m["publication"], "published": m["published"],
                            "files": m.get("files") or {}, "result": m.get("result"),
                            "meta": m.get("meta"), "current": False})
            if manifests is not None:
                manifests.add(nxt)             # a kept publication's, amending ones included
            nxt = m["replaces"][0] if m["replaces"] else None
        return out

    def _write_manifest(self, m: dict) -> str:
        data = _canonical(m)
        self.blobs.put_bytes(data)
        return _digest_of(data)

    def _predecessor(self, key: str, kind: str, entry, new: dict) -> list:
        """What a new manifest ``replaces``: the current one; for a format 1 entry, its
        whole history converted into a chain first (old tokens kept, so a local copy of it
        stays current); nothing for a first publication - or for a deletion of a format 1
        entry, whose history a tombstone would only have to keep alive to throw away."""
        if kind in ("live", "tombstone") and entry.get("format") == POINTER_FORMAT:
            return [entry["_manifest"]]
        if kind != "live" or new.get("deleted") is True:
            return []
        prev: list = []
        for g in reversed(_generations(entry)):        # oldest first
            if not _dated(g["published"]):
                # format 1's writer dropped a generation it could not date rather than keep
                # it forever; stamping it "now" here would have given it 30 more days
                continue
            prev = [self._write_manifest({
                "format": POINTER_FORMAT, "key": key, "publication": g["generation"],
                "published": g["published"], "files": g["files"], "result": g.get("result"),
                "meta": g.get("meta") or {}, "replaces": prev})]
        return prev

    def _swap(self, key: str, update):
        """Publish ``update(current)`` as a new manifest and point the ref at it by
        conditional write, rereading on every lost race; ``update`` returning None abandons
        the swap. Returns the written entry (a :func:`_view`), or None.

        ``update`` is given the LIVE entry or None, and returns the manifest's content:
        ``publication``, ``published``, ``files``, ``result``, ``meta`` (and ``amends``), or
        ``deleted``. ``replaces`` is this method's: it is what makes the history."""
        from obstore.exceptions import AlreadyExistsError, PreconditionError
        for _ in range(SWAP_ATTEMPTS):
            kind, entry, mode = self._read_ref(key, raise_faults=True)
            if kind == "unreadable" and self._is_newer_format(key):
                # an object is there that this version cannot read - most likely a newer
                # POINTER_FORMAT. Writing over it would take another host's current result
                # and its whole history out of the index in one write (review, 2026-09-20).
                raise ObjectStoreUnsuitable(
                    f"result {key[:12]}...: the store holds an entry this version cannot "
                    "read (a newer haversack wrote it); refusing to overwrite it. Upgrade "
                    "this host, or use a different --result-store prefix")
            new = update(entry if kind == "live" else None)
            if new is None:
                return None
            manifest = {"format": POINTER_FORMAT, "key": key, **new,
                        "replaces": self._predecessor(key, kind, entry, new)}
            data = _canonical(manifest)
            digest = _digest_of(data)
            self.blobs.put_bytes(data)
            ref = _canonical({"format": POINTER_FORMAT, "manifest": digest,
                              "body": data.decode("utf-8")})
            try:
                ops.put(self.store, self._pointer_path(key), ref,
                        mode=mode if mode is not None else "create")
            except (AlreadyExistsError, PreconditionError):
                continue                       # another writer moved it: read again
            return _view(manifest, digest)
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
                for g in self._chain(ptr)]

    def find_generation(self, key: str, digest: str) -> str | None:
        """Which kept generation of ``key`` published a primary output (labels, or a field)
        with this digest, if any.

        For `result:` references (main, 2026-09-20), which pin the referenced output's
        sha256 at submit and resolve AGAIN in the worker, refusing anything that is not
        those bytes. On one machine the pinned generation survives because the submit took
        a lease on it; across machines a lease means nothing, and another host may have
        republished the key in between - so the question the worker actually needs to ask
        is "which generation has these bytes", not "what is current".

        Bounded history answers it without a lease: the predecessor is listed, the sweep
        spares what history lists, and :meth:`fetch_generation` materializes it. Newest
        first, so a digest republished unchanged resolves to the current publication.

        Returns the generation token; ``fetch_generation`` then hands over the bytes.
        """
        ptr, _ = self._read_pointer(key)
        if ptr is None:
            return None
        for gen in self._chain(ptr):
            blob = (gen["files"] or {}).get(_primary_name(gen["files"]))
            if isinstance(blob, dict) and blob.get("digest") == digest:
                return gen["generation"]
        return None

    def fetch_generation(self, key: str, generation: str, dest) -> dict | None:
        """Materialize one kept generation into ``dest`` (which must exist); its entry, or
        None when that generation is not kept or its bytes have been swept.

        Into a directory the CALLER owns, never into the local cache: a historical read
        must not become what this host serves.
        """
        from .serve import ARTIFACT_NAMES
        ptr, _ = self._read_pointer(key)
        if ptr is None:
            return None
        for g in self._chain(ptr):
            if g["generation"] != generation:
                continue
            primary = _primary_name(g["files"])
            if primary is None:
                return None
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            written = []
            for name, blob in g["files"].items():
                if name not in (primary, *ARTIFACT_NAMES):
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
            return {**g, "written": written} if primary in written else None
        return None

    def get(self, key: str):
        """``(labels path, result)`` from the local copy of the CURRENT generation, filling
        it from the store first when the local copy is older or missing; None on a miss -
        including a pointer whose blobs have been swept.

        When the STORE cannot be reached and this host holds a complete copy, that copy is
        served. It may be stale - another host may have republished - but a cache serving
        what it already computed is what a cache is for, and answering a miss instead threw
        away a warm result and asked for a GPU on every request the outage lasted (review,
        2026-09-20). The deviation is reported, throttled.
        """
        try:
            self._pointer_path(key)            # a malformed key is not an outage
        except ValueError:
            return None
        try:
            ptr, _ = self._read_pointer(key, raise_faults=True)
        except Exception as e:                 # noqa: BLE001 - the store is unreachable
            _miss(f"reading the pointer for {key[:12]}", e)
            return self._fallback(key)
        if ptr is None:
            return None
        if not self._fill(key, ptr):
            return None
        self._confirm(key, ptr["generation"])
        return self.local.get(key)

    def _fill(self, key: str, ptr) -> bool:
        """Make the local copy hold ``ptr``'s generation with all of its files."""
        import shutil
        import tempfile
        import time as _time

        from .serve import ARTIFACT_NAMES
        gen, files = ptr["generation"], ptr.get("files") or {}
        primary = _primary_name(files)         # labels, or an encode job's field
        if primary is None:
            return False
        with self._fill_lock(key):
            local_dir = self.local._generation_dir(key, gen)
            # the PUBLICATION is here - its token current, its primary output and both
            # documents present - even if a late artifact is not. Asking `_holds` (every
            # file) instead re-downloaded the labels whenever a preview landed after this
            # host's copy, which with amending manifests is every result (step 3, 2026-09-23)
            have_gen = self._holds(key, ptr) or (
                self.local.generation(key) == gen
                and all((local_dir / n).exists() for n in (primary, "result.json", "meta.json")))
            # ONLY these names, and they are spelled out here rather than taken from the
            # pointer: a pointer is written by another host, and a name of its choosing
            # ("../..", an absolute path) would decide where these bytes land.
            wanted = [n for n in (primary, *ARTIFACT_NAMES) if n in files]
            file_sizes = {n: b["size"] for n, b in (ptr.get("files") or {}).items()
                          if isinstance(b, dict) and isinstance(b.get("size"), int)}
            if not have_gen and self.local.adopt(key, gen, names=wanted, sizes=file_sizes):
                return True                    # already here whole, only the pointer was
                                               # wrong: no download, no repair
            missing = [n for n in wanted if not (have_gen and (local_dir / n).exists())]
            if not missing:
                return True
            try:
                work = self._fresh_work_dir()
            except OSError as e:               # a full or read-only cache directory
                _miss(f"making room to fill {key[:12]}", e)
                return False
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
                    elif name == primary:
                        return False           # swept: a miss, and the next compute heals it
                    # an artifact that is gone is not a reason to lose the labels: a missing
                    # preview used to make the whole entry a miss, and a lost thumbnail is
                    # not worth a GPU recompute (review, 2026-09-19)
                if not have_gen:
                    try:
                        self.local.put(key, work / primary, ptr.get("result") or {},
                                       ptr.get("meta") or {},
                                       preview_path=_present(work / "preview.png"),
                                       statistics_path=_present(work / "statistics.json"),
                                       output_name=primary, generation=gen,
                                       # the work directory is inside this cache root, so
                                       # the files are handed over rather than copied: a
                                       # fill needed twice the result's size free
                                       move=True)
                    except FileExistsError as e:
                        if "being placed" in str(e):
                            # another PROCESS holds the claim (two servers over one
                            # --cache-dir). Its copy will be here in a moment; a miss here
                            # is a duplicate GPU compute of a key the store already holds
                            for _ in range(20):
                                _time.sleep(0.05)
                                if self._holds(key, ptr):
                                    return True
                            return False
                        # The generation directory is already here: a crash between its
                        # rename and the pointer write, another process placing it, or a
                        # pointer that moved back to it. Put back whatever is missing,
                        # atomically, and then make it current - returning a miss instead
                        # left the key unreadable on this host for ever, and on a compute
                        # server the miss became a recompute that overwrote a rollback
                        # (review, 2026-09-20).
                        for name in got:
                            _place(local_dir, name, work / name)
                        # the DOCUMENTS too: `local.put` writes them and it just raised, so
                        # a repair that placed only files left a hit whose result.json was
                        # missing - served as `{}`, a 200 with no outputs (review,
                        # 2026-09-20)
                        for name, doc in (("result.json", ptr.get("result") or {}),
                                          ("meta.json", ptr.get("meta") or {})):
                            doc_tmp = work / name
                            doc_tmp.write_text(json.dumps(doc), encoding="utf-8")
                            _place(local_dir, name, doc_tmp)
                        return (self.local.generation(key) == gen
                                or self.local.adopt(key, gen, names=wanted,
                                                    sizes=file_sizes))
                    except OSError as e:
                        # ENOSPC, EROFS, EACCES: the store gave good bytes and this host
                        # cannot keep them. A miss, like every other read failure - it used
                        # to leave the route a 500 (review, 2026-09-20)
                        _miss(f"keeping a local copy of {key[:12]}", e)
                        return False
                else:
                    for name in got:
                        self.local.add_artifact(key, name, work / name, generation=gen)
                return True
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _fallback(self, key: str):
        """This host's own copy, while the store cannot be reached - if it was confirmed
        current recently enough.

        An unbounded fallback serves a result that may have been DELETED elsewhere, and
        deletion is the one thing this design promises means gone. A host in an outage
        cannot tell; what it can know is when it last saw the entry alive in the store, so
        that is the bound (review, 2026-09-20).
        """
        import time as _time
        local = self.local.get(key)
        if local is None:
            return None
        try:
            seen = (Path(local[0]).parent / CONFIRMED_NAME).stat().st_mtime
        except OSError:
            seen = 0.0
        if _time.time() - seen > OUTAGE_GRACE_S:
            _warn_once(f"{key[:12]}...: the store is unreachable and this host last saw "
                       "the entry more than "
                       f"{OUTAGE_GRACE_S // 60:.0f} minutes ago; answering a miss rather "
                       "than serving a result that may since have been deleted")
            return None
        _warn_once(f"{key[:12]}...: the store is unreachable; serving this host's own "
                   "copy, confirmed current less than "
                   f"{OUTAGE_GRACE_S // 60:.0f} minutes ago")
        return local

    def _confirm(self, key: str, gen: str) -> None:
        """Record that the store just said this copy is current."""
        try:
            (self.local._generation_dir(key, gen) / CONFIRMED_NAME).touch()
        except OSError:
            pass                               # a read-only cache: the fallback shortens

    def _holds(self, key: str, ptr) -> bool:
        """Does this host already hold that publication COMPLETE - its files and both
        documents? One definition, because two disagreed: `pull` asked only about the
        pointer's files, so a copy whose `result.json` had gone was reported current and
        served as an empty result document (review, 2026-09-20)."""
        from .serve import ARTIFACT_NAMES
        gen, files = ptr["generation"], ptr.get("files") or {}
        primary = _primary_name(files)
        if primary is None or self.local.generation(key) != gen:
            return False
        where = self.local._generation_dir(key, gen)
        wanted = [n for n in (primary, *ARTIFACT_NAMES) if n in files]
        return all((where / n).exists() for n in (*wanted, "result.json", "meta.json"))

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
        import os
        from .cache_admin import _alive                 # noqa: PLC2701
        root = self.local.root
        mine = f"{WORK_PREFIX}{os.getpid()}-"
        for stale in root.glob(f"{WORK_PREFIX}*"):
            try:
                # Death is PROVED where it can be: a directory named by a process that is
                # gone from this host is reclaimed at once, however new it is - a reader
                # killed mid-fill otherwise left its download sitting for an hour, and a
                # dotted directory is invisible to `cache usage` and `cache clean`
                # (configuration sweep, 2026-09-20). Where death cannot be proved - a pid
                # from another host sharing this directory, a name this code did not write
                # - age is the fallback, and a live fill is never touched.
                pid, old = stale.name[len(WORK_PREFIX):].split("-")[0], (
                    _time.time() - stale.stat().st_mtime > WORK_GRACE_S)
                if pid.isdigit() and _alive(int(pid)) and not old:
                    continue                   # a fill this host can see running
                if pid.isdigit() and _alive(int(pid)):
                    # old AND apparently alive: either a fill that has run for hours, or a
                    # pid belonging to another host sharing this directory (whose numbers
                    # mean nothing here) or a number since reused. Age decides, because
                    # nothing else can (review, 2026-09-20)
                    shutil.rmtree(stale, ignore_errors=True)
                    continue
                if not pid.isdigit() and not old:
                    continue                   # a name this code did not write: age only
                shutil.rmtree(stale, ignore_errors=True)
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix=mine, dir=root))

    def put(self, key: str, labels_path, result: dict, meta: dict,
            preview_path=None, statistics_path=None, output_name: str | None = None) -> str:
        """Publish: blobs first, then the pointer, then the local copy. Returns the
        generation token, which the local copy shares.

        ``output_name`` is the primary output's name - labels, or an encode job's field
        (main, 2026-09-23) - checked BEFORE anything is uploaded, so a wrong name costs no
        blob and no pointer."""
        from .serve import PRIMARY_NAMES, RESULT_NAME
        output_name = output_name or RESULT_NAME
        if output_name not in PRIMARY_NAMES:
            raise ValueError(f"{output_name!r} is not a primary output ({', '.join(PRIMARY_NAMES)})")
        sources = {output_name: labels_path, "preview.png": preview_path,
                   "statistics.json": statistics_path}
        gen, now = uuid.uuid4().hex, time.time()
        def keep_the_work():               # one per `with`: a generator cannot be re-entered
            return _keeping_the_work(self, key, labels_path, result, meta, preview_path,
                                     statistics_path, gen, output_name)
        with keep_the_work():
            files = {name: self.blobs.put_file(src) for name, src in sources.items()
                     if src and Path(src).exists()}

        def publish(current):
            # what is being replaced becomes this manifest's `replaces` (in _swap)
            return {"publication": gen, "published": now, "files": files,
                    "result": result, "meta": meta}
        with keep_the_work():
            pointer = self._swap(key, publish)
        self._verify(pointer["files"], sources, manifest=pointer)
        try:
            self.local.put(key, labels_path, result, meta, preview_path=preview_path,
                           statistics_path=statistics_path, output_name=output_name,
                           generation=gen)
        except Exception as e:                 # noqa: BLE001
            # The publication HAPPENED - every host can read it - so a local copy that
            # cannot be written must not fail the job that just produced it (a full disk
            # did exactly that: `put` raised after the result was visible cluster-wide,
            # review 2026-09-19). The next read on this host fills the copy again.
            _miss(f"keeping a local copy of {key[:12]}", e)
        else:
            self._confirm(key, gen)            # it saw the entry current: it wrote it
        return gen

    def _verify(self, files: dict, sources: dict, *, manifest=None) -> None:
        """Put back any blob that was not uploaded because it already existed and has since
        been swept - the window between that check and the pointer write. The pointer now
        references them, so this is the last moment anything may quietly remove them.

        ``manifest`` (a written entry) is checked too: the ref carries a copy, so the
        present never needs the object - but the NEXT publication's history walk does."""
        for name, blob in files.items():
            try:
                if not self.blobs.has(blob["digest"]):
                    self.blobs.put_file(sources[name])
            except Exception as e:             # noqa: BLE001 - the pointer is already out
                _miss(f"re-checking {name}", e)
        if manifest is not None and manifest.get("format") == POINTER_FORMAT:
            try:
                if not self.blobs.has(manifest["_manifest"]):
                    self._write_manifest(_manifest_of(manifest))
            except Exception as e:             # noqa: BLE001
                _miss("re-checking a manifest", e)

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
            # an AMENDING manifest of the same publication: its token and date stay, so
            # the local copy is still current and history counts it once
            return {"publication": ptr["generation"],
                    "published": ptr["published"] if _dated(ptr.get("published"))
                    else time.time(),
                    "files": files, "result": ptr.get("result"), "meta": ptr.get("meta"),
                    "amends": True}
        try:
            blob = self.blobs.put_file(src_path)
            written = self._swap(key, update)
            if written is None:
                return False
            self._verify({name: blob}, {name: src_path}, manifest=written)
        except Exception as e:                 # noqa: BLE001
            _miss(f"placing {name} on {key[:12]}", e)
            return False
        self.local.add_artifact(key, name, src_path, generation=written["generation"])
        return True

    def delete(self, key: str) -> bool:
        """Remove the entry: the pointer, and this host's copy. True if anything went.

        **The BYTES are not reclaimed here** - `haversack cache sweep` does that, once
        nothing references them (2026-09-20). Reclaiming them at delete time meant deciding,
        against live publishers, whether a blob this entry named was also one that a
        publication happening right now had deduplicated onto. Four attempts went into that
        question - pre-listed candidates, a refreshed timestamp, a re-check before each
        delete, a wait for coarse clocks - and the reviewers were still finding holes in it.
        The guarantee it was buying, "a deletion removes the bytes the moment it returns",
        is one the operator can have instead by running a sweep, which decides the same
        question with nothing else moving.

        Asked of the OBJECT, not of a parse: a pointer this version cannot read is still an
        entry, and answering False for it told an operator deleting a patient's result that
        there had been nothing there. An entry a NEWER haversack wrote is refused outright:
        removing it would leave its blobs with nothing naming them, and this version's own
        sweep - which spares nothing it cannot account for only while that pointer is
        THERE - would then collect another host's live data.

        A live entry is replaced by a TOMBSTONE (format 2), not removed: a manifest saying
        "deleted", written by the same conditional write as a publication. It keeps nothing
        alive - the sweep marks no blob a tombstone's predecessors named, so the result's
        bytes and its whole history go at the next sweep - and it is what a SYNC will carry
        to a copy of this store, where a removed ref would be indistinguishable from one
        never copied. Garbage under the ref's name is nobody's data and is removed outright.
        """
        if self._is_newer_format(key):
            raise ObjectStoreUnsuitable(
                f"result {key[:12]}...: this entry was written by a newer haversack. "
                "Removing it here would leave its bytes with nothing naming them, and a "
                "later sweep would collect them - upgrade this host and delete it there")
        # anything the store raises comes out: a delete must not report success on doubt
        kind, _entry, _mode = self._read_ref(key, raise_faults=True)
        existed = False
        if kind == "unreadable":
            try:
                ops.delete(self.store, self._pointer_path(key))
            except FileNotFoundError:
                pass
            existed = True
        elif kind == "live":
            def tombstone(current):
                return (None if current is None
                        else {"deleted": True, "published": time.time()})
            existed = self._swap(key, tombstone) is not None
        local = self.local.delete(key)
        return existed or local

    def list(self, *, keys=None, limit: int | None = None, after=None, accept=None,
             match=None, memo=None, hold=None, workers: int | None = None) -> tuple:
        """``(rows, position)``, newest published first - ``ResultCache.list``'s contract,
        answered from the store (main's listing, 2026-09-21).

        The three mechanisms it rests on carry over, and two of them get cheaper here:

        1. ``keys`` - an identity filter COMPUTES the names it wants, so this reads exactly
           those pointers and never lists the bucket. A miss is one 404, not a search.
        2. Order and paging come from names and times the LISTING itself returns: an object
           store gives last-modified with every name, so what costs a stat per entry on a
           filesystem costs nothing extra here. ``after`` is the same ``(stamp, key)``
           position, in nanoseconds, so a cursor issued by either cache means the same
           thing.
        3. Content is read only for the page. The pointer IS the content - meta, sizes and
           which artifacts exist are all in the one document - so a row costs ONE request
           where the local cache pays a read plus three stats. ``workers`` reads a chunk in
           parallel, ``memo`` spares a long-lived server the re-read.

        ``hold`` is accepted and honored for symmetry with the local cache, where it is the
        Modal view lock; a bucket has no view to hold still, so it guards nothing here.
        """
        from concurrent.futures import ThreadPoolExecutor, wait

        from .serve import LIST_CHUNK, LIST_WORKERS, resource_links
        import contextlib
        hold = hold or contextlib.nullcontext
        workers = LIST_WORKERS if workers is None else int(workers)
        pool = []

        def order(position):                   # newest first; the key makes it total
            return -position[0], position[1]

        def each(fn, items) -> list:
            with hold():
                if len(items) < 4 or workers < 2:
                    return [fn(x) for x in items]
                if not pool:
                    pool.append(ThreadPoolExecutor(max_workers=workers,
                                                   thread_name_prefix="haversack-list"))
                futures = []
                try:
                    for x in items:
                        futures.append(pool[0].submit(fn, x))
                finally:
                    wait(futures)
                return [f.result() for f in futures]

        def row_of(key: str, stamp: int, ptr) -> dict | None:
            files = ptr.get("files") or {}
            primary = _primary_name(files)
            if primary is None:
                return None
            meta = ptr.get("meta") if isinstance(ptr.get("meta"), dict) else {}
            task, identity = meta.get("task"), meta.get("identity")
            options = meta.get("options")
            row = {"key": key, "task": task, "identity": identity, "options": options,
                   "computed": meta.get("computed"), "published": stamp / 1e9,
                   "bytes": files[primary]["size"]}
            if meta.get("kind") not in (None, "segment"):
                row["kind"] = meta["kind"]     # the local cache's rule: a field is said, never
                return row                     # linked as labels
            if resource_links(task, identity, options):
                row["links"] = resource_links(task, identity, options,
                                              preview="preview.png" in files,
                                              statistics="statistics.json" in files)
            return row

        def fetch(candidate):
            """The pointer for one candidate, remembered under the stamp the caller saw."""
            stamp, key = candidate
            if memo is not None:
                hit = memo.get(key, stamp)
                if hit is not None:
                    return hit[1]
            try:
                ptr, _ = self._read_pointer(key)
            except ValueError:                 # not a key this code would ever write
                return None
            if ptr is None:
                return None
            fields = {"_ptr": ptr}
            if memo is not None:
                memo.put(key, stamp, "", fields)
            return fields

        try:
            if keys is None:
                with hold():
                    found = sorted(self._stamps(), key=order)
            else:
                names = [k for k in dict.fromkeys(map(str, keys)) if self._listable(k)]
                stamped = each(self._stamp, names)
                found = sorted(((t, k) for k, t in zip(names, stamped) if t is not None),
                               key=order)
            if after is not None:
                found = [c for c in found if order(c) > order(after)]
            rows, i, clean = [], 0, True
            while i < len(found) and (limit is None or len(rows) < limit):
                need = len(found) - i if limit is None else limit - len(rows)
                chunk = found[i:i + (min(need, LIST_CHUNK) if clean else LIST_CHUNK)]
                said = each(fetch, chunk)
                built = {}
                for c, fields in zip(chunk, said):
                    if fields is None:
                        continue
                    ptr = fields["_ptr"]
                    meta = ptr.get("meta") if isinstance(ptr.get("meta"), dict) else {}
                    said_fields = {k: meta.get(k) for k in ("task", "identity", "options",
                                                            "computed")}
                    if meta.get("kind") not in (None, "segment"):
                        said_fields["kind"] = meta["kind"]     # as the local cache's fields
                    if match is not None and not match(said_fields):
                        continue
                    built[c] = row_of(c[1], c[0], ptr)
                for c in chunk:
                    if limit is not None and len(rows) >= limit:
                        break                  # read ahead of the page: remembered, not sent
                    i += 1
                    row = built.get(c)
                    if row is not None and (accept is None or accept(row)):
                        rows.append(row)
                    else:
                        clean = False
            return rows, (found[i - 1] if 0 < i < len(found) else None)
        finally:
            for pl in pool:
                pl.shutdown(wait=True)

    def _listable(self, key: str) -> bool:
        try:
            self._pointer_path(key)
        except ValueError:
            return False
        return True

    def _stamps(self) -> list:
        """``(stamp, key)`` for every entry, from the bucket listing alone - the store
        hands out last-modified with each name, so this is one request per thousand
        entries and no read at all."""
        import datetime as _dt

        base = f"{self.prefix}results/"
        out = []
        for batch in ops.list(self.store, base):
            for obj in batch:
                name = obj["path"][len(base):]
                if "/" in name or not name.endswith(".json"):
                    continue
                when = obj.get("last_modified")
                if isinstance(when, _dt.datetime):
                    when = when.timestamp()
                if isinstance(when, (int, float)):
                    out.append((int(when * 1e9), name[:-len(".json")]))
        return out

    def _stamp(self, key: str) -> int | None:
        """When this key was published, in ns, or None - one HEAD of its pointer."""
        import datetime as _dt

        try:
            meta = ops.head(self.store, self._pointer_path(key))
        except FileNotFoundError:
            return None
        except Exception as e:                 # noqa: BLE001 - a read degrades, see _miss
            _miss(f"stat of {key[:12]}", e)
            return None
        when = meta.get("last_modified")
        if isinstance(when, _dt.datetime):
            when = when.timestamp()
        return int(when * 1e9) if isinstance(when, (int, float)) else None

    def evict(self) -> None:
        """Bounds the LOCAL copy only. The store is bounded by ``sweep``: it keeps no
        access times, and a count-bounded LRU over it would need an index this protocol
        deliberately does not have."""
        self.local.evict()

    # -- store maintenance ---------------------------------------------------------------

    def _scan_pointers(self, *, newest_first: bool = False, limit: int | None = None,
                       tombstones: list | None = None):
        """``(pointers this code can read, how many it could NOT)``.

        An object under ``results/`` that is not a
        pointer this version understands - a stray upload, a `.tmp` file from someone's
        sync, a FORMAT FROM A NEWER WRITER - is skipped and counted, never raised: one
        stray object used to abort both ``list`` and ``sweep`` permanently, and a sweep
        that never runs is a bucket that never stops growing (review, 2026-09-19).

        ``newest_first`` orders by the pointers' own last-modified before reading any of
        them, so ``limit`` costs that many reads instead of one per entry in the bucket.
        """
        base = f"{self.prefix}results/"
        entries = []
        for batch in ops.list(self.store, base):
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
                kind, ptr, _mode = self._read_ref(key)
            except ValueError:                 # not a key this code would ever write
                kind, ptr = "unreadable", None
            if kind == "tombstone":
                # a deletion is READ, not unreadable: counting it would freeze the sweep
                # for as long as the tombstone stands
                if tombstones is not None:
                    tombstones.append(ptr)
                continue
            if kind != "live":
                unreadable += 1
                continue
            out.append({**ptr, "_key": key})
        return out, unreadable

    def sweep(self, *, max_age_s: float | None = None, grace_s: float = BLOB_GRACE_S,
              now: float | None = None, allow_empty: bool = False) -> dict:
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

        An index with NO entries is refused unless ``allow_empty`` says it was meant.
        "Everything here was deleted" and "I am looking in the wrong place" arrive as the
        same answer - no pointers - and only one of them means the bytes are garbage. After
        deleting the last entry in a store, that is the flag to pass.
        """
        from provender import EmptyKeepSet
        now = time.time() if now is None else now
        candidates = self.blobs.entries(older_than=now - grace_s)   # BEFORE the pointers
        referenced, expired, tombstones = set(), 0, []
        pointers, unreadable = self._scan_pointers(tombstones=tombstones)
        for tomb in tombstones:
            # the tombstone itself only: what it replaced - the result and its history -
            # is exactly what a deletion is for reclaiming
            referenced.add(tomb["_manifest"])
        for ptr in pointers:
            published = ptr.get("published")
            datable = isinstance(published, (int, float)) and not isinstance(published, bool)
            # an entry nothing can date is never expired: the one field cleanup judges by
            # must be one it can read, or it is not evidence (review, 2026-09-20)
            if max_age_s is not None and datable and published < now - max_age_s:
                try:
                    ops.delete(self.store, self._pointer_path(ptr["_key"]))
                except FileNotFoundError:
                    pass
                expired += 1
                continue
            # a margin past the listing bound: hosts do not share a clock, and one
            # running fast must not collect what the others still list. Every manifest the
            # walk READ is kept too - an amending one included - or the next walk would end
            # at the gap and the history behind it would be collected a sweep later.
            try:
                chain = self._chain(ptr, now=now - HISTORY_GC_MARGIN_S, raise_faults=True,
                                    manifests=referenced)
            except Exception as e:             # noqa: BLE001
                # a walk that faulted marked less than is live: that is an unreadable
                # entry for this sweep's purposes, and no blob is deleted (below)
                _miss(f"walking the history of {ptr['_key'][:12]}", e)
                unreadable += 1
                continue
            for gen in chain:
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
                                   now=now, allow_empty=bool(pointers) or allow_empty)
        except EmptyKeepSet:
            if candidates:
                print(f"warning: no entries under {self.prefix}results/, but "
                      f"{len(candidates)} blob(s) are stored there; deleting none. If they "
                      "really were all deleted, sweep again with --empty-index-ok; if this "
                      "is the wrong prefix, that flag would empty someone else's store.",
                      file=sys.stderr, flush=True)
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
        from .serve import ARTIFACT_NAMES, PRIMARY_NAMES
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
            sources = {n: where / n for n in (*PRIMARY_NAMES, *ARTIFACT_NAMES)
                       if (where / n).exists()}
            if _primary_name(sources) is None:         # none, or two: nothing to serve
                out["unreadable"] += 1
                continue
            # Asked BEFORE the bytes are touched, for every policy that can refuse: a rerun
            # over a cache of thousands should cost one pointer read each, not a re-hash
            # and re-upload of every result. `_swap` asks again under the conditional
            # write, so a key published in between is still not lost.
            try:
                current, _mode = self._read_pointer(key, raise_faults=True)
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
        out = {"pulled": 0, "current": 0, "failed": 0, "unreadable": 0, "evicted": 0}
        placed: list = []
        pointers, unreadable = self._scan_pointers(newest_first=True, limit=limit)
        out["unreadable"] = unreadable
        for ptr in reversed(pointers):                 # oldest first: see above
            key = ptr["_key"]
            if self._holds(key, ptr):
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
            if report:
                report(key, "pulled" if ok else "failed")   # a failure names its key too
        # the local cache is count-bounded, and a pull of more entries than it keeps cannot
        # leave them all servable. Asked only of what THIS pull placed - an entry that
        # failed was never there to evict - and reported rather than counted as pulled.
        gone = (sum(1 for key in placed if self.local.generation(key) is None)
                if len(placed) > self.local.keep else 0)
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
        # a re-push of the generation already current is the same publication: the history
        # walk counts it once, which `_kept_history` had to special-case in format 1
        return {"publication": gen, "published": now, "files": files, "result": result,
                "meta": meta}
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


@contextlib.contextmanager
def _keeping_the_work(cache, key, labels_path, result, meta, preview_path, statistics_path,
                      gen, output_name):
    """Keep this host's copy of a result the STORE refused.

    The segmentation is finished and its bytes are on this disk. Failing the job is right -
    the publication did not happen, and no other host can see it - but discarding the work
    as well means a GPU run per request for as long as the outage lasts (review,
    2026-09-20).

    What the kept copy is good for is narrower than it sounds: the store has no pointer, so
    an ordinary read is a miss, and only the OUTAGE fallback - reads failing too - serves
    it. It is kept for the case that motivated it, a store that is down rather than one
    that refuses: a refusal (a newer-format entry, a publication storm) is deliberate and
    repeatable, and keeping a copy for it only fills this disk with generations nothing
    will read.
    """
    from .errors import InputError
    try:
        yield
    except (InputError, RuntimeError):         # refused, not unreachable: nothing to keep
        raise
    except Exception:                          # noqa: BLE001 - the raise is the news
        try:
            cache.local.put(key, labels_path, result, meta, preview_path=preview_path,
                            statistics_path=statistics_path, output_name=output_name,
                            generation=gen)
        except Exception:                      # noqa: BLE001
            pass
        raise


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
