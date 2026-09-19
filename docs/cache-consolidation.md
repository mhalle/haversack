# One cache, three deployments — the consolidation, and how to get there safely

**Status (2026-09-19): design, nothing built beyond the shared result store on this branch.**
This document is the handoff: what exists, what is proposed, what must not break, and the
order to do it in. Read it with `AGENTS.md` ("Result-cache lifetimes", "The 2026-09-10 review
round — lifetimes") — the defects listed there are the evidence this design is answering.

## Where things stand

Three stores exist today, each with its own protocol, and every one of them was written
because the last one did not fit:

| store | code | keyed by | protocol | lives |
|---|---|---|---|---|
| result cache | `serve.ResultCache` | request digest (`result_key`) | generation directory + `current` pointer, renames, `flock` claims, `.lease` files, tombs, an LRU ceiling | local disk, Modal volume |
| input/series cache | `serve.SeriesCache` + `content.ContentStore` | series id, or the bytes' SHA-256 | one directory per entry, `.owner` claim published by `os.link`, `.done` marker, LRU by byte budget | local disk, Modal volume |
| shared result store | `objectcache.SharedResultCache` (new, commit `b9494c6`) | request digest | blobs named by SHA-256 + one pointer per key, replaced by conditional write | S3/GCS/R2 (tested on R2) |

The new store is deliberately in FRONT of nothing: it is authoritative, and the local
`ResultCache` is its read-through copy. So today a server using `--result-store` runs two
protocols at once, which is more machinery than before, not less.

**The part that keeps producing defects is the same part in all three:** deciding when it is
safe to DELETE something. Leases, writer claims, `.owner` links, "death is proved, never
inferred", the tomb-and-second-question dance, `GENERATION_GRACE_S` — every one of those
exists to make deletion safe while another process may be mid-read or mid-write.

## The observation this design turns on

A result is mutable ONLY because it is published as a directory behind a pointer. Its bytes
are not: a labels file is the output of one computation and never changes. The same is
already true of inputs, which `content.py` names by their own SHA-256.

If every byte in the cache is stored under the digest of its own content:

- **A reader needs no lease.** It opens the file; POSIX keeps the inode alive while the
  descriptor is open, so a concurrent delete cannot take it away mid-read. On the object
  store, an immutable object either exists or does not — and a miss is safe.
- **A writer needs no claim.** Two writers producing the same bytes write the same file, and
  the loser's temp file is discarded. A half-written file is never visible, because it is
  renamed into place only when complete (locally) or PUT atomically (remotely).
- **Cleanup needs no liveness proof.** Deleting a blob that someone is reading is harmless
  locally (the descriptor survives) and, remotely, is a miss.
- **What remains mutable is one small pointer per key**, and mutation of that pointer is a
  compare-and-swap: a conditional write remotely, an atomic rename locally.

That is the whole of it. The lease/claim/tomb apparatus can then be DELETED rather than
ported — which is the only outcome that makes the cache less defect-prone instead of more.

## The target

One protocol, two backends, three deployments:

```
        index (small, mutable, CAS-swapped)          blobs (immutable, digest-named)
local   <root>/index/<key>.json  (temp+rename)       <root>/blobs/sha256/<hex>
remote  results/<key>.json       (If-Match)          blobs/sha256/<hex>
```

- **One `BlobStore` interface**, two implementations (local directory, object store). The
  object-store one is written and tested (`objectcache.BlobStore`).
- **One `Index` interface**: read a pointer with a version token, write it only if unchanged.
  Locally that is a file and `os.replace` under one lock; remotely a conditional PUT.
- **A pointer is the same document in both**: format, generation, files (name → digest,
  size), the result and meta documents, published time.
- **Inputs join it.** `ContentStore` entries are already digest-named; a fetched series
  becomes a blob per file plus a small pointer (`inputs/<series>` → tree of digests), which
  also retires `SeriesCache`'s `.owner`/`.done` protocol and its hard-link requirement (the
  exFAT/FAT32 fallback, `sources.sole_file`, the case-folding hazard).
- **Eviction becomes boring**: blobs are deleted least-recently-used locally, and by
  unreferenced-and-old remotely. Neither needs to know who is reading.
- **Modal** then needs no cache volume semantics at all: the object store is the authority
  and each container's local blob directory is a cache of it. That removes the
  `commit()`/`reload()` ordering hazards from the result path (they stay for weights).

### What this does NOT fix

- The **result KEY** (`result_key`, `CACHE_EPOCH`, per-engine `cache_epoch`) is unchanged
  and stays the contract. Consolidation is about storage, not about what a key means.
- **Single-flight** (one computation per key) is a separate mechanism and stays where it is.
- **Weights** are not a cache in this sense; leave `weights_fetch` alone.
- **Ranked/duckn stores** are outputs, not caches. Out of scope.

## Home-rolled or existing?

Surveyed 2026-09-19. Nothing off the shelf does the whole job, and the two closest are worth
copying rather than adopting:

- **Bazel's remote cache** (ActionCache over a CAS; bazel-remote, BuildBuddy, Buildbarn) is
  the same shape, and its degrade-to-miss rule is the one this design uses. Adopting a server
  means gRPC, protobufs, and an extra service per deployment. **Copy the shape, not the
  software** — already done in `objectcache`.
- **python-diskcache** (SQLite index + files, LRU, safe across processes) is a credible
  LOCAL tier and would retire hand-rolled eviction. Unverified here: behaviour with
  multi-MB values, and whether its index survives the filesystems this repo supports (a
  cache root on exFAT is a supported configuration). Worth a measured experiment before it
  is trusted, not an assumption.
- **Icechunk** (transactional Zarr on object storage) is array-shaped; interesting for the
  ranked stores, not for this.
- **lakeFS / SlateDB / ZeroFS** were assessed and rejected: a server plus a database, a
  single writer, and a single writer whose locks vanish on restart, respectively.

**Recommendation: home-rolled, small, on obstore + the local filesystem** — but with the
local tier's eviction measured against diskcache before writing a third LRU by hand.

## The rule while this happens: do not disturb what runs today

1. **The default path does not change.** With no `--result-store` and no new flag, a server
   uses `ResultCache` exactly as it does now. The consolidation ships behind its own opt-in
   until it has been through an adversarial review round and a soak.
2. **No cache epoch bump.** Storage changes must not invalidate a single existing result;
   the key is untouched, so they do not.
3. **Both protocols readable during the transition.** The new local store reads a legacy
   generation directory (as `ResultCache` already reads a legacy flat entry) rather than
   requiring a flag day.
4. **Modal last.** Its deployment is the one with results that cost GPU time. It moves only
   after the local path has run in anger and after a migration exists that can be verified
   against a second, empty host.
5. **A rollback exists at every step**: the old code stays until the new one has run a
   release cycle, and `cache push`/`pull` moves entries either way.

## Order of work

1. **Adversarial review + soak of `objectcache` as it stands.** Multi-process publication and
   reading against a real bucket; a writer killed mid-publication; the sweep racing a
   publication. This is cheap and it either finds defects now or raises confidence in the
   protocol the rest of the plan rests on. **Do this before writing more.**
2. **`haversack cache push/pull`** (design in the session of 2026-09-19): migrate an existing
   local or Modal cache into the store, preserving each entry's generation token so the local
   copy is instantly current. Idempotent by construction.
3. **The local blob store + index**, behind a flag, reading legacy entries. Measure against
   diskcache first. Retire lease/claim/tomb code ONLY once its tests are green on both.
4. **Inputs onto the same store**, retiring `SeriesCache`'s claim protocol.
5. **Modal**, once 1–4 have run locally: the object store becomes the authority, the cache
   volume becomes a local blob directory, and `cache_get`'s commit/reload dance goes.
6. **Delete the old protocol** and the tests that pin it, in one commit, with the CHANGELOG
   saying what is no longer possible.

## Open questions for the user

- Is the local tier meant to become content-addressed (steps 3–4), or is the object store
  only for sharing between servers? Everything above assumes the former.
- Does a single-server, no-network install have to keep working with no object store at all?
  (Assumed yes: `segment` and a laptop server must not need a bucket.)
- How much history matters: should a republication keep the previous result (the store can,
  cheaply), or is last-writer-wins enough (today's behaviour)?
