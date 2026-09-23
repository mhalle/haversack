# One cache, three deployments — the consolidation, and how to get there safely

**Status (2026-09-20): the shared result store, its bounded history, `cache push`/`pull`/`sweep` and the provender extraction are built and reviewed three times; steps 3-6 below are still design.**
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

- **One blob-store interface**, two implementations (local directory, object store). The
  object-store one is `provender.Blobs` (extracted 2026-09-20, shared with feldglas); the
  local one is step 3.
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
  LOCAL tier and would retire hand-rolled eviction. Unverified here: behavior with
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

## Configurations actually run (2026-09-20)

Every row is a configuration nothing had run before - the reviewers could attack code but
not environments. Three of the six found defects, which is the usual ratio for this
subsystem.

| configuration | result |
|---|---|
| Two servers sharing ONE `--cache-dir`, against one store (the per-GPU deployment) | **Sound.** 446 reads, no torn read, no error, and the shared directory coherent afterwards: every current generation complete, no dotfiles left behind. |
| Two haversack versions against one bucket | **One defect.** The older host DELETED an entry it could not read - removing the index and leaving bytes it cannot name. It refuses now (409 on the wire). Publishing over it and sweeping its blobs were already refused. |
| A reader killed mid-fill (SIGKILL, 24 MB result) | **One defect.** Its work directory survived an hour, invisible to `cache usage` and `cache clean`, although its process was provably gone. Death is proved and reclaimed at once now; age remains the fallback for a pid this host cannot judge. |
| A store that is SLOW, not broken (1.5 s per read) | **Sound.** `/v1/health` answered in 0.01 s and `/v1/tasks` in 0.00 s while a store-touching route waited 1.52 s: the offload does what it claims. |
| A cache disk that runs OUT OF SPACE (12 MB image) | **One defect.** A fill downloaded into its work directory and then copied into place, so it needed TWICE the result's size free - a 6 MB result failed with 11 MB free. The files are handed over (`ResultCache.put(move=True)`) now, which also stops reading and writing every byte twice; and a failed publication no longer leaves an empty entry directory. |
| A cache root on exFAT: no hard links, case-INSENSITIVE | **Sound**, 9 checks: publish, read, pull to another host, republish, push, history, sweep, delete, and no leavings. This is the first time the shared-store protocol has run anywhere but APFS. |
| The purge race on a real store (`cfg7`, added after the fifth review) | **Sound**, and it corrected a premise. A deleting host and a deduplicating publisher over ONE blob, 40 s: 454 reads, 4 misses, no torn read, the survivor readable on a cold host at the end. R2 reports last-modified to the MILLISECOND through obstore, so the whole-second waiting path never engages there - the re-check alone carries it. AWS S3 documents second-resolution Last-Modified and is where the wait is expected to matter; unverified. A `delete` cost ~0.8 s here (its listing, its scan of the other entries, and its deletes). |

Still unrun: a cache root on a network filesystem; Modal; a store with object versioning;
a bucket large enough to make `list` and `sweep` cost real money.

## Order of work

1. ~~**Adversarial review + soak of `objectcache` as it stands.**~~ **DONE 2026-09-19.**
   Three review agents (protocol, its tests, the serve wiring) found **thirteen defects,
   all fixed and each pinned by a test that fails on the pre-fix commit** (17 of the new
   tests do). The soak - publishers, readers and a sweeper as separate processes against
   R2, one publisher killed mid-flight - saw no torn read and no error, at zero sweep grace
   and at the shipped one.

   What the round taught, for the consolidation to carry forward:
   - **A store fault is not an OSError.** obstore's exceptions do not subclass it, so the
     local cache's "any read failure is a miss" did not carry over and every fault became a
     500. The local blob store must make the same promise explicitly, and a test must hold
     both backends to it.
   - **A pointer written by another host is data, not a promise.** Validate it field by
     field, and never let a name in it decide a local path.
   - **Cleanup must refuse what it cannot account for.** A sweep meeting a pointer format
     it does not know deletes nothing - otherwise an old host collects a new one's results.
     This is the object-store form of "death is proved, never inferred", and the local
     content-addressed store will need its own version.
   - **Shared bytes make deletion collateral.** Deduplication means one blob belongs to
     many results; a single client's bad read is not grounds to delete it.
   - **A blocking call changes what a route may do.** A lookup that was a stat became a
     network round trip, and the async routes had to move it off the event loop. Every
     backend the consolidation adds has to be re-examined for this, not assumed.
2. ~~**`haversack cache push/pull`**~~ **DONE 2026-09-20.** `cache push [--conflict
   skip|newer|force]` and `cache pull`, both `--limit`-able, on the local cache and any
   store. Generation tokens are preserved, so a pushed entry is served locally without a
   download. Verified against R2: 12 entries pushed in 16.0 s, a second push skipped all
   12 in 1.7 s, and a pull onto an EMPTY machine took 4.7 s, after which that machine
   served every entry from its local cache with no store involved. The Modal half is
   the same call inside a container with the cache volume mounted, and is still to do.
3. **The local blob store + index**, behind a flag, reading legacy entries. Measure against
   diskcache first. Retire lease/claim/tomb code ONLY once its tests are green on both.
4. **Inputs onto the same store**, retiring `SeriesCache`'s claim protocol.
5. **Modal**, once 1–4 have run locally: the object store becomes the authority, the cache
   volume becomes a local blob directory, and `cache_get`'s commit/reload dance goes.
6. **Delete the old protocol** and the tests that pin it, in one commit, with the CHANGELOG
   saying what is no longer possible.

## Decided (2026-09-19, by the user)

**1. One protocol, with a TIME-LIMITED transition.** The local tier becomes
content-addressed (steps 3–4) and the lease/claim/tomb/ceiling apparatus is deleted, not
ported. The transition is bounded on purpose: reading a legacy generation directory is a
migration shim with an expiry, not a second supported protocol. Concretely:

- The new local store reads legacy entries from its first release, and `cache push`/`pull`
  migrate them in bulk.
- The shim is removed **two minor releases after the release that introduces the new store,
  or 90 days, whichever is later**, and the CHANGELOG says so in the release that adds it.
- After removal, a legacy entry is not read - it is ignored and evicted, and `cache clean`
  removes it. Nothing is silently reinterpreted.
- Deliberately NOT enforced by a dated test: a check that turns CI red on a calendar day
  fails the wrong person on the wrong morning. The deadline lives in the CHANGELOG, in this
  document, and in the shim's own docstring, and removing it is a scheduled task.

**2. Working with no network and no object store is a REQUIREMENT, not a default.** The
local store stands alone: no configuration, no bucket, no credentials, offline, in the lean
install, and on a filesystem without hard links (a cache root on exFAT is supported and
tested). The object store is an option that servers may share, never the substrate. Any
design step that would make a bucket necessary is out of bounds.

**3. Bounded history, in the shared store only.** A republication keeps its predecessors:
the pointer carries a bounded list of previous generations, and the sweep treats their blobs
as referenced. Bounded by count and by age, so the POINTER is bounded per key; the blobs a
generation leaves when it falls off are reclaimed by `cache sweep`, which nothing runs on a
schedule. Deduplication
makes this nearly free when a recomputation produces identical bytes. The local copy keeps
no history - it holds the current generation, as it does today. `delete` removes the entry
AND its history. History is read explicitly (`history()`, `fetch_generation()`); no
ordinary read can be served a superseded result by accident.

**Amended 2026-09-20.** The original decision added "for anything near patient data,
deletion means gone", meaning the BYTES went the moment `delete` returned. That guarantee
was withdrawn, by the user, after it proved to be the most expensive line in the design.
Reclaiming bytes at delete time means deciding, against live publishers, whether a blob the
entry named is also one that a publication happening right now has deduplicated onto. Four
attempts went into that question - pre-listing the candidates, refreshing a deduplicated
blob's timestamp, re-checking each candidate before deleting it, waiting out a coarse
clock - and reviewers were still finding holes. `delete` now removes the entry; a sweep reclaims the
bytes, deciding the same question with nothing else moving. A server with `--result-store`
sweeps its own store every `--sweep-interval-hours` (a day by default), so the reclamation
is not left to an operator remembering a cron line; `haversack cache sweep` does it on
demand, and a deployment that needs the bytes gone by a deadline sets the interval to it. The per-delete purge, its
scan limit, its freshness margin and its coarse-clock wait are all deleted - about 120
lines, and the hardest remaining reasoning in the module.
