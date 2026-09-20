# Result references: a haversack result as the input of another job

Design note, 2026-09-20. It records decisions reached while working out how an embedding
model (DAMO RADAR; `../feldglas`, `../medseg/docs/radar-idc-validation/EXPLORATION.md`) could
run on haversack. What was READ from the code is marked *(read)*; what is proposed is marked
*(proposed)*; what was BUILT is marked *(built)*, with what building changed and why.

**Status, 2026-09-20: step 1 is built** (`sources.ResultSource`, `labelmap.read_label_map`,
`tests/test_result_source.py`, `tests/test_result_source_modal.py`) and was deployed to Modal
under a throwaway name (see "Verified on Modal"). Steps 2 and 3 are not,
and decision 6 is untouched: nothing in the result-cache format, generations, leases or
publication changed. The section "What building step 1 changed" lists every place the note
below was wrong or silent; "What steps 2 and 3 should know" is for whoever builds them.

## The idea

A job's inputs are data references - `idc:<series>`, `s3:<bucket>/<key>`, an upload's `sha256:` -
bound to a task's declared input ROLES *(read: `schemas.bind_sources`, `sources.DataSource`)*. Every
one of them names data that came from OUTSIDE. Let a reference also name a result haversack itself
computed:

    POST /v1/jobs   task = radar:organs
                    sources = [{role: "field", source: "result:<key of a radar:encode job>"},
                               {role: "mask",  source: "result:<key of a ts.v2:total job>"}]

That one addition makes jobs COMPOSABLE: CT -> segmentation; CT -> token field; (field,
segmentation) -> organ vectors, finding scores, a deviation map. Each step is cached under the rules
haversack already has, and the expensive step (a GPU encode) is never repeated because a cheap one
(re-gating with another ecosystem's labels) changed.

It is not a RADAR feature. The same reference serves radiomics and statistics on (image, mask),
cascades (CADS gates T557/T558 on a brain T553 found; TotalSegmentator's own crop-by-organ models),
MONAI bundles that want a region of interest, registration. Build it as a source; RADAR is its
first customer.

## What already exists *(read)*

- **Multi-input jobs bound by role, never by position** - `schemas.bind_sources` returns
  `[(role, source), ...]` in the model's channel order and refuses unknown, missing and duplicate
  roles.
- **A source is a small class** - `prefix`, a strict `id_pattern` fullmatched before anything else
  (the input-validation boundary), `check()` with NO I/O at every door, `fetch()` on the worker,
  `identity()` giving the result-cache token, `describe_input()` giving origin / license / citation
  for `provenance.inputs`, `forget()` for `no-cache`.
- **A finished job already returns the handle** - `key` (`serve.result_key`: a digest of identity
  x task x options x weights versions x epoch), `links` (the path surface, only for a
  single-source identity with menu options; `resource_links` returns `{}` for uploads and
  multi-source), and `result.outputs = [{"name": "labels", "sha256", "bytes"}]`. Outputs are
  already a NAMED LIST.
- **A cacheless engine exists** - VoxTell's row sets `serve_from_cache=False`.
- **Readers of a cache entry take a lease** and no cleanup path deletes what a reader holds
  (AGENTS.md, "Result-cache lifetimes").

## Decisions

1. **The caller names a result by its `key`: `result:<key>`.** Always present, already returned,
   covers uploads and non-menu options (which have no path form), and pins what a downstream job
   should pin - weights versions and the epoch, not just "total of this series". Identifier grammar:
   the key's own alphabet, fullmatched. No host is involved, so the SSRF boundary is untouched; a
   reference to ANOTHER haversack server is out of scope and should stay out (it would be a
   caller-chosen host).
   *(built: `[0-9a-f]{64}`, plus an optional `@sha256:<digest>` pin - see "What building
   changed", 1.)*

2. **Its IDENTITY is the referenced output's content digest, not the key.** haversack keys results on
   content identity. `Cache-Control: no-cache` republishes new bytes under the SAME key, so a
   downstream result keyed on `result:<key>` could silently outlive the mask it was computed from.
   Resolve at submit: `result:<key>` -> the entry's current generation -> the named output's
   `sha256`; that digest is what enters the downstream `result_key` and what `provenance.inputs`
   records (with the key, the upstream task and its weights versions beside it, and the upstream
   result's own `provenance.inputs` nested or linked, so origin / license / citation of the
   ORIGINAL data survive the hop - `describe_input` is the place). Consequence to design around:
   `DataSource.identity(identifier)` does no lookup today; this source needs the executor's cache,
   so it is constructed BY the executor (`LocalExecutor`, the Modal side) rather than listed in
   `default_sources()`, or the resolution happens in `serve` before the source is asked. Pick one
   place; do not do it in two.
   *(built: a source constructed by the executor; `identity()` still does no lookup - see "What
   building changed", 2 and 3.)*

3. **Which output: the primary one unless named.** `result:<key>` means the result's primary output
   (`labels` today). `result:<key>!<name>` - the separator the archive sources already use for a
   member - selects another named output once there are any.
   *(built; the rule is `sources.select_output`, and the primary output is the FIRST entry of
   `outputs`, which is also the one `serve.etag_of` names.)*

4. **A referenced result that is not there is REFUSED at submit, with the fix in one line** ("no
   result `<key>` on this server - compute `<task>` first"; 404/409 family, never a queued job that
   fails in a worker). Submit-and-wait on the dependency is a later convenience (single flight
   exists), not part of this.
   *(built as 409 - see "What building changed", 5.)*

5. **A role can be a LABEL MAP.** Roles today are image channels. A label-map role needs: reading a
   `.seg.nrrd` WITH its segment names (the consumer gates by structure name, never by label value -
   the family's rule, and the reason rankfield stores exist), the geometry, and the upstream task
   name so a consumer can pick its convention. `io.read_image` is image-only; say where the
   label-map reader lives and keep `segment IN -o labels.nii.gz` importing none of it.
   *(built: `haversack/labelmap.py`; `inputs[].kind` is `image` or `labels` - see "What building
   changed", 6.)*

6. **Caching is wanted, and results need not be label maps.** (The user's decision, 2026-09-20: cache;
   license terms are made available in provenance and following them is the user's - no gate, see
   AGENTS.md "Known open".) A `radar:encode` result's primary output is a token FIELD (17-66 MB,
   recomputable in under a second of GPU, so eviction is harmless and the existing LRU ceiling and
   leases are the right policy - this is a cache, not an archive). So: a task declares its primary
   output's NAME and kind; `outputs` stops assuming `labels`; preview, statistics and the
   seg.nrrd conversions apply only to label-map outputs; the path surface links what exists. This is
   the one part that touches the publication / lifetime code three review rounds went through -
   design it, mutation-test it, keep generations and leases exactly as they are.
   *(not built.)*

## Beside it: a per-request list of light deliverables (decided 2026-09-20)

Previews and statistics are already "eventually-consistent artifacts, rendered after done is
served" *(read: `serve.py`, `artifact_overlap`)*, written into the result's own generation and
advertised in `links` - but WHICH are rendered is a deployment setting (`artifacts=("preview",
"statistics")` on the executor; `HAVERSACK_ARTIFACTS` on Modal). Making that a per-request list
(the deployment's set as default and ceiling) is the better FRONT DOOR for light postprocessing:
the postprocessor runs where the image and the labels are already local, so none of the hazards
below apply, lifetimes are untouched, and the client never handles a key. "Preview off" alone is
worth it. Two rules: a deliverable must NEVER enter the labels' result key (asking for a preview
must not recompute a segmentation), and each deliverable gets its own small identity (result digest
x name x its options x its own epoch) so a change to the statistics code recomputes statistics only.

The split, as decided: **light** deliverables (numpy-only: preview, statistics, conversions, meshes,
distance fields) run in the segmentation's worker from that list; **heavy** ones - a GPU model with
its own image, weights and worker, which is what RADAR is - are separate jobs that take `result:`
references. A "heavy deliverable" named on a segmentation request would only be the server
dispatching that second job for the caller; `result:` is the mechanism either way. RADAR's FIELD
hangs off the INPUT (the encode does not depend on any mask), so it is a job of its own
(`radar:encode`), never a deliverable of a segmentation.

*(step 1 and this, 2026-09-20: nothing here was built, and `result:` is a plain source entry
with a plain role binding - `{"kind": "result", "id": ..., "role": ...}` beside `upload`, `input`
and the hosted kinds - so nothing in its wire shape assumes it is the only way a consumer reaches
a result. What WOULD get in the way of the server itself submitting a follow-on job with a
`result:` input, as built:*

- *The door is in the ROUTE. `pin()`, the role-kind check and the identity tuple are assembled
  in `serve._accept`, interleaved with multipart parsing; `executor.submit` takes an identity
  already built and entries already pinned. A server-side dispatch that called
  `executor.submit` directly would skip all three. It needs that part of `_accept` lifted into
  a function of (task, source entries) -> (pinned entries, identity), raising `RequestError`
  rather than `HTTPException` - `ResultSource.pin` already does.*
- *The server can skip the lookup race entirely: at publication it holds the key AND
  `outputs[].sha256`, so it can write the pinned form `<key>!labels@<digest>` itself; `pin()`
  then only verifies.*
- *WHERE it dispatches matters on Modal: the worker publishes, but only the api container has
  `ModalExecutor.submit` (spawn, inflight markers). Either the worker gains a submit, or the api
  dispatches on observing `done` - and the api's view of the cache volume may trail the
  worker's commit, which `pin` answers with 503, not a refusal: the dispatcher has to retry.*
- *The follow-on's IMAGE input has to be re-nameable by the server. A hosted source entry and a
  stored `input` are (an upload is adopted into the content store under its digest, though the
  store is LRU); per-request source credentials are not - they are never persisted, so a
  follow-on that must re-fetch a credentialed input cannot.*
- *A reference's result is not path-addressable, so the caller reaches a server-dispatched
  job only if the first job's status names it (its id or key). On the local server the bounded
  queue can also refuse it (`QueueFull`), which a caller-submitted job sees as a 429 and a
  server-dispatched one has nowhere to report.)*

## Order of work

1. **The `result:` source and a label-map role** - small, general, no engine. Testable with two
   existing tasks and a test double that consumes (image, mask). *(built, 2026-09-20)*
2. **Non-label primary outputs** - medium; the cache / serve change in decision 6.
3. **A RADAR engine, two tasks, off by default** - the engine checklist (7 files / ~15 edit sites,
   `tests/test_engine_completeness.py`). `radar:encode` (ct -> field; GPU, or MPS: encoder-only fp16
   ran in 3.3 s / 4.8 GB on a 16 GB M2 and reproduced an L40S field at token cosine 0.999998 -
   `../feldglas/tools/radar_encode_local.py`) and `radar:organs` / `radar:map` (field + mask ->
   vectors, 146 finding scores, a map volume; CPU, numpy). The engine is THIN: haversack supplies
   fetch, IO, device, jobs and Modal; `feldglas` supplies preprocessing, the encoder call, gates,
   the head and the queries. It needs a structure -> organ-query table per segmentation ecosystem
   (RADAR pools under one of 36 organ queries; the study's table for `ts.v2:total` is
   `../medseg/docs/radar-idc-validation/results/probes/ts_total_to_radar36_lut.json`). Gating by
   haversack's labels means RADAR's decoder is never run - which is also what makes fp16 on MPS work.

Step 1 is worth doing whatever happens to RADAR. Steps 2 and 3 wait on a second encoder behind
feldglas's contract if the "embedding result" is to be designed against more than RADAR's quirks.

## Hazards the builder should expect

Each is followed by what step 1 did about it *(built)*.

- **Modal: a reload hides ITS volume from the container's other threads** (AGENTS.md, 2026-09-19).
  A worker's `fetch` of a `result:` reference reads the CACHE volume, possibly on the prefetch
  thread. It must follow the rules those fixes established (`_cache_view`, `_vol_lock`, local copies
  made under the lock, a miss believed only from a view newer than the request) or it will
  reintroduce the 410s and FileNotFoundErrors they removed. Smoke it with a deep queue, not one job.
  *(built: `modal_app._worker_result_entry` holds `_vol_lock` from the reload through the lookup
  AND the source's copy; `_api_result_entry` is `cache_get`'s own lookup plus the confirming
  reload, 503 when none takes. The prefetch thread never fetches a reference at all -
  `jobpolicy.prefetchable` excludes the kind - but the fetch takes the lock itself, so that is
  an economy and not what keeps it correct. Reproduced with the volume doubles: thirty jobs
  beside committing and reloading threads; the unlocked mutant fails that test 8 runs in 8.)*
- **A lease taken in the API container does not reach worker-side pruning on Modal** ("known open"):
  the window between resolving a reference at submit and reading it in a worker is exactly that.
  Re-resolve in the worker, compare the digest with the one the key was built from, and fail the
  job with "the referenced result changed" rather than compute from other bytes.
  *(built: the digest travels INSIDE the identifier, the worker's fetch resolves again and
  compares, and what it copied is hashed as well - a copy torn by a reload, or an entry whose
  result.json and labels disagree, fails the job too.)*
- **`resource_links` treats any single identity containing `:` and not starting with `sha256:` as
  path-addressable.** A single-input job whose identity is a result digest would get a
  `/v1/result/<digest>/<task>/...` path - decide whether that is wanted; if not, exclude it where
  the URL grammar is written (that function, nowhere else).
  *(decided: not wanted. The hazard was live already in another form - see "What building
  changed", 4.)*
- **`no-cache` on a downstream job** asks every source to `forget()`; for this source that means
  re-resolving the reference, NOT recomputing the upstream.
  *(built: the source remembers nothing, so there is nothing to forget - `pin` resolves at every
  submit and `fetch` at every fetch. A test counts the producer's runs across a `no-cache`
  consumer: none.)*
- **Cycles cannot occur** (a key exists only after its result does), but chains can be long;
  provenance should nest by reference, not by copying whole upstream records each hop.
  *(built FLAT rather than nested - see "What building changed", 7.)*
- **One fact in two places drifts** (the repo's recurring defect): the identifier grammar, the
  output-name rule and the digest-as-identity rule each get ONE home, and a test that reconciles
  two independent sources - not a check that compares a thing to itself.
  *(built: the three fragments of the grammar are three constants in `sources.py`, checked
  against a key the real `result_key` minted, a digest the real `digest_file` took and the name
  the real `result_payload` wrote; the output-name rule is `select_output`, checked against
  `etag_of`; the identity rule is checked against the UPLOAD door, which hashes the bytes as
  they stream and knows nothing of references. The first mutation run found one of these hollow:
  the "no host can be spelled" test inserted a character, so every case was refused for its
  length and a key alphabet widened to `[0-9a-f./]` passed. It substitutes at constant length
  now.)*

## What building step 1 changed (2026-09-20)

1. **The grammar has a third part: `result:<key>[!<name>][@sha256:<digest>]`.** The note had two
   forms. The worker has to compare what it reads with the digest the job was keyed on, and
   `fetch(identifier, dest)` is handed nothing but the identifier - the series-cache key is
   `<kind>:<id>` and that is all that reaches a worker. So the digest travels in the identifier:
   at submit `pin()` rewrites `result:<key>` to `<key>!labels@sha256:<digest>`, always spelled
   whole. Three things follow for free: the series-cache entry is content-pinned (a republished
   upstream is a different entry, so a stale staged copy can never be served for a new pin),
   `identity()` is a pure function of the string, and a caller may send the pinned form itself
   ("this result, but only if it is still these bytes" - 409 `result_changed` otherwise). ONE
   grammar rather than a wire form and an internal form, which would have been the same fact
   written twice.
2. **`DataSource` grew two hooks rather than `serve` growing a special case.** `pins_at_submit`
   (tested `is True`: sources are duck-typed and a Mock's attributes are truthy) and
   `pin(identifier, kind)`, asked once at submit, off the event loop; every repository inherits
   the identity function. `path_addressable` keeps a source off the path surface. `serve` never
   parses a reference or reads `outputs` for this.
3. **Resolution lives in the source, constructed by the executor, and in one method**
   (`ResultSource._open`), used by `pin` at submit and by `fetch` in the worker. The alternative
   - resolving in `serve` before the source is asked - leaves a Modal worker, which has a source
   registry and no app, needing a second implementation of the same question. The executor hands
   the source a `reader(key, fresh=...)`, a context manager yielding `(path, result.json)` with
   the path readable while it is open; three exist (local: the leased `cache_get`; Modal api:
   `_read_cache` / `_confirm_cache_absent`; Modal worker: under `_vol_lock`). **A stale view is
   asked first**: it can answer "missing" or "other bytes" wrongly but never "these bytes"
   wrongly, because the digest decides - so only a refusal costs a reload, and the warm path in
   a worker reloads nothing.
4. **The `resource_links` hazard was already live.** With the identity a bare `sha256:<hex>`
   (the same string an upload of those bytes has - so referring to a result and sending its
   bytes are one request and one cached answer, as `{"kind": "input"}` already was), the
   `sha256:` test in `resource_links` happened to exclude it. But that test let `sha256-tree:`
   through: a job over an uploaded DICOM series referred to by digest has been linked to
   `/v1/sha256-tree/<hex>/<task>/...`, which no route serves. The rule is asked of
   `content.is_digest` now, in that function only: no content digest is path-addressable. A
   consequence to know: a downstream result computed from an UPLOAD of the same label bytes and
   one computed from a reference share an entry, so a hit shows the provenance of whichever
   computed first - the trade `upload` and `input` already make.
5. **Refusals: 409, and the message cannot name the task.** `errors.UnresolvedReference`
   (`status = 409`): nothing is wrong with the request's shape and the same request succeeds once
   the state is put right, which is what a pinned task version this server does not run already
   answers. 422 for an output the result does not have (`unknown_output`, naming the ones it
   has) and for a label map bound to an image role (`wrong_input_kind`). On Modal a miss that
   cannot be verified is 503, never a refusal. The note's "compute `<task>` first" is not
   possible: a bare key does not say what made it once the entry is gone, and finding the job
   that did would be a scan of the jobs Dict per refusal (AGENTS.md, "Per-job work on Modal must
   not be O(jobs Dict)"). The message names the key and says to run the job that produces it.
6. **The label-map role.** `inputs[].kind` was always there and always `image`; `labels` is the
   second value (`schemas.label_input`, no `channel`). A reference's output kind is checked
   against the role's at submit - `ts.v2:total` handed its own labels would segment a label map
   as a CT and cache the answer. The reader is `haversack/labelmap.py`, not `io.py`: `io` is the
   image reader and every caller expects a bare image back. It reads the header keys
   `result.Segmentation.save` writes (named once, in `result.py`; they are 3D Slicer's), refuses
   a nameless label map by default and a layered one always. **The upstream task name is not in
   every header**: only the nnU-Net pipeline writes `task` into its provenance; FastSurfer,
   SynthStrip, VoxTell and the MONAI bundles do not. The result cache's `meta.json` has it for
   every engine, so a fetched reference's `.input.json` carries it and the reader asks that
   record first.
7. **Provenance is flat, not nested.** A `result` input record has `origin` {`result`, `output`,
   `task`, `weights`}, the upstream task's `attribution`, and `derived_from`: `results` - every
   earlier hop BY REFERENCE (key, digest, task, weights; a fixed few fields) - and `inputs` -
   each ORIGINAL input once, BY VALUE, with its origin, license and citation. Nesting the
   upstream's records would copy the whole ancestry at every hop; pure references would lose the
   original data's terms the day an upstream entry is evicted, and those are the one thing no
   identifier can recover (a series' license is per series). `license` on the record itself is
   None on purpose: a computed input's terms are not one license, and this does not invent a
   combination. **The upstream weights versions are not in the cache entry**: `meta.json` holds
   identity, task, options - not the weights the key was built from. They are read from the
   upstream result's own provenance (`models[]`, or an engine's `<engine>_version`), best
   effort; what pins them is the key. Writing them into `meta.json` at publication would be the
   clean fix and is a result-cache format change, so it belongs to step 2.

8. **"Fail if the result changed" is too strong; "never compute from OTHER bytes" is the
   rule.** The hazard above says to fail the job when the referenced result changed between
   submit and fetch. Deployed (`haversack-resultref-smoke`, below), twelve consumers were held
   QUEUED while their upstream results were republished or evicted by another container - and
   the warm workers that then ran them finished `done`. Correctly: a worker's view of the
   cache volume is a snapshot, theirs predated the change, it still held the pinned
   generation, the stale-first lookup found it, the digest matched and the copy hashed to the
   pin. They computed from exactly the bytes their keys were built from. A worker that
   STARTED after the change (the ordinary case under scale-to-zero) cannot read those bytes,
   and there the jobs failed by name: 4 `the referenced result changed`, 6 `no result <key>`.
   No job computed in a container that started after the change, and none failed any other
   way. The local server has no stale view - its lookup follows the `current` pointer - so
   there a changed result always fails the job, though the lease taken at submit keeps the
   pinned generation on disk for `GENERATION_GRACE_S`: a lookup BY DIGEST could still serve
   it, but that is new result-cache API and so step 2's to decide.

## Verified on Modal (2026-09-20, `haversack-resultref-smoke`, torn down)

A throwaway app under its own name (its three volumes and jobs Dict deleted after; the global
`haversack-weights` mounted and never written; no GPU), with a CPU double engine: a producer
writing real `.seg.nrrd` results and an (image, mask) consumer reporting per-structure means
by NAME, checked against means computed locally from the same volumes. 29 of 29 checks on a
clean deployment: the refusals against a real volume (409 `result_missing` from a verified
miss, 422 for a URL, an unknown output and a label map on an image role, 409 for a pin on
other bytes); twenty consumers submitted at once over five upstream results, all done and all
correct, with producers and consumers in different containers; the pinned spelling a cache
hit; a single-input label-map task; and the held-open window of item 8. The tail of the app
log (about a hundred lines) had no traceback, no refused reload and no FileNotFoundError, and
twelve artifact overlaps placed beside twelve fetches. Not a full log audit, and no timing
was compared.

**It found a defect every local test had passed.** `serve._accept`'s upload branch has a local
named `declared` (the upload's `part`); the role specs rode in under the same name, so an
UPLOAD bound before a reference - upload a CT, refer to its segmentation: the commonest
request there is - turned them into `None`, the mask's role read as an image, and every such
job was refused `wrong_input_kind`. The local tests had only ever put a hosted source or the
reference first. Renamed `role_specs`; `test_an_uploaded_image_and_a_referenced_mask` pins
both source orders and is the only test that fails with the collision put back.

## Adversarial review (2026-09-20, one agent in its own worktree, time-boxed)

It could not make a job compute from bytes other than the digest in its key - through any
ordering of sources, a client-sent pin, two roles on one reference, the grammar (uppercase,
Unicode digits, trailing newline, a 400 KB identifier), or the volume doubles - and found no
re-entry of the worker's non-reentrant `_vol_lock`. What it did find, each reproduced, fixed
and pinned by a test that fails without the fix:

- **`label_input(required=False)` declared an optional role `bind_sources` refuses** - one fact
  in two places, only one of them read. The knob is gone: an optional role is a change to the
  binder, not a flag on a declaration.
- **SERVER.md and the CHANGELOG promised more than item 4 above admits.** A reference and an
  upload of the same bytes share an answer, and provenance describes the computation that
  produced it: a hit on an answer first computed from an UPLOAD says "uploaded by the caller"
  and carries no `derived_from`, though the upstream CT was CC BY-NC. Both documents say so
  now, and that `no-cache` recomputes it from the reference.
- **The reader believed somebody else's record.** `_staged_task` trusted any `.input.json` two
  levels above the file with `kind: result` - the false provenance claim `_dicom_facts` was
  rewritten to stop making. It asks only beside a `series/` directory, and only a record whose
  `content.digest` is this file's.
- **`pin()` could mint an identifier its own grammar refuses**: `_judge` checked the stated
  digest with `content.is_digest` (tree digests, uppercase hex), not with the grammar's
  fragment. It is held to `RESULT_PIN_RE` and `OUTPUT_NAME_RE` - the one home.
- **A malformed upstream provenance failed a fetch whose bytes had verified**, and two input
  records stating no identity were folded into one, dropping a license.
- **The reader chose silently** between two segments on one label value (the last won) and
  two values under one name (`mask("kidney")` covered one kidney). Both are refused, as are
  non-integer voxels; names with no `LabelValue` say so.
- **`_stated_weights` read any `*_version` key as weights** - `haversack_version`, a nested
  value. Plain strings under a real name only.
- The hash-mismatch message now says what to do when it repeats (recompute the upstream).

Sixteen mutants survived the first guards - above all that nothing tested `pin` runs OFF the
event loop (the 2026-09-19 freeze class), that terms travel a chain whose middle never touches
the original input, the worker's retry over refused reloads and its sleeping outside the lock,
and the generic output-kind check step 2 relies on. Each has a test; 67 mutants are killed on
the final tree, and two that survived were equivalent mutants that exposed a redundant
condition in the leaf dedupe, since simplified. NOT reached by the review: concurrent fetches
of one pinned reference from real threads, a `/v1/segmentations` listing holding digest
identities, and restart recovery of a persisted queued job with a `result:` source.

## What steps 2 and 3 should know

- **`cache_get` hands back ONE path: the primary output's file.** `result:<key>!<name>` parses
  for any name, and `select_output` finds any listed output, but a non-primary output is refused
  `unsupported_output` because nothing says where its file is. Step 2 has to make the reader
  protocol hand over a per-output path (or the generation directory plus a `file` field per
  output); the one place to change on this side is `ResultSource._judge`.
- **An output needs a `kind`, and the check is already generic.** `pin(identifier, kind)`
  compares the role's declared kind with `outputs[].kind`, defaulting to `labels` when an entry
  says nothing (every entry published so far). A token field is a third kind in
  `schemas.INPUT_KINDS` and a `field` reader beside `labelmap`; nothing in `serve` changes.
- **The api container mirrors the WHOLE generation to resolve a reference.** `_read_cache`
  copies every file of the generation into `MIRROR_ROOT` (it was built to serve them). For a
  label map that is a few MB, once per generation. For a 17-66 MB field it is a copy made only
  to read two JSON files: step 2 should give the api a metadata-only lookup, and
  `HAVERSACK_API_MIRROR_GB` (default 2) is what bounds it until then.
- **A worker stages a copy per pinned reference in `/dev/shm`** (the series cache,
  `HAVERSACK_SHM_CACHE_GB`, default 8) and hashes it: about 0.1-0.2 s for 66 MB. Fine; worth
  knowing when sizing a cohort that gates one field under many masks.
- **The names in a `.seg.nrrd` are the structures PRESENT in that volume**, not the task's whole
  table (`Segmentation.save` writes a segment per present label). A consumer that must tell
  "absent from this scan" from "this task never segments it" - RADAR's structure -> organ-query
  table does - needs the full table, which is `names` in the entry's `result.json` and not in
  the file. The cheapest carrier is the record `fetch` already leaves beside the bytes.
- **Compare grids with a tolerance.** A CT that came through NIfTI (float32 header: 0.8 mm reads
  0.800000012) and its mask through NRRD (double) do not have equal geometry bit for bit. Found
  by the consumer double on its first run.
- **The first declared input is what previews and statistics render against**
  (`serve.reference_input`), so a consumer declares its image first; a task whose only input is
  a label map gets a preview of the mask over itself unless step 2 says otherwise.
- **`RemoteClient.submit` is single-input.** `haversack remote submit result:<key> --task ...`
  works for a task that takes one label map; an (image, mask) job has to be posted as a `source`
  list until the client learns roles.
