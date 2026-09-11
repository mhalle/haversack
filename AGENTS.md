# haversack — working notes

The README is the user guide and `SERVER.md` the server guide; this file is what an agent
needs that neither says. It is TRACKED, and it went 292 lines stale once because the
working copy that gets edited in a session is `CLAUDE.md`, which is excluded from the repo
(`.git/info/exclude`) - so everything learned about the 0.8.0 engine work existed only on
one machine. Keep the two in step: whatever a session learns belongs here before the
branch lands.

## What this is

Medical-image segmentation: nnU-Net-family models (TotalSegmentator, MOOSE, MRSegmentator,
stock nnU-Net folders) and other engines (FastSurfer, SynthStrip, VoxTell, MONAI bundles) on
MPS / CUDA / CPU, as a library + CLI + REST server, with the same job protocol deployable to
Modal. Split out of the `medseg` workspace on 2026-09-02 (formerly `nnseg`); the product path
is torch. MLX is oracle-only and this package never imports it.

## Where the rest lives (this repo has no design record of its own)

- `../medseg/AGENTS.md` — **read before any geometry, backend, resampling or orientation
  change.** Its "Established facts — don't relitigate" list is the reason things are the way
  they are. `../medseg/docs/` holds the design docs that docstrings here cite as
  `docs/<name>.md` (torch-port-design-notes, backend-decision, resampler-parity-finding,
  slicer-modal-design, segmentation-storage-and-duckn, monai-bundles).
- `docs/` here: `dependency-discipline.md` (enforced by `tests/test_layering.py`) and the five
  `ranked-*.md` records with measured numbers.
- Sibling repos pinned by git tag in pyproject: `../rankfield` (ranked encoding, format 0.4,
  `docs/format.md`), `../duckn` (store metadata), `../synthstrip-torch`, `../fastsurfer-lean`.
  To develop one: `uv pip install -e ../rankfield` **after** `uv sync`, which otherwise
  reinstalls the tagged release. Bumping a tag means pyproject `[tool.uv.sources]` **and**
  `.github/workflows/tests.yml` (CI hand-lists the git URLs) - no longer on trust:
  `test_ci_pins_the_same_git_refs_as_pyproject` fails when the two disagree on repository
  or ref. rankfield and duckn are both at v0.3.2.
- Oracle: `../medseg/nnunet-inference-mlx` at `main` ≥ `40ebe55`, plus its frozen fixtures.
- `upstream/*` clones live in medseg and are other people's repos. Never push to their `origin`.

## Environment and commands

```bash
uv sync --extra torch --extra test --extra modal --extra serve --extra duckn   # the dev env (.venv)
uv run pytest -m "not slow" -q                                                  # fast suite, ~780 tests
uv run pytest tests/test_layering.py -q                                         # import discipline
uv run --no-project python -c "import sys, haversack; assert 'torch' not in sys.modules"
UV_PROJECT_ENVIRONMENT=.venvs/synthstrip uv sync --extra synthstrip --extra serve   # engine venvs
uv sync --extra fastsurfer   # fastsurfer installs into the main env; not synced here today
```

- `--all-extras` does not resolve: `synthstrip` (numpy<2) conflicts with core and `duckn`.
- `.venv` today has torch, serve, modal, duckn/rankfield/zarr, cc3d. Not FastSurferCNN,
  synthstrip_torch, or triton (CUDA-only). Engine tests importorskip.
- Engines are pinned OFF by an autouse fixture (`conftest.py`); a test that wants one sets
  its `HAVERSACK_*` enable var.
- CI (`.github/workflows/tests.yml`) installs CPU torch by hand, not `uv sync`; `PYTHONPATH=src`.
- `uv.lock` and `data/` are gitignored on purpose. `src/haversack/data/*.json` manifests are
  tracked and ship in the wheel (`artifacts` in pyproject) — do not let a `data/` rule swallow them.
- Modal: `haversack modal deploy`; tear deploys down after smokes
  (`uv run --no-project modal app stop haversack-serve --yes`); redeploy does not preempt warm
  old-code containers. Never compare timings across Modal containers.

## Architecture (src/haversack, ~17k lines)

Two layers, enforced by `tests/test_layering.py`:

- **Kernel** — `grid mapping tables restore resample shuffleup reference backends/` — torch +
  numpy only; knows nothing of tasks, files, weights. scipy is call-time only in `resample`.
- **Pipeline** — `io preprocess frame network pipeline segmenter tasks ecosystems values
  envelope weights weights_fetch trainers result cache job progress cli fetchlib filelock`.
  `fetchlib` is the ONLY place a URL is opened (named agent, token stripped on a cross-origin
  redirect; tests mock `fetchlib.urlopen`); `tools/zippeek.py` is its counterpart for the
  `--no-project` generators. `filelock` is the portable install lock (fcntl / msvcrt).
  `attribution` + `data/attribution.json`: license / group / citations per ecosystem and engine,
  merged with each manifest's per-task facts. Every entry was READ from the project's own
  README/LICENSE (never from memory) and every PMID resolved through PubMed from the DOI -
  keep it that way when adding a catalog; `test_attribution.py` fails if one ships without.
  Inputs: `DataSource.describe_input()` per source (metadata only, never downloads) ->
  {origin, license, cite}; `sources.fetch_recording_origin` (the ONE fetch door for serve,
  Modal and `get`) adds the bytes' digest + DICOM UIDs and writes `.input.json` beside the
  fetch; `jobpolicy.input_records` reads it back into `provenance.inputs`.
- **Wire** — `serve` (FastAPI + LocalExecutor + sqlite `jobstore`), `modal_app` (same protocol,
  one `@app.cls` worker per engine: nnU-Net's in `modal_app`, every other engine's in
  `engines/modal_<engine>.py`, collected by a composer that iterates the registry), `client`
  (RemoteClient), `schemas` (pydantic: one declaration → JSON Schema, submit validation,
  OpenAPI), `sources` (idc/tcia/openneuro/zenodo/hf), `content` (content-addressed inputs),
  `preview`, `statistics`.
- **Engines** — `engines/registry.py` is the static ecosystem→engine map (ts, moose,
  mrsegmentator, dentalsegmentator, totalvibe, custom → nnunetv2; fastsurfer, synthstrip,
  voxtell, monai). Deliberately NOT a plugin system, and the reason is now in the module
  docstring: the torch-free import rule, Modal resolving `@app.cls` at import, and engines
  living in conflicting environments each rule discovered plugins out on their own.
  **Adding an engine really costs 7 files / 18 edit sites** to reach a green suite
  (schemas, the engine module, registry ×5, ecosystems ×2, modal_app ×4, pyproject ×3,
  attribution ×2) plus README/SERVER/CI/CHANGELOG. The old "one row plus a worker class"
  line here was wrong by about 4× — measured 2026-09-08 by an agent that actually added one.
  That evening `e389021` made the four modal_app edits ONE new file instead, the adapter
  `engines/modal_<engine>.py` (image, `@app.cls` worker, `ENGINE`, `WORKER`), and modal_app
  takes no edit: 15 sites by arithmetic, not re-measured. Copy a sibling adapter whole: its
  last line, registering itself into `ENGINE_WORKERS`, is the half of the import handshake
  that runs in a worker container (see the sixth Modal round below).
  **The rule that keeps it from growing: when adding an engine needs an edit somewhere new,
  add a FIELD to the Engine row, not a branch there.** `dist` (distributions, for
  `/v1/version`), `label_names` (a thunk, for engines whose labels are in their own
  namespace), and `cache_store` (subdir + env override, so `cache usage`/`clean` can see it)
  all exist because a consumer was keeping its own copy and the copies drifted.
  `tests/test_engine_completeness.py` is the checklist — one obligation per test, each
  failing with the step it found missing. **Read its docstring before trusting it**: its
  first version was substantially hollow (a broken sixth engine passed the whole suite),
  and every rule in it now kills something a reviewer got past.
- **Adding a catalog of nnU-Net weights** = a `ZipManifestEcosystem` subclass (data only:
  name, description, bucket, MANIFEST, generator) + a module-level `*_MANIFEST` path constant
  + a `tools/gen_*_manifest.py` using `tools/zippeek.py` + one line in `default_ecosystems()`
  + the ecosystem set in `tests/test_ecosystems.py` (hardcoded on purpose). No packaging
  change: `pyproject.toml` already globs `src/haversack/data/*.json` into the wheel. The checkpoint is the spec: the manifest holds only
  url, folder, tag and a digest, never labels. Override `_unpack_into` only if the zip has
  no `Dataset<id>` parent, and `spec` only to read a fact the checkpoint states somewhere
  `from_model_folder` does not look (TotalVibe's per-model orientation).
- **Adding a data source** = a `DataSource` subclass, or `ArchiveReadingSource` + one
  `resolve()` to get `!member` zip-by-Range for free, or `ObjectStoreSource` (obstore: `s3:`,
  `gs:`; one object, `!member` over `get_range`, or `<prefix>/` for a whole series) + one
  line in `default_sources()`.
  Strict fullmatch identifier, and the host chosen by us and never by the caller — that is
  the SSRF boundary and it is not negotiable. Prefer a pinned identity (a commit sha, a
  version-pinned uuid); where the repository offers none, `tcia:`, `s3:` and `github:` show
  the alternative — say plainly in the docstring that the identity is not cache-grade, so a
  stale cached result is a known cost rather than a surprise.
- **Ranked store** — `ranked` (shim over rankfield + `RankedSpec`/`emit`/distance/junction
  caches), `ranked_output`, `ranked_build`, `ranked_store`, `ranked_restore` (the STORED
  form is rankfield's: `rankfield.store.read_parts`/`array_geometry` and its version gate,
  never restated here - two copies of it drifted in one day), `duckn_io`,
  `view`. INTERNAL and undocumented (not README, not CHANGELOG, not `--help`); `view` and
  `restore` are unlisted CLI subcommands. Callers are `tools/ranked_*.py`, which NO test
  imports (they are `--no-project` scripts, several Modal- or GPU-only): a rankfield API
  change is invisible to the suite, which is how the 0.3.0 `Geometry` migration left one
  call site raising for a day. `tests/test_tools_rankfield_api.py` reads them instead -
  every rankfield name must exist and every call must bind against the INSTALLED signature
  (the sibling working tree in a dev clone, the pinned tag in CI). It is static only: it
  says nothing about a call whose meaning changed while its shape did not.

Key entry points: `pipeline.segment()` (the run), `Segmenter` (policy + warm `ModelCache`),
`tasks._resolve_spec` (torch-free task resolution used by `describe()` and the server),
`cli` (click since 2026-09-11: `_command_line()` builds every command, each handing what it
parsed to a `_cmd_*` function, and `_run` returns that function's status).

## Rules the tests enforce — keep them

1. `import haversack` pulls no torch and no pydantic. Torch-pulling exports go in `_LAZY` in
   `__init__.py`, never eager. Subpackages count (`backends/` leaked once).
2. Heavy imports are call-time on the light path. Only `network pipeline preprocess resample
   restore shuffleup` import torch at top level.
3. `describe()` / `/v1/tasks/{task}` stays in torch-free modules (`tasks`, `weights`, `values`,
   `errors`, `schemas`).
4. pydantic stops at the wire (`schemas`, `serve`, `registry`): never in value types or the kernel.
5. The lean install (README "Lean install": `--no-deps` + numpy SimpleITK pydantic click tqdm
   httpx obstore) must run `haversack tasks`, `remote`, `modal deploy`; `segment` must fail
   in one line naming what to add. A test blocks torch/nnunetv2/scipy/skimage/duckn to prove it.
6. The default path `segment IN -o labels.nii.gz` imports none of duckn/zarr/rankfield.

## Geometry facts — don't relitigate, don't reintroduce

- Canonical orientation is **RAS** for the TotalSegmentator lineage. LPS mirrors left/right;
  it was a real bug three times. Stock nnU-Net readers (`SimpleITKIO`, `NibabelIO`) do NOT
  reorient — only `*WithReorient` do — so a native model gets its stored axis order
  (`io.reader_reorients`). MRSegmentator forces LPS via `TaskSpec.orientation`.
- Resample convention: `"corner"` + no crop for TS lineage, `"center"` + crop-to-nonzero for
  nnU-Net-native. Getting it backwards is silent (0.995 → 0.888 Dice). Anti-aliasing is
  a distribution shift; opt-in only. Cubic (order 3) forward resample matches training.
- Normalization is per model. A union task shares the resampled grid, never the normalized
  array — `pipeline.predict_into` has a tripwire; keep it.
- Linear restore of the **deficit** field, not margin; nearest hides interpolation bugs.
- MPS: fp16 works because transposed convs are rewritten as 1×1×1 conv + depth-to-space
  (`ShuffleUp3d`, bit-exact). batch_size 1 is fastest on Apple silicon. MPS allocator is
  capped at the recommended working set because past it Metal returns zeros silently.
- Every run holds a per-device lock (`job.device_lock`): `TorchModel` loads folds in place.
- duckn `centering`: `cell` for acquired data, derived from the resample rule for model grids
  (corner rule → `node`). Never hardcode it. Use duckn as a library, not hand-written dicts.
- Any change to orientation, transpose or resampling needs a bit-exactness check against a
  known reference before it is believed. Geometry bugs here are silent, not loud.

## Server semantics (the three rules)

Token computes, anonymous reads. Plain GET is a read; `Prefer: wait=N` / `respond-async` is
the intent to compute. `Cache-Control: no-cache` recomputes and republishes under the same key.
Results are keyed on content identity + task + options + weights versions + `serve.CACHE_EPOCH`
— NOT `__version__`, which used to be there and threw the cache away on every release. Bump the
epoch when a build would compute different bytes from the same inputs (resampling, framing,
restore, label mapping, an engine's inference path); never for a release or a server change.
VoxTell refuses cache serving.

**Result-cache lifetimes (2026-09-10).** A result is a directory per generation
(`<key>/g-<gen>/`) behind one `current` pointer, switched by one rename. No cleanup path —
per-key pruning, the ceiling, eviction — deletes what a reader holds (a `.lease` file inside
the directory it was handed, touched on every resolve, valid `GENERATION_GRACE_S`) or what a
writer may still be publishing (`<key>/.writer-<gen>`, flock'd from before its staging exists
until its pointer moves; kept and marked `unlocked` where no lock can be taken). A reader's
acquisition (pointer → lease → check) holds `<key>/.lock` shared and reclamation holds it
exclusive, so nothing is decided while a reader is mid-acquisition; the `.reclaim-*` tomb and
second question remain only for entries no lock reaches. Death is proved, never inferred:
only a pruner on the host named in a locked claim may take its lock and conclude anything.
Unknown ownership — an unlocked or foreign claim, unclaimed staging — is protected on every
path, so it can pin an entry until an operator clears it; a failed `put` removes its own
work. Eviction skips what it may not take and evicts the next least recently used instead.
`delete` and `cache clean` are explicit and ignore all of this. On Modal, `cache_get` never
commits, so a lease taken in the API container
reaches worker-side pruning only if the volume commits it some other way — no worse than
before, and not solved.

## Release procedure

1. CHANGELOG entry at the top: `## [X.Y.Z] - YYYY-MM-DD`, prose bullets that say *why*.
2. Bump **both** `pyproject.toml` `version` and `src/haversack/__init__.py` `__version__`
   (0.5.0 and 0.6.1 each missed one of them).
3. Commit `Release X.Y.Z`; tag if asked. Reviews' findings get pinned as tests
   (`tests/test_release_review.py` is the pattern).

## House style

- Comments and docstrings explain *why*, often with the date the decision was made and the
  bug that forced it. Keep that: a bare "what" comment is below the local bar.
- Error messages name the fix (the extra to install, the flag to pass) in one line.
- Deviations from what the user asked (accumulator moved to host, fp16 retry) are never
  silent: `note:` on the CLI, a progress stage on the server, `provenance.deviations` in the result.
- An out-of-memory handler retries AFTER its `except` block, never inside it: until the block
  exits, the traceback pins every frame's tensors, so `empty_cache()` frees nothing and the
  retry runs beside the failed attempt (an fp32 run on a 22 GiB A10, 2026-09-11). Keep only
  strings in the handler. `TorchModel._sliding_window_with_fallback` and SynthStrip's fp16
  retry have this shape; `tests/test_oom_fallback.py` checks it with weak references.
- Tests are `unittest` classes under pytest, with the device matrix fixture in `conftest.py`.
- Don't commit this file, `data/`, `uv.lock`, or model weights. Don't vendor sibling repos.

## Guards added 2026-09-08 — what they cover, so they are not re-litigated

One fact written in two places, and the copies drift, is this repo's recurring defect. Each
of these reconciles TWO INDEPENDENT sources; a check that compares a thing to itself always
passes, which is how the first engine checklist came out hollow.

- `test_ci_pins_the_same_git_refs_as_pyproject` — CI's hand-written installs vs
  `[tool.uv.sources]`, repo and ref both. CI hand-lists because it installs CPU torch.
- `test_tools_rankfield_api` — every rankfield name/call under `tools/` binds against the
  INSTALLED rankfield. `tools/` is imported by no test, which is how the 0.3.0 `Geometry`
  migration left one call site raising. Static only.
- `test_engine_completeness` — the engine checklist (above). Modal obligations are STATIC
  text checks, never `importorskip("modal")`, which deleted them in the per-engine venvs
  where an author actually works. Also: git pins are non-placeholder offline, and resolve
  upstream under `-m slow` (out of CI on purpose — CI installs no engine extra, so it never
  resolves an engine source at all).
- `test_fastsurfer` — the shipped LUT vs FastSurfer's own table (identical on all 78 ids;
  upstream adds only id 0 Background). A verification done once by hand is a snapshot.
- `test_server_docs` / `test_engine_completeness` — SERVER.md's flag list and
  `haversack --help` must name every engine. `--help` had gone two engines stale.
- `modal_app._wiring_problems()` is GONE (`e389021`), with the `_WORKER_CLASSES` name-string
  map and the per-engine flags it guarded: the composer takes each adapter's `WORKER` class
  directly, so neither silent deploy-fatal mistake (a mistyped class name, a forgotten flag)
  can be written any more. What is left is checked at import — an adapter without `WORKER`,
  or whose `ENGINE` is not its filename, raises naming the module — and statically by
  `test_engine_completeness`: an adapter per optional engine, `ENGINE` = filename = registry
  key, `WORKER` bound to an `@app.cls` class, and a composer that iterates the registry.

### Real defects these found (not hypotheticals)

- `ranked_build.named_groups` fell back to nnU-Net's `GROUP_CLAIMS` for any engine without
  its own, so voxtell/monai/synthstrip stores emitted `g_lungs` with `exhaustive=True` — an
  anatomical guarantee TotalSegmentator makes and they do not. `claims_for` returns nothing
  now for an engine that declares nothing.
- `/v1/version` hand-listed engine packages and had missed voxtell and monai since each was
  added — the endpoint whose job is to say which rev is running, silent about the one engine
  pinned to a git rev for exactly that reason.

## Known-benign lint, don't "fix" without reading this

CI runs NO linter and there is no ruff config, so `ruff check` uses defaults: 192 findings
package-wide, 113 of them E702 (multiple statements on one line), which is house style.
Wiring ruff into CI needs a config encoding the style first or it fails red on day one.
All 9 F821 are false positives, checked 2026-09-08: 5 in `io.py` and 1 in `weights_fetch.py`
are lazily-imported names used in string annotations (the call-time import discipline; `io.py`
has `from __future__ import annotations`, so they never evaluate) — a `TYPE_CHECKING` block
would silence them. The 3 in `ranked_build` are lambdas closing over `rk_all`/`su_all`, which
are deleted later in the function; the consumer (`junction_sparse` → `_triple_tube`/
`_junction_at`) has no yields, stores neither callable and returns plain arrays, so the
closures never outlive the call. Real hazard class, not a real bug — it becomes one the day
that routine is made lazy. Binding them as lambda defaults would remove both.

## Still unverified, as of 0.10.0

0.10.0 is tagged and pushed with CI green BEFORE the tag (run 34614181942 on `8b3a1ee`: 1143
passed, 20 skipped), as 0.9.1 (run 34594374478 on `5677f36`: 1137 passed), 0.9.0 (run
34585608278 on `50caae6`: 1135 passed) and 0.8.0 (985 passed) were - the whole lesson of
0.7.0 and 0.7.1, both tagged onto a red CI and then deleted from origin. Each of 0.10.0 and
0.9.1 was also smoke-run through `uvx` from its release commit before the tag, the check
that found 0.9.0's bug. 0.10.0's `segment` gave labels byte-identical to 0.9.1's, with the
accumulator pinned to host so that the comparison could not depend on the machine's memory
pressure. What follows is what the work never proved, kept because it is still true.

- **The result cache's locks have run only on APFS.** Writer claims and the entry lock rely
  on flock; nothing has exercised them on a network filesystem or a Modal volume, where a
  lock may not reach across hosts. The design falls back to the move-aside and second
  question there, and never to taking what it cannot prove abandoned - but that fallback,
  too, has only run here.
- **None of 0.9.0's publication and lifetime code has been deployed to Modal** — the last
  deploy was the sixth round, before `6965e68`.

- **Engine venvs have not been run locally** (synthstrip, voxtell, monai, fastsurfer) since
  the shared code moved. All four DID run on Modal (see below), which exercises the same
  `_execute_job` body but not the per-venv install.
- **CI's environment differs from this one** — it hand-installs CPU torch with
  `PYTHONPATH=src` rather than `uv sync`, and several modules (`jobpolicy`, `fetchlib`,
  `filelock`, `attribution`) are new on that path.
- **`haversack.filelock`'s msvcrt branch has never run on Windows.** Written from the
  documentation; the tests exercise whichever facility the host has.
- Concurrency probes were mostly threads within one process; cross-process coverage is thinner.

### What Modal HAS been shown to do (2026-09-06/08, six rounds, torn down after each)

Cold fetch and cache hit; `no-cache` idle and queued behind a running job; the prefetcher
and read-ahead, including one worker warming only its own engine's jobs; uploads and the
upload pre-read; multi-input with two roles uploaded and two bound to one `s3:` object; all
five engine workers (FastSurfer, SynthStrip, VoxTell, a MONAI bundle, nnU-Net); the path
surface; and `provenance.inputs` carrying an IDC series' collection, licence and citation.
Three pre-existing defects were found and fixed that way - orphaned queued records starving
the prefetcher, the inflight-marker age rule, and the missing engine filter.

Fourth round (2026-09-07) closed the rest: a cancel mid-run (DELETE on a running job returns
cancelled and the record settles there), the anonymous twin under `HAVERSACK_PUBLIC=1`
(serves labels, meta, statistics, preview, HEAD and the listing to nobody in particular;
refuses to compute or mutate), a proxy-auth deployment (every route 401 without credentials,
including with a bogus key/secret, while the twin stays open by design), and a VoxTell prompt
outside the embedding bank - which pulls the ~8 GB text backbone to the weights volume and
finished in 289 s. It also found the image-build defect fixed in the same commit.

Fifth round (2026-09-07) took the last two. With `HAVERSACK_MAX_CONTAINERS=2`: four distinct
jobs ran with two containers genuinely overlapping in time, none of their identities crossed,
and single flight held ACROSS containers - a second submission of an uncomputed key never
started a compute of its own (`started` stayed null) and took the first's result under the
same key. And a worker killed mid-compute (`modal app stop` while the job was in `fetch`,
which leaves no terminal emit) is recovered by `_reconcile_orphans` on the next container's
setup: `[reconcile] 1 orphaned job(s) failed`, the record moving to `failed` with a message
telling the caller to resubmit. That is the code added in fab7706 doing deliberately what it
first did by accident on records a `modal app stop` had orphaned 76 hours earlier.

Sixth round (2026-09-08), after the workers moved into `engines/modal_<engine>.py`: two
defects that every static check and every in-process import passed, found only by
deploying. **`modal deploy` loads `modal_app.py` BY PATH**, under a synthetic module name,
so each adapter's `from haversack.modal_app import ...` executed the file a second time as
a different object and re-entered the composer half-built; `sys.modules.setdefault` before
the composer makes the two names one module (`e8599d6`). And **a worker container imports
the module its class LIVES IN** — the adapter, not modal_app — so there the adapter is the
entry point and the composer met it mid-initialization: every engine worker crashed on
start while the api container stayed healthy, and a submitted job sat in `queued` with
nothing wrong visible from outside. The composer skips an adapter whose
`__spec__._initializing` is set, and each adapter registers itself into `ENGINE_WORKERS` on
the way out (`97d0a0d`). Each has a test that reproduces its import mode — a by-path load
in process, and a subprocess per entry point. Then, deployed: all five workers registered
(Modal does discover a class defined in an imported submodule), a real job completed on
each of the four adapters, and `/v1/version` reported their packages as
unknown-because-remote.

Modal has no untouched PATH this file knows of, but it has untouched CODE: none of the
result-cache publication and lifetime commits after the sixth round (`6965e68` through
`3ec6699`, all released in 0.9.0) records a deploy, so generation-at-a-time publication,
leases, writer claims and the entry lock have run only in the local suite and CI. Beyond
that, what remains is judgment, not coverage: every smoke has been
small and short, so nothing says how the queue behaves under sustained load or how a
multi-hour job fares against the 3600 s function timeout.

## Known open, deliberately

- The two `_emit` functions are NOT consolidated: same name, different jobs (one merges
  persisted state terminal-wins, one pushes SSE snapshots). Merging them would invent a
  duplication. `_prefetch_next` and the inflight markers are still written twice.
- A one-member zip and a loose upload of the same bytes are now one identity; what remains
  is that a `.input.json` written before that change still records the old tree digest. It
  is provenance only - the content digest never reaches `result_key` - and a `no-cache`
  refetch rewrites it.
- `docs/totalvibe-region-names.md` records deferred work: naming TotalVibe's 11 regions,
  which needs deriving from a `ts:total` overlap table, NOT reading them off upstream's JPEG.
- **`segment`'s batch writes two inputs that share a stem onto one output (found 2026-09-11,
  not fixed).** `segment a/scan.nii.gz b/scan.nii.gz --format seg.nrrd -o out/` names both
  `out/scan_<task>.seg.nrrd` and exits 0 with only b's labels there. `get`'s batch refuses the
  second of such a pair, folding names as the filesystem does; `segment` has no such check.
- **A local folder holding several DICOM series is read as its first (found 2026-09-11, not
  fixed).** `io.read_image` asks GDCM for "the" series of a directory, so `segment ./study/`
  segments one series of several without a word and `get ./study/ -o x.nii.gz` converts one.
  `serve` refuses such a folder at upload (`dicom_series_ids`). Refusing in `read_image` would
  cover both, and would also refuse a fetched archive holding several series, which takes the
  same path today.
- **Two provenance digests pin less than they say (found 2026-09-11, not fixed).** A
  detached header (`.nhdr`, probably `.mhd`) is digested alone - changing its data file
  leaves the digest the same - and a directory holding exactly ONE top-level file is
  digested as that file (`sources.sole_file`), which is every bare duckn volume: a zarr v3
  array is one `zarr.json` over a `c/` chunk tree, so the record pins the metadata and not
  a voxel. The demo stores escape only because a `README.md` sits beside their `zarr.json`.
- **A NIfTI cannot hold a slightly tilted series (found 2026-09-11, not decided).**
  `_series_geometry` accepts a slice chord up to 2.56° off the image normal (`|dot| >= 0.999`)
  and gives it a sheared direction, exact in memory and in NRRD; ITK's NIfTI writer squares it
  off with only a stderr warning. On a synthetic 40-slice 2 mm series tilted 1.5°, the NIfTI
  from `get -o` misplaces the worst voxel corner by 1.02 mm (the bare series reader `get`
  used before: 2.04 mm; NRRD: 0). `segment`'s NIfTI labels on such a series carry the same
  squaring. Refusing it would need a tolerance that applies to NIfTI output only.

### Verified on a real filesystem (2026-09-07)

A cache root on a USB stick is an ordinary way to run this, and FAT32 and exFAT both answer
ENOTSUP to `os.link` and are case-INSENSITIVE. Both were mounted (`hdiutil create -fs
MS-DOS` / `-fs ExFAT`) and the whole protocol run on them: the no-hard-link fallback admits
exactly one writer of eight and leaves no staged token, two keys differing only in case stay
apart (which is what the uppercase escaping in `safe_path_component` is for), and commit,
discard and the graveyard rename all work. `tests/test_jobpolicy.py` carries it as an opt-in
test - point `HAVERSACK_TEST_NOLINK_ROOT` at such a mount; the recipe is in its docstring.

## Open on the engine/registry work — deliberately not done (2026-09-08)

- **CI installs no engine extra**, so nothing about an engine is exercised there and no
  engine git source is ever resolved. Adding one is a real cost (conflicting numpy ranges,
  heavy trees); the `-m slow` pin check is the cheap half of the answer.
- **README hand-lists engine families** in three places (`README.md:3/380/400/412`). Only
  `--help` and SERVER.md are pinned by tests.
- **`[tool.uv] conflicts` and the worker image's `uv_sync(extras=[...])` restate
  `Engine.extra`** — the image now in each `engines/modal_<engine>.py`. Omitting the
  conflicts entry breaks `uv sync` for everyone — loud, but only at sync time, never in the
  suite. Note `fastsurfer` is deliberately NOT in a conflicts group (it installs into the
  main env).
- `cache_admin.clean` addresses ONE path per category and `checkpoints` is a CLI category
  name, so only one engine may declare `cache_store`. A test tripwires the second one.

## Hazard when running review agents

An agent told to mutation-test will temporarily rewrite source files and restore them. Do
not edit those files while one is running, and verify findings against `git show HEAD:<path>`
rather than the working tree — a reviewer in this session read a file mid-mutation. Their
findings are worth the run: three agents on the 0.7.0 work found two real defects (a
one-file DICOM series losing its identifiers, a legacy store restoring misplaced instead of
failing) and proved two of my own guards hollow. Three more on the 0.8.0 engine work found a
false anatomical claim shipping in stores, two silent deploy-fatal Modal wiring holes, and
ten holes in a checklist that had just been "mutation-tested" with 8 kills.

**Mutation harnesses lie when they break.** Three runs this session reported every mutant
KILLED without executing the test once: an unquoted `$CMD` that zsh did not word-split, a
wrong test class name (`-q` prints "no tests ran", exit 4), and a shell heredoc passing a
literal `\n` so the anchor never matched. Rules: assert the anchor appears EXACTLY ONCE and
treat a failed apply as a HARNESS ERROR, not a survivor; run the unmutated baseline FIRST and
require it to pass; and prefer killing a mutant that the guard should NOT catch as a sanity
check. Also: mutate the facts the test does NOT check — 8 easy kills said nothing about the
gaps that mattered.

The reverse happens too (2026-09-11): a failing unittest `subTest` prints its parent test
`PASSED` and the failure on a line that STARTS with `SUBFAILED[<name>]`, so a harness matching
lines by their leading node id read two real kills as survivors. Require the exit status to
agree with the parsed outcomes, record which `haversack` was imported from INSIDE the session
(a `-p` plugin's `pytest_sessionstart`, not a separate `python -c`), and prefer separate test
methods to `subTest` where a mutant has to see each case.

## The 2026-09-09 review round — what it cost and what it taught

An external review raised 12 findings; all 12 held under verification, three needed
narrowing, and checking them turned up four more it had missed. Then three adversarial
agents on the fixes found six further real defects and proved **four of seven new guards
hollow** — including one written that same day. Treat "I added a guard" as a hypothesis
until a mutant has died.

**Ask the tool, do not model it.** The collection guard compared filenames against
`python_files`. Pytest has FOUR gates — directory recursion, `collect_ignore*`,
`python_files`, `python_classes`/`python_functions` — so a `tests/kernel/` subdirectory
reinstated the very bug it was written for, one level down, and dropping
`unittest.TestCase` from a class deleted it silently. **Every class in
`test_engine_completeness.py` is collected only because it subclasses `TestCase`** (none
is `Test`-prefixed), so one edit there removes a whole checklist section. It now runs a
real `--collect-only` and compares node ids against what each file declares.

**A filename collision is what the FILESYSTEM merges, not what `==` says.** APFS folds
case AND unicode normalization; exFAT and FAT32 fold case, and a cache root on one of
those is a supported way to run this. `T1.nii` and `t1.nii` are one file here. Anything
deciding "are these two names the same" must fold: `unicodedata.normalize("NFC",
n).casefold()`. This lost DICOM slices in a 32-thread prefix fetch — the survivor was an
interleaving of two objects, not either one.

**A dotfile is invisible to `cache_admin`.** `_entries` skips them at both levels, so
`cache clean` cannot sweep what you hide there and `cache usage` counts its bytes without
counting it as an item. Whoever creates `.staging-*`/`.lock-*` owns removing it — the
next fetch of the same input sweeps the staging, under the lock.

**Test the property, not the outcome, for anything concurrent.** The two-process race
test compared published bytes: it caught only the interleavings that happened to corrupt
(measured 4 runs in 10) and could not distinguish the shipped protocol from one with no
lock at all. Rewritten to assert MUTUAL EXCLUSION from timestamps the source records —
two fetches of one input must not overlap — it kills that mutant 10 in 10. A lock alone
still passes it, so a second test kills a writer mid-fetch; that is what staging is for.

**An AST test that pins a variable name is worse than none.** Requiring `_zip` to contain
`ck = (outer, credentials)` failed identical code spelled `key`, and passed a `_zip` that
kept the literals and then scanned `cache.items()` for `k[0] == outer` — the replay
defect restored. Deleted; the behavioural tests caught what it could not.

**A subprocess test must not REPLACE `PYTHONPATH`.** Here haversack comes from the venv's
.pth; in CI it comes only from `PYTHONPATH=src` with the project deliberately not
installed. Overwriting the inherited value passed locally and failed CI outright. Prepend.

### Review agents can destroy real data

One agent misread the input-cache knob (it is `HAVERSACK_CACHE_DIR`, not
`HAVERSACK_INPUT_CACHE`) and ran `cache_admin.clean("inputs")` against the real
`~/.cache/haversack/inputs`, **deleting 28 entries / 421 MB**. Re-fetchable, but gone.
Tell agents that probes touching cache admin, `clean`, or any `HAVERSACK_*` path must run
with `HAVERSACK_CACHE_DIR` pointed at a temp directory, and have them echo the resolved
root before the first destructive call. Two agents also ran concurrently and one read
files the other had mutated; either serialize them or scope each to disjoint files.

## The 2026-09-10 review round — lifetimes

The external review's second pass reproduced three cleanup defects in the generation work of
`e4e80b4`, and they were one mistake three times: cleanup judged by a timestamp that meant
something else.

- **A timestamp with two meanings protects neither.** A generation directory's mtime moved
  when it was created AND when a reader resolved it, so the ceiling — which must reclaim
  something — could not help taking exactly the generations readers had just resolved. The
  lease is a file of its own now: creation still ages a generation, only a read holds it.
- **Lookup and cleanup ask one marker, through one helper.** A legacy flat read touched the
  key directory while cleanup judged the labels file. The marker lives beside what it
  protects, and `_take_lease`/`_leased` are the only code that knows its name.
- **Age is not liveness.** A quiet writer is not a dead one — one long copy, a suspended
  laptop. Proof is a lock the kernel releases on exit, and only on the host that took it: an
  advisory lock need not reach across machines sharing a volume.
- **A fix can concentrate the hazard it did not look at.** Taking held generations out of the
  ceiling made its remaining candidates the NEWEST, and the newest can be a concurrent
  publication between its rename and its pointer switch; HEAD was safe from that only because
  its ceiling took the oldest. The writer's claim covers that instant and the pruner re-reads
  the pointer. Both tests pass on HEAD: a regression guard is written for what the fix could
  break, not only for the finding.
- **Strictness moves the cost somewhere: find where.** Once unclaimed staging is never
  reclaimed, a failed `put` that left its staging would leave it forever — so it removes its
  own, and a test says so.
- **A second question narrows a check-then-act; only exclusion closes it.** Moving a
  directory aside and asking again still left a move and a move back, and a reader already
  handed its path could open it in between (the review's third pass). Readers now acquire
  under the entry's lock held shared, reclamation takes it exclusive, and the second
  question remains only where no lock can be taken.
- **"Cannot verify" is not "nobody's".** The first version deleted a lockless writer's
  claim, reasoning that an unlocked claim looks like a dead writer's — which left its
  staging looking like nobody's, and eviction, honoring only claims it could verify, took
  the entry. A claim that cannot prove death still proves presence: keep it, mark it, and
  protect it on every path.
