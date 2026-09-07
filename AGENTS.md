# haversack — local working notes

Local to this clone (`.git/info/exclude`), never committed. The README is the user guide and
`SERVER.md` the server guide; this file is what an agent needs that neither says.

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
- Sibling repos pinned by git tag in pyproject: `../rankfield` (ranked encoding, format 0.3,
  `docs/format.md`), `../duckn` (store metadata), `../synthstrip-torch`, `../fastsurfer-lean`.
  To develop one: `uv pip install -e ../rankfield` **after** `uv sync`, which otherwise
  reinstalls the tagged release. Bumping a tag means pyproject `[tool.uv.sources]` **and**
  `.github/workflows/tests.yml` (CI hand-lists the git URLs).
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
  one `@app.cls` worker per engine), `client` (RemoteClient), `schemas` (pydantic: one
  declaration → JSON Schema, submit validation, OpenAPI), `sources` (idc/tcia/openneuro/
  zenodo/hf), `content` (content-addressed inputs), `preview`, `statistics`.
- **Engines** — `engines/registry.py` is the static ecosystem→engine map (ts, moose,
  mrsegmentator, dentalsegmentator, totalvibe, custom → nnunetv2; fastsurfer, synthstrip,
  voxtell, monai). Adding an engine = one Engine row + one extra + one `[tool.uv.sources]`
  entry + a Modal worker class.
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
  caches), `ranked_output`, `ranked_build`, `ranked_store`, `ranked_restore`, `duckn_io`,
  `view`. INTERNAL and undocumented (not README, not CHANGELOG, not `--help`); `view` and
  `restore` are unlisted CLI subcommands. Callers are `tools/ranked_*.py`.

Key entry points: `pipeline.segment()` (the run), `Segmenter` (policy + warm `ModelCache`),
`tasks._resolve_spec` (torch-free task resolution used by `describe()` and the server),
`cli._run` (hand-rolled argparse, not typer decorators).

## Rules the tests enforce — keep them

1. `import haversack` pulls no torch and no pydantic. Torch-pulling exports go in `_LAZY` in
   `__init__.py`, never eager. Subpackages count (`backends/` leaked once).
2. Heavy imports are call-time on the light path. Only `network pipeline preprocess resample
   restore shuffleup` import torch at top level.
3. `describe()` / `/v1/tasks/{task}` stays in torch-free modules (`tasks`, `weights`, `values`,
   `errors`, `schemas`).
4. pydantic stops at the wire (`schemas`, `serve`, `registry`): never in value types or the kernel.
5. The lean install (README "Lean install": `--no-deps` + numpy SimpleITK pydantic typer tqdm
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
Results are keyed on content identity + task + options + weights versions + `__version__`
— bumping `__version__` invalidates every cached result. VoxTell refuses cache serving.

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
- Tests are `unittest` classes under pytest, with the device matrix fixture in `conftest.py`.
- Don't commit this file, `data/`, `uv.lock`, or model weights. Don't vendor sibling repos.

## `protocol-and-shared-core` — merged to `main` 2026-09-06 (fast-forward, 851d240); NOT pushed. Trim this section once the unverified items below are closed.

Six commits + a CHANGELOG entry, on top of `main`, which itself carries 5 unpushed commits
from `sources-and-catalogs`. **Nothing is pushed to origin.** No version bump; everything
sits under `[Unreleased]`.

What it does: one executor contract derived from `create_app`'s AST; a new `jobpolicy`
module holding the decisions `serve` and `modal_app` both make; a pinned pre-read; and a
claim protocol where the claim and its owner arrive in one `os.link` instead of two steps.

### Modal smoke (2026-09-06) — DONE; the first on this branch

Deployed twice with `--no-proxy-auth`, torn down after. Exercised: idc cold fetch + cache
hit, s3 no-cache idle, s3 no-cache queued behind a running job (the finding-1 case: now
`refetching (no-cache)`, no `input_refresh_skipped`), a plain job queued behind a long one,
upload, path surface (HEAD/meta/statistics). No tracebacks. Found and fixed a pre-existing
Modal defect: five `queued` records orphaned by a `modal app stop` 76 h earlier starved the
prefetcher (it warms the oldest queued job) and poisoned single-flight for their keys;
`_reconcile_orphans` fails them via `FunctionCall.get(timeout=0)`. After the fix the
prefetcher worked for the first time (`[prefetch] … staged`, next job `fetch cached` /
`read preread`, 35.6 s → 8.4 s).

Second round, all five engines deployed: FastSurfer (95 structures, 33 s), SynthStrip,
VoxTell (`lung`/`heart` on CT), MONAI spleen (bundle installed on the volume), BraTS
multi-input with two roles uploaded and two bound to ONE `s3:` object under no-cache (the
refresh deduped: no `input_refresh_skipped`), and the upload pre-read. Found and fixed: the
prefetch scan had no engine filter, so with several workers each container warmed whichever
job was oldest - the SynthStrip container pre-read the nnU-Net worker's upload. After the
filter the nnU-Net worker pre-reads its own upload (`read preread`, verified live).

Third round (2026-09-07, ade4293): the worker records an input's rights at fetch time and
writes `provenance.inputs` - an `idc:` job carried collection nlst / CC BY 4.0 / the NLST
DOI / IDC's acknowledgment, an `s3:` object and an upload said "not determined" in words;
the path surface's meta.json carries the same. No tracebacks. Re-run after the reshape
(e3919a8): `inputs[i]` = content / origin / license / cite; the worker's tree digest of the
NLST series equals the local run's (`sha256-tree:c1e472…`), so the pin is reproducible.

What no smoke has touched: a VoxTell prompt outside the embedding bank (the 8 GB backbone
download to the weights volume), `HAVERSACK_PUBLIC=1` (the anonymous twin), proxy-auth
deployments (every smoke was `--no-proxy-auth`), a cancel mid-run, and anything on Windows.

### Review round 2 (2026-09-06), fixed in the working tree

- Modal had not received two of ba2e1c1's no-cache fixes (prefetcher skipping
  `refresh_input` jobs; never using a pre-read after a refresh). Both rules now live in
  `jobpolicy.prefetchable` / `take_pre_read`; `modal_app._prefetch_candidate` is the
  scan lifted to module level so it can be tested against a fake jobs dict.
- `_claim` leaked a live claim when the stale-clear raised after the link; `_hb_start`
  spelled `.owner` by hand. Both have behavioral tests.
- Windows: `haversack.filelock` (fcntl / msvcrt) replaces the bare `flock` that silently
  degraded to no lock; ranked_store's lock uses it too. Still never run on Windows.
- Textual guard `test_a_no_cache_job_never_uses_a_pre_read_image` rewritten structurally.

### Already verified — don't redo

900 fast tests; ruff identical to the `main` baseline on every touched file; a real `s3:`
fetch with two concurrent callers downloading once; a real MPS segmentation. Concurrency
probes after the fixes: 0 false "writer is dead" verdicts under 112,276 competing claims
(was 86/3000), and 1 claim attempt / 0% CPU against an unreadable claim (was 6,399 / 95% of
a core). `jobpolicy`'s behavior and the claim/commit paths are mutation-tested.

### NOT verified — this is where fresh eyes are worth most

1. **Modal has never been deployed or smoke-tested from this branch.** Half the
   consolidation exists to keep `modal_app` in step with `serve`, and `modal_app`'s changed
   paths run only under the Modal runtime — a `NameError` there already slipped past the
   whole suite once during this work. Highest-value gap by a distance.
2. **Engine venvs were never run** (synthstrip, voxtell, monai, fastsurfer). Shared code moved.
3. **CI's environment differs** — it hand-installs CPU torch with `PYTHONPATH=src` rather
   than `uv sync`, and `jobpolicy` is a new module in that path.
4. **The no-hard-link fallback** in `SeriesCache._claim` was only exercised by monkeypatching
   `os.link`; never on a real FAT32 or network mount.
5. **`_commit` now raises** where it used to pass, failing a job whose fetch had completed.
   Deliberate and tested, but the interleaving that reaches it is rare — worth a second read.
6. Probes were mostly threads within one process; cross-process coverage is thinner.

### Known open, deliberately

- The two `_emit` functions are NOT consolidated: same name, different jobs (one merges
  persisted state terminal-wins, one pushes SSE snapshots). Merging them would invent a
  duplication. `_prefetch_next` and the inflight markers are still written twice.
- `jobpolicy.purgeable` and `jobstore.reap` disagree when `finished == 0.0`. Unreachable
  (a finish stamp is `time.time()`) and pre-existing.
- `docs/totalvibe-region-names.md` records deferred work: naming TotalVibe's 11 regions,
  which needs deriving from a `ts:total` overlap table, NOT reading them off upstream's JPEG.

### Hazard when running review agents

An agent told to mutation-test will temporarily rewrite source files and restore them. Do
not edit those files while one is running, and verify findings against `git show HEAD:<path>`
rather than the working tree — a reviewer in this session read a file mid-mutation.
