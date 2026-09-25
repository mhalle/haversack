# Changelog

## [Unreleased]

- **Faster submits and finishes on Modal.** Profiled on a deployment with three fresh IDC CTs:
  a submit answered from the result cache copied the result into the api container to read its
  record, and concurrent ones queued behind each other's copies (1-4 s each); it now reads the
  record and checks which artifacts exist in place (0.9-1.0 s end to end, from 2.4-5.4 s). The
  route answers a submit from the record it just wrote instead of reading it back. A worker places
  the job's own scratch copy - the fallback used once a result is evicted - after reporting done
  rather than before (0.7-1.5 s of every job's latency), and the `.seg.nrrd` header no longer
  sorts the whole label map to list the labels present (1 s on a 418 M-voxel map). Results,
  keys and file bytes are unchanged.
- **An input copy is written a slab at a time.** Transcoding a DICOM series or a gzipped NIfTI
  held the volume several times over - 3.3 GB at peak for a 709-slice CT, where Modal's api
  container (which transcodes uploads) has 2 GB. The compressed copy is written a chunk of 32
  slices at a time and checked the same way: 526 MB for that CT, 130-350 MB for six others, every
  chunk byte-identical to the whole-volume path's, at 5-90 % more time (compression is no longer
  parallel across chunks). Other formats and the uncompressed form take the whole path as before.
- **A task's ranked store is a cached result, and holds the task's own field.** `POST /v1/jobs`
  with `kind=rankfield` (`haversack remote submit ... -o <name>.duckn.zip`) computes a task's
  output distribution - the store `segment -o x.duckn.zip` writes - and caches it as labels
  and embeddings are cached: a third result kind, keyed on the task's weights versions and the
  formats the store is written in (`rankfield=rf<rankfield>/seg<duckn seg>/h<rules>`), served as
  a zip by its job and, for a hosted input, by a read-only path
  (`/v1/<source>/<id>/<task>/rankfield.duckn.zip`). No options, no deliverables; a server without
  the `duckn` extra answers 501, and Modal runs the job on the task's own worker - the api,
  nnU-Net and FastSurfer images now carry the `duckn` extra (and SynthStrip's, below). No
  existing key moves.
- **What a store holds changed: the task's field, not its models'.** A cascade's crop stage is
  left out; a multi-model task's parts are composed into one layer (`ts.v2:total`'s five), whose
  winner is the painted label at every model-grid voxel and whose gaps across models are each
  model's painting margin, scaled to match its own field and encoded at clip 16 without a tail
  (`scores: composed`: not one softmax, no probabilities). Its linear restore differs from the
  direct labels at 0.013 % of voxels on a 418 M-voxel CT and scored as well against upstream
  TotalSegmentator (mean Dice 0.9363 against 0.9360). The distance field is computed on the
  task's field, seams between models included. A store written before this restores as it did;
  a multi-layer cascade store is still refused whole (its stages sit on different grids since
  the upstream crop), which the one-layer store no longer is. The ranked store is now
  documented (README, SERVER.md).
- **Cached inputs are kept decoded.** A DICOM series was decoded on every job that read it -
  13 s for a 709-slice CT on a Modal worker, 45 s cold from a volume another container wrote -
  whether or not it had been read a minute earlier. The series cache and the input store now keep
  each image input as its *input copy* instead of its files: the volume `io.read_image` produced,
  written once when the input is stored as a zarr zip (compressed by default - above; with
  `HAVERSACK_INPUT_COPY_COMPRESSION=uncompressed`, one uncompressed chunk read by mapping it, 0.25 s for
  the same CT), with duckn's geometry and the DICOM tags SimpleITK reports (series-level in
  `extensions.dicom`, per-slice in the slice axis' samples, in duckn's `dicom-spec` encoding) -
  voxels identical, geometry identical (within 1e-12 for a tilted series).
  One form per entry: the original exists only while it is transcoded, and nothing downstream is
  handed it. Label maps and anything the reader refuses keep their files; `HAVERSACK_INPUT_COPY=0`
  keeps originals. Results and keys do not move. A fetched input cached before this is fetched
  again once (its entry name now carries the reader version); `GET /v1/inputs/{digest}` gains
  `stored_form` and `stored_bytes`. The content store's raw-NRRD fast copy is gone. pydicom joins
  the `duckn` extra (as a data dictionary: SimpleITK's tag keys to keywords); every Modal image
  that stores inputs carries the extra, and one without it still reads uncompressed copies.
  `docs/input-copy.md` is the specification.
- **The review's second batch** (each pinned by a test that fails on c93b699; 13 of 13 mutants
  killed):
  - The embedding worker makes input copies: its image lacked pydicom (the `embed` extra has
    none), so every cached `idc:` embedding paid the full DICOM read. It also no longer pre-reads
    the next job's image, which the encoder discarded.
  - The anonymous twin reloads its weights volume (throttled) when a key would be `unknown`: a
    task a worker installed after the twin started had every result 404 on the anonymous path.
  - A read-only result-store server reaches rank fields and nnU-Net encoders' embeddings: every
    kind now records what a reader keys it on (only segmentations did), and the reader's versions
    function takes the kind.
  - Each deliverable renders on its own, and one that fails or has nothing to show is said on
    the job (`deliverables_unavailable`), so its link goes: a failed preview used to cost the
    statistics too, and the job kept linking both.
  - Advice the server can follow: a missing deliverable's 404 says a plain submit renders it
    only where a cache hit renders (the local server); on Modal it says to recompute with
    `Cache-Control: no-cache`. A rank field's or an embedding's job artifact 404 says only a
    segmentation renders deliverables. IDC refusals no longer point at a `/v1/resolve` that does
    not exist.
- **Fixes from the 2026-09-25 server review** (four adversarial reviewers, one black-box against
  the live deployment; each fix pinned by a test that fails on 9c69466, 10 of 10 mutants killed):
  - On Modal a submit of a key already computing JOINS that job, as the local server does and as
    SERVER.md says; it used to start a second computation and publish the key twice (seen on the
    live deployment). The in-flight marker is claimed atomically (`modal.Dict.put(...,
    skip_if_exists=True)`), so two api containers cannot both win; a flight is joined only with
    the same source credentials (recorded as a digest, never the token); what a joiner asks
    rendered joins a queued job's list; and a joiner's DELETE releases the flight rather than
    cancelling it for the others.
  - An upload job's input resolves only from a committed store entry: one evicted after the
    submit, being written again by a re-upload of the same bytes, was handed to the job half
    written, and its labels published under the whole content's key. The job now fails naming
    the fix (send the bytes again).
  - `PUT` and `POST /v1/inputs` store off the event loop: the input copy made the store's adopt a
    whole-volume decode, compression and verify, which stalled every other request to the
    container for the length of it.
  - A series the reader fails on with something other than a refusal (SimpleITK's error on
    slices of different sizes) keeps its original, as before the input copy: it had been thrown
    away - a fetched input downloaded again on every job, an upload answered 500.
- **The kind is `rankfield`** (named for rankfield's own object, a `RankField`, as the other
  kinds are named for what they return): `kind=rankfield`, `rankfield.duckn.zip`,
  `GET /v1/rankfields`, `links.rankfield`, `haversack remote rankfields`, and the key tag
  `rankfield=rf...`. It was `ranked` on main for a day; no deployment ran it, so there are no
  aliases. The prose and the modules keep "ranked store" for the duckn zip that holds the field.
- **Rank fields are listed**: `GET /v1/rankfields` (`haversack remote rankfields`,
  `RemoteClient.rankfields`) is the segmentations and embeddings listing for the third kind -
  filters by `identity` and `task`, cursors, newest published first, `links.rankfield` for a store
  with a path, and a key round trip so a store keyed under weights or formats this server no
  longer writes is not offered. The anonymous twin lists them when its operator opts in, as
  the other two.
- **Input copies are compressed by default** (`HAVERSACK_INPUT_COPY_COMPRESSION`, default `zstd`;
  `uncompressed` for the other form): the smallest form measured, at a read of up to about a second
  against the DICOM decode it replaces. The experimental VoxTell and MONAI images cannot read a
  compressed copy (no zarr); a deployment that runs them sets `uncompressed`.
- Input copies may be stored compressed: `HAVERSACK_INPUT_COPY_COMPRESSION=zstd` (zstd
  level 3 through blosc with bit shuffling, in 32-slice chunks). Across seven local datasets the
  copies were 3.1-5.8x smaller (the 709-slice CT 270 MB instead of 837 MB; int32 CTs 4.5-4.9x),
  and a read took 2-3x the mapped one; on Modal the difference was 0.3-0.5 s a read on that CT
  (measured with plain zstd, which reads alike). A cold read from a volume was no faster either
  way, since the volume's own latency swamps the bytes. It is the
  choice where cache room is the constraint - a Modal worker's series cache is RAM - and applies
  only to copies written after it is set. `GET /v1/inputs/{digest}` reports
  `stored_compression`. Both input-copy variables are now forwarded into Modal containers;
  `HAVERSACK_INPUT_COPY=0` had never reached them.
- SynthStrip runs on NumPy 2 and installs beside the other engines. synthstrip-torch 0.1.1 drops
  its `numpy<2` cap, which existed because surfa 0.6.3 (its last release) breaks in `reorient`
  on NumPy 2; it takes surfa at upstream's unreleased fix instead (`8aa4a5f6`). On CPU, NumPy
  2.5.3 with that surfa gives the same distance field and mask bit for bit as NumPy 1.26 with
  surfa 0.6.3. The `synthstrip` extra leaves every uv conflict group, so it no longer needs its
  own environment, and SynthStrip's Modal image takes the `duckn` extra and reads compressed
  input copies. synthstrip-torch is marked unsupported: kept runnable, not developed.
- The DICOM tag converter the input copy uses moved into duckn (`duckn.dicom_tags`, duckn 0.5.2),
  so the dicom extension's encoding rules live beside its specification; haversack's own copy is
  gone. duckn's converter leaves binary-VR values out (SimpleITK's strings of them are not the
  spec's base64), so the input copy's reader version is 2: a copy written by the first version is
  fetched again once. duckn is pinned at v0.5.2 and feldglas at v0.1.4 (which pins the same duckn).
- VoxTell and the MONAI bundles are documented as experimental: off unless a deployment sets
  `HAVERSACK_VOXTELL=1` / `HAVERSACK_MONAI=1` (as they already were - neither has an in-process
  runner), Modal-only, heavy, and not maintained for general use. They showed the engine model
  can carry a free-text prompter and a foreign model zoo; they are not what most users need.
  A VISTA3D evaluation against TotalSegmentator (four CTs, median Dice 0.82-0.92, pelvic
  structures reported on chest-only scans, non-commercial weights) found no case for adding it.

- **A Modal deployment can take the local server's bearer token, and `haversack remote`
  reaches it.** `haversack modal deploy --token T`, or `HAVERSACK_SERVER_TOKEN` in its
  environment, stores the token in the Modal Secret `<app name>-token` (through Modal's API,
  never a command line) and gates the api the way `haversack serve --token` does, in place of
  Modal proxy auth - whose `Modal-Key` / `Modal-Secret` headers the bundled client never
  sent, so until now it could reach a deployment only with no auth at all. Only the api
  function mounts the Secret; no image, worker or anonymous twin holds the token, and a
  container asked for one that finds none refuses to start instead of serving open. The
  deploy prints which auth it chose. Without a token nothing changes: proxy auth, as before.
- **`HAVERSACK_SERVER_TOKEN` is the server side's token variable** - read by `haversack
  serve` and `haversack modal deploy` after `--token`, and by nothing else. A token passed as
  `--token` sits in the process list, where `ps` shows it to every user of the machine, for
  as long as the command runs (a note says so); the variable does not. The client keeps
  `HAVERSACK_TOKEN`, which no server reads, so a token exported for `remote` can never
  become a server's or switch a deployment's auth mode.
- **Cached embeddings are listed, and have a path.** `GET /v1/embeddings` (`haversack remote
  embeddings`, `RemoteClient.embeddings` / `iter_embeddings`) is the segmentations listing
  asking for the other kind: the same token, paging and computed `identity` filter, `encoder`
  in place of `task`, a row kept only when its meta says it is an embedding, and a key round
  trip through the encoder's own versions, so a field keyed under weights the server no longer
  runs is not offered. A row with one source identity has `links.embedding`,
  `/v1/<source>/<id>/<encoder>/embedding.zarr.zip` (`embedding_int8.zarr.zip` for int8): a
  READ door beside the labels' - 200 with an `ETag` and 304, `HEAD`, 202 while a job for it
  runs, 404 `no-store` otherwise, and a 503 rather than a 404 from a stale Modal view. It
  computes nothing, whatever `Prefer` says: an embedding is computed by `POST /v1/jobs
  kind=embed`, whose cache hit costs the same. The anonymous twin serves the same paths.
  Before this a field was reachable only through the job that made it. Modal's
  `weights_versions` and the twin's `weights_fn` take the kind, as `submit` already keyed.
- **`{"int8": false}` is the default embedding, not a second one.** The options were kept as
  sent, so an explicit `false` keyed apart from `{}` - the same bytes, computed twice, and
  never found by a path, which asks for the default by `{}`.
- **An encoder's output is an EMBEDDING, and the commands and protocol say so.** 0.13.0 called
  it "encode" everywhere, the word rankfield already uses for the ranked encoding of a
  segmentation's logits - so "the encoder's budget" meant rankfield's and "the encoder
  worker" meant RADAR's, one night apart. feldglas already names the format the embedding
  field. Renamed outright, with no aliases, since no deployment ran the 0.13.0 protocol:
  `haversack encode` is `haversack embed`, `haversack remote encode` is `remote embed`,
  `RemoteClient.encode` is `RemoteClient.embed`, a job is `POST /v1/jobs` with `kind=embed`
  (its status and meta say `"kind": "embed"`), its output is `{"name": "embedding", "kind":
  "embedding"}` stored as `embedding.zarr.zip`, `haversack embed --json` reports it under
  `"embedding"`, `GET /v1/encoders` says `"embeds"`, the extra is `haversack[embed]`, and a
  Modal deployment turns the worker on with `HAVERSACK_EMBED=1` (`EmbedWorker`). An
  embedding's key moves (`embed@epoch` and the kind are hashed into it): fields cached under
  0.13.0 are computed again. The ENCODER - the network, `radar:pretrain`, `ts.v2:total_fast`
  - keeps its name: `haversack encoders`, `/v1/encoders`, `HAVERSACK_ENCODER_WEIGHTS`. On the
  ranked side, haversack's own names say "ranked": `network.ranked_encode_budget`,
  `RANKED_ENCODE_BUDGET_FRACTION` / `_CEILING`, and the provenance field
  `ranked_encode_budget_bytes` (it was `encode_memory_budget_bytes`; no key moves).
- **The ranked encoder's budget counts the memory torch already holds for reuse.** Right
  after a network, `network.ranked_encode_budget` read ZERO on an M2 (`lung_vessels` on a 0.625 mm
  CTPA: the host had 2.6 GiB available, under `device_budget_bytes`' 3 GiB headroom, while
  2.5-2.8 GiB of what the driver held was torch's cache of the network's freed
  activations), so every such encode ran one plane a slab. That was harmless while
  rankfield's selection was the cost, and costs 2.09 s against 0.75 s once rankfield's Metal
  kernel makes the selection cheap. `network.reusable_cache_bytes(device)` - the driver's
  holding less live tensors on MPS, reserved less allocated on CUDA, where `mem_get_info`
  also leaves that cache out - is added to the fresh reading; `device_budget_bytes` itself,
  which the accumulator's placement reads too, is unchanged. Same run: the budget is
  rankfield's full 1 GiB default, the peak footprint 5.87 GB as before, and the stored arrays
  identical. With rankfield 0.3.6's kernel the fine stage's encode took 0.75-0.78 s in two
  runs, 14.1 s on 0.13.0; one run in three read 3.7 s while the machine was swapping.
- rankfield is pinned at `v0.3.6` (floor `>=0.3.6` in the duckn and encode extras) and feldglas
  at `v0.1.3`, in `pyproject.toml` and CI's list.
  rankfield 0.3.6 runs its encoder's selection as a Metal kernel on MPS and as one stable sort
  when every class is kept (depth >= K) - byte-identical to its torch path, which stays the
  reference: a K=5 field's encode went 13.2 s -> 0.74 s on an M2, 1.12 -> 0.49 s on an A10.
  feldglas 0.1.3 is 0.1.2 with the same rankfield pin, which uv needs to resolve
  `haversack[encode]`.

## [0.13.0] - 2026-09-23

Embedding fields (`haversack encode`, and encode jobs on the server and on Modal);
TotalSegmentator v3 as its own catalog, and TotalSegmentator's crop and auxiliary-class rules
applied as upstream applies them; `result:` references from one job to another; deliverables
chosen per request; a filtered, paged listing; and a result store that servers can share
through a directory or an object store. **What recomputes, once:** the ts.v2 tasks that crop
(`crop=upstream`) and the three with auxiliary classes (`auxiliary=0`); no other key moves and
`CACHE_EPOCH` stays. **What to rebuild:** the ranked stores of cascade tasks, whose first layer
was misnamed. **Who must change:** a client that reads a ranked store's raw attributes (duckn
seg 0.9, below). A server started without `--result-store` runs as before.

### Embedding fields

- **`haversack encode`: embedding fields, with the weights managed here.** An encoder's token
  lattices are written as a feldglas embedding field (`.zarr.zip`, the `encode` extra), so the
  weights an encoder needs are fetched, pinned and verified by the same program that manages
  segmentation weights, instead of by a second tool. Encoders are named in the task grammar
  (`radar:pretrain`, `ts.v2:total_fast`, `ts.v2:total`; `haversack encoders` lists them), one
  generic path does reading, provenance and writing, and each algorithm family is one small
  module. RADAR's weights (1.6 GB, CC BY-NC-SA 4.0) are pinned to a Hugging Face revision and
  sha256: `weights fetch radar:pretrain` downloads them and `--from FILE` adopts a copy, both
  refusing any other bytes. On a sample CT the tokens are identical to the fields feldglas's
  tools made for the RADAR study, for all three encoders, lattice for lattice. The nnU-Net
  encoders run the encoder only, so `ts.v2:total` encodes in 18 s where the old tool took 55 s.
  RADAR joins the attribution record (its Science 2026 paper, DOI only until PubMed indexes it).
- **Encode jobs on the server.** `POST /v1/jobs` takes `kind=encode` and an encoder name, and
  the job's result is an embedding field (`.zarr.zip`) where a segmentation's is labels: the
  same queue, single flight, result cache and job routes, with `GET /v1/encoders`,
  `RemoteClient.encode`, and `haversack remote encode` / `remote encoders` beside them. A
  field's result key includes its kind, because `ts.v2:total_fast` is a task and an encoder
  and one key for both would serve a label map to a field reader; a segmentation's key
  leaves the kind out, so no existing result recomputes. On the sample CT, both encoders
  served by a local server gave tokens identical to `haversack encode`, and a repeat ask
  was a cache hit with the same bytes. On Modal, `HAVERSACK_ENCODE=1` deploys an encoder GPU
  worker: a smoke deployment encoded the same CT with both families to tokens agreeing with the
  local ones at median cosine 1.000000 and 0.999999 (CUDA against MPS, fp16), fetched RADAR's
  checkpoint once onto its own volume, served repeats from the cache, and encoded an `idc:`
  series (21 of 21 checks). A server fetches an encoder's pinned weights on first use, as it
  does a task's; the command line still wants `weights fetch`.

### TotalSegmentator

- **TotalSegmentator v3 is the `ts.v3` catalog: `ts.v3:total`, `ts.v3:total_fast`,
  `ts.v3:total_fastest`.** Upstream's `total_v3` (Datasets 831-835 at 1.5 mm, 836 at 3 mm,
  837 at 6 mm, release `v3.0.0-weights`) under v2's task names, the catalog carrying the
  version as the naming policy has it; `ts.v2` and every result it has keyed are untouched, and
  a bare `total` now names both catalogs. The registry is generated from TotalSegmentator
  2.18.0's own source by the new `tools/gen_ts_registry.py` (read out of the PyPI wheel, its
  digest checked; upstream's `get_task_config` executed, not restated) and checked against
  every checkpoint's plans and labels by range request. The labels are v2's 117 with value 26
  `vertebrae_L6` in place of `vertebrae_S1`, as the checkpoints and upstream's v3 class maps
  both say. Upstream states no license for the v3 weights (its README lists v2's `total` as
  Apache-2.0 and does not mention v3; the release is a prerelease), and the attribution
  record says that rather than guess. MR Datasets 870-873 are recorded in the weights manifest
  and offered by no task: no upstream task uses them yet.
- **`ts.v3` tiles at upstream's step, 0.8.** TotalSegmentator runs `total`, `total_v3` and
  `total_mr` with a sliding-window step of 0.8 (faster; its own note says 0.001 Dice worse)
  and everything else at nnU-Net's 0.5, which haversack used everywhere. On a CPTAC-CCRCC CT
  on Modal, matching it took `ts.v3:total`'s voxel agreement with upstream from 99.86 % to
  99.98 % (`total_fast` 99.88 % to 99.96 %). A registry entry may now state `step_size`; the
  generator reads upstream's rule out of nnunet.py, and a stated step enters the warm-model
  key and the result key (`step=0.8`). `ts.v2` states none and keeps 0.5, so none of its
  results or keys move.
- **A dataset holding two models of one configuration is refused, never chosen.** Datasets
  831-836 each ship upstream's default `nnUNetPlans` model and, beside it, the
  `nnUNetResEncUNetLPlans_8` model upstream runs only for `model_size="small"`. The resolver
  keyed a dataset's folders by configuration alone, so the one sorting last - the small model
  - would have run under the default's name without a word. A registry entry now states
  `models: {dataset: {trainer, plans}}`, every resolve of that weights id carries it (the
  load, `describe()` and so the result key's weights, `warm`, the orientation decision), and
  a dataset still holding more than one match raises `AmbiguousModel` - reported as
  `unresolved` by the doors that already report an installed model they cannot choose from.
- **`weights refresh` no longer adds a license-gated dataset** that TotalSegmentator has also
  published as a release asset: Dataset857 (`thigh_shoulder_muscles`, `commercial` upstream)
  appeared in `v3.0.0-weights`, and taking its URL would have made haversack download what
  upstream installs only through its licensed backend. It is reported as `license_gated`.
- **TotalSegmentator's crop tasks crop as TotalSegmentator does.** Every ts.v2 task that crops
  with a coarse model first - head_muscles, headneck_bones_vessels, liver_segments,
  lung_vessels and 22 more - now cuts the input to the crop classes' box plus the margin and
  runs its final model on that cut alone, labeling nothing outside it; an empty crop is an empty
  result. The crop used to be a speed approximation of whole-volume inference, grown to the
  network's patch with real image and dropped when it saved too little, and it labeled beyond
  upstream's box: on a neck CT, `headneck_bones_vessels` scored mean Dice 0.738 against upstream,
  with zygomatic arches only haversack found. The crop stage's labels are restored nearest-
  neighbor onto the input, as upstream restores them, and the margin is the 20 mm upstream
  actually applies to every task that crops with its `total`/`body` models, not the value in the
  task's config (upstream overrides it; teeth keeps its own 10 mm). These tasks' results are keyed
  anew (`crop=upstream`) and recompute once; no other task's key moves.
- **TotalSegmentator results no longer carry classes their task does not name.** Some
  TotalSegmentator models are trained with helper classes the task drops; upstream zeroes them
  after prediction, haversack wrote them as unnamed values. `ts.v2:kidney_cysts` wrote Dataset
  789's whole kidneys as values 3 and 4 - on one abdominal CT 45,234 and 55,244 voxels beside
  531 of cyst - under a label map naming only 1 and 2. Every value a TotalSegmentator task's
  label map does not name now becomes background, as upstream does, and the registry states each
  task's dropped classes (`auxiliary`, upstream's own lists: kidney_cysts, appendicular_bones,
  face_mr); a model emitting any other unnamed value is refused as not matching its catalog.
  The named labels are unchanged voxel for voxel. These three tasks' results are keyed anew
  (`auxiliary=0`) and recompute once; no other task's key moves.
- **`ts.v2:headneck_muscles`: TotalSegmentator's 23 neck muscles.** Sternocleidomastoid, the
  three scalenes, platysma, the three pharyngeal constrictors, the prevertebral muscles,
  sternothyroid, thyrohyoid, levator scapulae and trapezius, each side separately where that
  applies - one of upstream's openly available (Apache-2.0) tasks. The weights (Datasets 778
  and 779) were in the manifest all along; the task was not, because upstream runs it as a
  crop followed by a union - the 6 mm `total` model boxes the clavicles and C1/C5/T1/T4 plus
  40 mm, exactly as for `headneck_bones_vessels`, and both models then run on that one crop
  and are combined as `total`'s parts are, a later part over an earlier one - and a cascade
  here could only end in a single model. A cascade's last stage may now be a union; the
  registry refuses a union anywhere else, and a stage stating two things or none.

### Ranked stores

- **Ranked stores are written under duckn seg 0.8, and carry the model's classes and nothing
  derived from them.** duckn 0.4.0 changes what a segment is - it lists `label_values`, always
  an array; `background: true` is `role: "background"`; a color is a CSS string - and has no
  groups. The builder used to write two kinds: a `classes_<i>` partition per part, and named
  unions (`g_lungs` over TotalSegmentator's five lobes, a vertebral column, FastSurfer's
  subcortical sets) with `disjoint` and `exhaustive` claims. Those unions are ours, not the
  model's - no TotalSegmentator task emits a "lungs" class - and they are facts about a
  labeling scheme, the same for every store a model produces, so they move to a document
  outside the store. The partition needs no statement at all: a part's classes are disjoint
  because no two list the same value, and the background role means "none of the described
  structures is here". `GROUP_CLAIMS`, `named_groups` and `part_partition` are gone;
  `ranked_store.segment()` replaces `leaf()` and `group()`. **Clients that read `label_value`,
  `members`, or a `g_*` / `classes_*` id from a store's raw attributes must change** (so did
  the bundled `preview.html`, until its rebuild below). Stores already delivered keep reading:
  `read_segmentation`, `ranked_restore` and the tools read both shapes, duckn migrating the
  older one (a group becomes a segment listing its members' values).
  `tools/ranked_upgrade_seg.py` now upgrades a store to 0.8 in place: it removes every group
  the builder ever generated, whatever engine wrote it, and keeps one a user authored.
- **A store declares the labeling scheme it was produced under.** `labeling_scheme` names
  an entry of `terminologies`, and each class the catalog names carries its name as an exact
  designation in it - which is what lets a hierarchy, a color table or a cross-walk written
  once for a scheme find its segments in every store. For the `ts.v2` catalog the key is the
  ecosystem-qualified name of the task whose class list it is, the `system_uri` carries the
  catalog's major version and the task and not the release
  (`https://github.com/wasserth/TotalSegmentator#v2:total`), `version` is the package
  version the registry was generated from, and `url` is that release's tree. Checked
  against upstream at 2.13.0: all 51 label maps equal TotalSegmentator's own `class_map`,
  and the six `_fast` / `_fastest` variants have no class list of their own - they are their
  base task's classes from a coarser model - so they declare its scheme
  (`ts.v2:total_fast` writes `ts.v2:total`). A build handed its own names declares no
  scheme, and neither does an engine that names its own labels;
  `ModelEcosystem.labeling_scheme` is where the other catalogs will answer.
- **Every catalog with a published class list declares its scheme, and the scheme now
  reaches the stores the product writes.** The first cut never fired on the normal path:
  `segment_to_store` hands the builder the run's own names, and the builder declared a scheme
  only when it was handed none - so only `tools/ranked_build_store.py` ever wrote one.
  `build(model_names=True)` says whose names they are; a caller's own names, or a run that
  could not name its classes (`labels_unnamed`, a MONAI region head), still declare nothing.
  Schemes resolve against every catalog this build knows, not the ones a machine serves.
  Per catalog, each claim checked against upstream on 2026-09-21: **`moose`** per task,
  versioned by the asset's release stamp, its `fast_*` tasks NOT folded onto their base
  (separate upstream models whose lists coincide by fact), and `clin_ct_dental` declaring
  DentalSegmentator's scheme, whose model it is; **`mrsegmentator`** `base` only, versioned
  by upstream's `weights_version` (`1.2`, not the source tag) - `body_comp` names its classes
  in German inside its checkpoint while upstream publishes them in English, so it declares
  none; **`cads`** per task, all nine lists equal to upstream's own label-map module value for
  value (pinned in `tests/fixtures`); **`totalvibe`** with `vibe` and `vibe_sagittal` sharing one
  scheme, none for `body_regions` and `feet_bones` (digit-string names), and the repository
  spelled `VIBESegmentator` as upstream spells it - the manifest had it wrong, and a `system_uri` is
  compared byte for byte; **`dentalsegmentator`** identified by the weights' Zenodo concept
  DOI; **`monai`** per bundle, the bundle name in the `system_uri` and its version as the release;
  **`synthstrip`**, **`voxtell`** and **`custom`** declare none.
- **FastSurfer stores code their classes by id, and 17 of them carry no code.** FastSurfer's
  identifier for a class is the aparc+aseg number, so `ModelEcosystem.scheme_code` lets a
  catalog say how it spells a class - and whether it has an exact code for it at all. A
  ranked store holds the network's channels BEFORE `split_cortex_labels`, which lateralizes
  17 lh-numbered cortical ids spatially (it is why the LUT has 31 `ctx-lh-*` ids and 14
  `ctx-rh-*`): in a store, value 1003 is both caudal middle frontal cortices under a
  left-hemisphere name. A designation says a segment IS a concept, so those 19 carry none.
  Measured on a real run (ds000114 sub-01, on Modal): 17 channels moved 38-56% of their
  voxels to the right under upstream's split, no other channel moved any, and 1025 and
  1028 - which upstream's list names but which have rh channels of their own - moved none,
  so those two keep their codes. Their NAMES are still the left-hemisphere ones, which is a
  separate thing to put right.
- **A ranked store can be written for `fastsurfer:asegdkt`** (`-o case.duckn`): the engine
  already hands over its pre-argmax field, and the refusal that kept engine tasks out now
  admits the engines whose runner takes a ranked sink. `Segmenter.segment` accepts
  `probabilities=`.
- **A cascade's ranked store names each layer from its own model.** A crop stage outputs
  its own model's classes - stage 0 of `ts.v2:lung_vessels` is `ts.v2:total_fast`'s 118 -
  but the builder named every layer from the task's label map, so layer 0's spleen, kidneys
  and gallbladder were stored as `lung_airways` ... `lung_veins`, coded in
  `ts.v2:lung_vessels`, and values 5-117 as `label_<v>`. It meant to leave crop stages
  unnamed, but recognized them by a part name (`<task>:s<i>` on every part) the pipeline had
  stopped writing for the final stage, so the check never fired and `parts="last"` dropped
  nothing. The emit now records `labels_named_by` - the one task of the catalog that runs the
  stage's model alone (`TaskCatalog.stage_task`; all 26 ts.v2 cascades resolve), or none,
  which leaves a stage unnamed rather than misnamed - and the store's part block keeps it. A
  store whose layers follow two class lists declares both schemes: `labeling_scheme` is then
  an array, the task's own first, as duckn seg 0.9 allows. `tools/ranked_verify.py` fails a
  layer coded in another task's scheme (it had the same broken check, and reported the old
  stores as merely unnamed) and no longer reports parts on a 3 mm and a 0.7 mm grid as
  differently oriented; `tools/ranked_upgrade_seg.py` no longer drops `labeling_scheme`.
  **Cascade stores written before this are misnamed in layer 0: rebuild them.** No format
  version moves. An emit directory that predates the field is named from its `<task>:s<i>`
  part names, which covers every cascade but `ts.v2:teeth`: its middle part, cropped from
  `craniofacial_structures`' result, is still named from teeth's label map, and
  `ranked_verify` does not catch it. Emit teeth again rather than rebuilding an old emit.
- **The ranked encoder's slab is sized from the device's free memory** (rankfield 0.3.5's
  `memory_budget`, pinned below). `network.encode_budget` takes half of `device_budget_bytes` -
  on MPS the allocator's pool grew in ~1 GiB heaps to 1.5-2.2x rankfield's `slab_bytes` bound -
  capped at rankfield's 1 GiB default, which is also the answer on cpu. The cap is measured,
  not caution: on an M2, same process, conditions alternated, K=118 at 236x167x167 encoded in
  5.6 / 5.5 / 5.7 s at 7 / 15 / 30 planes and 8.1 s at 45, K=25 at 472x334x334 in 32.7 / 34.7 /
  40.8 s at 5 / 11 / 22 - a thicker slab buys nothing, so the measurement only ever shrinks it
  on a device too full for the default. CUDA is unmeasured; `ENCODE_BUDGET_CEILING` is the
  number to lift there. `ranked.emit` takes `memory_budget=` (keyword-only, never in the code's
  meta - it moves no byte) and the nnU-Net and FastSurfer paths pass it; the nnU-Net path
  records it per model as `encode_memory_budget_bytes`, beside `accumulate`, not as a
  deviation.
- **`haversack view` opens current stores again.** Its bundled `data/preview.html` dated from
  2026-09-05 and accepted only seg 0.6/0.7 and ranked 0.2/0.3, so it refused every store
  written since the seg 0.8 work above. Rebuilt from sdfview `1983b26` (byte-identical to a
  rebuild from that commit): it reads seg 0.9 segments by (layer, value) and ranked format
  0.4, and a cascade whose parts sit on different grids - a 3 mm whole-body crop stage under
  a 0.7 mm fine stage - is composed as haversack composes parts: each grid restored on its
  own, painted in `part_order`, later over earlier. Parts sharing a grid still share an atlas
  (a five-part `total` store renders as before, 7.0 s to load instead of 8.3 s). The lite
  viewer and the margin renderer still refuse stores on several grids.

### Jobs, inputs and references

- **A job's input can be a result the server itself computed: `result:<key>`.** Every
  source until now named data that came from outside. `<key>` is the `key` a finished job
  already reports, so jobs compose - CT to segmentation, then something computed from (CT,
  segmentation) - and each step is cached under the rules every other result is, so a cheap
  step run again never repeats the expensive one before it. `result:<key>!<name>` selects a
  named output (only `labels` exists) and `@sha256:<digest>` pins the bytes. The key is 64
  hex characters and nothing else, so no host can be spelled and a result on another server
  cannot be referenced. `GET /v1/sources` lists `result` on a server that keeps a result
  cache. This is step 1 of `docs/result-references.md`; results that are not label maps, and
  an engine that needs them, are not part of it.
- **The identity of a reference is the content digest of the output it names, not the key.**
  `Cache-Control: no-cache` republishes other bytes under the same key, and a downstream
  result keyed on the key would silently outlive the mask it was computed from. The server
  resolves the reference at submit and keys the job on the digest it finds - the same digest
  an upload of those bytes has, so referring to a result and sending its bytes are one
  request with one cached answer. `no-cache` on the downstream job resolves the reference
  again; it never recomputes the upstream result.
- **A reference that does not resolve is refused at submit, with what to compute first**
  (409 `result_missing`; 422 for an output the result does not have, or a label map bound to
  a role that takes an image) - never a queued job that fails minutes later in a worker,
  which on Modal is a GPU container started for nothing. The worker that fetches a reference
  resolves it a second time and hashes what it copied, because on Modal a lease taken in the
  api container does not reach a worker's pruning: a job computes only ever from the bytes it
  was keyed on, and where a result was recomputed or evicted in between and those bytes can
  no longer be read, it fails with "the referenced result changed" (or "no result ...")
  rather than compute from others. On
  Modal both resolutions read the result volume under the rules of 0.12.4 - never while
  another thread of the container may be reloading it, a local copy made under the lock, a
  miss believed only from a view newer than the request (503 otherwise) - and the prefetch
  thread never stages a reference at all.
- **An input role can take a label map.** A task's `inputs[].kind` is `image` or `labels`,
  and `haversack.labelmap.read_label_map` reads a `.seg.nrrd` with its segment names, its
  geometry and the task that made it. A consumer of (image, mask) selects structures by
  name - `liver` is a different label value in every catalog - so a label map that carries
  no names (a NIfTI) is refused unless the caller asks for label values, and one where a
  name cannot be matched to voxels without choosing - two segments on one value, two values
  under one name, layered segments, non-integer voxels - is refused always. The reader is
  its own module: `io.read_image` stays the image reader, and the default `segment` path
  imports none of it. Every declared role is required, as image roles are. No shipped task
  takes a label map yet.
- **`GET /v1/sources` says which prefixes have a path surface.** Every entry gains
  `path_addressable`, and a server with a result cache lists one more entry, `result`, for
  which it is false. Additive, and the only thing here an existing client can see besides
  the dead links below going away: no route, no accepted request, no refusal code and no
  cache key changed, so nothing stored is recomputed.
- **The terms of the original data survive the hop.** A result computed from a reference
  records, in `provenance.inputs`, the digest, the upstream key, output, task and weights
  versions, and that task's attribution; `derived_from` carries what is above that hop flat
  - earlier hops by reference, each original input once by value with its origin, license
  and citation - so a long chain does not copy its whole ancestry at every step, and the
  terms outlive the upstream entries' eviction. They are recorded by the computation, so
  the one exception is a cache hit on an answer first computed from an UPLOAD of the same
  label bytes: that answer says "uploaded by the caller", whoever asks for it next, as an
  upload and a stored input already shared theirs. `no-cache` recomputes it from the
  reference.
- **No content digest is path-addressable.** A job status linked
  `/v1/sha256-tree/<hex>/<task>/...` for an uploaded DICOM series referred to by digest - a
  path no route has ever served - because the rule was written against the `sha256:`
  spelling. It is asked of the digest grammar now, which is also what keeps a reference's
  result off the path surface: a path is keyed with no lookup, and a reference needs one.
- **A license a catalog states for one task now reaches the result, not only `describe`.**
  `segment` wrote a result's attribution from the catalog's name and the modality alone, so
  a license a manifest states per task never reached the `.seg.nrrd` header: it fell back to
  the catalog's. The engine path always handed over the catalog's own record of the task;
  the nnU-Net path now does too. Nothing shipped was misstated - every manifest that names a
  per-task license repeats its catalog's - but a catalog whose tasks differ in license would
  have been, in the one copy of the terms that travels with a download. A per-task license
  that only repeats its catalog's keeps the catalog's fuller record (it names the code's
  license as well), so no shipped task's header changes: without that, 17 would have lost
  `code` or changed case for the same facts, and a result's ETag is the digest of that file.
- **A job's event stream now sends the status `GET /v1/jobs/{id}` answers**, `key` and `links`
  included, as SERVER.md always said it did; it sent the executor's raw record, so a client
  whose wait ended on the stream held a status without the result's handle.
  The nnU-Net encoders now read their task's weights from the root the server's own
  `Segmenter` reads, where they used to read the default root whatever the server used.
- **A server describes an uninstalled task's structures.** `GET /v1/tasks/{task}` said
  "structures are read from the checkpoint once installed" for every task whose weights the
  server had not fetched - `moose:clin_ct_muscles` showed no structure list on a server whose
  own `GET /v1/segments` listed its ten muscles. The segments index was mined from that very
  checkpoint, and `haversack tasks TASK` has read it since the index shipped; describe now
  reads it too, through one function the CLI shares, marked `structures_from: segments index`
  with the version that pins it. Neither door shows a record whose catalog has since moved to
  another version (the CLI used to print it anyway), or a list for a request pinned with
  `@version`. A result's key reads nothing new: it is the same with the index, without it, and
  with a broken one.

### What is served beside a result, and how

- **What is rendered beside a result is the request's to say: `deliverables`.** The preview
  and the statistics were a deployment setting, so every job paid for both - a cohort run
  by upload rendered a preview per scan that no route can even serve - and "preview off"
  meant redeploying. `POST /v1/jobs` takes a `deliverables` form field, a JSON list
  (`["statistics"]`, `[]` for none); absent, a job gets the deployment's set, exactly as
  before, and that set is also the ceiling: a name this server does not render, or has
  never heard of, is refused at submit with what it offers (`GET /v1/health` lists it). A
  declined deliverable costs nothing - no render, and with an empty list no second read of
  the two volumes either. `RemoteClient.submit(..., deliverables=[...])` and `haversack
  remote submit --deliverables statistics` (or `none`) send it. This is the light half of
  `docs/result-references.md`: what is numpy-only and needs the image and the labels runs
  where both already are; a GPU model is a job of its own over `result:`.
- **A deliverable never enters a result's key.** Every option is hashed into the key, so the
  list is a field of its own and is refused inside `options`: declining a preview and then
  asking for one is one result - the same `key`, a cache hit, the same labels `ETag` - and
  no segmentation is ever recomputed to draw a picture of it. No cache key moved and no
  computed byte changed, so nothing stored is recomputed and `CACHE_EPOCH` stays.
- **A cache hit still honors the list.** A deliverable the request names and the stored
  result lacks - declined by the request that computed it, or never rendered - is rendered
  on the hit, into the generation that already holds the labels and through the path every
  artifact takes (its pending marker is the single flight, the cache's `add_artifact` the
  placement), so an artifact can no more land beside another publication's labels than it
  could before. The server never fetches an input again to do it: what it cannot render it
  SAYS, in the job's `deliverables_unavailable` with the reason and the way out
  (`no-cache`), rather than leave a link off without a word. On Modal artifacts are
  rendered by the worker that computes a result, and a hit reaches no worker, so there a
  hit reports what is missing and renders nothing; a render-only job is the follow-up.
- **`links` name what was asked for, and a declined artifact is absent at once.** A job's
  links advertised whatever the deployment renders; they are built from the job's own list
  now, less what a hit said it could not deliver. The pending marker records what its
  render will place, so a GET of a preview the job declined answers 404 immediately
  instead of 202 until the statistics land. A read never renders: a GET of an artifact a
  cached result lacks is a 404 that names the request which renders it, for an anonymous
  caller and an authorized one alike.
- **`statistics.json` says which axis order each spacing is in.** `grid_spacing_mm` is
  (x, y, z) in RAS - the labelmap's spacing AFTER the RAS reorientation, so a coronal series
  reads e.g. [0.78, 3.0, 0.78] - while `field_grid_spacing_mm` is the model grid's (z, y, x).
  Nothing said so, and a reader guessed wrong (2026-09-19). The JSON's `units` block now
  states the first; no number changes.
- **`HEAD /v1/jobs/<id>/result`, and a "gone" no cache may keep.** An adversarial pass on
  the artifact routes below found the one file route they left without a `HEAD` - a job's
  own labels, 405 until now, and invisible to the test that reads the router for file names
  because its name has no dot. It answers what `GET` would (status, `ETag`, the file's
  `Content-Length`, a 304 for a matching `If-None-Match`) and converts nothing: with
  `?format=nii.gz` it says 200 and no length. The job routes' 410 now says `Cache-Control:
  no-store` as their 404s do - RFC 9111 (4.2.2) lets a cache keep a 410 on a heuristic, and
  these URLs say 200 again once the key is recomputed with the same output. The new routes'
  record, in-flight and render-state lookups run off the event loop (on Modal each is a
  Dict round trip, and the anonymous twin now makes one per artifact miss). One capability
  went with the artifact work and is recorded here rather than restored: `preview.png` is
  sent from memory, so it no longer answers `Range` requests (the labels still do).
- **A result with no path can reach its deliverables: `/v1/jobs/<id>/preview.png`,
  `/statistics.json`, `/statistics.tsv`, `/meta.json`.** The artifacts beside a result were
  served only by its path, and a result whose identity has no path - an upload's, a
  `result:` reference's, a multi-input job's - had none: the job rendered `preview.png` and
  `statistics.json` into its cache entry, reported `deliverables: ["preview", "statistics"]`
  with nothing unavailable, and offered no link, because no route served them. The person
  who uploads a local scan from 3D Slicer is exactly who wants the statistics. The new
  routes are the job's own and authorized like `/result`, never anonymous - a job's preview
  shows what was uploaded - and resolve as `/result` does: through the job's key to its
  published entry, held to the digest this job reported, and by its answers (404, 409 not
  done, 410 gone, 503 not visible yet), then 202 with `Retry-After` while a render that will
  place the artifact is pending and 404 when none will, with the job's own reason. A
  path-less job's `links` carry them as `meta`, `preview` and `statistics` - the names a
  path-addressable result uses, so a client follows one name either way. Listing rows for
  such results stay without links: the listing is of results and knows no job. A job with no
  cache entry to render into (a server without a result cache) now says so in
  `deliverables_unavailable` rather than list what nothing would serve.
- **Fixed: an artifact's strong `ETag` did not change when its bytes did.** `meta.json`,
  `preview.png` and `statistics.json` / `.tsv` all carried one tag derived from the result
  KEY, under `Cache-Control: public, max-age=3600`. A `Cache-Control: no-cache` recompute
  republishes under the same key, so the same URL then served different bytes - a preview of
  8290 and then 8309 bytes, a `volume_ml` of 0.384 and then 0.512 - under an unchanged
  strong validator, which RFC 9110 (8.8.1) forbids, while the labels' tag (their content
  digest) moved as it should. Each artifact's `ETag` is now the digest of the body it sends.
  The content rather than the publication, because the read is already paid: three of the
  four bodies are built per request and a `HEAD` owes them to its `Content-Length`, the
  fourth is a PNG of kilobytes, and a digest is also right for a legacy entry with no
  generation and for a job's own copy. With a tag that can be trusted, `If-None-Match` is
  answered on all four as on the labels: a 304 that repeats `Cache-Control` and `Vary` and
  never `Preference-Applied`. The labels' tag, the result key and every computed byte are
  untouched.
- **Fixed: `HEAD` on an artifact was a 405, and its `Allow` named a method the URL never
  had.** `HEAD` had just become the compute-free probe for labels, and deliverables now land
  after `done` - so "has the preview rendered?" was exactly the request being refused, with
  `Allow: DELETE` on the api (the greedy `DELETE /v1/<source>/<id>/<task>` pattern also
  matches `.../<task>/preview.png`, was registered first, and the router answers a 405 from
  the first route whose path matches) and `Allow: GET` on the anonymous twin, which claims
  header parity with the api by construction. Every artifact route - by path with its grid
  tokens, on the twin, and through a job - now answers `HEAD` with `GET`'s `ETag`,
  `Content-Length`, `Cache-Control` and `Vary` and no body, honors `If-None-Match`, and
  never computes, renders or waits: 202 while the labels compute or a render that will place
  that deliverable is pending, 404 otherwise - at once for a deliverable nobody asked for,
  since no render is coming. A 405 now lists the methods of the URL asked about, whichever
  routes they live in.
- **Fixed: the anonymous twin called an artifact that was still rendering absent.** The
  twin's executor had no view of the pending-render marker, so between a job's `done` and
  its preview landing - on Modal, the worker's commit of it - a GET of `preview.png` there
  answered 404, to the anonymous poller told everywhere else that a 404 is final. Found by
  deploying (`haversack-doors-smoke`, a render slowed to 10 s), not by the suite: locally
  the window is milliseconds. `create_public_app` takes the writer's `artifact_state` as a
  read-only signal, as it takes `inflight`, and the Modal twin reads the marker (and never
  sweeps a dead one: it writes nothing). It answers 202 with `Retry-After`, as the api does.
- **"Not materialized" says `Cache-Control: no-store` too.** The labels' own 404 - HEAD
  probe and GET, the api and the anonymous twin, every grid token - stated no freshness,
  beside a 200 that says `public, max-age=3600`. It is the answer until somebody computes
  the result, which is the next thing an authorized caller does with it, so a shared cache
  that kept it on a heuristic (RFC 9111, 4.2.2) would go on hiding a result that exists.
  Every 404 of these routes, the unknown task's included: a task unknown today is served
  after the deploy that adds its catalog, and an uncached error costs one request.
- **An artifact's 404 says `Cache-Control: no-store`**, as its 202 always has. An artifact
  arrives late - after `done`, on a cache hit that asks for it, and with a shared result
  store from another host into the same generation - so "not here" only ever means "not as
  far as this request saw". The 200 beside it invites shared caches, and RFC 9111 (4.2.2)
  lets one keep a 404 that states no freshness on a heuristic; a proxy could have gone on
  answering 404 for a preview that landed a second later. The server remembers no absence
  either: every request looks again.
- **`If-None-Match` is compared weakly.** `W/"<tag>"` matches `"<tag>"`, as RFC 9110
  (13.1.2) requires of this header; the strings were compared whole, so a client or proxy
  that had weakened the tag - which one that re-encodes a body must - downloaded a label
  volume it already held. The safe direction, and still wrong.
- **Fixed: a 304 for a result by path dropped the caching fields its 200 carries.** A
  conditional `GET` of `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` answered 304 with
  the `ETag` alone, while the 200 for the same request says `Cache-Control: public,
  max-age=3600` and `Vary: Prefer`. RFC 9110 (15.4.5) has a 304 repeat the 200's
  `Cache-Control`, `Content-Location`, `Expires` and `Vary`, and these are the responses the
  server invites shared caches to store. Nothing was seen to break - a cache keeps the stored
  fields a 304 omits (RFC 9111, 4.3.4) - so this is the server saying what the standard has
  it say. `Preference-Applied` deliberately does not cross: a cache writes a 304's fields
  onto every stored response holding that validator, whatever `Prefer` it was stored under,
  so a `wait=30` echo would land on the variant kept for a plain `GET`. The job result
  route's 200 carries none of the four, and its 304 is what it was.
- **Fixed: `HEAD` on a result path named a validator no `GET` issues.** `GET
  /v1/<source>/<identifier>/<task>/labels.seg.nrrd` answers with the content digest as its
  `ETag`; `HEAD` on the same path answered with the first 32 hex characters of the result
  key, because the probe built its headers without the entry's result - which both of its
  lookups hand back. A client asking "has this result changed?" with `HEAD` compared a tag
  it had never been given, and the digest became the validator precisely so that a weights
  bump which leaves the bytes alone forces no re-download. The same 200 also said
  `Content-Length: 0`, the web framework's count of the empty body, where RFC 9110 (8.6)
  allows only the length `GET` would send. Both matter beyond tidiness: these responses are
  `Cache-Control: public`, and a shared cache that forwards a `HEAD` marks its stored `GET`
  stale when either field differs (RFC 9111, 4.3.5). `HEAD` now sends `GET`'s `ETag` and
  `Content-Length`, on the api and the anonymous twin, and on Modal also where the entry is
  found only by the reload that confirms a stale miss.
- **`HEAD` honors `If-None-Match` with a 304, as `GET` does.** RFC 9110 (13.1.2) names the
  two methods together, and a `HEAD` that says 200 to the request `GET` says 304 to is the
  same disagreement one header over. Only the 200 is conditional: a result in flight is
  still 202 and an absent one 404. (`meta.json` kept a key-derived `ETag` at that point, its
  body being the result record and not the labels; it has a validator of its own now - the
  artifact entries above.)

### The result listing and the cache

- **`GET /v1/segmentations` takes filters and pages, and its silent cap is gone.** It
  returned the newest 500 results and said nothing of the rest - a Modal cache of 2,083
  listed 500 - took no filter, and read every entry one after another, which on a Modal
  volume is the expensive way round: measured there, listing 2,083 names is 0.03 s and a stat
  0.4 ms, but the FIRST read of any file is ~24 ms whatever its size, so a serial walk of that
  cache took 161 s. `limit` (1-1000, default 100) and an opaque `cursor` replace the cap, and
  the answer carries `next_cursor`, null on the last page. `identity=` (repeatable, up to
  100: a cohort asks "which of these series are done" in one request) and `task=` filter.
  `haversack remote results` and `RemoteClient.segmentations` / `iter_segmentations` take
  the same filters and follow the cursors; against a server from before this, the client
  refuses to pass off its whole listing as a filtered one.
- **No index stands behind the listing, on purpose.** An identity -> key index would be one
  more thing to keep in step with every publication, eviction and delete, by every container;
  the measurements give three answers that keep nothing. A filter by identity is COMPUTED:
  the server derives the key of every task it serves, under the default options and each grid
  token, and looks those names up - what the path surface does for one task - so it reads no
  entry it does not return, and is as fast on thousands of results as on none. It finds the
  results that have a path; one computed with other options, or from several inputs, is in
  the plain and the `task` listing only. Order and paging come from names and stats - the
  mtime of each entry's pointer - and content is read for the requested page alone. A filter
  by task alone is the one that needs content: entries are read 32 at a time (the same walk
  took 4.5 s that way), and a running server remembers each publication's `meta.json` in
  memory, validated against the pointer's stamp on every use - a cache that rebuilds in
  seconds, not an index that can be wrong. The task filter is asked of that metadata before
  an entry's other files are stat'ed, because on the volume stats are the warm cost and,
  unlike reads, barely overlap. Deployed over that 2,083-result cache, mounted read-only
  (server-side, one container each): an identity filter 0.55 s, and 0.2-0.3 s with `task=`
  for one input or for fifty; a first page 0.6-0.7 s; a scan for a task no result has
  3.5-9.7 s cold and 0.3-0.5 s once remembered; all 2,083 rows by cursor in three requests.
  Every listing there includes a ~0.2 s volume reload. A key's real cost turned out to be
  its task's weights versions (a `describe()`), not the lookup, so a request reads them once
  a task - through `weights_versions` on an executor that pins its own keys, and
  `create_public_app(..., weights_fn=)` on a twin; asked key by key, fifty inputs across every
  task took 47 s, and 1.6 s after.
- **The listing is ordered by publication, and a cursor is a position.** Rows gain
  `published`, the mtime of the entry's pointer: one atomic rename moves it, at the instant
  the entry's content changes, and nothing a read does touches it (reads lease a file of their
  own and touch the key directory, whose mtime would have reshuffled the listing under
  traffic; `computed` is a job's start on a worker's clock). The cursor holds (time, key),
  not an offset, so pages stay put while results are published: what arrives after the first
  page sorts ahead of it, where an offset would repeat a row on every page after it.
- **A multi-input result is listed.** The listing's stale-key filter re-derived each
  entry's key from its FIRST identity alone, so every result of a multi-input task read as
  stale and was dropped. It is held to the same rule as the rest now - listed under the key
  this server derives from all of its identities, without links, as an upload's is.
- **On Modal the listing follows the volume rules of 0.12.4.** It is read from a view of
  the result volume newer than the request - a result missing from a listing reads as "not
  computed", the listing's form of the false 410 - and a reload that cannot be had is a 503
  with `Retry-After`, never a shorter list. Its reader threads count as "other threads" to a
  reload, so every batch of reads runs inside the view lock, held by the thread that started
  it until the batch has ended; per batch and not per listing, because a cold scan is seconds
  long and lookups queue behind a waiting reload.
- **A Modal deployment can name its result-cache volume** (`--cache-volume`,
  `HAVERSACK_CACHE_VOLUME`; default `<app>-cache`, as before). Every store was named after
  the app, so a deployment under a new name began with an empty result cache - though a
  result key holds no app name, which makes a cache portable. Only the cache can be named:
  scratch, the inputs store and the job store hold one deployment's job ids, uploads and
  flights. Adopting a cache whose first deployment is gone is completely safe. Two LIVE
  deployments on one cache behave like more containers of one app, except that single flight
  lives in the per-app job store, so the same key can be computed twice - duplicate work, not
  corruption. Either way every publication evicts down to the publishing app's
  `HAVERSACK_RESULTS_KEEP` (default 500): adopt a larger cache with a larger bound, or lose
  the difference at the first job. The knob is forwarded to every container like the rest;
  unforwarded, the containers would commit and reload `<app>-cache`, mounted nowhere.
- **A result cache the server cannot write is read whole.** A cache on a read-only
  filesystem, or in a directory another user owns, is a supported way to read one - the
  reader's lease and the entry lock have always allowed for it - but the lookup did its
  least-recently-used touch and its read of `result.json` in one `try`, so where the touch
  was refused the read was skipped and every hit came back with an empty result. Nothing
  failed, which is why it went unseen until a listing was smoked over a cache mounted
  read-only (2026-09-20), and three things quietly answered differently. The `ETag` fell
  back from the content digest to the key, so `If-None-Match` from a client holding those
  very bytes got the whole download again instead of a 304. A `result:<key>` reference was
  refused `result_unreadable`, with the advice to recompute a result that was fine. And
  `GET /v1/jobs/{id}/result`, which serves the cache entry only when its digest is the
  job's, fell through to the job's own copy - 410 "purged" once that was gone, with the
  bytes sitting in the entry. The touch is best effort and on its own now, as the input
  cache's has been since 2026-09-06: there the LRU loses a touch, that is all. An empty
  result is left meaning what it was always taken to mean, a `result.json` that is missing
  or not what was written - which now includes bytes that are not UTF-8 (they raised, a
  500) and JSON that is not an object. No cache format, lease, claim or eviction rule
  changed, and nothing stored is recomputed.

### A result store servers can share (opt-in)

- **`haversack serve --result-store s3://bucket/prefix` shares the result cache between
  servers through an object store** (also `gs://`, `az://`; `HAVERSACK_RESULT_STORE`). The
  POSIX cache's guarantees rest on rename and `flock`, which an object store does not have
  and a filesystem emulating them over one (ZeroFS, assessed) loses on restart, so the store
  gets its own protocol, the one build caches use: result bytes as blobs named by their
  SHA-256 and written only if absent, and one pointer per key replaced by a conditional
  write, so a publication is one write and an artifact can never land beside another
  publication's labels. `--cache-dir` stays in front as each server's local copy, keeping
  the store's generation token so a current copy downloads nothing. A store that does not
  refuse a stale conditional write is refused at startup, naming it - asked, not assumed:
  obstore's own local-disk store fails it. Anything missing or corrupt in the store reads as
  a miss that the next computation repairs. `SharedResultCache.sweep` removes unreferenced
  blobs after a day's grace and, optionally, entries past an age; nothing schedules it yet.
  Not yet wired into the Modal deployment. `tools/probe_result_store.py` checks a bucket and
  round-trips one result under a throwaway prefix, and `tools/soak_result_store.py` runs
  publishers, readers and a sweeper as separate processes against it. First real store,
  Cloudflare R2 (2026-09-19): both conditional writes honored, two servers sharing one
  bucket served each other's results - the second never ran its segmenter - and the soak
  saw no torn read and no error while a sweeper with no grace deleted blobs underneath it.
  A hit costs one pointer read - about 85 ms median from this Mac, measured before
  history existed; a key republished four times carries ~5x the pointer, which has not
  been re-measured on a real bucket.
- **The result store runs on a directory too: `--result-store file:///path`.** provender
  0.1.6 adds `DiskStore`, a directory that honors both conditional writes (obstore's own
  local store honors only one, so `file://` used to be refused at startup), and
  `objectcache` now talks to its store through `provender.ops`, which answers for a bucket
  and a directory alike. Same layout, same protocol, no network. Every store test runs
  twice - in memory and on a directory - with the fault tests included, and the
  multi-process soak passed on APFS and on a real exFAT volume. One host only: machines
  share a bucket, never a directory over a network filesystem. Step 2 of
  `docs/cache-consolidation.md`'s from-scratch design; nothing changes without the flag.
- **The result store's format 2: a ref naming an immutable manifest.** Each key's ref
  (`results/<key>.json`) names a manifest stored as a blob - files by digest, result, meta,
  the publication's token, and `replaces`, the manifest it superseded - and carries its
  exact bytes, checked against the digest, so the present is still one read. History is the
  `replaces` chain instead of a list copied into every pointer (a read no longer grows with
  history); a late artifact is an amending manifest of the same publication; a deletion is a
  tombstone, which the coming sync can carry where a removed ref could not. Format 1 is
  still read and is converted, tokens kept, by the first write to its key. **Every host
  sharing a store must upgrade together:** an older haversack reads format 2 as a newer
  format - a miss, a refused publication, a sweep that deletes nothing. Also fixed on the
  way: a host holding a result re-downloaded its labels whenever a late artifact had not
  reached it yet, and a generation token read from the store is now validated before it
  becomes part of a local path.
- **`haversack cache sync SOURCE DESTINATION`: one result store into another.** A
  server's directory store (`file:///path`) into a bucket, a bucket into a directory, or
  any store into any other. Each result is decided by its history, not by clocks: copied
  when the destination lacks it, fast-forwarded when the destination holds an older
  version, left alone when the destination's is newer, and merged when both computed it
  independently - the later computation wins and the other stays in its history. Every
  object a result's kept history needs is copied before the destination's index names it,
  and deletions travel as tombstones, which the sweep now removes after 30 days (a copy
  that goes unsynced longer than that can bring a deleted result back). Rerunning is
  cheap: a key already current costs two small reads.
- **`haversack serve-store URL`: a read-only server over a result store.** Every read
  route of `haversack serve` - results by path, meta, preview, statistics, the listing - and
  nothing else: no jobs, no computation, and not one write to the store, so it runs on a
  read-only credential, with no GPU, no weights and no torch. It is the Modal deployment's
  public twin over a bucket instead of a volume. The one thing a bucket did not hold was the
  result KEY, which is a digest of the task's weights versions; writers now record those
  (`tasks/<task>.json`, rewritten only when they change) as they publish, and the reader
  keys from them. A task no writer recorded, or a writer on another cache epoch, is a miss,
  never a wrong result. Served from R2 in a real run: 565 ms for a first read, 115 ms after.
- **`haversack cache push` and `cache pull` migrate a result cache to and from a shared
  store.** The transition the consolidation plan needs: a cache that has been filling for
  months is worth GPU-hours, and nothing else recovers it once the local protocol goes.
  Each entry keeps the generation token it already has, so a pushed result is still served
  from this machine afterwards without downloading anything, and an entry from before
  generations existed is given one. Idempotent by construction - blobs are
  create-if-absent, the pointer is written conditionally, and a rerun costs one pointer
  read per key rather than a re-hash - so an interrupted push is simply rerun and two hosts
  pushing overlapping caches upload the shared bytes once. `--conflict` decides what
  happens when the store already holds a key: keep theirs (the default, because theirs may
  be newer), take whichever was computed later, or take ours. `pull` makes a cold host warm
  and is also the way out: afterwards the local cache answers on its own.
- **A server with `--result-store` sweeps it, every `--sweep-interval-hours` (24 by
  default, 0 to disable).** Now that `delete` leaves bytes for a sweep and a republication
  leaves its predecessor's the same way, a store nothing sweeps only grows - and a
  reclamation that depends on an operator remembering a cron line is one that does not
  happen. The loop is deliberately dull: the shipped grace, no expiry by age, so it can
  only remove bytes no entry refers to; it waits on the server's own condition so a
  shutdown stops it at once; the first sweep is one interval away and the interval is
  jittered, so a fleet restarting together does not all sweep in the same second; and a
  failure is reported and retried rather than taken seriously enough to end the thread.
- **`delete` stops reclaiming bytes; `haversack cache sweep` does it** (decided
  2026-09-20, replacing the "deletion means gone" half of the history decision). Deciding at
  delete time whether a blob belongs only to the entry being removed means deciding it
  against live publishers, because deduplication lets a publication happening right now
  reference those same bytes. Four attempts went into that - pre-listed candidates, a
  refreshed timestamp on deduplicated writes, a re-check before each delete, a wait for
  coarse clocks - and reviewers were still finding holes in it. The entry goes
  immediately, everywhere; the bytes wait for a sweep, which answers the same question with
  nothing else moving. About 120 lines and the hardest remaining reasoning in the module go
  with it. A sweep of a store with NO entries left is refused unless `--empty-index-ok` says
  it was meant, because "everything was deleted" and "wrong prefix" look identical from
  there.
- **A republication keeps its predecessors: bounded history in the shared store.** The
  pointer carries the generations it replaced - up to `HISTORY_KEEP` (4) and
  `HISTORY_MAX_AGE_S` (30 days), whichever runs out first - and the sweep treats their
  blobs as referenced, so storage per key is bounded by both. Deduplication makes it nearly
  free: a recomputation that produced identical bytes adds one small pointer entry and no
  blob. It answers what this server published over the last 30 days, up to four
  republications back; a weights upgrade can be compared against what it replaced, and a
  bad one recovered - `fetch_generation()` writes it to a directory you own, and
  republishing it is manual. The local copy offers no history API and never serves a
  predecessor, though its own superseded directories linger until that key is published
  again; history in the store is read deliberately (`history()`, `fetch_generation()`,
  which materializes into a directory the caller owns). `delete` removes the entry, its
  local copy and the bytes of every generation its pointer lists, keeping what another
  entry shares - near patient data, deletion means gone - and says in its report when it
  could not establish that, which `haversack cache sweep` then finishes.
- **The blob half is now `provender`, a package shared with feldglas.** A second project
  needed content-addressed blobs on the same kind of store, and two copies of one protocol
  is how this repo's defects have always started - so the blob store moved out to
  https://github.com/mhalle/provender (pinned by tag, as rankfield and duckn are), leaving
  thin wrappers here that turn its errors into haversack's and haversack keeps the pointer, the policy, and the live set
  the sweep needs, which is the only part that knows what a result is. Reviewing the
  extraction from the other side found two more: a sweep with no grace by default deletes a
  blob uploaded a second ago (every client writes the blob before the index entry naming
  it), and the package's own probe tool had been broken by a guard the day before, because
  nothing imported it. Both fixed in provender 0.1.1, the second with a test that runs the
  probe.
- **The result store knows an encode job's field.** A pointer names its one primary output -
  labels or `field.zarr.zip` - and `put(output_name=)`, the listing (a field row says its
  `kind` and carries no label links), `find_generation` and `fetch_generation` answer for a
  field as for labels; a name that is not a primary output is refused before anything is
  uploaded, and a pointer naming two is served as neither. The local copy follows too - the
  fill, `pull` and `push` - so a field computed on one server is a hit on another sharing the
  store, without its encoder running.
- **The shared store answers the new listing contract** (main's `ResultCache.list` with
  `keys` / `after` / `match` / `accept` / `memo` / `hold` / `workers`, 2026-09-21). Its three
  mechanisms carry over, and two get cheaper against a bucket: a listing returns names AND
  last-modified, so ordering and paging cost no stats at all, and the pointer IS the content
  - meta, sizes and which artifacts exist are one document - so a row is ONE request where
  the local cache pays a read plus three stats. `keys` reads only the names it was given and
  never lists the bucket, a miss being one HEAD; the cursor is the same `(stamp, key)` in
  nanoseconds, so a position issued by either cache means the same thing. Per-request
  deliverables land as they were designed to: a hit renders what is missing into the SAME
  generation, which is what `add_artifact` has always been - a conditional write that cannot
  land beside another publication's labels, and a render for a generation that has moved on
  is refused.
- **`SharedResultCache.find_generation(key, digest)`**, for `result:` references across
  machines. A `result:<key>` pins the referenced output's sha256 at submit and resolves
  again in the worker, refusing other bytes; on one machine the submit's lease keeps that
  generation alive, but a lease means nothing to another host, which may have republished
  the key in between. Bounded history answers the question a lease was standing in for -
  which generation has these bytes - and the sweep spares what history lists, so the pinned
  result is still there and `fetch_generation` hands it over. Measured: another host
  republishes, the pin refuses the new bytes, and the pinned generation is still found,
  fetched and spared by a sweep.
- **Result store sync: a merge no longer repeats, and a sync costs a fifteenth of the time.**
  Found by soaking two real servers on one directory store for 35 minutes with syncs into
  R2 (`tools/soak_server_store.py`). A merge - what sync writes when it cannot see how two
  versions of a result are related - exists only at the destination, and every later sync
  took it for a new version and merged the key again; it now counts as the version it
  carries. And a sync walked the destination's history and refreshed every object in it on
  every run: it now learns what the destination holds from the source's copy of the same
  history, uploads a new object with one request, and syncs 8 results at once
  (`--workers`). On R2, 16 results: 73.7 s to 4.9 s for an update, 200 s to 19 s for a
  first copy.
- **Review round on the above, same day: three agents, thirteen defects, all fixed.** The
  one that mattered: obstore's errors do not subclass OSError, so a store fault - expired
  credentials, DNS, a 503 - left `cache_get` as a bare 500 on routes SERVER.md promises
  404/410 for, anonymous ones included; the same class of defect 0.12.3 had just fixed for
  the scratch read. Reads now degrade to a miss (reported, throttled) and only WRITES
  raise. Also: a cache lookup is a network round trip, so the async routes hand it to the
  threadpool instead of stalling the event loop; a pointer's every field is validated
  because another host wrote it, and only known filenames decide where bytes land; one
  stray object under `results/` no longer aborts `list` and `sweep` for good; a sweep that
  meets a pointer it cannot read deletes no blobs, so an old host cannot collect a newer
  writer's results; a corrupt blob is no longer deleted (it is shared by every identical
  result) but suspected and replaced on the next publication; a missing artifact blob no
  longer costs the whole result; an artifact's blob gets the same post-pointer re-check as
  the labels; `add_artifact` never raises on the overlap thread; a local copy that cannot
  be written no longer fails a publication that already succeeded; and `list` reads at most
  `limit` pointers rather than one per entry in the bucket.
- **Second review round (four agents, 2026-09-20): twenty defects, all fixed.** The worst
  three: `push` read the entry's directory and its generation token separately, so a server
  publishing that key in between bound one generation's bytes to another's token and the
  pushing host believed its copy current for ever (the token now comes from the directory
  in hand); `"history": null` - what another language emits for "no history" - made the
  CURRENT result unreadable and, because the pointer then counted as unreadable, froze blob
  deletion for the whole store; and the extraction had inverted the sweep's listing order,
  so a blob written mid-sweep could be deleted at `grace_s=0` where the old order made it
  structurally safe (provender 0.1.2 takes pre-listed candidates, and haversack lists
  before it reads pointers again). Also: `delete` now removes the bytes, not just the
  pointer, keeping what another entry shares and refusing loudly when a pointer it cannot
  read makes "unreferenced" unknowable - near patient data, deletion means gone; the
  history age bound applies on READ as well as on write, so a key published and then left
  alone stops keeping its old generations; one rule now decides what a history entry is, so
  an entry the readers rejected can no longer occupy a slot for ever; history entries no
  longer carry `meta`, which no reader ever read; `--limit` bounds the work rather than the
  entries examined, so a rerun makes progress; `pull` goes oldest-first, repairs a local
  copy whose files went missing instead of calling it current, and reports what did not fit
  in the local cache instead of counting it pulled; a store URL that is malformed says what
  forms are accepted rather than advising about credentials it never used; a corrupt blob
  is reported again (that warning was lost at the extraction seam); and a new guard
  reconciles the dependency floor with the sourced tag and with the installed package,
  which is the same drift the CI pin test exists for, one field over.
- **Third review round (four agents, same day): sixteen more, and `haversack cache sweep`.**
  The one that mattered: deduplication defeats the sweep's candidate listing, because a
  recomputation producing identical bytes uploads nothing - so the blob is old while the
  pointer naming it is new, and both the sweep and `delete` could take it out from under a
  result that had just been computed. provender 0.1.3 refreshes a deduplicated blob's
  timestamp with a server-side copy (one request at any size, verified at 32 MB on R2), so
  a blob is again as young as the reference to it. Also: `delete` no longer reads every
  entry in the store on an HTTP route (bounded, and it says to run the sweep instead),
  reclaims the bytes of generations the age bound has stopped listing, and REPORTS whether
  the bytes actually went rather than only saying so on stderr; a repaired local copy gets
  its result and meta documents back, not just its files, and "already complete here" is
  now one definition instead of two that disagreed; blobs outlive the history listing by a
  day so a host with a fast clock cannot collect what other hosts still list; `pull` names
  the key that failed. `cache sweep` exists because an error message already told operators
  to run it - and because a bound nothing can run is not a bound. Its help, and `cache
  pull`'s, are now their own rather than `push`'s text. Several claims in this entry and in
  the design doc were overstated and have been corrected against what the code does.
- **Fourth review round (one reviewer, the whole branch): eight more, including the same
  window a third time.** Deduplication defeats BOTH earlier attempts at it - pre-listed
  candidates and a refreshed timestamp - because the blob is genuinely old while only the
  reference to it is new, and the test that was supposed to pin it passed for the wrong
  reason (with one key in the store, an empty live set refused the whole sweep). provender
  0.1.4 re-checks each candidate against the state it was listed in immediately before
  deleting it, which with the refresh is what finally closes it; `delete` lists before it
  scans, for the same reason. Also: a generation directory that is present but not current
  - a crash between its rename and the pointer write, a second process placing it, a
  pointer rolled back to it - is now ADOPTED rather than answered as a miss, which had
  made such a key permanently unreadable on that host and, on a compute server, let the
  next request overwrite a rollback; a store outage no longer turns a complete local copy
  into a miss, and a publication the store refuses keeps its bytes on this disk rather
  than discarding finished GPU work; local write failures during a read (a full or
  read-only cache) are misses, not 500s; an entry written by a NEWER haversack is never
  overwritten, while garbage still is; `executor.submit` joins the calls that run off the
  event loop; a slow fill's work directory is no longer reaped by age while its process is
  alive; and two processes sharing one `--cache-dir` wait for each other instead of both
  computing.
- **Fifth review round (the same reviewer, on the fixes it asked for): nine more.** Two
  were serious and both were in the new code. A filler killed between its claim and its
  release left that claim behind, and every later fill on that host then read "someone is
  placing it" and missed - permanently on a host that never computes, since only a
  successful publication of that key prunes a claim; death is now proved the way it is
  everywhere else here, and a live writer's claim (which holds a lock) still stands. And
  `adopt` published on existence alone, so a zero-length labels file - exactly what a power
  loss leaves behind a rename - was served as a 200; it checks the sizes the pointer
  records now. The purge's re-check was narrowed rather than closed, because S3 and R2 date
  objects to the whole second: it waits out the remainder, and only on a store whose own
  timestamps say that is needed. The outage fallback is bounded by when this host last saw
  the entry alive (15 minutes), so an unreachable store cannot resurrect a DELETED result
  indefinitely; a publication the store REFUSES keeps no local copy, since a refusal is
  deliberate and repeatable where an outage is not; the work-directory reaper now does what
  its comment always claimed; and a DELETE that will be refused no longer cancels a running
  compute first. The six sweep scripts are in `tools/config_sweep/` - results nobody can
  re-run are not evidence.
- **Configuration sweep: six environments nobody had run, three defects.** The reviewers
  could attack code but not environments. Two servers sharing one `--cache-dir` (the
  per-GPU deployment) came through clean - 446 reads, no torn read, a coherent directory
  afterwards - as did a store made slow rather than broken (`/v1/health` answered in 0.01 s
  while a store-touching route waited 1.5 s) and, for the first time off APFS, a cache root
  on exFAT with no hard links and case-insensitive names (9 checks: publish, read, pull,
  republish, push, history, sweep, delete, no leavings). What broke: an older haversack
  DELETED an entry written by a newer one, removing the index and leaving bytes it cannot
  name - refused now, 409 on the wire; a reader killed mid-fill left its work directory for
  an hour although its process was provably gone, where `cache usage` and `cache clean`
  cannot see it - death is proved and reclaimed at once now; and a fill needed TWICE the
  result's size in free space, because it downloaded into its work directory and then
  copied into place - a 6 MB result failed with 11 MB free. Fills hand the files over
  instead, which also stops reading and writing every byte twice, and a publication that
  lands nothing no longer leaves an empty entry directory behind.

### Also

- **Fixed: `haversack remote submit -o labels.nii.gz` wrote a `.seg.nrrd` under a NIfTI
  name.** The client asked for the server's own bytes whatever the output was called, so every
  reader refused the file (0.12.4 did the same). A `.nii.gz` or `.nii` name now asks for the
  server's NIfTI conversion - the label values survive, the segment names do not - and a
  `.nrrd` or `.zarr.zip` name that does not match what the job produced (labels or an
  embedding field) is refused before anything is written, naming the suffix to use. Any other
  name gets the server's bytes, as before. `remote fetch` and `remote encode` go through the
  same download. On a real server, `.nii.gz`, `.nii` and `.seg.nrrd` were voxel-identical.
- **American spelling everywhere, held by a test.** 105 British spellings (center, neighbor,
  millimeters, labeled, color and license, among others, in their British forms) are gone from
  the package, its docs, tests and tools, and `tests/test_american_spelling.py` fails on any new
  one, in prose or inside an identifier (snake_case and camelCase are split before matching).
  The reason is drift: each British word
  in the tree is a template the next edit copies. A line that must quote someone else's spelling
  says so with `spelling: allow <word>`, and a pragma that no longer matches fails too. The job
  state `cancelled` is unchanged - it is on the wire. Nothing computed moves; newly written
  ranked stores carry the respelled format README.

### Dependencies

- duckn is pinned at `v0.5.1`, seg extension 0.9: a segment may state a union once, as
  `members`. Stores here list values and are read as 0.9 files unchanged; the builder writes
  none, since a union belongs in a store only when its scheme defines it. Since `v0.4.1` a
  registry entry's `uri` and `url` are `system_uri` and `definition_url` - a URI *of* the
  coding system, a URL *of* this version's definition - and the schemes here use those names.
- rankfield is pinned at `v0.3.5` and feldglas at `v0.1.2`, in `pyproject.toml` and CI's
  list. rankfield 0.3.5 sizes the encoders' slabs from `memory_budget` (1 GiB default) and
  reads fields straight from a store; no bytes move. feldglas moves with it only because uv
  resolves a git dependency's own `[tool.uv.sources]`: 0.1.1 pinned rankfield `v0.3.3`, and
  two URLs for one package leave `haversack[encode]` unresolvable. 0.1.2 is the same code
  with the new pin.
- provender is pinned at `v0.1.6`: its `DiskStore` (a directory that honors both conditional
  writes) and `provender.ops` are what the result store runs on, on a directory and on an
  object store alike.

## [0.12.4] - 2026-09-19

- **A finished Modal job no longer answers 410 "purged" while its result is there.** On
  2026-09-19 (300 ts.v2:total jobs on IDC, 16 concurrent clients) `GET
  /v1/jobs/{id}/result` said 410 for 191 jobs whose results the workers had committed, and
  200 minutes later. Reproduced on a throwaway deployment, the cause was in the api
  container itself: while a Modal volume reloads, every path on it is missing to the
  container's other threads (measured: 734,714 of 735,326 listings, and none with no
  reload running), and the api reloaded the result volume on every request's thread - so
  a lookup racing another request's reload missed, and the miss was believed. The api's
  whole view of the volume emptied and refilled every 0.5-3 s under that load. Lookups
  and reloads now exclude each other within the container, and the api serves a
  container-local copy of each result rather than streaming the volume's file - which
  also ends the second cause, reloads that Modal refused (silently, until now) while
  streamed files were open. A miss is an answer only when read from a view newer than
  the request; when no reload takes, the answer is 503 with `Retry-After` - never 410,
  which now means a purge the server has verified. The by-path routes (labels GET and
  HEAD, `meta.json`, preview, statistics) and the anonymous twin take the same rule: a
  miss there was a 404, and with `Prefer: wait` a second compute of a result that
  already existed. A refused reload is logged, once a minute per volume. The api
  container's copies are capped by `HAVERSACK_API_MIRROR_GB` (default 2).
- **A Modal worker's job no longer reads its own files from a volume another thread may
  be reloading.** While one thread reloads a volume, every path on it is ENOENT to the
  container's other threads (measured on Modal 2026-09-19), and a worker runs the job beside
  the prefetcher, which reloads the scratch volume whenever an upload is queued behind it.
  The job dropped its volume lock and then read its upload from scratch through the whole
  compute, and read its saved labels back from there for the digest, the artifact pair and
  the cache put, so a prefetcher reload in between failed the job with FileNotFoundError.
  The job's open upload also made the prefetcher's reload raise, which ended the prefetcher.
  The job now copies its uploads, and saves its labels, to container-local disk under the
  lock and works from those copies. The cache put and its commit take the lock too: Modal
  reloads after a commit whenever its server asks, although in 61 measured commits it never
  hid a file. A refused scratch reload no longer fails a job whose upload is already visible.
  On Modal (two throwaway deploys, 24 distinct uploads submitted at once to one L40S
  worker), the code before this change failed 14 of 24 jobs with FileNotFoundError on
  their own `labels.seg.nrrd`, raised in the cache put. With the change, all 24 finished,
  and every published generation (40 over both rounds) held its preview and statistics.
  Probes on Modal also settled what the lock must cover: a reload hides only its own volume,
  so the weights and inputs volumes, which only the job thread touches in a worker, need
  nothing.

- **A multi-input Modal job no longer takes one role's upload for another's.** The worker
  found each role's file by the prefix `input_{role}_`, and a role is the model's own
  spelling, so for roles `t1` and `t1_ce` the `t1` channel could be given the `t1_ce`
  scan; a role containing `/` would have named a subdirectory. The role is now
  hex-encoded in the file name and matched exactly. A job queued before a redeploy, or
  run by a warm worker from before it, fails with a message naming the role instead.

## [0.12.3] - 2026-09-19

- **`GET /v1/jobs/{id}/result` reads the job's published result, not the worker's scratch
  copy.** On a Modal deployment (2026-09-19, ts.v2:total on 440 IDC series) it answered 500
  for 162 finished jobs, intermittently per job: the api container did not see the file the
  worker wrote to its scratch volume, while the same result by path answered 200 for every
  one. The route, with or without `?format=`, and `POST /v1/inputs` `from_job` now resolve
  the job's key through the leased cache lookup the path surface uses, and fall back to the
  job's own copy only where there is no entry, or where a later recompute republished the
  key with different bytes. With neither copy the answer is 410, as SERVER.md always said,
  including when the file leaves between the check and the conversion. A Modal job answered
  from the cache at submit now records its key, as the local server's does.
- **A Modal read no longer joins a flight a stopped deployment left behind.** Stopping an app
  mid-job leaves its records `queued` or `running` with their `inflight:` markers, and only a
  worker's reconcile failed them - so after a restart with no new jobs, a plain GET of such a
  key joined a flight that would never land and waited its full 30 s (the 1,206-job run of
  2026-09-19 fell back to HEAD). The API's single-flight lookup now applies the reconcile's
  rule itself: a record older than two minutes whose call is gone is failed on the spot and
  the read answers at once. A live call is probed at most once per 30 s per job.
- **Review of both fixes and of 0.12.2's Modal changes** (four adversarial agents, each
  finding reproduced before it was fixed):
  - *Two concurrent uploads deadlocked the Modal api container* (since 2026-08-27). The
    upload held the volume guard, a thread lock, across an `await`; a second upload's
    acquire blocked the event loop the first needed. The write now runs in a thread.
  - *A transient Modal error failed a running job.* Every exception from a call probe
    counted as dead; only the answers that say a call ended do now, and anything else is
    treated as alive. This mattered once reads of a key began probing.
  - *A running job could lose its call id*, since the API's and the worker's updates of the
    record race; the worker now records its own. Prepare jobs never recorded one, and the
    reconcile failed any that ran past two minutes.
  - *The sweep deleted job records before their directories*, so a container stopped
    mid-sweep leaked directories nothing names. It now removes the directories first.
  - *The job result route* treated an entry without a digest as the job's bytes (it serves
    the job's own copy now), advertised an evicted job's result that then answered 410,
    and still gave a 500 when the job's own copy vanished just before a plain download.
- **Second review round and a Modal smoke** (`haversack-upload-smoke`, torn down):
  - *A warm Modal worker was poisoned by a cancelled job.* A cancel raises in the job
    wherever it is, and on the smoke it landed mid-import: the next job in the container
    failed on a half-initialized `torch._dynamo`. A container retires after any job
    cancelled while it ran, once earlier jobs' artifact threads have finished (a daemon
    killed by the exit left its `artifacts:` marker answering 202 for up to 15 minutes).
  - *A crashed worker's job would have stayed running forever* under round one's rule,
    because Modal reports a lost container as InternalFailure. Now only a short list of
    transient errors (the caller's own connection, throttling, service errors) reads as
    unknown, and every other failure of a probe means the call ended.
  - The content store's volume commit ran on the event loop during every upload; it runs
    in a thread now. The job result download closes its file when a client disconnects
    rather than at the next garbage collection. It no longer answers byte ranges, which
    FileResponse did and no haversack client uses.
  - The smoke: two and four concurrent 3.7 MB uploads all answered 202 in 5-12 s, and the
    worker's own call id matched the spawn's.
- **A Modal job DELETEd while it ran now stays cancelled.** On the smoke, a job cancelled
  during model loading finished, published its result, and reported `done`: Modal's cancel
  (a signal raising in the job) never reached the worker, the job's own check runs only at
  a patch, and a worker's `done` could replace the API's `cancelled`. The worker now asks
  for the cancel once more before it saves or publishes, as the local server does, and a
  `cancelled` record is final. Verified on a smoke (`haversack-cancel-smoke`, torn down): a
  job cancelled in loading stayed `cancelled`, its result answered 409, the same bytes
  resubmitted computed afresh (nothing was published), and the container retired.

## [0.12.2] - 2026-09-19

- **A Modal deployment accepts concurrent submits concurrently again.** `POST /v1/jobs`
  called the executor inline in an async route, and on Modal that is a dozen blocking RPCs:
  every submit held the container's event loop for 1.48 s, so 8 clients got 0.67 submits/s
  between them (0.62 on a 760-job IDC run). The hand-over now runs in a worker thread, and a
  submit commits the scratch volume only when it wrote something there - that commit was
  0.67 s of every `idc:` submit. Measured on throwaway deploys, 200 submits from 8 threads:
  6-8 submits/s with one worker or six.
- **A Modal worker no longer spends ~2 minutes between jobs on housekeeping.** After every job
  the worker swept the whole jobs Dict inline, holding its volume lock: one `get` per record,
  another per `inflight:` marker's target, and a call probe per record queued longer than two
  minutes. With the Dict a batch builds up (1342 keys, 558 of them queued, on 2026-09-19)
  that cost ~130 s per job. It set throughput, because `run_job` did not return until the
  sweep finished: 6 L40S workers ran ts.v2:total_fast at ~2.6 jobs/min when inference takes
  seconds. It also showed up in the log as `preview 127s`, because the preview's timer
  included the wait on that lock (the render itself takes ~1 s). Every worker scan of the Dict,
  including the prefetcher's every 2 s, is now one streamed `items()`. The sweep runs in a
  background thread at most once a minute per container, never two at once, and holds the
  volume lock only for file operations. The listing is not atomic, so whatever the sweep
  deletes or fails is re-read first, and a flight that started since the listing keeps its
  marker. The job's own input is still deleted inline. The overlap log's seconds now cover
  the render only. On smoke deploys with 2 L40S workers and a 1300-record Dict: ~1 job/min
  before, ~10.7 after. With six workers and 200 queued jobs, the jobs drained during a 30 s
  submit burst went from 15 to 199.

## [0.12.1] - 2026-09-19

- **A series with a directory marker in its bucket fetches again.** Some IDC series carry a
  zero-byte marker object `<uuid>/`, and obstore lists it with the slash stripped, as `<uuid>`,
  so the prefix fetch asked for an object that does not exist and failed the whole series with
  `NoSuchKey` (`idc:15fc0810-a2e2-4b32-8c3c-217ebc92ba32`, colorectal_liver_metastases: 1 of 5
  random abdominal CT series tried). The listing now drops a key that is the prefix itself or
  a zero-byte strict path-prefix of another listed key, for `idc:` and every `<bucket>/<prefix>/` fetch.

## [0.12.0] - 2026-09-13

- **The segments index: what every task produces, before anything is installed.** Most
  catalogs could list a task's segments only once its weights were on disk, so "which tasks
  produce a pancreas, and with what label value" meant downloading gigabytes. `haversack
  catalog mine` reads each list from where the model states it - a checkpoint's `dataset.json`
  read out of its remote zip by Range (a few KB per model), a MONAI bundle's metadata at its
  curated version with the commit that version names, an engine's own table, TotalSegmentator's
  registry - and records the version that pins it; `haversack catalog check` says, offline,
  which records a catalog change has made stale, for everything or for the catalogs and tasks
  named. `mine --all` mines all 103 tasks in seconds; naming a catalog or a task updates only
  that. An archive its manifest pins by digest is not read again while that digest - and the
  listing rules, a number in every record's version that `check` holds the index to - is
  unchanged, so a version change from outside the archive costs no network; `--reread` reads
  everything. A failed fetch keeps the previous record and fails the
  run, and where a model is installed at the same version its own labels must agree with its
  archive's, or the list is not recorded (all 23 installed on the development machine agreed).
  The index ships as `data/segments.json` (2142 segments) and is derived, never used to run
  anything: an installed model's own labels still decide every result. A segment is DICOM's
  and duckn's: an `id` - the model's own token, a code in its class list rather than a display
  name or an identity across models - a label `value`, and a `layer` where the output
  overlaps, since segments need not be disjoint. The record has room for duckn's group
  `members`, display names and designations without another schema. Mining is a method on the
  ecosystem, so the zip-manifest and image-baked shapes carry it and a new catalog on either
  needs no edit here; the suite fails for a catalog that cannot answer, and for a shipped
  record whose version no longer matches its manifest.
- **Search the segments: `haversack tasks --find` and `GET /v1/segments`.** Which tasks
  produce a pancreas, and with what label value, is one query: word prefixes in any order by
  default (`--find "kid left"` finds kidney_left and left_kidney), a glob over the whole id, or
  - locally only - a regular expression. It lives under `tasks` because the answer is tasks,
  and because a top-level `segments` would sit one letter from `segment`, the verb that runs a
  model. Ids are compared folded for case, spaces and hyphens, with each
  model's spelling kept beside them; grouping ids that fold alike is a match on spelling, not a
  claim that two models mean one thing, and it is not an ontology - an abbreviation or a
  synonym is not found. The server answers from the tasks it serves, to anonymous callers as
  `/v1/tasks` does, and refuses regex, since a pattern from anyone can take unbounded time to
  evaluate; SERVER.md gives the answer's shape and its errors. `RemoteClient.segments` speaks
  it. `haversack tasks TASK` now prints a model's segments from the index when it is not
  installed, where it used to say to install it first. A user index (`HAVERSACK_SEGMENTS`,
  else `~/.config/haversack/segments.json`) lays over the packaged one for search and
  `catalog check`, and is where `catalog mine` writes from an installed package - or from a
  checkout, when the variable is set.
- **Long answers are paged, and say where they end.** An agent whose harness cuts long tool
  output sees the start of an answer and cannot tell it was cut. `tasks --find` and
  `GET /v1/segments` now return `limit` ids from `offset` in a fixed order, with a precise
  `truncated` and a `next_offset` (null on the last page); `--count` / `count_only` answer
  with the counts alone. The counts come first and a receipt comes last - the CLI's `#` header
  and `# end:` line on stdout with the results, the JSON's trailing `end` - so an answer that
  lost its tail is missing its end, however complete it looks. The server's answer carries an
  `index` version, which changes if the index is rebuilt between pages. The task listings
  answer which tasks exist and leave the rest to one call per task: `haversack tasks --json`
  counts each task's structures (`n_structures`) instead of listing them with its label map,
  and neither it nor `GET /v1/tasks` carries each task's attribution any more - 204 KB of the
  CLI listing's 347 KB, largely the same citations repeated across a catalog's tasks.
  `haversack tasks TASK --json`, `haversack cite TASK` and `GET /v1/tasks/{task}` give one
  task's. The CLI listing is now 24 KB.
- **A MOOSE task's modality no longer changes when it installs.** MOOSE's catalog took the
  modality from the task name before install and from the checkpoint after, and two
  checkpoints misstate theirs: `preclin_mr_all`'s `dataset.json` names its channel "CT", so
  an MR model reported CT once installed, and `clin_pt_fdg_face` went from PT to "PET". The
  generator now records the modality the name states in the manifest, and every zip catalog
  applies its manifest's modality in `spec()` - one rule where TotalVibe alone had it - so
  `info()`, `describe()`, result provenance and the segments index give one answer. Only
  that metadata changes; no label is computed differently, so no cache epoch moves.
- **nnU-Net's `ignore` label is no longer listed as a segment.** nnU-Net never predicts it -
  its label manager skips the key by name and requires it to be the highest value, one past
  the rest - but haversack read it into an installed model's label map like any other, so
  TotalVibe's `vibe` and `vibe_sagittal` reported 73 structures for their 72, in `describe()`,
  in a result's `names` and in the segments index. It is a role a value plays, as background
  is, and is now dropped where background is, in the one parse installs and the miner share.
  No label value changes, and a `.seg.nrrd` lists only the segments present, so no result's
  labels differ and no cache epoch moves; a cached result of those two tasks keeps the unused
  name in its `names` until it is recomputed.
- **SynthStrip names its label on its engine row.** The ranked builder asks the engine row
  for label names and fell back to TotalSegmentator's for anything without them, which for
  `synthstrip:mask` meant none; the row now carries `{1: "Brain"}`, and the engine reads its
  mask's label from there too.

- **The server honors `task@version`.** It dropped the pin before a job existed, so
  `POST /v1/jobs` with `ts.v2:total@X`, and a pinned GET, ran or served whatever version was
  installed - the silent wrong version the grammar exists to prevent; only `prepare` held it.
  Now a pin the server provably does not run is a 409 naming what it runs, on every route. One
  it cannot check yet (nothing installed, or no version record) is carried with the job to the
  worker, locally and on Modal, whose catalog installs that version or refuses it. Such a job
  is never answered from the cache or by joining another flight, and no other request joins
  it until it has re-keyed on what it actually installed - a review found a plain request
  riding a pinned job and inheriting its failure. A read cannot install, so an unverifiable
  pin there is a 409 pointing at `POST /v1/jobs`; `GET /v1/tasks/{task}@v` still describes
  it. Modal's API container reloads its frozen weights volume before calling a pin
  unverifiable, the public twin checks pins against versions the deployment gives it, image-baked
  workers re-check a pin against their own build, and the MONAI worker hands the pin to its
  catalog (it used to read `bundle@version` as an unknown bundle). Jobs report `version`,
  on Modal too, including a pinned ask answered from the cache. Run on Modal before release:
  a wrong pin refused at submit on an installed task and by the worker's catalog on an
  uninstalled one, a plain request kept out of a pinned job's flight, a pinned read
  answered once the API's volume caught up with the worker's install, and pinned MONAI and
  FastSurfer 2.5.4 jobs completed.
- **`POST /v1/jobs` keeps a rename hint.** Submitting `fastsurfer:brain`, or `ts:total_fast`
  since 0.11.0, answered a generic "unknown task ... this server offers N tasks": the route
  kept the resolver's own words only when they said "needs its catalog". Now only a plain
  unknown task is replaced by that sample; a renamed task or catalog is named with its new
  form, as the CLI and the Python API already did.
- **One torch for every engine: VoxTell's `torch<2.9` is overridden, and everything resolves
  torch 2.14.** VoxTell pinned below 2.9 because torch 2.9.0 slowed 3D convolutions under
  mixed precision (pytorch#166122); that was fixed in 2.10, and VoxTell's authors report 2.10
  and 2.11 fine. `[tool.uv] override-dependencies = ["torch>=2.10"]` lifts the pin, and a
  VoxTell worker on torch 2.14 (CUDA 13, L40S) was run on Modal: a chest CT, four prompts,
  plausible volumes, about 8 GB of GPU memory. An override applies to the whole graph, so
  SynthStrip and MONAI move to 2.14 too, and both were run on it on Modal: SynthStrip (torch
  2.14 beside the numpy 1.26 surfa needs) stripped a T1 to a 1282 mL brain, and MONAI 1.6
  ran `spleen_ct_segmentation` and `wholeBody_ct_segmentation` (all 81 labels non-empty) on a
  chest CT. The FastSurfer fork asks for torch 2.14 and its torchvision (0.29) as its floor.
- **haversack runs on Python 3.12, for now (`requires-python = ">=3.12,<3.13"`).** FastSurfer
  fails on 3.14 - it passes `str | None` as an argparse type, which 3.14's argparse refuses -
  and upstream lists nothing past 3.13. The suite passes on 3.13 too, but every Modal image,
  the development environment and every measured run is 3.12, so that is what ships until
  FastSurfer is brought up to current Python. `.python-version` asks uv for 3.12.
- **FastSurfer's task is `fastsurfer:asegdkt`, FastSurfer's own name for the module.**
  `brain` was a name haversack made up, and it would not have survived FastSurfer's other
  segmentation modules (`cereb`, `hypothal`) arriving in the same catalog. `fastsurfer:brain`
  is refused naming the new form, as `ts:` was in 0.11.0.
- **haversack runs FastSurfer 2.5.4, a release, and says so.** The `fastsurfer-lean` fork was
  cut from an August dev snapshot - code the maintainers had not released or validated. It is
  now upstream's v2.5.4 tag with only `pyproject.toml` changed (`v2.5.4-lean2`), and it asks
  for the latest torch (2.14.0) and its torchvision rather than upstream's 2.7. Of what the
  engine calls, only `conform`'s handling of float noise in voxel sizes near 1 mm differs from
  before. The weights identity is the release (`fastsurfer=2.5.4`) rather than the checkpoints'
  `vinn-v2`, which every 2.x release shares, and a test holds it to the tag; FastSurfer's
  cached results recompute once under the new key.
- **A pinned version an image-baked engine does not run is refused.** `fastsurfer:asegdkt@X`
  (and SynthStrip's and VoxTell's) ran the one build there is whatever `X` said, because an
  engine task never reached the check an nnU-Net task's pin goes through. The version the
  build runs is accepted; any other is refused before the engine starts.

## [0.11.0] - 2026-09-12

Two changes break what worked before, and a catalog arrives. A task name now names its catalog -
`ts.v2:total_fast`, not `total_fast` - and TotalSegmentator's catalog is `ts.v2`, so a bare name
and the old `ts:` are refused, with the form to write instead. The body envelope is off by
default and `0` means the whole volume everywhere, which changes the labels a default run
computes. CADS arrives: nine whole-body CT models, 167 structures, open weights. Beside them,
the Triton restore takes fields past 2^31 logits, a first-use weights install is timed and
reported as its own step, FastSurfer runs finer-than-0.7 mm scans at 0.7 mm and restores on the
GPU in pieces, a Modal deploy made with `--app-name` runs as itself, and `rights` points a task
name at `cite`. Cached results recompute once: the global cache epoch moves to 3, and
TotalSegmentator's results are keyed by their new name.

- **Task names carry their catalog: `ts.v2:total_fast`, `cads:organs`, `totalvibe:vertebrae`.**
  A bare name resolved to whichever installed catalog offered it, so what a script meant
  depended on what else was installed: adding CADS turned `vertebrae` from `totalvibe:vertebrae`
  into an ambiguity error, and TotalSegmentator v3 reuses v2's names. A bare name is now refused
  even when one catalog alone offers it, in the CLI, the Python API and on the server, and the
  error names the qualified form (`task 'total_fast' needs its catalog: use ts.v2:total_fast`).
  Names stay the model makers'; haversack does not rename a task to avoid a collision. Results
  are keyed by the qualified name, so TotalSegmentator's cached results recompute once under
  `ts.v2`; none is served wrong. Shell completion offers qualified names and still finds them
  from the task's own name, and a cascade's reference to a task of its own catalog (`teeth`
  crops from `craniofacial_structures`) is looked up in that catalog. TotalSegmentator's catalog
  is `ts.v2` now - it is TotalSegmentator v2's, and v3 reuses v2's task names, so v3 can arrive
  as `ts.v3` beside it; the family is what comes before the dot. `ts:total_fast` is refused with
  the `ts.v2:` form to use, and a store written with a `ts:` or a bare name still reads.
- **`envelope_mm=0` cropped inference flush to the skin in Python and on the server, while
  `--envelope 0` ran the whole volume.** One number, two meanings: a CADS parity study passed 0
  believing it meant "no envelope" and measured, against upstream, mean Dice 0.64 on the face,
  0.69 on head muscles, 0.72-0.78 on the mammary glands and 0.82 on CSF, where the whole volume
  gives 0.95-1.0. `0` (or less, or `None`) now runs the whole volume at every door - `segment()`,
  `Segmenter`, the server's `envelope_mm` and the command line - and provenance records
  `envelope_mm: null` for it. A margin that is not a finite number is refused.
- **The envelope is off by default; it was 20 mm.** Cropping re-tiles nnU-Net's sliding window
  and the labels move with the tiles - not only near-ties at the new seams. On a chest CT (0.7
  mm, 465 slices), against the whole volume, the 20 mm envelope moved 0.50 % of `total_fast`'s
  voxels (pancreas Dice 0.74, portal and splenic vein 0.83), relabeled 17 % of `body`'s
  `body_extremities` as trunk (Dice 0.87), and left `breasts` at 0.995. No margin buys it off:
  `body_extremities` went 0.95, 0.94, 0.87 and 0.97 at 0, 10, 20 and 40 mm. The speedup is real -
  up to half the patches on the chest and whole-body CTs measured, none on some - and
  `--envelope 20` / `envelope_mm=20` still asks for it, but it is a trade and not a free win. The
  default now tiles the whole volume, as nnU-Net's own predictor does.
- **An envelope crop narrower than the network's patch is grown to it from the image.** nnU-Net
  pads such a crop with 0 after normalization, which is the model's mean foreground intensity -
  +120 HU for CADS's head model, -89 HU for TotalSegmentator's breasts model - so the network saw
  tissue a margin's width outside the skin. On a head CT that happens at any margin: at 1.5 mm a
  head is smaller than CADS's 160-192 voxel patch on every axis. The grown crop is the shape the
  padded one was, so it costs no network time; on the chest `total_fast` it took the pancreas
  from 0.74 to 0.91. Padding with air instead was tried and rejected: it also changes
  whole-volume runs, which pad a volume shorter than the patch with 0 just as upstream does (0.89
  % of voxels on an abdominal CT).
- **A crop that saves no patches runs the whole volume.** The envelope judged a crop by the box's
  volume, but the window steps half a patch, so a crop that removes a third of the volume can
  still need every tile: on a chest-abdomen-pelvis CT the 10-40 mm crops ran all 54 of the whole
  volume's tiles at TotalSegmentator's 128^3 patch, re-tiling for nothing. It counts tiles now,
  and a crop that saves fewer than 5 % of them gives way to the whole volume.
- **A cascade's crop follows the same two rules.** The fine model of a cascade task
  (`lung_vessels`, `abdominal_muscles`, ...) still runs only inside the box its coarse stage
  found - that crop is not the body envelope and stays on - but a box narrower than the patch is
  now grown from the image rather than padded, and one that saves no tiles runs the whole volume.
  Both can change a cascade's labels.
- **CADS** (`cads:*`, weights CC BY-SA 4.0): whole-body CT, 167 structures in nine models -
  `organs`, `vertebrae`, `cardiac`, `muscles`, `ribs`, `oar`, `head`, `headneck` and
  `bodyregions`. The weights are upstream's `open` release; its non-commercial `research` and
  challenge-gated `reference` weights are not offered. Upstream reorients to RAS before the
  network while its checkpoints say nothing about orientation, so the catalog runs them in the
  TotalSegmentator lineage: read as stock nnU-Net models every left/right structure lands on
  the wrong side, and with the lineage all nine matched upstream at 0.998-1.0 over the whole
  volume with `--interp nearest`, which is how upstream restores its labels (`cads:organs`
  through the catalog on MPS: 99.9995 % of voxels). The default linear restore moves
  boundaries off upstream's - `organs` at mean Dice 0.970, its adrenals at 0.91 - so a
  comparison with upstream wants `--interp nearest`. Each task is one model and there is no
  combined task, because the nine overlap by design (upstream's own combined map paints the
  thoracic cavity over every lung lobe). haversack does not gate `head` and `headneck` on a
  brain or post-process, and it pads the edge of a short volume the TotalSegmentator way (a
  34-slice head CT: 0.977-0.998).
- **`haversack rights` given a model task name says to use `cite`.** `rights totalvibe:vibe`
  ended in a raw `KeyError`, and `rights ts.v2:total` called the task a local file. `rights`
  reports on data sources; a prefix naming a model catalog now exits 2 pointing at
  `haversack cite <task>`, and any other unknown prefix exits 2 listing the source kinds
  `rights` takes. What it reports for a real source is unchanged.
- **FastSurfer runs a scan finer than 0.7 mm at 0.7 mm**, the finest voxel size FastSurfer
  validated (its own help calls finer "experimental"), and restores the labels onto the
  scan's own grid, with a deviation saying so. A 0.5 mm scan ran on a 512^3 grid whose
  79-class field is 21 GB of fp16 and cannot exist as one MPS buffer. A genuine 0.7 mm scan
  is not floored: a header's float32 0.7 is compared at FastSurfer's own precision.
- **A FastSurfer field too big for one MPS buffer is aggregated on the host**, as FastSurfer's
  `viewagg_device=cpu` does, with a deviation; `view_aggregation_device` in provenance is
  where the field actually lived. The host path keeps the field in fp16 instead of widening
  and transposing it: peak memory at 384^3 went from 51.5 to 24.7 GB, and the CPU restore is
  4x faster. Labels and ranked arrays are byte-identical to before on a 1 mm T1.
- **FastSurfer's GPU restore works a group of channels and a slab of the target at a time.**
  It widened the whole field to fp32 and sampled all 79 channels at once - 42 GB for a
  512^3 target - so a Modal A10G ran out of memory above 256^3. Labels are unchanged (0 of
  899,376 differ over four geometries); 384^3 and 512^3 inputs complete on an L40S at 14.4 GB.
- **An engine can move its own cache epoch.** The FastSurfer floor changes FastSurfer's bytes
  alone, so it bumps `Engine.cache_epoch` for FastSurfer (`fastsurfer@epoch=1` in the key)
  rather than the global epoch, and no nnU-Net result is recomputed for it. The Modal worker
  keyed FastSurfer results without that epoch, publishing them where the API never looked;
  it keys them as the API does now.
- **A Modal deploy made with `--app-name` runs as itself.** Its containers did not receive
  `HAVERSACK_APP_NAME` and ran as `haversack-serve`: the FastSurfer worker committed a scratch
  volume that was not mounted, so every job failed "volume ... not attached", and job records
  went into the default app's store. Every setting the module reads at import now reaches the
  container, checked by a test that reads those reads from the module's source.
- **An engine parameter that shadows a processing option is refused** when the wire model is
  built. The documented "loud error" had no test, and pydantic silently merged the two.
- **`segment` failed outright on CUDA when a model with many classes met a large grid.** The
  fused Triton restore addressed the whole logit field with 32-bit offsets and refused any field
  of K x Z x Y x X of 2^31 or more, with a `ValueError` nothing caught. The open CADS
  head-and-neck model (K=30) on a whole-body PET/CT's attenuation CT with `--envelope 0` is 30 x
  678 x 334 x 334 = 2.27e9 logits. The 20 mm body envelope, the default until this release, had
  kept most runs under the line; with the whole volume now the default, that model would have
  failed on such a scan every time. Only each channel's base needed 64 bits - the Metal kernel
  has always had it that way - so that field now restores on Triton. On an A10 its labels equal
  the torch backend's (fp16; in fp32 two voxels differ, both ties within 5e-7), and they are
  bit-identical to the old kernel's wherever the old one ran, at no measurable cost: over 30
  interleaved runs on one A10, a K=118 field restored in 54.9 ms (median) against the old
  kernel's 55.7, and a K=30 field in 23.5 against 24.1. Nothing that ran before computes
  different labels, so the cache epoch stays where it is.
- **What a fused kernel still cannot address, "auto" now restores with the torch backend and
  says so, instead of failing.** That is a channel of 2^31 voxels or more (a 1290^3 model grid,
  on either kernel), or for Triton an output of 2^31 or more (a whole body at about 0.5 mm),
  which the old kernel did not check at all: its 32-bit output index wrapped, and on an A10 a
  2048 x 1024 x 1025 output died of an illegal memory access. The torch backend has no such
  limit but is far slower (1.0 s against
  0.02 s for the field above), so the run records it: a `note: restore backend` line on the
  command line, a `restore` progress stage on the server, and an entry in
  `provenance.deviations`. `to_labels(backend="auto")` called directly warns instead; a
  backend asked for by name still refuses a field it cannot take, naming `backend='torch'`.
- **A weights install on first use is its own step in `segment`'s timings and progress.**
  `segment` resolved the task inside its `read+canonical` timer, and resolving a catalog task
  whose weights are not in place downloads them: on a fresh Modal container, 29 s of a
  `cads:headneck` run's "read+canonical" was its 760 MB weights, with no progress while they
  came, where the read itself takes about half a second. The install is now timed as
  `weights:<task>` and reported as a `weights` stage carrying its download progress, and
  `read+canonical` times the read. A cascade's coarse stage resolves the same way.

## [0.10.2] - 2026-09-11

One fix: a Ctrl-C now ends every command by SIGINT, wherever it lands. The flaky CI test that
found it waits for a waiting command now. Nothing `segment` computes changes, so no cache is
invalidated.

- **A Ctrl-C while a command was starting up ended it with "Aborted!" and exit 1.** Since the
  command line moved to click, a Ctrl-C inside a command is held and raised again once click
  is done, so the process dies of SIGINT - the status on which a shell's `for` loop over a
  folder of scans stops. One that landed in click's own code instead (parsing the command line,
  a lazy import, the dispatch before the command's function runs) became click's "Aborted!"
  and exit 1, and the loop went on to the next scan. Raised at 1006 points across the startup
  of `remote status`, the interrupt came out that way at 996. It now ends the process by
  SIGINT as well; click still prints its "Aborted!" first.

## [0.10.1] - 2026-09-11

Two groups of fixes. Three give back GPU memory the sliding window was holding - above all an
out-of-memory fallback that could not free what had run out. The rest stop `get` and `segment`
from quietly doing less than they were asked, or something else: a DICOM series converted
without the checks `segment` applies, a folder of several series read as one of them, a local
path not written at all, two outputs written over each other. None of them changes what
`segment` computes from an input it accepts, so the cache epoch stays where it is.

- **`get -o scan.nii.gz` wrote DICOM series that `segment` refuses, as if nothing were wrong.**
  Converting a series read it with a bare SimpleITK series reader, which skipped the two
  things `segment`'s reader does to every series: build the geometry from the slice
  positions, and refuse a stack of them that is not one uniform grid. ITK instead regrids a
  series with missing slices onto its mean step and says so only on stderr - IDC series
  `33754f28-f0fe-4bbe-bba9-a98e4a4926ef` (eay131), 3 mm slices with four 6 mm gaps, came out
  on a 3.0577 mm grid with slices up to ~2.9 mm from where they were acquired - and it takes
  the slice axis's sign from a negative SpacingBetweenSlices, the Philips head-to-foot
  convention behind 2026-08's head-down CPTAC-CCRCC body; on a synthetic series of that kind
  it reversed the axis about the first slice, putting the last of six 10 mm from its place.
  Once written, the NIfTI is a clean uniform grid, so `segment` on it could no longer see
  what it refuses on the source. A series is now read as `segment` reads it: the same
  refusal, before anything is written, and the geometry from the positions. There is
  deliberately no flag to write a gapped series anyway - any one volume of it misplaces or
  invents slices - so the refusal names the way out instead: `-o <directory>/` still copies
  the series as fetched. Where ITK was already right the output is unchanged, voxels and
  header geometry identical. Nothing `segment` computes changes, so no cache is invalidated.
- **The out-of-memory fallback could not free the memory that had run out.** When the sliding
  window ran out of GPU memory with the accumulator on the device, the host retry ran inside
  the `except` block. There the exception's traceback still held the failed attempt's
  activations and accumulator, so `empty_cache()` released nothing and the retry ran beside
  them. A CADS ResEnc-L checkpoint in fp32 on a 22 GiB A10 failed an 864 MiB allocation that
  way with 19 GB still held (2026-09-11). The retry now runs after the handler. A batch above 1
  is first retried at batch 1 on the device, which is what the measured placement policy
  approved, before the accumulator moves to the host. The deviation recorded when a forced
  `--accumulate device` falls back had always given an empty reason, because it read a key that
  was never set; it now says why.
- **Two patch outputs sat on the device where the working set was measured with one.** The
  first patch's output stayed on the device for the whole sliding window, kept alive by an
  argument binding: about 510 MB for an fp32 K=18 model at a 192³ patch. Every later patch's
  output also stayed alive through the next forward pass; on the batched path, a leftover row
  view kept the whole batch's. Each is now freed once it has been added. Neither change alters
  a voxel: both lower the peak that the placement and batch policies decide from, so no cache
  is invalidated.
- **`get` wrote nothing for a local path, and exited 0.** `haversack get ./series -o
  scan.nii.gz` printed the folder back: a local path was returned before `-o` was looked at,
  and a batch with `--format` printed its local paths and converted none. A local path - a
  file, or a folder holding one DICOM series - is now written as a fetched source is, converted
  for an image extension or `--format` and copied for a directory `-o`, which makes `get` a
  converter. Given neither option it is printed back as before, since there is nothing to
  fetch.
- **Three more ways `get` did less than it was asked, each exiting 0.** `--format` without
  `-o` converted several sources into the current directory but was dropped for one, which
  printed where the data was; it converts into the current directory either way now.
  `--no-cache` held only for conversions: every raw copy (`-o dir/`, `-o` a file, a batch
  without `--format`) fetched into the cache and left it there. And two sources of one batch
  that name one output - `a/scan.nii.gz` and `b/scan.nii.gz`, the ordinary layout of a folder
  of cases once local paths are written - went to one file, the second replacing the first,
  or, copied raw, into one folder, file over file; the second now fails, naming the first,
  and the run exits 1. Names are compared as APFS and FAT compare them, so `Scan` and `scan`
  are one.
- **`get` never writes a source into itself.** Reachable only now that local paths are
  written: `get ./scan.nii.gz -o .` would copy the file onto itself, `-o scan.nii.gz` rewrite
  it in place, and a folder copied into itself nest a copy one level deeper on every run. Each
  is refused in one line, decided by the filesystem, so a symlink or a case-folded spelling
  counts.
- **A local `.` is named by the folder it is** where an output is named after its source,
  and a `!` in a local name is part of the name: `segment . --format seg.nrrd -o out/` wrote
  `out/._<task>.seg.nrrd`, a hidden file, and `get` would have written `out/..nrrd`.
- **A folder holding several DICOM series was read as one of them, without a word.** Asked
  for the series in a directory, GDCM answers with the first it finds and says nothing of the
  rest, so a folder of two series - 3 and 5 slices - read as the 3-slice one: `segment
  ./study/` segmented one series of several, and `get -o scan.nii.gz` converted one. Both
  now refuse such a folder in one line that names its series and the fix, a folder holding
  one of them; there is no flag to pick one, since which was wanted is not something a reader
  can know. The server already refused one at upload. The refusal also reaches what a fetch
  can deliver: a zip's `!<folder>/` or a bucket's `<prefix>/` flattens every series under it
  into one directory, and ending the source at one series' folder reads that series. A CT
  beside its RTSTRUCT still reads, since an object without pixel data is not a series. A
  folder that read before reads the same bytes, so the cache epoch is not bumped - which
  means a server that cached a result from such a folder keeps serving it until it is
  deleted, rather than every result being recomputed to drop it.
- **`segment`'s batch wrote two inputs onto one output, and exited 0.** Each output is named
  `<stem>_<task><ext>` in `-o`, so two inputs sharing a stem - `a/scan.nii.gz` and
  `b/scan.nii.gz`, the ordinary layout of a folder of cases, or two Zenodo records each holding
  a `scan.nii.gz` - were both written to `out/scan_<task>.seg.nrrd`, and only the second's
  labels were there at the end, the first's inference spent for nothing. Such a pair is now
  refused in one line naming the fix, before anything is fetched or run: an output's name
  depends only on its input as given, so every name in the batch is known before the first
  input starts. Names are compared as APFS and FAT compare them, so `Scan` and `scan` are one,
  as is a name typed composed and decomposed. A lean install, asked for torch only after these
  checks, now also hears about a mistyped `--format` rather than about torch.

## [0.10.0] - 2026-09-11

The command line moves onto click, which gives it `--version` and shell completion and changes
one thing for scripts: an option must be spelled out in full. Nothing computed changes, so no
cache is invalidated and nothing is recomputed.

- **The command line is click's now, and says its version.** `haversack --version` prints the
  release - argparse's command line had no such option, and a `uvx` smoke run found it
  failing - and shell completion comes with it: commands, options, and task names from the
  catalog (the README has the one line for a shell's startup file). One thing changes for
  anyone scripting it: an option must be spelled out, since argparse took a unique prefix
  (`--env 5` meant `--envelope 5`) and click refuses one. Exit statuses are unchanged,
  a Ctrl-C's included - click would have made it exit 1, on which a shell's loop over a
  folder of scans goes on to the next. Help reads a little differently: defaults print as
  `[default: fp16]`, and no longer as `(default: None)` where there is none. The `weights`
  line in the command list, cut off mid-sentence since it was written, is finished. `click`
  becomes a core dependency, so the lean install adds it. Inside, `cli._run` stops being one
  ~850-line function: each command is a function of its own.

## [0.9.1] - 2026-09-11

A performance and correctness fix in how an input's provenance is recorded. The labels
computed are the same bytes, so no cache is invalidated and nothing is recomputed.

- **A local input's provenance could name a DICOM series it was not.** The DICOM
  identifiers recorded for an input were read from the directory it sits in, and for a
  file given on the command line that is whatever folder the caller keeps it in - so a
  NIfTI beside a DICOM file of another scan was recorded as THAT series, and GDCM parsed
  every file in the folder on the way (a scan in `~/Downloads` meant all of
  `~/Downloads`, with an ITK warning on stderr when none of it was DICOM). A file is now
  described by itself, and only a DICOM file carries DICOM identifiers; a directory is
  read as before, whatever number of files it holds.

## [0.9.0] - 2026-09-11

Acting on an external review of 0.8.0, on three adversarial reviews of the fixes, and on
the external review's further passes over the result cache those fixes rebuilt. Twelve
findings were raised first; all twelve held under verification, and checking them turned
up four more the review had missed. **Anyone running a server or the CLI should take this
one**: three of the defects silently changed what was segmented or what a result claimed
about itself. Both cache epochs move, so stored results are recomputed and fetched inputs
fetched again, once.

### Bytes that were lost, mixed, or claimed wrongly

- **A prefix fetch could lose slices of a DICOM series.** Every object under an `idc:`,
  `s3:` or `gs:` prefix was written under its basename with no deduplication, through a
  32-thread pool - so two objects sharing a basename were opened at one path by two
  threads and the file that survived was an interleaving of both. It matters more than a
  lost file sounds: a series that collapses onto one name stops being a series, because
  the pipeline is then handed a single file and the input's provenance digest turns from
  `sha256-tree:` into `sha256:`. The archive extractor had a second form of the same bug,
  disambiguating only on collision so that the name it invented could be overwritten by a
  real member of that name. There is one flattening rule now, and it decides collisions on
  what the FILESYSTEM merges rather than on string equality - APFS folds case and unicode
  normalization, exFAT and FAT32 fold case, so `T1.nii` and `t1.nii` are one file.

- **A `Cache-Control: no-cache` recomputation served the previous preview and statistics.**
  Publishing replaced the labels and metadata but kept any artifact it was not given, and
  nothing gives them - so a client that explicitly asked not to reuse a stored response
  reliably got new labels beside artifacts describing the old ones. Not a race. Each
  publication is now a directory of its own behind one pointer, switched by one rename:
  artifacts it does not supply are simply not in it, a worker still rendering for an
  earlier publication writes only into that one, and no reader can be handed labels from
  one publication beside metadata from another.

- **Nothing a reader has been handed is cleaned up under it.** The cache hands out a path
  that is opened later, when the response streams, so cleanup has to know who is holding
  what. A superseded result now stays while any reader holds it - a lease renewed on every
  read and honored by every cleanup path, the per-key ceiling and eviction included - and
  a reader's acquisition and a reclamation exclude each other on the entry's lock, so a
  path is never briefly missing either. A publication still being written is recognized
  by its writer's claim, never by its age: a writer quiet through one long copy or a
  suspended laptop is not taken for dead - only a lock the kernel released proves that,
  and only on the host that took it. Where nothing can be proved (no locks, another
  host's claim, staging from an older build) nothing is taken. Eviction's `keep` becomes
  a target: an entry in use outlives it, and the next least recently used goes instead.

- **Multi-input provenance paired roles with the wrong digests.** The cache identity is
  sorted so that permuting the source list cannot split a key, while the source entries
  stay in the model's channel order; the two were joined by index. For the shipped MONAI
  BraTS bundle - channels T1c, T1, T2, FLAIR, sorting to FLAIR, T1, T1c, T2 - three of
  four records carried another channel's identity, and for an all-upload submission the
  digest itself. The audit record only: channels were always bound by role, never by
  index. The join uses the role now.

- **A forced refresh did not reach the archive the source had already parsed.** It dropped
  the cached series and the read-ahead image and stopped, while the source kept a ZipFile,
  its central directory and up to 256 MiB of range blocks. Since a cache hit skips
  resolution, a refetch of a replaced `s3:` object or `github:` asset reused the old
  resolved URL and old member offsets - against an object that really had changed, that is
  a CRC failure rather than a refresh.

- **Two `haversack` commands on one input could destroy each other's download.** The input
  cache read a missing completion marker as "the writer is dead", so a second process
  deleted the first's in-progress tree and both could publish contents assembled from two
  fetches. Fetches now go to a directory the caller owns, under the advisory lock the
  model installs already take, and are published by rename.

- **A store upgrade kept the anatomical claim 0.8.0 stopped making.** The upgrader asked
  which groups the engine currently claims, so a legacy `g_lungs` written by the old
  nnU-Net fallback looked user-authored on a MONAI or VoxTell store and was re-emitted
  with `exhaustive=True`, while the provenance step said the unions had been rewritten.

- **`ObjectStoreSource` cached a parsed archive without its credential**, so a cache hit
  skipped the bucket allowlist - the same defect its sibling class documents as fixed.

**Two epochs move, and they answer different questions.** `serve.CACHE_EPOCH` goes to 2
because the same source identifier now computes from a complete series where it used to
compute from a truncated one - that discards stored results once. On its own it was not
enough: it left the truncated INPUT marked complete under its own identifier, so an
upgraded server would miss the result, reuse the bad input, and recompute from it without
the corrected download ever running. `sources.FETCH_EPOCH` versions the fetched-input
contract for that, and is part of a fetched entry's cache key, so inputs downloaded by an
older build are fetched again. Content-addressed uploads are exempt and keep their
identity - nothing fetched them, and there is nowhere to fetch them from.

### Things that were not being checked

- **98 kernel tests had never been collected, once.** Six files arrived misnamed and
  matched neither of pytest's default patterns, and `python_files` was never configured -
  so the grid, mapping, resample, backend and MLX-oracle parity checks were absent from
  every green run. They cost 1.3 seconds. The fast suite goes from 1032 to 1155.
- The engine registry is now authoritative for FastSurfer's checkpoint directory rather
  than being restated by the engine; the two ecosystem-to-engine routing declarations are
  reconciled; `/v1/version` reports a package it cannot see as unknown rather than
  dropping it, which on Modal is the normal case for four of the five engines; and the
  README's engine list is pinned the way `--help` and SERVER.md already were.
- Guards that could not fail were rewritten: the collection check now asks pytest what it
  collects instead of modeling one of its four gates, and the upstream pin check no
  longer lets one unreachable repository void the whole thing.

### Modal

- **Each engine's Modal image and worker now live beside the engine**, in
  `engines/modal_<engine>.py`, and `modal_app` composes them by iterating the engine
  registry. In 0.8.0 the workers were found by looking a class-name STRING up in
  `globals()`, each gated by a module-level flag declared a thousand lines from the class
  it gated - so a typo in either would have dropped that engine from every deploy while its
  environment variable was set to 1, and the error would have told the caller to set the
  variable that was already set. Neither mistake can be written now, and adding an engine
  no longer touches `modal_app` at all. `haversack modal deploy` is unchanged. Verified by
  deploying: all five workers registered, and a real job ran to completion on each of the
  four that moved.

## [0.8.0] - 2026-09-08

A cache that survives its own releases, and two guards for facts that were written twice.

- **A release no longer throws the result cache away.** The key carried `__version__`, so every
  version bump discarded every stored result - three times on 2026-09-08, for releases that
  touched no model, no resampling and no encoding. What a stored result depends on is already
  in the key: the input's identity, the task, the options and the weights versions. The build's
  own component is now `serve.CACHE_EPOCH`, which moves only when the same inputs would compute
  different bytes - the pipeline's resampling, framing or restore, the label mapping, an
  engine's inference path. This change invalidates the cache once more, and then stops.
- **rankfield moves to v0.3.2**, in `pyproject.toml` and CI's hand-written list. No encoder or
  decoder change and no bytes move: it is 0.3.1 plus a CUDA test runner and a corrected claim
  about where a store's tail plane can differ. The tag is what a fresh install resolves, so it
  is worth being current even when nothing in it is load-bearing here.
- **A rankfield API change under `tools/` is now caught by the suite.** Nothing there is
  imported by a test - they are `--no-project` scripts, several Modal- or GPU-only - so the
  0.3.0 `Geometry` migration left `tools/ranked_restore_modal.py` calling a constructor that
  no longer existed, raising on its first field, while every test stayed green. A new test
  reads the scripts and checks that every rankfield name still exists and every call binds
  against the installed signature. It is static: it says nothing about a call whose meaning
  changed while its shape did not.

## [0.7.2] - 2026-09-08

Three things CI was reporting while 0.7.0 and 0.7.1 went out without anyone reading it.

- **Text files are read as UTF-8, not as the machine's locale.** `Path.read_text()` with no
  encoding uses the locale's, which is ASCII under `LANG=C`; the store's own README has
  carried an em-dash for months, so building a ranked store there raised `UnicodeDecodeError`.
  Nothing about the data changed - the environment did, and any user with `LANG=C` would have
  met the same thing. Every read and write of text in the package, the tools and the tests
  names its encoding now, and a test refuses a bare `read_text()`.
- **`typer` is no longer a dependency.** It was declared core and imported nowhere; the command
  line is argparse. It comes out of the install, including the README's lean-install line.
- CI installs `obstore`, which has been a core dependency since 0.6.x and was missing from its
  hand-written list. It only began to matter when 0.7.0 moved `s3:` and `gs:` onto it, at which
  point those sources reported themselves disabled there. A test now fails when the list drifts
  from pyproject again - it found the `typer` entry immediately.

## [0.7.1] - 2026-09-07

- **A disabled engine no longer costs an image build, or blocks a deploy.** The four engine
  images were built on every `haversack modal deploy` however the enable flags were set, so an
  engine this deployment does not run could still stop it: Zenodo's file endpoint answered 504
  for a while on 2026-09-07 and two deploys with FastSurfer switched off died fetching its
  checkpoints. Each image is now built where its worker is defined, inside the flag's own
  branch. `HAVERSACK_FASTSURFER_CHECKPOINTS` remains the way to bake a local copy instead of
  fetching at build.
- **The cache works on a USB drive.** FAT32 and exFAT have no hard links, so the claim falls
  back to an exclusive create; that branch had only ever been exercised by monkeypatching
  `os.link`. Both were mounted and the protocol run on them: one writer of eight racing
  threads, nothing left staged, and commit, discard and eviction intact. Both are also
  case-insensitive, which is what the uppercase escaping in cache keys is for - two keys
  differing only in case stay apart there. Opt in with `HAVERSACK_TEST_NOLINK_ROOT`.
- Retention is one rule again: the Python side read `finished or created` where the SQL reads
  `COALESCE(finished, created)`, which differ at a finish stamp of exactly zero. No stamp is
  the epoch, so nothing could reach it - but the two are meant to be read as one rule.

## [0.7.0] - 2026-09-07

Four data sources and two model catalogs on the extension seams that already existed; one
home for the decisions the local server and the Modal worker both make; and a result that
says whose work it is - which model, under what license, from which data, under what terms.

### Sources

- **Every input's provenance - the bytes, their origin, their license and what to cite - from its own repository, into every result.** A model's
  license is half the story: a segmentation of a CC BY-NC series inherits a constraint, and
  nothing said so. Each source now answers `rights(identifier)` from its repository's own
  metadata, without downloading the data: IDC per series through its v3 API (the license
  belongs to the series in IDC; the answer carries the collection, the dataset's citation with
  its DOI, and IDC's own acknowledgment), TCIA per series from NBIA, Zenodo per record (license,
  creators, DOI), OpenNeuro by its CC0 policy, Hugging Face and GitHub as the uploader declared,
  and an IDC bucket prefix as the IDC series it is. `s3:`/`gs:` objects otherwise, uploads and
  stored content answer "not determined" - a bucket name is not a license label. The lookup
  runs once per fetch on every substrate and is recorded beside the bytes together with the
  bytes' own digest (the content store's, so a fetched series and a stored one hash alike; for
  a DICOM series the UIDs, modality and description its files carry), the IDC data release or
  Zenodo record that versions it, and when it was fetched. Every result's provenance carries
  `inputs`, one record per input in binding order, each `content` / `origin` / `license` /
  `cite` - the same shape as the model's `attribution` - on the local server, on Modal and from
  the command line alike; a local file is pinned by its digest. `haversack rights <input>`
  prints the record.
- **One file is one identity, however it reached the server.** A single volume sent loose, sent
  as a one-member zip, declared `kind=tree`, or fetched from a source used to carry two
  different digests - `sha256-tree:` over a directory of one for the first two paths,
  `sha256:` for the others - so a content-store lookup across them could never match. The
  store now records a one-file directory as the file it is, which is what its own rule
  ("an entry is a function of the content and nothing else") always implied, and `POST
  /v1/inputs` reports the kind it stored rather than the kind the request named. A real
  series is still a tree.
- **`s3:` and the new `gs:` read through obstore, and a trailing slash fetches a whole
  prefix.** The `s3:` source built path-style URLs by hand from a bucket-to-region map and
  read by HTTP Range; it now goes through the same object-store client `idc:` has always
  used, and `gs:` is the same source over Google Cloud Storage with its own allowlist (IDC's
  mirror bucket today). What that buys: `<bucket>/<prefix>/` downloads every object under a
  prefix in parallel, by basename - a DICOM series laid out in a bucket, which is what `idc:`
  does for its own buckets and what no HTTP source could do at all. One object and `!member`
  behave as before (the member read is the same remote-zip extraction over the store's
  ranged reads). Both stay anonymous and allowlisted; the cap is checked against the listing
  before any byte moves.
- **`idc:` can fetch from IDC's Google Cloud mirror.** `HAVERSACK_IDC_CLOUD=gcp` probes
  `gs://idc-open-data` first and falls back to the AWS buckets, because the mirror holds the
  main bucket only (`-two` and `-cr` do not exist on GCS). The identity is unchanged - the
  same uuid names the same bytes on both clouds - so cached results are unaffected. Forwarded
  to Modal containers like the other deploy-time knobs.
- **`s3:<bucket>/<key>[!member]`** reaches a fixed list of public buckets - `fcp-indi` (ABIDE,
  ADHD-200, CoRR, NKI-Rockland), `openneuro.org`, `msd-for-monai` and the IDC buckets. The
  bucket is an operator allowlist, not something the identifier chooses, because a source that
  fetched whatever bucket a caller named would fetch from anywhere. `msd-for-monai` holds tar
  archives, so `!member` does not apply there and its objects come down whole.
- **`github:<owner>/<repo>@<tag>/<asset>[!member]`** reaches release assets. Both are
  `ArchiveReadingSource` subclasses built around one `resolve()`, inheriting whole-file fetch,
  zip-member reading by HTTP range, the decompression cap and the member flattening that makes
  zip-slip impossible. Neither needs obstore, so both work in a lean install.
- **A release tag is readable, not immutable.** An asset can be replaced under a published tag
  and a tag can be moved, so `github:` identities are the `tcia:` kind rather than the `hf:`
  kind, and a result cached under one can outlive the bytes it describes. `s3:` keys are
  mutable the same way.
- **Neither accepts a credential.** Both read public data, a private asset fetched with a
  caller's token would be cached where every reader of that cache can ask for it, and a token
  cannot work in either case: S3 answers one with `400 InvalidArgument`, and github.com
  redirects an authenticated download to a host that rejects it. A supplied token is refused
  with that explanation rather than dropped.
- **`Cache-Control: no-cache` now reaches the input.** It skipped the result cache but not the
  fetched-series cache, so a forced recompute re-segmented stale bytes - and the read-ahead
  image was not dropped either, so the re-download was paid for and then discarded. Both are
  invalidated now, on the local dispatcher and on the Modal worker. A discard refuses while
  another job is reading those bytes, and that refusal is reported on the job as
  `input_refresh_skipped` from either deployment rather than passed off as a refresh. This
  only became necessary because these are the first two sources whose identity is not
  version-pinned.
- **A refusal that needs no network happens at submit.** An unlisted bucket, or a credential a
  source does not take, was refused inside the worker: the server answered 202 and then failed,
  or 502, which reads as an upstream outage rather than a bad request - and on Modal a GPU
  container had already started. `DataSource.check` is the seam and both doors call it.
- **`parse_input` knows the built-in prefixes**, computed once at import from the sources
  themselves. **This fixes a pre-existing bug:** `hf` is two characters and the old shape test
  needed three, so *every* `hf:` input was treated as a local filename by the command line and
  by `haversack get`. They work now.

### Catalogs

- **Every task credits its makers.** Who made the model, under what license, and what its
  authors ask to be cited - each reference with its DOI and PubMed ID - from three layers at
  once: the task's own facts from its catalog manifest (a MONAI bundle's authors, copyright and
  references; DentalSegmentator's CC BY 4.0; TotalVibeSegmentator's per-model release), its
  ecosystem (the group, the repository, the papers, the license, TotalSegmentator's list of
  license-gated models and the MRI paper for its MR tasks) and the engine that runs it
  (nnU-Net, which every nnU-Net catalog's authors ask to be cited). `haversack cite <task>`
  prints it; `describe()` and `GET /v1/tasks/<task>` carry it as `attribution`; every result's
  provenance carries the license and the identifiers. Two things this fixes on the way: the
  manifest's license and description used to reach a client only while a task was NOT
  installed - the installed path of `describe()` rebuilt its answer from the checkpoint and
  dropped them, so the credit vanished at the moment the model became usable - and MOOSE,
  MRSegmentator, TotalSegmentator and the engines recorded no license or citation at all.
  Every entry in `data/attribution.json` was read from the project's own README, LICENSE or
  documentation and every PMID resolved through PubMed from the DOI the authors publish; the
  MONAI manifest now records each bundle's authors, copyright, description and references from
  its own `metadata.json`.
  Two facts worth knowing that this surfaced: VoxTell's weights are CC BY-NC-SA 4.0 (non-commercial;
  the repository's Apache-2.0 covers the code only), and `moose:clin_ct_dental` is
  DentalSegmentator's checkpoint redistributed, so its credit and its CC BY 4.0 are Dot et al.'s.
- **DentalSegmentator** (`dentalsegmentator:base`, weights CC BY 4.0): dento-maxillo-facial
  CBCT and CT - upper skull, mandible, upper and lower teeth, mandibular canal. Its plans
  permute the axes, so it needs the transpose opt-in below. The reference Slicer extension also
  removes connected components under 60 mm3, which haversack does not, so its output carries
  speckle that tool would have cleaned.
- **TotalVibeSegmentator** (`totalvibe:*`, Apache-2.0): whole-torso VIBE MRI (`vibe`, 72
  structures, and `vibe_sagittal`), CT bone models (`ct_bones`, `feet_bones`), `body_regions`,
  `vertebrae` and `pancreas`. The release publishes nineteen models and the manifest accounts
  for every one: seven offered, twelve excluded with a reason each - multi-channel (three,
  which the nnU-Net path does not carry), published without metadata (two), or single-channel
  but never exercised here (seven). The generator checks each reason against the asset and
  refuses to run if the release grows a model in neither list. `feet_bones` keeps nnU-Net's
  label integers where upstream renumbers them into 99-117; the names are unchanged and
  correctly paired, but a numerical diff against an upstream volume will not line up.
- **MOOSE offers 24 of upstream's 25 models and records the one it does not.** The
  generator's name pattern was lowercase-only, so `clin_ct_ALPACA`, `clin_ct_PUMA`,
  `clin_ct_PUMA4` and `clin_mr_FVM` fell through it without a word: 21 models shipped and
  nothing said the registry held more. The three CT models are ordinary single-channel
  checkpoints under their own `Dataset*` folder and are offered now: `clin_ct_ALPACA` (aorta,
  pulmonary artery, both iliac arteries and both ventricles), `clin_ct_PUMA` (23 whole-body
  organs and tissues) and `clin_ct_PUMA4` (22, with the bowel and the fat compartments split
  out). `clin_mr_FVM` is excluded: its zip unpacks to `clin_mr_FVM/` with `Dataset501_FVM` a
  level down, so haversack's installer would refuse it and moosez's own extractor would not
  find a model where it looks. The manifest records that under `excluded`, the generator reads
  the reason off the archive rather than taking it on trust, and it now refuses to run when any
  `KEY_URL` block in the registry does not parse as an entry - so the next layout change is an
  error, not a missing model.
- **`base` is now an ambiguous short name**, since MRSegmentator already had one. That is the
  documented behavior for a collision, but it will break a script that says `task="base"`.
- **One install for three packagings, `ZipManifestEcosystem`.** MOOSE, DentalSegmentator and
  TotalVibeSegmentator all publish bare nnU-Net checkpoints as zips and all read their labels
  from the installed checkpoint, so the manifest loading, the version-pin rule, the install and
  the sidecar live once. A catalog declares data and overrides one method for a real difference
  in packaging: TotalVibeSegmentator's archives have no `Dataset<id>` parent, so that catalog
  says where the zip unpacks and nothing else. MRSegmentator deliberately stays off the base -
  its zips are flat, so the unit that must land atomically is different. (The pin rule compares
  the tag the install sidecar recorded, not a hash of the bytes; what it rules out is trusting
  the *manifest's* tag for a folder already on disk.)
- **The orientation a model needs is read, not hardcoded.** TotalVibeSegmentator reorients per
  model - RAS for the whole-body and CT models, LPS for the body-region, vertebra and pancreas
  ones - while the plans still declare a reader that does not reorient. That is the
  MRSegmentator LPS trap, except upstream states the answer in each checkpoint's own
  `dataset.json`. Anything that is not three letters naming each anatomical axis once is
  ignored rather than passed to `DICOMOrient`, and the generator refuses a model that declares
  none, because a missing one degrades silently into a left/right mirror.
- **`checkpoint_best.pth` is a fallback.** A published model may ship either; `body_regions`
  and `vertebrae` ship only `best`. `final` is still preferred, so every model that already ran
  is byte-identical, and one name is used across every requested fold.
- **Digests are whichever the publisher states** - Zenodo publishes md5, GitHub sha256, and
  the install sidecar records the one it verified under that name. Three
  TotalVibeSegmentator assets are published with no digest at all and are checked against
  nothing; the manifest records which and why. So are all 24 MOOSE assets, which predates this
  change and is not something it fixes.
- **`moose:clin_ct_dental` installs.** Its asset is the one MOOSE entry not hosted as a
  GitHub release, and the Cloudflare rule in front of `model.s.mdforge.com` answers 403 to
  Python's default `Python-urllib/3.x` User-Agent while answering anything named with 200. The
  zip installer sent the default, so a live asset failed mid-install and read as dead. Every
  weights download - the zip catalogs and TotalSegmentator's - now identifies itself as
  `haversack/<version>`, and `tools/zippeek.py` sends the same name so the generators see what
  the installer sees. The MOOSE generator, the one generator that never reads the assets it offers,
  now HEADs every URL before writing the manifest and refuses to record one that does not
  answer. (The same model is also `dentalsegmentator:base`, installed from its original
  Zenodo record and md5-verified; the MOOSE copy is checked against nothing, like the other
  23.)
- **Every HTTP request haversack makes goes through one door.** There were three: a
  hand-built opener in the sources that dropped `Authorization` on a cross-host redirect, the
  weights installers' bare `urlopen` calls that had just learned to name themselves, and an
  engine checkpoint fetch that did neither - so the User-Agent fix above reached the weights
  path and left every hosted *source* still sending `Python-urllib/3.x`, which is the path a
  server takes on a client's behalf. `haversack.fetchlib` (stdlib, no new dependency) now
  holds both properties, every request in the package passes through it, and a test fails if
  a `urlopen` grows anywhere else. The manifest generators, which cannot import the package,
  route through `tools/zippeek.py` the same way.
- **Three MOOSE models that were never offered now are: `moose:clin_ct_ALPACA`,
  `moose:clin_ct_PUMA` and `moose:clin_ct_PUMA4`.** The generator's name pattern was
  lowercase-only, so these and `clin_mr_FVM` fell through it without a word - 21 of 25 models
  offered and nothing said so. The three are ordinary single-channel CT checkpoints (7, 24 and
  23 structures) and are offered as such. `clin_mr_FVM` is not: its zip unpacks to
  `clin_mr_FVM/` with `Dataset501_FVM` a level down, which haversack's installer refuses and
  moosez's own extractor cannot find either. The manifest records that reason, and the
  generator checks it against the archive on every run and refuses to run when a registry
  entry fails to parse, so the next silent omission is an error instead.
- **An archive that would replace another task's weights is refused.** Unpacking replaces a
  directory of the archive's own top-level name, so with a stale manifest one task's download
  could overwrite another's model and leave it silently running the wrong network. The names
  are checked before anything is moved. This was true before these catalogs existed.
- **Installs of one model folder are serialized**, across threads and processes, by an advisory
  lock. Two callers wanting one task at once is ordinary - two prepares, a prepare racing an
  on-demand install, two workers on a shared volume - and unlocked they interleave a
  destroy-then-move; the second caller now finds the work done and fetches nothing.
- "Installed" means a model folder nnU-Net could load, not any `dataset.json` found underneath,
  so an interrupted unpack is never mistaken for a finished one. It is deliberately not the
  stricter `resolve_model_folder`: a folder shipping several configurations and no preferred
  one is a real install and must not be re-downloaded.
- macOS zip litter (`__MACOSX/`, `.DS_Store`, `._*`) is dropped on unpack - the DentalSegmentator
  asset is zipped that way - and interrupted installs no longer strand their part-downloaded
  archive, which for these models is 100 MB to 1 GB at a time.

### Command line and server

- **`--allow-transpose`** on `haversack segment` and `haversack serve`, and
  `HAVERSACK_ALLOW_TRANSPOSE` for the Modal worker. Models whose plans permute the axes are
  refused by default, and the refusal named a Python keyword argument no command line or HTTP
  client could pass - so `dentalsegmentator:base`, `totalvibe:vibe_sagittal` and
  `totalvibe:pancreas` were unreachable from every door but the API. It is deployment policy,
  so it is an operator flag rather than a request parameter.
- **The weights commands see every catalog.** `weights fetch` went through TotalSegmentator's
  manifest alone and rejected every moose, mrsegmentator, dentalsegmentator and totalvibe task,
  while `tasks` told users to run exactly that. `weights list` and `weights remove` now scan the
  per-ecosystem directories too, so a catalog's models can be listed and deleted rather than
  needing `rm -rf`, and all three accept `--model-root` as well as `--root`. (`weights
  coverage` is still TotalSegmentator-only.)
- **`tasks <name>` prints label order with the label**, and `--json` carries the mapping. It
  printed names alphabetically, which for a numerically-labeled model gives 1, 10, 11, 2.
  `describe()` carries `label_map` for the same reason: a caller reading a result cannot
  assume the labels are 1..N - `feet_bones` uses 1-17 and 99-117 - or that the names sort
  meaningfully, since several checkpoints name their structures with numbers. **This changes
  `haversack tasks --json` for existing tasks too**: `structures` is in label order rather
  than alphabetical for 51 of the 75 tasks a default catalog lists, so a script that relied
  on the old ordering will see different output.
- `tools/zippeek.py`: the range-based remote zip reader the manifest generators share, so
  describing a 1 GB asset costs a few kilobytes and the zip64 parsing exists once. New
  generators `tools/gen_dentalsegmentator_manifest.py` and `tools/gen_totalvibe_manifest.py`.

### Cache and job protocol

- **A staging claim is now taken and named in one operation.** It used to be two - create the
  directory, then write `.owner` - which left a window where a claim existed that nothing
  could identify, and `_owner_of` returning nothing could not distinguish a writer that
  crashed before naming itself from a successor two statements in. Every guard around the
  reclaim path existed to work around that: an mtime-based freshness heuristic, an owner
  re-read immediately before the reclaim, an extension counter. The claim is now published by
  a single `os.link` of a file that already holds the token, so a claim that exists always
  names its owner. `os.link` rather than `os.symlink`, which has the same property but needs
  Developer Mode or a privilege on Windows; a filesystem without hard links falls back to an
  exclusive create, where an empty claim reads as live and is never reclaimed.
- **A cached input can no longer be deleted while it is being read.** The read-ahead filled
  from a committed entry during the window where nothing holds a pin, so a concurrent
  `no-cache` discard or an eviction under budget pressure could remove a DICOM series
  directory while the reader was still walking it - a partial read of a series that no longer
  exists, rather than a clean failure. The pre-read now holds a pin for its duration.
- **`Cache-Control: no-cache` refreshes when a pre-read is in flight.** The pin above is
  refused by `discard`, which cannot say who holds it, and the prefetcher did not consult the
  job's own refresh flag - so a forced recompute could return the cached answer instead. It
  no longer pre-reads an input the queued job asked to refresh, and such a job never uses a
  pre-read image even if one landed. This matters most for `s3:` and `github:`, whose bytes
  can be replaced under one identifier. Both rules were first fixed on the local server only;
  the Modal worker kept selecting those jobs, and now asks the same shared rule
  (`jobpolicy.prefetchable`, `take_pre_read`) - which also stops it pre-reading a fraction of
  a multi-input job's inputs, another exclusion the two sides had disagreed on.
- **A claim that cannot be read is no longer treated as absent.** It reported as unclaimed,
  and the waiter retook it without pausing - measured at 95% of a core on the single
  dispatcher thread, indefinitely. The trigger is ordinary: under `umask 077` the claim file
  is owner-only, so a second user on a shared cache root reached it.
- **Stale staging is cleared when a claim takes over an entry**, and a long-lived server
  re-sweeps its graveyard rather than only at startup. Content sitting under a claim with no
  completion marker is an attempt that never finished, and adopting it produced a series
  assembled from two different fetches, marked complete.
- **A claim that fails after it is published is taken back.** Clearing stale content under a
  fresh claim could raise (an entry that is writable but not readable, found by probing), and
  the handler then returned "not claimed" with the token still linked: a claim nobody held,
  reported as staging, and the key unusable until the timeout reclaimed it. The heartbeat also
  named the claim file by its literal spelling rather than the constant.
- **The per-model install lock now holds on Windows.** It was a bare `flock`, and where that
  could not be imported the install ran with no lock at all - so two installs of one model
  there could still delete each other's weights, the bug the lock exists to stop. A small
  `haversack.filelock` module locks through `fcntl` or `msvcrt`, and the ranked store's
  one-writer lock uses it too. Two other Windows-hostile spots fixed on the way: a download
  temp file was unlinked while still open, and a checkpoint move used `rename` where the
  target may exist.
- **Modal: jobs orphaned by a stopped deployment are failed, not left queued forever.**
  `modal app stop` cancels the spawned calls but leaves their records `queued`, and queued
  records are never purged by age (deliberately). Five of them, 76 hours old, were found on a
  live deployment and had done three kinds of damage: probes of their keys reported a flight
  that would never land; the prefetcher, which warms the oldest queued job, warmed them on
  every job and so never staged a real one; and they were listed as queued indefinitely. The
  worker now asks Modal whether each active record's call still exists - at container start
  and after every job, skipping records younger than two minutes so it can never race a
  job's own completion - and fails the dead ones with a message that says to resubmit. The
  local server has reconciled its store at startup all along; this is the same rule on the
  other substrate. An `inflight:` marker is now also dropped as soon as its job is terminal,
  rather than only once it is strictly older than zero seconds.
- **Modal: each engine worker now warms only the jobs that will run on it.** The prefetcher
  scanned the shared jobs dict for the oldest queued job with no regard for which worker
  would run it, and every worker class is its own container with its own series cache and
  read-ahead. With five engines deployed, the SynthStrip container pre-read the nnU-Net
  worker's upload into a read-ahead nothing there would ever pop, the nnU-Net worker spent
  its one-ahead slot staging a FastSurfer job's series and ran its own next job cold, and
  every container downloaded the same MRI once. The scan is now filtered by the worker's
  engine.
- **Retention and terminal job states agree across deployments by construction.** The local
  server decides in SQL and the Modal deployment in Python, so they cannot share code; a test
  now drives both over the same cases instead of asserting in a comment that they match.
- Internal: `haversack.jobpolicy` holds the decisions both deployments have to make the same
  way - the no-cache rule, the pinned pre-read, the terminal states, retention, and how a job
  source becomes a cache key. Six places built that key by hand and two of the spellings
  disagreed when a source had no identifier. No public API change.

## [0.6.1] - 2026-09-05

Fixes from five adversarial reviews of 0.6.0 (the library, the store path, an outsider's
install, the default path against 0.5.0, the viewer), with rankfield moved to 0.1.2.

- **A lean install (README's own recipe, no torch) died on `haversack segment`**: the
  store-output name check imported a module that imported the pipeline. The check now lives in
  `haversack.io` with nothing behind it, and a test runs the CLI with torch, nnunetv2, scipy,
  skimage and the store extra all blocked.
- **The ranked API says what it needs.** `haversack.ranked` (`RankedSpec`, `margin`, `encode`,
  ...) is a shim over rankfield, which the `duckn` extra installs; in 0.6.0 a plain install
  got a raw ModuleNotFoundError. The import, `haversack restore` and `segment -o x.duckn.zip`
  now answer in one line naming the extra, before anything is computed.
- **Python 3.12 is the floor**, one floor for everything: the `duckn` extra's per-package
  markers made it empty on 3.10 and rankfield-only on 3.11, silently.
- **Store geometry follows the crop.** The array's true spacing and origin were derived from
  the full canonical grid, not from the crop-to-nonzero sub-grid the resample ran on (every
  nnU-Net-native lineage): a 60^3 crop of a 100^3 source placed the store 19 mm off with a
  67 % spacing error. One derivation now serves the builder and the emit; the restore was never
  affected, since it follows the frame. No shipped store is affected: TotalSegmentator and
  FastSurfer never crop.
- **rankfield 0.1.2**: `margin()` read a winner's sentinel byte as level 0 - the log curve's far
  end, 64 logits - instead of the clip, on 15 % of a real field's voxels; the field
  measurements in statistics consumed it. Encoding is now reproducible across devices (`topk`
  selected arbitrarily among tied keys; the tail summed in device order). The Metal and Triton
  kernels check the label table's length. The 0.6.0 note that the log byte made area free was
  that bug: under a correct margin the area a store gives up is again 0.3-1.1 %, always low,
  and the test says so.
- `haversack restore`: an output grid past 2^31 voxels, or one that does not fit in memory
  (torch raises RuntimeError, not MemoryError), is an InputError naming the grid; an unknown
  `--device` too.
- `segment -o x.duckn.zip`: the target is checked for writability before the network runs;
  `--spacing` is refused (the store is on the model grid); the report names the store, not the
  discarded label grid.
- Provenance records what was built (`parts_kept`, the derived layers present,
  `distance_voxels`) instead of constants, and the product path writes `provenance.sources`
  naming the input as given (`idc:...`, or the file); the verifier refuses a 0.3 store whose
  embedded README documents the 0.2 byte.
- Statistics asked for field measurements say in the JSON whether they came
  (`field_measurements.available`, with the reason), instead of dropping the columns silently.
- `haversack.RemoteClient` is exported as the README says; the README says which extras pip
  can install (the git-sourced engine and store extras need uv).
- The slice preview (sdfview) refuses a part without a clip and null curve keys; its 3-D
  renderer decodes the rival gap through the level table (it read log bytes with the uniform
  formula) and refuses parts on different grids, clips or curves.

## [0.6.0] - 2026-09-05

**The ranked encoding is its own library.** encode / decode / the store-backed restore on
torch, Metal and Triton / the float64 reference / `Frame`, `Grid`, `Mapping` and the level
tables moved out to **rankfield** (github.com/mhalle/rankfield, pinned at `v0.1.0`), which
has no serialization of its own. `haversack/ranked.py` is a shim over it - every name it used
to define is re-exported, so no import changes - and keeps what is haversack's: `RankedSpec`,
`emit`, and the distance and junction caches. rankfield joins the `duckn` extra (duckn + zarr
+ rankfield: the undocumented ranked store) as a git source at the tag; it is not core, and a
test now proves the default path - `haversack segment IN -o labels.nii.gz` - imports none of
the three.

- **Ranked format 0.3**, written by every new emit. Two changes to the bytes:
  - `keep: "shell"` - a class that wins at the voxel or at any of its 26 neighbors is stored
    with its TRUE gap, however far behind, then the non-winners within the clip. The clip is
    a relevance threshold and the floor for what is unnamed, no longer a cap on what is
    stored. Flooring a dropped class at `-clip` is an upper bound, so the interpolated winner
    was over-credited wherever a shell neighbor had been dropped, and thin structures grew.
  - `gap_curve: "log"`, `gap_range` 64, `gap_origin` 0.5 - the support byte follows a log
    curve over 0..64 logits: 0.004 logits per step at a tie, 0.12 at the clip, against 0.03
    everywhere for the uniform byte. Every decoder indexes the same 256-entry float32 table,
    `rankfield.levels(meta)`, so numpy, torch, the Metal and Triton kernels and the slice
    preview agree bit for bit; a 0.2 part gets the uniform table from the same call.

  Measured on idc-torso1 (`total_fast`, 3 mm restored to 1.5 mm, against the labels the run
  wrote from its own logits): **0.0037 % of voxels moved and the worst structure, the left
  12th rib, is +0.4 %**, against 0.0596 % / +18.8 % for the uniform byte at clip 8. 2.08 MB
  against 1.96 MB uniform-8 and 4.01 MB uniform-16 (which is 0.0056 % / +2.0 %). Depth was
  never the cause - depth 16 restores bit-identically to depth 6 - and a shell never exceeded
  6 classes, so depth 6 stays the default. Companding alone, without the shell, bought
  nothing. Reproduced through the product path: `segment -o torso.duckn.zip`, then `restore
  --spacing 1.5`.

  A 0.3 store cannot be made from 0.2 bytes - the shell's true gaps were never written - so
  `tools/ranked_upgrade_format.py` stops at 0.2 (on its own constant; it refuses a 0.3 store)
  and the demo stores stay 0.2, readable everywhere. The verifier reads both and requires
  `keep`, `gap_curve`, `gap_range` and `gap_origin` on a 0.3 block.

- **`haversack restore STORE -o LABELS`** (internal, undocumented, like the store and `view`).
  Labels from a ranked store on any grid - the input's own, or `--spacing S` - written in the
  input's orientation, `--interp linear|nearest`. `haversack.ranked_restore` is the store
  adapter: it reads each part into a `rankfield.Part` and hands the list to `rankfield.restore`
  in paint order. The builder now copies the run's `frame` record into the part block, so
  output-to-model coordinates are composed exactly as `predict_into` did, crop included; a
  part without one (every store built before this release) restores only onto grids in its
  own model-grid frame, and the verifier warns about it. Work is bounded by the region asked
  for. On that torso, 52.5 M voxels: 10 s linear and 1.6 s nearest on CPU, 0.36 s on Metal
  (M2), 0.15 s on Triton (A10, proved bit-identical on Modal); one structure's box in 0.02 s.

- The slice preview (`data/preview.html`, rebuilt from sdfview) reads the level table, so
  `haversack view` shows 0.3 stores.

- Fixed from the release review: the dense `junction_field` wrapper accepted `levels` and
  dropped it (the numpy reference then read log bytes as uniform ones, up to 81 quanta off
  torch; now pinned by a test); two renderer tools decoded 0.3 bytes with the uniform
  formula; `restore --spacing 0` meant the input grid; an output grid that does not fit in
  memory is an InputError, not a traceback. `__version__` had said 0.4.0 since 0.4.1 - it is
  stamped into every store and reported by the server - and is bumped with the release from
  now on. CI installs the ranked-store extra from its tags, so those suites run there.

- The `cuda` extra floors triton at 3.0: rankfield's restore kernel passes
  `enable_fp_fusion` as a launch option.

- The area a store gives up to quantization was 0.3-1.1 %, always low, under the uniform
  byte; under the log byte it is below 0.1 % with no sign left to it.

## [0.5.0] - 2026-09-05

**Ranked format 0.2** (internal, undocumented; every reader refuses older parts, and
`tools/ranked_upgrade_format.py` brings a store up in place). From a critique of the encoding:

- A rank whose support byte is 0 - a gap that rounds to the clip - is written as the
  sentinel, since it decoded as absent anyway; the verifier's deep check enforces it.
- Each part's block states `version`, `gap_unit` (`logit` for a softmax head) and `tail_max`.
- The tail is uint16 and is the mass of everything not stored, the classes past the depth
  and the masked ones alike.
- `decode_groups` has a byte-valued fast path for disjoint groups, bit-identical to the
  general one.
- A round-trip test: the store's argmax, composited in paint order and aligned through both
  geometries, equals the labels the run wrote. The truncation bias of a two-plane consumer is
  measured and stated in the module header (depth 2: 2.1 % of boundary cells against depth 6).

## [0.4.4] - 2026-09-04

- **Weights installs report progress.** A `prepare` job used to show one static `weights`
  stage until it finished, so a client could only run a timer against a 230 MB fetch. The
  job's `progress` object now carries what it carries for a segmentation: stage `weights`,
  one part per model (`part` / `n_parts`, the whole cascade chain counted up front), bytes
  received as `step` / `n_steps` (Content-Length, else the manifest's `size`), and a
  `fraction` that moves with the download. `detail` walks "downloading ...", "unpacking
  ...", "installed" (or "present" for a model already on disk). Byte snapshots are throttled
  to every 2 % (at least 4 MiB). Cancelling a prepare job now interrupts the download.
  TotalSegmentator, MOOSE, MRSegmentator and FastSurfer installs all report this way, on
  `haversack serve` and on Modal. The CLI's output is unchanged.
- `Reporter.advance(within, step=, n_steps=, detail=)` reports a position inside the current
  part directly, for work whose stages are not the pipeline's. `InstallProgress` is the
  adapter installers write through; it accepts a `Reporter`, a message callback, or nothing.

## [0.4.3] - 2026-09-04

- **`WeightsStore.have()` now recognizes zero-padded dataset folders.** TotalSegmentator
  ships weights id 8 as `Dataset008_HepaticVessel`; `have()` looked for `Dataset8_*` only, so
  `ts:liver_vessels` never showed as installed - `weights_installed` from the server stayed
  false after a successful fetch, a client polling it (the Slicer panel's fetch loop) spun
  until it timed out, and its segment pre-check refused a model that was on disk. Every
  other lookup already tolerated both spellings; this was the one that did not. Reported
  from SlicerGotthard.

## [0.4.2] - 2026-09-03

- **`haversack view STORE...` (internal, undocumented, like the ranked store it shows).**
  Serves a slice preview of one or more ranked stores on this machine and opens it: three
  orthogonal panels in radiological display, derived from the direction cosines. Each pixel
  is the class whose interpolated margin is largest there - the argmax after interpolation
  that the restore performs to write labels on the image grid, done in the slice plane - so
  boundaries fall where they fall in the labels and the field's edge artifacts show at that
  resolution; nothing is outlined. Structures toggle per part; the wheel scrolls slices on a
  clicked panel. The page is sdfview's single-file preview, shipped as `data/preview.html`;
  the server hands out directory stores file by file and zips by Range request, and lists
  the stores at `/stores`. Reviewed adversarially before release (paths, Range semantics,
  hosts, lifecycle).

## [0.4.1] - 2026-09-03

Two adversarial review rounds over 0.4.0 (four reviewers each), every agreed finding fixed.

- **Server.** `--no-token` is open, full stop: no origin or proxy heuristics; a machine you
  trust end to end. Every other server has a token, generated after the socket is bound and
  written with the server's pid to `~/.cache/haversack/serve/<port>.token`; the client uses
  that file only for a loopback address whose port has a live server behind it, and a server
  refuses to start when a live one already owns the port's file. `Cache-Control: no-cache`
  reaches the executor on the path surface and the artifacts. Cache entry names are injective
  under case folding. After a restart the single-flight marker is restored, terminal records
  keep answering from the job store (marked `evicted`, with their options and no result link
  once the bytes are gone), DELETE drops the record, and a cancel that lands during the save
  is still honored. A submit whose key is in flight joins that job; joiners must share the
  flight's source credentials, and a joiner's DELETE releases its seat rather than cancelling
  for the others. Failures are recorded with their cause. Describe never echoes the resolver.
  The 401 names where the token it sent came from; an unreachable server is one line.
- **`HAVERSACK_CACHE_DIR` is the cache root everywhere:** results, FastSurfer checkpoints,
  the input cache and the trainer shims all sit under it, `cache list` shows them all, and a
  root that is not a directory is refused up front.
- **Command line.** Output names are validated before the inference stack is demanded and
  before any input is downloaded; a name haversack cannot write (a directory, a bare name,
  `.zarr`) is refused at once. Unknown tasks, unreadable or absent inputs and half-built
  model folders are one-line errors, not tracebacks.
- **Ranked stores (internal, undocumented).** Writers build in a staging directory beside the
  target, hold a lock, and replace the target only after a complete build; anything at the
  path that is not a store is never replaced; symlinks are written through. Leaves are unique
  per (layer, value) - a cascade built with every stage used to lose the final stage's
  classes to a value-only dedupe - single-part stores omit `layer`, and the lungs claim is
  made only over the exact lobe set. duckn pinned at 0.3.2.

## [0.4.0] - 2026-09-03

- **Internal, undocumented, unstable: ranked stores and duckn.** The ranked-store tooling
  (emit and build) moved into the package, stores can live in a standard zarr zip as well as
  a directory, and every store attribute is now built and read through duckn's own metadata
  models (seg extension 0.7: leaves and groups, coverage and partition claims, background as
  a segment role). duckn volumes are accepted as input, and duckn is declared as the `duckn`
  extra, installed from its GitHub release tag. None of this is in the README or the command
  help on purpose: the format and duckn are being iterated together and nothing outside
  should depend on them yet. The demo stores were upgraded in place and now live under
  `data/duckn_demo`.

- **`get` and `segment` take several inputs (batch mode).** With more than one input the
  output is a directory (`-o`, default the current directory) and `--format` names the type:
  `segment` writes `<dir>/<name>_<task>.<ext>` for each, `get` writes `<dir>/<name>.<ext>`
  (converted) or copies each raw when `--format` is omitted. A single input is unchanged
  (`-o` the file, extension picks the format). One input failing does not sink the batch -
  it is reported and the run exits non-zero.

- **Modal workers fail fast when a volume was deleted under a snapshot.** A memory snapshot
  (on by default) captures the container with its volume handles; deleting one of the per-app
  `scratch`/`cache`/`inputs` volumes and redeploying left a restored worker holding a dead
  handle, and the first write failed deep in a job with a cryptic `volume vo-... not
  attached`. A one-line write probe at worker startup (`_check_volumes_attached`, post-restore)
  now surfaces it immediately and names the remedy: redeploy once with `HAVERSACK_SNAPSHOT=0`,
  or stop and redeploy the whole app. Root cause of the reference-deploy failures seen while
  verifying FastSurfer on torch 2.14 - self-inflicted by an earlier volume cleanup, not Modal.

- **FastSurfer shares the main environment.** The `torch==2.7.*` pin was upstream FastSurfer's
  CUDA-11 caution, not a need of the parcellation net; the `fastsurfer-lean` fork relaxes it to
  `>=2.7` (torchvision unbounded above), the `fastsurfer` extra leaves the uv conflict group,
  and `uv sync --extra fastsurfer` (or `haversack[fastsurfer]`) installs it beside torch 2.14
  and the nnU-Net path - no separate venv. Verified: the same T1 through the engine on torch
  2.7.1 and on 2.14 gave **bit-identical** labels (0 of 654,295 voxels differ, all 95
  structures, every per-label Dice 1.0), so the relaxation changes nothing about the output.
  SynthStrip still owns its environment (numpy<2).

- **FastSurfer checkpoints come from Zenodo, verified.** FastSurfer's own downloader tries
  `b2share.fz-juelich.de` first, whose server omits an intermediate certificate; because its
  helper catches only HTTPError, the resulting SSLError killed model load before the Zenodo
  fallback was tried (and a Modal build could not bake the weights). haversack now fetches the
  three DOI-versioned checkpoints from Zenodo itself (stdlib urllib, sha256-checked) into
  `~/.cache/haversack/fastsurfer-checkpoints` (`HAVERSACK_FASTSURFER_CHECKPOINTS` overrides),
  once per machine. The Modal image bakes them the same way at build (to `/opt`, off the
  mounted `/weights` volume that a shipped copy would have blocked).

- **`haversack get` fetches source data; `cache` and `weights list`/`remove` manage the disk.**
  `get idc:<uuid>` (or any `zenodo:`/`http` source) lands the data in the input cache and
  prints the path, so a later `segment` of the same id reuses it; with `-o` it also writes
  there - a directory takes the raw fetched files, an image-extension file (or `--format`)
  is converted to one volume (a DICOM series -> a single NIfTI/NRRD, geometry preserved),
  and `--no-cache` skips leaving a cached copy. `cache list`/`clean`/`path` show and sweep
  the transient stores (inputs, results, engine checkpoints); `weights list`/`remove` handle
  models. `cache clean` needs `--yes` (a dry run otherwise), takes `--older-than 30d`, and
  never touches weights - those are removed one dataset at a time by `weights remove`.

- **Users refresh their own weights manifest.** `haversack weights refresh` from an installed
  package writes `~/.config/haversack/ts_weights.json` (`HAVERSACK_TS_MANIFEST` moves it),
  laid over the packaged manifest on every read - a user's entry for a dataset replaces the
  packaged one - and kept across upgrades; before, a refresh edited the copy in
  site-packages, lost on reinstall. A source checkout still edits the repository's file, and
  `--to PATH` names any file. `weights coverage` says where its entries come from.

- **After a blind usability test.** An agent given only `--help` and `haversack docs`
  completed listing, a CLI segmentation, a Python-API liver volume and a server round trip
  first try; what it stumbled on is fixed: `haversack tasks <name>` prints a task's
  structures (they were only inside `--json`, unmentioned), `tasks --help` defines
  `materialized` and `task_spec`, `remote submit` ends with `done 100%` and `wrote <path>`,
  and the guide annotates the API's return values, names `total_fastest`, matches the CLI's
  timing keys, and says how to stop the server and when it is ready.

- **The command line explains itself.** Every option has help, every command a description
  and examples, defaults are shown, and the top line says what haversack is rather than how
  it works. New `haversack docs [topic] [--sections]` prints the user guide - the README,
  shipped inside the wheel as `haversack/data/GUIDE.md` (an editable checkout reads the
  repository's README; one source of truth) - whole or one section. A test walks the parser
  and fails on any future option without help.

- **obstore is core.** `haversack segment idc:<crdc_series_uuid>` works from a bare install;
  the `idc` extra is gone, and the Modal image definitions no longer name it. The lean recipe gains
  `obstore` so a client-only install can still fetch from IDC.

- **A run that departs from what was asked says so.** `provenance["deviations"]` is a list
  of `{what, requested, effective, why}` records (`result.deviation()`), and it travels
  everywhere provenance does: the seg.nrrd header, the server's result payload, and the
  CLI's closing `note:` lines; the engines also report each one as a progress stage.
  Recorded today: SynthStrip's fp16 rerun after a real MPS out-of-memory (`precision`, and
  `precision` itself is now in provenance); FastSurfer's view-aggregation field moving to
  the CPU on a small unified-memory machine (`device (view-aggregation field)`); and the
  nnU-Net accumulator's effective placement per model (`accumulate` on each `models[]`
  entry, a deviation when `device` was asked and `host` ran). The rule: a run may adapt
  to the machine, never silently.

- **Apple Silicon: the MPS allocator is capped, because past the cap Metal fails silently.**
  SynthStrip's first local run masked 94 % of the image. The cause, isolated with identity
  convolutions in pure torch: PyTorch's MPS allocator lets a process grow to 1.7x the
  device's recommended working set (`recommendedMaxWorkingSetSize`, 10.7 GiB on a 16 GB
  machine) before raising, and past 1.0x Metal cannot back the buffers - a conv3d returns
  all zeros, no exception (exact at 8 GiB of live activations, zeros at 11 GiB). The same
  lock on Modal's CUDA gave the sane 1282 mL mask, so no dependency was at fault, and torch
  2.7 cannot run the net on MPS at all (no max_pool3d), so nothing regressed. Fix:
  resolving an MPS device now calls `torch.mps.set_per_process_memory_fraction(1.0)` once
  (`HAVERSACK_MPS_MEMORY_FRACTION` overrides, `0` keeps PyTorch's default); under the cap
  the allocator reclaims its cache and the same forward runs correctly in fp32 - SynthStrip
  256^3 peaks at 6 GiB, Dice 0.999971 against the Modal CUDA fp32 mask (57 voxels) - and a
  real shortfall raises. SynthStrip catches that: retries with the net in fp16 (0.015 mm
  worst case on the SDT), then refuses naming `device='cpu'`; it also refuses any constant
  field outright. This is a known class of PyTorch bug (silent-correctness MPS convolution
  issues #96982, #142836, umbrella #149325), but the allocator-watermark form of it was not
  on file; the identity-convolution repro is in the workspace scratch and worth filing.

- **SynthStrip runs in-process too.** The second engine on the local seam: `synthstrip:mask`
  routes to `synthstrip.run_local` when `synthstrip_torch` is installed
  (`UV_PROJECT_ENVIRONMENT=.venvs/synthstrip uv sync --extra synthstrip --extra serve`; the
  fork needs numpy<2, which is why it has its own environment). The SDT restore takes the
  CPU path off CUDA, as before.

- **The inference stack is core.** torch, nnunetv2, scipy and scikit-image are plain
  dependencies, so `uvx git+https://github.com/mhalle/haversack segment ...` works with no
  extra - a segmentation tool that could not segment after `pip install` was a trap, and it
  caught the first user. `[torch]` remains as an empty alias so existing commands install.
  numpy is no longer pinned `>=2` in core (the synthstrip engine's numpy<2 fork must still
  resolve beside it), and `torch` leaves the conflict group. The lean, torch-free install
  (client / describe-only) is a documented `--no-deps` recipe (README "Lean install") -
  extras can only add - and such an install asked to segment still ends in one line naming
  what to add. The Modal images mount the package and hand-list their packages, unchanged.

- **`haversack segment` takes remote inputs.** `idc:<crdc_series_uuid>`,
  `zenodo:<recid>/<file>[!member]`, `tcia:`, `openneuro:`, `hf:` - the server's five sources,
  same grammar - and a bare `http(s)://` URL, with `!member` reading one file out of a remote
  zip by Range. `sources.materialize()` fetches once into `~/.cache/haversack/inputs`
  (`HAVERSACK_CACHE_DIR` moves it) and hands back the file, or the directory for a DICOM
  series. IDC needs the `idc` extra (obstore) and says so. The http(s) source is local only,
  deliberately: it is never in a server's registry, because a server fetching client-chosen
  URLs could be steered at anything it can reach and a URL is not a cache-grade identity.

- **Progress fractions move with the work.** A one-patch volume read `5 % loading`, `5 %
  preprocess`, `5 % predict`, `95 % restore`: every stage but restore started at the same
  number and only patch ticks moved it. Each stage now starts where the previous one's
  work ends (read 0, load 12, preprocess 22, predict 27 to 90, restore 90, finalize 97 %,
  sized from warm runs), the read and the final reorientation are reported as stages, and
  the last patch lands exactly on restore. The wire's `Haversack-Fraction` and `progress`
  fields change value, not shape.

- **Engines run in-process when their runtime is installed: FastSurfer locally.** The
  design note of 2026-08-26 planned a local FastSurfer for machines with the memory; the
  seams it named were already cut (`Engine.compute` reserved, grammar routing without Modal,
  `fastsurfer.segment` taking `mps`/`cpu`), and this fills them in. `Engine` gains
  `runtime_module` and `extra`; `registry.available()` asks `find_spec` whether the runtime
  is here, and `enabled()` treats an installed runtime as enabled when the env flag is
  unset (`=0` still switches it off; the Modal deploy's `=1` still switches it on).
  `Segmenter.segment`/`submit` route an engine task to `Engine.compute` and never to the
  nnU-Net pipeline; without the runtime they refuse with the per-engine install line
  (`UV_PROJECT_ENVIRONMENT=.venvs/fastsurfer uv sync --extra fastsurfer --extra serve`),
  because an engine owns its environment (pyproject's conflicts: fastsurfer-lean pins torch
  2.7.1). `haversack segment` and a local `haversack serve` follow, so a server started from
  the FastSurfer venv lists and runs `fastsurfer:brain`; the CLI's stack check is per engine
  and `haversack tasks` marks an engine installed when its runtime is. New in the engine:
  `viewagg_device` is plumbed through, and `local_viewagg()` places FastSurfer's 79-class
  aggregation field by reading the host - CUDA keeps FastSurfer's own rule, MPS keeps the
  field on the CPU below 32 GB of unified memory (it would otherwise swap), CPU is CPU.
  Verified FastSurfer-free (`tests/test_engines_local.py`) and by a dry call in the
  FastSurfer venv that stops at the reader; **not yet run end to end on a brain** - this
  16 GB Mac is below the field's comfortable size, and a machine with more memory should
  run the first one and compare against the Modal result.
- **A missing input is one line.** `haversack segment nope.nii.gz` traced back from inside
  SimpleITK; the CLI now says `input not found` before touching any stack.

- **No torch.jit.interface FutureWarning on every run.** torch 2.14 deprecates it, and
  `timm` (pulled in by nnunetv2's architecture package for its Primus/EVA models) trips it
  while nnunetv2 scans trainer modules for the checkpoint's class. Filtered at that scan
  (`trainers._resolvable`) and at the predictor.

- **A bare install says which extra it needs.** `uvx git+https://github.com/mhalle/haversack
  segment ...` installed the package without torch and died in a traceback from
  `pipeline.py` (2026-09-03). `segment` and the local `serve` now check for torch, nnunetv2
  and scipy before importing anything and end as one line naming `haversack[torch]` and the
  two install forms. A bare install stays torch-free on purpose: the client (`haversack
  remote`) and a describe-only front end never pay for it.

## [0.3.0] - 2026-09-02

- **nnseg is now haversack, in its own repository.** The package, the distribution, the
  command, the environment variables (`HAVERSACK_*`), the results cache
  (`~/.cache/haversack`), the Modal app and volumes (`haversack-serve`, `haversack-weights`,
  ...) and the error class (`HaversackError`) all carry the new name; nothing external keeps
  the old one except the weights sidecar, where `.nnseg-version.json` is still read so an
  installed model folder is not re-fetched. Wire: the progress and credential headers are
  `Haversack-Stage`, `Haversack-Fraction` and `Haversack-Source-Token` (were `NNSeg-*`); a
  client written against nnseg must follow. History came along: this repository is
  `nnunet-inference-mlx` filtered to nnseg's paths (275 commits), so `git log` and `git blame`
  reach back to the 2026-08-22 "increment 1" commit and beyond into the packaging files' MLX
  past; the MLX package itself stays behind in `nnunet-inference-mlx` as the oracle. The name
  is scope-neutral on purpose - the same kernel (restore any field onto any grid) and the same
  job protocol are meant to carry registration models next, and a `-seg` name would have been
  wrong the day that landed. Entries below this line say "nnseg"; they are history and were
  not rewritten. Test files lost their `test_nnseg_` prefix; the tests themselves are unchanged.

## [Unreleased before the rename]

- **The distribution is nnseg only (2026-09-02).** The wheel ships `src/nnseg` and one
  executable, `nnseg`. The MLX-era package and its three commands (`nnmlx`,
  `TotalSegmentator`, `totalseg-mlx`) are no longer packaged, and the `mlx` extra is gone:
  `uv tool install` had put all four commands on PATH, three of them crashing on `import mlx`
  and one shadowing a user's real TotalSegmentator. The MLX source stays in the repository as
  the oracle nnseg is checked against (its tests still run in the dev env with
  `uv pip install mlx`); its README is preserved as `docs/mlx-oracle-README.md`. The README is
  now nnseg's page (what was `docs/nnseg-getting-started.md`), with the venv, `uv tool
  install` and `uvx` install forms - all three verified from the pushed branch.

- **nnseg installs and runs from a clean Mac (2026-09-02).** Tried as an outsider would - fresh
  venv, `uv pip install "...[torch] @ git+https://...@feature/nnseg"`, empty weights root - and
  it failed twice before running anything. (1) uv applies `[tool.uv.sources]` to a git
  install, and the `nnunetv2 = { path = "../upstream/nnUNet" }` convenience source pointed at a
  path that exists only in the dev workspace; the source is gone (PyPI nnunetv2 2.8.1 is what
  nnseg is written against - stock APIs only), and the checkout goes in by hand for upstream
  work. (2) The wheel had no task catalog: hatch skips VCS-ignored files and the workspace
  `.gitignore`'s `data/` rule covered `src/nnseg/data`, so every task lookup died on a missing
  `ts_tasks.json`; `artifacts = ["src/nnseg/data/*.json"]` ships them regardless of gitignore
  (verified by building from the pushed branch's tree). With both fixed, `total_fast` on the
  chest CT ran on MPS in 68 s cold (38 s of it model load), 109/117 structures, 99.97 %
  agreement with the recorded result; a `nnseg serve` job on the same Mac was voxel-identical
  to the CLI. Around it: `nnseg tasks [--installed] [--json]` lists the catalog locally without
  importing torch ("installed" asks the weights store, not `materialized`, which for TS only
  says the spec is known); nnseg's own errors end as one `nnseg: <message>` line with status 2
  instead of a traceback (the missing-serve-extra `InputError` was reaching the user as a raw
  `ModuleNotFoundError`); single-model timing keys read `load:ts:total_fast`, not
  `load:ts:total_fast:ts:total_fast`; nnunetv2's non-CUDA `print` and its old-plans-format
  warning are silenced at the predictor. New: a getting-started page (requirements, install, weights, CLI, API, local server) -
  now the README itself, see the entry above.

- **Previews are radiological, and say so (nnseg).** The three-plane preview showed the
  patient's right on the image right - not by decision but because the loader's RAS array
  was handed to `imshow` as-is, and nothing stated a convention. Found while checking
  MRSegmentator's left/right; confirmed on `ts:total_fast` from the same CT (liver plotted
  at column 327 of 512). `preview.DISPLAY` now names the convention (`radiological`, 3D
  Slicer's default: patient right on the image left in axial and coronal, sagittal viewed
  from the patient's left so anterior is on the image left), `preview.display_planes()`
  turns it into a per-panel orientation frame and gets there through
  `io.orientation_transform` - the same DICOMOrient probe the pipeline trusts, applied as
  a transpose+flip view, never a hand-written `[:, ::-1]` - and the R/L, A/P, S/I edge
  letters are read back from the resulting direction cosines and drawn on every panel, so
  a wrong convention is visible instead of silent. `neurological` remains available as a
  named option. Tested on a laterality phantom stored in five axis orders and read back
  from the PNG's pixels (`tests/test_nnseg_preview.py`). `load_oriented_pair` is
  unchanged (statistics still read the RAS pair), and cache keys do not move; previews
  already in a results cache keep their old appearance until regenerated.

- **MRSegmentator ecosystem (nnseg).** `mrsegmentator:base` (40 abdominal / pelvic / thoracic
  structures on MRI, also usable on CT) and `mrsegmentator:body_comp` (10 body-composition
  classes) join the catalog as a fourth nnU-Net ecosystem beside `ts`, `moose` and `custom` -
  the same shape as MOOSE (bare checkpoints on public assets, labels read from each installed
  checkpoint's own `dataset.json`, `@version` pins checked against the bytes on disk), and
  `nnseg serve` lists and installs them through the existing prepare path with no worker change.
  Two things are properties of upstream's *packaging* rather than of the checkpoints, and both
  are recorded in the manifest / spec rather than guessed: the zips are **flat** (a
  configuration folder with no `Dataset*` parent), so each one is installed under
  `<root>/mrsegmentator/<Dataset>/<trainer>__<plans>__<config>/` through a staging directory
  and one rename, with the zip's own `version.json` checked against the manifest tag; and
  MRSegmentator's reader **forces LPS** on top of plans that declare the non-reorienting
  `SimpleITKIO`, so the spec carries `orientation="LPS"`. That needed one small generalization:
  `TaskSpec.orientation` (None = follow the declared reader, as before), `io.read(target=...)`,
  and `pipeline.canonical_orientation_for()`, the one place the decision is made; provenance
  and ranked-store metadata gain `canonical_orientation` beside the unchanged
  `reoriented_to_ras`. Existing tasks are byte-for-byte unaffected (RAS for `ts`, stored order
  for plain nnU-Net models). `tools/gen_mrsegmentator_manifest.py` regenerates the manifest
  from upstream's `MODEL_REGISTRY` and reads dataset name, trainer and version out of each 1.1
  GB zip by Range request (a few MB, no download). nnseg's default fold policy (fold 0, no
  mirroring) matches upstream's `--fast`; pass `folds=[0,1,2,3,4]` for its default ensemble.
  Validated on three AMOS22 MRI cases against AMOS ground truth (fold 0): mean Dice 0.799 vs
  TS `total_mr` 0.792 on the 13 shared organs, higher on 9, every paired structure on the
  annotated side (`bench/results/mrseg_vs_tsmr_amos/` in the workspace). Still owed:
  voxel-level parity against MRSegmentator's own package output.

- **Volume and surface area measured from the field, not the mask (nnseg).** `nnseg.measure`
  integrates the margin field over the cells between voxel centers — full cells exactly,
  straddling cells by the plane their corners imply — so `V = int H(m)` and
  `A = int delta(m)|grad m|` are evaluated on the interpolant the store already defines.
  Area is the volume expression differentiated in the level, so the two cannot disagree
  about where the surface is. Against closed-form phantom truth at 1.5 mm, counted area is
  **+39 to +54 %** and does not converge (four grid refinements leave a sphere at +50.4,
  +50.9, +49.3, +50.9, +50.7 %) where the field is within 0.3 % and converges at O(h^2). The
  fair raster baseline is not face counting but SimpleITK's Crofton `ComputePerimeter`,
  which is within a couple of percent on a smooth body; the field beats it 0.14 % vs 0.85 %
  on sub-voxel stability, −4.5 % vs −16.6 % on a crease, converges monotonically where
  Crofton bounces, and needs no raster at all. Meanwhile
  counted volume swings −10.3 to +4.2 % across shapes where the field holds −0.6 to −0.1 %.
  Under pure sub-voxel translation counting saws by 1.10 % of a sphere's volume and 2.69 %
  of its area; the field by 0.00 % and 0.14 %. On real TotalSegmentator output the
  count/field area ratio is 1.51–1.68 across six structures. **Pass
  `clip=code.meta["clip"]`**: a straddling cell's far corners saturate wherever the margin
  climbs faster than the clip over half a cell diagonal, which is 30–95 % of cells at real
  gradients (3.0–7.2 logits/mm), and reading those bounds as values costs 1–5 % of the
  surface. `nnseg.statistics` passes it. Full account, including what does *not* help
  (subdivision, depth) and what is still biased (creases, −4.5 %), in
  [`docs/ranked-measurement.md`](docs/ranked-measurement.md).

- **`statistics.json` can carry both measurements (nnseg).** `compute_statistics` takes an
  optional `ranked_code` and then reports `volume_ml_field` and `area_cm2_field` beside
  `volume_ml`, with the field's own `field_grid_spacing_mm` — the code lives on the model
  grid, the labelmap has been restored onto the input's, and an area must never be compared
  across grids even where a volume may be. Without a code the output is byte-identical to
  before, so both can ship until the comparison has run on a real cohort. **The server does
  not pass one yet**: `artifact_overlap` computes from the in-RAM pair with no disk
  dependence, so the served `statistics.json` and `.tsv` are unchanged for now.

- **Analytic phantoms (nnseg).** `nnseg.phantoms` turns geometry with closed-form volume and
  area into logits, so anything reading the ranked field can be scored against calculus
  rather than against another measurement. A segmentation model cannot supply that — render
  a synthetic image, run a network on it, and the ground truth evaporates, because the
  decision surface is not the surface that was drawn. Sphere, ellipsoid, torus, shell, box,
  rounded box, star and an n-sector partition; closed forms cross-checked against a
  Gauss–Legendre × trapezoid rule with partials by autograd (the ellipsoid area agrees with
  `scipy.special.elliprg` to 12 digits). `tools/field_vs_counting.py` is the standalone
  comparison harness.

- **Fixed: multi-model tasks ran on one model's normalization (nnseg).** `segment()`
  cached the preprocessed frame by spacing alone, but the cached tensor had nnU-Net's
  **per-model** normalization baked in, so any task running several models at one
  spacing handed models 2..N the first model's intensity statistics. For `total` that
  meant the organs model's CT clip at +276 HU applied to vertebrae, cardiac, muscles
  and ribs, collapsing every bone density above it: on CT_Abdo, ribs 273 -> 412.9 mL,
  sternum 12.1 -> 59.6, costal cartilage 5.8 -> 128.6. `to_model_frame` is now
  `to_model_grid` (crop + resample, model-independent and therefore shareable, and a
  different type from a network input so it cannot be fed to one) plus `normalize_for`
  (per-model, fresh tensor), and the input carries a normalization fingerprint the
  consumer checks. **`__version__` is bumped to 0.2.0, which invalidates every cached
  serve result** - `total` results cached before this are degraded. Unaffected:
  `total_mr` (ZScoreNormalization reads no dataset statistics), single-model tasks,
  and the engines layer.

- **The job queue is durable (nnseg).** Records live in a sqlite `jobs.db` under
  the server's workdir, so a restart re-queues work instead of dropping it - a job
  is content-keyed and idempotent, so an interrupted run is re-runnable. It also
  gives job directories an owner, so the ones a previous run left behind are
  reclaimed rather than stranded forever holding their uploads. `--jobs-ttl-hours`
  bounds how long a record lasts (`--keep-finished` bounds memory and files).
  `tools/jobs.py` reads the store without a running server.

- **Named inputs, typed parameters, introspection (nnseg).** A task now publishes
  what it takes: `describe()` carries `inputs` (roles), `parameters` grouped by
  owner (the algorithm's knobs vs ours, as real JSON Schema), and `behavior` -
  read-only facts about what an engine does that a caller cannot change. Sources
  bind to roles **by name, never by position**: MONAI's BraTS bundle orders its
  channels T1c-first while nnU-Net's own convention puts FLAIR there, so a
  positional wire would silently mis-serve one of them. New `nnseg.schemas`
  generates the published schema and enforces it from one declaration; pydantic
  becomes a core dependency but stops at the wire. Options outside a task's
  schema are now refused at submit, which means `device`/`dtype`/`weights`/
  `batch_size`/`accumulate` are no longer settable remotely - deployment policy,
  not per-request knobs.

- **A content-addressed input store (nnseg).** Upload once, refer by digest
  thereafter: `GET/PUT /v1/inputs/{digest}`, `POST /v1/inputs` for a DICOM series
  (parts or a zip) as one tree, and a `{"kind":"input","sha256":...}` source. The
  server does the addressing - a declared digest is checked, never trusted. A
  blob's key is exactly the pre-existing upload identity, so no cached result
  moved. A tree's key is rooted in the sorted digests of its members, so it
  survives arrival order, filenames and zip metadata.

- **Results carry their content digest.** A finished job publishes `outputs`; the
  ETag is that digest and `If-None-Match` is honored (304), so a client stops
  re-downloading label volumes it already holds. `POST /v1/inputs?from_job=<id>`
  promotes a result into the input store, so one job's output can be another's
  input without the bytes routing through the client.

- **MONAI bundles can be multi-channel.** The engine takes a role->image mapping,
  orders channels by the bundle's own `channel_def`, and refuses inputs that do
  not share a grid - nnseg does not register images, and says so in the task's
  `behavior.alignment`. `brats_mri_segmentation` is curated as the first
  multi-input task. A region head (overlapping output channels, no `background`
  entry) is no longer read as a labelmap on either side.

- **Fixed: engine workers published results under a key nobody computes.** Every
  MONAI job recomputed forever - the worker's describe shim reported "unknown"
  weights for an engine whose identity is per task, so `publish_completion`
  re-keyed the result onto a key the API never computes. Its docstring already
  claimed the two "cannot drift"; there is now a test that checks it.

- **Modal:** a new `nnseg-inputs` Volume for the input store, mounted on the API
  function and every worker; the `jobs` volume is now `scratch`.

- **Engine / Ecosystem split (nnseg).** The two ideas that one class used to carry
  are now separate: an **ecosystem** is the user-facing *catalog* (`ts`, `moose`,
  `custom`, `fastsurfer`, `synthstrip`, keeping the `eco:task@version` grammar) and
  an **engine** is the *runtime* that runs it (`nnunetv2`, `fastsurfer`,
  `synthstrip`) - many ecosystems to one engine. A static registry
  (`nnseg.engines.registry`) is now the single source of truth for dispatch,
  enablement, knob forwarding, and the weights identity that keys the result cache;
  the engine ecosystems stop faking a `TaskSpec`, the per-engine describe shims are
  gone, `_spawn_worker` routes by grammar instead of hardcoded task prefixes, and
  the three Modal workers share one base class. Adding an engine is now a registry
  row plus a worker class that declares its image and compute. `/v1/tasks` reports
  `engine` per task and `/v1/version` lists the engines a deployment can run.
- **Renames that came with it** (nothing is deployed, so no aliases): the `native`
  ecosystem is now **`custom`**; `TaskSpec.source` is now **`lineage`** with value
  `nnunet` -> `nnunetv2` (freeing "source", which already meant *data source* on the
  wire); `WeightsStore(ecosystem=)` is now **`layout=`** with values
  `totalsegmentator`/`nnunet` -> `ts`/`nnunetv2` (it selects a weights tree, not a
  catalog); and `_is_native` is now `_uses_nnunet_preprocessing`, which is what it
  actually selects. Upstream names (`TOTALSEG_WEIGHTS_PATH`, `nnUNet_results`) are
  untouched.

- **All-GPU forward pipeline: reorient + resample on Metal.** `to_model_frame`
  gains `interpolation="auto"` (now the default): a per-axis Metal resampler
  (`resample_volume_mlx`) — factor-scaled, clamped anti-aliased cubic on
  *downsampling* axes (no aliasing of thin/high-contrast structure), linear on
  *upsampling/near-identity* axes (no cubic ringing — important for thick-slice
  CT's through-plane axis); ~0.5 s on a 418 M-voxel volume. And a GPU reorient
  (`reorient_array_mlx`) — permutation+flips derived from the direction cosines,
  bit-identical to `sitk.DICOMOrient` across RAS/LPS/SPL and arbitrary codes,
  replacing the ~5.5 s of CPU `DICOMOrient` memory-shuffle (forward 3.2 s + inverse
  2.3 s) with ~0.9 s on Metal. `restore` uses it for the inverse too. Net: chest
  fast-mode end-to-end **75 → 57.6 s**, output **bit-identical** to the prior SITK
  path. SITK now does only file IO + geometry. `"linear"/"bspline"/"nearest"`
  still route to SITK.
- **Fused Metal kernel for the logit restore (~100×).** The default linear
  `restore` was memory-gather-bound — 8 full-array corner fetches of the K
  logit channels, blend, then a separate argmax, materializing ~8× the
  K-channel output transiently. A single `mx.fast.metal_kernel` now does the
  whole inverse resample with one thread per *output* voxel: trilinear-
  interpolate all K channels inline (only the 8 source corners per channel)
  and reduce to one integer label on the fly. Nothing K-channel-sized is
  materialized, so there's no slab budget to tune (`peak_working_memory_mb`
  is ignored on this path). On ct.nii (512×512×165, 117ch) restore drops
  **24.7 s → 0.25 s (~98×)**; end-to-end `segment` **43 s → 11.5 s**. The
  pure-MLX slab path is retained as a fallback (`use_fused_kernel=False`, or
  automatic on kernel error). Region models get the same treatment via a
  fused threshold-paint kernel. Numerics: same separable blend op-order as
  the slab path → bit-identical on synthetic logits; on real smooth fields a
  handful of boundary voxels (~4e-5) flip on FMA-contraction rounding,
  negligible next to MLX↔PyTorch divergence. (Supersedes the earlier ~28%
  flattened-`mx.take` gather, an interim win on the now-fallback slab path.)
- **Fast nearest-neighbor inverse resample (path A), opt-in.** `restore` /
  `LoadedModel.segment` / `segment` gain `interpolation`/`output_interpolation`
  (`"linear"` default = logit interpolation, higher fidelity, like nnU-Net;
  `"nearest"` = argmax-at-model-spacing then NN-resample the label map, like TS).
  Profiling showed the default logit restore is memory-gather-bound (8-corner
  fetch of K=117 logit channels): on `ct.nii` (512×512×165) it's ~32 s; the NN
  path is **~0.4 s (≈75× faster)** with **98.5% agreement** (differs only at
  boundaries). So MLX end-to-end with `--resample nearest` ≈ ~7 s vs ~43 s
  (linear) / ~23 s (TS-MPS). CLIs: `nnmlx segment --resample linear|nearest`;
  `totalseg-mlx` maps TS's `--higher_order_resampling` (default NN like TS, `-ho`
  → logit interp).
- **Native weight download is live for TotalSegmentator.** TS v2 weights are public
  GitHub release zips, keyed per dataset id. A build-time generator
  (`scripts/refresh_ts_weights.py`) extracts the id→URL map from TS's
  `download_pretrained_weights` source into a shipped `data/ts_weights.json` (42
  datasets; 1 license-gated, flagged) — TS is never imported at runtime, same
  relationship as `ts_tasks.json`. The `totalsegmentator` store now has a default
  `fetch` that downloads the URL → verifies → unpacks, so `store.download(id)` works
  out of the box (verified end-to-end against GitHub). License-gated/unknown ids raise
  actionably. `nnunet`/`moose` have no default fetch (place locally or inject one).
- **CLIs auto-download by default (TS-like), opt-out.** `TotalSegmentator`/`totalseg-mlx`
  gain `--no-download`; `nnmlx segment` gains `--download/--no-download` (default on).
  Missing weights for the requested task are fetched before inference (`segment`'s new
  `required_weights_ids` resolves single/cascade/union ids). The *library* default stays
  explicit (no auto-download) — only the executables mimic TS. `download_archive` shows a
  tqdm byte-progress bar (auto-enabled on a TTY via `disable=None`, silent in pipes/tests).
- **`ModelStore.download(ids, *, force=False, build=False)` contract** — idempotent
  "ensure present" (fetch only what's missing; a no-op for present ids — the disk-layer
  twin of `load`'s read-through), with `force` to re-fetch. Returns the ids actually
  fetched. The fetch is an injectable `fetch(id, model_root)` seam; with none configured,
  a missing id raises actionably (pointing at the upstream downloader) instead of silently
  doing nothing. Added `verify_and_unpack(archive, sha256, dest)` (checks the archive's
  SHA-256 against the recipe's `weights_sha256` *before* unpacking — `.pth` is pickle, so
  this is a supply-chain gate, not just corruption detection; writes a `.verified` sidecar
  to avoid re-hashing) + `download_archive(url, dest)` + `sha256_file`. CLI: `nnmlx models
  download <ids> [--force]`. (The actual remote fetch / recipe URLs land with 0.11.)
- **HTTP client is now `httpx`** (was `requests`): `download_archive` and the torch-free
  range loader (`_torchfree/rangefile.py`, `load_pth_url`/`smart_load_url`) use httpx with
  `follow_redirects=True` (release assets 302 to a CDN). The `remote` extra is now just
  `httpx` (dropped `requests`/`remotezip` — `load_pth_url` reuses our `CachingRangeFile`).
- **Confirmed RAS is nnU-Net v2's universal canonical** (not just TS): the installed
  nnU-Net readers reorient inputs to RAS (`SimpleITKIO.read_images(orientation="RAS")`,
  `NibabelIO` to RAS) and back on write. So the RAS default is correct for all
  nnU-Net v2 ecosystems (TS, MOOSE, raw nnUNet); rationale documented in
  `preprocess.to_model_frame`.
- **Migrated `examples/` to the toolkit API** (the old ones imported removed
  symbols): `01_single_volume`, `02_batch_folder`, `03_logits_and_resolution`,
  `04_cascade_and_union`, `05_toolkit_namespaces`, plus a rewritten `examples/README`.
- **Expanded the `@slow` real-weights suite** (`test_real_weights.py`): organ-volume
  sanity, a non-default-trainer model (`Dataset117`), and skip-guarded region /
  MOOSE tests that run automatically once such weights are present. (Region,
  multi-fold, and MOOSE remain unvalidated on real data — no such weights downloaded.)

## [0.10.0] - 2026-05-31 — toolkit rearchitecture

Lands the composable toolkit API and removes the old hidden-state surface. Still
**pre-1.0** — breaking changes are expected, and 1.0 is gated on broader testing
(real-weights integration coverage, more tasks/ecosystems exercised).

### Fixed — left/right mirror in segmentation output (canonical orientation RAS, not LPS)

The inference canonical orientation was wrongly set to **LPS**, which mirrors the
volume left↔right vs **RAS** (nibabel's `as_closest_canonical`, what nnU-Net /
TotalSegmentator train and serve in). Since the network is not L/R-equivariant,
it saw a mirrored volume and produced **left/right-swapped labels** — structures
in the right *places* but with `left`/`right` reversed. Confirmed against the TS
mainline reference (`mlx-LEFT-lung` matched `ref-RIGHT-lung` at 0.93 Dice under
LPS; 0.97 vs `ref-LEFT` under RAS) and by anatomy (liver landed on the patient's
left). Default `reorient_to` is now `"RAS"` across `segment` / `LoadedModel` /
`preprocess.to_model_frame` / the CLI. This was a **pre-existing** port bug
(invisible to synthetic tests, which have no L/R semantics, and to old-vs-new
parity, which shared the same flip); added a real-weights `@slow` regression
(`test_real_weights.py`) asserting liver/spleen sit on the correct sides.

### Added — `nnmlx` CLI (Typer)

A command-line shell over the toolkit, so real-weights runs are one command:

```
uv run nnmlx segment total_fast ct.nii.gz seg.nii.gz
uv run nnmlx tasks list --modality CT
uv run nnmlx tasks show total
uv run nnmlx models list
```

Command groups: `segment` (run a named task → NIfTI), `tasks list`/`tasks show`
(catalog inspection), `models list`/`models loaded` (store inspection). Shared
`--ecosystem` / `--model-root` / `--max-memory-mb` on the top-level callback;
each command builds an explicit request-scoped `ModelStore` + `TaskCatalog` (no
global state). `typer` added to core deps; entry point `nnmlx`, also
`python -m nnunet_inference_mlx`.

### Added — `totalseg-mlx`, a TotalSegmentator-compatible CLI

A drop-in front end mirroring TotalSegmentator's `TotalSegmentator` argparse
(every flag parses), so existing TS command lines/scripts run on the MLX backend
unchanged:

```
totalseg-mlx -i ct.nii.gz -o segmentations            # one mask per class (TS default)
totalseg-mlx -i ct.nii.gz -o seg.nii.gz --ml          # single multilabel file
totalseg-mlx -i ct.nii.gz -o seg --fast -rs liver spleen -s
```

Supported: `-i/-o`, `-ot nifti`, `-ml`, `-f/--fast` + `-ff/--fastest` (→ our
`_fast`/`_fastest` tasks), `-ta/--task`, `-rs/--roi_subset`, `-rmb/--remove_small_blobs`,
`-s/--statistics` (volume mm³ + mean intensity → `statistics.json`), `-ss/--skip_saving`,
`-q/-v`, `--version`. Per-class output writes `{roi_name}.nii.gz` into the `-o`
directory, exactly like TS. Flags the MLX backend doesn't implement (radiomics,
nora, dicom output, body_seg/force_split, save_probabilities, license, …) are
accepted and ignored with a warning, so command lines don't break. The native,
non-TS interface remains `nnmlx`.

Entry points **`TotalSegmentator`** (a literal drop-in name — existing command
lines run verbatim under `uv run`) and `totalseg-mlx` (same front end; doesn't
shadow a real TS install). Console output **mimics TS**: citation line,
`Using 'fast' option...`, `Predicting...` / `Predicting part i of N ...` (per
union part), `  Predicted in Xs`, `Saving segmentations...` with a tqdm bar,
`  Saved in Xs`. To support that, `segment()` / the per-shape runners gained an
optional `progress` callback (invoked with short phase strings) so CLIs report
progress without the toolkit owning any console output.

### Added — output resolution control

`segment` / `LoadedModel.segment` / `postprocess.restore` gain output-resolution
knobs (mutually exclusive; default = the input grid):

- `--output-scaling S` — resolution multiplier (2 = finer/half-spacing, 0.5 = coarser).
- `--output-spacing MM` — absolute isotropic spacing.
- `--at-model-spacing` — the model's native training grid (no upsample back).

The labels are rendered **from the logits** at the requested grid (then argmax/
paint), not nearest-neighbor-resampled from a finished label map — higher
quality. The output header (spacing/shape/origin/direction) is recomputed over
the same physical extent, so it still overlays the input; `scaling=1` is a true
identity. Single-model tasks only for now (cascade/union raise). Downsampling
stays Nyquist-limited. SITK is now a core dependency (segmenting == image I/O),
so `uv run nnmlx segment …` works with no extra flags.

A composable toolkit API with **no hidden state**: three nouns + one verb —
`TaskCatalog` (name→recipe), `ModelStore` (id→model; read-through, bounded,
freeable), `segment` — over frozen value types (`Geometry`/`Volume`/
`Segmentation`/`Prediction`/`LabelSchema`/`RestorePlan`/`BuildOptions`) and
pure-fn stage namespaces (`preprocess`/`infer`/`postprocess`/`geometry`). Now the
package's public surface.

### Removed — the old hidden-state surface (breaking; Phase 5 cutover)

- **Module-global engine cache** (`engine_cache.py`: `cached_engine_from_*`,
  `get_cached_engine`, `clear_engine_cache`, `resolve_moose_config_folder`) →
  replaced by the owned, bounded `ModelStore`.
- **Module-global task registry + dispatcher** (`tasks.py`: `register_task`/
  `get_task`/`run_named_task`/`list_registered_tasks`/…) → replaced by the owned
  `TaskCatalog` (lookup) + `segment()` (dispatch). The recipe vocabulary
  (`TaskSpec`/`CascadeStep`/`UnionPart`/`AmbiguousTaskError`) is kept.
- **Old SITK orchestration** `workflow.py` (`run_workflow`/`run_label_union_workflow`/
  `Stage`/`ParallelStage`/`Bbox`/`compute_fg_bbox`/`crop_image`/`paste_segmentation`)
  and `resampling.predict_with_resampling` → replaced by `segment` composing
  `preprocess`/`infer`/`postprocess` + `geometry`.
- **Weights-layout discovery** (`WeightsLayout`/`discover_weights`/
  `register_weights_layout`) and `ModelBundle.from_folder`/`from_task` → folder
  reading now lives in `ModelData.read_folder`.
- Kept the low-level resampling primitives (`resample_image_to_target`,
  `inverse_resample_*`, `reorient`, `get_orientation`), the label primitives
  (`remap_labels`/`paint_union`/…), and `InferenceEngine` (now a private compute
  core). Verified end-to-end on real TotalSegmentator weights via the new API.

### Changed — forward resample now runs in float32 (behavior change)

The new pipeline casts the input to **float32 at read** (`sitk_to_volume`) and
resamples in float, matching nnU-Net v2's reference preprocessing. The legacy
`predict_with_resampling` resampled the raw int16 SITK image, which rounded
interpolated HU values to integers. On real int16 CT (TotalSegmentator
Dataset297, abdominal) this rounding accounted for **all** of the new-vs-old
divergence: 99.973% of voxels identical, differences confined to organ
boundaries. With both paths resampling in float the outputs are **bit-identical**
(verified end-to-end on real weights, same engine). Float is the intended
behavior going forward.

## [0.9.2] - 2026-05-27

### Added — source-aware engine resolution (MOOSE models)

`run_named_task` now picks an engine factory by the task's `source`:

- `ts` / `user` → integer nnU-Net dataset IDs via `cached_engine_from_task` (unchanged)
- `moose` → string folder names via the new **`cached_engine_from_moose_model(folder_name, models_dir=...)`**

MOOSE stores nnU-Netv2 models one folder per model (`Dataset123_Organs`) under a flat models dir — same inner `{trainer}__{plans}__{res}` config layout TS uses, only the outer identifier is a string. New **`resolve_moose_config_folder(folder_name, models_dir)`** maps the folder name to the config folder `cached_engine_from_folder` consumes. The MOOSE models dir resolves from `models_dir=` → `NNUNET_MLX_MOOSE_MODELS` / `MOOSE_MODELS` env vars → the installed `moosez` package, with a clear error if none resolve. `run_named_task` gains a `moose_models_dir=` parameter. Both new functions are exported.

### Added — nested-task cascade (`CascadeStep.crop_from_task`)

A cascade step can now reference another *registered task by name* instead of an inline `weights_id`:

```python
CascadeStep(crop_from_task="craniofacial_structures", crop_to_classes=(2, 7))
```

The dispatcher flattens the reference — recursively — into the `run_workflow` stage list: the referenced task (which may itself be `single` or `cascade`) runs, and its final stage carries the outer step's crop. `CascadeStep` validates exactly one of `weights_id` / `crop_from_task`.

This unblocks **TS `teeth`** (previously skipped by the generator). teeth crops from `craniofacial_structures`, which is itself a cascade — so it flattens three deep: `298 (rough total, crop→skull) → 115 (craniofacial, crop→teeth 2,7) → 113 (teeth)`. The TS registry now ships **51 tasks** (was 50; 23 single / 25 cascade / 3 label_union). The same mechanism covers MOOSE's two FOV-limited models (0.9.3).

### Added — `int | str` weights identifiers (MOOSE-ready)

A source audit of MOOSE v3.1.6 confirmed MOOSE identifies models by **string** (`"clin_ct_organs"` / `"Dataset123_Organs"`), not the integer dataset IDs TS/nnU-Net use. So the weights identifier is now the union type **`WeightsId = int | str`** (exported from the package root):

- `TaskSpec.single`, `CascadeStep.weights_id`, `UnionPart.weights_id` accept `int | str`.
- JSON (de)serialization preserves the incoming type — an integer stays an nnU-Net dataset ID, a string stays a MOOSE model identifier; TS integer IDs are *not* silently stringified.
- `run_named_task`'s default `engine_factory` resolves only integer IDs (via `cached_engine_from_task`); a string identifier without a custom factory raises a clear `NotImplementedError` rather than globbing for a bogus `Dataset` folder. Source-aware string resolution lands with MOOSE support (0.9.3).

6 new tests (`TestStringWeightsId`) cover string-id validation, type-preserving round-trip, and the factory guard. The full MOOSE scope — grounded in the v3.1.6 audit — is in `docs/post-0.8.2-roadmap.md`.

### Added — source-qualified task names (anticipating multi-system registries)

The registry now keys on `source:name` (e.g. `ts:total`, `moose:total`, `user:mytask`) so two model systems can ship a task with the same bare name without colliding. This is forward-prep for MOOSE support (0.9.3) — done now, while TS is the only source and there's nothing to migrate.

- **`TaskSpec.qualified_name`** property → `"source:name"`.
- **`get_task(name)`** accepts bare or qualified names. A bare name resolves when exactly one source defines it (the common case today); when multiple sources define it, it raises the new **`AmbiguousTaskError`** with the qualified alternatives. A qualified name (`"ts:total"`) always resolves directly.
- **`register_task`** keys on the qualified name — `ts:total` and `moose:total` coexist; re-registering the *same* qualified name still needs `overwrite=True`.
- **`unregister_task`** / **`run_named_task`** accept bare or qualified names with the same resolution rules.
- **`list_registered_tasks(*, source=None)`** returns sorted qualified keys; pass `source` to filter to one system. **`list_tasks_by_modality`** likewise returns qualified keys.
- **`TaskSpec.name`** may no longer contain `:` (reserved as the separator) — validated in `__post_init__`.

`AmbiguousTaskError` is exported from the package root.

### Added — `uv run`-native admin CLI for the registry (PEP 723)

`scripts/refresh_ts_registry.py` is now a two-subcommand admin tool — the maintainer-side counterpart to the user-facing `mlxseg` CLI (which never needs TotalSegmentator). It declares its `totalsegmentator` dependency via a PEP 723 inline `# /// script` block, so `uv run` provisions TS in an ephemeral environment on demand. TS (and its torch stack) never enters the package's own dependency set.

```bash
# Regenerate the shipped registry in place
uv run scripts/refresh_ts_registry.py generate --write

# Or emit to stdout for inspection
uv run scripts/refresh_ts_registry.py generate

# Verify the committed JSON is in sync with the generator
uv run scripts/refresh_ts_registry.py check
```

`check` exit codes are designed for CI automation (deferred): `0` in sync, `1` drift at the same TS version (regenerate), `2` committed file built against a different TS version than is currently resolved, `3` TS unimportable / file missing.

The generator output is now **deterministic** — the wall-clock `generated` timestamp was removed from `_meta` (it made every run diff). `_meta.ts_version` records what the data was built from; git history records when it changed. Two runs at the same TS version are byte-identical, so `check` only reports a real change.

Note: the "is the committed registry stale?" check is deliberately *not* a pytest — it requires provisioning the heavy TS/torch stack, which we keep out of the test environment. It lives in the admin CLI (`check`), to be wired into CI later. The unit suite validates the committed fixture's schema and content against our own code (`TestBuiltinRegistry`), which is the part that belongs there.

### Added — TS registry generator + 50 populated TS task entries

`scripts/refresh_ts_registry.py` extracts task definitions from a `totalsegmentator` distribution (provisioned via `uv run`, see above) and emits `src/nnunet_inference_mlx/data/ts_tasks.json`. The shipped JSON contains **51 TS tasks** (23 single, 25 cascade, 3 label-union) — including `teeth` via the nested-task cascade above — covering essentially the full TS v2.13.0 catalog.

### How the generator works

TS's task dispatch lives in `totalsegmentator.python_api.totalsegmentator()` as a flat `if task == "..." / elif` chain. Each branch is pure variable assignment:

```python
elif task == "lung_vessels":
    task_id = 117
    resample = [0.703125, 0.703125, 1.0]
    trainer = "nnUNetTrainerSkeletonRecall"
    crop = ["lung_upper_lobe_left", ...]
    robust_crop = True
```

The generator:

1. AST-locates the dispatch chain in the function source (just for the line span — no walking)
2. Dedents and `exec()`s it in a controlled namespace with `task` / `fast` / `fastest` set to each combination
3. Reads the resulting locals (`task_id`, `resample`, `trainer`, `crop`, ...)
4. Classifies by shape: list of IDs → label-union, crop=None → single, crop=names → cascade
5. Builds TaskSpec entries, resolving crop class names against `class_map["total"]` / `class_map["total_mr"]` / `class_map["body"]` as appropriate
6. Writes JSON to stdout

The "exec a code block in a controlled namespace" approach is more robust than AST walking — it tolerates TS syntax changes (the dispatch could become a match/case in a future TS version and our generator still works) and offloads Python's syntactic complexity to Python itself.

### What's covered

| Shape | Count | Examples |
|---|---|---|
| `single` | 23 | `body`, `body_mr`, `appendicular_bones`, `tissue_types`, `total_fast` (297), `total_fastest` (298), `total_mr_fast` (852), all the focused single-shot models |
| `cascade` | 24 | `lung_vessels`, `liver_vessels`, `liver_segments`, `cerebral_bleed`, `coronary_arteries`, `craniofacial_structures`, `kidney_cysts`, etc. — all the "rough total → focused" patterns |
| `label_union` | 3 | `total` (5 parts on CT, datasets 291-295), `total_mr` (2 parts on MR, datasets 850/851), `test` |

The `fast`/`fastest` flags on `total` and `total_mr` are expanded into separate registered names (`total_fast`, `total_fastest`, `total_mr_fast`, `total_mr_fastest`) — each is a distinct logical task with its own dataset ID.

### Skipped (with reasons)

- `total_v1`, `covid`, `appendicular_bones_auxiliary`, `face_mr_auxiliary`, `kidney_cysts_auxiliary` — present in TS's `class_map` but absent from the dispatch chain (legacy, deprecated, or auxiliary internal tasks).

(`teeth` was skipped in the first generator pass — its `crop_model` reference needed nested-task cascade support, now added above; it's included.)

### Usage

```bash
# Run inside an env with totalsegmentator installed
python scripts/refresh_ts_registry.py > src/nnunet_inference_mlx/data/ts_tasks.json
git diff src/nnunet_inference_mlx/data/ts_tasks.json
```

Now `run_named_task("total_fast", img)`, `run_named_task("lung_vessels", img)`, etc. work out-of-the-box (provided the underlying TS weights are downloaded; we don't manage that yet — that's the 0.11.0+ remote-download work).

### Tests

New `test_tasks.py` coverage: `TestBuiltinRegistry` (TS-catalog breadth + canonical pins), `TestCrossSourceConflicts` (qualified-name resolution + `AmbiguousTaskError`), `TestStringWeightsId` (MOOSE-style string IDs), `TestMooseEngineResolution` (folder resolution + source routing), and `TestNestedCascade` (crop_from_task flattening incl. the 3-deep `teeth` case). Total: **244 passing** (`-m "not slow"`), 1 skipped, 2 heavy benchmarks deselected by default. ~7s for the fast suite.

- `test_ts_tasks_present` — assert ≥ 40 tasks registered (catches generator regression)
- `test_canonical_ts_tasks_resolve` — verifies popular task names (`total`, `total_fast`, `body`, `lung_vessels`, …) are all registered with `source="ts"`
- `test_total_is_label_union_with_5_parts` — flagship task structurally is 5-part union with dataset IDs `{291, 292, 293, 294, 295}`
- `test_total_fast_is_single_model` — `total_fast` → dataset 297
- `test_lung_vessels_is_cascade` — 2-stage cascade ending at dataset 117

These assertions are pinned to TS v2.13.0. Future TS releases that change dataset IDs will fail these tests, surfacing the change at refresh time rather than silently.

### What's still TBD

- **CI workflow** to auto-refresh on a cron schedule + open a PR if `ts_tasks.json` diffs — not blocking; manual refresh (`uv run scripts/refresh_ts_registry.py generate --write`) is one command.
- **MOOSE registry generator + populated `moose_tasks.json`** — the engine resolution (`cached_engine_from_moose_model`) and nested cascade (`crop_from_task`) groundwork is now in place; remaining is `refresh_moose_registry.py` + a verification pass on a downloaded MOOSE model's `plans.json` (0.9.3).
- **MOOSE PET multi-channel** — only 2 models; deferred.
- **CLI (`mlxseg`)** — 0.10.0; consumes this registry.

## [0.9.1] - 2026-05-27

### Added — declarative task registry + `run_named_task` dispatcher

Names tasks. `mlxseg --task lung_vessels ...` (coming in 0.10.0 CLI) needs a registry; here it is.

The schema is informed by both TS (which has three pipeline shapes — single-model, cascade, label-union) and MOOSE's `moosez/constants.py` (which contributed modality-as-first-class, body coverage hints, and weight provisioning slots).

**New module `tasks.py`** with:

- **`TaskSpec`** dataclass — the descriptor:
  ```python
  TaskSpec(
      name: str,
      source: Literal["ts", "moose", "user"],
      modality: Literal["CT", "MR", "PET"],
      shape: Literal["single", "cascade", "label_union"],
      # Exactly one of these, matching `shape`:
      single: int | None = None,
      cascade: tuple[CascadeStep, ...] | None = None,
      union: tuple[UnionPart, ...] | None = None,
      # Informational:
      label_map: dict[int, str] = {},
      expected_coverage: str = "any",       # MOOSE-inspired
      weights_url: str | None = None,        # slot for 0.11.0+ remote weights
      weights_sha256: str | None = None,
  )
  ```
  Validated in `__post_init__`.

- **`CascadeStep`** / **`UnionPart`** — shape-specific sub-types.

- **Registry API**:
  - `register_task(spec, *, overwrite=False)`
  - `get_task(name)`
  - `unregister_task(name)`
  - `list_registered_tasks()`
  - `list_tasks_by_modality(modality)`

- **`run_named_task(name, image_sitk, *, folds=None, reorient_to="LPS", peak_working_memory_mb=None, verbose=False, engine_factory=None)`** — the dispatcher:
  - `shape="single"` → `predict_with_resampling`
  - `shape="cascade"` → `run_workflow` with `Stage` per `CascadeStep`
  - `shape="label_union"` → `run_label_union_workflow` with `ParallelStage` per `UnionPart`
  - `engine_factory` hook allows injecting custom engine construction (tests, non-standard weight locations).

### Added — JSON registry file

`src/nnunet_inference_mlx/data/ts_tasks.json` ships in the wheel. Loaded lazily on first registry access. Schema version 1; currently ships with an empty `tasks: []` — `scripts/refresh_ts_registry.py` (a generator that imports an installed TotalSegmentator distribution and emits this file) is planned for a follow-up patch.

Users can register custom tasks at runtime via `register_task(TaskSpec(...))` regardless of whether the shipped JSON has entries.

### Architecture notes

- The dispatcher is thin glue over 0.9.0's `predict_with_resampling`, `run_workflow`, and `run_label_union_workflow`. The registry adds *naming and persistence*; the dispatch logic itself is one branch per shape.
- The `engine_factory` injection point makes the dispatcher testable without weight files and gives users an extension point for non-standard weight locations (MOOSE-style remote downloads, custom WeightsLayout entries, etc.).
- The JSON storage is reviewable in PRs (vs hand-written Python dict), language-agnostic (other tools can consume it), and decoupled from our Python module — a downstream CLI or evaluator can read `ts_tasks.json` without importing our package.

### Tests

30 new tests in `test_tasks.py`. Total: **212 passing.**

Coverage:
- `TaskSpec` validation: shape↔data field consistency, allow-list enforcement (modality / source / shape), cascade min-length, union min-length, multiple-shape-fields rejection, missing-shape-field rejection
- JSON round-trip for all three shapes; int-key recovery in `label_remap` / `label_map`; default-value omission on disk
- Registry API: register / get / unregister / list / list-by-modality, duplicate-rejection, overwrite-with-explicit-flag, name-collision errors
- Dispatcher: per-shape backend routing, `weights_id` forwarding to factory, factory called once per distinct ID in order, unknown-task error, output geometry preservation
- Builtin registry loadability sanity check

### What's not in this release

- **`scripts/refresh_ts_registry.py`** — the generator that imports an installed TS and writes `ts_tasks.json`. Planned for 0.9.1.1 once verified against a real TS install. Until then, the shipped registry is empty and users register tasks at runtime.
- **CI cron** to auto-refresh against new TS releases — same dependency.
- **MOOSE entries / MOOSE-compat fields** — those are 0.9.2 work.
- **CLI (`mlxseg`)** — still 0.10.0; consumes this registry.

## [0.9.0] - 2026-05-26

### Added — multi-task label-union orchestrator

The TotalSegmentator full-mode pattern as a first-class primitive: run multiple independent task models against the same input, remap each to a shared unified label space, fold by paint priority. Previously buildable as ~25 lines of user code; now a documented, tested primitive.

**`ParallelStage`** dataclass — one task in the union:

```python
@dataclass
class ParallelStage:
    engine: InferenceEngine
    label_remap: dict[int, int]      # task-local ID → unified ID
    part_name: str | None = None     # optional, for logging
```

**`run_label_union_workflow(image, stages, *, reorient_to="LPS", ...)`** — orchestrator:

```python
from nnunet_inference_mlx import (
    cached_engine_from_task, ParallelStage, run_label_union_workflow,
)

stages = [
    ParallelStage(cached_engine_from_task(291), TS_REMAP_ORGANS,    "organs"),
    ParallelStage(cached_engine_from_task(292), TS_REMAP_VERTEBRAE, "vertebrae"),
    ParallelStage(cached_engine_from_task(293), TS_REMAP_CARDIAC,   "cardiac"),
    ParallelStage(cached_engine_from_task(294), TS_REMAP_MUSCLES,   "muscles"),
    ParallelStage(cached_engine_from_task(295), TS_REMAP_RIBS,      "ribs"),
]
seg = run_label_union_workflow(image, stages)
```

Semantics: each stage runs against the same input volume independently; labels are remapped from task-local to unified space; later stages overwrite earlier ones at overlapping voxels (list order = priority — matching the existing `_slab_resample_paint` convention).

Unlike `run_workflow` (sequential cascade with inter-stage cropping), there's no data dependency between stages — the orchestrator only adds a single LPS canonicalization at the boundary and a unified output buffer.

The orchestrator is intentionally thin glue — every step is a public top-level function (see below). Callers building non-standard variants (custom paint priority, logit-confidence merging, persistent intermediates) write the same recipe themselves.

### Added — top-level toolbox primitives

Four new public functions, exposed so the orchestrator recipe is fully decomposable:

- **`get_orientation(image_sitk) -> str`** — read the 3-letter DICOM orientation code (`"LPS"`, `"RAS"`, `"SAR"`, …) of a SITK image. Wraps the longwinded `DICOMOrientImageFilter_GetOrientationFromDirectionCosines`.
- **`reorient(image_sitk, code) -> sitk.Image`** — symmetric primitive that reorients to any DICOM code. No-op (returns the same object) when already in the requested orientation. Used to be inlined inside `predict_with_resampling` / `run_workflow`.
- **`remap_labels(seg, mapping)`** — vectorized LUT remap of integer labels. Source IDs not in the mapping become background. Auto-picks the smallest unsigned-int target dtype.
- **`paint_union(target, source)`** — overwrite `target` with `source` wherever `source != 0`. Same convention as `_slab_resample_paint`: list order is priority.

### Changed — `predict_with_resampling` kwarg rename: `reorient=` → `reorient_to=`

The previous parameter name shadowed the new top-level `reorient` function. Renamed to `reorient_to` (which also reads more naturally: "reorient *to* LPS"). Same default (`"LPS"`), same semantics.

**Migration:** rename any `reorient="LPS"` / `reorient=None` call to `reorient_to=...`.

### Refactor — `predict_with_resampling` and `run_workflow` use the new primitives

Both functions previously had the DICOMOrient logic inlined. They now call `get_orientation` / `reorient` directly, removing duplication. No behavior change.

### Tests

41 new tests across three files. Total: **182 passing.**

- `test_label_primitives.py` (20 tests) — `remap_labels` auto-dtype tiers, unmapped-drops-to-background, negative-ID rejection, shape preservation; `paint_union` overwrite-where-nonzero, zero-source transparency, priority-via-call-order, shape-mismatch error; a composition test that hand-rolls the union recipe (two tasks → remap each → paint into shared unified) to prove the toolbox is usable without the orchestrator
- `test_label_union_workflow.py` (10 tests) — empty-stages error, single-stage remap dispatch, geometry preservation, SAR orientation round-trip (the 0.8.1 chest-CT canary), `reorient_to=None` skip, two-stage paint priority (defining property of the orchestrator), disjoint-stages paint cohabit, output-dtype resolution across stages
- `test_canonical_orientation.py` (5 new tests) — `get_orientation` code detection, `reorient` direction change, no-op when already at target, round-trip geometry preservation

### What's not in this release

- **Declarative `TS_TASKS` registry + `run_named_task` dispatcher.** Originally planned for the same 0.9.0 milestone; split out to keep this release focused on the orchestrator + primitives. Likely lands as 0.9.1.
- **CLI (`mlxseg`)** — still 0.10.0, depends on the task registry.

## [0.8.2] - 2026-05-24

### Fixed — `transpose_forward` / `transpose_backward` handling

nnU-Net's training pipeline permutes input volume axes by `transpose_forward` from `plans.json` before patches reach the network. The model has only ever seen volumes in that transposed axis order. For models where `transpose_forward != (0, 1, 2)`, feeding the engine canonical-order volumes was producing silently-wrong predictions — same class of bug as the orientation issue in 0.8.1, just at a different axis-mapping layer.

For TS Datasets 291–298 these are all identity, so this fix has zero effect on TS workloads. For research nnU-Net models with non-identity transposes (e.g. `(2, 0, 1)` for axially-acquired thoracic models), this closes a real correctness hole.

### Added — bundle transpose properties

- **`ModelBundle.transpose_forward`** — read from `plans.json`, defaults to `(0, 1, 2)` when absent.
- **`ModelBundle.transpose_backward`** — same.
- **`ModelBundle.target_spacing`** now applies `transpose_backward` to the raw plans spacing, returning canonical-order spacing (matching MOOSE's convention). For identity-transpose models the behavior is unchanged.

### Changed — engine round-trip

`InferenceEngine.predict()` / `predict_logits()` / `predict_segmentation()` now:

1. Apply `transpose_forward` to the input volume (numpy `np.transpose`) before the sliding-window backend sees it
2. Run inference in the model's expected axis order
3. Apply `transpose_backward` to the output predictions (spatial axes; K dimension preserved at axis 0)

Caller-facing axis order is canonical `(Z, Y, X)` everywhere — both for inputs and for outputs. For identity-transpose models the round-trip is two no-ops; for non-identity models it makes the engine "transparent" so the caller doesn't need to know about the internal permutation.

`SlidingWindowEngine`, `Predictor`, and `FoldEnsemble` are unchanged — they remain raw primitives operating on whatever axis order they're given. The transpose handling lives in the `InferenceEngine` facade, matching the architectural rule: primitives do one thing, facades compose with metadata-awareness.

### Tests

14 new tests in `test_transpose_handling.py`. Total: **167 passing.**

Notable coverage:

- `test_target_spacing_non_identity_transpose` — verifies `bundle.target_spacing` returns canonical-order spacing
- `test_non_identity_round_trip_preserves_canonical_layout` — runs the engine with five different transpose pairs (identity, swaps, rotations, full reverse) on an asymmetric `(20, 24, 28)` input and verifies output shape is always `(K, 20, 24, 28)` — the engine is transparent regardless of the model's internal axis convention
- `test_apply_transpose_backward_undoes_forward` — tests the spatial-axis-only K-channel transpose helper directly

### Migration

For identity-transpose models (every TS model): no behavior change.

For non-identity-transpose models: existing callers passing canonical-order volumes get the right answer for the first time. Callers who were manually pre-transposing inputs to compensate for the bug should remove that compensation.

## [0.8.1] - 2026-05-24

### Fixed — canonical orientation handling in `predict_with_resampling` and `run_workflow`

Inputs with non-canonical direction matrices (oblique / reformatted scans where voxel axes don't align with anatomical axes) were silently producing badly fragmented segmentations: the sliding window scans along numpy axes assuming they map to canonical anatomical directions, but for a volume where, e.g., voxel-X = patient-superior, vertebrae stretched along the patient's cranio-caudal direction land as thin needles perpendicular to the sliding-window primary axis. Most patches saw only fragments.

Reproduced on a chest CT with SAR orientation (voxel X→S, Y→A, Z→R, spacing 1.0×0.65×0.65 mm). Before the fix:

| Task | FG voxels |
|---|---|
| Dataset291 (organs) | 16.3 M |
| **Dataset292 (vertebrae)** | **18.5 k** (almost entirely missed) |
| Dataset293 (cardiac) | 187 k |
| Dataset294 (muscles) | 5.46 M |
| Dataset295 (ribs) | 215 k |
| **Total FG** | **22.0 M (58 classes)** |

After the fix:

| Task | FG voxels |
|---|---|
| Dataset291 (organs) | 25.5 M (1.6×) |
| **Dataset292 (vertebrae)** | **2.30 M (124×)** |
| Dataset293 (cardiac) | 2.38 M (12.7×) |
| Dataset294 (muscles) | 11.3 M (2.1×) |
| Dataset295 (ribs) | 1.43 M (6.7×) |
| **Total FG** | **42.9 M (111 classes)** |

Inference itself ~26% faster aggregate — patches now contain anatomically coherent context so the model spends less compute on garbage paths.

### Changed — `predict_with_resampling` API

New keyword argument `reorient: str | None = "LPS"`. The function now:

1. Reorients the input image to canonical LPS via `sitk.DICOMOrient` before forward resample (if not already canonical),
2. Runs the full pipeline (forward resample → inference → inverse resample) in canonical orientation,
3. Reorients the output segmentation back to the caller's original orientation before returning.

Pass `reorient=None` to skip the round-trip when the caller knows the input is already canonical and wants to save the ~3–5 s reorient cost on huge volumes. Pass `reorient="RAS"` or any other DICOM-style orientation code to target a different canonical orientation.

For axis-aligned inputs (the typical case — clinical CTs in LPS, neuroimaging volumes in RAS) the reorient is a no-op and there's no behavior change. The fix only affects volumes with non-identity direction matrices.

### Changed — `run_workflow` reorient at workflow boundary

The orchestrator now reorients to LPS once at the workflow entry and reorients the final output back to the caller's orientation at exit. Each `Stage`'s internal `predict_with_resampling` call is told `reorient=None` (saving the per-stage reorient cost) since the workflow has already done it. Inter-stage crop bboxes are computed in canonical-orientation voxel coordinates, then composed in the same space.

### Tests

7 new orientation tests in `test_canonical_orientation.py`. Total: **153 passing**.

Notable: `test_sar_input_returns_sar_output` and `test_sar_cascade` explicitly verify that an input with SAR direction matrix produces output with the same SAR direction matrix after the LPS round-trip — geometry preserved through both the single-stage and cascade paths.

### Migration

Existing callers of `predict_with_resampling(engine, image)` and `run_workflow(image, stages)` need no changes — the reorient defaults to on. Callers who were already manually canonicalizing inputs can pass `reorient=None` to opt out.

## [0.8.0] - 2026-05-23

Small, single-purpose primitive additions that complete the mix-and-match toolkit story. No god-methods, no consolidation. Each addition does one thing and composes with what's already there.

### Added — engine bundle property accessors

Four read-only properties on `InferenceEngine` that surface metadata previously buried in `engine._bundle`:

- **`engine.target_spacing`** — `(Z, Y, X)` voxel spacing in mm the model expects as input. Callers resample to this spacing before predict.
- **`engine.has_regions`** — `True` for BraTS-style models with independent sigmoid heads, `False` for standard mutually-exclusive classes.
- **`engine.regions_class_order`** — paint-priority tuple of label values for region-based conversion. Empty for standard datasets.
- **`engine.bundle`** — the underlying `ModelBundle` (escape hatch for everything else in `dataset` / `plans` / `metadata`).

Pure delegations to `engine._bundle`. The curated names cover the routine cases; `engine.bundle` is for the rare reach-through.

### Added — `engine.predict_logits()` returning `mx.array`

The MLX-native sibling of `engine.predict()`. Returns the per-channel predictions as an `mx.array` in unified memory, ready to feed directly into `inverse_resample_argmax` / `inverse_resample_paint` / multi-model arithmetic without a per-caller `mx.array(...)` wrap. Same underlying data as `predict()`; same per-volume materialization cost (the sliding-window accumulator runs in numpy internally). The win is API ergonomics — the `mx.array(...)` ceremony moves once into the engine rather than every call site.

### Added — `inverse_resample_paint` (region-based scheme-aware inverse resample)

New primitive paired with `inverse_resample_argmax`. Same slab-streaming + K-channel trilinear-gather pass; differs only in the per-slab finishing step: per-region threshold (default 0.0 ↔ raw-logit "sigmoid > 0.5") + paint-priority overwrite from `regions_class_order`. Use for BraTS-style models where argmax across channels is silently wrong (it picks "the region with the highest sigmoid" instead of "all regions above threshold, painted by priority").

```python
seg = inverse_resample_paint(
    logits_target,
    out_shape_zyx=acq_shape,
    target_spacing_zyx=target_spacing,
    acq_spacing_zyx=acq_spacing,
    regions_class_order=engine.regions_class_order,
    threshold=0.0,   # 0.5 if input is post-sigmoid
)
```

Memory shape identical to `inverse_resample_argmax` (slab budget bounds the K-channel acquisition-spacing slab). Output dtype auto-picks from the maximum label value in `regions_class_order` — typically `uint8`.

### Added — `inverse_resample_argmax` accepts `mx.array | np.ndarray`

Polymorphic input. The caller-side `mx.array(numpy_logits)` wrap is internalized; the function accepts whatever logit-shaped tensor you have. Same one-time conversion cost when called with numpy, zero conversion when called with `mx.array`.

### Added — `resample_volume` (numpy forward resample primitive)

Pure-scipy sibling of `resample_image_to_target` (which takes a `sitk.Image`). For users composing numpy-native mix-and-match pipelines without an SITK dependency:

```python
vol_target = resample_volume(vol_acq, in_spacing_zyx=acq, out_spacing_zyx=target, order=3)
```

`order=3` (cubic) matches nnU-Net's training-time forward resample quality; pass `order=0` for label volumes. Adds an optional scipy dependency that's already commonly installed.

### Fixed — `predict_with_resampling` scheme dispatch for region-based models

Previously called `inverse_resample_argmax` unconditionally. For region-based models, the sigmoid-averaged ensemble output would get argmax'd — picking "highest sigmoid channel" instead of doing proper threshold + paint priority. Silently wrong at ambiguous voxels.

Now dispatches on `engine.has_regions`:

- Standard scheme → `inverse_resample_argmax`
- Region-based scheme → `inverse_resample_paint` with `regions_class_order=engine.regions_class_order` and `threshold=0.5` (multi-fold, post-sigmoid mean) or `threshold=0.0` (single-fold, raw logits)

The non-resample path (`engine.predict_segmentation`) already handled this correctly via `convert_logits_to_segmentation`; this brings the resample path into alignment.

### Tests

38 new pytest tests across three new files (`test_predict_logits_and_properties.py`, `test_resample_primitives.py`, `test_predict_with_resampling_scheme.py`), bringing total coverage to 146 tests, all passing.

Notable: `test_predict_with_resampling_scheme.py::test_region_based_dispatch_uses_paint` explicitly verifies the fixed BraTS bug — output labels must come from `regions_class_order` (`{0, 1, 2, 4}` in the test fixture), not channel indices (`{0, 1, 2, 3}` which would indicate argmax was incorrectly applied).

### Migration

`predict_with_resampling` callers don't need to change anything — the scheme dispatch is internal. Direct callers of `inverse_resample_argmax` who were passing region-based logits should switch to `inverse_resample_paint`.

Internal use of `engine._bundle.target_spacing` should migrate to `engine.target_spacing` (the underscore-prefixed access still works but is no longer the recommended pattern).

## [0.7.0] - 2026-05-23

### Added — pluggable weights-folder discovery (`WeightsLayout` + registry)

Different consumers of nnU-Net weights store them in slightly different places (vanilla nnU-Net under `$nnUNet_results`, TotalSegmentator under `$TOTALSEG_WEIGHTS_PATH` or `~/.totalsegmentator/nnunet/results`, MOOSE elsewhere). The `WeightsLayout` dataclass captures a layout as data — env var, default path, preferred trainer/plans/model naming — and a module-level registry walks layouts in order. Two built-in layouts cover nnU-Net and TS; downstream packages register their own.

- **`WeightsLayout`** — frozen dataclass: `name`, `env_var`, `default_path`, `trainer`, `plans`, `model`. `resolve_weights_dir()` returns the dir if the env var points at one, falls back to `default_path` if that exists, returns `None` otherwise.
- **`register_weights_layout(layout, *, prepend=False)`** — add a layout. Append by default; pass `prepend=True` to put it ahead of the built-ins.
- **`list_weights_layouts()`** — current lookup order.
- **`discover_weights()`** — first matching layout wins, returns `(path, layout)`. Raises `FileNotFoundError` with a diagnostic listing of what was tried when nothing resolves.

`ModelBundle.from_task()` now walks the registry when `weights_dir=None` and inherits the resolved layout's trainer preference automatically. New keyword args `trainer` / `plans` / `model` disambiguate Datasets that ship multiple trainer variants — needed for TS's `Dataset291` (both `nnUNetTrainer` and `nnUNetTrainerNoMirroring` under one Dataset folder).

### Added — process-wide engine cache (`engine_cache.py`)

Building an `InferenceEngine` from a model folder takes ~3–5 s on M2 base (read disk + build network + load weights + compile + warmup). For workflows that touch the same model many times — batch inference, multi-stage cascade, interactive UI — keeping the engine alive across calls eliminates that cost.

- **`cached_engine_from_folder(model_folder, *, configuration, folds, step_size, compile, batch_size, use_mirroring, ...)`** — high-level helper: build-and-cache or return-cached, keyed on everything that affects engine state.
- **`cached_engine_from_task(task_id, *, folds, trainer, plans, model, ...)`** — same, but resolves the folder via the `WeightsLayout` registry first. Drop-in replacement for TS-style `find_model_folder` + build-engine code.
- **`get_cached_engine(key)`** / **`cache_engine(key, engine)`** — low-level for callers managing custom keys (e.g. nnInteractive's session state).
- **`cache_enabled()`** — auto-tiers by detected unified memory (≥32 GB on by default; below disabled to avoid memory pressure from holding ~600 MB per cached engine). Override via the `NNUNET_MLX_CACHE_ENGINES` env var.
- **`clear_engine_cache()`** — release everything and clear Metal buffers; safe on empty cache.

Measured on TS Dataset297 fast task:
- Cold load: ~4.5 s
- Cache hit: ~3 ms (**~1500× speedup**)

Verbose / progress flags don't bust the cache; `step_size`, `use_mirroring`, `batch_size`, fold list, configuration, and folder all do.

### Added — multi-stage workflow orchestrator (`workflow.py`)

The MOOSE-style cascade pattern (low-res body detector → high-res organ model) and the generalization of TS's FOV-limited inference, as a first-class primitive. Each stage runs `predict_with_resampling`; between stages, the prior stage's output bbox optionally crops the next stage's input.

- **`Bbox`** — frozen dataclass for `(Z, Y, X)` voxel-coordinate boxes. `shape_zyx` / `slices` / `clamped()` / `dilated()` / `compose()` / `Bbox.full()`.
- **`compute_fg_bbox(labels, *, classes=None, dilation_mm=0, spacing_zyx=None)`** — find FG bbox of a label volume, optionally restricted to specific classes and dilated by a physical margin. Returns `None` when no FG is found, signaling "skip cropping" to workflow callers.
- **`crop_image(sitk_image, bbox)`** — extract a sub-volume preserving world-coordinate geometry (origin shifts to track the crop).
- **`paste_segmentation(small_seg, full_shape_zyx, bbox, *, fill=0)`** — paste a cropped-space label volume back into a full-shape canvas.
- **`Stage`** — `engine + crop_to_classes + dilation_mm + interpolation + peak_working_memory_mb + remove_small_components_mm3`. Default dilation is 10 mm.
- **`run_workflow(image_sitk, stages, *, verbose=False)`** — chain stages, crop input between stages where requested, paste-back at the end so output geometry matches input.

Two-stage cascade in user code:

```python
from nnunet_inference_mlx import Stage, run_workflow, cached_engine_from_task

body  = cached_engine_from_task(298, folds=0)              # low-res body detector
liver = cached_engine_from_task(291, folds=0,              # high-res organ model
                                trainer="nnUNetTrainerNoMirroring")

stages = [
    Stage(engine=body,  crop_to_classes=(BODY_TRUNK,), dilation_mm=10.0),
    Stage(engine=liver, remove_small_components_mm3=200.0),
]

seg = run_workflow(image_sitk, stages, verbose=True)
```

Composes naturally with the engine cache — each `Stage.engine` survives across workflow invocations. nnInteractive-style sub-volume re-runs use the same crop/paste primitives at finer granularity. The geometric primitives are exported as public API so callers building bespoke pipelines (FOV-limited inference, manual region updates) use the same building blocks.

### Fixed

- `compute_fg_bbox`: previously cast user-supplied class IDs to the labels array's dtype, which raised `OverflowError` for IDs outside the dtype's range (asking about class 999 on a `uint8` label volume crashed). Now filters out-of-range values up front and returns `None` when no in-range classes remain.

### Tests

97 new pytest tests across five new files (`test_weights_layout_registry.py`, `test_engine_cache.py`, `test_workflow.py`, `test_postprocessing.py`, `test_resampling.py`), all passing in ~3.7 s with no real weights or CT data required. Backfills coverage for the 0.6.0 resampling + postprocessing modules as well as the new 0.7.0 modules.

## [0.6.0] - 2026-05-23

### Added — spacing-aware resampling subsystem (`resampling.py`)

New module covering the forward + inverse resampling pieces that nnU-Net inference needs around model space. SimpleITK on the way in (CPU, B-spline/linear/nearest), MLX on the way out (Metal, K-channel trilinear with slab streaming). All resampling code is opt-in via the new `[preprocessing]` extra so the core package stays SimpleITK-free.

- **`resample_image_to_target(sitk_image, target_spacing_zyx, *, interpolation='linear')`** — forward resample an acquisition-spacing image to model target spacing via SITK. `interpolation` accepts `'linear'`, `'bspline'`, or `'nearest'`. Preserves geometry; output has the correct origin/spacing/direction.

- **`inverse_resample_argmax(logits_target, out_shape_zyx, target_spacing_zyx, acq_spacing_zyx, *, out_dtype=np.uint8, peak_working_memory_mb=None, cascade_downsample=False, verbose=False)`** — resample target-spacing K-channel logits to acquisition-spacing labels (path B: continuous-logit interpolation + argmax). Single Z-slab loop with explicit-coordinate trilinear over all K channels; slab depth auto-sized from `peak_working_memory_mb` so the K-channel slab fits the budget. When the whole output fits, equivalent to a one-shot materialize.

  `peak_working_memory_mb=None` (default) auto-detects from system RAM: 200 MB on `<32 GB` Macs, 2000 MB on `≥32 GB`.

  `cascade_downsample` (default `False`) enables a multi-step path for aggressive downsampling (source-ratio > 2× in any axis). Each step does a 2× K-channel downsample, preserving the continuous decision surface; final pass slabs to acquisition spacing. Trades more smoothing for less boundary aliasing; **not** a strict win on aggressive downsamples — small-vessel structures can vanish more than single-step preserves them. Documented with the trade-off in the docstring.

- **`predict_with_resampling(engine, image_sitk, *, interpolation='linear', peak_working_memory_mb=None, remove_small_components_mm3=0.0)`** — full path-B pipeline. Caller hands over a SITK image at any acquisition spacing, gets back a SITK image at the same spacing with integer labels. Forward resample on CPU/SITK, inference on Metal, inverse resample on Metal with slab+channel streaming, optional cc3d cleanup. K-channel logits are transient — never materialized at acquisition spacing.

  ```python
  import SimpleITK as sitk
  from nnunet_inference_mlx import InferenceEngine, predict_with_resampling

  img = sitk.ReadImage("scan.nii.gz")
  seg = predict_with_resampling(engine, img)              # default path B
  seg = predict_with_resampling(engine, img,
                                remove_small_components_mm3=200.0)  # + cleanup
  ```

Why path B: nnU-Net's standard inverse resampling is either nearest-neighbor on labels (aliased boundaries) or per-class one-hot resize (memory-prohibitive at K≥100). Trilinear interpolation of the K-channel logits with argmax-at-the-end gives smoother boundaries while remaining feasible on Apple Silicon — we slab-stream the K-channel inverse so the working set is bounded by a tunable budget (default ~200 MB on 16 GB Macs, 2 GB on 32 GB+). On unified memory the logits stay resident across the full pipeline without disk round-trips.

### Added — multi-label connected-component postprocessing (`postprocessing.py`)

New module exposing `remove_small_components` for dropping label islands below a physical-volume threshold. Backed by [cc3d](https://github.com/seung-lab/connected-components-3d) for the multi-label CC pass — two neighboring voxels are connected iff they share the same nonzero label, so disconnected pieces of the same class are filtered independently. Original label IDs are preserved.

- **`remove_small_components(labels, spacing_zyx, *, min_volume_mm3=200.0, connectivity=26, in_place=False)`** — drop components smaller than `min_volume_mm3`. Default threshold matches TotalSegmentator's `--remove_small_blobs` flag. Pass `0` for a no-op. Opt-in via the new `[postprocessing]` extra.

Measured on a 256×178×255 K=117 CT segmentation (`min_volume_mm3=665`):

| Method | Time | Speedup |
|---|---|---|
| `cc3d.dust` | **80 ms** | baseline |
| SITK `ScalarConnectedComponent + RelabelComponent + Mask` | 820 ms | 10× slower |
| scipy per-label loop (TS-style) | 7800 ms | 90× slower |

Wired into `predict_with_resampling` as a new `remove_small_components_mm3=0.0` keyword arg (off by default; pass `200.0` for TS-equivalent cleanup).

### Added — optional extras

- `[preprocessing]` = `["SimpleITK"]` — required for `resample_image_to_target`, `predict_with_resampling`, and any SITK-image I/O path.
- `[postprocessing]` = `["connected-components-3d"]` — required for `remove_small_components`.

Both raise a clear `ImportError` pointing users at the right `pip install` invocation when the underlying package is missing.

### Notes

- Path B inverse throughput is dominated by MLX's native trilinear+gather kernels (~11 ns / voxel-K on M2 base, ~3 ns / voxel-K on M1 Max). `peak_working_memory_mb` is a memory-bound knob, not a perf knob — slab size affects what fits, not how fast it runs.
- Slab boundaries are continuous: explicit-coordinate trilinear ensures interp values across slab borders match the values that would come from a single materialize pass. Earlier per-slab `mx.nn.Upsample` formulation had visible staircase artifacts in non-axial views; the current explicit-coordinate path is artifact-free.
- Output dtype on `predict_with_resampling` follows the bundle's label scheme: `uint8` for standard `K ≤ 255`, auto-widened for region-based / large-K datasets via `label_dtype`.

## [0.5.3] - 2026-05-22

### Changed — auto-tier Metal cache by detected unified memory

`Predictor` now accepts `cache_limit_fraction=None` (the new default) and auto-picks based on detected unified memory:

- **< 32 GB RAM**: 0.30 (unchanged from previous behavior; leaves room for sliding-window accumulators on constrained Macs).
- **≥ 32 GB RAM**: 0.50 (M1 Max / M3 Pro / Studio / Ultra). Maps to ~30 GB+ Metal cache on a 64 GB Mac — well under Apple's recommended GPU working-set size, but enough to keep compiled-graph buffers resident between forward passes instead of evicting and rebuilding for every patch.

Effect: small but consistent latency win on batch / multi-file workloads on big-RAM Macs. No change on 16 GB Macs. `self.cache_limit_fraction` is now a `Predictor` attribute callers can introspect to confirm what auto-detection picked. Explicit fractions still work as a hard override.

Why now: downstream consumers (TotalSegmentator's MLX backend in particular) want to ship engine-caching workflows that hold ≥5 InferenceEngines resident on big-RAM Macs. The previous 0.30 default sized the Metal cache for the constrained case and left ~50 GB unused on 64 GB systems, forcing more buffer churn than necessary in those flows.

## [0.5.2] - 2026-05-22

### Added — region-based label handling (BraTS-style models)

New `labels.py` module brings the LabelManager-equivalent post-processing piece that was the last gap vs `nnUNetPredictor`. Closes a silent-correctness hole: a BraTS-style checkpoint loaded into 0.5.1 and consumed via `np.argmax(engine.predict(vol), axis=0)` would have produced nonsense labels because the model's region heads are independent sigmoids, not a softmax distribution.

- **`labels.has_regions(dataset_json)`** — True if any label value is a list/tuple of underlying classes (the union form).
- **`labels.regions_class_order(dataset_json)`** — paint-priority tuple, empty for standard datasets.
- **`labels.label_dtype(dataset_json)`** — smallest unsigned integer dtype that fits every label value across both region member-class lists and `regions_class_order` output values. Returns `uint8` / `uint16` / `uint32` per nnUNetv2's convention.
- **`labels.convert_logits_to_segmentation(pred, dataset_json, threshold=0.0, dtype=None)`** — uniform post-processing. Standard datasets get `argmax`; region-based get threshold + paint-priority overwrite. Auto-detects scheme from `dataset_json`. `dtype=None` (default) auto-picks via `label_dtype`; pass explicit dtype to override.
- **`labels.sigmoid_inplace`** — float32-safe in-place sigmoid (matches `softmax_inplace` in style).

`ModelBundle` gains two derived properties:

- **`bundle.has_regions`** — True for BraTS-style datasets.
- **`bundle.regions_class_order`** — the paint-priority tuple.

### Added — `InferenceEngine.predict_segmentation()`

The one-call high-level path that returns integer labels directly. Handles all four (scheme × fold-count) combinations:

| Scheme | Single fold | Multi fold |
|---|---|---|
| Standard | logits → argmax | softmax-avg → argmax |
| Region-based | per-region logits → threshold + paint | sigmoid-avg → threshold + paint |

```python
seg = engine.predict_segmentation(volume)        # auto dtype
seg = engine.predict_segmentation(volume, dtype="uint16")  # explicit dtype
```

Auto-routes through the right scheme; auto-picks output dtype (the smallest unsigned int that fits every label value). Pass an explicit dtype to force a specific integer width regardless of what the dataset needs.

Also exposed: **`engine.label_dtype`** read-only property for introspecting what auto-detection would pick.

### Fixed — `Predictor.num_classes` for region-based datasets

Previously `Predictor.num_classes = len(dataset["labels"])` — which counts background plus all regions/labels indiscriminately. For region-based datasets this gave the wrong network output channel count (e.g. 4 instead of 3 for BraTS, where background is implicit). Now correctly counts foreground regions (`sum(k != "background")`) for region-based datasets while preserving the standard `len(labels)` rule for softmax datasets. Also handles bare-int size-1 regions like `"ET": 3` correctly — they count as a head, same as `"ET": [3]`.

### Changed — `FoldEnsemble` averaging

The fold-ensemble averaging branches on label scheme:

- **Standard** models (heads are softmax-related): softmax-then-average — unchanged.
- **Region-based** models (heads are independent sigmoids): sigmoid-then-average. Auto-set from `bundle.has_regions` when the facade builds the ensemble; can be forced via `FoldEnsemble(..., region_based=True)` for direct callers.

### Changed — `predict_nifti` default dtype

The `dtype` parameter on `nnunet_inference_mlx.io.predict_nifti` now defaults to `None` (auto-detect from dataset) instead of hardcoded `np.uint8`. Pass an explicit dtype to force one. Also internally routes through `engine.predict_segmentation` instead of a raw `np.argmax`, so region-based models produce correct labels through the NIfTI helper too.

### Migration

Purely additive; no API removals. Code calling `engine.predict()` + `np.argmax` continues to work unchanged for standard models. Code using region-based models that was silently broken in 0.5.1 (no public consumer that we know of) now works correctly via `engine.predict_segmentation()`.

## [0.5.1] - 2026-05-22

### Added — nnInteractive enabler set

Two small additive changes that close the last gaps for a third-party port (notably nnInteractive) to use `nnunet-inference-mlx` as its core instead of forking `model.py` / `weights.py`.

- **Nested `cfg["architecture"]` plans block.** `build_network_from_plans` now accepts the form
  ```json
  "architecture": {
    "network_class_name": "...ResidualEncoderUNet",
    "arch_kwargs":        {...}
  }
  ```
  emitted by modern nnUNetv2 trainers via the `dynamic_network_architectures` configuration manager. The flat `network_arch_init_kwargs` form (TS Dataset29x plans) still works. Lookup order: flat-top-level → flat-in-config → nested-in-config → "old plans" fallback. Pure additive, no behavior change on existing models.
- **`dtype=` weight cast on load.** `load_model_weights`, `load_checkpoint_with_metadata`, and `ModelBundle.from_folder` / `from_task` gain a `dtype` parameter. Pass `"float16"`, `"bfloat16"`, `"float32"`, or an `mx.Dtype` to cast on load. Default `None` preserves source precision. Verified 2.00× memory reduction on Dataset291 (125 MB → 62 MB). Callers using fp16 weights still need matching activation precision in the forward — this is a weight-cast convenience, not a full mixed-precision pipeline.

### Changed — internal cleanup

- **`predict_sliding_window_streaming` dropped** in favor of the simpler `predict_sliding_window` kernel. The streaming variant's rolling-Z accumulator was optimizing peak memory that isn't where TS's pressure actually sits (the 5-models-in-one-process Metal cache, addressed separately by `engine.close()`). `SlidingWindowEngine.predict` now calls the non-streaming kernel; ~231 lines removed from `inference.py`. Test suite runs ~15% faster on the same volumes. No public-API impact — the variant was never exported.

## [0.5.0] - 2026-05-22

### Added — layered inference architecture

The monolithic `InferenceEngine` is decomposed into four composable layers so downstream consumers (TotalSegmentator, MOOSE, nnInteractive, future ports) can pick what they need instead of inheriting concerns they don't.

- **`Predictor`** (Layer 2) — one compiled, weight-swappable MLX network. Owns `mx.compile`, warmup, Metal cache discipline. Exposes `forward(x)`, `reload_weights(w)`, public `.network`. Knows nothing about sliding windows, ensembling, or normalization.
- **`SlidingWindowEngine`** (Layer 3) — wraps a `Predictor` and adds Gaussian importance weighting, the streaming accumulator, the shape cache, and per-channel normalization. The whole-volume `(Z, Y, X) → (K, Z, Y, X)` path.
- **`FoldEnsemble`** (Layer 4) — orthogonal composable. Wraps either a `Predictor` (patch-level ensemble) or a `SlidingWindowEngine` (volume-level ensemble). Loops the bundle's fold weight dicts via `Predictor.reload_weights` between forwards and averages softmax. Single-fold wrap is a no-op cost.
- **`InferenceEngine`** survives as a thin back-compat facade — given a bundle, builds `Predictor → SlidingWindowEngine`, wraps in `FoldEnsemble` automatically when `len(bundle.fold_weights) > 1`. The 90% caller stays one line.

### Added — multi-fold bundle loading

- **`ModelBundle.from_folder(path, folds=…)`** and the same kwarg on `from_task`. Single union-typed parameter:
  - `folds=int` — single fold
  - `folds=Iterable[int]` — multi-fold ensemble
  - `folds="all"` (default) — auto-detect every `fold_*/` subdir
  
  The `"all"` default loads whatever is on disk, which works for single-fold release builds (e.g. TotalSegmentator) and multi-fold trained models (e.g. MOOSE) without the caller knowing upfront.
- **`ModelBundle.metadata: dict`** — first fold's non-weights checkpoint metadata (`init_args`, `trainer_name`, `inference_allowed_mirroring_axes`, …) captured during load. Used by `Predictor` to auto-detect the configuration name from `init_args["configuration"]` when the caller doesn't pass `configuration=`.
- **`ModelBundle.fold_ids: tuple[int, ...]`** — which folds were loaded, in order.
- **`load_checkpoint_with_metadata(folder, fold)`** in `weights.py` — returns `(mlx_weights, metadata)` for a single fold.
- **`discover_folds(folder)`** — list of int fold IDs present as `fold_*/` subdirs.

### Added — TTA mirroring

- **`use_mirroring: bool = False`** kwarg on `SlidingWindowEngine` and `InferenceEngine`. When enabled, flip-averages predictions along every spatial axis the trained model allows. Mirror axes auto-read from `bundle.metadata["inference_allowed_mirroring_axes"]`; if the model wasn't trained with mirroring (e.g. TotalSegmentator's `NoMirroring` variants), `use_mirroring=True` is silently a no-op. Default `False` for back-compat and because mirroring doubles cost per axis (3 spatial axes → 8 forwards per patch).
- **`ModelBundle.mirroring_axes`** read-only property — `tuple(metadata["inference_allowed_mirroring_axes"] or ())`. Removes a magic dict-key from consumer code.

This closes a latent gap on the MLX path: upstream `nnUNetPredictor` defaults TTA on when the model allows it, but TS's `mlx_predict.py` silently dropped its `tta=` kwarg. MOOSE relies on the same upstream default. Consumers can now opt in by passing `use_mirroring=True` (and should, on models trained with mirroring axes set).

### Added — `Predictor` parameters for future consumers

- **`num_input_channels=None`** override on `Predictor` and `InferenceEngine`. When `None`, derives from `dataset["channel_names"]` as today; nnInteractive passes `8` explicitly (1 image + 7 interaction channels).
- **`configuration=None`** auto-detect — order is explicit arg → `metadata["init_args"]["configuration"]` → the only configuration in plans → `"3d_fullres"` fallback. Lets nnInteractive's non-default `"3d_fullres_ps192_bs24"` config work without explicit caller knowledge.
- **`cache_limit_fraction=0.3`** parameter on `Predictor` — was previously hard-coded.

### Changed — breaking API

- **`ModelBundle.weights` → `ModelBundle.fold_weights`**. Singular dict becomes a list of dicts (length 1 for single-fold bundles). Known consumers (this package, TotalSegmentator's `mlx_predict.py`) never read the attribute directly, so impact is limited to anyone constructing a bundle by hand. Test fixtures updated.
- **`InferenceEngine.predict()` return type depends on bundle shape**: single-fold → raw logits (unchanged behavior); multi-fold → softmax-averaged probabilities (the standard nnU-Net ensemble convention). `np.argmax(axis=0)` on either output yields the segmentation, so argmax-only callers don't need to branch.
- **`InferenceEngine.__init__(configuration=...)`** default changed from `"3d_fullres"` to `None` (auto-detect). Explicit pass-through still works; `None` now reads the configuration from the checkpoint's `init_args`.

### Migration

- Callers of `ModelBundle.from_folder(path, fold=N)` must rewrite as `folds=N`. The `fold=` keyword no longer exists.
- The default behavior changed: `from_folder(path)` with no fold args now loads *all* available folds, not just fold 0. For TotalSegmentator-style single-fold release builds this is a no-op (there's only one fold to load). For multi-fold trained models the engine now auto-ensembles. Pass `folds=0` explicitly if you want fold-0-only on a multi-fold model.
- Code constructing `ModelBundle(plans, dataset, weights=...)` directly must pass `fold_weights=[weights]` instead. Engine/facade users (the 90% case) need no changes.
- Code that wants raw single-patch forward without sliding-window scaffolding (e.g. nnInteractive-style) should build `Predictor` directly: `Predictor(bundle).forward(patch)`. Skips Layers 3-4.

## [0.4.0] - 2026-05-22

### Added
- Vendored torch-free `.pth` loader at `src/nnunet_inference_mlx/_torchfree/` (`torchfree_load.py`, `rangefile.py`). Reads zip-format PyTorch checkpoints (>= 1.6) into numpy via a restricted unpickler — only storage rebuild symbols, `OrderedDict`, and numpy reconstructors are allowed. Materializes only the `network_weights` subtree, so optimizer state and other large storages are never read.
- `load_pth_url` and `smart_load_url` for HTTP-range remote loading of `.pth` checkpoints. Opt-in via the new `remote` extra (`pip install nnunet-inference-mlx[remote]` → `requests`, `remotezip`).
- NIfTI I/O helpers: `load_nifti_zyx`, `save_segmentation_zyx`, `predict_nifti`, `predict_folder`.
- `InferenceEngine.close()` plus context-manager / `__del__` support. Clears the MLX Metal cache between runs — needed for TotalSegmentator's full mode, which runs inference 5× in one process and previously OOM'd without explicit buffer release.

### Changed
- `load_model_weights` now reads `<fold_dir>/<checkpoint_name>` directly via the vendored loader. TotalSegmentator release `.pth` files load with no conversion step and no PyTorch import.
- Runtime dependencies trimmed to `mlx>=0.25`, `numpy`, `tqdm`. `safetensors` is no longer required.

### Removed
- **Breaking**: the entire safetensors load/convert pipeline. Removed public symbols `load_weights_safetensors`, `convert_pth_to_safetensors`, `convert_model_folder`, and the `WEIGHT_LAYOUT_TORCH` constant.
- **Breaking**: `convert_weights_cli` and the `nnunet-inference-mlx-convert` console script entry in `[project.scripts]`.
- **Breaking**: `ModelBundle.from_task(auto_convert=...)` parameter (auto-conversion no longer exists).
- `safetensors` from required dependencies.
- `convert = ["torch"]` optional extra (no path through the package needs torch anymore).

### Migration
- The typical inference workflow (`predict_nifti`, `InferenceEngine`, `ModelBundle.from_task(task_id)`) is unchanged — `.pth` files in `~/.totalsegmentator/nnunet/results/...` load directly.
- Code that imported any of the removed symbols, called the CLI, or passed `auto_convert=` must be updated. There is no migration shim.
- Users who deleted `.pth` files to keep only `.safetensors` siblings need to re-fetch: `rm -rf <model_folder> && totalseg_download_weights -t <task>`.

### Performance
- Cold `.pth` load (Dataset291 fold_0, M2 17 GB): ~162 ms — *faster* than the previous safetensors path's ~848 ms, because the torchfree reader skips the optimizer-state subtree (reads ~125 MB of network_weights, not the full 354 MB safetensors file).
- Vendored loader micro-optimization: removed a redundant `.copy()` in `_LazyStorage.array` that previously memcopied every storage. Downstream consumers (`mx.array`, `np.moveaxis + ascontiguousarray`) copy into their own buffers, so the upstream copy was wasted. ~12 ms saved per fold on warm cache; arrays returned from `load_pth` are now read-only views into the underlying zip bytes.

## [0.3.1] - 2026-04-07

### Added
- `progress: bool = False` parameter on `InferenceEngine` and on the three sliding-window functions (`predict_sliding_window`, `predict_sliding_window_streaming`, `predict_sliding_window_segmentation`). When `True`, a tqdm progress bar is shown for each patch processed during inference. Mirrors the equivalent bar that nnUNetPredictor shows on the PyTorch path.
- `tqdm` is now a runtime dependency (small, pure Python).

### Fixed
- The MLX inference path was missing a per-patch progress bar that the PyTorch/MPS path through `nnUNetPredictor` has always shown. Long inference runs now have visible progress feedback when the caller passes `progress=True`. The corresponding `mlx_predict.py` wrapper in TotalSegmentator now passes `progress=not quiet` to enable the bar by default.

## [0.3.0] - 2026-04-07

### Added
- `ModelBundle` and `InferenceEngine` are now separate classes — bundles hold weights/plans/dataset, engines hold inference-time configuration. Construct each independently or use `InferenceEngine(ModelBundle.from_task(...))`.
- Streaming sliding-window accumulator: rolling Z-direction buffer keeps memory bounded for large volumes. Skipped automatically when the volume fits in a single accumulator.
- `softmax_inplace` helper for converting logits to probabilities without an extra copy.
- `convert_pth_to_safetensors` public helper for one-shot conversion of legacy PyTorch checkpoints to the canonical layout.

### Changed
- **Adopt the nnU-Net canonical safetensors layout as the only on-disk format.** Files now live at `<base>.safetensors` (PyTorch-layout tensors with a `weight_layout=torch_ncdhw` metadata header), matching what `nnUNetTrainer` writes natively after the upstream safetensors PR. Models trained with new nnU-Net drop onto a Mac and load with no conversion step.
- The loader transposes conv weights at load time using `safetensors.numpy` (no torch round-trip). Runtime stays torch-free.
- `convert_model_folder` and the convert CLI now write the canonical layout directly. Output is byte-identical in shape and tagging to `nnUNetTrainer`'s native output.
- Package renamed from internal references to `nnunet-inference-mlx`; TotalSegmentator-specific defaults and hardcoding removed from `ModelBundle` and `InferenceEngine`. The package no longer assumes any particular weights directory.
- `nnUNet_results` environment variable is consulted before falling back to TotalSegmentator's default location.
- Metal cache limit set to 30% of system RAM by default for better large-volume behavior.

### Removed
- Legacy `<base>_mlx.safetensors` (MLX-pre-transposed) format. Existing files become orphaned and can be deleted; auto-conversion handles re-generation from `.pth` on first call.
- `save_weights_safetensors` and the `WEIGHT_LAYOUT_MLX` constant (unused after the rewrite).
- Hardcoded `Task` enum for TotalSegmentator model IDs.
- Source-layout fallback hint on `load_weights_safetensors`. The loader now requires the metadata header and rejects untagged files with an actionable error.

### Fixed
- Tests in `test_engine.py` use plain asserts instead of `return bool`, eliminating `PytestReturnNotNoneWarning`.

### Migration
- Pre-existing `_mlx.safetensors` files are no longer read. Delete them and let `ModelBundle.from_task` auto-convert from `.pth` on the next call (one-time torch dependency at conversion time only).
- Code that imported `save_weights_safetensors` should switch to `convert_pth_to_safetensors`.

## [0.2.0] - 2026-04-03

### Changed
- Replace custom `_AvgPool3d` with built-in `mlx.nn.AvgPool3d` in residual encoder blocks.
- Bump minimum MLX version from 0.22 to 0.25 (adds native 3D pooling, 3D conv speedups).

## [0.1.0] - 2025-04-14

Initial release: MLX inference backend for nnU-Net on Apple Silicon.
