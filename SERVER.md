# haversack server

The server guide. `haversack docs --server` prints it; `haversack docs --server jobs` prints
one section. The machine-readable reference for every route, parameter, and response
schema is the OpenAPI document the server publishes itself, at `/docs` (Swagger UI) and
`/openapi.json`. This guide is the part a schema cannot say: what the routes mean, which
rules they obey, and how to run the thing.

## What it is

One job protocol, two deployments. `haversack serve` runs it on the machine in front of you;
`haversack modal deploy` runs the same protocol on a GPU worker in your Modal account. A
client - the `haversack remote` command, a 3D Slicer panel, a notebook - talks to either the
same way.

The server holds a `Segmenter` with warm models, a bounded job queue, a durable job store,
and a result cache addressed by what was segmented rather than by which job did it. Every
input resolves to a content identity (an upload to its sha256, an IDC series to its crdc
UUID), and a result is keyed on that identity, the task and its options, the weights
versions, and a cache epoch that moves only when a build would compute different bytes from
the same inputs - not on every release. Ask twice, compute once.

## Quick start

```bash
haversack serve --port 8790

export HAVERSACK_SERVER=http://127.0.0.1:8790
haversack remote tasks
haversack remote submit scan.nii.gz --task ts.v2:total_fast -o labels.seg.nrrd
haversack remote submit idc:<crdc_series_uuid> --task ts.v2:total_fast -o labels.seg.nrrd
```

`submit` uploads (or names a hosted series), streams progress, and downloads the labels;
`--no-wait` returns a job id for `status`, `fetch`, and `cancel`. `GET /v1/health` answers
when the server is ready. That is the personal setup: the server generated a token, printed
it, and left it in `~/.cache/haversack/serve/<port>.token`, readable by you alone and
stamped with the server's process id, and `haversack remote` on the same machine read it
back without being told. The client uses that file only for a loopback address whose port
has a live server behind it; a file a crashed server left is ignored, and every server
start clears one for its port. For other machines, choose the token and pass it on both
sides:

```bash
haversack serve --host 0.0.0.0 --token choose-a-secret
HAVERSACK_SERVER=http://gpu-box:8790 haversack remote --token choose-a-secret submit scan.nii.gz --task ts.v2:total_fast -o labels.seg.nrrd
```

`remote` takes the token from `--token`, then `HAVERSACK_TOKEN`, then the local file.

## Reads and computes

Three rules decide every request, and they are the same on the local server and on Modal.

**A token computes; anonymous reads.** Authorization is `Authorization: Bearer <token>`,
and every server has a token: the one you gave it, or the one it generated and left in a
file only your user can read. Without the token a caller can read health, the version, the
task list and descriptions, the sources, and any result already in the cache - and nothing
else. Anonymous never computes and never stores, so it never spends your GPU or your disk.
The bind address decides nothing here, on purpose: a server on the loopback interface is
routinely put on a network by a reverse proxy or a tunnel, and it must still face a token
then. `--no-token` is the one way to run open, and open means open: anything that reaches
the port computes, a web page you have open included, and nothing in the server pretends
otherwise. It exists for a machine you trust end to end and for nothing else. On Modal,
auth is the platform's proxy tokens.

**A plain GET is a read; `Prefer` is the intent to compute.** Fetching a result by its path
(below) returns it if the cache holds it and 404 "not materialized" otherwise; on a key
that is already being computed, a plain GET waits the default up to that flight's finish
before answering, so a probe that must not block is a `HEAD`. An authorized
caller who wants it computed says so with RFC 7240: `Prefer: wait=N` holds the connection
up to N seconds (default 30, at most 110) and returns the bytes if they arrive in time, else
202 with progress in the headers; `Prefer: wait=0` or `Prefer: respond-async` starts the
computation and returns 202 at once. A `HEAD` on the same path probes without computing:
200 cached, 202 in flight (with the same progress headers), 404 absent. The header never
goes in the URL, so the URL stays the pure cache key.

**`Cache-Control: no-cache` recomputes.** On a submit or an authorized result GET (with
`Prefer`) it means
"do not serve me a stored result" in the RFC 9111 sense: the job runs again and the new
result is published under the same key, so links and artifacts survive. It is the lever for
"the model behind this task changed." There is no `no-store`. An engine may refuse to be
served from cache at all; VoxTell does, because free-text prompts make its key space
unbounded.

## Jobs

`POST /v1/jobs` (multipart) takes `task`, an optional `options` JSON object, and the input:
either a `file` part - shorthand that stays valid - or a `source` JSON list, one entry per
declared input role: `{"kind": "upload"}`, `{"kind": "input", "sha256": ...}` for content the
server already holds, a hosted identifier such as `{"kind": "idc", "crdc_series_uuid":
...}`, or `{"kind": "result", "id": "<key>"}` for a result this server computed (see
"Results as inputs"). Options are validated at submit against the task's published parameter schema, and
sources are bound to the task's declared inputs by role name, never by position, so a wrong
request is refused with a 422 naming the problem rather than failing minutes later in a
worker. On the local server the queue is a bounded FIFO (`--max-pending`, default 16) and
past it the answer is 429 with `Retry-After`; on Modal the platform queues without bound. A
submit whose key is already in flight joins that job rather than starting a second one.
The response is 202 with the job id.

A job's status (`GET /v1/jobs/{id}`) carries its `state` (`queued`, `running`, `done`,
`failed`, `cancelled`), timestamps, a `progress` snapshot (stage, detail, part, fraction,
elapsed), the `input_identity`, the result `key`, and once done a `result` block with the
structure names, volumes in ml, provenance, timings, and the content digest of the output,
plus a `links` object: `self`, `events`, `result`, and for a path-addressable result the
labels and the artifacts this deployment produces. Follow the links rather than building
URLs.

`GET /v1/jobs/{id}/events` is Server-Sent Events: each event is the same status snapshot,
so a dropped stream needs no replay - resubscribe, or poll the status URL. `GET
/v1/jobs/{id}/result` returns the labels as `.seg.nrrd` (names, colors, extents, and the
full provenance in the header); `?format=nii.gz` converts on the way out and is the lossy
option. The result's `ETag` is its content digest, so `If-None-Match` gets a 304. A job that
is not done answers 409; a result whose bytes were purged answers 410. A server that cannot
yet see a finished result - on Modal, the api container's view of the result volume can trail
the worker's publication - answers 503 with `Retry-After` instead: retry it, it is not gone.
The path surface answers the same 503 where it cannot tell a miss from a stale view, rather
than 404 or a second compute. A job with a key is
served from the result cache's entry for that key - the same bytes the path surface serves -
and from the job's own copy only when there is no entry or the key has since been recomputed
to different bytes. The download does not honor `Range`. On the local
server the record itself keeps answering `GET /v1/jobs/{id}` (marked `evicted`) after it
leaves memory or the server restarts, until `DELETE` removes it; Modal's records live in
its job store for `HAVERSACK_JOBS_TTL_H`. `DELETE` cancels an active job or deletes a
finished one; the local server reports a running job as `cancelling` and moves it to
`cancelled` at the next patch, while Modal reports `cancelled` at once.

Job records are durable: a sqlite `jobs.db` in the work directory outlives the process, so a
restart re-queues what was queued instead of dropping it and reclaims job directories no
record owns. `--keep-finished` bounds how many finished jobs stay in memory and on disk;
`--jobs-ttl-hours` bounds how long a record lasts.

## Results by path

A result of a hosted input is addressable without its job:

```
/v1/<source>/<identifier>/<task>/labels.seg.nrrd
/v1/<source>/<identifier>/<task>/meta.json
/v1/<source>/<identifier>/<task>/preview.png
/v1/<source>/<identifier>/<task>/statistics.json     (also .tsv)
```

`<source>` is a prefix from `/v1/sources` whose entry says `path_addressable` (every hosted
source; not `result`), `<identifier>` that source's own id (an IDC
series UUID, a TCIA SeriesInstanceUID, an OpenNeuro path, ...), `<task>` a catalog name in
either its canonical `eco:name` form or its bare alias. A grid variant is a token in the
filename: `labels_res-1mm.seg.nrrd` is the same result restored at 1 mm isotropic, and every
artifact takes the same token. Reads obey the rules above: a cache hit is served with
`Cache-Control: public` and an `ETag`; a miss is 404 unless the caller is authorized and
sends `Prefer`. `GET /v1/segmentations` lists every cached result this server can still
resolve (its mounted sources, its current weights) with its links, for authorized callers,
because the listing reveals the identities of uploaded content.

Uploads are not path-addressable - their identity is a digest nobody else can guess - so an
uploaded input's result is fetched through its job's `result` link, which is where the ETag
revalidation earns its keep. No content digest is: not a stored input's, and not a `result`
reference's, whose identity is the digest of the output it names.

## Sources and the input store

`GET /v1/sources` lists what this server can fetch for itself and the identifier grammar of
each: `idc` (NCI Imaging Data Commons, by crdc_series_uuid), `tcia` (by SeriesInstanceUID),
`openneuro` (`ds<number>/<file path>`), `zenodo` (`<record>/<file>`, `!member` for a file
inside a zip), `hf` (Hugging Face, `<owner>/<repo>@<revision>/<path>`), `s3` (`<bucket>/<key>`,
where the bucket must be one the server serves - the response lists them) and `github`
(`<owner>/<repo>@<tag>/<asset>`, the tag required). A server that keeps a result cache also
lists `result` - not a repository but a result this server computed, named by its key (see
"Results as inputs"). Each entry carries its `prefix`, `id_pattern`, `description`, whether it
is `enabled`, and `path_addressable`: whether results of its inputs have the path surface
above. Every hosted source does; `result` does not. A fetch happens
at dispatch, as a visible "fetch" progress stage, so submits stay small and a full queue
never wastes an upload. Fetched series live in a bounded cache in the work directory.

The input store lets a client send bytes once and refer to them by digest. `GET
/v1/inputs/{digest}` says whether the server holds that content; `PUT /v1/inputs/{digest}`
stores a single file, checking the digest against the bytes and refusing anything it cannot
identify as a medical image (NIfTI, NRRD, MetaImage, DICOM - a blob nothing can open is a
job that was always going to fail, so it fails here); `POST /v1/inputs` stores a multi-file
input such as a DICOM series as one tree whose digest is taken over its members, so the same
series zipped twice is the same identity. `POST /v1/inputs` with a `from_job=<id>` form
field promotes a job's result into the store, so one job's output becomes another's input
without the bytes passing through the client. No route ever hands input bytes back. All of it is authorized only.

## Results as inputs

A job's input can be a result this server computed: `{"kind": "result", "id": "<key>"}`,
where `<key>` is the `key` a finished job reports. That makes jobs composable - CT to
segmentation, then something computed from (CT, segmentation) - with each step cached under
the rules above, so a cheap step run again never repeats the expensive one before it.
`GET /v1/sources` lists `result` on a server that has a result cache; one started with
`--no-result-cache` does not offer the kind.

```
result:<key>                       the result's primary output (labels)
result:<key>!<name>                a named output; only `labels` exists today
result:<key>@sha256:<digest>       pinned: refused unless the output is still those bytes
```

The `id` of the source entry is whatever follows `result:` - the key alone, or with `!<name>`,
`@sha256:<digest>`, or both (`<key>!labels@sha256:<digest>`). `haversack remote submit
result:<key> --task ...` sends the same thing for a task that takes one label map.

The key is 64 hex characters and nothing else, so a reference can only name a result of
THIS server: there is no way to spell a host, and a result on another server cannot be
referenced.

**The identity of a reference is the content digest of the output it names, not the key.**
`Cache-Control: no-cache` republishes other bytes under the same key, and a downstream result
keyed on the key would outlive the mask it was computed from. So the server resolves the
reference at submit - key, current generation, the named output's `sha256` - and that digest
is what the job's `input_identity` and its own `key` are built from. It is the same digest
an upload of those bytes has, so referring to a result and sending its bytes are one request
and share one cached answer. After a recompute of the upstream result, the same reference is
a new downstream request; a reference pinned with `@sha256:` to the old bytes is refused
instead (409 `result_changed`).

**A reference that does not resolve is refused at submit**, never queued to fail in a
worker: 409 `result_missing` when the server holds no such result (never computed here, or
evicted - run the job that produces it first, then submit again), 422 `unknown_output`
naming the outputs the result has, 422 `wrong_input_kind` when a label map is bound to a
role that takes an image, 409 `result_unreadable` for an entry that states no usable digest
for its output (recompute it with `no-cache`), and 422 `unsupported_output` for a listed
output this build cannot hand over yet. On Modal a miss is believed only from a view of the result volume
newer than the request; when none can be had the answer is 503 with `Retry-After`, as it is
for a read. The worker that fetches a reference resolves it AGAIN and hashes what it copied,
so a job computes only ever from the bytes it was keyed on. If the result was recomputed or
evicted between the submit and the fetch, the job fails with "the referenced result changed"
(or "no result ...") rather than compute from other bytes - unless that worker can still read
the pinned bytes themselves, which on Modal a worker whose view of the result volume predates
the change can: then it computes from them, and the result is exactly what its key says.
`Cache-Control: no-cache` on the downstream job resolves the reference again and recomputes
the downstream result; it never recomputes the upstream one, which is that job's own
`no-cache`.

**A role says what it takes.** Each entry of a task's `inputs` has a `kind`: `image`, an
intensity volume, or `labels`, a label map read with its segment names
(`haversack.labelmap.read_label_map`: the names, the geometry, and the task that made it). A
consumer selects structures by NAME - `liver` is a different label value in every catalog -
so a label map without names (a NIfTI) is refused by default, and one where a name could not
be matched to voxels without choosing is refused always: two segments on one label value, two
label values under one name, overlapping (layered) segments, or voxels that are not integers.
A `.seg.nrrd` uploaded or stored by digest binds to a `labels` role as a reference does; the
task that made it is known for a reference, and for an upload only if its header says.

A result computed from a reference says so in `provenance.inputs`: the record has `kind:
result`, the digest, and an `origin` naming the upstream `result` key, `output`, `task` and
the `weights` versions its provenance states, with the upstream task's `attribution`.
`derived_from` carries what is above that hop, flat: `results` lists every earlier hop by
reference (key, digest, task, weights), and `inputs` holds each ORIGINAL input once, by
value, with its origin, license and citation - so the terms of the data a chain started from
survive every hop, and survive the upstream entries being evicted. One thing follows from a
reference and an upload of the same bytes being one request: they share a cached answer, and
provenance describes the computation that PRODUCED it. If the same label bytes were first sent
as an upload, a later job that refers to the result is a cache hit whose `provenance.inputs`
says "uploaded by the caller" and carries no `derived_from` - the terms of the original data
are then on the upstream result, not on this one. `Cache-Control: no-cache` recomputes it from
the reference. Results of references are not path-addressable; fetch them through the job's
`result` link.

## Tasks and options

`GET /v1/tasks` lists catalog names; `GET /v1/tasks/{task}` describes one: its `engine`,
`lineage`, `modality`, the `structures`, the `weights` and whether they are installed, the
`inputs` it takes (each with a role name and a kind - `image` or `labels`; every declared role is required), its
`parameters` as two JSON Schemas, and its `behavior`. Task names cross the wire as qualified catalog
names only (a bare `total_fast` is a 404 that names `ts.v2:total_fast`); the in-process API's ability to run a model folder by path stops at this
boundary. The grammar `eco:name@version` names an ecosystem, a task, and a weights version;
all spellings of one task converge on one cache key. A version is a pin, and the server holds
it the way every catalog does: the INSTALLED version decides, and an unknown one satisfies
nothing. A pin the server provably does not run is a `409` on every route, naming what it runs.
One it cannot check yet - the weights are not installed, or carry no version record - is
accepted by `POST /v1/jobs` and `POST /v1/tasks/{task}/prepare`, which carry it to the
worker's catalog to install that version or refuse it (such a job is never answered from the
cache); on a read, which cannot install anything, it is a `409` pointing at `POST /v1/jobs`.
A job submitted with a pin reports it as `version` beside the canonical `task`.

`GET /v1/segments?q=...` answers which tasks produce a segment, and with what label value,
before anything is installed. A segment is one item of a task's segment table, as in a
`.seg.nrrd` or a DICOM segmentation: the label `value` it is written with, the `layer` it lives
in where the output overlaps (no `layer` means layer 0), and its `id` - the model's own token
for it, a code in that model's class list rather than a display name or an identity across
models. A task's `structures` in `GET /v1/tasks/{task}` are those ids, in label order. The
search runs over every task's segments as its model states them, limited to the tasks this
deployment serves (not to be confused with `/v1/segmentations`, which lists cached results).
`q` is word prefixes in any order by default (`kid left` finds `kidney_left` and
`left_kidney`); `mode=glob` matches a shell pattern against the whole id. `field=key` (the
default) compares ids folded for case, spaces and hyphens; `field=id` holds a glob to the
model's own spelling. `catalog`, `modality` and `limit` (ids returned, 1-1000, default 100)
narrow it.

The answer groups segments by folded id: `results` is a list of `{key, ids, segments}` - `ids`
the spellings the models use, `segments` each `{task, value, id, modality, layer?}` - after
`index`, `key_count`, `segment_count`, `offset`, `truncated` and `next_offset`. `notes` maps a
task to a caveat its values need (a MONAI head whose declared outputs are not the labelmap it
writes), and `open_vocabulary` lists served tasks with no fixed segment list (VoxTell), which
may segment anything a prompt names. Answers are paged, for a reader that cannot be trusted to
see a long one whole: `limit` ids from `offset`, in an order fixed for a given `index` (a short
version, which changes if the index is rebuilt between pages); `truncated` means more ids
follow and `next_offset` is where they start, null on the last page; `count_only=true` returns
the counts alone. The last key is always `end`, a receipt repeating the page and the next
offset - an answer without it lost its tail on the way, however complete it looks. A `422` is
a query the search refuses - a negative `offset`, empty or longer than 200 characters,
`mode=regex` (a pattern from anyone can take unbounded time to evaluate, so regex stays with
`haversack tasks --find PATTERN --regex`, locally), `field=id` with word search, a catalog with
no tasks here, a `limit` out of range - and a `503` means the index or this server's own task
list is unavailable. The index is `data/segments.json`, with the version each list was read
at; `haversack catalog check` says which records a catalog change has made stale.

`parameters.algorithm` is the engine's own knobs, empty for nnU-Net tasks, a `prompt` for
VoxTell. `parameters.processing` is haversack's, offered only where haversack owns the chain:

| option | meaning |
|---|---|
| `grid` | output grid: `"input"` (default), `"model"` for the network's own spacing, or an isotropic size in mm |
| `interp` | how the result is restored to the output grid: `linear` (sub-voxel boundaries) or `nearest` |
| `envelope_mm` | crop the network's field of view to this margin around the body, in mm: faster, and not the same labels. Unset, `0` or `null` runs the whole volume (the default) |
| `folds` | which trained folds to ensemble |
| `configuration` | nnU-Net configuration, when a model ships more than one |
| `resampling_order` | spline order of the forward resample |
| `convention` | grid-alignment convention; `auto` follows the model's lineage |

`{"grid": 1}` and the `_res-1mm` path token are the same request. `{"no_cache": true}` in the
options is the same as the header.

## Storage and caches

- **Work directory** (`--workdir`, default a temp directory): `jobs.db`, the job directories,
  and the fetched-series cache. Transient by design.
- **Result cache** (`--cache-dir`, default `~/.cache/haversack/results`, or
  `HAVERSACK_CACHE_DIR`; `--no-result-cache` keeps nothing): durable, content-addressed,
  shared by every server run on the machine. `haversack cache list` shows its size and
  `haversack cache clean results` sweeps it - with `--result-store` in use, that clears only
  this machine's copy, never the shared store.
- **Deleting from a shared store** (`DELETE` on a path-addressed result, or
  `haversack cache delete`): removes the ENTRY - the pointer every server reads, and this
  server's copy. The bytes stay until a sweep reclaims what nothing references any more -
  which a server with `--result-store` does every `--sweep-interval-hours` (24 by default,
  0 to turn it off), and which `haversack cache sweep <store>` does on demand. If your
  deployment needs "deleted" to mean the bytes are gone by a deadline, set that interval to
  it; the sweep only ever removes bytes no result refers to.
- **Shared result store** (`--result-store s3://bucket/prefix`, or `gs://`, `az://`, or
  `HAVERSACK_RESULT_STORE`; credentials from the environment as obstore reads them): every
  server naming the same store serves every result any of them computed. The store is the
  authority and `--cache-dir` becomes this server's local copy of it. The store must honor
  conditional writes; the server checks at startup and refuses one that does not.
  Cannot be combined with `--no-result-cache`: the store needs a local copy in front of
  it. Unreferenced bytes are removed by `haversack cache sweep <store>`, which nothing runs
  on a schedule; `haversack cache push`/`pull` move results between a machine and a store.
- **Weights** (`--model-root`, or the same default as the command line): shared with
  `haversack segment`, so a model either side downloaded is warm for both.
- **Not shared with the command line:** `haversack segment` neither reads nor writes the
  result cache, and the inputs `haversack get` fetches go to `~/.cache/haversack/inputs`,
  not the server's series cache.

`--cache-models` (default 5) is how many models stay warm; a `total` union cycles through
five, and fewer than that means a reload per part.

## Engines

`--allow-transpose` serves the tasks whose plans permute the axes
(`dentalsegmentator:base`, `totalvibe:vibe_sagittal`, `totalvibe:pancreas`). They are refused
by default because that path has not been confirmed against an outside implementation for
each model. It is deployment policy, not a request parameter, so a client cannot ask for it;
without it those tasks are listed and described but refuse to run. On Modal the same switch is
`HAVERSACK_ALLOW_TRANSPOSE=1` at deploy.

The nnU-Net engine is always on. The others are switched on per deployment by environment
variable, `HAVERSACK_FASTSURFER=1`, `HAVERSACK_SYNTHSTRIP=1`, `HAVERSACK_VOXTELL=1`,
`HAVERSACK_MONAI=1`, and each needs its runtime installed in the environment the server runs
in - SynthStrip's pins numpy below 2, so it runs from its own environment. `GET /v1/version`
reports which engines are enabled, and `GET /v1/tasks/{task}` names each task's engine.

## Deploying to Modal

```bash
haversack modal deploy [--gpu L40S] [--app-name haversack-serve] [--scaledown 120] [--no-proxy-auth]
modal app stop haversack-serve --yes
```

The deploy prints the URL. Modal is the queue there: submits spawn, the autoscaler drains
them onto up to `HAVERSACK_MAX_CONTAINERS` warm workers (default 1, so parallel requests run
serially on one warm GPU - the economical posture), and a worker lingers `--scaledown`
seconds after its last job. Auth is Modal proxy auth: per-person tokens minted and revoked in
the Modal dashboard, sent as `Modal-Key` and `Modal-Secret` headers, with no auth code in
haversack. The bundled `haversack remote` client sends a bearer token only, so today it
reaches a Modal deployment only when that deployment was made with `--no-proxy-auth`, which
is for smoke tests: anyone with the URL can spend the GPU.

Deploy-time knobs, all environment variables because Modal resolves decorators at import:
`HAVERSACK_GPU` (default L40S; A10 is the economical fast-mode choice), `HAVERSACK_APP_NAME`,
`HAVERSACK_SCALEDOWN`, `HAVERSACK_MAX_CONTAINERS`, `HAVERSACK_SNAPSHOT` (memory snapshots,
default on), `HAVERSACK_WARM_TASK` (the task loaded at startup, default `ts.v2:total_fast`),
`HAVERSACK_JOBS_TTL_H` (default 72), `HAVERSACK_RESULTS_KEEP` (default 500),
`HAVERSACK_INPUTS_GB` (default 50), `HAVERSACK_API_MIRROR_GB` (default 2: the api container's
local copies of the results it serves), `HAVERSACK_ARTIFACTS` (default `preview,statistics`),
`HAVERSACK_IDC_CLOUD` (`aws`, or `gcp` to read IDC's Google Cloud mirror first - for a
deployment that lives there),
and `HAVERSACK_PUBLIC=1`, which adds an anonymous read-only twin that serves cache hits and
nothing else. A redeploy does not preempt warm containers running the old code; stop the
app first. Costs run while a worker is warm.

## Operating it

`GET /v1/health` is readiness (version, device, task count, whether the queue accepts,
which sources are enabled); `GET /v1/version` is what is deployed (contract, package and
engine versions, weights). The server logs to stderr. Stop it with Ctrl-C; queued jobs are
kept and re-queued on the next start. Read the job store without a server:

```bash
uv run python tools/jobs.py --db <workdir>/jobs.db list      # also show, stats, reap
```

The bearer token gates computation, not confidentiality: nothing is encrypted and the
result cache is readable by anyone who can reach the port. Loopback by default for that
reason; put it on a network only with a token you chose, and on a private network. The
authenticated deployment is the Modal one.

## Routes

The complete list; `/docs` has every parameter and schema. Auth: `read` works anonymously,
`token` needs the bearer token.

| method | path | auth | what |
|---|---|---|---|
| GET | `/v1/health` | read | readiness |
| GET | `/v1/version` | read | what is deployed |
| GET | `/v1/tasks` | read | task names |
| GET | `/v1/tasks/<task>` | read | describe a task |
| GET | `/v1/segments` | read | which tasks produce a segment, and with what label value |
| POST | `/v1/tasks/<task>/prepare` | token | install a task's weights now |
| GET | `/v1/sources` | read | the hosted sources (and `result`), their identifier grammar, and which have a path surface |
| GET | `/v1/segmentations` | token | every cached result, with links |
| POST | `/v1/jobs` | token | submit |
| GET | `/v1/jobs` | token | brief status of every known job |
| GET | `/v1/jobs/<id>` | token | full status, result metadata, links |
| GET | `/v1/jobs/<id>/events` | token | status snapshots as Server-Sent Events |
| GET | `/v1/jobs/<id>/result` | token | the labels (`?format=nii.gz` converts) |
| DELETE | `/v1/jobs/<id>` | token | cancel or delete |
| GET | `/v1/inputs/<digest>` | token | is this content already here |
| PUT | `/v1/inputs/<digest>` | token | store one file, digest checked |
| POST | `/v1/inputs` | token | store a multi-file input as one tree |
| HEAD | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | read | probe: cached, in flight, absent |
| GET | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | read, `Prefer` needs token | the labels |
| DELETE | `/v1/<source>/<identifier>/<task>` | token | drop the cached result and every artifact |
| DELETE | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | token | the same, addressed by the labels file |
| GET | `/v1/<source>/<identifier>/<task>/meta.json` | read | provenance and structure names |
| GET | `/v1/<source>/<identifier>/<task>/preview.png` | read | a rendered preview |
| GET | `/v1/<source>/<identifier>/<task>/statistics.json` | read | per-structure volumes |
| GET | `/v1/<source>/<identifier>/<task>/statistics.tsv` | read | the same as a table |

Every path-addressed route that names a file also exists with the `_res-1mm` token
before the extension.

## Reference

`/docs` is the interactive OpenAPI view and `/openapi.json` the document. The routes there
are tagged `service`, `tasks`, `jobs`, `inputs`, and `results`, and the document's own
description is the "What it is" and "Reads and computes" sections of this guide, so the two
cannot say different things.
