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
`--no-wait` returns a job id for `status`, `fetch`, and `cancel`. `haversack remote results`
lists what the server has computed, newest first (`--identity idc:<crdc_series_uuid>`,
`--task`, `--limit`, `--json`; see "Listing results"). `GET /v1/health` answers
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
200 cached, 202 in flight (with the same progress headers), 404 absent. Its 200 carries
the `ETag` and `Content-Length` the `GET` would, and like the `GET` it answers a matching
`If-None-Match` with 304. A 404 on a result path says `Cache-Control: no-store`, as a 202
does: "not materialized" lasts only until somebody computes the result, and a cache that
kept it would hide the `public` 200 that follows. Every artifact beside the labels answers `HEAD` too (see
"Results by path"). The header never goes in the URL, so the URL stays the pure
cache key.

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
worker. An optional `deliverables` JSON list says what is rendered beside the labels (see
"Deliverables"). On the local server the queue is a bounded FIFO (`--max-pending`, default 16) and
past it the answer is 429 with `Retry-After`; on Modal the platform queues without bound. A
submit whose key is already in flight joins that job rather than starting a second one.
The response is 202 with the job id.

A job's status (`GET /v1/jobs/{id}`) carries its `state` (`queued`, `running`, `done`,
`failed`, `cancelled`), timestamps, a `progress` snapshot (stage, detail, part, fraction,
elapsed), the `input_identity`, the result `key`, and once done a `result` block with the
structure names, volumes in ml, provenance, timings, and the content digest of the output,
plus `deliverables` - what this job renders beside its labels - and a `links` object: `self`,
`events`, `result`, and `meta`, `preview` and `statistics` for the metadata and the
deliverables this job was asked for. For a path-addressable result those three (and `labels`)
are its paths; for a result with no path - an upload's, a `result:` reference's, a multi-input
job's, options off the grid menu - they are the job's own artifact routes, below. Follow the
links rather than building URLs.

`GET /v1/jobs/{id}/events` is Server-Sent Events: each event is the same status snapshot,
so a dropped stream needs no replay - resubscribe, or poll the status URL. `GET
/v1/jobs/{id}/result` returns the labels as `.seg.nrrd` (names, colors, extents, and the
full provenance in the header); `?format=nii.gz` converts on the way out and is the lossy
option. The result's `ETag` is its content digest, so `If-None-Match` gets a 304. A job that
is not done answers 409; a result whose bytes were purged answers 410. A server that cannot
yet see a finished result - on Modal, the api container's view of the result volume can trail
the worker's publication - answers 503 with `Retry-After` instead: retry it, it is not gone.
The path surface answers the same 503 where it cannot tell a miss from a stale view, rather
than 404 or a second compute.

`GET /v1/jobs/{id}/meta.json`, `/preview.png`, `/statistics.json` and `/statistics.tsv` are
the artifacts beside that result, for every job - and the only door to them for a result
with no path, whose deliverables were rendered and then unreachable before 2026-09-21. They
need the token like the rest of the jobs API, never less: a job's preview shows what was
uploaded. They resolve as `/result` does and answer by its rules first - 404 no such job, 409
not done, 410 the bytes are gone, 503 not visible yet - and then by the artifact's own: 202
with `Retry-After` while a render that will place it is still running (`Prefer: wait=N` on a
GET waits it out), 404 when none will, with the job's own `deliverables_unavailable` reason
where it gave one. `HEAD` answers the same without the body and never waits. Each carries an
`ETag` that is the digest of what is sent, honors `If-None-Match`, and is `Cache-Control:
private, no-cache`: not for a shared cache, and revalidated rather than assumed, which the
ETag makes cheap.

A job with a key is
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
sends `Prefer`. `GET /v1/segmentations` lists the cached results (next section).

Every file here has a validator of its own, and it is the digest of what is sent: the labels'
`ETag` is their content digest, and each artifact's is the digest of its own body. So a
`Cache-Control: no-cache` recompute, which republishes under the same key and the same URLs,
moves the tag of every file whose bytes moved and of no other, and a client holding an old
tag gets the new bytes, not a 304. (Before 2026-09-21 the four artifacts shared one tag
derived from the key, which a republication did not change.) `If-None-Match` is answered
on all of them, compared weakly as RFC 9110 has it (`W/"x"` matches `"x"`), and a 304 repeats
the 200's `Cache-Control` and `Vary` and never a `Preference-Applied`.

`HEAD` works on all of them and never computes, renders or waits, whatever `Prefer` says:
the `ETag`, `Content-Length`, `Cache-Control` and `Vary` of the GET with no body, or the 304.
For `preview.png` and `statistics.*` it is the probe for "has it rendered yet": 200 there,
202 with `Retry-After` while the labels are still computing or a render that will place that
deliverable is pending, and 404 otherwise - a deliverable nobody asked for is a 404 at once,
not a 202, because no render is coming. The anonymous twin answers the same: it reads the
api's pending-render marker (before 2026-09-21 it could not, and said 404 for a preview
seconds from landing). `meta.json` is 404 until the labels are published, under either verb. An artifact's absence is never for a cache to keep: its 404, like its
202, says `Cache-Control: no-store`, because an artifact arrives late - after `done`, on a
later request that asks for it, from another host where several share a result store - and
a 404 followed a moment later by a 200 is a legitimate sequence. Read a 404 for a
deliverable you asked for as "not as far as this request saw", and ask again. A method a
URL does not have is a 405 whose `Allow` lists the ones it has.

Uploads are not path-addressable - their identity is a digest nobody else can guess - so an
uploaded input's result is fetched through its job's `result` link, which is where the ETag
revalidation earns its keep. No content digest is: not a stored input's, and not a `result`
reference's, whose identity is the digest of the output it names.

## Listing results

`GET /v1/segmentations` lists the cached results this server can still resolve - its mounted
sources, its tasks, its CURRENT weights: an entry whose key this server would no longer
derive is left out, because its link would 404 and suggest a recompute. It is for authorized
callers, because the listing reveals the identities of uploaded content, and a filter by
digest would confirm one. Newest published first, a page at a time:

```
GET /v1/segmentations?identity=idc:<crdc_series_uuid>&task=ts.v2:total&limit=100&cursor=<token>

{"segmentations": [{"key": ..., "task": ..., "identity": [...], "options": {...},
                    "computed": ..., "published": ..., "bytes": ..., "links": {...}}, ...],
 "next_cursor": "<token>" | null}
```

`links` is there when the result has a path (one hosted input, default options or a grid
token); an upload's result, a `result:` reference's and a multi-input one are listed without:
their artifacts are reached through a job (`/v1/jobs/{id}/preview.png`, ...), and the listing
is of results, not of jobs - it knows no job id to build such a link from.
`published` is what the order is by: the time of the publication the entry holds now, so a
result recomputed with `no-cache` moves to the head, and nothing a READ does moves anything.

- **`limit`** is 1-1000, default 100. There is no other cap: before 2026-09-20 the listing
  returned the newest 500 and said nothing of the rest.
- **`cursor`** is the previous answer's `next_cursor`, handed back as given; it is null on the
  last page. It is a position, not an offset, so pages stay put while results are published:
  what is published (or republished) after your first page sorts ahead of it and is at the
  head of your next listing - never a repeated row, never a shifted one. A cursor the server
  did not issue is a 422.
- **`identity`** keeps the results computed from one input: `<source>:<identifier>` for a
  source from `/v1/sources` with a path surface, or a content digest (`sha256:<hex>`, what an
  upload's job reports as its `input_identity`). Repeat it, up to 100 times, for several
  inputs at once - the answer is the union, which is how a cohort asks "which of these series
  are done". It is COMPUTED, not searched: the server derives the key of every task it serves,
  under the default options and each grid token, and looks those names up - what a `HEAD` on
  each path would tell you, in one request, and as fast on a cache of thousands as on an
  empty one. So it finds exactly the results that have a path, plus an upload's under those
  same options. A result computed with other options (`interp`, `folds`, ...) or from several
  inputs is not found this way; the plain listing and the `task` filter show it.
- **`task`** keeps one task's results (any spelling the task routes take). With `identity`
  it is one key per option set. Alone it is the one filter that has to read every entry's
  metadata, which the server does in parallel and remembers in memory for as long as it
  runs - there is no index on disk to fall out of step with the cache.

A 422 names what was refused (an identity that is neither form, an unknown task, a `limit`
out of range, a foreign cursor). On Modal the listing is read from a view of the result
volume newer than the request, like every other statement that something is absent; when no
such view can be had the answer is 503 with `Retry-After`, never a shorter list.
## Deliverables

Beside the labels a server renders light deliverables: `preview`, a three-plane overlay
(`preview.png`), and `statistics`, per-structure volumes and intensities (`statistics.json`,
also as `.tsv`). They are rendered after the job already reports `done`, into the result's
own cache entry, so they are eventually consistent: a GET or a HEAD of one that is still
rendering answers 202 with `Retry-After`. They are served by path where the result has one,
and through the job (`/v1/jobs/{id}/preview.png`, ...) always. A job whose result has no cache
entry to render into - a server run without a result cache - says so in
`deliverables_unavailable` instead of listing what no route would serve.

**Which are rendered is the request's to say.** `POST /v1/jobs` takes a `deliverables` form
field, a JSON list of names: `["statistics"]`, or `[]` for none. Absent, the job gets the
deployment's set - `GET /v1/health` lists it as `deliverables` - and that set is also the
ceiling: a name this server does not render (`deliverable_not_offered`) or has never heard
of (`unknown_deliverable`) is refused at submit, with a 422 that says what the server
offers. The list is a field of its own, beside `options`; inside them it is refused
(`misplaced_deliverables`).

**A deliverable is never part of the result.** Options are part of a result's key; the list
is not. Declining the preview computes the same labels under the same `key`, with the same
`ETag`, and a later request for that input and task is a cache hit whatever either list
said.

**A cache hit still honors the list.** When the labels are cached and the stored result
lacks a deliverable the request names - an earlier request declined it, or it never
rendered - the local server renders it then, from the input it still holds (the upload
just sent, a stored input, a fetched series still in its cache), into the same cache entry:
no recompute, no new generation, and one render per result at a time. It never fetches an
input again to do so. What it cannot deliver it says: the job's `deliverables_unavailable`
maps each such name to the reason - the input is no longer staged on the server, or a
render of this result that does not include it is still running - and `links` leaves it
out. `Cache-Control: no-cache` recomputes the result with its deliverables. A Modal
deployment renders only in the worker that computes a result, so there a cache hit never
renders: it reports what the stored result lacks in the same field (or, when the api
container cannot see the result volume's latest state, that it cannot tell yet).

A job's status carries `deliverables`, the list it renders, and its `links` name only what
was asked for and is, or will be, there. A submit that joins a job already in flight adds
its names to that job's list, until the job publishes.

**A read never renders.** `preview.png` and `statistics.json` by path serve what exists, to
anyone. When the result is cached and the artifact is not, the answer is 404 - to an
anonymous caller and to an authorized one, with `Prefer` or without - and it names the door
that renders it: an authorized `POST /v1/jobs` of the same input and task with the
deliverable in its list. (A result that is not cached at all is still computed by an
authorized GET with `Prefer`, with the deployment's set.)

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
  `haversack cache clean results` sweeps it.
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
haversack modal deploy [--gpu L40S] [--app-name haversack-serve] [--cache-volume NAME] [--scaledown 120] [--no-proxy-auth]
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
local copies of the results it serves), `HAVERSACK_ARTIFACTS` (default `preview,statistics`:
the deliverables this deployment renders - the default of a request that names none, and the
most one may name),
`HAVERSACK_IDC_CLOUD` (`aws`, or `gcp` to read IDC's Google Cloud mirror first - for a
deployment that lives there), `HAVERSACK_CACHE_VOLUME` (the result cache's volume, below),
and `HAVERSACK_PUBLIC=1`, which adds an anonymous read-only twin that serves cache hits and
nothing else. A redeploy does not preempt warm containers running the old code; stop the
app first. Costs run while a worker is warm.

A deployment's stores are named after the app - `<app>-scratch`, `<app>-inputs`, the
`<app>-jobs` job store, `<app>-cache` - except the weights volume, `haversack-weights`, which
every deployment shares. `--cache-volume` / `HAVERSACK_CACHE_VOLUME` names the result cache
instead, so a deployment under a new name can start with an earlier one's results: a result
key holds no app name (input identity, task, options, weights versions, cache epoch), so a
cache answers the same requests whichever app mounts it. Only the cache can be named;
scratch, inputs and the job store hold one deployment's job ids, uploads and flights. Nothing
is renamed or migrated, and unset means `<app>-cache` as before. Two cases:

- **The cache's first deployment is gone** (stopped, or redeployed under another name):
  completely safe. It is the same volume, read and written by the same code.
- **Two live deployments share one cache**: they behave like more worker containers of one
  app, which is a case the cache is built for - a result is published as a whole generation
  behind one pointer. The one difference is single flight: "a submit whose key is already in
  flight joins that job" is decided in the per-app job store, so the two apps can each
  compute the same key at the same time. That is duplicate GPU work, not corruption: both
  publish, one pointer wins, the other generation is pruned.

In both, mind `HAVERSACK_RESULTS_KEEP`: every publication evicts down to the PUBLISHING app's
bound (default 500), least recently used first. A deployment that adopts a cache of 2,000
results with the default evicts 1,500 of them at its first job - deploy it with a bound at
least the size of the cache it adopts. With two live apps the smaller bound is the real one.

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
| GET | `/v1/segmentations` | token | cached results, newest first: `identity`, `task`, `limit`, `cursor` |
| POST | `/v1/jobs` | token | submit |
| GET | `/v1/jobs` | token | brief status of every known job |
| GET | `/v1/jobs/<id>` | token | full status, result metadata, links |
| GET | `/v1/jobs/<id>/events` | token | status snapshots as Server-Sent Events |
| GET | `/v1/jobs/<id>/result` | token | the labels (`?format=nii.gz` converts) |
| GET | `/v1/jobs/<id>/meta.json` | token | the job's result: provenance and structure names |
| HEAD | `/v1/jobs/<id>/meta.json` | token | the same, no body |
| GET | `/v1/jobs/<id>/preview.png` | token | the job's rendered preview; 202 while it renders |
| HEAD | `/v1/jobs/<id>/preview.png` | token | probe: rendered, rendering, absent |
| GET | `/v1/jobs/<id>/statistics.json` | token | the job's per-structure volumes |
| HEAD | `/v1/jobs/<id>/statistics.json` | token | probe: rendered, rendering, absent |
| GET | `/v1/jobs/<id>/statistics.tsv` | token | the same as a table |
| HEAD | `/v1/jobs/<id>/statistics.tsv` | token | probe: rendered, rendering, absent |
| DELETE | `/v1/jobs/<id>` | token | cancel or delete |
| GET | `/v1/inputs/<digest>` | token | is this content already here |
| PUT | `/v1/inputs/<digest>` | token | store one file, digest checked |
| POST | `/v1/inputs` | token | store a multi-file input as one tree |
| HEAD | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | read | probe: cached, in flight, absent |
| GET | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | read, `Prefer` needs token | the labels |
| DELETE | `/v1/<source>/<identifier>/<task>` | token | drop the cached result and every artifact |
| DELETE | `/v1/<source>/<identifier>/<task>/labels.seg.nrrd` | token | the same, addressed by the labels file |
| GET | `/v1/<source>/<identifier>/<task>/meta.json` | read | provenance and structure names |
| HEAD | `/v1/<source>/<identifier>/<task>/meta.json` | read | the same, no body |
| GET | `/v1/<source>/<identifier>/<task>/preview.png` | read | a rendered preview |
| HEAD | `/v1/<source>/<identifier>/<task>/preview.png` | read | probe: rendered, rendering, absent |
| GET | `/v1/<source>/<identifier>/<task>/statistics.json` | read | per-structure volumes |
| HEAD | `/v1/<source>/<identifier>/<task>/statistics.json` | read | probe: rendered, rendering, absent |
| GET | `/v1/<source>/<identifier>/<task>/statistics.tsv` | read | the same as a table |
| HEAD | `/v1/<source>/<identifier>/<task>/statistics.tsv` | read | probe: rendered, rendering, absent |

Every path-addressed route that names a file also exists with the `_res-1mm` token
before the extension.

## Reference

`/docs` is the interactive OpenAPI view and `/openapi.json` the document. The routes there
are tagged `service`, `tasks`, `jobs`, `inputs`, and `results`, and the document's own
description is the "What it is" and "Reads and computes" sections of this guide, so the two
cannot say different things.
