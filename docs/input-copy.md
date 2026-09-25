# The input copy: a decoded, single-file form of a cached input

*Specification, 2026-09-25; built the same day on branch `claude/input-copy`
(`haversack.input_copy`, the caches in `serve.py`; the tag converter, first `haversack.dicom_tags`, is
`duckn.dicom_tags` since duckn 0.5.2). The decisions in §2
are the user's; the numbers in §3 were measured the same day (Modal and an M2, one CT). §11
records what building it changed.*

## 1. What it is for

A job reads its input every time it runs. For a DICOM series that is a full decode of every
slice, whether the same series was read a minute earlier or not: on the 709-slice CPTAC-CCRCC CT
`idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c` the decode is 13 s on a Modal worker - most of a warm
`ts.v2:total_fast` job - and a fresh container reading a series another container cached paid
45 s. The input copy is the input decoded once, when the cache stores it, and kept INSTEAD of
the original: an entry holds one form or the other, never both.

It never changes what a job computes (§8). The original's facts - its digest, its DICOM
identifiers - are recorded when it is fetched or uploaded (`.input.json`, the content store's
digest-named entry) and survive it; its bytes do not (§7a lists what that costs).

## 2. Decisions

- **One form per entry.** When the reader accepts the input, the entry keeps the copy and the
  original is deleted; when it does not (uneven spacing, an unreadable file), the entry keeps
  the original as fetched, so the job's error is the reader's own and nothing is lost.
- **Always on, never a request option.** The copy gives the network the same voxels and the
  same geometry as the original (§8), so it is not part of any result's identity, and a request
  option would split the cache over identical results. An operator may turn it off (§7).
- **Format: duckn, uncompressed, one chunk** (§4). Measured against raw NRRD and sharded duckn
  (§3): as fast as NRRD to read, the house format, and it carries the DICOM header.
- **The header is whatever SimpleITK gives.** No second DICOM reader (no pydicom reading
  files). The tags are SimpleITK's per-slice dictionaries with its defaults - public tags, no
  private ones.
- **Patient data is kept as SimpleITK gives it.** haversack is not a de-identifier and does not
  pretend to be one. Stripping, if it comes, is an option on what is SERVED, never on the cache.
- **Per-slice tags are stored,** in the slot duckn's DICOM extension defines for them, since it
  costs nothing to capture them (§5). Per-slice GEOMETRY is not written (§6).
- **The reader accepts what it accepts today** - uniform spacing, a gantry tilt up to the
  existing bound as a sheared direction. The copy records faithfully what the reader accepts; it
  does not widen it. Uneven spacing stays refused.

## 3. Measured

One CT, 709 x 768 x 768 int16 (836 MB); Modal CPU containers (8 cores) on a volume, and an M2.

| read of the input | warm | cold, fresh container | DICOM header on the image | voxels, geometry |
|---|---|---|---|---|
| DICOM series (today) | 13.1 s | 45 s | none (0 keys) | reference |
| raw NRRD | 0.21-0.44 s | 1.3-9.8 s | none | identical |
| duckn, one chunk, **mapped read** (§7) | **0.25 s** (M2: 0.15 s) | 2.2-4.3 s | 86 series tags | identical |
| duckn, one chunk, generic duckn reader | 0.62-0.67 s | 2.2-2.4 s | none | identical |
| duckn, 64^3 chunks in one shard | 0.95-1.18 s | 2.3-9.0 s | none | identical |
| duckn, `duckn.io.write` default chunks | 1.18 s | 5.1 s | none | identical |

- The DICOM time is the DECODE: 12.6 s on the container's local disk, and reorientation to
  canonical is 0.05 s.
- Reading the raw bytes of the 709 files cold from a freshly committed volume once took 278 s
  (0.39 s a file). A single file avoids the per-file cost entirely.
- Cold reads of freshly written files vary several-fold from run to run; compare formats on
  warm reads.
- Writing the copy: 1.1 s (after the decode it needs anyway).
- `zipfile.read` of the chunk was 1.08 s warm - it copies the member and CRCs every byte - which
  is why the reader maps it instead (§7).

## 4. The file

`<entry>/decoded/input.duckn.zip` - one zarr v3 array at the root of a zip, two members, both
`ZIP_STORED`:

```
input.duckn.zip
  zarr.json      a few KB (plus per-slice tags, ~120 KB for 709 slices)
  c/0/0/0        the voxels: shape[0]*shape[1]*shape[2]*itemsize bytes, C order
```

`zarr.json`:

| field | value |
|---|---|
| `zarr_format` / `node_type` | `3` / `"array"` |
| `shape` | `[Z, Y, X]`, as SimpleITK's array of the image |
| `data_type` | the image's pixel type as read (`int16` for most CT; `float32` when the reader rescaled to non-integer values) |
| `chunk_grid` | `regular`, `chunk_shape` equal to `shape` - exactly one chunk |
| `codecs` | `[{"name": "bytes", "configuration": {"endian": "little"}}]` - no compression, no other codec |
| `chunk_key_encoding` | `default`, separator `/` (so the chunk is `c/0/0/0`) |
| `fill_value` | `0` |
| `attributes.duckn` | §5 |

The values are the **calibrated** values the reader produced - no `value_transforms`. A copy is
the image the pipeline would have received, not a re-encoding of the stored DICOM integers.

## 5. duckn attributes

Built through duckn's own models and serializer (`from_sitk`, `duckn_attrs`), never by hand:

- `space`: `left-posterior-superior` - SimpleITK's own, so neither direction applies a flip.
- `space_origin`: the image origin (LPS, mm).
- `axes`: three `space` axes in array order z, y, x, `centering: cell`, `unit: mm`,
  `space_direction` = direction cosine x spacing - exactly the geometry `io.read_image`
  produced, including the IPP-derived override and a tilt's sheared direction.
- `extensions.dicom` (duckn `dicom-spec.md`) - DICOM series only:
  - `version`: the spec version written.
  - `tags`: the tags whose value is the same on every slice, keyed by PS3.6 keyword (private
    tags would be hex, but SimpleITK's defaults give none), values JSON-native per the spec's
    §4.2 (DS/IS as numbers, multi-valued tags as arrays by VM).
  - Tags the convention fields capture are left out (spec §2, §9): `ImagePositionPatient`,
    `ImageOrientationPatient`, `PixelSpacing`, `RescaleSlope`, `RescaleIntercept`,
    `RescaleType`, and the other geometry and value-mapping tags the spec lists.
  - `anonymized`: absent (not determined; haversack does not judge it).
- Per-slice tags, on the slice axis: `axes[0].samples[i].metadata.dicom`, one sample per
  slice in the volume's z order (the order the series reader stacks, which `read_image` uses),
  holding the tags that differ between slices. On the measured CT that is ten: instance number
  and SOP instance UID, instance and content time, slice location, tube current, exposure and
  two dose fields. `samples[i]` carries no `position` or `origin` (§6).
- `extensions.haversack`:

  ```json
  {"kind": "input_copy", "version": 1,
   "source": "idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c",
   "source_digest": "sha256-tree:…",
   "reader_version": 1,
   "reader": {"haversack": "0.14.0", "SimpleITK": "2.x"}}
  ```

  `source` and `source_digest` are the ORIGINAL's record (`.input.json`, or the content store's
  digest), and `source_files` / `source_bytes` its size - kept because the original is not
  (`GET /v1/inputs/{digest}` reports them); `version` is this document's format version, bumped
  when the file's meaning changes; `reader_version` is §7a's.

## 6. What the copy does not hold

- Per-slice geometry. The reader accepts a uniform grid (and a tilt as a sheared direction),
  which `axes` states exactly; `samples[i].position` / `origin` would state it a second time.
  Should the reader ever accept uneven spacing, the true positions go there.
- Sequences and binary tags: SimpleITK does not pass them through.
- Private tags: not asked for (SimpleITK's default).
- Anything the original was not: the copy never has more than the DICOM it came from.

## 7. Writing and reading

**Where.** In place of the original, in the entry that holds it, in both caches: the series
cache (hosted sources: `idc:`, `tcia:`, `s3:`, …) and the content store (uploads and digests) -
on the local server and on Modal. The content store's current raw-NRRD copy (`content.nrrd`,
`decode_for_fast_read`), which sits BESIDE its original, goes: one form per entry. The entry's
committed byte count is the copy's.

**When.** When the entry is stored - a fetch completing, an upload being put - before the
entry is committed. The original exists only inside that step, as the transcoder's input: an
implementation detail. Nothing downstream - segmenters, engines, `io.read_image`'s callers, the
read-ahead - is ever handed the original; a committed entry is the copy (or, for an input the
reader refuses, the original, which fails at read as it does today). **Whoever stores the entry
transcodes it:** a worker what it fetches, the api container what it is uploaded (on Modal ~13 s
of CPU for the measured series at upload, where it only hashed before - the submit waits for it,
as it waits for the upload's hash today). (The content store decodes
lazily today, so a preloaded input nobody runs pays nothing; now every stored input pays its
decode once, at ingest - 13 s for the measured CT.)

**Which inputs.** Image inputs that are a DICOM series, or a single file whose read
decompresses (`.nii.gz`, gzip-encoded NRRD). An input already raw and single (NRRD raw, MHA/MHD
raw, an uncompressed duckn) is kept as it is - it is its own efficient form. **Label-map inputs
are never converted** (a `result:` reference, an uploaded `.seg.nrrd`): their segment names and
codes live in the file.

**How it is written** (inside the writer's claim on the entry, before `.done`):

1. Record the original's facts first, as today: the digest of the fetched or uploaded bytes and
   the DICOM identifiers (`.input.json`); for an upload, the `expect` digest check.
2. Read the original with `io.read_image` - the reference - and, for a series, the series
   reader's per-slice dictionaries (`MetaDataDictionaryArrayUpdateOn()`; measured free). If the
   reader refuses the input, stop: the entry keeps the original.
3. Write `decoded/.input.duckn.zip.partial` (§4, §5).
4. Read it back with the mapped reader; compare the voxel digest, the geometry and the tags
   with step 2. Any difference: delete the copy, warn, and keep the original.
5. Rename the copy into place, THEN delete the original (`series/`), then commit the entry (on
   Modal, the volume commit under the lock the caches already use). A crash between the steps
   leaves an uncommitted entry, which the cache already treats as nobody's and refetches.

**How it is read.** The cache resolves an entry to whichever form it holds; `io.read_image`,
which every engine and door already goes through, reads a copy with the mapped reader when its
`extensions.haversack` says `input_copy` and its `reader_version` is current (§7a). The mapped
reader:

1. reads `zarr.json` and requires exactly §4: one regular chunk equal to the shape, the `bytes`
   codec alone, a stored member, a member size equal to the array's, no `value_transforms`, the
   geometry fields present. Anything else falls back to the generic path - never an error;
2. finds the chunk's data offset from its zip local header and maps it (`mmap` +
   `np.frombuffer`, read-only): no copy and no CRC until SimpleITK takes the buffer;
3. builds the image through duckn's `to_sitk` - the geometry conversion is duckn's, not ours;
4. sets the series-level tags on the image as `gggg|eeee` strings, as a single-file SimpleITK
   read would show them. Per-slice tags stay in the file; `input_copy.slice_tags(path, keyword)`
   returns one tag as a z-ordered list.

**Invalidation.** A refetch (`no-cache`, `refresh_input`) rewrites the entry. Eviction and
`cache clean` remove it. **Operator switch:** `HAVERSACK_INPUT_COPY=0` stores originals only -
the behavior before this change, for a host that wants the DICOM kept.
**Compression** - the default since 2026-09-25 (`HAVERSACK_INPUT_COPY_COMPRESSION`, `zstd`; `none`
for the uncompressed form; §13):
zstd level 3 through blosc with bit shuffling, in chunks of 32 whole slices, one zip member each, still a stored zip, format
version 2. It is read through zarr, whose codec pipeline decodes the chunks in parallel - no
mapping. Existing copies are not rewritten; a cache holds whichever form each entry was written
in, and the reader takes each by its layout. A reader that knows only version 1 sees a version-2
copy as stale and fetches again, never misreads it.

### 7a. What keeping one form costs

- **A reader fix cannot be re-applied from the cache.** The copy records `reader_version` (a
  haversack counter, bumped whenever `io.read_image` would produce different voxels, geometry
  or tags from the same bytes - the input-side analog of `CACHE_EPOCH`). A copy with another
  version is stale: a hosted input is refetched from its source; an UPLOAD is treated as
  evicted - the existing 410 `input_gone`, "send the bytes again" - because its bytes exist
  nowhere else.
- **What SimpleITK does not pass is gone from the cache:** sequences, private tags, binary
  values. A consumer that needs the full DICOM header refetches the series from its archive.
- **`get` cannot hand back the raw files from the cache**; asked for them, it refetches (hosted
  sources only).
- **The digest and DICOM facts are recorded, not re-derivable.** They were computed from the
  original when it was stored and are kept; nothing can re-check them against bytes later.
- **Disk:** one form, not two - 836 MB for the measured CT instead of 839 + 836.

**Never served.** The copy is internal. It is not an output and has no route.

## 8. What must be proven (tests)

- **Exactness, as for any geometry change.** Original vs copy: identical voxels, identical
  geometry (bit for bit on the measured CT - hold it to that, with a stated tolerance only if a
  case needs one), on: a real series; a synthetic slightly tilted series (the sheared direction
  must survive); a gzip NIfTI; a duckn input stored in RAS from elsewhere (the flip path our own
  copies never take).
- **Pipeline equality.** `segment()` labels from an input with and without its copy are
  byte-identical, and result keys do not move (the copy never reaches `result_key`).
- **The reader's refusals.** Each §7 requirement violated in turn falls back to the generic
  path; an all-zero volume whose chunk zarr did not write falls back; a truncated file is never
  read (the rename protocol) and a size mismatch falls back.
- **One form.** After a store the entry holds the copy and no `series/`; a refused or failed
  conversion leaves the original and no copy; a crash mid-way leaves an uncommitted entry.
- **Staleness.** A copy with another `reader_version` is not read: a hosted input is refetched,
  an upload answers 410 `input_gone`.
- **Tags.** Series-level tags round-trip to the image; per-slice tags come back in z order and
  in the spec's encoding; convention-captured tags are absent.
- **Modal.** The write, lock and commit on the worker; a copy written by one container read by
  another.

## 9. Changes outside haversack

In duckn (a release, then a pin bump here and in CI, with feldglas kept equal):

- `tags_from_sitk(per_slice_dicts)` → `(series_tags, per_slice_tags)` in `dicom-spec.md`'s
  encoding, beside the pydicom converter, so the spec's rules live in one repo. It uses
  `pydicom.datadict` only as a keyword/VR/VM table - no file is read - so `pydicom` joins the
  `dicom` extra there and haversack's `duckn` extra here.
- `from duckn import io` recurses forever in a fresh process: `__getattr__("io")` answers with
  `from . import io`, which asks `__getattr__` again. Use `importlib.import_module`.
- Reconcile `dicom-spec.md` §6.3 with the core convention and the model: §6.3 puts per-slice tags
  in `samples[i].extensions.dicom`, but `SampleMetadata` forbids unknown fields and has
  `metadata`, the core spec says per-sample data goes in `metadata` keyed by standard, and
  duckn's own converter writes `metadata.dicom` - which this copy follows. (Checked: samples
  carrying only `metadata` and `thickness` are read as uniformly spaced by `to_sitk`.)

## 10. Open

- Whether an input copy should ever be served - e.g. as a `get` output, where the copy is
  exactly what a client wants. Not for this change; if it is, patient-tag stripping comes with
  it (§2).

## 11. What building it changed (2026-09-25)

- **Per-slice tags are `samples[i].metadata.dicom`**, not `extensions.dicom` (§9, duckn's own
  converter and model).
- **Geometry within 1e-12, not bit for bit, for a tilted series.** duckn stores each axis as
  direction x spacing and a reader takes it apart again; for a sheared direction that is off by
  one unit in the last place (1.1e-16 measured - a sample moved ~1e-13 mm). An axis-aligned grid
  is exact. Voxels and pixel type are always identical. `input_copy.GEOMETRY_TOLERANCE`.
- **Slice Thickness, Pixel Spacing, Spacing Between Slices and Rescale/Modality LUT Type** are
  left out of `tags` (dicom-spec §2); thickness goes to the slice axis' `thickness` (per sample if
  it varies), Rescale Type to `sample_units`. An empty numeric tag is absent, never `null` (which
  would claim redaction).
- **The reader version is in a fetched entry's NAME** (`e<FETCH_EPOCH>.r<READER_VERSION>!<key>`),
  like the fetch epoch: a stale copy is never found and the input is fetched again; old entries
  age out by LRU. A content-addressed entry has no source: a stale copy reads as absent (410
  input_gone), and the same bytes uploaded again replace it. (The first version dropped stale
  entries through `SeriesCache.discard`, which `test_only_jobpolicy_decides_to_discard_a_cached_input`
  refused: that policy is jobpolicy's.)
- **Reading a copy needs neither duckn nor pydicom.** An engine image that cannot take the duckn
  extra (VoxTell's, MONAI's; SynthStrip's until its numpy<2 went, 2026-09-25) reads copies another container wrote: without duckn the one layout
  this module writes (LPS, z/y/x, direction x spacing) is converted directly - held bit for bit
  to duckn's `to_sitk` by a test, tilted series included - and without duckn or pydicom the tags
  are not restored (provenance, never needed to compute; since duckn 0.5.2 the converter is
  `duckn.dicom_tags`, so a duckn-free read restores none). WRITING needs both, so an environment
  without them keeps originals (`input_copy.enabled`). A compressed copy also needs zarr.
- **`GET /v1/inputs/{digest}`** reports the original's members and bytes (recorded in the copy)
  and adds `stored_form: "input_copy"` and `stored_bytes`.
- **The content store's raw-NRRD copy (`decode_for_fast_read`, `content.nrrd`) is gone**, and
  `ContentStore.fast_path` is `resolve`. On Modal the api image, the nnU-Net worker's and
  FastSurfer's carry the duckn extra (pydicom joins it).
- **Measured locally** on the 709-slice CT: transcode 9.5 s (a 6.4 s decode, the write, the
  read-back check), then `io.read_image` of the copy 0.145 s; 61 series tags on the image, 9
  per-slice tags on each of 709 samples.

## 12. Verified on Modal (2026-09-25)

Smoke `haversack-inputcopy-smoke` from `41e4636` (L40S, bearer token, optional engines off;
stopped, its three volumes, Dict and Secret deleted). 17 of 18 scripted checks passed; the one
failure is explained below and is not the copy's.

- **The read is gone from every job.** IDC `a05fb365-...` (709 slices, 418 M voxels):
  `read+canonical` 0.85 s on the first job, which fetched and transcoded the series, and 0.68 s
  on the next task over the same series - the earlier ranked smoke spent 11-14 s reading this
  series in EVERY job. A warm `ts.v2:total_fastest` job was 16 s wall end to end. A fresh
  DICOM read in the same worker took 12.5 s.
- **The worker's entry holds one form**: `e2.r1!idc%3A.../decoded/input.duckn.zip`, 836.8 MB,
  and no `series/`. An uploaded `CT_Abdo.nii.gz` answers `GET /v1/inputs/<digest>` with
  `stored_form: input_copy`, `bytes` 7,753,434 (the upload's own size), `stored_bytes`
  23,241,778; its labels lie on the input's grid, and a second task over it read in 0.15 s.
- **The copy is the image.** Inside the worker, the copy and a fresh fetch read by
  `io.read_image` from DICOM have the same voxel sha256 (`e29f1dcb...`), and equal origin,
  spacing, direction (compared as floats, not printed) and pixel type. `ts.v2:total_fast` run in
  that one container on each, and again on the copy, gave byte-identical labels.
- **The failed check:** this deployment's `total_fast` labels differ from an EARLIER
  deployment's (the ranked-jobs smoke) in 7,121 of 418 M voxels (0.0017 %), the same 7,121 on
  the first job and on two `no-cache` recomputes, while all three agree with each other exactly.
  With the input proven identical, that is the other deployment - another container and another
  branch's build - not the copy. Comparing labels across deployments is the wrong check for an
  input change; compare in one container.

## 13. Compressed copies (2026-09-25)

`HAVERSACK_INPUT_COPY_COMPRESSION=zstd` stores new copies as zstd level 3 - since the dataset
sweep below, through blosc with bit shuffling - in chunks of 32 whole slices, one member per chunk, in a stored zip; `extensions.haversack.version` is 2 (`FORMATS`),
so a version-1 reader treats it as stale rather than mapping it. The reader takes each file by its
layout, so a cache may hold both forms and the setting can change at any time. An unknown value
keeps the original and says why. Both input-copy variables are forwarded into Modal containers
(`HAVERSACK_INPUT_COPY=0` had never reached them).

**Chosen locally** (M2, 8 cores, the 709-slice CT, warm): one compressed chunk decodes on one
thread (zstd-3 2.4 s); whole-slice chunks decode in parallel through zarr - zstd-3 in 32-slice
chunks 0.64 s at 2.6x, blosc-zstd bitshuffle 0.84 s at 3.1x, gzip-6 1.1 s but an 8 s write. zstd
was first taken over blosc "for being a core zarr v3 codec every reader has" - but blosc is one
too, and the sweep across datasets below made blosc-zstd with bitshuffle the choice.

**Measured on Modal** with plain zstd, before the switch to blosc (L40S containers, 17 CPUs
each, torn down afterwards; locally the two read alike, so the read times carry over and the
compressed size is now 270 MB):

| | uncompressed (mapped) | zstd |
|---|---|---|
| file | 836.8 MB | 322.1 MB (2.6x) |
| transcode, written to a volume | 18.2 s | 14.9 s |
| read from `/dev/shm` (a worker's series cache), 12 containers | 0.24-0.43 s | 0.49-0.85 s |
| warm re-read from the volume | 0.24-0.43 s | 0.54-0.88 s |
| cold first read from the volume, imports warmed, 8 containers | 0.81, 0.97, 1.0, 1.4, 2.5, 8.8, 9.4, 16.5 s | 0.69, 1.0, 1.1, 1.1, 2.6, 6.3, 10.7, 11.8 s |
| `read+canonical` in a deployed worker's job | 0.85 s, 0.68 s | 1.0 s, 1.29 s |

- **Cold reads are the volume's latency, not its bytes**: the medians are equal (1.9 s) and both
  tails reach 10-17 s, in the same containers. A first round that did not warm the imports
  first seemed to favor zstd (medians 5.3 vs 1.5 s); with the imports warmed, that difference
  disappeared. Compression does not make a cold volume read faster.
- **What it buys is room**: the worker's `/dev/shm` series cache (`HAVERSACK_SHM_CACHE_GB`, 8 GiB by
  default) holds about 10 copies of this CT uncompressed and about 26 compressed, and the inputs
  volume holds 2.6x as many uploads (3.1x with blosc: about 32 copies in 8 GiB).
- **What it costs**: about 0.3-0.5 s a read on this CT, against the 12.5 s DICOM decode either form
  replaces.
- **Exactness**: every read, in every container, hashed to the DICOM read's voxels
  (`e29f1dcb...`), with the 61 series tags restored; the deployed worker's own zstd copy in
  `/dev/shm` held the same hash and no `series/` beside it. The deployment's labels differ from
  the uncompressed deployment's in 7,858 of 418 M voxels - across deployments, as §12 found for
  two uncompressed ones; the input is proven identical.
- **Tests**: 7 new in `test_input_copy.py`, the knob forwarding in `test_modal_app.py`, the status
  field in `test_serve.py`. 11 of 11 mutants killed (the last, a deflated zip accepted, by a test
  added for it); a comment mutant survived as it must. Fast suite 2609 passed / 4 skipped.
  After the switch to blosc: 12 of 12 (bitshuffle held by the layout test). A first reader rule
  that required blosc's cname to be zstd let two mutants survive - it was redundant, since a blosc
  chunk either decodes exactly through zarr or fails into NotACopy - so it was removed, and the
  case of a blosc codec over raw bytes now holds the decode-failure path.

**Across datasets (local, 2026-09-25).** The CT above was not aberrant; it was the least
compressible case. M2, 8 cores, warm page cache, reads the median of 5, every read voxel-exact:

| dataset | array | mapped read | zstd3/32 (first choice) | blosc-zstd3 bitshuffle/32 (shipped) |
|---|---|---|---|---|
| idc-torso1 CT (DICOM) | 709x768x768 int16, 836 MB | 0.15 s | 2.60x, 0.36 s | 3.10x, 0.36 s |
| NLST low-dose chest CT (DICOM) | 249x512x512 int32, 261 MB | 0.05 s | 3.73x, 0.10 s | 4.87x, 0.11 s |
| C3N-00704 CTPA 0.625 mm (DICOM) | 418x512x512 int32, 438 MB | 0.08 s | 3.45x, 0.18 s | 4.48x, 0.18 s |
| MSB-02664 CT (DICOM) | 409x512x512 int32, 429 MB | 0.07 s | 3.58x, 0.19 s | 4.67x, 0.17 s |
| ds000114 T1 MR (.nii.gz) | 256x156x256 float32, 41 MB | 0.005 s | 4.96x, 0.015 s | 5.84x, 0.017 s |
| ct_RAS CT (.nii.gz) | 165x512x512 float32, 173 MB | 0.03 s | 2.78x, 0.08 s | 3.40x, 0.09 s |
| Visible Human male CT (raw .nii, not copied) | 834x512x512 float32, 875 MB | - | 4.10x, 0.36 s | 5.12x, 0.36 s |

- Chunk size (16, 32, 64 slices) moves neither size nor read time materially; 64 was slightly
  slower to read on some sets. zstd level 1 is 5-7 % larger, level 6 4-8 % smaller at twice
  the write; neither changes the read.
- blosc-zstd with bitshuffle was 1.2-1.4x smaller than plain zstd on EVERY set, at the same read
  and write time. blosc is one of zarr v3's core codecs as zstd is, so the reason given above
  for preferring zstd does not hold: the compressed form was switched to it the same day, before
  it had been deployed anywhere (the format version stays 2; no stored copy was plain zstd).
- Three of the four DICOM CTs read as int32 (a rescale SimpleITK widens), so their copies carry
  twice the bytes an int16 would; compression absorbs most of that (3.5-3.7x). Narrowing the type
  would change the image the reader produces, which a copy must not do.
- The compressed read costs 2-3x the mapped one here, which in seconds is 0.01-0.2 s on these
  sets and 0.2 s on the large CT; on Modal the same CT's difference was 0.3-0.5 s.

**Every engine image can read a compressed copy (2026-09-25).** Reading one needs zarr, which the
SynthStrip image could not take while synthstrip-torch pinned numpy<2. synthstrip-torch 0.1.1
drops the cap (surfa at upstream's unreleased NumPy 2 fix, `8aa4a5f6`; on CPU, bit-identical to
numpy 1.26 with surfa 0.6.3), and the SynthStrip image takes the `duckn` extra. Smoke
`haversack-sst-smoke` (L40S, compression on; torn down): the worker ran numpy 2.5.3, surfa at the
fix, synthstrip-torch 0.1.1, zarr 3.4.0; the ds000114 T1 from OpenNeuro and the same T1 uploaded
(stored compressed, 7.0 MB) each computed, and their masks agreed voxel for voxel - 1,282 mL, the
figure the 2026-09-12 smoke gave - and with the CPU numpy<2 mask to Dice 0.99999 (16 voxels).
VoxTell's and MONAI's images still lack zarr: with compression on, an upload they read must be
stored uncompressed, or they must take zarr first.

**Compressed by default (2026-09-25, the user's decision).** The default became `zstd` - the
blosc-zstd bitshuffle form this section chose - because the measurements above favor it wherever
room matters and cost ~0.3-0.9 s a read against the 12-13 s decode it replaces either way.
`none` stays for a host that wants the mapped read, or runs VoxTell or MONAI: their images lack
zarr, so they cannot read a compressed copy (both are experimental and opt-in).
