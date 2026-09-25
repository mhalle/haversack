# The input copy: a decoded, single-file form of a cached input

*Specification, 2026-09-25. Nothing here is built yet. The decisions in §2 are the user's; the
numbers in §3 were measured the same day (Modal and an M2, one CT).*

## 1. What it is for

A job reads its input every time it runs. For a DICOM series that is a full decode of every
slice, whether the same series was read a minute earlier or not: on the 709-slice CPTAC-CCRCC CT
`idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c` the decode is 13 s on a Modal worker - most of a warm
`ts.v2:total_fast` job - and a fresh container reading a series another container cached paid
45 s. The input copy is the input decoded once, kept beside the original in the cache that
already holds it, and read instead of it from then on.

It is **pure cache**. It is derived from the original, never replaces it, never changes what a
job computes, and losing it costs one decode and never an answer. The original stays because
the recorded digest (`.input.json`) and every DICOM fact are about the original's bytes.

## 2. Decisions

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
- Per-slice tags, on the slice axis: `axes[0].samples[i].extensions.dicom`, one sample per
  slice in the volume's z order (the order the series reader stacks, which `read_image` uses),
  holding the tags that differ between slices. On the measured CT that is ten: instance number
  and SOP instance UID, instance and content time, slice location, tube current, exposure and
  two dose fields. `samples[i]` carries no `position` or `origin` (§6).
- `extensions.haversack`:

  ```json
  {"kind": "input_copy", "version": 1,
   "source": "idc:a05fb365-dfd2-4116-ab8e-a7262d2c169c",
   "source_digest": "sha256-tree:…",
   "reader": {"haversack": "0.14.0", "SimpleITK": "2.x"}}
  ```

  `source` and `source_digest` are the entry's own record (`.input.json`, or the content store's
  digest); `version` is this document's format version, bumped when the file's meaning changes.

## 6. What the copy does not hold

- Per-slice geometry. The reader accepts a uniform grid (and a tilt as a sheared direction),
  which `axes` states exactly; `samples[i].position` / `origin` would state it a second time.
  Should the reader ever accept uneven spacing, the true positions go there.
- Sequences and binary tags: SimpleITK does not pass them through.
- Private tags: not asked for (SimpleITK's default).
- Anything the original was not: the copy never has more than the DICOM it came from.

## 7. Writing and reading

**Where.** Beside the original, in the entry that holds it, in both caches: the series cache
(hosted sources: `idc:`, `tcia:`, `s3:`, …) and the content store (uploads and digests) - on the
local server and on Modal. The content store's existing raw-NRRD copy (`content.nrrd`,
`decode_for_fast_read`) is replaced by this one, so there is one form. The copy counts toward
its cache's byte budget (as `ContentStore._restamp` does now) and is evicted with its entry.

**When.** Lazily, on the first read of an entry, as the content store does now: a preloaded
input nobody runs never pays for a decode.

**Which inputs.** DICOM series, and single files whose read decompresses (`.nii.gz`,
gzip-encoded NRRD). An input that is already raw and single (NRRD raw, MHA/MHD raw, an
uncompressed duckn) is read as it is.

**How it is written.**

1. Read the original with `io.read_image` - the reference - and, for a series, the series
   reader's per-slice dictionaries (`MetaDataDictionaryArrayUpdateOn()`; measured free).
2. Write `decoded/.input.duckn.zip.partial` (§4, §5).
3. Read it back with the mapped reader; compare the voxel digest, the geometry and the tags
   with step 1. Any difference: delete it, warn once, keep reading the original.
4. Rename into place; on Modal, commit under the volume lock the caches already use.

**How it is read.** Inside `io.read_image`, which every engine and door already goes through:
the copy is used when it exists AND its `extensions.haversack.source_digest` equals the entry's
current digest; otherwise the original is read (and the copy rewritten). The mapped reader:

1. reads `zarr.json` and requires exactly §4: one regular chunk equal to the shape, the `bytes`
   codec alone, a stored member, a member size equal to the array's, no `value_transforms`, the
   geometry fields present. Anything else falls back to the generic path - never an error;
2. finds the chunk's data offset from its zip local header and maps it (`mmap` +
   `np.frombuffer`, read-only): no copy and no CRC until SimpleITK takes the buffer;
3. builds the image through duckn's `to_sitk` - the geometry conversion is duckn's, not ours;
4. sets the series-level tags on the image as `gggg|eeee` strings, as a single-file SimpleITK
   read would show them. Per-slice tags stay in the file; `input_copy.slice_tags(path, keyword)`
   returns one tag as a z-ordered list.

**Invalidation.** A refetch (`no-cache`, `refresh_input`) rewrites the entry and so the copy; a
digest mismatch is caught at read (above). Eviction and `cache clean` remove the entry with its
copy. **Operator switch:** `HAVERSACK_INPUT_COPY=0` reads originals only (a disk-bound host).

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
- **Binding.** A copy whose `source_digest` is not the entry's is not used.
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
- Confirm (or fix) that an axis whose `samples` carry only `extensions` - no `position` or
  `origin` - is still read as uniformly spaced by `to_sitk` and the readers.

## 10. Open

- Whether the Modal content store's copy is written by the api container (which stores uploads)
  or by the worker on first read. Lazy-on-read puts it on the worker, under the worker's lock.
- Whether an input copy should ever be served - e.g. as a `get` output, where the copy is
  exactly what a client wants. Not for this change; if it is, patient-tag stripping comes with
  it (§2).
