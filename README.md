# haversack

`haversack` runs nnU-Net-family segmentation models on PyTorch - TotalSegmentator, MOOSE,
MRSegmentator, and any stock nnU-Net v2 model folder - on an Apple Silicon GPU (MPS), a CUDA
card, or the CPU. It is a library with a command line and a small REST server on top.

This page is what a new user needs to run it. The design record is in `docs/` and the
workspace it lives in; the API is documented in the docstrings.

## Requirements

- **Apple Silicon** (M1 or later) or a CUDA machine. There are no PyTorch wheels for Intel
  Macs anymore, so an Intel Mac cannot run it.
- **Python 3.12 to 3.14** (3.12 is what the tests run on).
- **Memory:** 16 GB runs every task. The 3 mm `total_fast` task is comfortable; whole-body
  `total` at 1.5 mm fits (the sliding-window accumulator moves to host memory when the GPU
  budget is short) but takes about 25 minutes on an M2. More memory means the accumulator
  stays on the GPU and the run is faster; nothing is hard-coded to a laptop.
- About **1.1 GB** of packages (torch is most of it), plus model weights: 160 MB for
  `total_fast`, 1.1 GB for MRSegmentator, a few hundred MB per TotalSegmentator part.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "haversack @ git+https://github.com/mhalle/haversack"
```

`pip install` of the same URL works too for the bare install and the extras that resolve
from PyPI (`serve`, `remote`, `modal`); the extras whose packages come from git (`duckn`,
`fastsurfer`, `synthstrip`, `voxtell`) need `uv pip install`, which reads the sources this
project declares. Add `serve` to the extras for the local server, `remote` for the client:

```bash
uv pip install "haversack[serve,remote] @ git+https://github.com/mhalle/haversack"
```

For a command-line user, a tool install puts `haversack` on PATH in its own environment, away
from any project:

```bash
uv tool install --python 3.12 "haversack[serve,remote] @ git+https://github.com/mhalle/haversack"
```

and `uvx git+https://github.com/mhalle/haversack segment ...` runs it without installing
anything (`uvx --from "haversack[serve,remote] @ git+..." haversack ...` for the extras).

Check it:

```bash
haversack tasks
```

**Lean install: the library, the client, the Modal operator.** The inference stack (torch,
nnunetv2, scipy, scikit-image, about 1 GB) is part of the package, because a segmentation
tool that cannot segment after `pip install` is a trap. But `import haversack` is torch-free
(everything heavy imports lazily), so a client-only, describe-only or deploy-only install can
skip it. Extras can only add, so this is a two-line `--no-deps` recipe, about 200 MB, most of
it SimpleITK:

```bash
uv pip install --no-deps "haversack @ git+https://github.com/mhalle/haversack"
uv pip install numpy SimpleITK pydantic typer tqdm httpx obstore   # add `modal` to deploy the server
```

That environment runs `haversack tasks`, `haversack remote ...`, `haversack modal deploy`, and
`from haversack import RemoteClient` (or `Segmenter.describe`). Asked to segment, it says so
in one line and names what to add.

lists every task the catalog knows, with the engine, modality, and whether its weights are
already on disk. `haversack tasks --installed` shows what will run without a download, and
`haversack tasks total_fast` prints the structures a task produces, one per line. This
guide itself ships with the package: `haversack docs` prints it, `haversack docs weights`
one section, and every command answers `--help` with its options, defaults and examples.

## Weights

Weights live under one root, shared with TotalSegmentator's own installation so the two never
download the same model twice:

| Source | Location |
|---|---|
| default | `~/.totalsegmentator/nnunet/results` |
| environment | `TOTALSEG_WEIGHTS_PATH`, then `nnUNet_results` |
| command line | `--model-root DIR` |

They download on first use: TotalSegmentator models from the TotalSegmentator GitHub releases
(with the sha256 the manifest records), MOOSE and MRSegmentator from their own hosting. To
provision ahead of time:

```bash
haversack weights fetch total          # every part the task needs
haversack weights coverage             # what the manifest can provision, and what it cannot
```

`refresh` reads TotalSegmentator's GitHub releases and records new datasets and versions, so
you need not wait for a haversack release to use newly published weights. From an installed
package it writes your own manifest, `~/.config/haversack/ts_weights.json`
(`HAVERSACK_TS_MANIFEST` to move it), which is laid over the packaged one on every read and
survives upgrades; in a source checkout it edits the repository's file. `--dry-run` reports
without writing, and existing entries are never repointed at a newer release unless you say
`--update-existing`, because that changes the segmentations.

`coverage` marks the TotalSegmentator tasks whose weights are behind TotalSegmentator's
commercial license (`appendicular_bones`, `brain_structures`, `coronary_arteries`,
`heartchambers_highres`, `tissue_types`, ...). haversack does not handle that license: install those
with TotalSegmentator's own `totalseg_set_license` flow into the same root and haversack will find
them.

## Segment from the command line

```bash
haversack segment scan.nii.gz --task total_fast -o labels.nii.gz
```

The input is a NIfTI, NRRD, or MetaImage file, or a DICOM series directory - local, or a
remote source fetched on demand (see [Remote inputs](#remote-inputs)):

```bash
haversack segment idc:<crdc_series_uuid> --task total_fast -o labels.seg.nrrd
haversack segment "zenodo:<recid>/amos22.zip!amos22/imagesVa/amos_0575.nii.gz" --task mrsegmentator:base -o amos.seg.nrrd
haversack segment https://example.org/scan.nii.gz --task total -o labels.nii.gz
```

**Batch** - pass several inputs to segment them all into one output directory:

```bash
haversack segment a.nii.gz b.nii.gz dicom_dir/ --task total_fast --format seg.nrrd -o out/
```

With more than one input, `-o` is a directory (default: the current directory) and `--format`
names the output type; each result is written as `<name>_<task>.<ext>`, where `<name>` is the
input's filename stem (or the identifier for a remote source), so an output never overwrites
its input. If one input fails it is reported and the run exits non-zero, but the others still
complete. A single input keeps the plain form (`-o` the file, its extension picking the format).

Output format follows the extension
(`.nii.gz`, `.nrrd`, `.seg.nrrd`, `.mha`); labels come back on the input grid, in the input's
orientation. Task names are `ecosystem:task`, and a bare name is looked up across ecosystems
(`total_fast` is `ts:total_fast`).

`total_fast` is the 3 mm whole-body model; `total_fastest` is the 6 mm one (coarser, faster
still), and `total` runs the five 1.5 mm models. Useful options:

| Option | Meaning |
|---|---|
| `--spacing 1.0` | isotropic output spacing in mm instead of the input grid |
| `--interp nearest` | TotalSegmentator's label semantics; the default `linear` gives sub-voxel boundaries from the logits |
| `--device mps|cuda|cpu` | default `auto` |
| `--dtype fp16|bf16|fp32` | default `fp16` (the network runs fp16 on MPS) |
| `--envelope 20` | restrict inference to the body plus this margin in mm; `0` for the whole volume |
| `--accumulate device|host` | force the sliding-window accumulator's placement; `auto` decides from free memory |

What to expect on an M2 for `total_fast` on a 709 x 768 x 768 chest CT, one run per process:

| Stage | First run after install | Later runs |
|---|---|---|
| `read+canonical` (read + orientation) | 8 s | 8 s |
| model load (checkpoint, architecture, GPU upload) | 38 s | 5 s |
| network (8 patches, fp16 MPS) | 19 s | 15 s |
| total | 68 s | 30 s |

The first run after an install pays about 30 s once for compiling and caching (torch's MPS
kernels, bytecode); the second run in the same environment loads the model in 5 s. The Python
API and the server keep models resident, so a second case in the same process pays only the
read and the network.

## Python API

```python
from haversack import segment, Segmenter

r = segment("scan.nii.gz", "total_fast")        # a Segmentation
r.save("labels.nii.gz")
liver = r.mask("liver")                          # boolean array (Z, Y, X) on the output grid
r.present()                                      # {label: name} for what was found, e.g. {5: "liver", ...}
r.volumes_ml()                                   # {name: millilitres}, e.g. {"liver": 1424.3, ...}
r.timings, r.provenance                          # per-stage seconds; what ran, with what, and any deviations

seg = Segmenter(cache_models=5)                  # models stay warm across calls
for path in paths:
    seg.segment(path, "total").save(path.with_suffix(".labels.nii.gz"))
job = seg.submit("scan.nii.gz", "total", on_progress=print)   # off-thread, cancellable
```

`segment()` takes the same options as the command line as keyword arguments (`grid=1.0`,
`interp="nearest"`, `device="mps"`, `envelope_mm=20`, ...). A stock nnU-Net model folder
works as a task: `segment("scan.nii.gz", "/path/to/Dataset123_x/nnUNetTrainer__nnUNetPlans__3d_fullres")`.
Errors are one family, `haversack.HaversackError` (`InputError`, `ModelNotFound`,
`UnsupportedModel`, `ResourceError`, `Cancelled`).

## Remote inputs

Both `segment` and `get` accept remote sources anywhere a path would go. A source is fetched
once into the input cache (`~/.cache/haversack/inputs`) and reused, so segmenting and getting
the same identifier - or the same identifier twice - downloads it only once.

| Prefix | Identifier | Example |
|---|---|---|
| `idc:` | an NCI Imaging Data Commons `crdc_series_uuid` (a DICOM series) | `idc:4682f41a-65d7-4a7b-8050-952f73abb746` |
| `zenodo:` | `<record id>/<filename>` on a Zenodo record | `zenodo:7262581/amos22.zip` |
| `tcia:` | a TCIA series | `tcia:<series-uid>` |
| `openneuro:` | an OpenNeuro dataset file | `openneuro:ds000114/.../sub-01_T1w.nii.gz` |
| `hf:` | a file in a Hugging Face repo | `hf:<org>/<repo>/<path>` |
| `http://`, `https://` | any URL (command line only, never a server) | `https://example.org/scan.nii.gz` |

Add `!member` to read one file out of a remote **zip** by HTTP range, without downloading the
archive - `zenodo:7262581/amos22.zip!amos22/imagesVa/amos_0575.nii.gz`. The trailing-slash
form (`...zip!amos22/imagesVa/`) extracts every member under a prefix.

`idc:` needs the `idc` extra's runtime (`obstore`), which is part of the normal install; the
others use the standard library. The hosted prefixes are exactly the sources a `haversack
serve` accepts; bare URLs are local-only, because a server must not be pointed at arbitrary
hosts.

## Getting data with `get`

`haversack get` acquires a source without segmenting it - when you want the images themselves,
not only the labels.

```bash
haversack get idc:<crdc_series_uuid>                          # into the cache; prints the path
haversack get idc:<crdc_series_uuid> -o case1/scan.nii.gz     # the DICOM series as one NIfTI
haversack get idc:<crdc_series_uuid> --format nrrd -o out/    # converted, auto-named <id>.nrrd
haversack get idc:<crdc_series_uuid> -o raw_dicom/            # the raw DICOM series directory
```

What `-o` and `--format` do, in order:

| Given | Result |
|---|---|
| no `-o` | fetched into the cache; the cache path is printed (a later `segment` of the same id reuses it) |
| `-o` a **directory** (or a path ending `/`) | the raw fetched content copied in - a DICOM series stays a directory of files, named by the source |
| `-o` a **file** with an image extension | read and written as that one volume (a DICOM series collapses to a single NIfTI/NRRD), geometry preserved |
| `--format <type>` | convert to that type (`nifti`, `nrrd`, `seg.nrrd`, `mha`); with `-o` a directory the file is auto-named `<source>.<ext>` |
| `--no-cache` | fetch straight to `-o` and leave nothing in the cache (for a large one-off); requires `-o` |

**Batch** - pass several sources at once:

```bash
haversack get idc:<uuid-a> idc:<uuid-b> --format nrrd -o out/   # convert each -> out/<id>.nrrd
haversack get idc:<uuid-a> idc:<uuid-b>                         # cache each, print the paths
haversack get idc:<uuid-a> idc:<uuid-b> -o raw/                 # raw-copy each into raw/
```

With more than one source, `-o` is a directory (default: the current directory); `--format`
converts each into it, and without `--format` each is raw-copied. As with `segment`, a failing
source is reported and the run exits non-zero without stopping the rest.

## Managing the cache and weights

`haversack cache` shows and clears what haversack keeps on disk; `haversack weights` handles
the models.

```bash
haversack cache list                       # every store, its location and size
haversack cache path                       # just the locations, one per line
haversack cache clean inputs               # a dry run (prints what would go)
haversack cache clean inputs --yes         # actually delete the fetched inputs
haversack cache clean inputs idc:<uuid>    # drop one cached input by its spec
haversack cache clean all --older-than 30d --yes   # inputs + results + checkpoints, aged out
haversack weights list                     # installed models, versions, sizes
haversack weights remove 297               # delete one dataset's weights
```

The stores:

| Store | Location | Swept by `cache clean` |
|---|---|---|
| `inputs` | `~/.cache/haversack/inputs` (fetched sources) | yes |
| `results` | `~/.cache/haversack/results` (server result cache) | yes |
| `checkpoints` | `~/.cache/haversack/fastsurfer-checkpoints` | yes |
| `serve` | `~/.cache/haversack/serve` (generated server tokens) | **no** - a running server owns its file |
| `weights` | `~/.totalsegmentator/nnunet/results` (models) | **no** - use `weights remove` |

`cache clean` is a dry run unless you pass `--yes`, takes `--older-than <n>d|h|w` to keep
recent entries, and never touches model weights - those are slow to re-download and
license-gated ones you installed by hand are not re-fetchable, so `weights remove` deletes
them one dataset at a time. `HAVERSACK_CACHE_DIR` moves the `~/.cache/haversack` root;
`--model-root` (or `TOTALSEG_WEIGHTS_PATH`) sets the weights location.

## When a run adapts to the machine

haversack chooses placements and precisions from the machine's memory, but never silently.
Anything that ran differently from what you asked - the SynthStrip net rerun in fp16 after
a real out-of-memory on MPS, FastSurfer's aggregation field kept in CPU memory, the nnU-Net
accumulator on the host when you asked for the device - is a `note:` line at the end of a
command-line run, a progress stage on the server, and a record in the result's provenance
(`deviations`, in the seg.nrrd header and the job's result payload), each with what was
asked, what ran, and why. Explicit choices are explicit: `--device cpu` is never overridden,
and `--dtype fp32` on the nnU-Net path is never lowered.

## Local server

The server is the same job protocol haversack deploys on Modal, run on the machine itself:

```bash
haversack serve --port 8790
```

It builds a `Segmenter` with warm models (`--cache-models 5` by default, enough for a whole
`total` union), queues jobs, streams progress, and keeps a durable result cache under
`~/.cache/haversack/results`. Computation needs a bearer token and reads never do. Given
none, the server generates one, prints it, and leaves it in a file only you can read, which
`haversack remote` on the same machine picks up by itself when it talks to a loopback address
(`127.0.0.1`, `localhost`) - so personal use has no ceremony,
and a proxy or tunnel in front of the server still faces a token. (`--no-token` runs open,
with no protection of any kind; it is for a machine you trust end to end.) To reach it from other
machines, bind it to a network interface with a token of your choosing, `--host 0.0.0.0
--token <secret>`, and pass that to `remote --token`. The client:

```bash
export HAVERSACK_SERVER=http://127.0.0.1:8790
haversack remote tasks
haversack remote submit scan.nii.gz --task total_fast -o labels.seg.nrrd
```

`submit` uploads, shows progress, and downloads the labels; `--no-wait` returns a job id for
`status`, `fetch`, and `cancel`. The endpoints are under `/v1/` (`/v1/health`, `/v1/tasks`,
`/v1/jobs`); the OpenAPI document is at `/docs`. The server is ready when `GET /v1/health`
answers; stop it with Ctrl-C (or kill the process) - queued jobs are kept in its `jobs.db`
and re-queued when it starts again. On this M2 a `total_fast` job through the
server produced labels voxel-identical to the command line's.

That is the whole of it for a single machine. The server has its own guide for the rest -
the rules that decide reads and computes, results addressed by what was segmented, the
hosted sources, the caches, and deploying to Modal: `haversack docs --server`.

## Engines: FastSurfer and friends

Not everything is an nnU-Net checkpoint. FastSurfer (whole-brain parcellation from a T1),
SynthStrip, VoxTell and the MONAI bundles are *engines*: other networks behind the same
tasks, API, CLI and server. An installed runtime is the switch: `haversack tasks` shows an
engine's task as installed when its package is importable, and refuses it otherwise with
the install line.

FastSurfer installs into the main environment, and its checkpoints (66 MB, DOI-versioned)
download once from Zenodo into `~/.cache/haversack/fastsurfer-checkpoints`:

```bash
uv pip install "haversack[fastsurfer] @ git+https://github.com/mhalle/haversack"   # or --extra fastsurfer with uv sync
haversack segment t1.nii.gz --task fastsurfer:brain -o brain.seg.nrrd
```

SynthStrip's dependencies pin numpy below 2, so it owns a separate environment:

```bash
UV_PROJECT_ENVIRONMENT=.venvs/synthstrip uv sync --extra synthstrip --extra serve
.venvs/synthstrip/bin/haversack segment t1.nii.gz --task synthstrip:mask -o mask.seg.nrrd
```

FastSurfer and SynthStrip have in-process runners; VoxTell and the MONAI bundles run on
Modal. FastSurfer's
view-aggregation field is large (2.6 GB in half precision at 1 mm), so on Apple Silicon it
stays in CPU memory unless the machine has 32 GB or more; 16 GB is tight for it. On Apple
Silicon haversack caps PyTorch's MPS allocator at the device's recommended working set,
because past it Metal returns zeros instead of an error; a real shortfall then raises, and
SynthStrip retries in fp16 before refusing.

## What haversack does not do yet

- Multi-channel nnU-Net inputs, region (sigmoid) heads, and the `3d_lowres`, cascade and `2d`
  configurations are not on the nnU-Net path.
- VoxTell and the MONAI bundles have no in-process runner yet; they run on Modal.
  FastSurfer and SynthStrip run locally from their own environments (above).
- Versioning: `haversack.__version__` is haversack's own number and is what the server reports; the
  distribution's version belongs to the repository as a whole.
