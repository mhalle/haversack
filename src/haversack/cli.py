"""haversack segment IN --task total_fast -o OUT [--spacing 1.0] [--interp nearest|linear]"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    """Run the CLI. haversack's own errors (bad input, missing weights, a missing extra) are
    reported as one line on stderr with exit status 2; a traceback is for bugs."""
    from .errors import HaversackError
    try:
        return _run(argv)
    except HaversackError as e:
        print(f"haversack: {e}", file=sys.stderr)
        return 2
    except LookupError as e:
        # the catalogs answer an unknown or ambiguous task name with a bare LookupError
        # (never a KeyError - those are bugs and keep their traceback)
        if type(e) is not LookupError:
            raise
        print(f"haversack: {e}", file=sys.stderr)
        return 2


INSTALL_HINT = ('this is a lean install (--no-deps); a normal install has them: '
                'uv pip install torch nnunetv2 scipy scikit-image, or reinstall with '
                'uv tool install "haversack @ git+https://github.com/mhalle/haversack"')



STORE_EXTRA_HINT = ("the ranked store needs the duckn extra: uv sync --extra duckn "
                    "(or uv pip install 'haversack[duckn]')")


def _need_store_extra():
    """The three packages of the ranked-store extra, checked by name before anything is
    imported or computed - a missing one is a one-line answer, not a traceback after the
    network has run."""
    import importlib.util

    def absent(name):
        try:
            return importlib.util.find_spec(name) is None
        except ModuleNotFoundError:          # a finder that refuses the name outright
            return True
    missing = [n for n in ("rankfield", "zarr", "duckn") if absent(n)]
    if missing:
        from .errors import InputError
        raise InputError(f"{STORE_EXTRA_HINT} - missing {', '.join(missing)}")


def _need_inference_stack(task=None) -> None:
    """Refuse early, with the install line, when this environment cannot run ``task``.

    The inference stack is core, so this fires only on a lean install (``--no-deps``: the
    client and a describe-only front end never pay for torch) - and then it must say what to
    do rather than trace back from `import torch`. An engine task (``fastsurfer:brain``)
    needs that engine's runtime, which lives in its own environment; ``task=None`` (the
    server) needs only torch - it checks per task at run time.
    """
    import importlib.util
    from .engines import registry
    from .errors import InputError
    eng = registry.engine_for_task(str(task)) if task is not None else None
    if eng is None:
        need = ["torch"]
    elif eng.name == registry.NNUNETV2:
        need = ["torch", "nnunetv2", "scipy"]
    else:
        need = ["torch", eng.runtime_module]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if eng is not None and eng.name != registry.NNUNETV2:
        raise InputError(f"{task} runs on the {eng.name} engine ({', '.join(missing)} not installed), "
                         f"which has its own environment: UV_PROJECT_ENVIRONMENT=.venvs/{eng.name} "
                         f"uv sync --extra {eng.extra} --extra serve, then run haversack from it")
    raise InputError(f"segmenting needs {', '.join(missing)}, not installed here: {INSTALL_HINT}")


GUIDES = {"user": ("data/GUIDE.md", "README.md"), "server": ("data/SERVER.md", "SERVER.md")}


def _installed_datasets(root) -> list:
    """Every installed nnU-Net dataset folder under a weights root.

    Two levels, because that is how the weights are actually laid out: TS and
    stock models sit directly under the root, while every ecosystem catalog
    (moose, mrsegmentator, dentalsegmentator, totalvibe) installs under its own
    ``<root>/<ecosystem>/`` bucket. Scanning only the top level - which is what
    this did - reported "0 dataset(s)" for a root holding a working catalog
    install, and left no way to remove one.
    """
    import re as _re
    from pathlib import Path
    root = Path(root).expanduser()

    def real_children(d):
        """Entries that are really inside ``d`` - a symlink is not descended.

        This list feeds `weights remove`, which rmtree's what it matches, and
        rmtree follows a symlinked *final component* through to real data. A
        `mybackup -> /elsewhere` link under the weights root would otherwise put
        /elsewhere within reach of `weights remove <id>`.
        """
        try:
            return [c for c in sorted(d.iterdir()) if not c.is_symlink() and c.is_dir()]
        except OSError:
            return []                      # unreadable: not ours to list, never fatal

    if not root.is_dir():
        return []
    is_dataset = _re.compile(r"Dataset\d+").match
    out = [c for c in real_children(root) if is_dataset(c.name)]
    for sub in real_children(root):        # one level of ecosystem buckets
        if not sub.name.startswith(".") and not is_dataset(sub.name):
            out.extend(c for c in real_children(sub) if is_dataset(c.name))
    return out


def guide_text(which: str = "user") -> str:
    """A guide's Markdown: the file shipped inside the wheel, or, in a checkout (editable
    install), the repository's own file - one source of truth. ``which`` is ``user`` (the
    README) or ``server`` (SERVER.md, the job server and its deployment)."""
    from importlib.resources import files
    shipped_name, local_name = GUIDES[which]
    shipped = files("haversack").joinpath(shipped_name)
    if shipped.is_file():
        return shipped.read_text()
    from pathlib import Path
    local = Path(__file__).resolve().parents[2] / local_name
    if local.is_file():
        return local.read_text()
    raise RuntimeError(f"the {which} guide is missing from this installation")


def user_guide() -> str:
    return guide_text("user")


def guide_sections(text: str) -> list:
    """``[(heading, body)]`` for every ``## `` section of the guide, in order."""
    out, head, buf = [], None, []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            if head is not None:
                out.append((head, "".join(buf)))
            head, buf = line[3:].strip(), [line]
        elif head is not None:
            buf.append(line)
    if head is not None:
        out.append((head, "".join(buf)))
    return out


def _docs(topic, list_sections: bool, which: str = "user") -> int:
    text = guide_text(which)
    if list_sections:
        for head, _ in guide_sections(text):
            print(head)
        return 0
    if topic is None:
        sys.stdout.write(text)
        return 0
    hits = [(h, b) for h, b in guide_sections(text) if topic.lower() in h.lower()]
    if not hits:
        from .errors import InputError
        raise InputError(f"no guide section matches {topic!r}; the sections are: "
                         + ", ".join(h for h, _ in guide_sections(text)))
    for _, body in hits:
        sys.stdout.write(body if body.endswith("\n") else body + "\n")
    return 0


def _run(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["view"]:
        # undocumented, with the ranked store it shows: the slice preview served locally.
        # Parsed apart so it is not listed among the commands (argparse prints a
        # suppressed subcommand's help as '==SUPPRESS==' under this formatter).
        from .view import main_view
        return main_view(argv[1:])
    if argv[:1] == ["restore"]:
        # undocumented, the same way: labels from a ranked store, on any grid
        _need_store_extra()
        from .ranked_restore import main_cli
        return main_cli(argv[1:])
    F = argparse.ArgumentDefaultsHelpFormatter

    class Fmt(argparse.RawDescriptionHelpFormatter, F):
        """Defaults in the help, and epilogs kept as written."""

    ap = argparse.ArgumentParser(
        prog="haversack", formatter_class=Fmt,
        description="Medical-image segmentation with many model families behind one command: "
                    "TotalSegmentator, MOOSE, MRSegmentator, stock nnU-Net, FastSurfer, SynthStrip. "
                    "Runs on Apple Silicon (MPS), CUDA or CPU; also a local REST server and a client for one.",
        epilog="""examples:
  haversack tasks                                          what can be segmented
  haversack segment scan.nii.gz --task total_fast -o labels.seg.nrrd
  haversack segment idc:<crdc_series_uuid> --task total -o labels.seg.nrrd
  haversack serve --port 8790                              a local server (it generates a token); then: haversack remote submit ...
  haversack docs                                           the user guide; `haversack docs weights` for one section""")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command", help="what to do")
    s = sub.add_parser("segment", formatter_class=Fmt,
                       help="segment one or more images: NIfTI, NRRD, MetaImage, a DICOM series directory, a URL or a hosted id",
                       description="Segment one image and write the labels, or several into a directory (batch). "
                                   "Weights download on first use. "
                                   "The output format follows the extension (.nii.gz, .nrrd, .seg.nrrd, .mha); "
                                   "labels come back on the input grid, in the input's orientation.",
                       epilog="""examples:
  haversack segment ct.nii.gz --task total_fast -o labels.seg.nrrd
  haversack segment dicom_dir/ --task total --spacing 1 -o labels.nii.gz
  haversack segment t1.nii.gz --task fastsurfer:brain -o brain.seg.nrrd      (from the fastsurfer venv)
  haversack segment "zenodo:<recid>/amos22.zip!amos22/imagesVa/amos_0575.nii.gz" --task mrsegmentator:base -o amos.seg.nrrd
  haversack segment a.nii.gz b.nii.gz dicom_dir/ --task total_fast --format seg.nrrd -o out/   (batch: out/<name>_total_fast.seg.nrrd)""")
    s.add_argument("input", nargs="+",
                   help="one or more inputs; several = batch mode. Each is a NIfTI / NRRD / MetaImage file, a DICOM "
                   "series directory, an http(s) URL (!member reads one file out of a remote zip), or a hosted "
                   "identifier: idc:<crdc_series_uuid>, zenodo:<recid>/<file>[!member], tcia:..., openneuro:..., "
                   "hf:<org>/<repo>@<sha>/<path>, s3:<bucket>/<key>, github:<owner>/<repo>@<tag>/<asset>")
    s.add_argument("--task", required=True,
                   help="what to segment: a name from `haversack tasks` (total_fast, total, fastsurfer:brain, ...), "
                        "or a path to a stock nnU-Net model folder")
    s.add_argument("-o", "--output", default=None,
                   help="one input: the output file (its extension picks the format). Several inputs: an output "
                   "directory (default: the current directory), each written as <input>_<task> in --format")
    s.add_argument("--format", default=None,
                   help="output type for batch mode (nifti, nrrd, seg.nrrd, mha) - required when segmenting several inputs")
    s.add_argument("--spacing", type=float, default=None, help="isotropic output spacing in mm (default: the input grid)")
    s.add_argument("--interp", choices=("linear", "nearest"), default="linear",
                   help="logit interpolation for the restore: linear = sub-voxel boundaries; nearest = TotalSegmentator semantics")
    s.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto (the best available)")
    s.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16",
                   help="network precision on the nnU-Net path (fp16 runs on MPS; an explicit choice is never lowered)")
    s.add_argument("--accumulate", choices=("auto", "device", "host"), default="auto",
                   help="sliding-window accumulator placement: auto (from free device memory), device (fastest, needs headroom), host")
    s.add_argument("--batch-size", default="auto", help="patches per forward pass: auto (default), or an int")
    s.add_argument("--envelope", type=float, default=20.0,
                   help="restrict inference to the body's bounding box plus this margin in mm; 0 or negative = whole volume")
    s.add_argument("--model-root", default=None,
                   help="where model weights live (default: TOTALSEG_WEIGHTS_PATH, nnUNet_results, "
                        "or ~/.totalsegmentator/nnunet/results)")
    s.add_argument("--allow-transpose", action="store_true",
                   help="run a model whose plans permute the axes (transpose_forward). Refused by "
                        "default because the transposed path had not been checked against an "
                        "outside implementation; dentalsegmentator:base, totalvibe:vibe_sagittal "
                        "and totalvibe:pancreas need it")
    s.add_argument("--quiet", action="store_true", help="no progress or timings on stderr")

    g = sub.add_parser("get", formatter_class=Fmt, help="fetch source data (idc:/zenodo:/http...) into the cache, or out to a file",
                       description="Acquire a remote input without segmenting it. With no -o, it lands in the cache "
                                   "(~/.cache/haversack/inputs) and the path is printed; a later `segment <same id>` "
                                   "reuses it. With -o, it is also written there: a directory gets the raw fetched "
                                   "content (a DICOM series stays a directory), an image-extension file (or --format) "
                                   "is converted to that one volume (a DICOM series -> one NIfTI/NRRD), geometry "
                                   "preserved. The raw data stays cached unless --no-cache.",
                       epilog="""examples:
  haversack get idc:<crdc_series_uuid>                     into cache; prints the path
  haversack get idc:<crdc_series_uuid> -o case1/scan.nii.gz  the series as one NIfTI
  haversack get idc:<crdc_series_uuid> --format nrrd -o out/  converted, auto-named <uuid>.nrrd
  haversack get idc:<crdc_series_uuid> -o raw_dicom/         the raw DICOM series directory""")
    g.add_argument("source", nargs="+",
                   help="one or more remote inputs (several = batch): idc:<uuid>, zenodo:<recid>/<file>[!member], "
                   "tcia:, openneuro:, hf:<org>/<repo>@<sha>/<path>, s3:<bucket>/<key>[!member], "
                   "github:<owner>/<repo>@<tag>/<asset>[!member], or an http(s) URL")
    g.add_argument("-o", "--output", default=None, help="where to put it: a directory (raw copy) or a file (converted by extension)")
    g.add_argument("--format", default=None, help="output format (nifti, nrrd, seg.nrrd, mha): convert, and name by it into a directory")
    g.add_argument("--no-cache", action="store_true", help="do not keep the raw data in the cache (only with -o)")
    # (batch: -o is an output directory, default the current one; --format names each converted file)

    tl = sub.add_parser("tasks", formatter_class=Fmt, help="list every task the catalog can segment, or one task's structures",
                        description="One line per task: name, engine, modality, and whether its weights are on disk "
                                    "(or, for an engine task, whether the engine's runtime is installed here). "
                                    "With a task name, prints that task's structures, one per line, in label order.",
                        epilog="""examples:
  haversack tasks                        every task
  haversack tasks --installed            what runs without a download
  haversack tasks total_fast             the 117 structure names total_fast produces
  haversack tasks --json                 full records: name, ecosystem, engine, modality, structures, installed;
                                         `materialized` = the task's definition is known here without a download,
                                         `task_spec` = it is an nnU-Net model (false for FastSurfer, SynthStrip, ...)""")
    tl.add_argument("task", nargs="?", default=None,
                    help="a task name: print its structures instead of the list. An nnU-Net "
                         "task prints `<label>\t<name>` in label order; an engine task, whose "
                         "labels are its own, prints names only")
    tl.add_argument("--model-root", default=None, help="weights root to check for installed models")
    tl.add_argument("--installed", action="store_true", help="only tasks whose weights are already on disk")
    tl.add_argument("--json", action="store_true", help="the full per-task info records")

    ci = sub.add_parser("cite", formatter_class=Fmt, help="who made a task's model, its license, and what to cite",
                        description="The credit for one task, from all three layers: the task's own facts (a "
                                    "bundle's authors, a per-model license), its ecosystem (the group, the "
                                    "repository, the license, the papers) and the engine that runs it (nnU-Net "
                                    "asks to be cited alongside every model trained with it). Every reference "
                                    "carries its DOI and PubMed ID where one exists. Nothing is downloaded.",
                        epilog="""examples:
  haversack cite total_fast              TotalSegmentator's CT paper, nnU-Net, the license
  haversack cite totalvibe:body_regions  the TUM group, European Radiology 2026, Apache-2.0
  haversack cite monai:brats_mri_segmentation --json   the bundle's own references, as data""")
    ci.add_argument("task", help="a task name, in any accepted form")
    ci.add_argument("--json", action="store_true", help="the full attribution record")

    ri = sub.add_parser("rights", formatter_class=Fmt, help="the license and citation of a remote input, without fetching it",
                        description="What governs reuse of one input, as its repository states it: the license, the "
                                    "collection or dataset it belongs to, and the citation its publisher asks for. "
                                    "Metadata only - nothing is downloaded. The same record is written beside every "
                                    "fetched input and into every result's provenance as `inputs`.",
                        epilog="""examples:
  haversack rights idc:19ecafc9-d05a-4c6c-8727-ce1a78190d11   the NLST collection, CC BY 4.0, its DOI
  haversack rights zenodo:7262581/amos22.zip                 the record's license, creators and DOI
  haversack rights openneuro:ds000114/x.nii.gz               CC0, by OpenNeuro's policy""")
    ri.add_argument("input", help="a remote input: idc:, tcia:, zenodo:, openneuro:, hf:, s3:, gs:, github:")
    ri.add_argument("--json", action="store_true", help="the record as JSON")

    w = sub.add_parser("weights", formatter_class=Fmt, help="download model weights ahead of time, or see what can be",
                       description="Weights download on first use; these commands do it ahead of time, or report "
                                   "what the manifest can provision (some TotalSegmentator tasks are behind its license).")
    wsub = w.add_subparsers(dest="wcmd", required=True, metavar="action", help="what to do with the weights")
    wf = wsub.add_parser("fetch", help="download everything a task needs")
    wf.add_argument("task", help="a task name from `haversack tasks`; every model it needs is fetched")
    wf.add_argument("--root", "--model-root", dest="root", default=None,
                    help="weights root (default: the ecosystem's location)")
    wsub.add_parser("coverage", help="which catalog tasks the manifest can provision")
    wl = wsub.add_parser("list", help="installed model weights on disk, with sizes")
    wl.add_argument("--root", "--model-root", dest="root", default=None,
                    help="weights root (default: the ecosystem's location)")
    wrm = wsub.add_parser("remove", help="delete one dataset's installed weights")
    wrm.add_argument("weights_id", help="a dataset id, e.g. 297 (see `weights list`)")
    wrm.add_argument("--root", "--model-root", dest="root", default=None,
                     help="weights root (default: the ecosystem location)")
    wrm.add_argument("--yes", action="store_true", help="do not prompt")
    wr = wsub.add_parser("refresh", formatter_class=Fmt, help="merge newly published weights into the manifest",
                         description="Reads TotalSegmentator's GitHub releases and records new datasets and versions. "
                                     "From an installed package this writes YOUR manifest "
                                     "(~/.config/haversack/ts_weights.json, or HAVERSACK_TS_MANIFEST), laid over the "
                                     "packaged one and kept across upgrades; in a source checkout it edits the "
                                     "repository's file. Set GITHUB_TOKEN to lift GitHub's 60 requests/hour.")
    wr.add_argument("--repo", default=None, help="GitHub repo to read releases from")
    wr.add_argument("--to", default=None, help="write this file instead of the default target")
    wr.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    wr.add_argument("--update-existing", action="store_true",
                    help="also repoint datasets at newer releases (changes which weights download)")

    sv = sub.add_parser("serve", formatter_class=Fmt, help="run the REST job server on this machine (needs the serve extra)",
                        description="A job server with warm models, progress streaming and a durable result cache; "
                                    "the same protocol haversack deploys on Modal. Computation needs a bearer "
                                    "token; reads never do. Without --token the server generates one, prints it, "
                                    "and leaves it in a file only you can read, which `haversack remote` on this "
                                    "machine picks up by itself - so personal use has no ceremony, and a proxy or "
                                    "tunnel in front of the server still faces a token. --no-token runs open, with "
                                    "no protection of any kind.",
                        epilog="""examples:
  haversack serve                                                   (a token is generated for you)
  HAVERSACK_SERVER=http://127.0.0.1:8790 haversack remote submit scan.nii.gz --task total_fast
  haversack serve --host 0.0.0.0 --token secret                      (other machines pass --token secret)""")
    sv.add_argument("--host", default="127.0.0.1", help="interface to listen on (0.0.0.0 for the whole network)")
    sv.add_argument("--port", type=int, default=8790, help="port to listen on")
    sv.add_argument("--device", default="auto", help="cuda, mps, cpu, or auto")
    sv.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16", help="network precision on the nnU-Net path")
    sv.add_argument("--cache-models", type=int, default=5,
                    help="models kept warm across jobs (5 covers a total union)")
    sv.add_argument("--model-root", default=None, help="where model weights live (see `segment --model-root`)")
    sv.add_argument("--max-pending", type=int, default=16, help="queue bound; past it POST returns 429")
    sv.add_argument("--keep-finished", type=int, default=50, help="finished jobs (and files) retained")
    sv.add_argument("--jobs-ttl-hours", type=float, default=24.0,
                    help="how long a job RECORD lasts (keep-finished bounds memory "
                         "and files; this bounds the durable record)")
    sv.add_argument("--workdir", default=None, help="job storage (default: a temp directory)")
    sv.add_argument("--cache-dir", default=None,
                    help="result cache (default: ~/.cache/haversack/results; durable, unlike the workdir)")
    sv.add_argument("--no-result-cache", action="store_true", help="compute every request; keep nothing durable")
    sv.add_argument("--token", default=None,
                    help="the bearer token that gates computation (reads stay open); generated when omitted")
    sv.add_argument("--allow-transpose", action="store_true",
                    help="serve tasks whose plans permute the axes (dentalsegmentator:base, "
                         "totalvibe:vibe_sagittal, totalvibe:pancreas). Deployment policy, so "
                         "it cannot come from a request; without it those tasks are listed and "
                         "described but refuse to run")
    sv.add_argument("--no-token", action="store_true",
                    help="run WITHOUT a token: anything that can reach the port can compute, a proxy or "
                         "tunnel in front included. No guards of any kind - a machine you trust end to end")

    mo = sub.add_parser("modal", formatter_class=Fmt, help="deploy the server to your Modal account (needs the modal extra)",
                        description="Deploys the same server to Modal, one GPU worker per engine. Images build in Modal's "
                                    "cloud; the deploy prints the URL. Costs run while a worker is warm; stop with "
                                    "`modal app stop haversack-serve --yes`.")
    mosub = mo.add_subparsers(dest="mcmd", required=True, metavar="action", help="what to do")
    md = mosub.add_parser("deploy", help="deploy the packaged app to your Modal account")
    md.add_argument("--gpu", default=None, help="worker GPU (default L40S; A10 is the economical fast-mode choice)")
    md.add_argument("--app-name", default=None, help="Modal app name (default: haversack-serve)")
    md.add_argument("--scaledown", type=int, default=None,
                    help="seconds a warm worker lingers after its last job (Modal caps at 1200)")
    md.add_argument("--no-proxy-auth", action="store_true",
                    help="deploy WITHOUT auth - smoke tests only; anyone with the URL can spend your GPU credit")
    mosub.add_parser("app-path", help="print the deployable app file's path")

    rc = sub.add_parser("remote", formatter_class=Fmt, help="talk to a haversack server, local or on Modal (needs the remote extra)",
                        description="The client: upload an image (or name a hosted one) to a server, follow progress, "
                                    "download the labels. The server is --server or HAVERSACK_SERVER.")
    rc.add_argument("--server", default=None,
                    help="server URL, e.g. http://gpu-box:8790 (or set HAVERSACK_SERVER)")
    rc.add_argument("--token", default=None,
                    help="bearer token (or HAVERSACK_TOKEN); a server on this machine that generated its own "
                         "token needs neither - the client reads it from the file the server left")
    rsub = rc.add_subparsers(dest="rcmd", required=True, metavar="action", help="what to ask the server")
    rs = rsub.add_parser("submit", help="upload, wait with progress, download the labels")
    rs.add_argument("input", help="a local image file, or idc:<crdc_series_uuid> to segment straight from the Imaging Data Commons")
    rs.add_argument("--task", required=True, help="a task name the server lists (`haversack remote tasks`)")
    rs.add_argument("-o", "--output", default=None, help="where to save the labels (default: <input>_<task>.seg.nrrd)")
    rs.add_argument("--no-wait", action="store_true", help="print the job id and return")
    rst = rsub.add_parser("status", help="one job's status, as JSON")
    rst.add_argument("job_id", help="the id `submit --no-wait` printed")
    rf = rsub.add_parser("fetch", help="download a finished job's labels")
    rf.add_argument("job_id", help="the id `submit --no-wait` printed")
    rf.add_argument("-o", "--output", required=True, help="where to save the labels")
    rx = rsub.add_parser("cancel", help="cancel an active job / delete a finished one")
    rx.add_argument("job_id", help="the id `submit --no-wait` printed")
    rsub.add_parser("tasks", help="what the server can segment")

    dc = sub.add_parser("docs", formatter_class=Fmt, help="print a guide (Markdown), whole or one section",
                        description="The guides that ship with the package. The user guide: requirements, install, "
                                    "weights, the command line, the Python API, engines. The server guide (--server): "
                                    "the job protocol, its rules, results by path, sources, caches, deploying to Modal. "
                                    "Pipe either to a pager or a Markdown viewer; a running server's /docs has the "
                                    "route-by-route OpenAPI reference.",
                        epilog="""examples:
  haversack docs | less
  haversack docs weights            just the section whose heading contains 'weights'
  haversack docs --sections         the section headings
  haversack docs --server           the server guide
  haversack docs --server jobs      one section of it""")
    dc.add_argument("topic", nargs="?", default=None, help="print only the section whose heading contains this (case-insensitive)")
    dc.add_argument("--sections", action="store_true", help="list the section headings and exit")
    dc.add_argument("--server", action="store_true", help="the server guide instead of the user guide")

    ca = sub.add_parser("cache", formatter_class=Fmt, help="show and clean haversack's on-disk stores",
                        description="Lists every store haversack keeps (fetched inputs, server results, engine "
                                    "checkpoints, and the model-weights root) with sizes, and cleans the transient "
                                    "ones. Weights are never swept here - remove a model with `weights remove`.")
    casub = ca.add_subparsers(dest="ccmd", required=True, metavar="action", help="what to do")
    casub.add_parser("list", help="every store, its location and size")
    casub.add_parser("path", help="print the store locations, one per line")
    cc = casub.add_parser("clean", formatter_class=Fmt, help="remove cached inputs / results / checkpoints",
                          description="Sweeps a transient cache. Shows what would go and needs --yes to act.")
    cc.add_argument("category", choices=("inputs", "results", "checkpoints", "all"), help="which cache to sweep")
    cc.add_argument("item", nargs="?", default=None, help="one input to drop, by its spec (only with `inputs`)")
    cc.add_argument("--older-than", default=None, help="keep entries touched within this window, e.g. 30d, 12h")
    cc.add_argument("--dry-run", action="store_true", help="report what would be removed, delete nothing")
    cc.add_argument("--yes", action="store_true", help="actually delete (without this, it is a dry run)")

    args = ap.parse_args(argv)
    if args.cmd == "docs":
        return _docs(args.topic, args.sections, "server" if args.server else "user")
    if args.cmd == "modal":
        from importlib.resources import files
        apppath = str(files("haversack").joinpath("modal_app.py"))
        if args.mcmd == "app-path":
            print(apppath)
            return 0
        try:
            import modal  # noqa: F401
        except ImportError:
            print("needs the modal extra: uv sync --extra modal "
                  "(or pip install 'haversack[modal]')", file=sys.stderr)
            return 2
        import os
        import subprocess
        env = dict(os.environ)
        if args.gpu:
            env["HAVERSACK_GPU"] = args.gpu
        if args.app_name:
            env["HAVERSACK_APP_NAME"] = args.app_name
        if args.scaledown:
            env["HAVERSACK_SCALEDOWN"] = str(args.scaledown)
        if args.no_proxy_auth:
            env["HAVERSACK_PROXY_AUTH"] = "0"
        return subprocess.call([sys.executable, "-m", "modal", "deploy", apppath], env=env)
    if args.cmd == "serve":
        if args.token and args.no_token:
            from .errors import InputError
            raise InputError("--token and --no-token contradict each other")
        _need_inference_stack()          # the local server runs models in-process
        from .serve import main_serve
        return main_serve(args)
    if args.cmd == "remote":
        import json
        import os
        from .client import RemoteClient
        server = args.server or os.environ.get("HAVERSACK_SERVER")
        if not server:
            print("no server: pass --server or set HAVERSACK_SERVER", file=sys.stderr)
            return 2
        from .cache_admin import local_token_for, serve_token_path
        from urllib.parse import urlsplit
        token, token_source = args.token, "--token"
        if not token and os.environ.get("HAVERSACK_TOKEN"):
            token, token_source = os.environ["HAVERSACK_TOKEN"], "HAVERSACK_TOKEN"
        if not token:
            token = local_token_for(server)
            port = urlsplit(server if "://" in server else f"http://{server}").port or 80
            token_source = f"the local server's file {serve_token_path(port)}"
        c = RemoteClient(server, token=token, token_source=token_source if token else None)
        if args.rcmd == "tasks":
            for t in c.tasks():
                print(t)
        elif args.rcmd == "status":
            print(json.dumps(c.status(args.job_id), indent=2))
        elif args.rcmd == "fetch":
            print(c.fetch(args.job_id, args.output))
        elif args.rcmd == "cancel":
            print(json.dumps(c.cancel(args.job_id)))
        elif args.rcmd == "submit":
            if args.no_wait:
                print(c.submit(args.input, args.task))
                return 0
            stem = args.input[4:16] if args.input.startswith("idc:") else args.input.rsplit(".nii", 1)[0].rstrip("/")
            out = args.output or f"{stem}_{args.task}.seg.nrrd"
            last = {}
            def show(s, _last=last):
                p = s.get("progress") or {}
                line = (f"  {s['state']:9s} " + (f"[queue {s['queue_position']}] " if s.get("queue_position") is not None else "")
                        + f"{p.get('stage', '')} {p.get('detail', '')} "
                        + (f"{p.get('fraction', 0) * 100:3.0f}%" if p else ""))
                if line != _last.get("line"):
                    print(line, file=sys.stderr, flush=True)
                    _last["line"] = line
            final = c.run(args.input, args.task, out, on_status=show)
            if final["state"] == "done":
                print("  done      100%", file=sys.stderr, flush=True)
                print(f"wrote {out}", file=sys.stderr, flush=True)
                print(out)
            else:
                print(f"job ended {final['state']}", file=sys.stderr)
                return 1
        return 0
    if args.cmd == "rights":
        import json
        from .errors import InputError
        from .sources import HttpSource, check_identifier, default_sources, parse_input, registry
        reg = registry(default_sources() + [HttpSource()])
        parsed = parse_input(args.input, known=reg)
        if parsed is None:
            raise InputError(f"{args.input}: a local file; haversack cannot know its origin or license")
        kind, ident = parsed
        src = reg["http" if kind == "https" else kind]
        check_identifier(src, ident)
        rights = src.rights(ident)
        record = {"identity": f"{kind}:{ident}", "rights": rights}
        if args.json:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        elif rights is None:
            print(f"{kind}:{ident}\n  rights: not determined - the {kind} source cannot say what "
                  "license this input is under")
        else:
            print(f"{kind}:{ident}")
            for k, v in rights.items():
                if isinstance(v, dict):
                    v = "; ".join(f"{a}: {b}" for a, b in v.items() if b)
                elif isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                if v:
                    print(f"  {k + ':':<20}{v}")
        return 0
    if args.cmd == "cite":
        import json
        from . import attribution
        from .ecosystems import EcosystemCatalog
        from .weights import WeightsStore
        cat = EcosystemCatalog(root=WeightsStore(None, fetch=False).root)
        info = cat.info(args.task)
        if args.json:
            print(json.dumps(info["attribution"], indent=2, ensure_ascii=False))
        else:
            print(attribution.format(info["name"], info))
        return 0
    if args.cmd == "tasks":
        import json
        from .ecosystems import EcosystemCatalog
        from .weights import WeightsStore
        store = WeightsStore(args.model_root, fetch=False)
        cat = EcosystemCatalog(root=store.root)
        if args.task:
            info = cat.info(args.task)
            names = info.get("structures") or []
            if not names:
                from .errors import InputError
                if info.get("unresolved"):
                    # installed, but not runnable as it stands - `weights fetch`
                    # would do nothing, so say what actually helps
                    raise InputError(f"{info['name']}: {info['unresolved']}")
                raise InputError(f"{info['name']}: no structure list until its model is installed "
                                 f"(haversack weights fetch {args.task})")
            if args.json:
                out = {"name": info["name"], "structures": list(names)}
                if info.get("label_map"):
                    out["label_map"] = info["label_map"]   # what a JSON consumer needs most
                print(json.dumps(out, indent=2))
            else:
                # label order, with the label - which is what a caller needs to read
                # a result, and the only way to make sense of a catalog whose
                # checkpoints name their structures with numbers
                labels = info.get("label_map")
                if labels:
                    for k in sorted(labels, key=int):
                        print(f"{k}\t{labels[k]}")
                else:
                    for n in names:
                        print(n)
            return 0

        def installed(info) -> bool:
            # "materialized" is "the spec is answerable without installing" - for TS that is
            # always true (the catalog ships the specs), so ask the store about the weights
            # themselves. Never call cat.get() on an unmaterialized task: it would install.
            if not info.get("materialized"):
                return False
            if not info.get("task_spec", True):
                from .engines import registry     # an engine: installed = its runtime is here
                return registry.available(info.get("engine", ""))
            try:
                return all(store.have(w) for w in cat.get(info["name"]).weights_ids)
            except Exception:
                return False

        rows = []
        for name in cat.names():
            try:
                info = cat.info(name)
            except Exception as e:                      # one broken catalog entry must not hide the rest
                info = {"name": name, "error": str(e)}
            info["installed"] = installed(info)
            if args.installed and not info["installed"]:
                continue
            rows.append(info)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            for i in rows:
                print(f"{i['name']:44s} {i.get('engine', ''):12s} {i.get('modality') or '':4s} "
                      f"{'installed' if i['installed'] else ''}".rstrip())
        return 0
    if args.cmd == "get":
        import shutil
        import tempfile
        from pathlib import Path
        from . import io, sources
        from .errors import InputError
        say = lambda m: print(f"  {m}", file=sys.stderr, flush=True)
        srcs = args.source
        if len(srcs) > 1 and args.output and not args.format and not str(args.output).rstrip("/").endswith(tuple(io.IMAGE_SUFFIXES)):
            pass  # multiple raw-copies into a directory is allowed; --format only needed to convert

        def get_one(src_spec, out_target):
            """Fetch one source; write per out_target (None=cache only). Returns the path produced."""
            if sources.parse_input(src_spec) is None:
                p = Path(src_spec)
                if not p.exists():
                    raise InputError(f"not a remote source and not a local path: {src_spec}")
                return p
            tmp = tempfile.mkdtemp(prefix="haversack-get-") if args.no_cache else None
            try:
                src = sources.materialize(src_spec, cache_dir=tmp, progress=say)
                if out_target is None:
                    return src
                out, want_convert = out_target
                if want_convert:
                    io.convert(src, out)
                elif Path(src).is_dir():
                    shutil.copytree(src, out, dirs_exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, out)
                return out
            finally:
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)

        batch = len(srcs) > 1
        if args.no_cache and not args.output:
            raise InputError("--no-cache needs -o (there would be nowhere to put the data)")
        if not batch:
            src_spec = srcs[0]
            if sources.parse_input(src_spec) is None:
                p = Path(src_spec)
                if not p.exists():
                    raise InputError(f"not a remote source and not a local path: {src_spec}")
                print(p); return 0
            if not args.output:
                print(get_one(src_spec, None)); return 0
            out = Path(args.output)
            want_convert = bool(args.format) or bool(io.image_suffix(out.name))
            if want_convert:
                ext = io.format_extension(args.format) if args.format else io.image_suffix(out.name)
                if out.is_dir() or str(args.output).endswith("/") or not io.image_suffix(out.name):
                    out = out / (sources.source_stem(src_spec) + ext)
                print(get_one(src_spec, (out, True))); return 0
            if out.is_dir() or str(args.output).endswith("/"):
                name = sources.source_stem(src_spec)
                src = sources.materialize(src_spec, progress=say)
                dest = out / (name if Path(src).is_dir() else Path(src).name)
                if Path(src).is_dir():
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    out.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
                print(dest); return 0
            src = sources.materialize(src_spec, progress=say)
            if Path(src).is_dir():
                raise InputError(f"{src_spec} is a DICOM series (a directory); give a directory -o, "
                                 "or a file with an image extension (or --format) to convert it")
            out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, out); print(out); return 0

        # batch: several sources
        if not args.output and not args.format:          # no destination: cache each, print paths
            for src_spec in srcs:
                print(get_one(src_spec, None))
            return 0
        outdir = Path(args.output or ".")                # default the output directory to the cwd
        outdir.mkdir(parents=True, exist_ok=True)
        ext = io.format_extension(args.format) if args.format else None
        failures = 0
        for src_spec in srcs:
            try:
                if ext:                                   # convert each into the directory
                    out = get_one(src_spec, (outdir / (sources.source_stem(src_spec) + ext), True))
                else:                                     # raw copy each into the directory
                    src = sources.materialize(src_spec, progress=say)
                    dest = outdir / (sources.source_stem(src_spec) if Path(src).is_dir() else Path(src).name)
                    if Path(src).is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dest)
                    out = dest
                print(out)
            except Exception as e:
                failures += 1; print(f"  FAILED {src_spec}: {e}", file=sys.stderr)
        if failures:
            print(f"{failures} of {len(srcs)} sources failed", file=sys.stderr); return 1
        return 0
    if args.cmd == "cache":
        from .cache_admin import check_cache_root
        check_cache_root()
        import json
        from . import cache_admin as ca
        if args.ccmd == "path":
            for st in ca.stores():
                print(st["path"])
            return 0
        if args.ccmd == "list":
            for r in ca.usage():
                tag = ("" if r["sweepable"] else "  (weights - use `weights remove`)"
                       if r["name"] == "weights" else "  (not swept by clean)")
                print(f"{r['name']:12s} {r['human']:>10s}  {r['items']:>4d} items  {r['path']}{tag}")
            return 0
        if args.ccmd == "clean":
            days = None
            if args.older_than:
                m = {"d": 1, "h": 1 / 24, "w": 7, "m": 30}.get(args.older_than[-1].lower())
                if m is None:
                    from .errors import InputError
                    raise InputError(f"--older-than {args.older_than!r}: use a number then d/h/w/m, e.g. 30d")
                days = float(args.older_than[:-1]) * m
            r = ca.clean(args.category, older_than_days=days, item=args.item, dry_run=not args.yes)
            verb = "would remove" if not args.yes else "removed"
            print(f"{verb} {len(r['removed'])} entr{'y' if len(r['removed']) == 1 else 'ies'}, {r['human']}",
                  file=sys.stderr)
            for pth in r["removed"]:
                print(f"  {pth}", file=sys.stderr)
            if not args.yes and r["removed"]:
                print("  (dry run - pass --yes to delete)", file=sys.stderr)
            return 0
    if args.cmd == "segment":
        import json as _json
        from pathlib import Path
        from . import io
        from .errors import InputError
        from .engines import registry
        from .sources import materialize, parse_input, source_stem
        progress = None if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr, flush=True))
        inputs = args.input
        batch = len(inputs) > 1 or args.format is not None
        # The cheap mistakes first - before the inference stack is demanded (a lean
        # install should hear about its typo, not about torch) and before any input is
        # downloaded or a minute of inference is spent on an output the writer cannot name.
        from .io import is_store_output
        if not batch:
            if not args.output:
                raise InputError("segment needs -o (the output file), or --format with -o a directory for batch")
            if not is_store_output(args.output) and io.image_suffix(args.output) is None:
                raise InputError(f"{args.output}: not an output haversack writes; labels take "
                                 ".seg.nrrd, .nrrd, .nii.gz, .nii or .mha (a directory or a bare "
                                 "name is not a file), and a ranked store is named .duckn or "
                                 ".duckn.zip")
            if is_store_output(args.output):
                # the store's grid is the model's; a label spacing has nothing to apply to,
                # and the target has to be writable BEFORE minutes of inference
                if args.spacing:
                    raise InputError("a ranked store is written on the model grid; --spacing applies "
                                     "to labels only")
                _need_store_extra()
                from .ranked_store import check_target
                check_target(args.output)
        else:
            if is_store_output(args.output or "."):
                raise InputError("a ranked store output takes exactly one input and no --format")
            if str(args.output or "").lower().endswith((".zarr", ".zip", ".duckn")):
                raise InputError(f"{args.output}: a batch output is a directory of labels; that "
                                 "name says something else")
            if args.format is None:
                raise InputError("segmenting several inputs needs --format (the output type, e.g. seg.nrrd)")
        bs = args.batch_size if args.batch_size == "auto" else int(args.batch_size)
        _need_inference_stack(args.task)
        engine_task = registry.engine_for_task(args.task).name != registry.NNUNETV2

        def resolve(spec):
            if parse_input(spec) is not None:
                return str(materialize(spec, progress=progress))
            if not Path(spec).exists():
                raise InputError(f"input not found: {spec}")
            return spec

        def run_one(spec):
            r = _run_one(spec)
            from .sources import input_record
            prov = getattr(r, "provenance", None)      # in place: Segmentation is frozen
            if isinstance(prov, dict):
                prov["inputs"] = [input_record(spec)]
            return r

        def _run_one(spec):
            img = resolve(spec)
            if engine_task:
                from .segmenter import Segmenter
                return Segmenter(device=args.device, weights=args.model_root, batch_size=bs,
                                 allow_transpose=args.allow_transpose).segment(
                    img, args.task, progress=progress)
            from .pipeline import segment
            return segment(img, args.task, weights=args.model_root, device=args.device, dtype=args.dtype,
                           grid=args.spacing if args.spacing else "input", interp=args.interp,
                           accumulate=args.accumulate, batch_size=bs,
                           envelope_mm=args.envelope if args.envelope > 0 else None,
                           allow_transpose=args.allow_transpose, progress=progress)

        def report(r, where):
            if args.quiet:
                return
            for k, v in r.timings.items():
                print(f"  {v:7.2f} s  {k}", file=sys.stderr)
            for d in (r.provenance or {}).get("deviations", ()):
                print(f"  note: {d['what']}: asked {d['requested']}, ran {d['effective']} - {d['why']}", file=sys.stderr)
            print(f"wrote {where}: {tuple(r.grid.shape)}, {len(r.present())}/{len(r.schema.names)} structures present",
                  file=sys.stderr)

        if not batch:
            if is_store_output(args.output):
                # undocumented: a `.duckn` / `.duckn.zip` output is a ranked store - the whole
                # output distribution, not the labels (see haversack.ranked_output)
                if engine_task:
                    raise InputError("a ranked store output is available for nnU-Net tasks only")
                from .ranked_output import input_source, segment_to_store
                img = resolve(inputs[0])
                r, out = segment_to_store(
                    img, args.task, args.output, case=source_stem(inputs[0]),
                    source=input_source(inputs[0]),          # the spec as given, not the cache path
                    weights=args.model_root, device=args.device, dtype=args.dtype,
                    grid=args.spacing if args.spacing else "input", interp=args.interp,
                    accumulate=args.accumulate, batch_size=bs,
                    envelope_mm=args.envelope if args.envelope > 0 else None, progress=progress)
                if not args.quiet:
                    for k, v in r.timings.items():
                        print(f"  {v:7.2f} s  {k}", file=sys.stderr)
                    print(f"wrote {out}: ranked store on the model grid, "
                          f"{len(r.present())}/{len(r.schema.names)} structures present", file=sys.stderr)
            else:
                r = run_one(inputs[0])
                r.save(args.output)
                report(r, args.output)
        else:
            ext = io.format_extension(args.format)
            outdir = Path(args.output or "."); outdir.mkdir(parents=True, exist_ok=True)
            task_tag = str(args.task).replace(":", "-")
            failures = 0
            for spec in inputs:
                out = outdir / f"{source_stem(spec)}_{task_tag}{ext}"
                if not args.quiet:
                    print(f"[{spec}] -> {out}", file=sys.stderr)
                try:
                    r = run_one(spec)
                    r.save(out)
                    report(r, out)
                except Exception as e:                    # one bad input must not sink the batch
                    failures += 1
                    print(f"  FAILED {spec}: {e}", file=sys.stderr)
            if failures:
                print(f"{failures} of {len(inputs)} inputs failed", file=sys.stderr)
                return 1
    if args.cmd == "weights":
        from pathlib import Path
        from . import weights_fetch as wfm
        say = lambda m: print(m, file=sys.stderr, flush=True)
        if args.wcmd == "fetch":
            # through the ecosystem catalog, not TotalSegmentator's manifest: every
            # catalog installs its own weights, and `tasks` sends people here for
            # any of them. TS still ends up in ensure_task_weights - via its own
            # ecosystem - so nothing about that path changes.
            from .ecosystems import EcosystemCatalog
            from .weights import WeightsStore
            store = WeightsStore(args.root, fetch=False)
            cat = EcosystemCatalog(root=store.root)
            info = cat.prepare(args.task, progress=lambda m: say(f"  {m}"))
            # No count. `weights_installed` is only populated by engine ecosystems,
            # and the spec's own weights_ids omits the crop_from_task chains a
            # cascade installs - ts:teeth pulls three models and either number
            # says one. `weights list` reports what is actually on disk.
            if info.get("task_spec", True):
                print(f"{info['name']}: weights ready under {store.root}")
            else:
                # an engine's weights ship inside its image; nothing was installed
                # here and nothing is under this root
                print(f"{info['name']}: runs on the {info.get('engine')} engine, whose weights "
                      "ship with it - nothing to fetch")
        elif args.wcmd == "list":
            from .tasks import weights_root
            from .cache_admin import _du, _human
            root = Path(args.root or weights_root("ts")).expanduser()
            if not root.exists():
                print(f"no weights installed under {root}"); return 0
            datasets = _installed_datasets(root)
            total = 0
            for d in datasets:
                _, b = _du(d); total += b
                ver = (wfm.installed_version(d) or {}).get("tag", "")
                shown = str(d.relative_to(root))      # <bucket>/Dataset* for a catalog
                print(f"  {shown:52s} {_human(b):>10s}  {ver}")
            print(f"{len(datasets)} dataset(s), {_human(total)} under {root}")
            return 0
        elif args.wcmd == "remove":
            import re
            import shutil
            from .tasks import weights_root, _dataset_dirs
            from .errors import InputError
            root = Path(args.root or weights_root("ts")).expanduser()
            wanted = str(args.weights_id)
            # The id reaches a glob and then an rmtree. `weights remove '*'` matched
            # and deleted every dataset; `weights remove moose` deleted a whole
            # ecosystem bucket - and the "listed as <ecosystem>/Dataset<id>" hint
            # makes typing the bucket name the natural mistake.
            seg = r"[A-Za-z0-9][A-Za-z0-9._-]*"
            if not re.fullmatch(rf"{seg}(?:/{seg})?", wanted):
                raise InputError(
                    f"{wanted!r} is not a dataset id - give an id, or the name "
                    "`haversack weights list` prints (`Dataset297_total`, or "
                    "`totalvibe/Dataset278` for a catalog's)")
            # `_dataset_dirs` falls back to globbing the id, which matches a
            # bucket directory by name: `weights remove moose` deleted the whole
            # ecosystem. Only a Dataset folder is a thing this command removes.
            dirs = [d for d in _dataset_dirs(root, wanted)
                    if re.match(r"Dataset\d+", d.name)]
            if not dirs:                       # the ecosystem catalogs' own subtrees
                def names(d):
                    """The spellings that identify one installed dataset folder:
                    its own name, its `<bucket>/<name>` path as `weights list`
                    prints it, and the dataset id with or without zero padding
                    (`Dataset001_x` answers to 1, 001 and Dataset001_x)."""
                    yield d.name
                    yield str(d.relative_to(root))
                    m = re.match(r"Dataset(\d+)", d.name)
                    if m:
                        yield m.group(1)
                        yield str(int(m.group(1)))
                        yield f"Dataset{m.group(1)}"

                dirs = [d for d in _installed_datasets(root)
                        if wanted in set(names(d))
                        or d.name.startswith(f"Dataset{wanted}_")]
            if not dirs:
                raise InputError(
                    f"no installed weights match {wanted!r} under {root} - `haversack weights "
                    "list` names what is there - give a name exactly as it prints it, "
                    "including the <ecosystem>/ prefix, since one dataset id can appear "
                    "in more than one catalog")
            # nothing outside the root, whatever the match was
            base = root.resolve()
            for d in dirs:
                if not d.resolve().is_relative_to(base):
                    raise InputError(f"{d} resolves outside {root}; refusing to delete it")
            for d in dirs:
                print(f"  {d}", file=sys.stderr)
            if not args.yes:
                print(f"pass --yes to delete the above", file=sys.stderr); return 1
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            print(f"removed {len(dirs)} folder(s) for dataset {args.weights_id}", file=sys.stderr)
            return 0
        elif args.wcmd == "coverage":
            c = wfm.coverage()
            src = c["sources"]
            where = (f"{src['package']} packaged" + (f" + {src['user']} from {src['user_path']}"
                                                     f"{' (overriding ' + ', '.join(src['user_overrides']) + ')' if src['user_overrides'] else ''}"
                                                     if src["user"] else ""))
            print(f"{len(c['covered'])}/{c['n_tasks']} tasks provisionable from {c['n_weights']} manifest entries ({where})")
            for name, ids in sorted(c["license_required"].items()):
                print(f"  LICENSE  {name:32s} {','.join(ids)}  (TotalSegmentator licensed backend)")
            for name, ids in sorted(c["missing"].items()):
                print(f"  MISSING  {name:32s} {','.join(ids)}")
            return 1 if c["missing"] else 0
        elif args.wcmd == "refresh":
            kw = {"write": not args.dry_run, "update_existing": args.update_existing, "progress": say,
                  "path": args.to or wfm.refresh_target()}
            say(f"target: {kw['path']}" + (" (dry run)" if args.dry_run else ""))
            if args.repo:
                kw["repo"] = args.repo
            r = wfm.refresh_manifest(**kw)
            for wid, e in sorted(r["added"].items(), key=lambda kv: int(kv[0])):
                print(f"  + {wid:5s} new dataset, default {e['default']}")
            for wid, tags in sorted(r["new_versions"].items(), key=lambda kv: int(kv[0])):
                print(f"  v {wid:5s} versions recorded: {', '.join(tags)}")
            for wid, (ours, theirs) in sorted(r["behind_upstream"].items(), key=lambda kv: int(kv[0])):
                print(f"  ~ {wid:5s} default {ours}, TotalSegmentator pins {theirs}"
                      + ("" if args.update_existing else "   [not repointed]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
