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


INSTALL_HINT = ('this is a lean install (--no-deps); a normal install has them: '
                'uv pip install torch nnunetv2 scipy scikit-image, or reinstall with '
                'uv tool install "haversack @ git+https://github.com/mhalle/haversack"')


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


def user_guide() -> str:
    """The user guide's Markdown: the README shipped inside the wheel, or, in a checkout
    (editable install), the repository's README itself - one source of truth."""
    from importlib.resources import files
    shipped = files("haversack").joinpath("data/GUIDE.md")
    if shipped.is_file():
        return shipped.read_text()
    from pathlib import Path
    local = Path(__file__).resolve().parents[2] / "README.md"
    if local.is_file():
        return local.read_text()
    raise RuntimeError("the user guide is missing from this installation")


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


def _docs(topic, list_sections: bool) -> int:
    text = user_guide()
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
  haversack serve --port 8790 --token secret               then: haversack remote submit ...
  haversack docs                                           the user guide; `haversack docs weights` for one section""")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command", help="what to do")
    s = sub.add_parser("segment", formatter_class=Fmt,
                       help="segment one image: NIfTI, NRRD, MetaImage, a DICOM series directory, a URL or a hosted id",
                       description="Segment one image and write the labels. Weights download on first use. "
                                   "The output format follows the extension (.nii.gz, .nrrd, .seg.nrrd, .mha); "
                                   "labels come back on the input grid, in the input's orientation.",
                       epilog="""examples:
  haversack segment ct.nii.gz --task total_fast -o labels.seg.nrrd
  haversack segment dicom_dir/ --task total --spacing 1 -o labels.nii.gz
  haversack segment t1.nii.gz --task fastsurfer:brain -o brain.seg.nrrd      (from the fastsurfer venv)
  haversack segment "zenodo:<recid>/amos22.zip!amos22/imagesVa/amos_0575.nii.gz" --task mrsegmentator:base -o amos.seg.nrrd""")
    s.add_argument("input", help="NIfTI / NRRD / MetaImage file, a DICOM series directory, an http(s) URL "
                   "(!member reads one file out of a remote zip), or a hosted identifier: idc:<crdc_series_uuid>, "
                   "zenodo:<recid>/<file>[!member], tcia:..., openneuro:..., hf:... (fetched once to ~/.cache/haversack/inputs)")
    s.add_argument("--task", required=True,
                   help="what to segment: a name from `haversack tasks` (total_fast, total, fastsurfer:brain, ...), "
                        "or a path to a stock nnU-Net model folder")
    s.add_argument("-o", "--output", required=True, help="where to write the labels; the extension picks the format")
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
    s.add_argument("--quiet", action="store_true", help="no progress or timings on stderr")

    tl = sub.add_parser("tasks", formatter_class=Fmt, help="list every task the catalog can segment",
                        description="One line per task: name, engine, modality, and whether its weights are on disk "
                                    "(or, for an engine task, whether the engine's runtime is installed here).")
    tl.add_argument("--model-root", default=None, help="weights root to check for installed models")
    tl.add_argument("--installed", action="store_true", help="only tasks whose weights are already on disk")
    tl.add_argument("--json", action="store_true", help="the full per-task info records")

    w = sub.add_parser("weights", formatter_class=Fmt, help="download model weights ahead of time, or see what can be",
                       description="Weights download on first use; these commands do it ahead of time, or report "
                                   "what the manifest can provision (some TotalSegmentator tasks are behind its license).")
    wsub = w.add_subparsers(dest="wcmd", required=True, metavar="action", help="what to do with the weights")
    wf = wsub.add_parser("fetch", help="download everything a task needs")
    wf.add_argument("task", help="a task name from `haversack tasks`; every model it needs is fetched")
    wf.add_argument("--root", default=None, help="weights root (default: the ecosystem's location)")
    wsub.add_parser("coverage", help="which catalog tasks the manifest can provision")
    wr = wsub.add_parser("refresh", help="merge newly published weights into the manifest")
    wr.add_argument("--repo", default=None, help="GitHub repo to read releases from")
    wr.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    wr.add_argument("--update-existing", action="store_true",
                    help="also repoint datasets at newer releases (changes which weights download)")

    sv = sub.add_parser("serve", formatter_class=Fmt, help="run the REST job server on this machine (needs the serve extra)",
                        description="A job server with warm models, progress streaming and a durable result cache; "
                                    "the same protocol haversack deploys on Modal. Without --token, requests can read "
                                    "health, the task list and cached results but never compute.",
                        epilog="""example:
  haversack serve --port 8790 --token secret
  HAVERSACK_SERVER=http://127.0.0.1:8790 haversack remote --token secret submit scan.nii.gz --task total_fast""")
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
                    help="bearer token; without it a request gets health/tasks/cached reads only")

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
    rc.add_argument("--token", default=None, help="bearer token, if the server wants one")
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

    dc = sub.add_parser("docs", formatter_class=Fmt, help="print the user guide (Markdown), whole or one section",
                        description="The guide that ships with the package: requirements, install, weights, the command "
                                    "line, the Python API, the local server, engines. Pipe it to a pager or a Markdown viewer.",
                        epilog="""examples:
  haversack docs | less
  haversack docs weights            just the section whose heading contains 'weights'
  haversack docs --sections         the section headings""")
    dc.add_argument("topic", nargs="?", default=None, help="print only the section whose heading contains this (case-insensitive)")
    dc.add_argument("--sections", action="store_true", help="list the section headings and exit")

    args = ap.parse_args(argv)
    if args.cmd == "docs":
        return _docs(args.topic, args.sections)
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
        c = RemoteClient(server, token=args.token)
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
                print(out)
            else:
                print(f"job ended {final['state']}", file=sys.stderr)
                return 1
        return 0
    if args.cmd == "tasks":
        import json
        from .ecosystems import EcosystemCatalog
        from .weights import WeightsStore
        store = WeightsStore(args.model_root, fetch=False)
        cat = EcosystemCatalog(root=store.root)

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
    if args.cmd == "segment":
        from pathlib import Path
        from .errors import InputError
        from .sources import materialize, parse_input
        progress = None if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr, flush=True))
        if parse_input(args.input) is not None:
            # idc: / zenodo: / tcia: / openneuro: / hf: identifiers and http(s) URLs: fetched
            # once into ~/.cache/haversack/inputs, then segmented like any local path
            args.input = str(materialize(args.input, progress=progress))
        elif not Path(args.input).exists():
            raise InputError(f"input not found: {args.input}")
        _need_inference_stack(args.task)
        from .engines import registry
        batch = args.batch_size if args.batch_size == "auto" else int(args.batch_size)
        if registry.engine_for_task(args.task).name != registry.NNUNETV2:
            # an engine task: its runtime is in this environment (checked above); the
            # Segmenter routes it to the engine's in-process compute
            from .segmenter import Segmenter
            r = Segmenter(device=args.device, weights=args.model_root, batch_size=batch).segment(
                args.input, args.task, progress=progress)
        else:
            from .pipeline import segment
            r = segment(args.input, args.task, weights=args.model_root, device=args.device, dtype=args.dtype,
                        grid=args.spacing if args.spacing else "input", interp=args.interp,
                        accumulate=args.accumulate, batch_size=batch,
                        envelope_mm=args.envelope if args.envelope > 0 else None, progress=progress)
        r.save(args.output)
        if not args.quiet:
            for k, v in r.timings.items():
                print(f"  {v:7.2f} s  {k}", file=sys.stderr)
            for d in (r.provenance or {}).get("deviations", ()):
                print(f"  note: {d['what']}: asked {d['requested']}, ran {d['effective']} - {d['why']}",
                      file=sys.stderr)
            print(f"wrote {args.output}: {tuple(r.grid.shape)}, "
                  f"{len(r.present())}/{len(r.schema.names)} structures present", file=sys.stderr)
    if args.cmd == "weights":
        from . import weights_fetch as wfm
        say = lambda m: print(m, file=sys.stderr, flush=True)
        if args.wcmd == "fetch":
            from .tasks import weights_root
            root = args.root or weights_root("ts")
            paths = wfm.ensure_task_weights(args.task, root, progress=lambda m: say(f"  {m}"))
            print(f"{len(paths)} model(s) under {root}")
        elif args.wcmd == "coverage":
            c = wfm.coverage()
            print(f"{len(c['covered'])}/{c['n_tasks']} tasks provisionable from {c['n_weights']} manifest entries")
            for name, ids in sorted(c["license_required"].items()):
                print(f"  LICENSE  {name:32s} {','.join(ids)}  (TotalSegmentator licensed backend)")
            for name, ids in sorted(c["missing"].items()):
                print(f"  MISSING  {name:32s} {','.join(ids)}")
            return 1 if c["missing"] else 0
        elif args.wcmd == "refresh":
            kw = {"write": not args.dry_run, "update_existing": args.update_existing, "progress": say}
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
