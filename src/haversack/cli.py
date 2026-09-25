"""haversack segment IN --task ts.v2:total_fast -o OUT [--spacing 1.0] [--interp nearest|linear]"""
from __future__ import annotations

import sys

import click


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
    do rather than trace back from `import torch`. An engine task (``fastsurfer:asegdkt``)
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
        return shipped.read_text(encoding="utf-8")
    from pathlib import Path
    local = Path(__file__).resolve().parents[2] / local_name
    if local.is_file():
        return local.read_text(encoding="utf-8")
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
        # It parses its own command line, so it gets the rest before click sees any - which
        # also keeps it out of the listed commands.
        from .view import main_view
        return main_view(argv[1:])
    if argv[:1] == ["restore"]:
        # undocumented, the same way: labels from a ranked store, on any grid
        _need_store_extra()
        from .ranked_restore import main_cli
        return main_cli(argv[1:])
    # Standalone, so --help, --version and a usage error print and exit as a command line
    # should, with argparse's exit codes. A command's own status comes back through `state`,
    # because standalone mode discards it - and so does a Ctrl-C (see `_dispatch`).
    state = {}
    try:
        COMMAND_LINE.main(args=argv, prog_name="haversack", obj=state)
    except SystemExit as e:
        if "interrupted" in state:
            raise state["interrupted"] from None
        # A Ctrl-C in click's own code - parsing, a lazy import, the dispatch before the
        # command's function runs - never reaches `_dispatch`: click turns it into Abort, prints
        # "Aborted!" and exits 1, with the interrupt as the Abort's cause. It is the same Ctrl-C,
        # so it ends the process the same way (2026-09-11: raised at 1006 points across the
        # startup of `remote status`, the interrupt came out as exit 1 at 996).
        abort = e.__context__
        if isinstance(abort, click.Abort) and isinstance(abort.__cause__, KeyboardInterrupt):
            raise abort.__cause__ from None
        if "rc" not in state:
            raise
    return state.get("rc", 0)


#: What every command shares: `-h` works as it did under argparse, defaults show in the help,
#: and a wide terminal gets up to 100 columns rather than click's 80.
_CONTEXT = {"help_option_names": ["-h", "--help"], "show_default": True, "max_content_width": 100}


def _verbatim(text: str) -> str:
    """An epilog printed as written: click rewraps a paragraph unless it opens with ``\\b``."""
    return "\b\n" + text


class _Described:
    """What a command and a group share. One with only a one-line summary shows it as its
    description too, rather than an empty page. And `-h` is described in argparse's words,
    lower case like every other option here, where click's are a capitalized sentence."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("help") is None:
            kwargs["help"] = kwargs.get("short_help")
        super().__init__(*args, **kwargs)

    def get_help_option(self, ctx):
        option = super().get_help_option(ctx)
        if option is not None:
            option.help = "show this help message and exit"
        return option


class _Command(_Described, click.Command):
    """Positionals print their own help - click 8.5 added that, argparse's command line had
    it, and it is why 8.5 is the floor."""


class _Group(_Described, click.Group):
    """Subcommands in the order they are defined - `segment` first, as a new user needs them -
    rather than alphabetical."""

    def list_commands(self, ctx):
        return list(self.commands)


def _version_option() -> click.Option:
    """`haversack --version`: the release, as `/v1/version` and the User-Agent report it. The
    argparse command line had none, and a `uvx` smoke run found it failing (2026-09-11)."""
    def show(ctx, param, value):
        if value and not ctx.resilient_parsing:
            from . import __version__
            click.echo(f"haversack {__version__}")
            ctx.exit()
    return click.Option(["--version"], is_flag=True, expose_value=False, is_eager=True,
                        callback=show, help="print the version and exit")


def _complete_task(ctx, param, incomplete):
    """Shell completion for a task name, from the catalog with nothing downloaded - as `tasks`
    lists them. Only qualified names are offered, since a bare one is refused, but what was
    typed is matched against the task's own part too: people start with `total_f`."""
    try:
        from .ecosystems import EcosystemCatalog
        from .weights import WeightsStore
        names = EcosystemCatalog(root=WeightsStore(None, fetch=False).root).names()
    except Exception:                      # completion must never put a traceback in a shell
        return []
    return [n for n in names
            if n.startswith(incomplete) or n.partition(":")[2].startswith(incomplete)]


def _dispatch(handler, cmd: str, subkey: str | None = None):
    """The callback for one command: it hands ``handler`` the parsed command line as one
    namespace - the leaf's parameters over its groups', with the command and subcommand names
    where argparse put them - so that moving to click changed none of the `_cmd_*` bodies."""
    def callback(**_):
        from types import SimpleNamespace
        ctx = click.get_current_context()
        params, c = {}, ctx
        while c is not None:
            params = {**c.params, **params}
            c = c.parent
        args = SimpleNamespace(cmd=cmd, **params)
        if subkey:
            setattr(args, subkey, ctx.info_name)
        try:
            rc = handler(args)
        except KeyboardInterrupt as e:
            if not isinstance(ctx.obj, dict):
                raise
            # Click turns Ctrl-C into "Aborted!" and exit 1. Under argparse it reached the
            # interpreter and the process died of SIGINT - the status on which a shell's `for`
            # loop over a folder of scans stops, where exit 1 goes on to the next scan. So it
            # is held here, and `_run` raises it again once click is done.
            ctx.obj["interrupted"] = e
            return None
        if isinstance(ctx.obj, dict):
            ctx.obj["rc"] = rc
        return rc
    return callback


def _command_line() -> click.Group:
    """The whole command line's shape: every command, its options and their help. Built
    once, at import; each command hands what it parsed to its `_cmd_*` function."""
    root = _Group(
        "haversack", context_settings=_CONTEXT, params=[_version_option()],
        help=('Medical-image segmentation with many model families behind one command: '
              'TotalSegmentator, MOOSE, MRSegmentator, stock nnU-Net, FastSurfer, SynthStrip, '
              'and (experimental, opt-in) VoxTell and MONAI bundles. Runs on Apple Silicon (MPS), '
              'CUDA or CPU; also a local '
              'REST server and a client for one.'),
        epilog=_verbatim("""examples:
  haversack tasks                                          what can be segmented
  haversack segment scan.nii.gz --task ts.v2:total_fast -o labels.seg.nrrd
  haversack segment idc:<crdc_series_uuid> --task ts.v2:total -o labels.seg.nrrd
  haversack serve --port 8790                              a local server (it generates a token); then: haversack remote submit ...
  haversack docs                                           the user guide; `haversack docs weights` for one section"""))

    segment = _Command(
        'segment', callback=_dispatch(_cmd_segment, 'segment'),
        short_help=('segment one or more images: NIfTI, NRRD, MetaImage, a DICOM series '
                    'directory, a URL or a hosted id'),
        help=('Segment one image and write the labels, or several into a directory (batch). '
              'Weights download on first use. The output format follows the extension (.nii.gz, '
              ".nrrd, .seg.nrrd, .mha); labels come back on the input grid, in the input's "
              'orientation.'),
        epilog=_verbatim("""examples:
  haversack segment ct.nii.gz --task ts.v2:total_fast -o labels.seg.nrrd
  haversack segment dicom_dir/ --task ts.v2:total --spacing 1 -o labels.nii.gz
  haversack segment t1.nii.gz --task fastsurfer:asegdkt -o brain.seg.nrrd      (from the fastsurfer venv)
  haversack segment "zenodo:<recid>/amos22.zip!amos22/imagesVa/amos_0575.nii.gz" --task mrsegmentator:base -o amos.seg.nrrd
  haversack segment a.nii.gz b.nii.gz dicom_dir/ --task ts.v2:total_fast --format seg.nrrd -o out/   (batch: out/<name>_total_fast.seg.nrrd)"""),
        params=[
            click.Argument(['input'], nargs=-1, required=True,
                           help=('one or more inputs; several = batch mode. Each is a NIfTI / '
                                 'NRRD / MetaImage file, a DICOM series directory, an http(s) '
                                 'URL (!member reads one file out of a remote zip), or a hosted '
                                 'identifier: idc:<crdc_series_uuid>, '
                                 'zenodo:<recid>/<file>[!member], tcia:..., openneuro:..., '
                                 'hf:<org>/<repo>@<sha>/<path>, s3:<bucket>/<key>, '
                                 'github:<owner>/<repo>@<tag>/<asset>')),
            click.Option(['--task'], required=True, shell_complete=_complete_task,
                         help=('what to segment: a name from `haversack tasks` (total_fast, '
                               'total, fastsurfer:asegdkt, ...), or a path to a stock nnU-Net '
                               'model folder')),
            click.Option(['-o', '--output'],
                         help=('one input: the output file (its extension picks the format). '
                               'Several inputs: an output directory (default: the current '
                               'directory), each written as <input>_<task> in --format')),
            click.Option(['--format'],
                         help=('output type for batch mode (nifti, nrrd, seg.nrrd, mha) - '
                               'required when segmenting several inputs')),
            click.Option(['--spacing'], type=float,
                         help='isotropic output spacing in mm (default: the input grid)'),
            click.Option(['--interp'], type=click.Choice(['linear', 'nearest']), default='linear',
                         help=('logit interpolation for the restore: linear = sub-voxel '
                               'boundaries; nearest = TotalSegmentator semantics')),
            click.Option(['--device'], default='auto',
                         help='cuda, mps, cpu, or auto (the best available)'),
            click.Option(['--dtype'], type=click.Choice(['fp16', 'bf16', 'fp32']), default='fp16',
                         help=('network precision on the nnU-Net path (fp16 runs on MPS; an '
                               'explicit choice is never lowered)')),
            click.Option(['--accumulate'], type=click.Choice(['auto', 'device', 'host']), default='auto',
                         help=('sliding-window accumulator placement: auto (from free device '
                               'memory), device (fastest, needs headroom), host')),
            click.Option(['--batch-size'], default='auto',
                         help='patches per forward pass: auto, or an int'),
            click.Option(['--envelope'], type=float, default=0.0,
                         help=("restrict inference to the body's bounding box plus this margin "
                               'in mm: faster, and not the same labels (cropping re-tiles the '
                               'sliding window); 0 = the whole volume')),
            click.Option(['--model-root'],
                         help=('where model weights live (default: TOTALSEG_WEIGHTS_PATH, '
                               'nnUNet_results, or ~/.totalsegmentator/nnunet/results)')),
            click.Option(['--allow-transpose'], is_flag=True,
                         help=('run a model whose plans permute the axes (transpose_forward). '
                               'Refused by default because the transposed path had not been '
                               'checked against an outside implementation; '
                               'dentalsegmentator:base, totalvibe:vibe_sagittal and '
                               'totalvibe:pancreas need it')),
            click.Option(['--quiet'], is_flag=True, help='no progress or timings on stderr'),
        ])
    root.add_command(segment)

    embed = _Command(
        'embed', callback=_dispatch(_cmd_embed, 'embed'),
        short_help="an image's embedding field (token lattices placed in the patient)",
        help=('Run an encoder - a model whose output is an embedding FIELD, not labels - on one '
              'image, and write the field as <name>.zarr.zip (read it with feldglas). The input is '
              'anything `segment` takes. The encoder\'s weights must be installed first: '
              '`haversack weights fetch <encoder>`. The field inherits the weights\' license.'),
        epilog=_verbatim("""examples:
  haversack encoders                                       the encoders, and which are installed
  haversack weights fetch radar:pretrain                   its weights (1.6 GB, CC BY-NC-SA 4.0)
  haversack embed scan.nii.gz --encoder radar:pretrain -o scan.zarr.zip
  haversack embed idc:<crdc_series_uuid> --encoder radar:pretrain -o scan.zarr.zip --int8"""),
        params=[
            click.Argument(['input'], help='an image file or folder, or a remote input (idc:, s3:, ...) as `segment` takes'),
            click.Option(['--encoder', '-e'], required=True, help='an encoder from `haversack encoders`'),
            click.Option(['-o', '--output'], required=True, help='the field to write (<name>.zarr.zip)'),
            click.Option(['--int8'], is_flag=True, help='store tokens as int8 with a per-channel scale (about half the size)'),
            click.Option(['--device'], default='auto', show_default=True, help='auto, mps, cuda or cpu'),
            click.Option(['--dtype'], type=click.Choice(['fp16', 'fp32']), help='default: fp16 on a GPU, fp32 on the CPU'),
            click.Option(['--slab'], type=click.IntRange(min=0), default=16, show_default=True,
                         help='slices at a time through the full-resolution stages (0: whole volume); exact either way'),
            click.Option(['--json', 'as_json'], is_flag=True, help='print what was done as JSON'),
            click.Option(['--quiet', '-q'], is_flag=True, help='no progress lines on stderr'),
        ])
    root.add_command(embed)
    encoders = _Command(
        'encoders', callback=_dispatch(_cmd_encoders, 'encoders'),
        short_help='list the encoders: what each is, its weights, license and whether they are installed',
        params=[click.Option(['--json', 'as_json'], is_flag=True, help='the list as JSON, with weights and attribution')])
    root.add_command(encoders)

    get = _Command(
        'get', callback=_dispatch(_cmd_get, 'get'),
        short_help='fetch source data (idc:/zenodo:/http...) into the cache, or out to a file',
        help=('Acquire a remote input without segmenting it. With no -o, it lands in the cache '
              '(~/.cache/haversack/inputs) and the path is printed; a later `segment <same id>` '
              'reuses it. With -o, it is also written there: a directory gets the raw fetched '
              'content (a DICOM series stays a directory), an image-extension file (or '
              '--format) is converted to that one volume (a DICOM series -> one NIfTI/NRRD), '
              'geometry preserved; a series `segment` would refuse (a missing slice, a tilted '
              'gantry) is refused, not regridded. The raw data stays cached unless --no-cache.'),
        epilog=_verbatim("""examples:
  haversack get idc:<crdc_series_uuid>                     into cache; prints the path
  haversack get idc:<crdc_series_uuid> -o case1/scan.nii.gz  the series as one NIfTI
  haversack get idc:<crdc_series_uuid> --format nrrd -o out/  converted, auto-named <uuid>.nrrd
  haversack get idc:<crdc_series_uuid> -o raw_dicom/         the raw DICOM series directory
  haversack get ./dicom_dir -o scan.nii.gz                   a local series, converted"""),
        params=[
            click.Argument(['source'], nargs=-1, required=True,
                           help=('one or more remote inputs (several = batch): idc:<uuid>, '
                                 'zenodo:<recid>/<file>[!member], tcia:, openneuro:, '
                                 'hf:<org>/<repo>@<sha>/<path>, s3:<bucket>/<key>[!member], '
                                 'github:<owner>/<repo>@<tag>/<asset>[!member], or an http(s) '
                                 'URL; or a local file or DICOM folder, written the same way')),
            click.Option(['-o', '--output'],
                         help=('where to put it: a directory (raw copy) or a file (converted by '
                               'extension)')),
            click.Option(['--format'],
                         help=('output format (nifti, nrrd, seg.nrrd, mha): convert, and name '
                               'by it into a directory (-o, or else the current one)')),
            click.Option(['--no-cache'], is_flag=True,
                         help='do not keep the raw data in the cache (only with -o)'),
        ])
    root.add_command(get)

    tasks = _Command(
        'tasks', callback=_dispatch(_cmd_tasks, 'tasks'),
        short_help="list every task the catalog can segment, or one task's structures",
        help=('One line per task: name, engine, modality, and whether its weights are on disk '
              "(or, for an engine task, whether the engine's runtime is installed here). With a "
              "task name, prints that task's structures - its segments' ids - one per line in "
              'label order; for a model not installed here, from the segments index. With '
              '--find, which of the tasks listed here produce a segment - from the index, so '
              'nothing is installed: word prefixes in any order by default ("kid left" finds '
              'kidney_left and left_kidney), --glob or --regex for patterns. Ids are compared '
              'folded (case, spaces and hyphens); it is not an ontology, and an abbreviation or '
              'a synonym is not found. --find exits 1 when nothing matches, 2 on a usage error.'),
        epilog=_verbatim("""examples:
  haversack tasks                        every task
  haversack tasks --installed            what runs without a download
  haversack tasks ts.v2:total_fast          the 117 structure names total_fast produces
  haversack tasks --json                 full records: name, ecosystem, engine, modality, n_structures, installed;
                                         `materialized` = the task's definition is known here without a download,
                                         `task_spec` = it is an nnU-Net model (false for FastSurfer, SynthStrip, ...)
  haversack tasks --find pancreas        every task that produces a pancreas, with its label value
  haversack tasks --find "kid left" --installed        ...among what runs without a download
  haversack tasks --find 'vertebra*_[ct]*' --glob      cervical and thoracic vertebrae, either spelling
  haversack tasks --find '^rib_(left|right)_1[0-2]$' --regex
  haversack tasks ts.v2:total --find liver             search one task's segments
  haversack tasks --find vertebra --count              how many, before reading them
  haversack tasks --find vertebra --offset 50          the next page, from the end line"""),
        params=[
            click.Argument(['task'], required=False, shell_complete=_complete_task,
                           help=('a task name: print its structures instead of the list. An '
                                 'nnU-Net task prints `<label>\t<name>` in label order; an '
                                 'engine task, whose labels are its own, prints names only')),
            click.Option(['--model-root'], help='weights root to check for installed models'),
            click.Option(['--installed'], is_flag=True,
                         help='only tasks whose weights are already on disk'),
            click.Option(['--json'], is_flag=True,
                         help=('the per-task records, structures counted (`tasks TASK --json` '
                               "lists one task's); with --find, the search answer")),
            click.Option(['--find'], metavar='TEXT',
                         help=('which of the tasks listed here produce a segment: each id, then '
                               'a line per task - the task, its label value ("1 L2" = value 1 in '
                               'layer 2, where the output overlaps), its modality, and the '
                               "model's own spelling in parentheses where it differs. With a "
                               'task name, searches that task')),
            click.Option(['--glob'], is_flag=True,
                         help='with --find: a shell pattern (*, ?, [...]) against the whole id'),
            click.Option(['--regex'], is_flag=True,
                         help='with --find: a regular expression anywhere in the id'),
            click.Option(['--exact'], is_flag=True,
                         help=("with --find and --glob or --regex: the model's own spelling of "
                               'the id, case included, rather than its folded form')),
            click.Option(['--catalog'],
                         help='with --find: only this catalog (ts.v2, moose, monai, ...)'),
            click.Option(['--modality'],
                         help='with --find: only tasks whose modality contains this (CT, MR, ...)'),
            click.Option(['--limit'], type=int,
                         help='with --find: at most this many ids (default 50)'),
            click.Option(['--offset'], type=int,
                         help=('with --find: the first id to show - the next offset the previous '
                               "answer's end line gives")),
            click.Option(['--count'], is_flag=True,
                         help='with --find: the counts alone, to size a search before reading it'),
        ])
    root.add_command(tasks)

    cite = _Command(
        'cite', callback=_dispatch(_cmd_cite, 'cite'),
        short_help="who made a task's model, its license, and what to cite",
        help=("The credit for one task, from all three layers: the task's own facts (a bundle's "
              'authors, a per-model license), its ecosystem (the group, the repository, the '
              'license, the papers) and the engine that runs it (nnU-Net asks to be cited '
              'alongside every model trained with it). Every reference carries its DOI and '
              'PubMed ID where one exists. Nothing is downloaded.'),
        epilog=_verbatim("""examples:
  haversack cite ts.v2:total_fast           TotalSegmentator's CT paper, nnU-Net, the license
  haversack cite totalvibe:body_regions  the TUM group, European Radiology 2026, Apache-2.0
  haversack cite monai:brats_mri_segmentation --json   the bundle's own references, as data"""),
        params=[
            click.Argument(['task'], shell_complete=_complete_task,
                           help='a task name, in any accepted form'),
            click.Option(['--json'], is_flag=True, help='the full attribution record'),
        ])
    root.add_command(cite)

    rights = _Command(
        'rights', callback=_dispatch(_cmd_rights, 'rights'),
        short_help=('where a remote input comes from, its license, and what to cite - without '
                    'fetching it'),
        help=("What one input's repository says about it: where it came from (the collection or "
              'dataset, its identifier, DOI and version), under what license, and the citation '
              'its publisher asks for. Metadata only - nothing is downloaded. The same record, '
              "plus the bytes' own digest, is written beside every fetched input and into every "
              "result's provenance as `inputs`."),
        epilog=_verbatim("""examples:
  haversack rights idc:19ecafc9-d05a-4c6c-8727-ce1a78190d11   the NLST collection, CC BY 4.0, its DOI
  haversack rights zenodo:7262581/amos22.zip                 the record's license, creators and DOI
  haversack rights openneuro:ds000114/x.nii.gz               CC0, by OpenNeuro's policy"""),
        params=[
            click.Argument(['input'],
                           help=('a remote input: idc:, tcia:, zenodo:, openneuro:, hf:, s3:, '
                                 'gs:, github:')),
            click.Option(['--json'], is_flag=True, help='the record as JSON'),
        ])
    root.add_command(rights)

    catalog = _Group(
        'catalog',
        short_help="maintain the segments index: what every catalog's tasks produce",
        help=("Maintainer commands for the segments index (data/segments.json): every task's "
              'segments - the label value each is written with, its layer where the output '
              'overlaps, and the id its model gives it. `mine` reads them from where each model '
              "states them (a checkpoint's dataset.json read out of its remote zip by Range, a "
              "MONAI bundle's metadata at its curated version, an engine's own table) and records "
              'the version that pins each list; `check` says, offline, which records a catalog '
              'change has made stale. Searching the index is `haversack tasks --find`.'))
    root.add_command(catalog)
    catalog_mine = _Command(
        'mine', callback=_dispatch(_cmd_catalog, 'catalog', 'ccmd'),
        short_help='read segment lists from the models and update the index',
        help=("Mine every catalog (--all), or the catalogs and tasks named, and merge the result "
              'into the index: records not named are left as they are, an unchanged list keeps '
              'its record, and a task that fails keeps its previous record and fails the run. '
              'Naming a catalog also drops its records of tasks it no longer offers. Where a '
              "model is installed at the same version, its own labels must agree with its "
              "archive's, or the list is not recorded. It writes --to, else HAVERSACK_SEGMENTS, "
              'else in a source checkout the packaged index (src/haversack/data/segments.json), '
              'else ~/.config/haversack/segments.json - which search then lays over the '
              'packaged index.'),
        epilog=_verbatim("""examples:
  haversack catalog mine --all                     every catalog (a few KB read per model)
  haversack catalog mine moose                     one catalog
  haversack catalog mine cads:organs ts.v2:total   single tasks
  haversack catalog mine --all --dry-run           report what would change, write nothing"""),
        params=[
            click.Argument(['target'], nargs=-1, shell_complete=_complete_task,
                           help=('catalogs (ts.v2, moose, monai, ...) or tasks '
                                 '(moose:clin_ct_organs) to mine; or give --all')),
            click.Option(['--all'], is_flag=True,
                         help='mine every catalog this build knows, engines enabled here or not'),
            click.Option(['--to'], help='write this index file instead of the default'),
            click.Option(['--dry-run'], is_flag=True, help='report what would change, write nothing'),
            click.Option(['--reread'], is_flag=True,
                         help=("read every archive again, even one whose manifest digest says it "
                               'is unchanged since its record was mined')),
            click.Option(['--model-root'],
                         help=('weights root whose installed models are held against their '
                               'archives (default: the usual weights root)')),
        ])
    catalog.add_command(catalog_mine)
    catalog_check = _Command(
        'check', callback=_dispatch(_cmd_catalog, 'catalog', 'ccmd'),
        short_help="which records the catalogs' current versions have made stale",
        help=("Compare the index with this build's catalogs, offline: every task should have a "
              'record whose version is the one its catalog offers now. Checks the index search '
              'reads - the packaged one with your user index laid over it - or --file; every '
              'task by default, or the catalogs and tasks named. Exits 1 when any record is '
              'stale, missing, or no longer a task, 2 on a usage error.'),
        epilog=_verbatim("""examples:
  haversack catalog check                every task of every catalog
  haversack catalog check moose          one catalog, its vanished tasks included
  haversack catalog check cads:organs    one task"""),
        params=[
            click.Argument(['target'], nargs=-1, shell_complete=_complete_task,
                           help='catalogs or tasks to check (default: all of them)'),
            click.Option(['--file'], help='the index to check (default: the one `mine` writes)'),
        ])
    catalog.add_command(catalog_check)

    weights = _Group(
        'weights', short_help='download model weights ahead of time, or see what can be fetched',
        help=('Weights download on first use; these commands do it ahead of time, or report '
              'what the manifest can provision (some TotalSegmentator tasks are behind its '
              'license).'))
    root.add_command(weights)
    weights_fetch = _Command(
        'fetch', callback=_dispatch(_cmd_weights, 'weights', 'wcmd'),
        short_help='download everything a task needs',
        params=[
            click.Argument(['task'], shell_complete=_complete_task,
                           help=('a task name from `haversack tasks`, or an encoder from '
                                 '`haversack encoders`; every model it needs is fetched')),
            click.Option(['--root', '--model-root'],
                         help="weights root (default: the ecosystem's location)"),
            click.Option(['--from', 'from_path'],
                         help=('an encoder only: install a copy you already have (a file, or a '
                               'directory holding its files), verified against the pinned digest')),
        ])
    weights.add_command(weights_fetch)
    weights_coverage = _Command(
        'coverage', callback=_dispatch(_cmd_weights, 'weights', 'wcmd'),
        short_help='which catalog tasks the manifest can provision')
    weights.add_command(weights_coverage)
    weights_list = _Command(
        'list', callback=_dispatch(_cmd_weights, 'weights', 'wcmd'),
        short_help='installed model weights on disk, with sizes',
        params=[
            click.Option(['--root', '--model-root'],
                         help="weights root (default: the ecosystem's location)"),
        ])
    weights.add_command(weights_list)
    weights_remove = _Command(
        'remove', callback=_dispatch(_cmd_weights, 'weights', 'wcmd'),
        short_help="delete one dataset's installed weights",
        params=[
            click.Argument(['weights_id'], help='a dataset id, e.g. 297, or an encoder name (see `weights list`)'),
            click.Option(['--root', '--model-root'],
                         help='weights root (default: the ecosystem location)'),
            click.Option(['--yes'], is_flag=True, help='do not prompt'),
        ])
    weights.add_command(weights_remove)
    weights_refresh = _Command(
        'refresh', callback=_dispatch(_cmd_weights, 'weights', 'wcmd'),
        short_help='merge newly published weights into the manifest',
        help=("Reads TotalSegmentator's GitHub releases and records new datasets and versions. "
              'From an installed package this writes YOUR manifest '
              '(~/.config/haversack/ts_weights.json, or HAVERSACK_TS_MANIFEST), laid over the '
              'packaged one and kept across upgrades; in a source checkout it edits the '
              "repository's file. Set GITHUB_TOKEN to lift GitHub's 60 requests/hour."),
        params=[
            click.Option(['--repo'], help='GitHub repo to read releases from'),
            click.Option(['--to'], help='write this file instead of the default target'),
            click.Option(['--dry-run'], is_flag=True,
                         help='report what would change, write nothing'),
            click.Option(['--update-existing'], is_flag=True,
                         help=('also repoint datasets at newer releases (changes which weights '
                               'download)')),
        ])
    weights.add_command(weights_refresh)

    serve = _Command(
        'serve', callback=_dispatch(_cmd_serve, 'serve'),
        short_help='run the REST job server on this machine (needs the serve extra)',
        help=('A job server with warm models, progress streaming and a durable result cache; '
              'the same protocol haversack deploys on Modal. Computation needs a bearer token; '
              'reads never do. Without --token the server generates one, prints it, and leaves '
              'it in a file only you can read, which `haversack remote` on this machine picks '
              'up by itself - so personal use has no ceremony, and a proxy or tunnel in front '
              'of the server still faces a token. --no-token runs open, with no protection of '
              'any kind.'),
        epilog=_verbatim("""examples:
  haversack serve                                                   (a token is generated for you)
  HAVERSACK_SERVER=http://127.0.0.1:8790 haversack remote submit scan.nii.gz --task ts.v2:total_fast
  haversack serve --host 0.0.0.0 --token secret                      (other machines pass --token secret)"""),
        params=[
            click.Option(['--host'], default='127.0.0.1',
                         help='interface to listen on (0.0.0.0 for the whole network)'),
            click.Option(['--port'], type=int, default=8790, help='port to listen on'),
            click.Option(['--device'], default='auto', help='cuda, mps, cpu, or auto'),
            click.Option(['--dtype'], type=click.Choice(['fp16', 'bf16', 'fp32']), default='fp16',
                         help='network precision on the nnU-Net path'),
            click.Option(['--cache-models'], type=int, default=5,
                         help='models kept warm across jobs (5 covers a total union)'),
            click.Option(['--model-root'],
                         help='where model weights live (see `segment --model-root`)'),
            click.Option(['--max-pending'], type=int, default=16,
                         help='queue bound; past it POST returns 429'),
            click.Option(['--keep-finished'], type=int, default=50,
                         help='finished jobs (and files) retained'),
            click.Option(['--jobs-ttl-hours'], type=float, default=24.0,
                         help=('how long a job RECORD lasts (keep-finished bounds memory and '
                               'files; this bounds the durable record)')),
            click.Option(['--workdir'], help='job storage (default: a temp directory)'),
            click.Option(['--cache-dir'],
                         help=('result cache (default: ~/.cache/haversack/results; durable, '
                               'unlike the workdir)')),
            click.Option(['--no-result-cache'], is_flag=True,
                         help='compute every request; keep nothing durable'),
            click.Option(['--result-store'], envvar='HAVERSACK_RESULT_STORE',
                         help=('share the result cache through an object store '
                               '(s3://bucket/prefix, gs://..., az://..., or a directory '
                               'as file:///path; credentials from the environment); '
                               '--cache-dir becomes its local copy')),
            click.Option(['--sweep-interval-hours'], type=float, default=24.0,
                         help=('how often this server reclaims bytes in the shared store '
                               'that no result refers to any more (0 turns it off). Only '
                               'with --result-store; a store nothing sweeps only grows')),
            click.Option(['--token'],
                         help=('the bearer token that gates computation (reads stay open); '
                               'HAVERSACK_SERVER_TOKEN when omitted, which keeps it out of '
                               'the process list; generated when neither is given')),
            click.Option(['--allow-transpose'], is_flag=True,
                         help=('serve tasks whose plans permute the axes '
                               '(dentalsegmentator:base, totalvibe:vibe_sagittal, '
                               'totalvibe:pancreas). Deployment policy, so it cannot come from '
                               'a request; without it those tasks are listed and described but '
                               'refuse to run')),
            click.Option(['--no-token'], is_flag=True,
                         help=('run WITHOUT a token: anything that can reach the port can '
                               'compute, a proxy or tunnel in front included. No guards of any '
                               'kind - a machine you trust end to end')),
        ])
    root.add_command(serve)
    serve_store = _Command(
        'serve-store', callback=_dispatch(_cmd_serve_store, 'serve-store'),
        short_help='serve the results in a result store, read-only (needs the serve extra)',
        help=('Every read route of `haversack serve` - results by path or key, their meta, '
              'preview and statistics, the listing - over a result store that writers publish '
              'into, and nothing else: no jobs, no computation, and not one write to the '
              'store, so it runs on a read-only credential, with no GPU and no weights '
              'installed. Keys come from what the writers recorded about their weights, so '
              'run the same haversack version as they do.'),
        epilog=_verbatim("""examples:
  haversack serve-store s3://bucket/results --host 0.0.0.0
  haversack serve-store file:///srv/haversack/store                  (a writer's directory store)"""),
        params=[
            click.Argument(['store'], required=False,
                           help='the result store, e.g. s3://bucket/prefix or file:///path'),
            click.Option(['--result-store'], envvar='HAVERSACK_RESULT_STORE',
                         help='the store, if not given as the argument'),
            click.Option(['--host'], default='127.0.0.1',
                         help='interface to listen on (0.0.0.0 for the whole network)'),
            click.Option(['--port'], type=int, default=8791, help='port to listen on'),
            click.Option(['--cache-dir'],
                         help=('where hits are copied to be served (default: a temporary '
                               'directory; disposable - every read asks the store)')),
            click.Option(['--no-listing'], is_flag=True,
                         help='do not serve /v1/segmentations (the store\'s index stays private)'),
        ])
    root.add_command(serve_store)

    modal = _Group(
        'modal', short_help='deploy the server to your Modal account (needs the modal extra)',
        help=('Deploys the same server to Modal, one GPU worker per engine. Images build in '
              "Modal's cloud; the deploy prints the URL. Costs run while a worker is warm; stop "
              'with `modal app stop haversack-serve --yes`.'))
    root.add_command(modal)
    modal_deploy = _Command(
        'deploy', callback=_dispatch(_cmd_modal, 'modal', 'mcmd'),
        short_help='deploy the packaged app to your Modal account',
        params=[
            click.Option(['--gpu'],
                         help=('worker GPU (default L40S; A10 is the economical fast-mode '
                               'choice)')),
            click.Option(['--app-name'], help='Modal app name (default: haversack-serve)'),
            click.Option(['--cache-volume'],
                         help=('the Modal volume holding the result cache (default: <app '
                               'name>-cache). Result keys hold no app name, so a deployment '
                               'under a new name can adopt an earlier one\'s results; set '
                               'HAVERSACK_RESULTS_KEEP at least as high as the cache it adopts')),
            click.Option(['--scaledown'], type=int,
                         help=('seconds a warm worker lingers after its last job (Modal caps at '
                               '1200)')),
            click.Option(['--token'],
                         help=('gate computation with this bearer token, as `haversack serve '
                               '--token` does, in place of Modal proxy auth, so `haversack '
                               'remote` reaches the deployment; stored in the Modal Secret '
                               '<app name>-token. Prefer HAVERSACK_SERVER_TOKEN, read when this '
                               'is omitted: a flag\'s value shows in the process list')),
            click.Option(['--no-proxy-auth'], is_flag=True,
                         help=('deploy WITHOUT auth - smoke tests only; anyone with the URL can '
                               'spend your GPU credit')),
        ])
    modal.add_command(modal_deploy)
    modal_app_path = _Command(
        'app-path', callback=_dispatch(_cmd_modal, 'modal', 'mcmd'),
        short_help="print the deployable app file's path")
    modal.add_command(modal_app_path)

    remote = _Group(
        'remote', short_help=('talk to a haversack server, local or on Modal (needs the remote '
                            'extra)'),
        help=('The client: upload an image (or name a hosted one) to a server, follow progress, '
              'download the labels. The server is --server or HAVERSACK_SERVER.'),
        params=[
            click.Option(['--server'],
                         help='server URL, e.g. http://gpu-box:8790 (or set HAVERSACK_SERVER)'),
            click.Option(['--token'],
                         help=('bearer token (or HAVERSACK_TOKEN); a server on this machine '
                               'that generated its own token needs neither - the client reads '
                               'it from the file the server left')),
        ])
    root.add_command(remote)
    remote_submit = _Command(
        'submit', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='upload, wait with progress, download the labels',
        params=[
            click.Argument(['input'],
                           help=('a local image file; <source>:<identifier> for a source the '
                                 'server lists, e.g. idc:<crdc_series_uuid>; or result:<key>, a '
                                 'result that server computed, for a task that takes a label map')),
            click.Option(['--task'], required=True,
                         help='a task name the server lists (`haversack remote tasks`)'),
            click.Option(['-o', '--output'],
                         help='where to save the labels (default: <input>_<task>.seg.nrrd); a '
                              '.nii.gz or .nii name gets NIfTI, without the segment names'),
            click.Option(['--deliverables'], metavar='LIST',
                         help=('what the server renders beside the labels, comma-separated: '
                               'preview, statistics - or "none" for the labels alone '
                               "(default: the server's own set). Not part of the result: "
                               'declining a preview recomputes nothing, and asking for one '
                               'later is a cache hit that renders it')),
            click.Option(['--no-wait'], is_flag=True, help='print the job id and return'),
        ])
    remote.add_command(remote_submit)
    remote_encode = _Command(
        'embed', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='an embedding field of an image, computed by the server',
        help=('Submit an embedding job (POST /v1/jobs, kind=embed), follow it, and download the '
              'field (<name>.zarr.zip). The server fetches, runs and caches it like a '
              'segmentation; `haversack remote encoders` lists the encoders it embeds with.'),
        params=[
            click.Argument(['input'],
                           help=('a local image file, or <source>:<identifier> for a source the '
                                 'server lists, e.g. idc:<crdc_series_uuid>')),
            click.Option(['--encoder', '-e'], required=True,
                         help='an encoder the server lists (`haversack remote encoders`)'),
            click.Option(['-o', '--output'],
                         help='where to save the field (default: <input>_<encoder>.zarr.zip)'),
            click.Option(['--int8'], is_flag=True,
                         help='store tokens as int8 with a per-channel scale (another result than fp16)'),
            click.Option(['--no-wait'], is_flag=True, help='print the job id and return'),
        ])
    remote.add_command(remote_encode)
    remote_encoders = _Command(
        'encoders', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='the encoders the server embeds with')
    remote.add_command(remote_encoders)
    remote_status = _Command(
        'status', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help="one job's status, as JSON",
        params=[
            click.Argument(['job_id'], help='the id `submit --no-wait` printed'),
        ])
    remote.add_command(remote_status)
    remote_fetch = _Command(
        'fetch', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help="download a finished job's labels",
        params=[
            click.Argument(['job_id'], help='the id `submit --no-wait` printed'),
            click.Option(['-o', '--output'], required=True,
                         help='where to save the labels; a .nii.gz or .nii name gets NIfTI, '
                              'without the segment names'),
        ])
    remote.add_command(remote_fetch)
    remote_cancel = _Command(
        'cancel', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='cancel an active job / delete a finished one',
        params=[
            click.Argument(['job_id'], help='the id `submit --no-wait` printed'),
        ])
    remote.add_command(remote_cancel)
    remote_tasks = _Command(
        'tasks', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='what the server can segment')
    remote.add_command(remote_tasks)
    remote_results = _Command(
        'results', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='the results the server holds, newest first',
        help=('Lists cached results (GET /v1/segmentations; needs the token): when each was '
              'published, its task, its input and the path of its labels - or its key, for a '
              'result with no path (an upload, several inputs, non-default options). '
              '--identity is computed by the server, not searched, so it is immediate on a '
              'cache of any size; it finds the results that have a path.'),
        params=[
            click.Option(['--identity'], multiple=True,
                         help=('only results of this input: <source>:<identifier>, e.g. '
                               'idc:<crdc_series_uuid>, or a content digest (sha256:<hex>). '
                               'Repeat for several inputs')),
            click.Option(['--task'], help='only results of this task'),
            click.Option(['--limit'], type=int, default=100, show_default=True,
                         help=("stop after this many results; 0 follows the server's cursors "
                               'to the end')),
            click.Option(['--json'], is_flag=True,
                         help='the rows as one JSON document instead of a table'),
        ])
    remote.add_command(remote_results)
    remote_embeddings = _Command(
        'embeddings', callback=_dispatch(_cmd_remote, 'remote', 'rcmd'),
        short_help='the embedding fields the server holds, newest first',
        help=('Lists cached embedding fields (GET /v1/embeddings; needs the token): when each was '
              'published, its encoder, its input and its path - a plain GET of it downloads the '
              'field - or its key, for one with no path (an upload). --identity is computed by the '
              'server, as for `remote results`.'),
        params=[
            click.Option(['--identity'], multiple=True,
                         help=('only embeddings of this input: <source>:<identifier>, or a '
                               'content digest (sha256:<hex>). Repeat for several inputs')),
            click.Option(['--encoder'], help='only embeddings by this encoder'),
            click.Option(['--limit'], type=int, default=100, show_default=True,
                         help=("stop after this many; 0 follows the server's cursors to the "
                               'end')),
            click.Option(['--json'], is_flag=True,
                         help='the rows as one JSON document instead of a table'),
        ])
    remote.add_command(remote_embeddings)

    docs = _Command(
        'docs', callback=_dispatch(_cmd_docs, 'docs'),
        short_help='print a guide (Markdown), whole or one section',
        help=('The guides that ship with the package. The user guide: requirements, install, '
              'weights, the command line, the Python API, engines. The server guide (--server): '
              'the job protocol, its rules, results by path, sources, caches, deploying to '
              "Modal. Pipe either to a pager or a Markdown viewer; a running server's /docs has "
              'the route-by-route OpenAPI reference.'),
        epilog=_verbatim("""examples:
  haversack docs | less
  haversack docs weights            just the section whose heading contains 'weights'
  haversack docs --sections         the section headings
  haversack docs --server           the server guide
  haversack docs --server jobs      one section of it"""),
        params=[
            click.Argument(['topic'], required=False,
                           help=('print only the section whose heading contains this '
                                 '(case-insensitive)')),
            click.Option(['--sections'], is_flag=True,
                         help='list the section headings and exit'),
            click.Option(['--server'], is_flag=True,
                         help='the server guide instead of the user guide'),
        ])
    root.add_command(docs)

    cache = _Group(
        'cache', short_help="show and clean haversack's on-disk stores",
        help=('Lists every store haversack keeps (fetched inputs, server results, engine '
              'checkpoints, and the model-weights root) with sizes, and cleans the transient '
              'ones. Weights are never swept here - remove a model with `weights remove`.'))
    root.add_command(cache)
    cache_list = _Command(
        'list', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help='every store, its location and size')
    cache.add_command(cache_list)
    cache_path = _Command(
        'path', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help='print the store locations, one per line')
    cache.add_command(cache_path)
    cache_clean = _Command(
        'clean', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help='remove cached inputs / results / checkpoints',
        help='Sweeps a transient cache. Shows what would go and needs --yes to act.',
        params=[
            click.Argument(['category'], type=click.Choice(['inputs', 'results', 'checkpoints', 'all']),
                           help='which cache to sweep'),
            click.Argument(['item'], required=False,
                           help='one input to drop, by its spec (only with `inputs`)'),
            click.Option(['--older-than'],
                         help='keep entries touched within this window, e.g. 30d, 12h'),
            click.Option(['--dry-run'], is_flag=True,
                         help='report what would be removed, delete nothing'),
            click.Option(['--yes'], is_flag=True,
                         help='actually delete (without this, it is a dry run)'),
        ])
    cache.add_command(cache_clean)
    def _store_params(extra=()):
        return [
            click.Argument(['store'], required=False,
                           help='the object store, e.g. s3://bucket/prefix'),
            click.Option(['--result-store'], envvar='HAVERSACK_RESULT_STORE',
                         help='the store, if not given as the argument'),
            click.Option(['--cache-dir'],
                         help='the local result cache (default: ~/.cache/haversack/results)'),
            *extra,
            click.Option(['--quiet'], is_flag=True, help='counts only, no per-entry lines'),
        ]

    cache.add_command(_Command(
        'push', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help="copy this machine's results INTO a shared store",
        help=('Publishes the results cached on this machine to an object store, so other '
              'servers can serve them. Each result keeps the generation it already has, '
              'so a pushed entry is still served from this machine without downloading '
              'anything - except an entry cached before generations existed, which is '
              'given a fresh one and costs one download the first time it is read. Safe '
              'to interrupt and rerun: nothing is uploaded twice and nothing in the store '
              'is overwritten unless you ask for it.'),
        params=_store_params([
            click.Option(['--limit'], type=int,
                         help=('stop after this many results are transferred; a key the '
                               'store already has does not use up a slot, so a rerun '
                               'makes progress')),
            click.Option(['--conflict'],
                         type=click.Choice(['skip', 'newer', 'force']), default='skip',
                         help=('what to do when the store already has a key - keep theirs '
                               '(the default, since theirs may be newer), take whichever '
                               'was computed later, or take ours')),
        ])))
    cache.add_command(_Command(
        'pull', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help="copy a shared store's results ONTO this machine",
        help=('Downloads results from an object store into this machine\'s cache, so it '
              'can serve them without asking the store again - to warm a new server, or '
              'to stop depending on the store at all. Entries already complete here cost '
              'one small read and no bytes. The local cache keeps a bounded number of '
              'entries, so pulling more than it holds evicts the oldest as it goes; use '
              '--limit to take the newest few.'),
        params=_store_params([
            click.Option(['--limit'], type=int,
                         help='at most this many entries, newest first'),
        ])))
    cache.add_command(_Command(
        'sweep', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help='reclaim storage in a shared store',
        help=('Deletes bytes in an object store that no cached result refers to any more - '
              'what a republication replaced, and what a deleted entry left behind when it '
              'could not be reclaimed at the time. Refuses to delete anything it cannot '
              'account for: an object it cannot read as an entry, or an index it cannot '
              'find, stops the run rather than emptying the store. Nothing runs this on a '
              'schedule.'),
        params=_store_params([
            click.Option(['--older-than-days'], type=float,
                         help=('also remove ENTRIES last published longer ago than this. '
                               'Without it, only unreferenced bytes go')),
            click.Option(['--grace-hours'], type=float, default=24.0,
                         help=('spare bytes written within this window; they may belong to '
                               'a result being published right now')),
            click.Option(['--empty-index-ok'], is_flag=True,
                         help=('sweep even when the store holds no results at all. Without '
                               'it an empty index stops the run, because "everything was '
                               'deleted" and "wrong prefix" look identical from here')),
        ])))
    cache.add_command(_Command(
        'sync', callback=_dispatch(_cmd_cache, 'cache', 'ccmd'),
        short_help='copy one result store into another',
        help=('Makes DESTINATION hold what SOURCE holds - a server\'s directory store '
              '(file:///path) into a bucket, or any store into any other. Each result is '
              'decided by its history, not by clocks: a result the destination lacks or '
              'holds an older version of is copied; one the destination holds a NEWER '
              'version of is left alone; one computed independently on both keeps the later, '
              'with the other in its history. Deletions travel too. Safe to interrupt and '
              'rerun: nothing is copied twice.'),
        params=[
            click.Argument(['source'], help='the store to copy from'),
            click.Argument(['destination'], help='the store to copy into'),
            click.Option(['--key', 'keys'], multiple=True,
                         help='only this result key (repeatable); default every key'),
            click.Option(['--workers'], type=int, default=8,
                         help='results synced at once (a sync waits on the store, not on work)'),
            click.Option(['--quiet'], is_flag=True, help='counts only, no per-entry lines'),
        ]))
    return root


def _cmd_docs(args) -> int:
    """`haversack docs`."""
    return _docs(args.topic, args.sections, "server" if args.server else "user")


def _cmd_modal(args) -> int:
    """`haversack modal`."""
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
    if args.cache_volume:
        env["HAVERSACK_CACHE_VOLUME"] = args.cache_volume
    if args.scaledown:
        env["HAVERSACK_SCALEDOWN"] = str(args.scaledown)
    if args.no_proxy_auth:
        env["HAVERSACK_PROXY_AUTH"] = "0"
    # Auth is decided here, out loud, and from nothing but --token / HAVERSACK_SERVER_TOKEN:
    # the lower-level knob naming the Secret is dropped from what the deploy inherits, and so
    # are both token variables, which the deploy process has no use for.
    from .cache_admin import TOKEN_FLAG_NOTE, server_token
    token, token_source = server_token(args.token)
    for k in ("HAVERSACK_TOKEN_SECRET", "HAVERSACK_SERVER_TOKEN", "HAVERSACK_TOKEN"):
        env.pop(k, None)
    app_name = env.get("HAVERSACK_APP_NAME") or "haversack-serve"
    if token:
        if token_source == "--token":
            print(TOKEN_FLAG_NOTE, file=sys.stderr)
        secret = f"{app_name}-token"
        _put_token_secret(secret, token)
        env["HAVERSACK_TOKEN_SECRET"] = secret
        print(f"auth: bearer token from {token_source} (Modal Secret {secret})", file=sys.stderr)
    elif args.no_proxy_auth:
        print("auth: NONE - anyone with the URL can compute", file=sys.stderr)
    else:
        print("auth: Modal proxy auth (Modal-Key / Modal-Secret)", file=sys.stderr)
    return subprocess.call([sys.executable, "-m", "modal", "deploy", apppath], env=env)


def _put_token_secret(name: str, token: str) -> None:
    """Create or overwrite the Modal Secret ``name`` as ``{HAVERSACK_TOKEN: token}``, through
    the SDK so the value never becomes a command's argument (``modal secret create`` would
    take it on its command line). The key is the client's variable name, so one value serves
    both ends; the api function alone mounts it (``modal_app.TOKEN_SECRET``)."""
    import modal
    from modal.exception import NotFoundError
    try:
        modal.Secret.from_name(name).update({"HAVERSACK_TOKEN": token})
    except NotFoundError:
        modal.Secret.objects.create(name, {"HAVERSACK_TOKEN": token})


def _cmd_serve(args) -> int:
    """`haversack serve`."""
    if args.token and args.no_token:
        from .errors import InputError
        raise InputError("--token and --no-token contradict each other")
    _need_inference_stack()          # the local server runs models in-process
    from .serve import main_serve
    return main_serve(args)


def _cmd_serve_store(args) -> int:
    """`haversack serve-store`: the read-only server over a result store (no torch needed)."""
    import tempfile
    from pathlib import Path
    from .errors import InputError
    url = args.store or args.result_store
    if not url:
        raise InputError("serve-store: name the store, e.g. `haversack serve-store "
                         "s3://bucket/prefix` (or set HAVERSACK_RESULT_STORE)")
    try:
        import uvicorn
    except ImportError as e:
        raise InputError("the server needs the serve extra: uv sync --extra serve "
                         "(or pip install 'haversack[serve]')") from e
    from .objectcache import read_only_app
    from .serve import _version
    local = Path(args.cache_dir or tempfile.mkdtemp(prefix="haversack-serve-store-"))
    try:
        local.mkdir(parents=True, exist_ok=True)
        app = read_only_app(url, local_dir=local, listing=not args.no_listing)
    except InputError:
        raise
    except OSError as e:
        raise InputError(f"--cache-dir {local}: {e.strerror or e}") from None
    except Exception as e:                     # noqa: BLE001
        raise InputError(f"serve-store {url}: {type(e).__name__}: {e}; check the bucket name "
                         "and that credentials are in the environment") from None
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    try:
        sock = config.bind_socket()
    except OSError as e:
        raise InputError(f"cannot listen on {args.host}:{args.port}: {e}") from e
    print(f"haversack {_version()} serving {url} read-only on http://{args.host}:{args.port}",
          flush=True)
    server.run(sockets=[sock])
    return 0


def _deliverables_arg(value):
    """`remote submit --deliverables`: None when the flag was not given (the server's own
    set), ``[]`` for ``none``, else the names as written. Which names exist, and which
    this server renders, is the SERVER's to say - it refuses the rest naming what it
    offers - so no list of them is kept here to fall behind it."""
    if value is None:
        return None
    from .errors import InputError
    names = [n.strip() for n in str(value).split(",") if n.strip()]
    if not names:
        raise InputError('--deliverables needs names (e.g. preview,statistics), or "none" '
                         "for the labels alone")
    return [] if [n.lower() for n in names] == ["none"] else names


def _cmd_remote(args) -> int:
    """`haversack remote`."""
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
    elif args.rcmd == "encoders":
        for e in c.encoders().get("encoders") or []:
            state = {True: "installed", False: "not installed", None: "-"}.get(e.get("installed"), "-")
            print("\t".join([str(e.get("name")), str(e.get("license")), state]))
    elif args.rcmd == "embed":
        opts = {"int8": True} if args.int8 else {}
        if args.no_wait:
            print(c.submit(args.input, args.encoder, kind="embed", **opts))
            return 0
        stem = args.input[4:16] if args.input.startswith("idc:") else args.input.rsplit(".nii", 1)[0].rstrip("/")
        out = args.output or f"{stem}_{_file_stem(args.encoder)}.zarr.zip"
        final = c.embed(args.input, args.encoder, out, int8=args.int8,
                         on_status=lambda st: print(f"  {st['state']}", file=sys.stderr, flush=True))
        if final["state"] != "done":
            print(f"job ended {final['state']}", file=sys.stderr)
            return 1
        print(f"wrote {out}", file=sys.stderr, flush=True)
        print(out)
    elif args.rcmd == "results":
        import datetime
        import itertools
        if args.limit < 0:
            from .errors import InputError
            raise InputError("--limit must be 0 (every result) or more")
        # a page no larger than what is still wanted, and never over the server's maximum
        rows = c.iter_segmentations(identity=list(args.identity) or None, task=args.task,
                                    page_size=min(args.limit, 1000) if args.limit else 1000)
        rows = list(itertools.islice(rows, args.limit) if args.limit else rows)
        if args.json:
            print(json.dumps({"segmentations": rows}, indent=2))
            return 0
        for e in rows:
            when = e.get("published") or e.get("computed")
            stamp = (datetime.datetime.fromtimestamp(when, datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ") if when else "-")
            where = (e.get("links") or {}).get("labels") or f"key:{e.get('key')}"
            print("\t".join([stamp, str(e.get("task")),
                             ",".join(map(str, e.get("identity") or [])) or "-", where]))
    elif args.rcmd == "embeddings":
        import datetime
        import itertools
        if args.limit < 0:
            from .errors import InputError
            raise InputError("--limit must be 0 (every embedding) or more")
        rows = c.iter_embeddings(identity=list(args.identity) or None, encoder=args.encoder,
                                 page_size=min(args.limit, 1000) if args.limit else 1000)
        rows = list(itertools.islice(rows, args.limit) if args.limit else rows)
        if args.json:
            print(json.dumps({"embeddings": rows}, indent=2))
            return 0
        for e in rows:
            when = e.get("published") or e.get("computed")
            stamp = (datetime.datetime.fromtimestamp(when, datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ") if when else "-")
            where = (e.get("links") or {}).get("embedding") or f"key:{e.get('key')}"
            print("\t".join([stamp, str(e.get("encoder")),
                             ",".join(map(str, e.get("identity") or [])) or "-", where]))
    elif args.rcmd == "status":
        print(json.dumps(c.status(args.job_id), indent=2))
    elif args.rcmd == "fetch":
        print(c.fetch(args.job_id, args.output))
    elif args.rcmd == "cancel":
        print(json.dumps(c.cancel(args.job_id)))
    elif args.rcmd == "submit":
        wanted = _deliverables_arg(getattr(args, "deliverables", None))
        if args.no_wait:
            print(c.submit(args.input, args.task, deliverables=wanted))
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
        final = c.run(args.input, args.task, out, on_status=show, deliverables=wanted)
        if final["state"] == "done":
            print("  done      100%", file=sys.stderr, flush=True)
            # asked for and not delivered is a deviation, and deviations are never silent
            for name, why in (final.get("deliverables_unavailable") or {}).items():
                print(f"note: no {name}: {why}", file=sys.stderr, flush=True)
            print(f"wrote {out}", file=sys.stderr, flush=True)
            print(out)
        else:
            print(f"job ended {final['state']}", file=sys.stderr)
            return 1
    return 0


def _file_stem(name: str) -> str:
    """A name as a file-name fragment: ``radar:pretrain`` -> ``radar_pretrain``."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _cmd_rights(args) -> int:
    """`haversack rights`."""
    import json
    from .errors import InputError
    from .sources import HttpSource, check_identifier, default_sources, parse_input, registry
    reg = registry(default_sources() + [HttpSource()])
    parsed = parse_input(args.input, known=reg)
    # A task name is shaped like a remote input, and `totalvibe:vibe` reached
    # `reg[kind]` as a raw KeyError (2026-09-12); `ts.v2:total` is too short a prefix
    # to parse as one and was called a local file. Both are models, whose rights
    # `cite` reports - say so before either refusal.
    kind = parsed[0] if parsed else str(args.input).partition(":")[0]
    if kind not in reg:
        from .ecosystems import registry as ecosystems
        if kind in ecosystems():
            raise InputError(f"{args.input}: {kind} is a model catalog, not a data source; a model's "
                             f"license and citation come from `haversack cite {args.input}`")
    if parsed is None:
        raise InputError(f"{args.input}: a local file; haversack cannot know its origin or license")
    if kind not in reg:
        raise InputError(f"{args.input}: unknown source kind {kind!r}; rights takes "
                         + ", ".join(f"{p}:" for p in reg if p != "http") + " or an http(s):// URL")
    kind, ident = parsed
    src = reg["http" if kind == "https" else kind]
    check_identifier(src, ident)
    said = src.describe_input(ident)
    record = {"kind": kind, "identity": f"{kind}:{ident}",
              **(said or {"origin": None, "license": None, "cite": []})}
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    print(f"{kind}:{ident}")
    if said is None:
        print(f"  not determined - the {kind} source cannot say where this input came "
              "from or what license it is under")
        return 0
    for k, v in (said.get("origin") or {}).items():
        if v:
            print(f"  {k + ':':<14}{', '.join(map(str, v)) if isinstance(v, list) else v}")
    lic = said.get("license")
    print("  license:      " + ((lic.get("name", "") + (f"  {lic['url']}" if lic.get("url") else ""))
                                if lic else "not stated"))
    for ref in said.get("cite") or []:
        print(f"  cite ({ref.get('for', '')}): {ref.get('text')}")
    return 0


def _cmd_cite(args) -> int:
    """`haversack cite`."""
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


def _cmd_tasks(args) -> int:
    """`haversack tasks`."""
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

    from .errors import InputError
    if args.find is None:
        if (args.glob or args.regex or args.exact or args.catalog or args.modality
                or args.limit is not None or args.offset is not None or args.count):
            raise InputError("--glob, --regex, --exact, --catalog, --modality, --limit, "
                             "--offset and --count shape a search: give --find TEXT as well")
    else:
        # Which tasks produce a segment, from the segments index - nothing installed, nothing
        # downloaded. Here rather than as a `segments` command: the answer is tasks, and a
        # top-level `segments` sat one letter from `segment`, which runs a model.
        from . import segments
        if args.glob and args.regex:
            raise InputError("--glob and --regex are two ways to read --find: give one")
        if args.exact and not (args.glob or args.regex):
            raise InputError("--exact matches the model's own spelling with --glob or --regex; "
                             "a word search always compares folded ids")
        # Among the tasks this catalog lists, as `tasks` itself lists them: searching the
        # whole index named engine tasks that `tasks TASK` then called unknown (2026-09-13).
        only = set(cat.names())
        if args.task:
            only &= {cat.resolve(args.task)[2]}
        if args.installed:
            here = set()
            for name in only:
                try:
                    if installed(cat.info(name)):
                        here.add(name)
                except Exception:                   # one broken entry must not hide the rest
                    pass
            only = here
        res = segments.index().search(
            args.find, mode="glob" if args.glob else "regex" if args.regex else "words",
            field="id" if args.exact else "key", catalog=args.catalog, modality=args.modality,
            tasks=only, limit=50 if args.limit is None else args.limit,
            offset=args.offset or 0, count_only=args.count)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            # A header saying what to expect and an end line proving it all arrived, both on
            # stdout with the results and marked `#` (`grep -v '^#'` drops them): an agent whose
            # harness cuts long output silently finds the end line missing, instead of a short
            # answer that looks complete. The summary used to go to stderr, after the results -
            # the part such a harness cuts first (2026-09-13).
            n, end = res["key_count"], res["end"]
            page = ("counts only" if args.count else
                    f"showing ids {end['shown']}" if end["shown"] != "none" else "no ids here")
            print(f"# {n} id(s), {res['segment_count']} segment(s); {page}")
            for g in res.get("results") or ():
                print(g["key"])
                for s in g["segments"]:
                    # the label value, and the layer where the output overlaps
                    where = str(s["value"]) + (f" L{s['layer']}" if s.get("layer") else "")
                    spelled = "" if s["id"] == g["key"] else f"  ({s['id']})"
                    print(f"  {s['task']:44s} {where:>6s}  {s['modality'] or ''}{spelled}")
            for task, note in (res.get("notes") or {}).items():
                print(f"# note: {task}: {note}")
            if res["open_vocabulary"]:
                print(f"# also: {', '.join(res['open_vocabulary'])} - "
                      f"{res['open_vocabulary_note']}")
            if not n:
                last = "nothing matched"
            elif args.count:
                last = "counts only; the results start at --offset 0"
            elif end["next_offset"] is not None:
                last = f"ids {end['shown']} of {n}; next: --offset {end['next_offset']}"
            elif end["shown"] == "none":
                last = f"no ids at --offset {res['offset']}; there are {n}"
            elif res["offset"] == 0:
                last = f"all {n} id(s) shown"
            else:
                last = f"ids {end['shown']} of {n}, the last page"
            print(f"# end: {last}")
        return 0 if res["key_count"] else 1

    if args.task:
        info = cat.info(args.task)
        names = info.get("structures") or []
        if not names:
            if info.get("unresolved"):
                # installed, but not runnable as it stands - `weights fetch`
                # would do nothing, so say what actually helps
                raise InputError(f"{info['name']}: {info['unresolved']}")
            # Not installed: the segments index has what the model states, read from its
            # archive at the pinned version. "Install it first" was the answer until the index
            # existed, and a review watched --find list what this said it could not (2026-09-13).
            # One reading with the server's describe (`before_install`, 2026-09-22), which
            # also passes over a stale record rather than print what the model no longer says.
            from . import segments
            segments.index()                 # a broken index is its own one-line error here
            known = segments.before_install(cat, args.task)
            if known is None:
                raise InputError(f"{info['name']}: no current structure list until its model "
                                 f"is installed (haversack weights fetch {args.task}); "
                                 "`haversack catalog check` says why the index has none")
            segs = known["segments"]
            print(f"{info['name']}: not installed here; from the segments index - the "
                  "installed model's own labels decide a result", file=sys.stderr)
            if args.json:
                print(json.dumps({"name": info["name"], "structures": [s["id"] for s in segs],
                                  "segments": segs, "from": "segments index"}, indent=2))
            else:
                for s in segs:
                    print(f"{s['value']}\t{s['id']}"
                          + (f"\tlayer {s['layer']}" if s.get("layer") else ""))
            return 0
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
        # One lean record per task, as /v1/tasks has them. A task's structure list and label
        # map (117 names for ts.v2:total) are counted, not listed - `tasks TASK --json` has one
        # task's - and its attribution is left to `haversack cite TASK`: measured 2026-09-13,
        # the listing was 347 KB, 204 KB of it attribution, largely the same citations repeated
        # across a catalog's tasks. A listing answers which tasks exist; credit is per task.
        lean = []
        for i in rows:
            i = dict(i)
            names = i.pop("structures", None)
            i.pop("label_map", None)
            i.pop("attribution", None)
            if names:
                i.setdefault("n_structures", len(names))
            lean.append(i)
        print(json.dumps(lean, indent=2, default=str))
    else:
        for i in rows:
            print(f"{i['name']:44s} {i.get('engine', ''):12s} {i.get('modality') or '':4s} "
                  f"{'installed' if i['installed'] else ''}".rstrip())
    return 0


def _not_into_itself(spec, src, out, convert: bool) -> None:
    """Refuse a `get` write that lands on its own source, which a local source can do by
    accident (`get ./scan.nii.gz -o .`, `get . -o out/`). Every way of landing there is bad: a
    file copied onto itself ends in shutil's traceback, a folder copied into itself nests a
    copy one level deeper on every run, and a conversion onto itself rewrites the input in
    place. Asked of the filesystem, never of the names: on APFS or FAT `Scan.nii` IS
    `scan.nii`, and a symlink or a hard link reaches the same bytes by another name."""
    import os
    from pathlib import Path
    from .errors import InputError
    # A conversion reads all of its source before it writes, so only the file itself is at
    # risk; a copied folder must not land anywhere inside itself. Absolute, so that every
    # folder above a relative -o is asked too (`get .. -o out/` from inside the source).
    here = Path(os.path.abspath(out))
    for p in [here] if convert else [here, *here.parents]:
        try:
            same = os.path.samefile(p, src)
        except OSError:                                  # not there (yet), so not the source
            continue
        if same:
            raise InputError(f"{spec}: -o would write it into itself ({out}); give -o a path "
                             "outside it")


def _cmd_get(args) -> int:
    """`haversack get`."""
    import shutil
    import tempfile
    import unicodedata
    from pathlib import Path
    from . import io, sources
    from .errors import InputError
    say = lambda m: print(f"  {m}", file=sys.stderr, flush=True)
    srcs = args.source

    def get_one(src_spec, out_target=None):
        """Fetch one source; with ``out_target``, write it where ``out_target(fetched path)``
        says - a ``(path, convert)`` pair - and return that path, else the fetched one.

        Every write `get` makes is made here. A local path is a fetch already done and is
        written exactly as a fetched one; until 2026-09-11 it was returned before `-o` was
        looked at, so `get ./series -o scan.nii.gz` exited 0 having written nothing. The raw
        copies were written elsewhere, each fetching without the temporary cache below, so
        `--no-cache` left their data cached."""
        local = sources.parse_input(src_spec) is None
        if local and not Path(src_spec).exists():
            raise InputError(f"not a remote source and not a local path: {src_spec}")
        tmp = tempfile.mkdtemp(prefix="haversack-get-") if args.no_cache and not local else None
        try:
            src = Path(src_spec) if local else sources.materialize(src_spec, cache_dir=tmp, progress=say)
            if out_target is None:
                return src
            out, want_convert = out_target(src)
            _not_into_itself(src_spec, src, out, want_convert)
            if want_convert:
                try:
                    io.convert(src, out)
                except InputError as e:
                    if local or not Path(src).is_dir():
                        # a local folder was never fetched, so there is no "as fetched" to
                        # point to: copying it gives back the folder `segment` refuses too
                        raise
                    # A series `segment` refuses is refused here in its words, and no flag
                    # writes it anyway (decided 2026-09-11): one volume of a gapped series
                    # either misplaces slices (ITK's mean step) or invents them (a filled gap),
                    # and once it is a NIfTI the gap is gone - `segment` on that file cannot
                    # see what it refuses on the source, and a `note:` here would not travel
                    # with the file. The fetched series keeps everything, so name the way to it.
                    raise InputError(f"{e}; `-o <directory>/` without --format copies the "
                                     "series as fetched") from None
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
        if not args.output and not args.format:       # fetched only; a local path is printed back
            print(get_one(src_spec)); return 0
        # --format alone converts into the current directory, as it always did for several
        # sources; for one it was dropped, the cache path printed and nothing converted
        out = Path(args.output or ".")
        to_dir = not args.output or out.is_dir() or str(args.output).endswith("/")
        if args.format or io.image_suffix(out.name):
            ext = io.format_extension(args.format) if args.format else io.image_suffix(out.name)
            if to_dir or not io.image_suffix(out.name):
                out = out / (sources.source_stem(src_spec) + ext)
            print(get_one(src_spec, lambda src: (out, True))); return 0
        if to_dir:                                    # the raw content, named by the source
            name = sources.source_stem(src_spec)
            print(get_one(src_spec, lambda src: (out / (name if src.is_dir() else src.name), False)))
            return 0

        def as_named(src):
            if src.is_dir():
                raise InputError(f"{src_spec} is a directory, not one file; give a directory -o to "
                                 "copy it, or a file with an image extension (or --format) to convert it")
            return out, False
        print(get_one(src_spec, as_named)); return 0

    # batch: several sources
    if not args.output and not args.format:          # no destination: cache each, print paths
        for src_spec in srcs:
            print(get_one(src_spec))
        return 0
    outdir = Path(args.output or ".")                # default the output directory to the cwd
    outdir.mkdir(parents=True, exist_ok=True)
    ext = io.format_extension(args.format) if args.format else None
    # Each name this run has written, folded as APFS and FAT fold names (case, and unicode
    # normalization), with the source written there. Two sources sharing a stem -
    # `a/scan.nii.gz` and `b/scan.nii.gz`, the ordinary layout of a folder of cases once local
    # paths are written at all - land on one name: a converted file replaced by the second, a
    # copied series merged into the first's folder, file over file.
    fold = lambda name: unicodedata.normalize("NFC", name).casefold()
    written = {}

    def into_outdir(src_spec):
        def place(src):
            stem = sources.source_stem(src_spec)
            name = stem + ext if ext else (stem if src.is_dir() else src.name)
            if fold(name) in written:
                raise InputError(f"{outdir / name} was already written for {written[fold(name)]} "
                                 f"in this run; get {src_spec} on its own, with -o naming another file")
            return outdir / name, bool(ext)
        return place

    failures = 0
    for src_spec in srcs:
        try:
            out = get_one(src_spec, into_outdir(src_spec))
            written[fold(out.name)] = src_spec
            print(out)
        except Exception as e:
            failures += 1; print(f"  FAILED {src_spec}: {e}", file=sys.stderr)
    if failures:
        print(f"{failures} of {len(srcs)} sources failed", file=sys.stderr); return 1
    return 0


def _cmd_cache(args) -> int:
    """`haversack cache`."""
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
    if args.ccmd in ("push", "pull", "sweep"):
        return _cmd_cache_move(args)
    if args.ccmd == "sync":
        return _cmd_cache_sync(args)
    return 0


def _cmd_cache_sync(args) -> int:
    """`haversack cache sync SOURCE DESTINATION`."""
    from .errors import InputError
    from .objectcache import SharedResultCache, sync

    def opened(url, check):
        try:
            return SharedResultCache.index(url, check=check)
        except InputError:
            raise
        except Exception as e:                 # noqa: BLE001
            raise InputError(f"cache sync {url}: {type(e).__name__}: {e}; check the bucket "
                             "name and that credentials are in the environment") from None
    # the source is only READ: it needs no write probe, and may be a read-only credential
    src, dst = opened(args.source, False), opened(args.destination, True)

    def say(key, what):
        print(f"  {key[:12]}... {what}", file=sys.stderr, flush=True)
    got = sync(src, dst, keys=list(args.keys) or None, report=None if args.quiet else say,
               workers=args.workers)
    print(f"copied {got['copied']}, fast-forwarded {got['fast_forwarded']}, merged "
          f"{got['merged']}, already current {got['current']}, newer at the destination "
          f"{got['newer_there']}, failed {got['failed']}"
          + (f", format 1 left for its first write {got['legacy']}" if got["legacy"] else "")
          + (f", unreadable {got['unreadable']}" if got["unreadable"] else ""),
          file=sys.stderr)
    return 1 if got["failed"] or got["unreadable"] else 0


def _cmd_cache_move(args) -> int:
    """`haversack cache push` / `cache pull`: migrate results to or from a shared store."""
    from .cache_admin import results_dir
    from .errors import InputError
    from .objectcache import SharedResultCache
    from .serve import ResultCache
    url = args.store or getattr(args, "result_store", None)
    if not url:
        raise InputError(
            f"cache {args.ccmd}: name the store, e.g. `haversack cache {args.ccmd} "
            "s3://bucket/prefix` (or set HAVERSACK_RESULT_STORE)")
    local = ResultCache(getattr(args, "cache_dir", None) or results_dir())
    try:
        shared = SharedResultCache.open(url, local)
    except InputError:
        raise
    except Exception as e:                     # noqa: BLE001
        raise InputError(f"cache {args.ccmd} {url}: {type(e).__name__}: {e}; check the "
                         "bucket name and that credentials are in the environment") from None

    def say(key, what):
        print(f"  {key[:12]}... {what}", file=sys.stderr, flush=True)
    report = None if args.quiet else say
    if args.ccmd == "push":
        got = shared.push(conflict=args.conflict, limit=args.limit, report=report)
        print(f"pushed {got['pushed']}, replaced {got['replaced']}, "
              f"skipped {got['skipped']} (already there), failed {got['failed']}, "
              f"unreadable {got['unreadable']}", file=sys.stderr)
    elif args.ccmd == "pull":
        got = shared.pull(limit=args.limit, report=report)
        print(f"pulled {got['pulled']}, already current {got['current']}, "
              f"failed {got['failed']}, unreadable {got['unreadable']}"
              + (f", evicted again {got['evicted']}" if got.get("evicted") else ""),
              file=sys.stderr)
    else:
        max_age = (args.older_than_days * 86400) if args.older_than_days else None
        got = shared.sweep(max_age_s=max_age, grace_s=args.grace_hours * 3600,
                           allow_empty=args.empty_index_ok)
        print(f"deleted {got['deleted_blobs']} unreferenced object(s), expired "
              f"{got['expired_pointers']} entr{'y' if got['expired_pointers'] == 1 else 'ies'}"
              + (f", left {got['unreadable_pointers']} unreadable object(s) alone"
                 if got["unreadable_pointers"] else ""), file=sys.stderr)
        return 1 if got["unreadable_pointers"] else 0
    return 1 if got["failed"] else 0


def _cmd_segment(args) -> int:
    """`haversack segment`."""
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
        # An output's name depends only on its spec, so every name in the batch is known here,
        # and the loop below writes to exactly these. Until 2026-09-11 two inputs sharing a stem
        # - `a/scan.nii.gz b/scan.nii.gz`, the ordinary layout of a folder of cases - were both
        # written to out/scan_<task><ext>, exiting 0 with only b's labels there. Compared as the
        # filesystem compares names, not by `==`: on APFS or FAT `Scan` and `scan` are one file.
        from .content import _fs_equivalent
        outdir = Path(args.output or ".")
        task_tag = str(args.task).replace(":", "-")
        ext = io.format_extension(args.format)
        outs = [outdir / f"{source_stem(spec)}_{task_tag}{ext}" for spec in inputs]
        first = {}
        for spec, out in zip(inputs, outs):
            key = _fs_equivalent(out.name)
            if key in first:
                raise InputError(f"{first[key][1]} would be written for both {first[key][0]} and "
                                 f"{spec}; segment {spec} on its own, with -o naming another file")
            first[key] = spec, out
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
                       envelope_mm=args.envelope,
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
            from .ranked_output import input_source, segment_to_store, supports_store_output
            if not supports_store_output(args.task):
                raise InputError("a ranked store output is available for nnU-Net tasks and "
                                 "FastSurfer only: this task's engine returns labels, not the "
                                 "distribution a store holds")
            img = resolve(inputs[0])
            r, out = segment_to_store(
                img, args.task, args.output, case=source_stem(inputs[0]),
                source=input_source(inputs[0]),          # the spec as given, not the cache path
                weights=args.model_root, device=args.device, dtype=args.dtype,
                grid=args.spacing if args.spacing else "input", interp=args.interp,
                accumulate=args.accumulate, batch_size=bs,
                envelope_mm=args.envelope, progress=progress)
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
        outdir.mkdir(parents=True, exist_ok=True)     # outdir and outs: named, and checked, above
        failures = 0
        for spec, out in zip(inputs, outs):
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
    return 0


def _cmd_catalog(args) -> int:
    """`haversack catalog`."""
    from collections import Counter
    from pathlib import Path
    from . import segments
    say = lambda m: print(m, file=sys.stderr, flush=True)
    marks = {"added": "+", "changed": "~", "unchanged": "=", "removed": "-", "failed": "!",
             "ok": "=", "stale": "~", "missing": "?", "orphan": "-", "error": "!"}

    def show(rows):
        for name, status, detail in rows:
            print(f"  {marks[status]} {name:48s} {status:9s} {detail}")

    if args.ccmd == "mine":
        from .weights import WeightsStore
        plan = segments.plan(args.target, all_=args.all)
        path = Path(args.to).expanduser() if args.to else segments.target()
        say(f"index: {path}" + (" (dry run)" if args.dry_run else ""))
        say(f"mining {sum(len(t) for _, t, _ in plan)} task(s) from {len(plan)} catalog(s)")
        report = segments.mine(plan, path, root=WeightsStore(args.model_root, fetch=False).root,
                               write=not args.dry_run, prune=args.all, reuse=not args.reread)
        show(report["results"])
        counts = Counter(s for _, s, _ in report["results"])
        say(", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
            + ("; written" if report["written"] else "; nothing written"))
        return 1 if report["failed"] else 0
    path = Path(args.file).expanduser() if args.file else None
    say(f"index: {path}" if path else
        f"index: {segments.PACKAGED}, with {segments.user_path()} laid over it where it exists")
    rows = segments.check(path, targets=args.target)
    show([r for r in rows if r[1] != "ok"])
    counts = Counter(s for _, s, _ in rows)
    say(", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    return 0 if set(counts) <= {"ok"} else 1


def _encoder_owning_weights(name):
    """The encoder ``name`` resolves to, when that encoder downloads weights of its own (an
    nnU-Net encoder uses its task's, and ``weights`` treats its name as the task)."""
    if not name:
        return None
    from .encoders import resolve
    from .errors import InputError
    try:
        spec = resolve(str(name))
    except InputError:
        return None
    return spec if spec.weights else None


def _list_encoder_weights():
    from .cache_admin import _human
    from .encoders import ENCODERS, weights as ew
    for spec in ENCODERS.values():
        for wf in spec.weights:
            p = ew.path(spec, wf)
            if p.is_file():
                print(f"  {spec.name + ' / ' + wf.name:52s} {_human(p.stat().st_size):>10s}  {spec.revision[:12]}")


def _cmd_encoders(args) -> int:
    """`haversack encoders`."""
    import json as _json
    from .encoders import ENCODERS, ALIASES, weights as ew
    from .encoders.serving import describe as describe_encoder, installed_locally
    # the record `GET /v1/encoders` serves per encoder, plus the names fields were once written
    # under; `installed` answers for THIS machine, as the server's does for its own
    rows = [{**describe_encoder(spec, installed_locally(spec)),
             "aliases": sorted(a for a, t in ALIASES.items() if t == spec.name)}
            for spec in ENCODERS.values()]
    if args.as_json:
        print(_json.dumps({"encoders": rows}, indent=1))
        return 0
    for r in rows:
        size = sum(w["bytes"] for w in r["weights"]) / 1e9
        state = ("installed" if r["installed"] else f"not installed ({size:.1f} GB): haversack weights fetch {r['name']}") \
            if r["weights"] else (f"installed (the weights of {r['uses_task']})" if r["installed"]
                                  else f"not installed: haversack weights fetch {r['uses_task']}")
        print(f"{r['name']:24s} {r['license']:18s} {state}")
        print(f"  {r['description']}")
        cite = "; ".join(f"doi:{c['doi']}" for c in r["attribution"]["cite"] if c.get("doi"))
        if cite:
            print(f"  cite: {cite}")
        print("  lattices: " + ", ".join(f"{l['layer']} {l['kernel']} x {l['channels']}" for l in r["lattices"])
              + (f"; also known as {', '.join(r['aliases'])}" if r["aliases"] else ""))
    return 0


def _cmd_embed(args) -> int:
    """`haversack embed`."""
    import json as _json
    from .encoders.pipeline import embed
    progress = None if (args.quiet or args.as_json) else (lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    r = embed(args.encoder, args.input, args.output, device=args.device, dtype=args.dtype, int8=args.int8,
               slab=args.slab, progress=progress)
    if args.as_json:
        print(_json.dumps(r, indent=1))
    else:
        print(f"{r['field']}: {r['encoder']} on {r['device']} ({r['dtype']}{', int8' if r['int8'] else ''}), "
              f"model grid {tuple(r['model_grid'])}, {sum(r['tokens'])} tokens, {r['bytes'] / 1e6:.1f} MB, "
              f"{r['seconds']['total']} s; license {r['license']}")
    return 0


def _cmd_weights(args) -> int:
    """`haversack weights`."""
    from pathlib import Path
    from . import weights_fetch as wfm
    say = lambda m: print(m, file=sys.stderr, flush=True)
    enc = _encoder_owning_weights(getattr(args, "task", None) or getattr(args, "weights_id", None))
    if args.wcmd == "fetch" and enc is not None:
        from .encoders import weights as ew
        got = (ew.adopt(enc, args.from_path, progress=lambda m: say(f"  {m}")) if args.from_path
               else ew.fetch(enc, progress=lambda m: say(f"  {m}")))
        print(f"{enc.name}: weights ready under {ew.directory(enc)} ({len(got)} file(s)); license {enc.license}")
        return 0
    if args.wcmd == "fetch" and getattr(args, "from_path", None):
        from .errors import InputError
        raise InputError("--from installs an encoder's weights; tasks fetch from their catalog")
    if args.wcmd == "remove" and enc is not None:
        from .encoders import weights as ew
        if not args.yes and not click.confirm(f"delete {enc.name}'s weights under {ew.directory(enc)}?", default=False):
            return 1
        gone = ew.remove(enc)
        print(f"{enc.name}: removed {len(gone)} file(s)" if gone else f"{enc.name}: nothing installed")
        return 0
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
        # cascade installs - ts.v2:teeth pulls three models and either number
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
        _list_encoder_weights()
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


#: The command line, built once: `_run` invokes it, and the help tests walk it.
COMMAND_LINE = _command_line()


if __name__ == "__main__":
    sys.exit(main())
