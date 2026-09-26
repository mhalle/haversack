"""Regenerate src/haversack/data/moose_weights.json from a moosez checkout.

The checkpoint is the spec (labels come from each model's own dataset.json at
install time); this manifest holds only what the checkpoint cannot know -
the task name, where to download it, the release tag parsed from the asset
filename, and the modality the name states, which two checkpoints misstate (see
``MODALITY``). Run after updating upstream/MOOSE.

**Every entry in the registry is accounted for.** The registry is parsed with a
pattern, and a pattern matches what it was written against: the name group was
``[a-z0-9_]+`` and upstream's ``clin_ct_ALPACA``, ``clin_ct_PUMA``,
``clin_ct_PUMA4`` and ``clin_mr_FVM`` fell through it without a word, so the
manifest offered 21 of 25 models and nothing said so. Now every ``KEY_URL:``
block in the file must parse as an entry, and one that does not stops the run -
a future layout change (a swapped key order, a name character the pattern does
not admit, a value that is not a string literal) is an error here rather than a
model that quietly goes missing.

**A model that is not offered is recorded with a reason, and the reason is
checked against the asset**, the way tools/gen_totalvibe_manifest.py records its
exclusions: `NOT_OFFERED` names the entry and a category, the generator reads the
archive's layout by Range to confirm the category is true, and a reason that has
stopped being true is an error rather than prose that quietly rots. An entry
that leaves the registry while still listed here is an error too.

Every offered URL is checked with a HEAD before the manifest is written, and one
that does not answer 200 stops the run. The offered assets are never read - the
other generators read each zip's layout by Range, which fails on a dead URL by
itself, and this one has no such reason to touch them - so without the check a
URL that had stopped serving would be recorded and found by a user mid-install.
The HEAD is sent with the same kind of User-Agent the installer sends
(tools/zippeek.py says why that matters).

**Every offered task states the orientation its model was trained in**
(``MODEL_ORIENTATION``), and a task missing from that table stops the run. It is stated,
never inferred: nothing in a checkpoint says it, and an asset's file name (six carry
``_ras_``) is a label someone typed, not a fact - two RAS-era models carry no such mark.
Found 2026-09-26 from a report that ``clin_ct_ribs`` swapped left and right: haversack fed
every MOOSE model the input's own axis order. See the table for what was measured.

**A multi-step moosez workflow is stated too** (``WORKFLOWS``, from moosez's
``workflows.WORKFLOW_REGISTRY``), with the crop model's class VALUES as moosez states them
and their NAMES as the crop checkpoint states them; the ecosystem checks the two agree when
it builds the task.

Usage (stdlib only):
    uv run --no-project python tools/gen_moose_manifest.py [path/to/MOOSE]
"""
import json
import re
import sys
from pathlib import Path

import zippeek

DEST = Path(__file__).parent.parent / "src/haversack/data/moose_weights.json"

#: Registry entries not offered as tasks, and why. The reason is a category the
#: generator checks against the asset; see REASONS.
NOT_OFFERED = {
    # The zip's top level is ``clin_mr_FVM/``, with ``Dataset501_FVM`` one
    # level down. haversack's installer refuses an archive whose top-level name
    # is not the manifest folder (moving it into place would replace whatever
    # owns that name), and moosez's own extractor unpacks into its models
    # directory and then looks for ``Dataset501_FVM`` there, so the model is
    # unusable as packaged upstream too. Checked 2026-09-06 against
    # moosez-v.3.2.0's asset.
    "clin_mr_FVM": "mispackaged",
}
REASONS = {
    "mispackaged": "the archive does not unpack to the folder the registry names, so "
                   "haversack's installer would refuse it and moosez's own extractor "
                   "would not find a model where it looks",
}

#: The orientation (``DICOMOrient`` code, or ``None`` for the input's own axis order) each
#: task's model was trained in, with why. moosez has fed its models two ways. Until
#: 2025-07-18 its reader forced only the first image axis toward patient left, after
#: dicom2nifti had written DICOM input as LAS (``reorient_image``: "in LAS space") - so the
#: models trained then saw LAS. Commit 21331c6 ("Improved Image IO and new RAS Models",
#: 2025-07-18) replaced six models with ones trained on RAS and made moosez reorient every
#: input to RAS, the old models included: upstream issue #232 (2026-03-17) reports
#: ``clin_ct_lungs`` flipped left-right under current moosez, which is that. So each model
#: gets the orientation it was TRAINED in, not the one current moosez feeds it.
#:
#: Measured 2026-09-26 on an NLST chest CT (LPS DICOM, local MPS), mean Dice against
#: ``ts.v2:total``'s same-named structures, input as is (LPS) / LAS / RAS: organs 0.00 / 0.17 /
#: 0.91, cardiac 0.09 / 0.48 / 0.93, muscles 0.00 / 0.00 / 0.95, peripheral_bones 0.00 / 0.00 /
#: 0.94, ribs 0.01 / 0.04 / 0.51 (union 0.54 / 0.83 / 0.83; the rest is rib numbering, not
#: placement), vertebrae union 0.46 / 0.95 / 0.96; lungs 0.57 / 0.99 / 0.00, fast_cardiac 0.32 /
#: 0.80 / 0.25. fast_organs (0.63 all three) and digestive (0.83 / 0.80 / 0.77) were trained
#: with mirroring and barely care; fast_vertebrae 0.78 / 0.76 / 0.76.
_RAS_RETRAIN = "retrained for moosez's RAS input (21331c6, 2025-07-18); measured 2026-09-26"
_RAS_ERA = "published after moosez began feeding every model RAS (2025-07-18); not measured"
_LAS_MEASURED = "trained when moosez fed dicom2nifti's LAS; measured 2026-09-26"
_LAS_ERA = "trained when moosez fed dicom2nifti's LAS (before 2025-07-18); not measured"
MODEL_ORIENTATION = {
    "clin_ct_organs": ("RAS", _RAS_RETRAIN),
    "clin_ct_ribs": ("RAS", _RAS_RETRAIN),
    "clin_ct_muscles": ("RAS", _RAS_RETRAIN),
    "clin_ct_peripheral_bones": ("RAS", _RAS_RETRAIN),
    "clin_ct_vertebrae": ("RAS", _RAS_RETRAIN),
    "clin_ct_cardiac": ("RAS", _RAS_RETRAIN),
    "clin_ct_face": ("RAS", _RAS_ERA),
    "clin_pt_fdg_face": ("RAS", _RAS_ERA),
    "clin_ct_lungs": ("LAS", _LAS_MEASURED),
    "clin_ct_fast_cardiac": ("LAS", _LAS_MEASURED),
    "clin_ct_fast_organs": ("LAS", _LAS_MEASURED + " (trained with mirroring: barely sensitive)"),
    "clin_ct_digestive": ("LAS", _LAS_MEASURED + " (trained with mirroring: barely sensitive)"),
    "clin_ct_fast_vertebrae": ("LAS", _LAS_MEASURED + " (barely sensitive)"),
    "clin_ct_body": ("LAS", _LAS_ERA),
    "clin_ct_ALPACA": ("LAS", _LAS_ERA),
    "clin_ct_PUMA": ("LAS", _LAS_ERA),
    "clin_ct_PUMA4": ("LAS", _LAS_ERA),
    "clin_ct_body_composition": ("LAS", _LAS_ERA),
    "clin_ct_all_bones_v1": ("LAS", _LAS_ERA),
    "clin_pt_fdg_brain_v1": ("LAS", _LAS_ERA),
    "clin_ct_fat_old": ("LAS", _LAS_ERA),
    "preclin_mr_all": ("LAS", _LAS_ERA),
    "preclin_ct_legs": ("LAS", _LAS_ERA),
    # DentalSegmentator's own checkpoint (Dataset112 v100), which its authors run through
    # nnU-Net's predictor with no reorientation - as the dentalsegmentator catalog does, checked
    # against that predictor (99.86 % of voxels). One checkpoint, one contract.
    "clin_ct_dental": (None, "DentalSegmentator's checkpoint: run in the input's own order, "
                             "as its authors' predictor and the dentalsegmentator catalog do"),
}

#: moosez's multi-step workflows (``workflows.WORKFLOW_REGISTRY``, moosez 3.2.2): a crop
#: model's labels cut the image along the named axes, the target model runs on the cut, and
#: its result keeps only the band the ``band`` classes span. Values are moosez's
#: (``fov_intensities``, ``crop_label``), names the crop checkpoint's own. ``clin_ct_face`` is
#: listed there as a one-step workflow and needs nothing.
WORKFLOWS = {
    "clin_ct_body_composition": {
        "crop_task": "clin_ct_fast_vertebrae",
        "crop_classes": {"20": "vertebra_L1", "21": "vertebra_L2", "22": "vertebra_L3",
                         "23": "vertebra_L4", "24": "vertebra_L5"},
        "crop_axes": ["SI"],
        "margin_mm": 0,
        "band_classes": {"22": "vertebra_L3"},
        "band_largest_component": True,
    },
}


#: One registry entry: ``"<name>": {KEY_URL: "<url>", KEY_FOLDER_NAME: "<folder>"``.
#: Names carry uppercase (``clin_ct_PUMA4``); ``\w`` is not used because it
#: also admits non-ASCII letters that no task name here should have.
ENTRY = re.compile(
    r'"(?P<name>[A-Za-z0-9_]+)"\s*:\s*\{\s*'
    r'KEY_URL\s*:\s*"(?P<url>[^"]+)"\s*,\s*'
    r'KEY_FOLDER_NAME\s*:\s*"(?P<folder>[^"]+)"')
#: Every place the registry starts a URL, whatever follows. ``KEY_URL`` alone
#: also appears in the import and in attribute lookups; the colon is what makes
#: it a dict key.
URL_KEY = re.compile(r'KEY_URL\s*:\s*(?P<value>"[^"]*"|[^,\s}]*)')


#: The modality a model takes, read from its registry name (``clin_ct_organs``,
#: ``clin_pt_fdg_brain_v1``, ``preclin_mr_all``) and recorded in the manifest, because the
#: checkpoints do not say it reliably: ``preclin_mr_all``'s dataset.json names its channel
#: "CT", and ``clin_pt_fdg_face``'s says "PET" where its sibling says "PT". Read from the
#: checkpoint alone, both tasks changed modality when they installed (found 2026-09-13, when
#: the segments index put the two answers side by side). ``PT`` is DICOM's code for PET.
MODALITY = re.compile(r"(?:clin|preclin)_(ct|mr|pt|fdg_pt|pt_fdg)_")
MODALITY_CODES = {"ct": "CT", "mr": "MR"}


def modality_of(name: str) -> str:
    """The modality ``name`` states, or a clear stop: a model whose modality cannot be
    read here would report whatever its checkpoint says, the flip this field exists for."""
    m = MODALITY.match(name)
    if m is None:
        raise SystemExit(f"{name}: the name states no modality this generator can read "
                         f"({MODALITY.pattern}) - add the rule rather than ship a model whose "
                         "modality changes when it installs")
    return MODALITY_CODES.get(m.group(1), "PT")


def release_tag(url: str) -> str:
    tag = (re.search(r"_(\d{8})\.zip$", url) or re.search(r"v[\d.]+", url))
    return tag.group(0).strip("_.zip") if tag else "unknown"


def parse(models_py: Path) -> dict:
    """``{name: {url, folder, tag}}`` for every entry in the registry, or a
    clear error naming the ``KEY_URL`` blocks that did not parse as one."""
    src = models_py.read_text(encoding="utf-8")
    out = {}
    for m in ENTRY.finditer(src):
        name, url, folder = m.group("name", "url", "folder")
        out[name] = {"url": url, "folder": folder, "tag": release_tag(url)}
    blocks = [m.group("value") for m in URL_KEY.finditer(src)]
    if len(blocks) != len(out):
        parsed = {e["url"] for e in out.values()}
        stray = [v for v in blocks if v.strip('"') not in parsed] or blocks
        raise SystemExit(
            f"{models_py}: {len(blocks)} KEY_URL blocks but {len(out)} parsed as model "
            f"entries - the registry's layout has changed and this parser would drop the "
            f"rest silently. Not matched: {stray}")
    return out


def unreachable(entries: dict, status=zippeek.head_status) -> dict:
    """``{task: what a HEAD of its URL answered}`` for every entry that did not
    answer 200 - an HTTP status, or the reason when no HTTP answer came."""
    out = {}
    for name, e in entries.items():
        try:
            code = status(e["url"])
        except OSError as err:              # URLError, and a bare socket timeout
            out[name] = str(getattr(err, "reason", err))
            continue
        if code != 200:
            out[name] = code
    return out


def excluded_with_reasons(entries: dict, not_offered: dict, top_level=zippeek.top_level) -> dict:
    """``{name: reason}`` for every entry in ``not_offered``, each reason checked
    against the asset it describes."""
    unknown = sorted(set(not_offered) - set(entries))
    if unknown:
        raise SystemExit(f"{unknown} are listed in NOT_OFFERED but are not in the registry - "
                         "upstream dropped or renamed them; update the list")
    out = {}
    for name, reason in not_offered.items():
        e = entries[name]
        tops = top_level(e["url"])
        if reason == "mispackaged":
            if tops == [e["folder"]]:
                raise SystemExit(
                    f"{name}: {e['url']} unpacks to {e['folder']!r}, exactly the folder the "
                    "registry names - it is not mispackaged; offer it")
            out[name] = (f"{REASONS[reason]} (the archive's top level is {tops}, the "
                         f"registry names {e['folder']!r}; {e['url']})")
        else:
            raise SystemExit(f"{name}: {reason!r} is not a reason this generator checks")
    return out


def generate(models_py: Path, dest: Path = DEST, status=zippeek.head_status,
             top_level=zippeek.top_level, not_offered: dict = NOT_OFFERED,
             orientation: dict = MODEL_ORIENTATION, workflows: dict = WORKFLOWS) -> dict:
    entries = parse(models_py)
    if not entries:
        raise SystemExit(f"{models_py}: no models found - not the moosez registry?")
    dead = unreachable(entries, status)
    if dead:
        lines = [f"  {name}: {entries[name]['url']} -> {answer}" for name, answer in dead.items()]
        raise SystemExit(f"{len(dead)} of {len(entries)} URLs do not answer a HEAD with 200; "
                         "refusing to write a manifest that names them:\n" + "\n".join(lines))
    excluded = excluded_with_reasons(entries, not_offered, top_level)
    offered = [name for name in entries if name not in excluded]
    unstated = sorted(set(offered) - set(orientation))
    stale = sorted(set(orientation) - set(offered))
    if unstated or stale:
        raise SystemExit(
            (f"{unstated} state no model orientation - find out what each was trained in "
             "and add it to MODEL_ORIENTATION (never read it off a file name). " if unstated
             else "") +
            (f"{stale} are in MODEL_ORIENTATION but not offered; remove them." if stale else ""))
    for name, w in workflows.items():
        if name not in offered or w["crop_task"] not in offered:
            raise SystemExit(f"workflow {name}: it and its crop task {w['crop_task']!r} must "
                             "both be offered")
    tasks = {}
    for name in offered:
        code, basis = orientation[name]
        tasks[name] = {**entries[name], "modality": modality_of(name),
                       "model_orientation": code or "native", "orientation_basis": basis}
        if name in workflows:
            tasks[name]["workflow"] = workflows[name]
    dest.write_text(json.dumps({"source": "moosez/models.py",
                                "excluded": excluded,
                                "tasks": tasks}, indent=1) + "\n", encoding="utf-8")
    return tasks


if __name__ == "__main__":
    moose = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "../../upstream/MOOSE") / "moosez/models.py"
    tasks = generate(moose)
    print(f"{len(tasks)} models, {len(NOT_OFFERED)} excluded, every URL answering -> {DEST}")
