"""Regenerate src/haversack/data/totalvibe_weights.json from the TotalVibeSegmentator release.

TotalVibeSegmentator (Graf et al., Apache-2.0) publishes its nnU-Net v2 weights as
one zip per dataset id on the VibeSegmentator GitHub release. The checkpoint is
the spec - labels, and here also the orientation the model must see, come from
each zip's own dataset.json at install time - so this manifest holds only what
the checkpoint cannot say: where to download it, the folder it unpacks to, the
release it belongs to, and the two facts upstream states in prose rather than in
the archive (the task name and the modality; every model declares its channel as
`any` or `ct`, which is a contrast claim, not a modality).

Each zip's top level is the bare ``<trainer>__<plans>__<config>`` folder, with no
``Dataset<id>`` parent - upstream's own downloader supplies that parent, and so
does :class:`haversack.ecosystems.TotalVibeEcosystem`, which is why ``folder``
here is the parent this catalog creates rather than something read from the zip.

**Every base asset is accounted for.** The release carries more models than this
catalog offers, and a drop with no reason recorded is indistinguishable from an
oversight - so `CURATED` and `NOT_OFFERED` between them must name every `NNN.zip`
in the release, and the generator fails if one appears that is in neither. Each
reason is then *verified against the asset itself*: a model called multi-channel
must really declare more than one input channel, one called unreadable must
really be missing its metadata, and one called single-channel-but-not-integrated
must really be runnable - so a wrong reason is an error here rather than prose
that quietly rots. The `_1`/`_2` assets are additional folds fetched through
upstream's `other_downloads.json`; haversack's default is fold 0, so only the
base asset is recorded.

The generator also refuses a curated model that does not declare a valid
`orientation`, because a missing one degrades silently into "do not reorient" -
the left/right mirroring this project has been bitten by three times.

Usage (stdlib only):  uv run --no-project python tools/gen_totalvibe_manifest.py
                      (GITHUB_TOKEN lifts GitHub's 60 requests/hour)
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import zippeek

REPO = "robert-graf/VibeSegmentator"
TAG = "v1.0.0"
API = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
#: The licence of the repository the release is published from. Recorded per task
#: because it travels with the weights, and `info()` publishes it.
LICENSE = "Apache-2.0"
DEST = Path(__file__).parent.parent / "src/haversack/data/totalvibe_weights.json"

#: dataset id -> (task name, modality, what it segments). The names are ours:
#: upstream identifies these models by number only, so the number is recorded in
#: every entry and this table is the whole mapping.
CURATED = {
    "100": ("vibe", "MR", "whole-torso VIBE MRI (axial), the flagship model"),
    "099": ("vibe_sagittal", "MR", "whole-torso VIBE MRI, sagittal acquisition"),
    "012": ("ct_bones", "CT", "whole-body bone and organ CT at 0.8 mm"),
    "278": ("body_regions", "MR", "body-region splitter (head, thorax, abdomen, ...)"),
    "511": ("vertebrae", "MR", "vertebra and intervertebral-disc instances, C1 to the sacrum"),
    "538": ("feet_bones", "CT", "bones of the feet, CT at 0.8 mm"),
    "540": ("pancreas", "MR", "pancreas"),
}
#: Every other base asset in the release, and why it is not offered. The reason is
#: a category the generator checks against the asset; see REASONS.
NOT_OFFERED = {
    "085": "multi-channel", "086": "multi-channel", "282": "multi-channel",
    "098": "unreadable", "520": "unreadable",
    "001": "not-integrated", "011": "not-integrated", "080": "not-integrated",
    "087": "not-integrated", "512": "not-integrated", "524": "not-integrated",
    "527": "not-integrated",
}
REASONS = {
    "multi-channel": "takes more than one input image; multi-channel nnU-Net input is "
                     "not on haversack's nnU-Net path",
    "unreadable": "the published asset carries no dataset.json / plans.json beside its "
                  "checkpoint, so nothing can describe or run it",
    "not-integrated": "single-channel and would run, but is outside the curated set - "
                      "never exercised here, so not offered as a task",
}


class Unreadable(Exception):
    """The asset does not carry the metadata needed to describe or run it."""


def describe_zip(url: str) -> dict:
    cd = zippeek.central_directory(url)
    configs = zippeek.config_folders(cd)
    if len(configs) != 1 or "/" in configs[0]:
        raise Unreadable(f"expected one top-level trainer__plans__config folder, found {configs}")
    config = configs[0]
    if f"{config}/dataset.json" not in cd or f"{config}/plans.json" not in cd:
        raise Unreadable("no dataset.json / plans.json beside the checkpoint")
    dataset = zippeek.read_json(url, cd, f"{config}/dataset.json")
    labels = dataset.get("labels", {})
    channels = dataset.get("channel_names") or dataset.get("modality") or {}
    plans = zippeek.read_json(url, cd, f"{config}/plans.json")
    orientation = dataset.get("orientation")
    # nnU-Net's ignore label is a training device that no output can carry, so it
    # is not a structure - `vibe` and `vibe_sagittal` declare one
    ignore = dataset.get("ignore_label")
    ignored = {int(ignore)} if ignore is not None else set()
    ignored |= {int(v) for k, v in labels.items() if str(k).lower() == "ignore"}
    return {"config": config, "dataset_name": plans.get("dataset_name"),
            "n_channels": len(channels), "channels": channels, "labels": labels,
            "region_based": any(isinstance(v, (list, tuple)) for v in labels.values()),
            "structures": sum(1 for v in labels.values()
                              if int(v) != 0 and int(v) not in ignored),
            "ignored_labels": sorted(ignored),
            "orientation": "".join(orientation) if isinstance(orientation, list) else orientation,
            "roi": dataset.get("roi"),
            "folds": zippeek.folds(cd), "checkpoints": zippeek.checkpoints(cd)}


def valid_orientation(code) -> bool:
    """Three letters naming each anatomical axis once - the same rule
    haversack.ecosystems.TotalVibeEcosystem applies when it reads this back."""
    code = str(code or "").upper()
    return len(code) == 3 and all(sum(c in pair for c in code) == 1
                                  for pair in ("RL", "AP", "SI"))


def base_assets(assets: dict) -> list:
    """The `NNN.zip` base assets, without upstream's `_1`/`_2` extra-fold ones."""
    # [0-9]+ , not \d{3}: a four-digit or two-digit asset must still enter the
    # completeness check, and \d would also admit non-ASCII digits
    return sorted(n[:-4] for n in assets if re.fullmatch(r"[0-9]+\.zip", n))


def main() -> None:
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json"})
    if os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        release = json.load(r)
    assets = {a["name"]: a for a in release["assets"]}

    published = base_assets(assets)
    accounted = set(CURATED) | set(NOT_OFFERED)
    unaccounted = [i for i in published if i not in accounted]
    if unaccounted:
        raise SystemExit(
            f"release {TAG} publishes {unaccounted} which are neither curated nor listed in "
            "NOT_OFFERED - classify them (a silent drop is indistinguishable from an oversight)")
    missing = sorted(accounted - set(published))
    if missing:
        raise SystemExit(f"{missing} are named here but not published in release {TAG}")

    def describe(idx: str):
        a = assets[f"{idx}.zip"]
        print(f"describing {idx}.zip ({a['size'] / 1e6:.0f} MB) ...", file=sys.stderr)
        return a, describe_zip(a["browser_download_url"])

    tasks = {}
    for idx, (name, modality, what) in CURATED.items():
        a, d = describe(idx)
        if d["n_channels"] != 1:
            raise SystemExit(f"{idx}.zip: {d['n_channels']} input channels ({d['channels']}) - "
                             "multi-channel models do not run on the nnU-Net path; "
                             "move this id to NOT_OFFERED")
        if d["region_based"]:
            raise SystemExit(f"{idx}.zip: region-based labels are not on the nnU-Net path")
        if not valid_orientation(d["orientation"]):
            raise SystemExit(
                f"{idx}.zip: declares orientation {d['orientation']!r}, which is not three "
                "letters naming each anatomical axis once. haversack reads this key to decide "
                "how to reorient, and a missing or malformed one degrades silently into 'do "
                "not reorient' - a left/right mirror. Refusing rather than shipping it")
        # the curated modality is prose upstream states outside the checkpoint, so it
        # cannot be read - but where the checkpoint DOES name one, it must not disagree
        declared = str(next(iter(d["channels"].values()), "")).lower()
        if declared in ("ct", "mr", "mri", "pt", "petsuv") and not declared.startswith(modality.lower()):
            raise SystemExit(f"{idx}.zip: channel says {declared!r} but this table says "
                             f"{modality!r} - one of them is wrong")
        entry = {"url": a["browser_download_url"], "tag": TAG, "folder": f"Dataset{idx}",
                 "release": TAG, "dataset_id": idx, "modality": modality,
                 "structures": d["structures"], "description": what,
                 "license": LICENSE}
        digest = str(a.get("digest") or "")
        if digest.startswith("sha256:"):
            entry["sha256"] = digest.split(":", 1)[1]
        tasks[name] = entry
        print(f"  {name}: {d['dataset_name']}/{d['config']}  folds={d['folds']} "
              f"{d['checkpoints']}  orientation={d['orientation']}  "
              f"structures={d['structures']}  digest={'yes' if 'sha256' in entry else 'NONE'}",
              file=sys.stderr)

    # every other published asset, with its reason CHECKED against the asset
    excluded = {}
    for idx, reason in sorted(NOT_OFFERED.items()):
        try:
            _a, d = describe(idx)
        except Unreadable as e:
            if reason != "unreadable":
                raise SystemExit(f"{idx}.zip is unreadable ({e}) but is recorded as {reason!r}")
            excluded[idx] = REASONS["unreadable"]
            print(f"  {idx}: unreadable ({e})", file=sys.stderr)
            continue
        if reason == "unreadable":
            raise SystemExit(f"{idx}.zip reads fine but is recorded as 'unreadable'")
        if reason == "multi-channel" and d["n_channels"] < 2:
            raise SystemExit(f"{idx}.zip has {d['n_channels']} channel(s) but is recorded "
                             "as multi-channel")
        if reason == "not-integrated" and d["n_channels"] != 1:
            raise SystemExit(f"{idx}.zip has {d['n_channels']} channels - it is excluded for a "
                             "stronger reason than 'not-integrated'; say so")
        text = REASONS[reason]
        if reason == "multi-channel":
            text = (f"{d['n_channels']} input channels ({', '.join(d['channels'].values())}); "
                    + REASONS[reason].split("; ", 1)[1])
        if d.get("roi"):
            text += f"; also a cascade cropped by dataset {d['roi']}'s output"
        excluded[idx] = text
        print(f"  {idx}: {reason} ({d['n_channels']} ch, {d['structures']} structures)",
              file=sys.stderr)

    no_digest = sorted(n for n, e in tasks.items() if "sha256" not in e)
    DEST.write_text(json.dumps(
        {"source": f"{API} + each zip's own dataset.json / plans.json (read by Range)",
         "note": "labels and the orientation the model must see are read from each "
                 "installed checkpoint's own dataset.json, never from here",
         "repository": f"https://github.com/{REPO}",
         "release_assets": published,
         "unverified": {"tasks": no_digest,
                        "why": "the GitHub API publishes no digest for these assets, so their "
                               "download cannot be checked against one; every other task is "
                               "verified against the sha256 the release states"},
         "excluded": excluded,
         "tasks": tasks}, indent=1) + "\n")
    print(f"{len(tasks)} models, {len(excluded)} excluded -> {DEST}")
    if no_digest:
        print(f"NOTE: no digest published for {no_digest} - recorded in 'unverified'")


if __name__ == "__main__":
    main()
