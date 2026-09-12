"""Regenerate src/haversack/data/cads_weights.json from the CADS open-weights release.

CADS (Xu et al., arXiv 2507.22953) publishes nine nnU-Net v2 ResEnc-L checkpoints - one per
model group, T551-T559, 167 CT structures between them - in three license-stratified
releases of the same nine. This catalog offers the ``open`` one (CC BY-SA 4.0,
``cads-model-open_v1.0.0``): the ``research`` weights are non-commercial and the
``reference`` weights add challenge datasets under their own terms, and haversack has no
license gate to offer either behind.

The checkpoint is the spec - labels come from each installed checkpoint's own dataset.json
- so this manifest holds only what the checkpoint cannot say: where to download it, the
digest GitHub publishes, the Dataset folder it unpacks to, the release, and a name and a
description for each model group, which upstream identifies by number. The one fact the
checkpoints leave out that haversack needs - CADS reorients to RAS before the network,
while its plans say SimpleITKIO - is the catalog's to state, in ``CADSEcosystem.spec``.

Every asset is checked by Range against what the catalog assumes of it: one Dataset
parent, named as the asset is, holding one configuration folder; a readable dataset.json
and plans.json; one CT input channel; labels that are not region-based; plans that do not
permute the axes; and no orientation declared (if upstream starts declaring one, the
catalog's RAS has to be looked at again). Every asset must carry the sha256 GitHub
publishes, and the release must hold exactly the nine models, naming 167 structures
between them as the paper says - one that appears, disappears or changes stops the run.

Usage (stdlib only):  uv run --no-project python tools/gen_cads_manifest.py
                      (GITHUB_TOKEN lifts GitHub's 60 requests/hour)
"""
import json
import os
import re
import sys
from pathlib import Path

import zippeek

REPO = "murong-xu/CADS"
TAG = "cads-model-open_v1.0.0"
API = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
#: The license of this release's weights, as its title and cads/LICENSE_MODEL state it.
#: (The repository root's LICENSE_MODEL still says CC BY-NC-SA 4.0 for every variant; the
#: file under cads/ is the one the README points to.)
LICENSE = "CC-BY-SA-4.0"
#: The paper's count: Supplementary Table 2 lists 167 targets across the nine groups.
STRUCTURES = 167
DEST = Path(__file__).parent.parent / "src/haversack/data/cads_weights.json"

#: dataset id -> (task name, what it segments). The names follow upstream's own label-map
#: names (cads/dataset_utils/bodyparts_labelmaps.py: labelmap_part_organs, ...). Two groups
#: hold more than their name says, and the description says so rather than rename them.
CURATED = {
    "551": ("organs", "abdominal organs, the aorta, the inferior vena cava, the portal and "
                      "splenic veins, and the five lung lobes"),
    "552": ("vertebrae", "vertebrae C1 to L5"),
    "553": ("cardiac", "heart chambers and myocardium, esophagus, trachea, pulmonary artery - "
                       "and also the brain, common iliac vessels, small bowel, duodenum, "
                       "colon, urinary bladder and face"),
    "554": ("muscles", "gluteal, deep back and iliopsoas muscles - and also the humeri, "
                       "scapulae, clavicles, femurs, hips and sacrum"),
    "555": ("ribs", "ribs 1 to 12, left and right"),
    "556": ("oar", "radiotherapy organs at risk: spinal canal, larynx, heart, bowel space, "
                   "sigmoid, rectum, prostate, seminal vesicles, mammary glands, sternum, "
                   "psoas major and rectus abdominis"),
    "557": ("head", "head tissues from a brain atlas: white and gray matter, CSF, scalp, "
                    "eyeballs, compact and spongy bone, blood, head muscles (the paper's "
                    "rectus eye muscles)"),
    "558": ("headneck", "head-and-neck organs at risk after HaN-Seg; the open weights are "
                        "weaker here than upstream's reference weights, upstream notes"),
    "559": ("bodyregions", "body composition: subcutaneous tissue, muscle, abdominal and "
                           "thoracic cavities, bones, glands, pericardium, mediastinum, "
                           "spinal cord, breast implants"),
}


def describe_zip(url: str) -> dict:
    """What the catalog needs to know about one asset, read by Range."""
    cd = zippeek.central_directory(url)
    configs = zippeek.config_folders(cd)
    if len(configs) != 1 or configs[0].count("/") != 1:
        raise SystemExit(f"{url}: expected one Dataset*/<trainer>__<plans>__<config> folder, "
                         f"found {configs}")
    config = configs[0]
    for member in ("dataset.json", "plans.json"):
        if f"{config}/{member}" not in cd:
            raise SystemExit(f"{url}: no {member} beside the checkpoint")
    dataset = zippeek.read_json(url, cd, f"{config}/dataset.json")
    plans = zippeek.read_json(url, cd, f"{config}/plans.json")
    labels = dataset.get("labels", {})
    region_based = any(isinstance(v, (list, tuple)) for v in labels.values())
    return {"folder": config.split("/")[0], "config": config.split("/")[1],
            "tops": sorted({n.split("/")[0] for n in cd if not zippeek.is_junk(n)}),
            "channels": dataset.get("channel_names") or dataset.get("modality") or {},
            "region_based": region_based,
            "structures": 0 if region_based else sum(1 for v in labels.values() if int(v) != 0),
            "orientation": dataset.get("orientation"),
            "training": dataset.get("numTraining"),
            "transpose": plans.get("transpose_forward"),
            "folds": zippeek.folds(cd), "checkpoints": zippeek.checkpoints(cd)}


def main() -> None:
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    release = zippeek.get_json(API, headers=headers)
    assets = {a["name"]: a for a in release["assets"]}

    published = {}
    for name in assets:
        m = re.fullmatch(r"Dataset([0-9]+)_[A-Za-z0-9]+\.zip", name)
        if not m:
            raise SystemExit(f"release {TAG} publishes {name!r}, which is not a "
                             "Dataset<id>_<name>.zip model - classify it before regenerating")
        published[m.group(1)] = name
    if set(published) != set(CURATED):
        raise SystemExit(f"release {TAG} publishes models {sorted(published)}, and this catalog "
                         f"names {sorted(CURATED)} - a model gained or lost is a decision, "
                         "not a regeneration")

    tasks = {}
    for idx, (task, what) in CURATED.items():
        a = assets[published[idx]]
        url = a["browser_download_url"]
        print(f"describing {a['name']} ({a['size'] / 1e6:.0f} MB) ...", file=sys.stderr)
        d = describe_zip(url)
        where = f"{a['name']}"
        if d["folder"] != a["name"][:-len(".zip")] or d["tops"] != [d["folder"]]:
            raise SystemExit(f"{where}: unpacks to {d['tops']} around {d['folder']!r}; the "
                             "installer takes exactly the Dataset folder the asset is named for")
        if len(d["channels"]) != 1:
            raise SystemExit(f"{where}: {len(d['channels'])} input channels ({d['channels']}); "
                             "the nnU-Net path takes one image")
        declared = str(next(iter(d["channels"].values()), "")).lower()
        if declared and declared != "ct":
            raise SystemExit(f"{where}: its channel is {declared!r}, not CT")
        if d["region_based"]:
            raise SystemExit(f"{where}: region-based labels are not on the nnU-Net path")
        if d["transpose"] not in (None, [0, 1, 2]):
            raise SystemExit(f"{where}: plans permute the axes ({d['transpose']}), which "
                             "haversack refuses by default - the catalog would have to say so")
        if d["orientation"] is not None:
            raise SystemExit(f"{where}: dataset.json now declares orientation "
                             f"{d['orientation']!r}. CADSEcosystem.spec sets RAS because the "
                             "checkpoints said nothing; look at it again before regenerating")
        digest = str(a.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise SystemExit(f"{where}: GitHub publishes no sha256 for it; every CADS asset had "
                             "one, and an unverifiable download is not recorded silently")
        tasks[task] = {"url": url, "sha256": digest.split(":", 1)[1], "tag": TAG,
                       "folder": d["folder"], "release": TAG, "dataset_id": idx,
                       "modality": "CT", "structures": d["structures"], "description": what,
                       "license": LICENSE}
        print(f"  {task}: {d['folder']}/{d['config']}  folds={d['folds']} {d['checkpoints']}  "
              f"structures={d['structures']}  numTraining={d['training']}", file=sys.stderr)

    total = sum(e["structures"] for e in tasks.values())
    if total != STRUCTURES:
        raise SystemExit(f"the nine models name {total} structures between them; the paper "
                         f"says {STRUCTURES} - the release changed")
    DEST.write_text(json.dumps(
        {"source": f"{API} + each zip's own dataset.json / plans.json (read by Range)",
         "note": "labels are read from each installed checkpoint's own dataset.json, never "
                 "from here; the RAS orientation the models need is stated by "
                 "CADSEcosystem.spec, because the checkpoints do not state it",
         "repository": f"https://github.com/{REPO}",
         "release_assets": sorted(published.values()),
         "tasks": tasks}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(tasks)} models, {total} structures -> {DEST}")


if __name__ == "__main__":
    main()
