"""Regenerate src/haversack/data/dentalsegmentator_weights.json from Zenodo.

DentalSegmentator (Dot et al., Journal of Dentistry 2024; CC BY 4.0) publishes one
stock nnU-Net v2 model for dento-maxillo-facial CT and CBCT as a single Zenodo
record. The checkpoint is the spec - the five structure names come from its own
dataset.json at install time - so this manifest holds only what the checkpoint
cannot say: the download URL, the digest Zenodo publishes, the release, and the
Dataset folder the zip unpacks to, which haversack needs *before* it can read
anything.

Zenodo publishes an **md5** per file and no sha256, so that is what is recorded
and what the installer verifies: verifying the digest the host actually publishes
beats recording a stronger one we could only get by downloading the asset here.

The folder is read out of the archive by Range request (tools/zippeek.py), so
describing the 230 MB asset costs a few kilobytes.

Usage (stdlib only):  uv run --no-project python tools/gen_dentalsegmentator_manifest.py
"""
import json
import sys
import urllib.request
from pathlib import Path

import zippeek

RECORD = "10829675"
API = f"https://zenodo.org/api/records/{RECORD}"
DEST = Path(__file__).parent.parent / "src/haversack/data/dentalsegmentator_weights.json"
#: The catalog name for the record's one model. `base` follows MRSegmentator's
#: naming for a catalog's principal model.
TASK = "base"
#: What the model is, in the shape the other catalogs' entries carry - upstream
#: states it in its paper, not in the checkpoint.
DESCRIPTION = ("dento-maxillo-facial CBCT and CT: upper skull, mandible, upper and "
               "lower teeth, mandibular canal")


def describe_zip(url: str) -> dict:
    """``folder`` (the Dataset directory the zip carries) and the structure count."""
    cd = zippeek.central_directory(url)
    configs = zippeek.config_folders(cd)
    if len(configs) != 1:
        raise SystemExit(f"{url}: expected exactly one trainer__plans__config folder, "
                         f"found {configs}")
    config = configs[0]
    if "/" not in config:
        raise SystemExit(f"{url}: the archive has no Dataset* parent above {config!r} - "
                         "this generator describes MOOSE-shaped zips")
    folder = config.split("/")[0]
    dataset = zippeek.read_json(url, cd, f"{config}/dataset.json")
    labels = dataset.get("labels", {})
    if any(isinstance(v, (list, tuple)) for v in labels.values()):
        raise SystemExit(f"{url}: region-based labels are not on the nnU-Net path")
    return {"folder": folder,
            "structures": sum(1 for v in labels.values() if int(v) != 0),
            "channels": dataset.get("channel_names") or dataset.get("modality"),
            "checkpoints": zippeek.checkpoints(cd), "folds": zippeek.folds(cd)}


def main() -> None:
    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    meta = rec.get("metadata") or {}
    access = meta.get("access_right")
    if access not in ("open", "public"):
        raise SystemExit(f"record {RECORD} is {access!r}, not open - refusing to record it")
    zips = [f for f in rec["files"] if str(f.get("key", "")).endswith(".zip")]
    if len(zips) != 1:
        raise SystemExit(f"record {RECORD} has {len(zips)} zip files, expected 1")
    f = zips[0]
    url = f"https://zenodo.org/records/{RECORD}/files/{f['key']}?download=1"
    checksum = str(f.get("checksum") or "")
    algo, _, digest = checksum.partition(":")
    print(f"describing {f['key']} ({f['size'] / 1e6:.1f} MB) ...", file=sys.stderr)
    d = describe_zip(url)
    entry = {"url": url, "tag": str(meta.get("version") or d["folder"].rsplit("_", 1)[-1]),
             "folder": d["folder"], "release": f"zenodo:{RECORD}",
             "modality": "CT", "structures": d["structures"], "description": DESCRIPTION,
             "license": (meta.get("license") or {}).get("id")}
    if algo in ("md5", "sha256") and digest:
        entry[algo] = digest
    DEST.write_text(json.dumps(
        {"source": f"{API} + the zip's own dataset.json (read by Range)",
         "tasks": {TASK: entry}}, indent=1) + "\n")
    print(f"  {d['folder']}  folds={d['folds']}  checkpoints={d['checkpoints']}  "
          f"channels={d['channels']}  structures={d['structures']}", file=sys.stderr)
    print(f"1 model -> {DEST}")


if __name__ == "__main__":
    main()
