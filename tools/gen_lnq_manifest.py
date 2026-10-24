"""Regenerate src/haversack/data/lnq_weights.json from lnq-segmenter's registry.

lnq-segmenter (Steve Pieper, Isomics; github.com/pieper/lnq-segmenter) publishes four CT
lymph-node models as GitHub releases: per model a small ``meta.zip`` (dataset.json,
plans.json, the fingerprint and a model card) and one zip per fold holding a whole nnU-Net
training checkpoint, ``<folder>/fold_N/checkpoint_final.pth`` - about 815 MB, half of it the
optimizer's momentum. Its registry pins every asset by sha256 and size.

haversack installs only each fold's ``network_weights``, read in place over HTTP Range
(``LNQEcosystem``, :mod:`haversack.torchfree`), so a sha256 of the whole zip cannot be checked
at install. This generator is where that check happens: it downloads every fold ONCE, verifies
it against the registry's sha256 and size, reads its weights with the SAME reader the install
uses, and records their digest (``torchfree.state_digest``: names, dtypes, shapes, values) -
which each install then checks. It also records what the checkpoint says it is (trainer,
configuration, mirroring axes), checked against the registry and the plans, and the folder
name the install publishes under.

Downloads about 15 GB in all (19 folds). Run it where that is cheap - it was run in a Modal
container on 2026-10-22 - with the project's own environment, since it imports haversack:

    .venv/bin/python tools/gen_lnq_manifest.py [--commit SHA] [--only NAME ...]

The registry is read at a pinned commit (``COMMIT``); move it deliberately, and read the
repository's MODELS.md (the authoritative licensing statement) when you do.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

from haversack import fetchlib, torchfree
from haversack.ecosystems import fold_checkpoints_digest
from haversack.sources import RangeFile

REPO = "pieper/lnq-segmenter"
#: "Name the trainer, plans and configuration of every registry model" (2026-10-21), after
#: MODELS.md's license correction (2026-10-19): every model CC BY 4.0.
COMMIT = "274abe275916a4942e47fca95cb1ffb6d0b3b72b"
DEST = Path(__file__).resolve().parents[1] / "src" / "haversack" / "data" / "lnq_weights.json"
#: The registry's license strings, as SPDX. Anything else stops the run: a new license is
#: read, not mapped.
SPDX = {"CC BY 4.0": "CC-BY-4.0"}


def registry_url(commit: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{commit}/src/lnq_segmenter/_registry.json"


def fetch_verified(url: str, sha256: str, size: int, dest: Path) -> Path:
    """Download ``url`` to ``dest``, refusing anything but ``size`` bytes with ``sha256``."""
    h, n = hashlib.sha256(), 0
    with fetchlib.open(url, timeout=1800) as r, open(dest, "wb") as out:
        while chunk := r.read(8 << 20):
            n += len(chunk)
            if n > size:
                raise SystemExit(f"{url}: more than the registry's {size} bytes")
            h.update(chunk)
            out.write(chunk)
    if n != size or h.hexdigest() != sha256:
        raise SystemExit(f"{url}: {n} bytes, sha256 {h.hexdigest()}; the registry says "
                         f"{size} bytes, {sha256}")
    return dest


def describe_fold(zip_path: Path) -> dict:
    """What one fold zip holds, read the way the install reads it: the stored .pth member's
    place in the archive, then ``network_weights`` through a range reader over that span."""
    with zipfile.ZipFile(zip_path) as zf:
        pths = [n for n in zf.namelist() if n.endswith("/checkpoint_final.pth")]
        if len(pths) != 1:
            raise SystemExit(f"{zip_path.name}: {len(pths)} checkpoint_final.pth members")
        offset, size = torchfree.stored_member(zf, pths[0])
    f = open(zip_path, "rb")

    def read(lo, hi):
        f.seek(offset + lo)
        return f.read(hi - lo + 1)

    try:
        ck = torchfree.load(RangeFile.over(read, size, block=8 << 20, max_blocks=2),
                            realize=("network_weights",))
    finally:
        f.close()
    state = ck["network_weights"]
    distinct = {id(a): a.nbytes for a in state.values()}
    return {"member": pths[0], "weights_sha256": torchfree.state_digest(state),
            "weights_bytes": sum(distinct.values()),
            "trainer": ck.get("trainer_name"),
            "configuration": (ck.get("init_args") or {}).get("configuration"),
            "mirroring_axes": list(ck.get("inference_allowed_mirroring_axes") or [])}


def entry_for(model: dict, project: dict, work: Path) -> dict:
    name, version = model["name"], model["version"]
    url = lambda fn: model["weights_url_template"].format(name=name, version=version, filename=fn)
    assets = {a["role"]: a for a in model["weights_assets"]}
    meta = assets["meta"]
    mpath = fetch_verified(url(meta["filename"]), meta["sha256"], meta["size_bytes"],
                           work / meta["filename"])
    with zipfile.ZipFile(mpath) as zf:
        plans = json.loads(zf.read("plans.json"))
        dataset = json.loads(zf.read("dataset.json"))
    folds = {}
    for fold in model["folds"]:
        a = assets[f"fold_{fold}"]
        print(f"  {name} fold {fold}: {a['filename']} ({a['size_bytes'] / 1e6:.0f} MB)",
              file=sys.stderr, flush=True)
        zpath = fetch_verified(url(a["filename"]), a["sha256"], a["size_bytes"],
                               work / a["filename"])
        try:
            d = describe_fold(zpath)
        finally:
            zpath.unlink(missing_ok=True)
        folds[str(fold)] = {"url": url(a["filename"]), "sha256": a["sha256"],
                            "size": a["size_bytes"], **d}
    # one model: every fold the same trainer and configuration, and what the registry says
    seen = {(f["trainer"], f["configuration"], tuple(f["mirroring_axes"])) for f in folds.values()}
    if len(seen) != 1:
        raise SystemExit(f"{name}: folds disagree about what model they are: {sorted(seen)}")
    (trainer, configuration, mirroring), = seen
    for key, have in (("trainer_class", trainer), ("plans_identifier", plans["plans_name"]),
                      ("configuration", configuration)):
        if model.get(key) and model[key] != have:
            raise SystemExit(f"{name}: the registry's {key} is {model[key]!r}, the "
                             f"checkpoint's {have!r}")
    folder = f"{plans['dataset_name']}/{trainer}__{plans['plans_name']}__{configuration}"
    dsid = re.match(r"Dataset(\d+)_", plans["dataset_name"])
    labels = {k: v for k, v in dataset["labels"].items() if str(v) != "0"}
    if not labels or model.get("modality") != "CT":
        raise SystemExit(f"{name}: expected a CT model with labels, found {labels} / "
                         f"{model.get('modality')}")
    lic = SPDX.get(model.get("license"))
    if lic is None:
        raise SystemExit(f"{name}: license {model.get('license')!r} - read MODELS.md and add it")
    combined = fold_checkpoints_digest(folds)            # the install's own formula
    return {"url": url(meta["filename"]), "sha256": meta["sha256"], "size": meta["size_bytes"],
            "folder": folder, "tag": version, "release": f"{name}-{version}",
            "dataset_id": dsid.group(1) if dsid else None, "modality": "CT",
            "description": model.get("recommended_use"),
            # the bare SPDX id: a task licensed as its catalog is then credited with the
            # catalog's fuller record (attribution's repeat rule), the license URL included
            "license": lic,
            "attribution": model.get("attribution"),
            "segment": {k: model.get(k) for k in ("segment_name", "snomed_code", "snomed_term",
                                                  "color")},
            "mirroring_axes": list(mirroring),
            "weights_sha256": combined,
            "folds": folds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", default=COMMIT)
    ap.add_argument("--only", nargs="*", help="models to (re)describe; the others are kept")
    ap.add_argument("--out", type=Path, default=DEST)
    args = ap.parse_args()
    registry = fetchlib.get_json(registry_url(args.commit), timeout=60)
    if registry.get("schema") != "lnq-segmenter-registry/1":
        raise SystemExit(f"registry schema {registry.get('schema')!r}")
    project = registry.get("project") or {}
    names = [m["name"] for m in registry["models"]]
    if len(names) != len(set(names)):
        # the manifest holds one version per task; keeping whichever came last would be a
        # choice nobody made (2026-10-22)
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"the registry lists {dup} more than once - choose a version per "
                         "model here before regenerating")
    old = (json.loads(args.out.read_text(encoding="utf-8")) if args.only and args.out.exists()
           else {"tasks": {}})
    tasks = dict(old.get("tasks") or {})
    with tempfile.TemporaryDirectory(prefix="gen-lnq-") as tmp:
        for model in registry["models"]:
            if args.only and model["name"] not in args.only:
                continue
            print(model["name"], file=sys.stderr, flush=True)
            tasks[model["name"]] = entry_for(model, project, Path(tmp))
    args.out.write_text(json.dumps(
        {"source": registry_url(args.commit), "repository": f"https://github.com/{REPO}",
         "commit": args.commit,
         "release_status": project.get("release_status_note"),
         "note": "labels are read from each installed checkpoint's own dataset.json, never "
                 "from here. Each fold's url/sha256/size are the registry's, verified on a full "
                 "download by this generator; weights_sha256 is torchfree.state_digest of its "
                 "network_weights, which the install checks after reading them in place. The "
                 "task-level weights_sha256 digests the folds' digests.",
         "tasks": dict(sorted(tasks.items()))}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}: {len(tasks)} models", file=sys.stderr)


if __name__ == "__main__":
    main()
