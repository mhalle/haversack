"""Regenerate src/haversack/data/mrsegmentator_weights.json from upstream MRSegmentator.

The checkpoint is the spec (labels come from each model's own dataset.json at
install time); this manifest holds only what the checkpoint cannot know - the
task name, where to download it, its digest and version - plus the folder the
flat zip must be installed under, which haversack needs *before* it can read the
checkpoint. That folder is read FROM the zip without downloading it: two Range
requests fetch the central directory and the small members (plans.json for the
dataset name, version.json for the version, the head of fold_0's checkpoint for
the trainer name), so a 1.1 GB asset costs a few MB to describe.

Usage (stdlib only):  uv run --no-project python tools/gen_mrsegmentator_manifest.py
                      [path/to/MRSegmentator/src/mrsegmentator/config.py]
"""
import ast
import json
import re
import struct
import sys
from pathlib import Path

import zippeek

UPSTREAM_CONFIG = ("https://raw.githubusercontent.com/hhaentze/MRSegmentator/master/"
                   "src/mrsegmentator/config.py")
DEST = Path(__file__).parent.parent / "src/haversack/data/mrsegmentator_weights.json"


def parse_registry(config_py: str) -> dict:
    """MODEL_REGISTRY = {name: {version, url, sha256, ...}} out of config.py, by AST."""
    tree = ast.parse(config_py)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MODEL_REGISTRY" for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "MODEL_REGISTRY":
            return ast.literal_eval(node.value)
    raise SystemExit("MODEL_REGISTRY not found in config.py")


def describe_zip(url: str) -> dict:
    cd = zippeek.central_directory(url)
    names = set(cd)
    if "plans.json" not in names or "dataset.json" not in names:
        raise SystemExit(f"{url}: not a flat nnU-Net configuration folder ({sorted(names)[:8]}...)")
    plans = json.loads(zippeek.member_head(url, cd, "plans.json"))
    version = json.loads(zippeek.member_head(url, cd, "version.json")) if "version.json" in names else {}
    ckpt = next(n for n in sorted(names) if re.fullmatch(r"fold_\w+/checkpoint_final\.pth", n))
    head = zippeek.member_head(url, cd, ckpt, limit=3_000_000)      # data.pkl is the first member
    # the value is a pickle BINUNICODE: opcode 'X', a 4-byte little-endian length, the bytes -
    # read exactly that many, or the memo opcode that follows rides along as garbage
    m = re.search(rb"trainer_name.{0,40}?X(.{4})", head, re.DOTALL)
    if not m:
        raise SystemExit(f"{url}: trainer name not found in the head of {ckpt}")
    n, = struct.unpack("<I", m.group(1))
    trainer = head[m.end():m.end() + n].decode()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", trainer):
        raise SystemExit(f"{url}: implausible trainer name {trainer!r}")
    configs = [c for c, v in plans["configurations"].items()
               if v.get("patch_size") and len(v["patch_size"]) == 3 and not v.get("previous_stage")]
    config = "3d_fullres" if "3d_fullres" in configs else configs[0]
    folds = sorted(n.split("/")[0] for n in names if re.fullmatch(r"fold_\w+/checkpoint_final\.pth", n))
    return {"folder": f"{plans['dataset_name']}/{trainer}__{plans['plans_name']}__{config}",
            "weights_version": version.get("weights_version"), "folds": folds,
            "structures": sum(1 for v in json.loads(zippeek.member_head(url, cd, "dataset.json"))["labels"].values()
                              if v != 0)}


def main(argv: list) -> None:
    if argv:
        config_py = Path(argv[0]).read_text()
        source = str(argv[0])
    else:
        with zippeek.open_url(UPSTREAM_CONFIG) as r:
            config_py = r.read().decode()
        source = UPSTREAM_CONFIG
    registry = parse_registry(config_py)
    tasks = {}
    for name, cfg in registry.items():
        if not cfg.get("url"):
            print(f"skip {name}: no download URL", file=sys.stderr)
            continue
        print(f"describing {name} from {cfg['url']} ...", file=sys.stderr)
        d = describe_zip(cfg["url"])
        tag = str(d["weights_version"] if d["weights_version"] is not None else cfg.get("version"))
        if str(cfg.get("version")) != tag:
            print(f"  note: config.py says version {cfg.get('version')}, the zip says {tag}; "
                  "the zip wins (it is what ensure() verifies on disk)", file=sys.stderr)
        rel = re.search(r"/releases/download/([^/]+)/", cfg["url"])
        zen = re.search(r"zenodo\.org/records?/(\d+)", cfg["url"])
        tasks[name] = {"url": cfg["url"], "sha256": cfg.get("sha256"), "tag": tag,
                       "folder": d["folder"],
                       "release": rel.group(1) if rel else (f"zenodo:{zen.group(1)}" if zen else None),
                       "structures": d["structures"]}
        print(f"  {d['folder']}  folds={d['folds']}  structures={d['structures']}", file=sys.stderr)
    DEST.write_text(json.dumps({"source": f"{source} MODEL_REGISTRY + each zip's own plans.json / "
                                          "version.json / fold_0 checkpoint",
                                "tasks": tasks}, indent=1) + "\n")
    print(f"{len(tasks)} models -> {DEST}")


if __name__ == "__main__":
    main(sys.argv[1:])
