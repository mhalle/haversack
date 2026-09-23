"""Generate a TotalSegmentator task registry (``src/haversack/data/ts_*_tasks.json``) from
upstream's own source, read out of its published wheel.

    uv run --no-project python tools/gen_ts_registry.py generate ts.v3 --write
    uv run --no-project python tools/gen_ts_registry.py check ts.v3      # exit 1 on drift

Why a new generator (2026-09-21). ``ts_tasks.json`` (the ``ts.v2`` catalog) was written by
``../medseg/nnunet-inference-mlx/scripts/refresh_ts_registry.py`` against TotalSegmentator
2.13.0: it exec'd the ``if task == ...`` chain in ``python_api.totalsegmentator``. By 2.18.0
that chain is gone - task dispatch is the data table ``TASK_CONFIGS`` in
``map_tasks_config.py``, read by ``python_api.get_task_config`` - and that script writes
``source`` where haversack's registries say ``lineage``. ``ts.v2`` is deliberately NOT
regenerated here: its entries are what its result keys and label digests were built on.

What this reads, all from the wheel PyPI publishes for the pinned version (digest checked
against PyPI's own record), never from an import - so no torch, and no TotalSegmentator
install, just like the other ``--no-project`` generators:

- ``map_tasks_config.py`` and ``map_to_binary.py``: pure data, exec'd whole.
- ``python_api.get_task_config``: upstream's own resolution of a task and its fast/fastest
  mode, exec'd from its source (found by AST) - the same trick the old generator used on the
  if-chain, so the answer is upstream's code and not a restatement of it. ``plans`` and
  ``model_size`` are passed as ``totalsegmentator()``'s own signature defaults, read by AST.
- ``nnunet.py``: which part class map and dataset->part map a multi-model task paints with -
  the ``if task_name == ...: class_map_parts = ...; map_taskid_to_partname = ...`` chain,
  read by AST. ``total_v3`` has its own (``class_map_5_parts_total_v3``: v2's parts with
  vertebrae label 2 renamed ``vertebrae_L6``). And the sliding window's tile step - the
  ``if task_name in [...]: step_size = 0.8 else: step_size = 0.5`` rule, read by AST - as each
  entry's ``step_size`` (added 2026-09-21: matching it took ts.v3:total from 99.86 % to
  99.98 % voxel agreement with upstream).

Each entry states ``models: {dataset: {trainer, plans}}`` - v3's Datasets 831-836 each ship
an ``nnUNetPlans`` AND an ``nnUNetResEncUNetLPlans_8`` 3d_fullres folder, and haversack's
resolver refuses a dataset it cannot narrow to one (``tasks.resolve_model_folder``).

``generate`` then checks every entry against the checkpoints themselves (``--no-verify``
skips it): each dataset's plans must be 3d_fullres at the task's resample spacing, and its
``dataset.json`` labels must be exactly what the entry paints them as. Read by range request
through ``tools/zippeek.py`` from the URLs in ``ts_weights.json`` - a few MB, no download.

Task names are the makers' (the user's naming policy): a catalog maps upstream task names to
its own only where the catalog's version IS the difference - ``ts.v3``'s ``total`` is
upstream's ``total_v3``, whose ``_v3`` the catalog name ``ts.v3`` carries. Upstream's
``total`` is still v2 (Datasets 291-295) and stays ``ts.v2:total``.
"""
import argparse
import ast
import hashlib
import io
import json
import sys
import textwrap
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import zippeek  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "src" / "haversack" / "data"
GENERATOR_VERSION = 1

#: catalog -> the TotalSegmentator release it is generated from, the file it is written to,
#: and {upstream task: the name this catalog lists it under}.
CATALOGS = {
    "ts.v3": {"version": "2.18.0", "path": DATA / "ts_v3_tasks.json",
              "tasks": {"total_v3": "total"}},
}
MODES = ("default", "fast", "fastest")
#: what a task config may say beyond what haversack mirrors for these tasks. A task that
#: crops, cascades, masks or runs another configuration is refused here, not approximated.
UNMIRRORED = {"crop": None, "crop_model": None, "cascade": None, "remove_outside": None,
              "remove_outside_dilation": None, "remove_mask": None, "model": "3d_fullres",
              "folds": [0]}


def fetch_wheel(version: str) -> tuple[zipfile.ZipFile, str]:
    meta = zippeek.get_json(f"https://pypi.org/pypi/TotalSegmentator/{version}/json")
    whl = next(u for u in meta["urls"] if u["filename"].endswith("-py3-none-any.whl"))
    with zippeek.open_url(whl["url"], timeout=120) as r:
        blob = r.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != whl["digests"]["sha256"]:
        raise SystemExit(f"{whl['filename']}: sha256 {got} != PyPI's {whl['digests']['sha256']}")
    return zipfile.ZipFile(io.BytesIO(blob)), got


def _source(zf, name: str) -> str:
    return zf.read(f"totalsegmentator/{name}").decode("utf-8")


def _function(src: str, name: str) -> ast.FunctionDef:
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        raise SystemExit(f"upstream has no function {name!r}: the generator needs an update")
    return fn


def _signature_default(src: str, fn_name: str, arg: str):
    fn = _function(src, fn_name)
    args = fn.args.args
    defaults = dict(zip([a.arg for a in args[len(args) - len(fn.args.defaults):]],
                        fn.args.defaults))
    return ast.literal_eval(defaults[arg])


def _part_maps(nnunet_src: str) -> dict:
    """``{task_name: (class-map-parts name, dataset->part map name)}`` from nnunet.py's chain."""
    out = {}
    for node in ast.walk(ast.parse(nnunet_src)):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "task_name"
                and isinstance(node.test.ops[0], ast.Eq)
                and isinstance(node.test.comparators[0], ast.Constant)):
            continue
        assigned = {t.targets[0].id: t.value.id for t in node.body
                    if isinstance(t, ast.Assign) and isinstance(t.value, ast.Name)
                    and isinstance(t.targets[0], ast.Name)}
        if {"class_map_parts", "map_taskid_to_partname"} <= set(assigned):
            out[node.test.comparators[0].value] = (assigned["class_map_parts"],
                                                   assigned["map_taskid_to_partname"])
    return out


def _step_rule(nnunet_src: str) -> tuple[set, float, float]:
    """``(task names, their step, everyone else's step)`` from nnunet.py's step_size rule."""
    def step_of(body):
        for t in body:
            if (isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
                    and t.targets[0].id == "step_size"):
                return float(ast.literal_eval(t.value))
        return None
    for node in ast.walk(ast.parse(nnunet_src)):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "task_name"
                and isinstance(node.test.ops[0], ast.In)):
            yes, no = step_of(node.body), step_of(node.orelse)
            if yes is not None and no is not None:
                return set(ast.literal_eval(node.test.comparators[0])), yes, no
    raise SystemExit("nnunet.py has no `if task_name in [...]: step_size = ...` rule: "
                     "the generator needs an update")


def load_upstream(zf) -> dict:
    ns: dict = {}
    exec(_source(zf, "map_tasks_config.py"), ns)                # noqa: S102 - upstream data
    exec(_source(zf, "map_to_binary.py"), ns)                   # noqa: S102 - upstream data
    api = _source(zf, "python_api.py")
    get_cfg = _function(api, "get_task_config")
    ns["show_license_info"] = lambda: None                      # a print, for commercial tasks
    exec(textwrap.dedent(ast.get_source_segment(api, get_cfg)), ns)  # noqa: S102
    ns["_defaults"] = {"plans": _signature_default(api, "totalsegmentator", "plans"),
                       "model_size": _signature_default(api, "totalsegmentator", "model_size")}
    ns["_part_maps"] = _part_maps(_source(zf, "nnunet.py"))
    ns["_step_rule"] = _step_rule(_source(zf, "nnunet.py"))
    return ns


def _config(up: dict, task: str, mode: str) -> dict:
    return up["get_task_config"](task, fast=mode == "fast", fastest=mode == "fastest",
                                 quiet=True, plans=up["_defaults"]["plans"],
                                 model_size=up["_defaults"]["model_size"])


def entry_for(up: dict, task: str, mode: str, name: str) -> dict:
    cfg = _config(up, task, mode)
    for key, allowed in UNMIRRORED.items():
        if cfg.get(key) != allowed:
            raise SystemExit(f"{task} ({mode}): {key}={cfg.get(key)!r} - haversack does not "
                             f"mirror that for a generated task; extend the generator first")
    ids = cfg["task_id"] if isinstance(cfg["task_id"], list) else [cfg["task_id"]]
    choice = {"trainer": cfg["trainer"], "plans": cfg["plans"]}
    label_map = up["class_map"][task]
    out = {"name": name, "lineage": "ts", "modality": "MR" if task.endswith("_mr") else "CT"}
    if len(ids) == 1:
        out.update(shape="single", single=ids[0])
    else:
        parts_name, partmap_name = up["_part_maps"][task]
        parts, partmap = up[parts_name], up[partmap_name]
        inv = {v: k for k, v in label_map.items()}
        union = []
        for wid in ids:
            local = parts[partmap[wid]]
            missing = sorted(n for n in local.values() if n not in inv)
            if missing:
                # upstream's class_map_inv[...] would KeyError on these too; never drop one
                # silently - a dropped label paints 0 over whatever an earlier part wrote
                raise SystemExit(f"{task}: part {partmap[wid]} names {missing}, not in the union")
            union.append({"weights_id": wid, "name": partmap[wid].removeprefix("class_map_part_"),
                          "label_remap": {str(k): inv[v] for k, v in sorted(local.items())}})
        out.update(shape="label_union", union=union)
    out["models"] = {str(w): choice for w in ids}
    names, special, other = up["_step_rule"]
    out["step_size"] = special if task in names else other
    out["label_map"] = {str(k): v for k, v in sorted(label_map.items())}
    # classes the model emits and the task drops (upstream's remove_auxiliary_labels);
    # haversack refuses a TS model whose unnamed values are not exactly these (2026-09-22)
    aux = up["class_map"].get(f"{task}_auxiliary")
    if aux:
        out["auxiliary"] = {str(k): v for k, v in sorted(aux.items())}
    out["upstream"] = {"task": task, "mode": mode, "resample": cfg["resample"]}
    return out


def build(catalog: str) -> dict:
    spec = CATALOGS[catalog]
    zf, digest = fetch_wheel(spec["version"])
    up = load_upstream(zf)
    tasks = []
    for task, base in spec["tasks"].items():
        seen = []
        for mode in MODES:
            cfg = _config(up, task, mode)
            if any(cfg == s for s in seen):                      # a mode that falls back
                continue
            seen.append(cfg)
            tasks.append(entry_for(up, task, mode, base if mode == "default" else f"{base}_{mode}"))
    return {"_meta": {"schema_version": 1, "catalog": catalog, "ts_version": spec["version"],
                      "wheel_sha256": digest, "generator": "tools/gen_ts_registry.py",
                      "generator_version": GENERATOR_VERSION, "task_count": len(tasks)},
            "tasks": tasks}


def verify(registry: dict) -> list[str]:
    """Check each entry against its checkpoints; returns the problems (empty when clean)."""
    manifest = json.loads((DATA / "ts_weights.json").read_text(encoding="utf-8"))["weights"]
    problems, cache = [], {}

    def checkpoint(wid, choice):
        if wid not in cache:
            e = manifest.get(str(wid))
            if e is None:
                cache[wid] = None
            else:
                url = e["versions"][e["default"]]["url"]
                cache[wid] = (url, zippeek.central_directory(url))
        if cache[wid] is None:
            return None, f"Dataset{wid}: no entry in ts_weights.json"
        url, cd = cache[wid]
        folder = f"{choice['trainer']}__{choice['plans']}__3d_fullres"
        found = [c for c in zippeek.config_folders(cd) if c.rstrip("/").endswith("/" + folder)]
        if len(found) != 1:
            return None, f"Dataset{wid}: {len(found)} folders named {folder}"
        base = found[0].rstrip("/")
        return (zippeek.read_json(url, cd, f"{base}/dataset.json"),
                zippeek.read_json(url, cd, f"{base}/plans.json")), None

    for t in registry["tasks"]:
        spacing = t["upstream"]["resample"]
        spacing = [float(spacing)] * 3 if not isinstance(spacing, list) else spacing
        parts = ([(t["single"], None)] if t["shape"] == "single"
                 else [(p["weights_id"], p["label_remap"]) for p in t["union"]])
        for wid, remap in parts:
            got, err = checkpoint(wid, t["models"][str(wid)])
            if err:
                problems.append(f"{t['name']}: {err}")
                continue
            ds, plans = got
            fullres = plans["configurations"]["3d_fullres"]
            if [float(v) for v in fullres["spacing"]] != [float(v) for v in spacing[::-1]]:
                problems.append(f"{t['name']}: Dataset{wid} plans spacing {fullres['spacing']} "
                                f"!= upstream resample {spacing}")
            labels = {int(v): k for k, v in ds["labels"].items() if int(v) != 0}
            painted = ({int(k): n for k, n in {**t["label_map"], **t.get("auxiliary", {})}.items()}
                       if remap is None else
                       {int(k): t["label_map"][str(v)] for k, v in remap.items()})
            if labels != painted:
                diff = {k: (labels.get(k), painted.get(k)) for k in set(labels) | set(painted)
                        if labels.get(k) != painted.get(k)}
                problems.append(f"{t['name']}: Dataset{wid} labels differ (checkpoint, entry): {diff}")
    return problems


def _dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("generate", "check"):
        p = sub.add_parser(cmd)
        p.add_argument("catalog", choices=sorted(CATALOGS))
        p.add_argument("--no-verify", action="store_true",
                       help="skip checking entries against the checkpoints' plans and labels")
        if cmd == "generate":
            p.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    reg = build(a.catalog)
    if not a.no_verify:
        problems = verify(reg)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"  verified {reg['_meta']['task_count']} tasks against their checkpoints",
              file=sys.stderr)
    path = CATALOGS[a.catalog]["path"]
    if a.cmd == "check":
        same = path.exists() and path.read_text(encoding="utf-8") == _dumps(reg)
        print(f"  {path.name}: {'in sync' if same else 'OUT OF SYNC - regenerate with --write'}",
              file=sys.stderr)
        return 0 if same else 1
    if a.write:
        path.write_text(_dumps(reg), encoding="utf-8")
        print(f"  wrote {path}", file=sys.stderr)
    else:
        sys.stdout.write(_dumps(reg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
