"""Upgrade an existing ranked store's segment metadata to duckn seg extension 0.7, in place.

The builder writes 0.7 for new stores. This is for stores that already exist - the demo
packages - where re-emitting to rebuild is not on. It reads the root's `seg` extension back
through duckn's own model, which migrates the 0.6 shape (group membership carried as an
array-valued `label_value`) to `members`, then states what the builder now states and the
old stores never did:

  * a background LEAF per part (class 0 of the softmax, flagged `background`), so every
    value in `ranks[0] - 1` resolves to a segment and the part's partition can include it;
  * one partition group per part (`classes_<i>`: every class, background included,
    `disjoint` and `exhaustive` of the model's domain);
  * the engine's named unions re-derived with their claims (GROUP_CLAIMS in the builder), so
    an upgraded store carries exactly what a fresh build would.

The result goes through duckn's consistency validator before it is written, a processing
step is appended to the store's provenance, and the README is refreshed. Works on a
directory or a zarr zip (staged and repacked).

usage: uv run python tools/ranked_upgrade_seg.py STORE.duckn [STORE.duckn ...]
"""
import copy
from pathlib import Path

from duckn import SegmentationExtension

from haversack.ranked_build import GROUP_CLAIMS, named_groups, part_partition, write_readme
from haversack.ranked_store import leaf, open_store, root_attrs, segmentation

STEP = "Segment metadata upgraded to seg 0.7"


def upgrade(store: Path) -> None:
    with open_store(store, "a") as st:
        _upgrade(st, store)


def _upgrade(st, store: Path) -> None:
    root = st.root
    attrs = root.attrs.asdict()
    ext = copy.deepcopy((attrs.get("duckn") or {}).get("extensions") or {})
    if "haversack" not in ext and "nnseg" not in ext:
        raise SystemExit(f"{store}: no `haversack` (or legacy `nnseg`) extension block at the "
                         "root - not a haversack ranked store")
    if "seg" not in ext:
        raise SystemExit(f"{store}: no `seg` extension block at the root; nothing to upgrade")
    # Stores built before the rename (2026-09-02) carry the block under its old name; the
    # verifier and every reader look for `haversack`, so the rename travels with the upgrade.
    if "haversack" not in ext and "nnseg" in ext:
        old = ext.pop("nnseg")
        if "nnseg_version" in old:
            old["haversack_version"] = old.pop("nnseg_version")
        ext["haversack"] = old
    hv = ext["haversack"]
    engine, order = hv.get("engine", "nnunetv2"), hv.get("part_order")
    if not order:
        raise SystemExit(f"{store}: the haversack block has no `part_order`")

    # duckn migrates the 0.6 shape on read; from here on everything is 0.7 objects
    was = str(ext["seg"].get("version"))
    seg = SegmentationExtension.model_validate(ext["seg"])
    leaves = [s for s in seg.segments if s.label_value is not None]
    multi = len(order) > 1
    for i, _o in enumerate(order):
        if not any(s.background and (s.layer or 0) == i for s in leaves):
            leaves.append(leaf(f"background_{i}", "background", 0, layer=i if multi else None,
                               background=True))
    if not multi:
        # `layer` states which part a leaf belongs to; a single-part store has nothing to say
        leaves = [s.model_copy(update={"layer": None}) if s.layer is not None else s
                  for s in leaves]
    # the named unions are re-derived (their claims are the builder's decision); any other
    # group the store had - none today - is kept as it was
    known = {gid for gid, *_ in GROUP_CLAIMS.get(engine, GROUP_CLAIMS["nnunetv2"])}
    kept = [s for s in seg.segments if s.members is not None
            and s.id not in known and not s.id.startswith("classes_")]
    groups = [part_partition(i, o["name"], leaves) for i, o in enumerate(order)]
    groups += named_groups(engine, leaves) + kept
    new_seg = segmentation(leaves + groups)             # duckn's validator runs here

    pv = dict(ext.get("provenance") or {"version": "1.0"})
    steps = [s for s in pv.get("processing", []) if s.get("name") != STEP]
    steps.append({
        "name": STEP,
        "description": "root `seg` extension rewritten as duckn seg 0.7: leaves and groups "
                       "(`members`), a background leaf and a partition group per part, the "
                       "named unions with their disjoint/exhaustive claims; in place",
        "software": {"name": "ranked_upgrade_seg.py",
                     "url": "https://github.com/mhalle/haversack"},
        "parameters": {"from_version": was, "to_version": new_seg.version,
                       "segments": len(new_seg.segments)}})
    pv["processing"] = steps
    others = {k: v for k, v in ext.items() if k not in ("seg", "provenance")}
    root.attrs.update(root_attrs(new_seg, provenance=pv, **others))
    write_readme(st)
    n_leaf = sum(1 for s in new_seg.segments if s.label_value is not None)
    print(f"  {store.name}: seg {was} -> {new_seg.version}, {n_leaf} leaves, "
          f"{len(new_seg.segments) - n_leaf} groups", flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="works on a directory or a zarr zip (staged and repacked)")
    ap.add_argument("store", nargs="+", help="a ranked store: STORE.duckn or STORE.duckn.zip")
    a = ap.parse_args(argv)
    for s in a.store:
        print(s, flush=True)
        upgrade(Path(s))


if __name__ == "__main__":
    main()
