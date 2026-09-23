"""Upgrade an existing ranked store's segment metadata to the current duckn seg extension, in place.

The builder writes the current version for new stores. This is for stores that already exist - the demo
packages - where re-emitting to rebuild is not on. It reads the root's `seg` extension
through duckn's reader, which migrates 0.6 and 0.7 (a group becomes a segment listing its
members' values, `background: true` becomes the background role, colors become CSS
strings), then states what the builder now states:

  * a background segment per part (class 0 of the softmax, `role: "background"`);
  * NO generated groups. Until 0.8 the builder wrote a partition group per part
    (`classes_<i>`) and the engine's named unions (`g_lungs`, ...). Those are facts about a
    labeling scheme, not about a store, and seg 0.8 keeps them outside it; the upgrade
    removes the ones the builder generated and keeps any a user authored, as the union
    segments the migration made of them.

The result is checked against duckn's consistency rules before it is written, a processing
step is appended to the store's provenance, and the README is refreshed. Works on a
directory or a zarr zip (staged and repacked).

usage: uv run python tools/ranked_upgrade_seg.py STORE.duckn [STORE.duckn ...]
"""
import copy
from pathlib import Path

from duckn import read_seg_extension

from haversack.ranked_build import GENERATED_GROUP_IDS, write_readme
from haversack.ranked_store import open_store, root_attrs, segment, segmentation

STEP = "Segment metadata upgraded to the current seg extension"


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

    # duckn migrates older shapes on read, and reports what it changed
    was = str(ext["seg"].get("version"))
    seg, reported = read_seg_extension(ext["seg"])
    for d in reported:
        print(f"    {d}", flush=True)
    # What the builder generated goes: a generated id belongs to the builder whether or not
    # this engine still generates it (`GENERATED_GROUP_IDS`), and so does a part's partition.
    segments = [s for s in seg.segments
                if s.id not in GENERATED_GROUP_IDS and not s.id.startswith("classes_")]
    multi = len(order) > 1
    for i, _o in enumerate(order):
        if not any(s.role == "background" and (s.layer or 0) == i for s in segments):
            segments.append(segment(f"background_{i}", "background", 0,
                                    layer=i if multi else None, role="background"))
    if not multi:
        # `layer` states which part a segment belongs to; a single-part store has nothing to say
        segments = [s.model_copy(update={"layer": None}) if s.layer is not None else s
                    for s in segments]
    new_seg = segmentation(                              # duckn's rules are checked here
        segments, terminologies={k: v.model_dump(exclude_none=True)
                                 for k, v in (seg.terminologies or {}).items()},
        labeling_scheme=seg.labeling_scheme)             # one key, or a cascade's several

    pv = dict(ext.get("provenance") or {"version": "1.0"})
    steps = [s for s in pv.get("processing", []) if s.get("name") != STEP]
    steps.append({
        "name": STEP,
        "description": "root `seg` extension rewritten as the current duckn seg extension: segments listing "
                       "their label values, a background segment per part, the builder's "
                       "generated groups removed; in place",
        "software": {"name": "ranked_upgrade_seg.py",
                     "url": "https://github.com/mhalle/haversack"},
        "parameters": {"from_version": was, "to_version": new_seg.version,
                       "segments": len(new_seg.segments)}})
    pv["processing"] = steps
    others = {k: v for k, v in ext.items() if k not in ("seg", "provenance")}
    root.attrs.update(root_attrs(new_seg, provenance=pv, **others))
    write_readme(st)
    print(f"  {store.name}: seg {was} -> {new_seg.version}, "
          f"{len(new_seg.segments)} segments", flush=True)


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
