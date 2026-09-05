"""Check a ranked store against the invariants its README states.

Every failure this session was a SILENT one. A sixth positional argument that never arrived, so
`envelope_mm` kept its default and the run was subtly not what was asked for. A label-name
lookup that failed and renamed 78 segments to `label_<id>` while the build reported success. An
occupancy index copied instead of rebuilt, so it under-reported the one class the padding had
changed. None of these raised; all of them produced a store that looked fine.

So the pipeline needs a step that can say no. These are the README's claims turned into
assertions - if a check here fails, either the store is wrong or the documentation is, and
either way somebody has to look.

Cheap checks always run. `--deep` adds the ones that touch every voxel: occupancy
conservatism, segment extents, and the decode identity.

usage: uv run python tools/ranked_verify.py STORE.duckn [--deep] [--quiet]
       uv run python tools/ranked_verify.py DIR --all [--deep]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from duckn import validate_seg_data

from haversack.ranked_store import open_store, read_segmentation, validate_array

REQUIRED_RANKED = ("version", "mode", "classes", "depth", "clip", "gap_unit", "support_max",
                   "rank_sentinel", "labels", "part", "task", "model_grid", "envelope")
RANKED_VERSIONS = ("0.2", "0.3")   # what this verifier (and every reader in the ecosystem) knows
CURVE_KEYS_03 = ("gap_curve", "gap_range", "gap_origin", "keep")
README_MUST_MENTION = {"0.3": "format 0.3"}     # formats whose bytes an older README misreads


class Report:
    def __init__(self, name, quiet=False):
        self.name, self.quiet, self.fail, self.warn, self.ok = name, quiet, [], [], 0

    def check(self, cond, msg, *, warn_only=False):
        if cond:
            self.ok += 1
        elif warn_only:
            self.warn.append(msg)
        else:
            self.fail.append(msg)
        return bool(cond)

    def emit(self):
        status = "FAIL" if self.fail else ("warn" if self.warn else "ok")
        if not (self.quiet and status == "ok"):
            print(f"  {self.name:<44} {self.ok:>3} checks  {status}")
        for m in self.fail:
            print(f"      FAIL  {m}")
        for m in self.warn:
            print(f"      warn  {m}")
        return not self.fail


def verify(path: Path, deep: bool = False, quiet: bool = False) -> bool:
    st = open_store(path, "r")           # a directory, or a zarr zip
    r = st.root
    root = r.attrs.asdict().get("duckn", {})
    ext = root.get("extensions", {})
    rep = Report(f"{path.parent.name}/{path.name}", quiet)

    rep.check(st.exists("README.md"), "no README.md - the format reference is missing")
    readme = st.read_text("README.md").lower() if st.exists("README.md") else ""
    # The standard's own reading of the metadata, before any of ours: duckn's models parse
    # the root and every array, and its validators check the seg extension's consistency
    # rules and each array's geometry against its shape.
    seg_model = None
    try:
        seg_model = read_segmentation(r)
    except Exception as e:                     # noqa: BLE001 - the rejection is the finding
        rep.check(False, f"duckn rejects the seg extension: {e}")
    rep.check("seg" in ext, "no seg extension on the root group")
    rep.check("haversack" in ext, "no haversack block on the root group")
    pv = ext.get("provenance") or {}
    rep.check(bool(pv), "no duckn provenance extension - nothing says where this came from")
    rep.check(bool(pv.get("sources")), "provenance has no sources - the case is unidentifiable",
              warn_only=True)
    steps = pv.get("processing") or []
    rep.check(bool(steps), "provenance has no processing steps - nothing says what made this")
    rep.check(all(s.get("software", {}).get("name") for s in steps),
              "a processing step names no software")
    rep.check(all(s.get("name") for s in steps),
              "a processing step has no name (required by the extension)")
    order = ext.get("haversack", {}).get("part_order", [])
    rep.check(bool(order), "empty part_order - paint order is external knowledge and must "
                           "be recorded")

    segs = ext.get("seg", {}).get("segments", [])
    leaves = [s for s in segs if "label_value" in s]          # groups carry `members`
    # a cascade's earlier stages have their own classes, which the task's label map does not
    # name; numbered leaves there are honest, not a degraded lookup
    from haversack.ranked_build import CASCADE_PART
    cascade = bool(order) and len(order) > 1 and all(CASCADE_PART.search(str(o.get("name", "")))
                                                      for o in order)
    named = [s for s in leaves if not (cascade and (s.get("layer") or 0) != len(order) - 1)]
    rep.check(all("name" in s and not str(s["name"]).startswith("label_") for s in named),
              f"{sum(1 for s in named if str(s.get('name','')).startswith('label_'))} of "
              f"{len(named)} segments are unnamed (label_<id>) - the name lookup degraded")
    rep.check(all(0 <= s.get("layer", 0) < max(len(order), 1) for s in leaves),
              "a segment's layer is not a valid part index")
    rep.check(all(any(s.get("background") and s.get("layer", 0) == i for s in leaves)
                  for i in range(len(order))),
              "a part has no background leaf - its partition cannot include class 0")
    rep.check(all(any(s.get("members") and s.get("exhaustive") and s.get("disjoint")
                      and s["id"] == f"classes_{i}" for s in segs)
                  for i in range(len(order))),
              "a part has no partition group (classes_<i>) - the softmax's own partition "
              "goes unstated")

    origins, directions = set(), set()
    for i, _p in enumerate(order):
        if f"parts/{i}" not in r:
            rep.check(False, f"parts/{i} declared in part_order but absent")
            continue
        g = r[f"parts/{i}"]
        m = g.attrs.asdict().get("duckn", {}).get("extensions", {}).get("ranked", {})
        miss = [k for k in REQUIRED_RANKED if k not in m]
        rep.check(not miss, f"parts/{i}: ranked block missing {miss}")
        if miss:
            continue

        soft = m.get("softmax") or {}
        rep.check(bool(soft), f"parts/{i}: no softmax block - a reader cannot tell which "
                              "normalization these classes competed in")
        if soft:
            rep.check(soft.get("classes") == m["classes"],
                      f"parts/{i}: softmax.classes {soft.get('classes')} != classes "
                      f"{m['classes']}")
            rep.check(bool(soft.get("weights")), f"parts/{i}: softmax names no weights")

        # The README is the format reference a cold reader follows. A 0.3 part under the 0.2
        # README is decoded 8x wrong at every gap; a 0.2 part under a later README is fine
        # (the reference documents every format it knows), so only the versions whose
        # bytes changed are checked.
        if str(m["version"]) in README_MUST_MENTION:
            rep.check(README_MUST_MENTION[str(m["version"])] in readme,
                      f"parts/{i}: the store's README does not document ranked format {m['version']} - "
                      f"a reader following it would decode these bytes wrongly")
        rep.check(str(m["version"]) in RANKED_VERSIONS,
                  f"parts/{i}: ranked block version {m['version']!r} is not one this reader "
                  f"knows {RANKED_VERSIONS} - upgrade with tools/ranked_upgrade_format.py")
        if str(m["version"]) == "0.3":
            miss3 = [k for k in CURVE_KEYS_03 if k not in m]
            rep.check(not miss3, f"parts/{i}: ranked 0.3 block missing {miss3}")
            # Values, not just presence: a block that names a curve no reader knows, or a
            # range or origin that describes none, passes a key check and then fails inside
            # every decoder. Ask the reference reader itself - what it refuses, this refuses.
            rep.check(m.get("keep") in ("shell", "clip"),
                      f"parts/{i}: keep {m.get('keep')!r} is not a keep rule (shell, clip)")
            try:
                import rankfield
                rankfield.levels(m)
            except Exception as e:                 # noqa: BLE001 - the refusal is the finding
                rep.check(False, f"parts/{i}: the level table cannot be built from this block: {e}")
        rep.check(m["gap_unit"] == "logit",
                  f"parts/{i}: gap_unit {m['gap_unit']!r} - margins in another unit are not "
                  "comparable to logits and no reader here handles them yet")
        K, smax, sent = m["classes"], m["support_max"], m["rank_sentinel"]
        rep.check(len(m["labels"]) == K,
                  f"parts/{i}: labels has {len(m['labels'])} entries for {K} classes")
        rep.check(sent == 0, f"parts/{i}: rank_sentinel is {sent}, not 0 - the whole "
                             "'zero means absent' contract assumes 0")

        env, grid = m["envelope"], m["model_grid"]
        rep.check(len(env) == 6, f"parts/{i}: envelope must be 6 inclusive bounds, got {len(env)}")
        if len(env) == 6:
            lo, hi = env[0::2], env[1::2]
            rep.check(all(a <= b for a, b in zip(lo, hi)),
                      f"parts/{i}: envelope has an inverted bound {env} - inclusive, not half-open?")
            rep.check(all(0 <= a and b < n for a, b, n in zip(lo, hi, grid)),
                      f"parts/{i}: envelope {env} escapes model_grid {grid}")

        ranks = g["ranks"]
        rep.check(ranks.ndim == 4, f"parts/{i}: ranks must be 4-D, got {ranks.ndim}")
        shape = tuple(ranks.shape[1:])
        if len(env) == 6:
            want = tuple(b - a + 1 for a, b in zip(env[0::2], env[1::2]))
            rep.check(shape == want,
                      f"parts/{i}: array is {shape} but envelope says {want}")
        rep.check("support" in g, f"parts/{i}: no support array")
        if "support" in g:
            rep.check(g["support"].shape[0] == ranks.shape[0] - 1,
                      f"parts/{i}: support has {g['support'].shape[0]} planes for "
                      f"{ranks.shape[0]} rank planes (want one fewer)")
            rep.check(tuple(g["support"].shape[1:]) == shape,
                      f"parts/{i}: support grid {tuple(g['support'].shape[1:])} != ranks {shape}")
        rep.check(m.get("exhaustive") or "tail" in g,
                  f"parts/{i}: not exhaustive but no tail array", warn_only=True)
        if "tail" in g:
            want_tail = "uint16" if int(m.get("tail_max") or 0) > 255 else "uint8"
            rep.check(str(g["tail"].dtype) == want_tail,
                      f"parts/{i}: tail is {g['tail'].dtype} but tail_max {m.get('tail_max')} "
                      f"says {want_tail}")

        rep.check("frame" in m or "target_grid" in m,
                  f"parts/{i}: no frame (or target_grid) - the store restores only onto grids "
                  "given in its own model-grid frame, never onto the input's", warn_only=True)
        if "distance" in g:
            # The quantum is truncation/max, so without both the array is a uint8 with no
            # scale - the same roles `clip`/`support_max` play for `support`.
            rep.check("distance_truncation" in m,
                      f"parts/{i}: distance array but no distance_truncation - "
                      "the field cannot be decoded")
            rep.check("distance_max" in m,
                      f"parts/{i}: distance array but no distance_max - "
                      "the field cannot be decoded")
            rep.check(tuple(g["distance"].shape) == shape,
                      f"parts/{i}: distance grid {tuple(g['distance'].shape)} "
                      f"!= ranks {shape} - it must be ONE 3-D field, like tail")
            t = m.get("distance_truncation")
            rep.check(t is None or t > 0, f"parts/{i}: distance_truncation {t} is not positive")

        if "junction" in g or "junction_pair" in g:
            # The pair is meaningless without the byte and vice versa, and the byte is a uint8
            # with no scale without its truncation and half-range.
            rep.check("junction" in g and "junction_pair" in g,
                      f"parts/{i}: junction and junction_pair must travel together")
            rep.check("junction_truncation" in m and "junction_zero" in m and "junction_span" in m,
                      f"parts/{i}: junction array but no junction_truncation/junction_zero/"
                      "junction_span - the field cannot be decoded")
            if "junction" in g:
                rep.check(tuple(g["junction"].shape) == shape,
                          f"parts/{i}: junction grid {tuple(g['junction'].shape)} != ranks {shape}")
            if "junction_pair" in g:
                rep.check(tuple(g["junction_pair"].shape) == (2,) + shape,
                          f"parts/{i}: junction_pair must be (2, Z, Y, X), got "
                          f"{tuple(g['junction_pair'].shape)}")
                rep.check(g["junction_pair"].dtype == ranks.dtype,
                          f"parts/{i}: junction_pair dtype {g['junction_pair'].dtype} != "
                          f"ranks {ranks.dtype} - it holds class + 1 like ranks")
            if "junction" in g and "junction_pair" in g:
                jn = np.asarray(g["junction"][:])
                jp = np.asarray(g["junction_pair"][:])
                present = jn > 0
                rep.check(bool((jp[0][present] > 0).all() and (jp[1][present] > 0).all()),
                          f"parts/{i}: junction byte present with an absent pair")
                rep.check(bool((jp[0][~present] == 0).all() and (jp[1][~present] == 0).all()),
                          f"parts/{i}: junction pair present where the byte is the sentinel")
                rep.check(bool((jp[0][present] < jp[1][present]).all()),
                          f"parts/{i}: junction_pair is not in canonical order (lower class first)")
                rep.check(bool((jp[0][present] != 1).all()),
                          f"parts/{i}: junction_pair names the background as a structure")
                frac = present.mean()
                rep.check(frac < 0.25,
                          f"parts/{i}: junction written on {100 * frac:.1f} % of voxels - "
                          "the layer is meant to be tubes around triple lines, not a band",
                          warn_only=True)

        # Centering is the sample-count-to-extent relationship, so it decides what a resampler
        # holds fixed. It was hardcoded to "cell" on grids the corner rule produced, which is
        # duckn's "node" - harmless while duckn's resample() ignored the field, a half-voxel
        # shift now that it honors it. Restated here rather than imported: a verifier that
        # imports the builder inherits the builder's mistakes.
        want_data = {"corner": "node", "center": "cell"}.get(m.get("resample_alignment"))
        for nm in ("ranks", "support", "tail", "occupancy", "distance", "junction",
                   "junction_pair"):
            if nm not in g:
                continue
            err = _rejects(lambda: validate_array(g[nm]))
            rep.check(err is None, f"parts/{i}/{nm}: duckn rejects the geometry: {err}")
            if nm in ("distance", "junction", "junction_pair"):
                continue
            ax = g[nm].attrs.asdict().get("duckn", {}).get("axes", [])
            cen = {a.get("centering") for a in ax if a.get("space_direction")}
            rep.check(len(cen) <= 1,
                      f"parts/{i}/{nm}: spatial axes disagree on centering {cen}")
            # occupancy is a brick summary whose samples do own a cell, whatever grid the
            # data arrays landed on.
            expect = "cell" if nm == "occupancy" else want_data
            rep.check(expect is None or cen == {expect},
                      f"parts/{i}/{nm}: centering {cen or 'unset'} but the grid is "
                      f"{m.get('resample_alignment')!r}-aligned, which is {expect}")
            # a single-chunk array is already one file; sharding it would add an index for
            # nothing. Only multi-chunk arrays need it.
            nchunks = int(np.prod([int(np.ceil(s / c))
                                   for s, c in zip(g[nm].shape, g[nm].chunks)]))
            rep.check(g[nm].shards is not None or nchunks == 1,
                      f"parts/{i}/{nm}: {nchunks} loose chunks - ~2.5x the disk and one "
                      "request per chunk", warn_only=True)

        at = g["ranks"].attrs.asdict().get("duckn", {})
        ax = at.get("axes", [])
        rep.check(len(ax) == 4 and ax[0].get("kind") == "list",
                  f"parts/{i}: ranks axes must be [list, space, space, space], got "
                  f"{[a.get('kind') for a in ax]}")
        rep.check("space_origin" in at, f"parts/{i}: ranks has no space_origin")
        origins.add(tuple(round(float(v), 4) for v in at.get("space_origin", [])))
        directions.add(tuple(round(float(x), 6) for a in ax if a.get("kind") == "space"
                             for x in a.get("space_direction", [])))

        if "occupancy" in g:
            rep.check("brick" in m, f"parts/{i}: occupancy present but no declared brick - "
                                    "the index would be tied to the storage layout")
            if "brick" in m:
                b = m["brick"]
                want = (K,) + tuple(int(np.ceil(s / bb)) for s, bb in zip(shape, b))
                rep.check(tuple(g["occupancy"].shape) == want,
                          f"parts/{i}: occupancy is {tuple(g['occupancy'].shape)}, want {want}")

        if deep:
            rk0 = np.asarray(ranks[0])
            if "support" in g:
                # a present rank whose support is at the clip decodes as absent: since
                # format 0.2 the encoder writes the sentinel there, so a nonzero rank byte
                # over a zero support byte is a byte that means nothing
                sup_arr = g["support"]
                prev = None
                for j in range(1, ranks.shape[0]):
                    rj, sj = np.asarray(ranks[j]), np.asarray(sup_arr[j - 1])
                    n = int(((rj != 0) & (sj == 0)).sum())
                    rep.check(n == 0, f"parts/{i}: rank plane {j} has {n} present entries whose "
                                      "support is 0 (at the clip) - masked since format 0.2")
                    if prev is not None:
                        # sentinels form a suffix: once a plane is absent every deeper one is.
                        # The encoder guarantees it (gaps grow with depth) and the restore
                        # kernels stop scanning at the first sentinel because of it.
                        bad = int(((prev == 0) & (rj != 0)).sum())
                        rep.check(bad == 0, f"parts/{i}: rank plane {j} is present at {bad} voxels "
                                            f"where plane {j - 1} is the sentinel")
                    prev = rj
                    del sj
            rep.check(not (rk0 == 0).any(),
                      f"parts/{i}: ranks[0] holds the sentinel at "
                      f"{int((rk0 == 0).sum())} voxels - every voxel must have a winner")
            lut = np.asarray(m["labels"])
            glob = lut[rk0.astype(np.int64) - 1]
            # seg spec 0.7 rule 9, duckn's own check: every value present in this layer's
            # labels, other than its background, has a leaf
            if seg_model is not None:
                err = _rejects(lambda: validate_seg_data(seg_model, glob, layer=i))
                rep.check(err is None, f"parts/{i}: {err}")
            if "occupancy" in g and "brick" in m:
                occ, b = np.asarray(g["occupancy"]), m["brick"][0]
                missed = 0
                for c in range(K):
                    if lut[c] == 0:
                        continue
                    hit = glob == lut[c]
                    if not hit.any():
                        continue
                    z, y, x = np.nonzero(hit)
                    truth = np.zeros(occ.shape[1:], bool)
                    truth[z // b, y // b, x // b] = True
                    missed += int((truth & (occ[c] != smax)).sum())
                rep.check(missed == 0,
                          f"parts/{i}: occupancy missed {missed} bricks that contain the class "
                          "- the index is NOT conservative and skipping it loses data")
            for s in leaves:
                if s.get("layer", 0) != i or "extent" not in s:
                    continue
                hit = glob == s["label_value"]
                if not hit.any():
                    rep.check(False, f"parts/{i}: segment {s['name']} has an extent but no voxels")
                    continue
                idx = np.nonzero(hit)
                want = [int(v) for d in range(3) for v in (idx[d].min(), idx[d].max())]
                rep.check(list(s["extent"]) == want,
                          f"parts/{i}: {s['name']} extent {s['extent']} != actual {want}")

    softmaxes = [(r[f"parts/{i}"].attrs.asdict().get("duckn", {}).get("extensions", {})
                  .get("ranked", {}).get("softmax") or {}).get("weights")
                 for i, _ in enumerate(order) if f"parts/{i}" in r]
    if len(order) > 1 and all(softmaxes):
        rep.check(len(set(softmaxes)) == len(softmaxes)
                  or len(set(softmaxes)) == 1,
                  f"parts share weights inconsistently ({softmaxes}) - either every part is a "
                  "distinct model or they are all one; a mix means the part split is wrong",
                  warn_only=True)

    if len(order) > 1:
        rep.check(len(origins) == 1,
                  f"parts do not share one origin ({len(origins)} distinct) - a reader must "
                  "then carry per-part offsets", warn_only=True)
        rep.check(len(directions) == 1,
                  f"parts do not share one orientation ({len(directions)} distinct)")
    st.close()
    return rep.emit()


def _rejects(fn):
    """The error a validator raises, or None when it accepts."""
    try:
        fn()
    except Exception as e:                     # noqa: BLE001 - any rejection is the finding
        return e
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--all", action="store_true", help="verify every *.duckn under path")
    ap.add_argument("--deep", action="store_true", help="add the whole-volume checks")
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    a = ap.parse_args()

    stores = sorted(a.path.glob("*/*.duckn")) if a.all else [a.path]
    if not stores:
        sys.exit(f"no stores found under {a.path}")
    print(f"verifying {len(stores)} store(s){' (deep)' if a.deep else ''}")
    bad = [s for s in stores if not verify(s, a.deep, a.quiet)]
    print(f"\n{len(stores) - len(bad)}/{len(stores)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
