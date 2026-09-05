"""Upgrade a ranked store's PARTS to ranked format 0.2, in place.

What 0.2 changed (2026-09-05), and what this does to a store written before it:

  * a rank whose support byte is 0 (a gap that rounds to the clip) decodes as absent, so
    the encoder now writes the sentinel there - the rank bytes are masked to 0;
  * the tail plane is uint16 (quantum 1/65535 of the mass) - an old uint8 tail is widened
    exactly (x257); its definition also changed to the mass of everything NOT stored,
    which an old store cannot recompute without the logits, so the value is carried as it
    was and `max_tail` is left alone;
  * the block states `version` "0.2", `gap_unit` "logit" and `tail_max`.

Every reader (verify, sdfview, the preview) refuses a part without these. Works on a
directory or a zarr zip (staged and repacked). Idempotent.

usage: uv run python tools/ranked_upgrade_format.py STORE.duckn [STORE.duckn ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from haversack.ranked import GAP_UNIT, RANKED_VERSION, TAIL_MAX
from haversack.ranked_store import open_store

STEP = "Ranked format upgraded to 0.2"


def _upgrade_part(g, index: int, say) -> dict:
    attrs = g.attrs.asdict()
    block = dict(attrs["duckn"]["extensions"]["ranked"])
    if block.get("mode") not in (None, "ranked"):
        raise SystemExit(f"parts/{index}: mode {block.get('mode')!r} is not handled by this tool")
    masked = 0
    if "support" in g and "ranks" in g:
        ranks, sup = g["ranks"], g["support"]
        for j in range(1, ranks.shape[0]):
            rj = np.asarray(ranks[j])
            sj = np.asarray(sup[j - 1])
            bad = (rj != 0) & (sj == 0)
            n = int(bad.sum())
            if n:
                rj[bad] = 0
                ranks[j] = rj                    # a chunk that became all-zero is dropped
                masked += n
    widened = False
    if "tail" in g and str(g["tail"].dtype) == "uint8":
        old = g["tail"]
        data = np.asarray(old[:]).astype(np.uint16) * np.uint16(257)   # 255 -> 65535 exactly
        meta = old.metadata
        arr_attrs = old.attrs.asdict()
        del g["tail"]
        z = g.create_array("tail", shape=data.shape, dtype=np.uint16,
                           chunks=meta.chunk_grid.chunk_shape if hasattr(meta.chunk_grid, "chunk_shape") else None,
                           shards=getattr(meta, "shards", None) or None,
                           compressors=None if False else _compressors(meta),
                           attributes=arr_attrs)
        z[:] = data
        widened = True
    block.update({"version": RANKED_VERSION, "gap_unit": GAP_UNIT,
                  "tail_max": (TAIL_MAX if "tail" in g else None)})
    attrs["duckn"]["extensions"]["ranked"] = block
    g.attrs.update(attrs)
    say(f"  parts/{index}: masked {masked} rank bytes, tail {'widened to uint16' if widened else 'unchanged'}")
    return block


def _compressors(meta):
    """The zstd compressor of the old array's inner codecs, whatever the sharding."""
    import zarr
    codecs = list(meta.codecs)
    inner = codecs
    for c in codecs:
        if c.__class__.__name__ == "ShardingCodec":
            inner = list(c.codecs)
    for c in inner:
        if c.__class__.__name__ == "ZstdCodec":
            return c
    return zarr.codecs.ZstdCodec(level=9)


def upgrade(store: Path, quiet: bool = False) -> None:
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, file=sys.stderr, flush=True))
    with open_store(store, "a") as st:
        root = st.root
        i = 0
        while f"parts/{i}" in root:
            _upgrade_part(root[f"parts/{i}"], i, say)
            i += 1
        if i == 0:
            raise SystemExit(f"{store}: no parts")
        attrs = root.attrs.asdict()
        ext = attrs.get("duckn", {}).get("extensions", {})
        pv = dict(ext.get("provenance") or {"version": "1.0"})
        steps = [s for s in pv.get("processing", []) if s.get("name") != STEP]
        steps.append({"name": STEP,
                      "description": "ranks masked where the support byte is 0; tail widened to "
                                     "uint16; version, gap_unit and tail_max stated; in place",
                      "software": {"name": "ranked_upgrade_format.py",
                                   "url": "https://github.com/mhalle/haversack"},
                      "parameters": {"to_version": RANKED_VERSION}})
        pv["processing"] = steps
        ext["provenance"] = pv
        attrs["duckn"]["extensions"] = ext
        root.attrs.update(attrs)
    say(f"{store.name}: {i} part(s) at ranked {RANKED_VERSION}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("store", nargs="+", help="a ranked store: STORE.duckn or STORE.duckn.zip")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    for s in a.store:
        upgrade(Path(s), quiet=a.quiet)


if __name__ == "__main__":
    main()
