"""Prove the CUDA (Triton) store-backed restore on Modal: bit-identical to the torch path.

Runs on a GPU worker: random rankfield codes (format 0.3: shell keep, log byte) through
`rankfield.restore` on CUDA and on the CPU (torch), painted and not, on grids that stay
inside and grids that leave the box; and the torso store if `HAVERSACK_RESTORE_STORE`
names one, with timings.

usage (from the haversack checkout, rankfield checked out beside it):
    uv run --no-project modal run tools/ranked_restore_modal.py
    HAVERSACK_RESTORE_STORE=path/to/torso.duckn.zip uv run --no-project modal run tools/ranked_restore_modal.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "haversack"
RANKFIELD = HERE.parent.parent / "rankfield" / "src" / "rankfield"
STORE = Path(os.environ["HAVERSACK_RESTORE_STORE"]) if os.environ.get("HAVERSACK_RESTORE_STORE") else None


def _duckn_spec() -> str:
    """The duckn git source haversack's own pyproject pins (a tag), as a pip requirement."""
    text = (HERE.parent / "pyproject.toml").read_text()
    m = re.search(r'duckn\s*=\s*\{\s*git\s*=\s*"([^"]+)"\s*,\s*tag\s*=\s*"([^"]+)"', text)
    if not m:
        raise SystemExit("pyproject.toml: no duckn git source found")
    return f"duckn @ git+{m.group(1)}@{m.group(2)}"


# Both packages are mounted from the sibling checkouts (so a change under review is what runs,
# not the released tag) and the image is built from pip, not a uv sync of haversack's project.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch>=2.7", "triton>=3.0", "numpy>=2", "zarr>=3.3", "SimpleITK", "pydantic>=2",
                 _duckn_spec() if modal.is_local() else "duckn")   # the image is built locally
    .add_local_dir(str(PKG), remote_path="/root/pkg/haversack")
    .add_local_dir(str(RANKFIELD), remote_path="/root/pkg/rankfield")
)
if STORE is not None:
    image = image.add_local_file(str(STORE), remote_path=f"/root/data/{STORE.name}")

app = modal.App("haversack-ranked-restore-check", image=image)


@app.function(gpu="A10G", timeout=1200)
def check(store_name: str | None) -> dict:
    import time

    import numpy as np
    import torch
    sys.path.insert(0, "/root/pkg")
    import rankfield as rf
    from rankfield.backends import triton_gpu
    assert triton_gpu.available(), triton_gpu.why_unavailable()
    out = {"gpu": torch.cuda.get_device_name(0), "rankfield_format": rf.FORMAT_VERSION}

    def field(K, depth, shape, labels, seed):
        rng = torch.Generator().manual_seed(seed)
        zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
        lg = []
        for _ in range(K):
            c = torch.rand(3, generator=rng) * torch.tensor(shape, dtype=torch.float32)
            d = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).sqrt()
            lg.append(6.0 - 0.9 * d + 0.3 * torch.randn(shape, generator=rng))
        code = rf.encode(torch.stack(lg), depth=depth)
        geo = rf.Geometry(spacing_zyx=(2.0, 2.0, 2.0), shape_zyx=tuple(shape))
        return rf.RankField(ranks=code.ranks, support=code.support, tail=code.tail, meta=code.meta,
                            labels=labels, geometry=geo)

    # ---- synthetic: random codes, several grids, one part and two painted parts ----
    worst = 0
    shape = (11, 13, 17)
    for seed in range(6):
        K, depth = (10, 6) if seed % 2 == 0 else (5, 5)
        luts = [list(range(K)), [5] + list(range(1, K - 1)) + [0]]      # plain, and a LUT with a
        part = rf.Part(field(K, depth, shape, luts[seed % 2], seed))    # zero-mapped class
        second = rf.Part(field(K, depth, shape, [0] + [20 + i for i in range(1, K)], seed + 100))
        like = rf.array_grid(part)
        for spacing, paint, origin in ((0.7, False, (0.0, 0.0, 0.0)), (1.3, True, (0.0, 0.0, 0.0)),
                                       (2.0, False, (0.0, 0.0, 0.0)), (1.0, True, (-9.0, -3.0, 4.0))):
            grid = rf.Grid.isotropic(spacing, like=like)
            if any(origin):                                       # a grid past the box's edges
                grid = rf.Grid(shape=tuple(n + 12 for n in grid.shape), spacing=grid.spacing, origin=origin)
            parts = [part, second] if paint else [part]
            res = {}
            for dev in ("cpu", "cuda"):
                res[dev] = rf.restore(parts, grid=grid, interp="linear", device=torch.device(dev),
                                      slab_voxels=(1 << 18) if dev == "cuda" else 700).labels
                res[dev] = np.asarray(res[dev])
            diff = int((res["cpu"] != res["cuda"]).sum())
            worst = max(worst, diff)
            if diff and "synthetic_example" not in out:
                ref, margin = rf.reference_restore(part, grid)
                where = np.argwhere(res["cpu"] != res["cuda"])
                ex = []
                for w in where[:4]:
                    z, y, x = (int(v) for v in w)
                    ex.append({"at": [z, y, x], "cpu": int(res["cpu"][z, y, x]),
                               "cuda": int(res["cuda"][z, y, x]), "ref64": int(ref[z, y, x]),
                               "ref_margin": float(margin[z, y, x])})
                out["synthetic_example"] = {"seed": seed, "spacing": spacing, "paint": paint, "cases": ex}
    out["synthetic_max_differing_voxels"] = worst

    # ---- the real store, if given ----
    if store_name:
        from haversack import ranked_restore as rr
        p = Path("/root/data") / store_name
        rr.restore(p, grid=1.5, device="cuda", roi=((0, 8), (0, 40), (0, 40)))     # JIT + warm
        torch.cuda.synchronize()
        t0 = time.time()
        g_ = rr.restore(p, grid=1.5, device="cuda")
        torch.cuda.synchronize()
        tg = time.time() - t0
        t0 = time.time()
        c_ = rr.restore(p, grid=1.5, device="cpu")
        tc = time.time() - t0
        nd = int((g_.labels != c_.labels).sum())
        out["torso_1p5mm"] = {"voxels": int(g_.labels.size), "cuda_s": round(tg, 3), "cpu_s": round(tc, 1),
                              "differing_voxels": nd}
        t0 = time.time()
        g1 = rr.restore(p, grid=1.0, device="cuda")
        torch.cuda.synchronize()
        out["torso_1p0mm"] = {"voxels": int(g1.labels.size), "cuda_s": round(time.time() - t0, 3)}
    return out


@app.local_entrypoint()
def main():
    r = check.remote(STORE.name if STORE is not None else None)
    for k, v in r.items():
        print(f"{k}: {v}")
