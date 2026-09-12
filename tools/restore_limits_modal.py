"""Prove the label restore's offset limits on CUDA (Modal, A10G): what each backend can take.

Written 2026-09-11 after a K=30 model on a whole-body CT with no envelope (30 x 678 x 334 x 334
= 2.27e9 logits) failed `segment` in the Triton kernel's 32-bit channel offset. Three checks,
one container, so the timings compare:

- ``reported``: that field, fp16 and fp32. "auto" must take Triton; its labels against the
  torch backend's, every disagreement measured against a float64 reference at that voxel.
- ``channel``: a K=2 field whose channel is 2^31 voxels (8.6 GB fp16), which no fused kernel
  can address. "auto" must take torch, and torch must get the labels right - they are known
  in closed form, the field being affine.
- ``speed``: fields both kernels take, if ``HAVERSACK_BASELINE_TRITON`` names a
  ``triton_gpu.py`` of another revision (e.g. ``git show <rev>:src/haversack/backends/triton_gpu.py``):
  labels must be bit-identical to the baseline's, and the time is compared, interleaved.

usage (from the haversack checkout):
    uv run --no-project modal run tools/restore_limits_modal.py
    HAVERSACK_BASELINE_TRITON=/tmp/triton_gpu_old.py uv run --no-project modal run tools/restore_limits_modal.py
    HAVERSACK_RESTORE_GPU=L40S uv run --no-project modal run tools/restore_limits_modal.py   # another card
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "haversack"
BASELINE = Path(os.environ["HAVERSACK_BASELINE_TRITON"]) if os.environ.get("HAVERSACK_BASELINE_TRITON") else None
#: The Modal GPU to prove it on. The limits are the kernel's, not the card's, but a timing only
#: means something beside another taken in the same container, on the same card.
GPU = os.environ.get("HAVERSACK_RESTORE_GPU", "A10G")

# the package is mounted from the checkout, so the change under review is what runs
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "triton>=3.0", "numpy>=2", "SimpleITK", "pydantic>=2")
    .add_local_dir(str(PKG), remote_path="/root/pkg/haversack")
)
if BASELINE is not None:
    image = image.add_local_file(str(BASELINE), remote_path="/root/baseline_triton_gpu.py")

app = modal.App("haversack-restore-limits-check", image=image)


@app.function(gpu=GPU, timeout=1800)
def check(with_baseline: bool) -> dict:
    import gc
    import statistics
    import time
    import warnings

    import torch
    sys.path.insert(0, "/root/pkg")
    import haversack as hv
    from haversack import Mapping, backends, build_tables
    from haversack.backends import triton_gpu
    assert triton_gpu.available(), triton_gpu.why_unavailable()
    cuda = torch.device("cuda")
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "triton": __import__("triton").__version__}

    def voronoi(K, shape, dtype, seed):
        """K soft Voronoi cells with noise, built a channel at a time on the GPU."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        lg = torch.empty((K, *shape), dtype=dtype, device=cuda)
        axes = [torch.arange(s, dtype=torch.float32, device=cuda) for s in shape]
        for k in range(K):
            c = torch.rand(3, generator=g, device=cuda) * torch.tensor(shape, dtype=torch.float32, device=cuda)
            d = ((axes[0] - c[0])[:, None, None] ** 2 + (axes[1] - c[1])[None, :, None] ** 2
                 + (axes[2] - c[2])[None, None, :] ** 2).sqrt_()
            d.mul_(-0.05).add_(6.0).add_(torch.randn(shape, generator=g, device=cuda), alpha=0.3)
            lg[k].copy_(d)
            del d
        return lg

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return r, time.perf_counter() - t

    def margins_at(lg, tables, where):
        """float64 top-1 minus top-2 of the interpolated field at the given output voxels."""
        tz, ty, tx = tables
        idx = torch.as_tensor(where, device=cuda)
        z, y, x = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy(), idx[:, 2].cpu().numpy()
        cz = [torch.as_tensor(a[z], device=cuda).long() for a in (tz.i0, tz.i1)]
        cy = [torch.as_tensor(a[y], device=cuda).long() for a in (ty.i0, ty.i1)]
        cx = [torch.as_tensor(a[x], device=cuda).long() for a in (tx.i0, tx.i1)]
        fz, fy, fx = (torch.as_tensor(t.f[i], device=cuda, dtype=torch.float64) for t, i in ((tz, z), (ty, y), (tx, x)))
        v = 0
        for a, wz in ((0, 1 - fz), (1, fz)):
            for b, wy in ((0, 1 - fy), (1, fy)):
                for c, wx in ((0, 1 - fx), (1, fx)):
                    v = v + lg[:, cz[a], cy[b], cx[c]].double() * (wz * wy * wx)
        top = v.topk(2, dim=0).values
        return (top[0] - top[1]).cpu().numpy()

    # ---- the reported field: 30 x 678 x 334 x 334, restored onto the CT's 311 x 512 x 512 ----
    src, dst = (678, 334, 334), (311, 512, 512)
    mapping = Mapping.center(dst, src)
    tables = build_tables(dst, src, mapping)
    for dtype in (torch.float16, torch.float32):
        lg = voronoi(30, src, dtype, seed=1)
        rec = {"logits": int(lg.numel()), "past_2_31": lg.numel() >= 2 ** 31}
        choice = backends.select("auto", cuda, tuple(lg.shape), dst)
        rec["auto"] = choice.name
        rec["fallback"] = choice.fallback
        hv.to_labels(lg, dst, mapping, backend="triton")                       # JIT
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tri, rec["triton_s"] = timed(lambda: hv.to_labels(lg, dst, mapping, backend="auto"))
        tor, rec["torch_s"] = timed(lambda: hv.to_labels(lg, dst, mapping, backend="torch"))
        diff = (tri != tor).nonzero()
        rec["differing_voxels"] = int(diff.shape[0])
        rec["labels_present"] = int(torch.unique(tor).numel())
        if diff.shape[0]:
            m = margins_at(lg, tables, diff[:20000].cpu().numpy())
            rec["max_margin_at_differences"] = float(m.max())
        if with_baseline:
            import importlib.util
            spec = importlib.util.spec_from_file_location("baseline_triton_gpu", "/root/baseline_triton_gpu.py")
            old = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old)
            try:
                old.run(lg, torch.zeros(dst, dtype=torch.uint8, device=cuda), tables, list(range(30)),
                        mode="argmax", paint=False, background=0, threshold=0.0)
                rec["baseline"] = "ran"
            except ValueError as e:
                rec["baseline"] = f"ValueError: {e}"
        out[f"reported_{str(dtype).split('.')[-1]}"] = rec
        del lg, tri, tor
        gc.collect()
        torch.cuda.empty_cache()

    # ---- one channel of 2^31 voxels: past every fused kernel, so torch ----
    src, dst = (2048, 1024, 1024), (64, 96, 80)
    lg = torch.empty((2, *src), dtype=torch.float16, device=cuda)
    lg[0].zero_()
    # affine in the source index, so trilinear interpolation is exact and the labels are known
    a = [(torch.arange(n, dtype=torch.float32, device=cuda) - (n - 1) / 2) / n for n in src]
    coef = (1.0, 0.7, -0.4)
    for z0 in range(0, src[0], 256):
        blk = (coef[0] * a[0][z0:z0 + 256, None, None] + coef[1] * a[1][None, :, None]
               + coef[2] * a[2][None, None, :] + 0.013)
        lg[1, z0:z0 + 256].copy_(blk)
        del blk
    rec = {"channel_voxels": int(lg[0].numel())}
    choice = backends.select("auto", cuda, tuple(lg.shape), dst)
    rec["auto"], rec["fallback"] = choice.name, choice.fallback
    mapping = Mapping.center(dst, src)
    tz, ty, tx = build_tables(dst, src, mapping)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, rec["torch_s"] = timed(lambda: hv.to_labels(lg, dst, mapping, backend="auto"))
    rec["warned"] = [str(w.message) for w in caught]
    coords = [(t.i0 + t.f).astype("float64") for t in (tz, ty, tx)]
    val = (coef[0] * ((coords[0] - (src[0] - 1) / 2) / src[0])[:, None, None]
           + coef[1] * ((coords[1] - (src[1] - 1) / 2) / src[1])[None, :, None]
           + coef[2] * ((coords[2] - (src[2] - 1) / 2) / src[2])[None, None, :] + 0.013)
    import numpy as np
    want = (val > 0).astype(np.uint8)
    sure = np.abs(val) > 5e-3                                  # beyond fp16 storage error
    g = got.cpu().numpy()
    rec["wrong_voxels"] = int(((g != want) & sure).sum())
    rec["labels_present"] = sorted(int(v) for v in np.unique(g))
    try:
        hv.to_labels(lg, dst, mapping, backend="triton")
        rec["triton_by_name"] = "ran"
    except ValueError as e:
        rec["triton_by_name"] = f"ValueError: {e}"
    out["channel_2_31"] = rec
    del lg, got
    gc.collect()
    torch.cuda.empty_cache()

    # ---- speed and identity against a baseline kernel, on fields both take ----
    if with_baseline:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baseline_triton_gpu", "/root/baseline_triton_gpu.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cases = {"near_limit_k30": (30, (620, 334, 334), (311, 512, 512)),
                 "total_k118": (118, (200, 224, 224), (209, 512, 512)),
                 "small_k8": (8, (128, 128, 128), (256, 256, 256))}
        for name, (K, src, dst) in cases.items():
            lg = voronoi(K, src, torch.float16, seed=7)
            mapping = Mapping.center(dst, src)
            tables = build_tables(dst, src, mapping)
            lut = list(range(K))
            res, times = {}, {"baseline": [], "new": []}
            kernels = {"baseline": old.run, "new": triton_gpu.run}
            for mode in ("argmax", "regions"):
                for which, run in kernels.items():             # JIT both first
                    run(lg, torch.zeros(dst, dtype=torch.uint8, device=cuda), tables, lut, mode=mode,
                        paint=False, background=0, threshold=0.0)
            for rep in range(9):
                for which in (("baseline", "new") if rep % 2 == 0 else ("new", "baseline")):
                    o = torch.zeros(dst, dtype=torch.uint8, device=cuda)
                    _, t = timed(lambda: kernels[which](lg, o, tables, lut, mode="argmax", paint=False,
                                                        background=0, threshold=0.0))
                    times[which].append(t)
                    res[which] = o
            same = bool(torch.equal(res["baseline"], res["new"]))
            reg = {}
            for which, run in kernels.items():
                o = torch.zeros(dst, dtype=torch.uint8, device=cuda)
                run(lg, o, tables, lut, mode="regions", paint=False, background=0, threshold=5.0)
                reg[which] = o
            out[f"speed_{name}"] = {
                "logits": int(lg.numel()),
                "baseline_median_s": round(statistics.median(times["baseline"]), 4),
                "new_median_s": round(statistics.median(times["new"]), 4),
                "baseline_min_s": round(min(times["baseline"]), 4),
                "new_min_s": round(min(times["new"]), 4),
                "argmax_identical": same,
                "regions_identical": bool(torch.equal(reg["baseline"], reg["new"]))}
            del lg, res, reg
            gc.collect()
            torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main():
    r = check.remote(BASELINE is not None)
    for k, v in r.items():
        print(f"{k}: {v}")
