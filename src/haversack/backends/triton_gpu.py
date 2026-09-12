"""CUDA backend: the fused restore as a Triton kernel.

The same shape as the Metal backend - one program per block of output voxels, gathering the 8
corners and keeping a running decision over K, so nothing K-channel-sized is ever materialized.
Measured on an NVIDIA A10 against the best pure-torch alternative: 3.5-4.7x faster
(`docs/backend-decision.md`), exact against the float64 reference on the first run.

The kernel lives in this module rather than being built from a string because ``@triton.jit``
recovers its source with ``inspect``.
"""
from __future__ import annotations

import functools

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR = None
except Exception as _e:                                     # pragma: no cover - depends on the box
    triton = None
    _TRITON_IMPORT_ERROR = _e

    class _Sham:                                            # so the @triton.jit below still parses
        @staticmethod
        def jit(fn):
            return fn

    triton_jit = _Sham.jit
else:
    triton_jit = triton.jit


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def why_unavailable() -> str:
    if triton is None:
        return f"triton is not installed ({_TRITON_IMPORT_ERROR})"
    if not torch.cuda.is_available():
        return "no CUDA device"
    return ""


# Offsets within one channel and into the output are 32-bit; only each channel's base is 64-bit.
OFFSET_LIMIT = 2 ** 31


def cannot_take(logits_shape, out_shape) -> str | None:
    """Why the kernel cannot address this field, or None if it can.

    ``backends.select`` asks this before "auto" takes the kernel, and ``run`` refuses on it, so
    the two cannot disagree. K itself is unlimited: the channel base is 64-bit (2026-09-11,
    after a K=30 field of 2.27e9 logits was refused). A channel of 2^31 voxels is a 1290^3 model
    grid; an output of 2^31 is a whole body at about 0.5 mm.
    """
    _, Zs, Ys, Xs = (int(v) for v in logits_shape)
    Za, Ya, Xa = (int(v) for v in out_shape)
    if Zs * Ys * Xs >= OFFSET_LIMIT:
        return (f"a channel of {Zs}x{Ys}x{Xs} = {Zs * Ys * Xs:,} voxels is 2^31 or more, "
                f"past its 32-bit in-channel offsets")
    if Za * Ya * Xa >= OFFSET_LIMIT:
        return (f"an output of {Za}x{Ya}x{Xa} = {Za * Ya * Xa:,} voxels is 2^31 or more, "
                f"past its 32-bit output offsets")
    return None


if triton is not None:

    @triton.jit
    def fused_restore(logits_ptr, z0_ptr, z1_ptr, zf_ptr, y0_ptr, y1_ptr, yf_ptr,
                      x0_ptr, x1_ptr, xf_ptr, lut_ptr, out_ptr,
                      K, Zs, Ys, Xs, Ya, Xa, n_out, threshold, background,
                      MODE: tl.constexpr, PAINT: tl.constexpr, BLOCK: tl.constexpr):
        """One program per BLOCK output voxels.

        ``MODE`` 0 = argmax over channels, 1 = per-region threshold + paint in channel order.
        ``PAINT`` leaves the output untouched where the decision is background, which is how
        multi-model tasks composite into one buffer.

        Offsets within a channel and into the output are 32-bit, as in the Metal backend where
        64-bit ones cost 2x; each channel's base is 64-bit, one scalar per channel, so K
        channels together may pass 2^31. ``cannot_take`` is the host-side check.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out

        x = offs % Xa
        y = (offs // Xa) % Ya
        z = offs // (Xa * Ya)

        z0 = tl.load(z0_ptr + z, mask=mask, other=0)
        y0 = tl.load(y0_ptr + y, mask=mask, other=0)
        x0 = tl.load(x0_ptr + x, mask=mask, other=0)
        # -1 marks an output voxel that falls outside the source extent
        inside = (z0 >= 0) & (y0 >= 0) & (x0 >= 0)
        z0 = tl.where(inside, z0, 0)
        y0 = tl.where(inside, y0, 0)
        x0 = tl.where(inside, x0, 0)
        z1 = tl.load(z1_ptr + z, mask=mask, other=0)
        y1 = tl.load(y1_ptr + y, mask=mask, other=0)
        x1 = tl.load(x1_ptr + x, mask=mask, other=0)
        zf = tl.load(zf_ptr + z, mask=mask, other=0.0)
        yf = tl.load(yf_ptr + y, mask=mask, other=0.0)
        xf = tl.load(xf_ptr + x, mask=mask, other=0.0)

        plane = Ys * Xs
        b000 = z0 * plane + y0 * Xs + x0
        b001 = z0 * plane + y0 * Xs + x1
        b010 = z0 * plane + y1 * Xs + x0
        b011 = z0 * plane + y1 * Xs + x1
        b100 = z1 * plane + y0 * Xs + x0
        b101 = z1 * plane + y0 * Xs + x1
        b110 = z1 * plane + y1 * Xs + x0
        b111 = z1 * plane + y1 * Xs + x1
        chan = Zs * plane

        wx0 = 1.0 - xf
        wy0 = 1.0 - yf
        wz0 = 1.0 - zf
        load_mask = mask & inside

        best = tl.full((BLOCK,), float("-inf"), tl.float32)
        best_k = tl.zeros((BLOCK,), tl.int32)
        label = tl.full((BLOCK,), background, tl.int32)
        hit = tl.zeros((BLOCK,), tl.int1)
        # Each channel's base is a pointer advanced by `chan`, so it is 64-bit. Until 2026-09-11
        # it was an int32 `k * chan`, which wrapped once K channels passed 2^31 logits (a K=30
        # model on a whole-body CT at 1.5 mm is 2.27e9). A `k.to(tl.int64) * chan` base fixed
        # that too but cost 3-4 % on an A10; the carried pointer costs nothing measurable.
        ch = logits_ptr
        for k in range(0, K):
            c00 = (tl.load(ch + b000, mask=load_mask, other=0.0).to(tl.float32) * wx0
                   + tl.load(ch + b001, mask=load_mask, other=0.0).to(tl.float32) * xf)
            c01 = (tl.load(ch + b010, mask=load_mask, other=0.0).to(tl.float32) * wx0
                   + tl.load(ch + b011, mask=load_mask, other=0.0).to(tl.float32) * xf)
            c10 = (tl.load(ch + b100, mask=load_mask, other=0.0).to(tl.float32) * wx0
                   + tl.load(ch + b101, mask=load_mask, other=0.0).to(tl.float32) * xf)
            c11 = (tl.load(ch + b110, mask=load_mask, other=0.0).to(tl.float32) * wx0
                   + tl.load(ch + b111, mask=load_mask, other=0.0).to(tl.float32) * xf)
            v = (c00 * wy0 + c01 * yf) * wz0 + (c10 * wy0 + c11 * yf) * zf
            if MODE == 0:
                upd = v > best
                best = tl.where(upd, v, best)
                best_k = tl.where(upd, k, best_k)
            else:
                over = v > threshold                       # channel order is paint priority
                label = tl.where(over, tl.load(lut_ptr + k), label)
                hit = hit | over
            ch += chan
        if MODE == 0:
            label = tl.load(lut_ptr + best_k, mask=mask, other=background)
            hit = best_k != 0

        out_dtype = out_ptr.dtype.element_ty
        if PAINT:
            tl.store(out_ptr + offs, label.to(out_dtype), mask=mask & inside & hit)
        else:
            tl.store(out_ptr + offs, tl.where(inside, label, background).to(out_dtype), mask=mask)


@functools.lru_cache(maxsize=4)
def warmup(mode: str = "argmax", paint: bool = False) -> bool:
    """Compile the kernel on a token input so the first real restore does not pay for it.

    Triton JITs on first call - about 1.3 s, which is most of a single restore and pure waste
    on a one-shot job. Cached per (mode, paint) since those are ``constexpr``, and cheap to
    call from a background thread while the network runs.
    """
    if not available():
        return False
    from ..mapping import Mapping
    from ..tables import build_tables
    src, out_shape = (4, 4, 4), (5, 5, 5)
    logits = torch.zeros((2, *src), dtype=torch.float16, device="cuda")
    out = torch.zeros(out_shape, dtype=torch.uint8, device="cuda")
    tables = build_tables(out_shape, src, Mapping.center(out_shape, src))
    run(logits, out, tables, np.arange(2, dtype=np.int32), mode=mode, paint=paint,
        background=0, threshold=0.0)
    torch.cuda.synchronize()
    return True


@torch.no_grad()
def run(logits: torch.Tensor, out: torch.Tensor, tables, lut, *, mode: str, paint: bool,
        background: int, threshold: float, block: int = 256) -> None:
    if not available():
        raise RuntimeError(f"haversack.backends.triton_gpu: {why_unavailable()}")
    if logits.device.type != "cuda" or out.device.type != "cuda":
        raise ValueError("haversack.backends.triton_gpu: logits and out must be on a CUDA device")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"haversack.backends.triton_gpu: logits must be float16/bfloat16/float32; got {logits.dtype}")
    if out.dtype not in (torch.uint8, torch.uint16):
        raise TypeError(f"haversack.backends.triton_gpu: out must be uint8 or uint16; got {out.dtype}")
    if not out.is_contiguous():
        raise ValueError("haversack.backends.triton_gpu: out must be contiguous")
    logits = logits.contiguous()
    K, Zs, Ys, Xs = (int(v) for v in logits.shape)
    Za, Ya, Xa = (int(v) for v in out.shape)
    why = cannot_take((K, Zs, Ys, Xs), (Za, Ya, Xa))
    if why is not None:
        raise ValueError(f"haversack.backends.triton_gpu: {why}; restore with backend='torch'")
    n_out = Za * Ya * Xa
    dev = logits.device

    def to_i(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int32)).to(dev)

    def to_f(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    tz, ty, tx = tables
    grid = (triton.cdiv(n_out, block),)
    fused_restore[grid](
        logits, to_i(tz.i0), to_i(tz.i1), to_f(tz.f), to_i(ty.i0), to_i(ty.i1), to_f(ty.f),
        to_i(tx.i0), to_i(tx.i1), to_f(tx.f), to_i(lut), out,
        K, Zs, Ys, Xs, Ya, Xa, n_out, float(threshold), int(background),
        MODE=0 if mode == "argmax" else 1, PAINT=bool(paint), BLOCK=block)
