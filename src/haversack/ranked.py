"""haversack's side of the ranked encoding: the emit hook and the two derived caches.

The encoding itself - what is stored, the level table, the keep rule, the fields and the
restore - is the `rankfield` library (a sibling repo); this module re-exports its names so
the pipeline, the tools and the tests keep one import path, and adds what belongs to the
pipeline: :class:`RankedSpec` (where a run's output distribution goes), :func:`emit` (the
one seam every engine hands its logits through), and the distance and junction caches
(derived layers a store carries beside the encoding; see docs/ranked-distance-gpu.md).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

try:
    from rankfield import (CLIP, DEFAULT_DEPTH, FORMAT_VERSION, GAP_UNIT, RankField, decode_groups,
                           deficit, encode, encode_regions, levels, margin, probabilities, settle_ties,
                           to_device)
    from rankfield import SUPPORT_MAX, TAIL_MAX, ZERO_LEVEL  # noqa: F401 - re-exported for tools and tests
except ModuleNotFoundError as _e:
    if _e.name != "rankfield":
        raise
    raise ImportError("haversack.ranked is a shim over rankfield, which the duckn extra installs: "
                      "uv sync --extra duckn (or uv pip install 'haversack[duckn]'; on pip alone, "
                      "rankfield @ git+https://github.com/mhalle/rankfield.git)") from None

RankedCode = RankField                  # the name the pipeline and the tools grew up with
RANKED_VERSION = FORMAT_VERSION
_settle_ties = settle_ties

def _lut_np(levels, clip):
    if levels is None:                                # format 0.2: uniform over the clip
        return ((1.0 - np.arange(256, dtype=np.float64) / 255.0) * clip).astype(np.float32)
    return np.ascontiguousarray(levels, dtype=np.float32)


__all__ = ["CLIP", "DEFAULT_DEPTH", "GAP_UNIT", "RANKED_VERSION", "RankedCode", "RankedSpec",
           "decode_groups", "deficit", "distance_field", "emit", "encode", "encode_regions",
           "junction_field", "levels", "margin", "probabilities", "to_device"]


@dataclass(frozen=True)
class RankedSpec:
    """Ask a run to emit its output distribution, and say where each part's code goes.

    A sink rather than a return value, deliberately: a multi-model task holds one part's
    logits at a time and frees them, and the uncompressed codes are far larger than the
    stored form - so each part is handed over as it is produced and can be written and
    dropped. It is also the shape the layers want, since one part is one independent file.
    """

    sink: Callable[[str, RankedCode], None]
    depth: int = DEFAULT_DEPTH
    clip: float = CLIP


def emit(spec, part, logits, /, **meta) -> "RankedCode | None":
    """Encode ``logits`` into ``spec``'s sink, stamping ``meta`` onto the code.

    The one seam every engine hands its output distribution through. It exists because the
    logits are alive only between the network and the restore, so each engine must do this
    for itself at its own call site - and four copies would drift, in the depth and clip the
    spec asked for and in what lands in ``meta``.

    ``spec`` may be ``None``, so a call site can be unconditional rather than guarded.

    The first three are positional-only on purpose: ``meta`` is open-ended and engine-chosen,
    and the nnU-Net path really does stamp ``part`` into it, so a keyword ``part`` here would
    collide with the sink key and raise.

    What an engine puts in ``meta`` is its own business - the grids differ, the label
    mapping differs - but it must be enough to redo the restore later: the grid the field
    was computed on, the grid it restores onto, and channel -> label. Without that the
    arrays are only a picture of one run. Stamp ``engine`` too; with more than one engine
    emitting, a reader cannot otherwise tell what produced the file.
    """
    if spec is None:
        return None
    code = encode(logits, depth=spec.depth, clip=spec.clip)
    code.meta.update(meta)
    spec.sink(str(part), code)
    return code


def distance_field(ranks, support, *, clip: float, spacing_zyx, truncation: float,
                   distance_max: int = 255, device=None, levels=None) -> np.ndarray:
    """``(Z, Y, X)`` uint8: distance in mm to the nearest surface, on a GPU when there is one.

    The nearest surface is the nearest place the argmax changes, whichever class pair forms
    it - found from ``ranks[0]``, never from a rank pair, and located sub-voxel by the two
    deficits of the pair that actually swaps. Encoded counting up from the truncation
    (``distance_max`` on the surface, 0 at or beyond ``truncation``) so zero stays the
    sentinel and empty chunks elide. See docs/ranked-distance-gpu.md for the design and
    tools/ranked_build_store.py for the numpy reference this must agree with to one quantum.

    DENSE, FOR CUDA. Same math as the reference, Jacobi form, no sweep ordering, no
    atomics: seeding is per-axis slice writes and propagation is elementwise over the whole
    grid, with none of the band bookkeeping. On an L40S a 52 Mvoxel part takes 0.5-0.6 s,
    five times the banded numpy reference on an M2 (2.7 s), which is why the emit computes it
    there. The same kernel does NOT transfer to Apple hardware: 15.4 s on MPS and 11.6 s on
    torch's CPU backend for the same part (byte-identical results), because dense work is
    memory traffic and only the CUDA part has the bandwidth for it. Locally the reference is
    the path; a banded torch version is the obvious next step if a local GPU path matters.
    """
    from .resample import best_device

    dev = torch.device(device) if device is not None else best_device()
    def as_t(a):
        return a.to(dev) if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    rk = as_t(ranks)
    su = as_t(support)
    h = [float(v) for v in spacing_zyx]
    T = float(truncation)
    clip = float(clip)
    lut = torch.as_tensor(_lut_np(levels, clip), device=dev)     # the block's level table
    big = T * 4.0
    win = rk[0]
    inf = torch.tensor(float("inf"), device=dev)

    def deficit(rk_s, su_s, want):
        """Logit deficit of class ``want`` under the slice's own winner (dense, on device)."""
        d = torch.full(want.shape, clip, dtype=torch.float32, device=dev)
        d = torch.where(rk_s[0] == want, torch.zeros((), device=dev), d)
        for j in range(1, rk_s.shape[0]):
            gap = lut[su_s[j - 1].long()]
            d = torch.where(rk_s[j] == want, gap, d)
        return d

    # ---- seed: argmax flips per axis, crossing interpolated from the swapping pair ----
    d = torch.full(win.shape, float("inf"), dtype=torch.float32, device=dev)
    for axis, step in enumerate(h):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        flip = win[lo] != win[hi]
        if not bool(flip.any()):
            continue
        dq_a = deficit(rk[(slice(None),) + lo], su[(slice(None),) + lo], win[hi])
        dp_b = deficit(rk[(slice(None),) + hi], su[(slice(None),) + hi], win[lo])
        denom = dq_a + dp_b
        t = torch.where(denom > 1e-9, dq_a / denom, torch.full((), 0.5, device=dev))
        d[lo] = torch.minimum(d[lo], torch.where(flip, t * step, inf))
        d[hi] = torch.minimum(d[hi], torch.where(flip, (1.0 - t) * step, inf))

    seed_mask = torch.isfinite(d)
    if not bool(seed_mask.any()):
        return np.zeros(win.shape, np.uint8)
    seed_vals = torch.where(seed_mask, d, torch.full((), big, device=dev))
    d = seed_vals.clone()

    # ---- propagate: dense Jacobi Godunov, |grad d| = 1 ----
    n_iter = int(np.ceil(T / min(h))) + 4
    hs = [torch.full((), v, dtype=torch.float32, device=dev) for v in h]
    for _ in range(n_iter):
        p = torch.nn.functional.pad(d, (1, 1, 1, 1, 1, 1), value=big)
        trip = [
            (torch.minimum(p[:-2, 1:-1, 1:-1], p[2:, 1:-1, 1:-1]), hs[0]),
            (torch.minimum(p[1:-1, :-2, 1:-1], p[1:-1, 2:, 1:-1]), hs[1]),
            (torch.minimum(p[1:-1, 1:-1, :-2], p[1:-1, 1:-1, 2:]), hs[2]),
        ]
        for i, j in ((0, 1), (1, 2), (0, 1)):          # 3-element sort, h travels with its axis
            ai, hi_v = trip[i]
            aj, hj_v = trip[j]
            swap = ai > aj
            trip[i] = (torch.where(swap, aj, ai), torch.where(swap, hj_v, hi_v))
            trip[j] = (torch.where(swap, ai, aj), torch.where(swap, hi_v, hj_v))
        (a0, h0), (a1, h1), (a2, h2) = trip
        w0, w1, w2 = 1.0 / (h0 * h0), 1.0 / (h1 * h1), 1.0 / (h2 * h2)

        sol = a0 + h0                                          # one active axis
        use2 = sol > a1
        A2, B2 = w0 + w1, a0 * w0 + a1 * w1
        C2 = a0 * a0 * w0 + a1 * a1 * w1
        disc2 = B2 * B2 - A2 * (C2 - 1.0)
        d2 = (B2 + torch.sqrt(torch.clamp(disc2, min=0.0))) / A2
        ok2 = use2 & (disc2 >= 0) & (d2 <= a2)
        sol = torch.where(ok2, d2, sol)
        A3, B3 = A2 + w2, B2 + a2 * w2
        C3 = C2 + a2 * a2 * w2
        disc3 = B3 * B3 - A3 * (C3 - 1.0)
        d3 = (B3 + torch.sqrt(torch.clamp(disc3, min=0.0))) / A3
        sol = torch.where(use2 & ~ok2 & (disc3 >= 0), d3, sol)

        d = torch.where(seed_mask, seed_vals,
                        torch.minimum(d, torch.clamp(sol, max=big)))

    q = torch.round((1.0 - d / T) * distance_max)
    q = torch.where(d < T, torch.clamp(q, 0, distance_max),
                    torch.zeros((), device=dev))
    return q.to(torch.uint8).cpu().numpy()


def junction_field(ranks, support, *, clip: float, spacing_zyx, truncation: float,
                   reach: int | None = None, junction_zero: int = 128, junction_span: int = 127,
                   device=None, levels=None) -> tuple[np.ndarray, np.ndarray]:
    """``(junction, pair)``: the triple-line layer, on a GPU when there is one.

    The signed distance in mm to the level set where the two leading real classes' logits
    tie, positive on the lower class's side, and which two they are - written only in tubes
    around the triple lines, where such an interface meets a third label. It answers the one
    question the distance field cannot: where two structures divide a surface they share. The
    numpy reference in tools/ranked_build_store.py says why and how; this is the same
    algorithm on tensors and must agree with it to one quantum (tests/test_ranked_junction.py).

    Same shape of work as the reference: the triple cells are found densely (eight corner
    gathers, elementwise), the tubes by `reach` dilations, and the deficits are gathered only
    at the tube voxels - a fraction of a per cent of the volume - so the field is cheap on
    either device. Measured on a 52 Mvoxel part (1.5 mm `total`, part 0, M2 Air): the numpy
    reference 0.8 s, this 1.4 s on MPS and 10.7 s on torch's CPU backend, byte-identical.
    So locally the reference is the path and this one belongs where the arrays already sit on
    a CUDA device - the emit on the Modal worker, beside the distance field. A class absent
    from a voxel's rank list is floored at the clip.
    """
    from .resample import best_device

    dev = torch.device(device) if device is not None else best_device()
    def as_t(a):
        return a.to(dev) if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    rk = as_t(ranks)
    su = as_t(support)
    h = [float(v) for v in spacing_zyx]
    T = float(truncation)
    clip = float(clip)
    lut = torch.as_tensor(_lut_np(levels, clip), device=dev)     # the block's level table
    if reach is None:
        reach = int(np.ceil(T / min(h))) + 1
    win = rk[0]
    Z, Y, X = win.shape

    # ---- triple cells: three or more labels among a cell's eight corners ----
    corners = [win[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx]
               for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
    lo = corners[0]
    hi = corners[0]
    for c in corners[1:]:
        lo = torch.minimum(lo, c)
        hi = torch.maximum(hi, c)
    third = torch.zeros(lo.shape, dtype=torch.bool, device=dev)
    for c in corners:
        third |= (c != lo) & (c != hi)
    tube = torch.zeros(win.shape, dtype=torch.bool, device=dev)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                tube[dz:Z - 1 + dz, dy:Y - 1 + dy, dx:X - 1 + dx] |= third
    # dilation: a 3x3x3 max-pool per voxel of reach, on a float view (pooling has no bool)
    if reach > 0 and bool(tube.any()):
        f = tube.to(torch.float32)[None, None]
        for _ in range(reach):
            f = torch.nn.functional.max_pool3d(f, kernel_size=3, stride=1, padding=1)
        tube = f[0, 0] > 0.5

    junction = torch.zeros(win.shape, dtype=torch.uint8, device=dev)
    pair = torch.zeros((2,) + tuple(win.shape), dtype=rk.dtype, device=dev)
    idx = torch.nonzero(tube, as_tuple=True)
    N = idx[0].numel()
    if N == 0:
        return junction.cpu().numpy(), pair.cpu().numpy()

    # ---- the pair: the first two real classes in each voxel's rank list ----
    cols = rk[(slice(None),) + idx]                      # (planes, N)
    real = (cols != 1) & (cols != 0)                     # background is class 0, held as 1
    a = torch.zeros(N, dtype=rk.dtype, device=dev)
    b = torch.zeros(N, dtype=rk.dtype, device=dev)
    seen = torch.zeros(N, dtype=torch.int64, device=dev)
    for j in range(cols.shape[0]):
        r = real[j]
        first, second = r & (seen == 0), r & (seen == 1)
        a = torch.where(first, cols[j], a)
        b = torch.where(second, cols[j], b)
        seen += r.to(torch.int64)
    have = seen >= 2
    swap = have & (b < a)
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)

    def deficit(rk_c, su_c, want):
        d = torch.full(want.shape, clip, dtype=torch.float32, device=dev)
        d = torch.where(rk_c[0] == want, torch.zeros((), device=dev), d)
        for j in range(1, rk_c.shape[0]):
            gap = lut[su_c[j - 1].long()]
            d = torch.where(rk_c[j] == want, gap, d)
        return d

    z, y, x = idx

    def m_at(zz, yy, xx):
        q = (zz, yy, xx)
        rc = rk[(slice(None),) + q]
        sc = su[(slice(None),) + q]
        return deficit(rc, sc, b) - deficit(rc, sc, a)

    m0 = m_at(z, y, x)
    grad2 = torch.zeros(N, dtype=torch.float32, device=dev)
    for axis, (arr, n) in enumerate(((z, Z), (y, Y), (x, X))):
        plus = [z, y, x]
        minus = [z, y, x]
        plus[axis] = torch.clamp(arr + 1, max=n - 1)
        minus[axis] = torch.clamp(arr - 1, min=0)
        span = (plus[axis] - minus[axis]).to(torch.float32) * h[axis]
        diff = m_at(*plus) - m_at(*minus)
        g = torch.where(span > 0, diff / torch.where(span > 0, span, torch.ones_like(span)),
                        torch.zeros_like(diff))
        grad2 += g * g
    gmag = torch.sqrt(grad2)
    s = torch.where(gmag > 1e-6, m0 / torch.where(gmag > 1e-6, gmag, torch.ones_like(gmag)),
                    torch.sign(m0) * T)
    s = torch.clamp(s, -T, T)
    q = torch.clamp(torch.round(junction_zero + s / T * junction_span), 1, 255).to(torch.uint8)
    q = torch.where(have, q, torch.zeros((), dtype=torch.uint8, device=dev))
    junction[idx] = q
    zero = torch.zeros((), dtype=rk.dtype, device=dev)
    pair[(0,) + idx] = torch.where(have, a, zero)
    pair[(1,) + idx] = torch.where(have, b, zero)
    return junction.cpu().numpy(), pair.cpu().numpy()

