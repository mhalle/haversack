"""One ranked field for a union task, composed from its models' own fields (2026-09-24).

A union such as ``ts.v2:total`` is several models whose labels are painted in order: each
model claims a voxel where its argmax is not its background, and a later model's claim paints
over an earlier one. Their logits share no scale (docs/ranked-composition.md, "across parts
there is no algebra"), so no joint softmax exists to store. A store of the parts side by side
is a record of the models, not of the task: every reader has to repeat the painting, and the
seams between parts appear in no part's field. The store holds the TASK's field instead, built
here from the parts' stored deficits by one rule.

Per voxel, with ``d^p`` part p's deficits (the restore field: 0 for its winner, negative
behind, the true gap for a kept shell class, ``-clip`` for an unnamed one):

- ``c_p = max_{k>=1} d^p_k - d^p_0``, part p's claim: positive exactly where it paints.
- ``C_p = min(c_p, min_{q>p} -c_q)``: p wins the painting contest - it claims, and no later
  part does. ``B = min_p -c_p``: nobody claims.
- label ``lut_p[k]`` scores ``s * C_p + d^p_k - max_{k>=1} d^p_k``; label 0 scores ``s * B``.
  A label two parts can paint takes the larger score.

Exactly one contest term is positive at a voxel, so the composed argmax IS the painted label
on the model grid (measured: every voxel of ``total`` on a 418 M-voxel CT). Inside the winning
part its classes keep their own softmax gaps. ``s`` is :data:`MARGIN_SCALE`: without it a
part's claim enters twice - the loser's score falls by it and the winner's rises by it - so a
gap between parts was twice the part's own, reached the clip in half the distance, and a
linear restore swelled thin structures (ribs 3.6x, 0.057 % of voxels from the direct labels).
Scaled by 1/2 the gap between two parts' labels is the claim itself: 0.013 % from the direct
labels, the rest being where a minimum does not commute with interpolation, and against
upstream TotalSegmentator 2.18.0 the composed labels scored mean Dice 0.93634 where the direct
ones scored 0.93597 (one CT, linear restore, clip 16; clip 8 left the ribs slightly worse).

The composed scores are NOT a softmax: within one part their differences are logit gaps, across
parts they are painting margins. So the field is encoded without a tail and says so
(``scores: composed``); a probability read from it would be a number with no meaning.

Encoded slab by slab with one plane of halo: the encoder's keep rule looks at a voxel's 26
neighbors and nothing further, so each slab's interior is what one encode of the whole field
would write (held byte for byte by tests/test_ranked_compose.py).
"""
from __future__ import annotations

import numpy as np

from .errors import InputError

#: The clip a composed field is encoded at. Measured against 8 on ``total``: 8 floored enough
#: cross-part gaps to leave thin ribs slightly worse against upstream; 16 did not, at no size cost.
COMPOSED_CLIP = 16.0
#: The scale each part's painting margin enters the composed scores at (see the module docstring).
MARGIN_SCALE = 0.5
#: The least a stored non-winner trails its part's winner by, in logits. A gap under half the
#: level table's first step is stored as the byte whose level is 0, so it decodes as a tie - and
#: a tie handed to the encoder is broken by label index, which can pick a class the part did not
#: (or count a part as not painting where its argmax says it does: its claim decodes to 0). The
#: stored winner (``ranks[0]``) knows the order, so every other kept class is held this far
#: behind it: far below any step of the table, and enough to keep the part's own order.
TIE_FLOOR = 1e-4
#: The rule's name as the store states it, and its version: a change that moves a composed field's
#: bytes bumps ``RULE_VERSION`` (it is part of ``ranked_output.ranked_tag``, so stores recompute).
RULE = "painting"
RULE_VERSION = 1


def _torch():
    import torch
    return torch


def _device(device):
    if device not in (None, "auto"):
        return device
    from .resample import best_device
    return str(best_device())


def deficits(ranks, support, meta, z0: int, z1: int, device):
    """``(K, z1 - z0, Y, X)`` float32 on ``device``: every channel's stored deficit over planes
    ``z0:z1`` - ``rankfield.deficit`` for all channels at once (0 for the winner, ``-levels[s]``
    for a kept class, ``-clip`` for an unnamed one)."""
    torch = _torch()
    from rankfield import levels
    K, clip = int(meta["classes"]), float(meta["clip"])
    lv = torch.as_tensor(levels(meta), device=device)
    rk = torch.from_numpy(np.asarray(ranks[:, z0:z1]).astype(np.int64)).to(device)
    su = torch.from_numpy(np.asarray(support[:, z0:z1]).astype(np.int64)).to(device)
    out = torch.full((K + 1,) + tuple(rk.shape[1:]), -clip, dtype=torch.float32, device=device)
    out.scatter_(0, rk[0:1], 0.0)
    for j in range(1, rk.shape[0]):
        out.scatter_(0, rk[j:j + 1], -lv[su[j - 1]].unsqueeze(0))
    return out[1:]                                   # row 0 took the sentinel (rank value 0)


def _ordered(d, ranks, z0, z1, device):
    """``d`` with every channel but the stored winner held at least :data:`TIE_FLOOR` behind it."""
    torch = _torch()
    win = torch.from_numpy(np.asarray(ranks[0, z0:z1]).astype(np.int64) - 1).to(device)
    held = torch.clamp(d, max=-TIE_FLOOR)
    return held.scatter_(0, win.unsqueeze(0), 0.0)


def _check_one_grid(parts):
    first = parts[0][2]
    for name, _arrays, meta in parts[1:]:
        for key in ("model_grid", "envelope", "spacing_zyx"):
            if meta.get(key) != first.get(key):
                raise InputError(
                    f"part {name!r} is not on part {parts[0][0]!r}'s grid ({key} differs): a union's "
                    "parts are composed voxel by voxel, so they must share one grid - an envelope "
                    "crop (--envelope) can give each model its own; run without one")


def compose(parts, *, depth: int, device=None, slab: int = 8):
    """The task field of a union: ``parts`` is ``[(name, (ranks, support), meta), ...]`` in paint
    order, each on the same grid. Returns ``(ranks, support, meta, labels)`` - ``labels`` the
    channel -> label value table, ascending with 0 first - encoded at :data:`COMPOSED_CLIP`
    without a tail. ``meta`` is the encoder's plus ``scores``/``composed``; the caller adds the
    placement it took from the parts."""
    torch = _torch()
    from rankfield import encode
    if len(parts) < 2:
        raise ValueError("compose takes a union: two or more parts")
    _check_one_grid(parts)
    dev = torch.device(_device(device))
    luts = [np.asarray(meta["labels"], dtype=np.int64) for _n, _a, meta in parts]
    labels = sorted({0} | {int(v) for lut in luts for v in lut})
    index = np.full(max(labels) + 1, -1, dtype=np.int64)
    index[labels] = np.arange(len(labels))
    chan = [torch.as_tensor(index[lut[1:]], device=dev) for lut in luts]   # channel 0 never paints
    G = len(labels)
    Z, Y, X = (int(v) for v in np.asarray(parts[0][1][1]).shape[1:])
    s = float(MARGIN_SCALE)

    def scores(lo, hi):
        D = [_ordered(deficits(a[0], a[1], m, lo, hi, dev), a[0], lo, hi, dev)
             for _n, a, m in parts]
        claims = [d[1:].amax(0) - d[0] for d in D]
        later, run = [None] * len(D), None           # min over later parts of -c_q
        for p in range(len(D) - 1, -1, -1):
            later[p] = run
            run = -claims[p] if run is None else torch.minimum(run, -claims[p])
        L = torch.full((G,) + tuple(run.shape), -1e4, dtype=torch.float32, device=dev)
        L[0] = s * run                               # nobody claims
        for p, d in enumerate(D):
            win = claims[p] if later[p] is None else torch.minimum(claims[p], later[p])
            top = d[1:].amax(0)
            sp = s * win.unsqueeze(0) + d[1:] - top.unsqueeze(0)
            L.scatter_reduce_(0, chan[p].view(-1, 1, 1, 1).expand_as(sp), sp, reduce="amax")
        return L

    ranks = support = meta = None
    for z0 in range(0, Z, slab):
        z1 = min(Z, z0 + slab)
        lo, hi = max(0, z0 - 1), min(Z, z1 + 1)      # one plane of halo: the keep rule's reach
        enc = encode(scores(lo, hi), depth=depth, clip=COMPOSED_CLIP, with_tail=False)
        a = z0 - lo
        if ranks is None:
            ranks = np.empty((enc.ranks.shape[0], Z, Y, X), enc.ranks.dtype)
            support = np.empty((enc.support.shape[0], Z, Y, X), np.uint8)
            meta = dict(enc.meta)
        ranks[:, z0:z1] = enc.ranks[:, a:a + (z1 - z0)]
        support[:, z0:z1] = enc.support[:, a:a + (z1 - z0)]
    meta["shape"] = [Z, Y, X]
    meta["scores"] = "composed"
    meta["composed"] = {"rule": RULE, "rule_version": RULE_VERSION, "margin_scale": s,
                        "parts": [{"part": n, "labels": [int(v) for v in m["labels"]],
                                   **({"softmax": m["softmax"]} if "softmax" in m else {})}
                                  for n, _a, m in parts]}
    return ranks, support, meta, labels
