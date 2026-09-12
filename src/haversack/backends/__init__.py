"""Backend registry. Every backend exposes ``available()`` and
``run(logits, out, tables, lut, *, mode, paint, background, threshold, **opts)``
writing labels into ``out`` in place. A fused one also exposes
``cannot_take(logits_shape, out_shape)``: why its kernel cannot address a field, or None."""
from __future__ import annotations

from typing import NamedTuple

import torch

from . import metal, torch_gather, triton_gpu

BACKENDS = {"torch": torch_gather, "metal": metal, "triton": triton_gpu}
# the fused kernel "auto" takes on each device type, if it is installed and can address the field
FUSED = {"mps": "metal", "cuda": "triton"}


class Choice(NamedTuple):
    name: str
    module: object
    # Why "auto" did not take the device's fused kernel, which is several times faster than
    # the torch backend: a deviation for the caller to report. None when it did, or when the
    # device has no fused kernel to take.
    fallback: str | None = None


def select(name: str, device: torch.device, logits_shape, out_shape) -> Choice:
    """The backend that restores a ``(K, Zs, Ys, Xs)`` field onto ``out_shape`` on ``device``.

    "auto" asks the fused kernel whether it can address the field before taking it, and
    otherwise takes the torch backend, which has no offset limit, saying why. Until 2026-09-11
    it went by device type alone, and the Triton kernel's refusal of a 2.27e9-logit field (a
    K=30 model on a whole-body CT with no envelope) went straight through ``segment``. A
    backend asked for by name is never replaced: it raises, naming the one that would run.
    """
    shapes = (tuple(int(v) for v in logits_shape), tuple(int(v) for v in out_shape))
    if name == "auto":
        fused = FUSED.get(device.type)
        if fused is None or not BACKENDS[fused].available():
            return Choice("torch", torch_gather)
        why = BACKENDS[fused].cannot_take(*shapes)
        if why is None:
            return Choice(fused, BACKENDS[fused])
        return Choice("torch", torch_gather, f"the {fused} kernel cannot take this field: {why}")
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)} or 'auto'")
    if name != "torch":
        homes = sorted(d for d, n in FUSED.items() if n == name)
        if device.type not in homes:
            raise ValueError(f"backend={name!r} needs logits on a {' or '.join(map(repr, homes))} device")
        if name == "triton" and not triton_gpu.available():
            raise ValueError(f"backend='triton' unavailable: {triton_gpu.why_unavailable()}")
        why = BACKENDS[name].cannot_take(*shapes)
        if why is not None:
            raise ValueError(f"backend={name!r} cannot take this field: {why}; "
                             f"use backend='torch' (or 'auto', which does)")
    return Choice(name, BACKENDS[name])


def available_backends() -> list[str]:
    return [n for n, m in BACKENDS.items() if m.available()]
