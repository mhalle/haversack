"""Listing and cleaning haversack's on-disk stores (2026-09-03).

Two kinds of thing live on disk: TRANSIENT caches - fetched inputs, server results, engine
checkpoints - all of which are re-fetchable, so `cache clean` may sweep them; and PROVISIONED
weights (nnU-Net / TotalSegmentator / MOOSE / MRSegmentator models), which are slow to
re-download and, when license-gated, installed by hand and not re-fetchable at all. This
module lists both but only ever cleans the transient ones; weights are removed one at a time
through `weights remove`, never swept.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path


def _du(path: Path) -> tuple[int, int]:
    """(item count, total bytes) under ``path``; (0, 0) if it does not exist. An item is a
    top-level entry (a cached input, a dataset), not every file."""
    if not path or not Path(path).exists():
        return 0, 0
    p = Path(path)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return -1, total    # byte total only; item counts are per-store (see usage)


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024


def stores() -> list[dict]:
    """Every store haversack keeps on disk: name, path, whether `cache clean` may sweep it."""
    from .sources import default_input_cache
    from .tasks import weights_root

    def results_dir() -> Path:
        import os
        base = os.environ.get("HAVERSACK_CACHE_DIR")
        root = Path(base) if base else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "haversack"
        return root / "results"

    def checkpoint_dir() -> Path:
        # Computed here (not imported from the fastsurfer engine) so cache admin does not pull
        # in an engine module; the location is the fixed convention.
        import os
        env = os.environ.get("HAVERSACK_FASTSURFER_CHECKPOINTS")
        if env:
            return Path(env).expanduser()
        base = os.environ.get("XDG_CACHE_HOME")
        return (Path(base) if base else Path.home() / ".cache") / "haversack" / "fastsurfer-checkpoints"

    try:
        wroot = weights_root("ts")
    except Exception:
        wroot = None
    out = [
        {"name": "inputs", "path": default_input_cache(), "sweepable": True},
        {"name": "results", "path": results_dir(), "sweepable": True},
        {"name": "checkpoints", "path": checkpoint_dir(), "sweepable": True},
    ]
    if wroot is not None:
        out.append({"name": "weights", "path": Path(wroot), "sweepable": False})
    return out


def usage() -> list[dict]:
    """:func:`stores`, each with ``items`` and ``bytes`` measured now."""
    rows = []
    for st in stores():
        _, b = _du(st["path"])
        n = len(_entries(st["path"], nested=(st["name"] == "inputs")))
        rows.append({**st, "items": n, "bytes": b, "human": _human(b)})
    return rows


def _entries(path: Path, *, nested: bool = False) -> list[Path]:
    """The removable cache entries under ``path`` (skipping dotfiles). ``nested`` (the input
    cache, ``inputs/<kind>/<hash>``) descends one extra level so an entry is one fetched item,
    not a whole source kind."""
    p = Path(path)
    if not p.is_dir():
        return []
    tops = [c for c in p.iterdir() if not c.name.startswith(".")]
    if not nested:
        return tops
    out = []
    for kind in tops:
        out += [e for e in kind.iterdir() if not e.name.startswith(".")] if kind.is_dir() else [kind]
    return out


def input_entry(spec, cache_dir=None) -> Path | None:
    """The cache directory a remote input spec maps to, or None for a local path."""
    import hashlib

    from .sources import default_input_cache, parse_input
    parsed = parse_input(spec)
    if parsed is None:
        return None
    kind, ident = parsed
    root = Path(cache_dir) if cache_dir else default_input_cache()
    return root / kind / hashlib.sha1(ident.encode()).hexdigest()[:20]


def clean(category: str, *, older_than_days: float | None = None, item: str | None = None,
          dry_run: bool = False) -> dict:
    """Remove transient cache content and report what went (or would go, when ``dry_run``).

    ``category`` is ``inputs``, ``results``, ``checkpoints`` or ``all`` (the three together).
    ``item`` removes one input entry by its spec (only with ``inputs``). ``older_than_days``
    keeps entries touched more recently. Weights are never a category here - use
    ``weights remove``.
    """
    cats = ["inputs", "results", "checkpoints"] if category == "all" else [category]
    valid = {"inputs", "results", "checkpoints"}
    bad = [c for c in cats if c not in valid]
    if bad:
        from .errors import InputError
        raise InputError(f"cannot clean {bad[0]!r}; categories are: inputs, results, checkpoints, all "
                         + ("(weights are removed one at a time: haversack weights remove <id>)"
                            if bad[0] == "weights" else ""))
    by_name = {s["name"]: s for s in stores()}
    removed, freed, cutoff = [], 0, (time.time() - older_than_days * 86400) if older_than_days else None

    def gone(pth: Path):
        nonlocal freed
        _, b = _du(pth)
        freed += b
        removed.append(str(pth))
        if not dry_run:
            if pth.is_dir():
                shutil.rmtree(pth, ignore_errors=True)
            else:
                pth.unlink(missing_ok=True)

    for cat in cats:
        root = Path(by_name[cat]["path"])
        if not root.exists():
            continue
        if item is not None and cat == "inputs":
            entry = input_entry(item)
            if entry and entry.exists():
                gone(entry)
            continue
        for entry in _entries(root, nested=(cat == "inputs")):
            if cutoff is not None and entry.stat().st_mtime >= cutoff:
                continue
            gone(entry)
    return {"removed": removed, "bytes": freed, "human": _human(freed), "dry_run": dry_run}
