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


def cache_root() -> Path:
    """The one cache root: ``HAVERSACK_CACHE_DIR``, else ``$XDG_CACHE_HOME/haversack``
    (``~/.cache/haversack``). Every store below hangs off it - the server's results too."""
    import os
    base = os.environ.get("HAVERSACK_CACHE_DIR")
    if base:
        return Path(base).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "haversack"


def check_cache_root() -> Path:
    """The cache root, refused as an input error when something that is not a directory
    sits at it (a `HAVERSACK_CACHE_DIR` pointing at a file listed four empty stores)."""
    from .errors import InputError
    root = cache_root()
    if root.exists() and not root.is_dir():
        raise InputError(f"cache root {root} is not a directory (HAVERSACK_CACHE_DIR)")
    return root


def trainer_shim_dir() -> Path:
    """Where generated nnU-Net trainer shims go: ``HAVERSACK_TRAINER_SHIMS``, else under
    the cache root."""
    import os
    env = os.environ.get("HAVERSACK_TRAINER_SHIMS")
    return Path(env).expanduser() if env else cache_root() / "trainer_shims"


def results_dir() -> Path:
    """The server's durable result cache."""
    return cache_root() / "results"


def stores() -> list[dict]:
    """Every store haversack keeps on disk: name, path, whether `cache clean` may sweep it."""
    from .sources import default_input_cache
    from .tasks import weights_root

    def checkpoint_dir() -> Path:
        # Computed here (not imported from the fastsurfer engine) so cache admin does not pull
        # in an engine module; the location is the fixed convention.
        import os
        env = os.environ.get("HAVERSACK_FASTSURFER_CHECKPOINTS")
        if env:
            return Path(env).expanduser()
        return cache_root() / "fastsurfer-checkpoints"

    try:
        wroot = weights_root("ts")
    except Exception:
        wroot = None
    out = [
        {"name": "inputs", "path": default_input_cache(), "sweepable": True},
        {"name": "results", "path": results_dir(), "sweepable": True},
        {"name": "checkpoints", "path": checkpoint_dir(), "sweepable": True},
        # a running server's generated token lives here; never swept, never a credential
        # that `cache clean` could take away from under a server
        {"name": "trainer_shims", "path": trainer_shim_dir(), "sweepable": True},
        {"name": "serve", "path": cache_root() / "serve", "sweepable": False},
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
    valid = {"inputs", "results", "checkpoints", "trainer_shims"}
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


# -- the local server's generated token ------------------------------------------------
#
# `haversack serve` without --token generates one and writes it here, readable by this
# user only; `haversack remote` on the same machine reads it back when the server it is
# given is on this host and no token was passed. The Jupyter pattern: personal use needs
# no ceremony, and a proxy or tunnel in front of the server exposes something that still
# demands a token nobody outside holds. Keyed by port, since that is what a client knows.

def serve_token_path(port: int) -> Path:
    """Where the server on ``port`` leaves its generated token (0600, this user only)."""
    return cache_root() / "serve" / f"{int(port)}.token"


def write_serve_token(path: Path, token: str, *, host: str, port: int) -> None:
    """Write the token file: JSON with the owning process, created exclusively (no
    following of a pre-planted symlink), readable by this user only, and unlinked at
    exit only if it is still ours."""
    import atexit
    import json
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"token": token, "pid": os.getpid(), "host": host, "port": int(port),
                   "started": time.time()}, f)
        f.write("\n")

    def _unlink_if_mine(p=path, pid=os.getpid()):
        try:
            if json.loads(p.read_text()).get("pid") == pid:
                p.unlink()
        except (OSError, ValueError):
            pass
    atexit.register(_unlink_if_mine)


def _is_loopback(host: str) -> bool:
    """A loopback ADDRESS, or the name localhost - never a string prefix: an attacker's
    ``127.0.0.1.evil.example`` must not read as local."""
    import ipaddress
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _alive(pid) -> bool:
    import os
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def server_address(server_url: str) -> tuple[str, int]:
    """``(host, port)`` of a server URL, the port defaulted by scheme - the one place
    the client and its error messages take it from."""
    from urllib.parse import urlsplit
    u = urlsplit(server_url if "://" in server_url else f"http://{server_url}")
    return (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80)


def local_token_for(server_url: str) -> str | None:
    """The generated token of a server on THIS machine, or None.

    Only for a URL whose host is a loopback address and whose port has a token file
    written by a server process that is still alive - a file left by a crashed server
    is ignored, so the token is never handed to whatever answers on that port next
    (a tunnel, another user's process). Anything else - another host, an unknown
    port - gets nothing and the caller passes a token itself."""
    import json
    host, port = server_address(server_url)
    if not _is_loopback(host):
        return None
    try:
        info = json.loads(serve_token_path(port).read_text())
    except (OSError, ValueError):
        return None
    tok = info.get("token") if isinstance(info, dict) else None
    if not tok or not _alive(info.get("pid")):
        return None
    # the server wrote where it listens: a token for a server bound to some other
    # interface is not the token of whatever answers on the loopback port
    bound = str(info.get("host") or "").lower()
    if not (_is_loopback(bound) or bound in ("0.0.0.0", "::", "")):
        return None
    return str(tok)
