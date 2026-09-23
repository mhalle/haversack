"""Encoder weights: pinned files fetched once, verified by digest, kept under one root.

``<root>/<family>/<revision>/<file>`` - the root is ``$HAVERSACK_ENCODER_WEIGHTS`` or
``~/.haversack/encoders`` (not under the result or input caches: ``cache clean`` must never throw
away 1.5 GB of weights). A file is installed when it is there with a sidecar recording the size,
mtime and sha256 it was verified at, so a later check need not hash it again; a file that changed
since is re-hashed, and refused if its digest is not the pinned one.

A download streams to a temporary file beside the target, hashed on the way, with a byte cap of the
pinned size (a server cannot fill the disk), and is renamed into place only once its digest matches
- a half or wrong file never sits under the real name. ``adopt`` installs a copy the user already
has (``weights fetch --from``), verified the same way.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..errors import InputError
from .registry import EncoderSpec, WeightsFile

ROOT_ENV = "HAVERSACK_ENCODER_WEIGHTS"


def root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or Path.home() / ".haversack" / "encoders").expanduser()


def directory(spec: EncoderSpec) -> Path:
    return root() / spec.family_name.replace(".", "_") / spec.revision


def path(spec: EncoderSpec, wf: WeightsFile) -> Path:
    return directory(spec) / wf.name


def _sidecar(p: Path) -> Path:
    return p.with_name(p.name + ".verified.json")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified(spec: EncoderSpec, wf: WeightsFile) -> bool:
    """Is ``wf`` installed and the pinned bytes? Cheap when its sidecar still describes it."""
    p = path(spec, wf)
    if not p.is_file():
        return False
    st = p.stat()
    try:
        side = json.loads(_sidecar(p).read_text(encoding="utf-8"))
        if (side.get("sha256"), side.get("size"), side.get("mtime_ns")) == (wf.sha256, st.st_size, st.st_mtime_ns):
            return True
    except (OSError, ValueError):
        pass
    if st.st_size != wf.size or _sha256(p) != wf.sha256:
        return False
    _sidecar(p).write_text(json.dumps({"sha256": wf.sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}), encoding="utf-8")
    return True


def installed(spec: EncoderSpec) -> bool:
    return all(verified(spec, wf) for wf in spec.weights)


def _place(tmp: Path, spec: EncoderSpec, wf: WeightsFile, digest: str, n: int) -> Path:
    if n != wf.size or digest != wf.sha256:
        tmp.unlink(missing_ok=True)
        raise InputError(f"{wf.name}: {n} bytes with sha256 {digest[:16]}..., not the pinned {wf.size} bytes / "
                         f"{wf.sha256[:16]}... - refused")
    target = path(spec, wf)
    os.replace(tmp, target)
    st = target.stat()
    _sidecar(target).write_text(json.dumps({"sha256": wf.sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}), encoding="utf-8")
    return target


def fetch(spec: EncoderSpec, progress=None) -> list[Path]:
    """Download every weights file ``spec`` pins that is not installed yet."""
    from .. import fetchlib
    out = []
    for wf in spec.weights:
        if verified(spec, wf):
            out.append(path(spec, wf)); continue
        d = directory(spec); d.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"{spec.name}: fetching {wf.name} ({wf.size / 1e9:.2f} GB) from {wf.source or wf.url}")
        fd, tmp = tempfile.mkstemp(prefix=wf.name + ".", suffix=".partial", dir=d)
        tmp = Path(tmp)
        h, n = hashlib.sha256(), 0
        try:
            with fetchlib.open(wf.url, timeout=120) as r, os.fdopen(fd, "wb") as out_fh:
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > wf.size:
                        raise InputError(f"{wf.url}: more than the pinned {wf.size} bytes - refused")
                    h.update(chunk); out_fh.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        out.append(_place(tmp, spec, wf, h.hexdigest(), n))
    return out


def adopt(spec: EncoderSpec, source, progress=None) -> list[Path]:
    """Install a copy the user already has: ``source`` is a file (for a one-file encoder) or a
    directory holding the files by name. Verified against the pins like a download."""
    source = Path(source).expanduser()
    out = []
    for wf in spec.weights:
        src = source if source.is_file() and len(spec.weights) == 1 else source / wf.name
        if not src.is_file():
            raise InputError(f"{src}: no {wf.name} to adopt")
        d = directory(spec); d.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"{spec.name}: verifying and copying {src}")
        fd, tmp = tempfile.mkstemp(prefix=wf.name + ".", suffix=".partial", dir=d)
        os.close(fd)
        tmp = Path(tmp)
        shutil.copyfile(src, tmp)
        out.append(_place(tmp, spec, wf, _sha256(tmp), tmp.stat().st_size))
    return out


def remove(spec: EncoderSpec) -> list[Path]:
    """Delete ``spec``'s installed weights; returns what was removed."""
    gone = []
    for wf in spec.weights:
        p = path(spec, wf)
        for q in (p, _sidecar(p)):
            if q.exists():
                q.unlink(); gone.append(q)
    return gone
