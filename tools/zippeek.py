"""Read a remote zip's layout without downloading it - the shared half of the
manifest generators.

A weights manifest has to state the folder a zip installs to *before* anything is
downloaded, and the folder's name is inside the archive. A zip keeps its index in
a trailing central directory, so two Range requests fetch the whole listing and a
third fetches any small member: describing a 1.1 GB asset costs a few megabytes.

Extracted from tools/gen_mrsegmentator_manifest.py (2026-09-05) when the second
and third catalog needed the same reader. Stdlib only, like its callers - these
run under `uv run --no-project python tools/<generator>.py`.
"""
import json
import re
import struct
import urllib.request
import zlib


def content_length(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def central_directory(url: str) -> dict:
    """``{member name: (method, compressed size, local header offset)}`` via two
    Range reads. Handles zip64, which every multi-gigabyte weights asset needs."""
    total = content_length(url)
    tail = fetch_range(url, max(0, total - 66000), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    _, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF:                                  # zip64
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = fetch_range(url, cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, off):
            q = 0
            while q < len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (hsz // 8),
                                              extra[q + 4:q + 4 + hsz - hsz % 8]))
                    if usize == 0xFFFFFFFF:
                        usize = vals.pop(0)
                    if csize == 0xFFFFFFFF:
                        csize = vals.pop(0)
                    if off == 0xFFFFFFFF:
                        off = vals.pop(0)
                q += 4 + hsz
        out[name] = (method, csize, off)
        p += 46 + nlen + elen + clen
    return out


def member_head(url: str, cd: dict, name: str, limit: int | None = None) -> bytes:
    """The first ``limit`` bytes of one member (all of it when limit is None)."""
    method, csize, off = cd[name]
    lh = fetch_range(url, off, off + 29)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    start = off + 30 + nlen + elen
    data = fetch_range(url, start, start + (csize if limit is None else min(csize, limit)) - 1)
    if method == 8:
        return zlib.decompressobj(-15).decompress(data)
    return data


def read_json(url: str, cd: dict, name: str) -> dict:
    return json.loads(member_head(url, cd, name))


def is_junk(name: str) -> bool:
    """macOS Finder's zip litter, which is not part of any model."""
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX/") or base == ".DS_Store" or base.startswith("._")


def config_folders(cd: dict) -> list:
    """Every ``<trainer>__<plans>__<config>`` folder in the archive, as the path
    prefix it sits at ('' when it is the archive root itself)."""
    out = set()
    for name in cd:
        if is_junk(name):
            continue
        parts = name.split("/")
        for i, seg in enumerate(parts[:-1]):
            if seg.count("__") == 2:
                out.add("/".join(parts[:i + 1]))
    return sorted(out)


def folds(cd: dict) -> list:
    return sorted({m.group(1) for n in cd if not is_junk(n)
                   for m in [re.search(r"(fold_\w+)/checkpoint_\w+\.pth$", n)] if m})


def checkpoints(cd: dict) -> list:
    return sorted({n.rsplit("/", 1)[-1] for n in cd
                   if not is_junk(n) and re.search(r"fold_\w+/checkpoint_\w+\.pth$", n)})
