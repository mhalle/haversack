"""Hammer one result key on a real object store from several PROCESSES at once.

    AWS_... uv run --no-sync python tools/soak_result_store.py s3://bucket/soak --seconds 45

Why a soak and not a unit test (2026-09-19): the suite drives ``objectcache`` in ONE process
against an in-memory store, and every cache defect this repo has had came from an
interleaving - a reader between a check and a delete, a writer whose work was reclaimed under
it. So: publishers republishing the same key as fast as the store allows, readers reading it,
a sweeper deleting unreferenced blobs with NO grace (the most hostile schedule the design
permits), and one publisher killed mid-flight. Each process has its own local cache, as
separate servers would.

What must hold, and what the readers assert on every hit:

1. **A hit is one publication.** The labels bytes must hash to what that hit's own
   ``result.json`` says, and a preview that is present must name the same publication. A
   reader must never see one publication's labels beside another's metadata.
2. **A miss is allowed; an exception is not.** The sweeper can take a blob a pointer still
   names (documented: it degrades to a miss), and a reader must report that as a miss.
3. **It ends healthy.** After the sweeper stops, one more publication must be readable by a
   host with an EMPTY local cache.

Cleans up its own prefix at the end, whatever happened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

KEY = "so" + "ak" * 31 + "00"


def cache(url: str, root: Path, *, check: bool = False):
    from haversack.objectcache import SharedResultCache, open_store
    from haversack.serve import ResultCache
    store, prefix = open_store(url)
    return SharedResultCache(store, ResultCache(root, keep=50), prefix=prefix, check=check)


def publisher(url: str, seconds: float, marker: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        c = cache(url, d / "local")
        labels, preview = d / "labels", d / "preview.png"
        stats = {"published": 0, "artifacts": 0, "errors": []}
        end = time.time() + seconds
        n = 0
        while time.time() < end:
            n += 1
            body = f"labels {marker} {n} ".encode() + os.urandom(16)
            labels.write_bytes(body)
            preview.write_bytes(f"preview {marker} {n}".encode())
            result = {"marker": marker, "n": n,
                      "sha": hashlib.sha256(body).hexdigest()}
            try:
                gen = c.put(KEY, labels, result, {"task": "soak", "computed": time.time()})
                stats["published"] += 1
                if c.add_artifact(KEY, "preview.png", preview, generation=gen):
                    stats["artifacts"] += 1
            except Exception as e:                # noqa: BLE001 - the point of the soak
                stats["errors"].append(f"{type(e).__name__}: {e}")
            time.sleep(random.uniform(0.05, 0.4))
        return stats


def reader(url: str, seconds: float) -> dict:
    with tempfile.TemporaryDirectory() as d:
        c = cache(url, Path(d) / "local")
        stats = {"hits": 0, "misses": 0, "errors": [], "torn": []}
        end = time.time() + seconds
        while time.time() < end:
            try:
                hit = c.get(KEY)
            except Exception as e:                # noqa: BLE001
                stats["errors"].append(f"{type(e).__name__}: {e}")
                time.sleep(0.1)
                continue
            if hit is None:
                stats["misses"] += 1
            else:
                path, result = hit
                body = Path(path).read_bytes()
                if hashlib.sha256(body).hexdigest() != result.get("sha"):
                    stats["torn"].append(f"labels do not match result.json: {result}")
                png = Path(path).parent / "preview.png"
                if png.exists():
                    want = f"preview {result['marker']} {result['n']}".encode()
                    if png.read_bytes() != want:
                        stats["torn"].append(
                            f"preview {png.read_bytes()!r} beside result {result}")
                stats["hits"] += 1
            time.sleep(random.uniform(0.01, 0.1))
        return stats


def sweeper(url: str, seconds: float, grace: float = 0.0) -> dict:
    """``grace`` 0 is the hostile schedule: a blob is a candidate the instant it exists, so
    the sweep's one documented race (a publication landing between its listing of blobs and
    its reading of pointers) is hit constantly. The shipped default is a day, where a blob
    young enough to be in that race is never a candidate at all."""
    with tempfile.TemporaryDirectory() as d:
        c = cache(url, Path(d) / "local")
        stats = {"sweeps": 0, "deleted": 0, "errors": [], "grace_s": grace}
        end = time.time() + seconds
        while time.time() < end:
            try:
                got = c.sweep(grace_s=grace)
                stats["sweeps"] += 1
                stats["deleted"] += got["deleted_blobs"]
            except Exception as e:                # noqa: BLE001
                stats["errors"].append(f"{type(e).__name__}: {e}")
            time.sleep(0.5)
        return stats


ROLES = {"publisher": publisher, "reader": reader, "sweeper": sweeper}


def child(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role"), ap.add_argument("url")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--marker", default="m")
    ap.add_argument("--grace", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.role == "publisher":
        stats = publisher(a.url, a.seconds, a.marker)
    elif a.role == "sweeper":
        stats = sweeper(a.url, a.seconds, a.grace)
    else:
        stats = reader(a.url, a.seconds)
    print("RESULT " + json.dumps(stats))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--publishers", type=int, default=3)
    ap.add_argument("--readers", type=int, default=3)
    ap.add_argument("--sweep-grace", type=float, default=0.0,
                    help="the sweeper's grace in seconds (0, the default here, is hostile)")
    a = ap.parse_args()

    import obstore

    from haversack.objectcache import open_store
    url = f"{a.url.rstrip('/')}/run-{uuid.uuid4().hex[:8]}"
    store, prefix = open_store(url)
    print(f"soak {prefix} for {a.seconds:.0f}s: {a.publishers} publishers, {a.readers} "
          f"readers, 1 sweeper (grace {a.sweep_grace:g}s), 1 publisher killed mid-flight")
    procs, killed = [], None
    try:
        def spawn(role, *extra):
            return subprocess.Popen(
                [sys.executable, __file__, "--child", role, url,
                 "--seconds", str(a.seconds), *extra],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for i in range(a.publishers):
            procs.append(("publisher", spawn("publisher", "--marker", f"p{i}")))
        for _ in range(a.readers):
            procs.append(("reader", spawn("reader")))
        procs.append(("sweeper", spawn("sweeper", "--grace", str(a.sweep_grace))))
        killed = spawn("publisher", "--marker", "doomed")
        time.sleep(min(8.0, a.seconds / 3))
        killed.send_signal(signal.SIGKILL)        # a publisher that never comes back
        print("killed the doomed publisher")
        bad = 0
        for role, p in procs:
            out, err = p.communicate(timeout=a.seconds + 120)
            line = next((ln for ln in out.splitlines() if ln.startswith("RESULT ")), None)
            stats = json.loads(line[len("RESULT "):]) if line else {"no output": err[-400:]}
            for field in ("errors", "torn"):
                bad += len(stats.get(field, []))
            print(f"  {role}: {json.dumps(stats)}")
        # a fresh host must get a complete, current result once the hostility stops
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            c = cache(url, d / "fresh")
            labels = d / "labels"
            body = b"final " + os.urandom(16)
            labels.write_bytes(body)
            c.put(KEY, labels, {"sha": hashlib.sha256(body).hexdigest(), "marker": "final",
                                "n": 0}, {"task": "soak", "computed": time.time()})
            fresh = cache(url, d / "fresh2").get(KEY)
            ok = fresh is not None and Path(fresh[0]).read_bytes() == body
            print(f"  fresh host after the storm: {'HIT and correct' if ok else 'FAILED'}")
            bad += 0 if ok else 1
        print("SOAK PASSED" if not bad else f"SOAK FAILED: {bad} problem(s)")
        return 0 if not bad else 1
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.kill()
        if killed is not None and killed.poll() is None:
            killed.kill()
        n = 0
        for batch in obstore.list(store, prefix):
            for obj in batch:
                obstore.delete(store, obj["path"])
                n += 1
        print(f"cleaned {n} object(s) under {prefix}")


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.argv.remove("--child")
        sys.exit(child(sys.argv[1:]))
    sys.exit(main())
