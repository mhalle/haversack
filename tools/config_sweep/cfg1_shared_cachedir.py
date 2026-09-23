"""Two servers sharing ONE --cache-dir, against one store.

The deployment CLAUDE.md describes for a multi-GPU box: a server per GPU, each with its own
port, and nothing stopping them from defaulting to the same ~/.cache/haversack/results.
`_fill_lock` is per process, so everything that keeps two fills apart here is on disk.
"""
import hashlib
import multiprocessing as mp
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

KEYS = [f"{i:02x}" * 32 for i in range(6)]


def store_url(suffix: str) -> str:
    """The store to run against: $HAVERSACK_TEST_STORE, plus a fresh prefix per run.

    Named by the environment rather than written down here: this script goes to a public
    repository, and where one person's bucket lives is neither a secret nor anyone else's
    default. Everything is written under the returned prefix and deleted afterwards.
    """
    import os
    import sys
    import uuid
    base = os.environ.get("HAVERSACK_TEST_STORE")
    if not base:
        sys.exit("set HAVERSACK_TEST_STORE, e.g. s3://your-bucket/haversack-sweep "
                 "(with AWS_* credentials in the environment)")
    return f"{base.rstrip('/')}/{suffix}/run-{uuid.uuid4().hex[:8]}"


def cache(url, cache_dir):
    from haversack.objectcache import SharedResultCache, open_store
    from haversack.serve import ResultCache
    store, prefix = open_store(url)
    return SharedResultCache(store, ResultCache(cache_dir, keep=200), prefix=prefix,
                             check=False)


def worker(role, url, cache_dir, seconds, out):
    c = cache(url, Path(cache_dir))
    src = Path(tempfile.mkdtemp()) / "labels"
    stats = {"role": role, "published": 0, "hits": 0, "misses": 0, "torn": [], "errors": []}
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        n += 1
        key = KEYS[n % len(KEYS)]
        try:
            if role == "publisher":
                body = f"{key[:4]}-{n}".encode() + os.urandom(4096)
                src.write_bytes(body)
                c.put(key, src, {"sha": hashlib.sha256(body).hexdigest()},
                      {"task": "cfg1", "computed": time.time()})
                stats["published"] += 1
            else:
                hit = c.get(key)
                if hit is None:
                    stats["misses"] += 1
                else:
                    path, result = hit
                    body = Path(path).read_bytes()
                    if hashlib.sha256(body).hexdigest() != result.get("sha"):
                        stats["torn"].append(key)
                    stats["hits"] += 1
        except Exception as e:                 # noqa: BLE001
            stats["errors"].append(f"{type(e).__name__}: {e}")
        time.sleep(0.02)
    out.put(stats)


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    url = store_url("cfg1")
    shared = tempfile.mkdtemp(prefix="cfg1-shared-")
    print(f"one cache dir: {shared}")
    from provender import ops

    from haversack.objectcache import open_store
    store, prefix = open_store(url)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, url, shared, seconds, q))
             for r in ("publisher", "publisher", "reader", "reader")]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(seconds + 120)
        bad = 0
        while not q.empty():
            st = q.get()
            bad += len(st["torn"]) + len(st["errors"])
            print(" ", st)
        # the shared directory must still be coherent: every entry resolves, and every
        # generation directory that is current holds a complete result
        c = cache(url, Path(shared))
        broken = []
        for key in KEYS:
            gen = c.local.generation(key)
            if gen is None:
                continue
            where = c.local._generation_dir(key, gen)
            for name in ("labels.seg.nrrd", "result.json", "meta.json"):
                if not (where / name).exists():
                    broken.append(f"{key[:8]} current generation missing {name}")
            hit = c.local.get(key)
            if hit is None:
                broken.append(f"{key[:8]} current generation does not resolve")
        leftovers = sorted(p.name for p in Path(shared).glob(".*"))
        print("  incoherent entries:", broken or "none")
        print("  dotfiles left in the cache root:", leftovers or "none")
        bad += len(broken)
        print("CFG1", "PASSED" if not bad else f"FAILED ({bad})")
    finally:
        n = 0
        for batch in ops.list(store, prefix):
            for obj in batch:
                ops.delete(store, obj["path"])
                n += 1
        print(f"cleaned {n} object(s)")
