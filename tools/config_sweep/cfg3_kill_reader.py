"""A READER killed mid-fill. The soak only ever killed publishers.

A fill downloads into a work directory, then publishes the generation locally. SIGKILL in
the middle leaves that work behind and may leave a generation directory without a pointer -
the state the adopt path was written for. What must hold: the next read on that host
succeeds, no partial result is ever served, and the leavings are reclaimed.
"""
import hashlib
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

KEY = "ab" * 32
SIZE = 24 << 20                                 # a feldglas-sized result: seconds to fetch


def cache(url, cache_dir):
    from haversack.objectcache import SharedResultCache, open_store
    from haversack.serve import ResultCache
    store, prefix = open_store(url)
    return SharedResultCache(store, ResultCache(Path(cache_dir)), prefix=prefix, check=False)


def doomed_reader(url, cache_dir):
    c = cache(url, cache_dir)
    c.get(KEY)                                  # killed somewhere in here


if __name__ == "__main__":
    url = f"s3://haversack-backing/cfg3/run-{uuid.uuid4().hex[:8]}"
    shared = tempfile.mkdtemp(prefix="cfg3-")
    import obstore

    from haversack.objectcache import open_store
    store, prefix = open_store(url)
    try:
        pub = cache(url, tempfile.mkdtemp(prefix="cfg3-pub-"))
        src = Path(tempfile.mkdtemp()) / "labels"
        body = os.urandom(SIZE)
        src.write_bytes(body)
        pub.put(KEY, src, {"sha": hashlib.sha256(body).hexdigest()},
                {"task": "cfg3", "computed": time.time()})
        print(f"published {SIZE >> 20} MB")

        killed = 0
        for attempt, delay in enumerate((0.6, 1.2, 2.0), start=1):
            p = mp.Process(target=doomed_reader, args=(url, shared))
            p.start()
            time.sleep(delay)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)
                killed += 1
            p.join(30)
            leftovers = sorted(x.name for x in Path(shared).glob(".fill-*"))
            entry = Path(shared) / KEY
            gens = sorted(x.name for x in entry.glob("g-*")) if entry.exists() else []
            print(f"  kill at {delay}s: work dirs {len(leftovers)}, generations {len(gens)},"
                  f" pointer {(entry / 'current').exists() if entry.exists() else False}")
        print(f"killed {killed} of 3 readers mid-fill")

        c = cache(url, shared)
        hit = c.get(KEY)
        ok = hit is not None and Path(hit[0]).read_bytes() == body
        print("  the next read on that host:", "HIT and correct" if ok else "FAILED")
        # a second read must not re-download: the copy is whole now
        again = c.get(KEY)
        ok2 = again is not None and Path(again[0]).read_bytes() == body
        left = sorted(x.name for x in Path(shared).glob(".fill-*"))
        print("  work directories left after two good reads:", left or "none")
        print("CFG3", "PASSED" if (ok and ok2 and not left) else "FAILED")
    finally:
        n = 0
        for batch in obstore.list(store, prefix):
            for obj in batch:
                obstore.delete(store, obj["path"])
                n += 1
        print(f"cleaned {n} object(s)")
