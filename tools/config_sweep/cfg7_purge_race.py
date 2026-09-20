"""The purge race against a store whose timestamps are whole seconds (S3, R2).

`delete` purges the bytes of the entry it removed, sparing any blob whose timestamp moved
since it listed them - which is how a publication that DEDUPLICATED onto those bytes keeps
them. On a store that records last-modified to the second, a refresh inside the listing's
own second moves nothing, so the purge waits out the remainder first (`objectcache._settled`,
measured from the store's own timestamps rather than assumed).

This drives that: one process deletes key A in a loop while another republishes key B with
the SAME bytes, so there is exactly one blob and every delete is a chance to take it from B.
B must stay readable on a host that has to fetch it.
"""
import hashlib
import multiprocessing as mp
import sys
import tempfile
import time
import uuid
from pathlib import Path

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


SHARED = b"identical output from two different requests " + b"x" * 4096
A, B = "aa" * 32, "bb" * 32


def cache(url, tag):
    from haversack.objectcache import SharedResultCache, open_store
    from haversack.serve import ResultCache
    store, prefix = open_store(url)
    return SharedResultCache(store, ResultCache(Path(tempfile.mkdtemp(prefix=tag)) / "c"),
                             prefix=prefix, check=False)


def publisher(url, seconds, out):
    c = cache(url, "pub-")
    src = Path(tempfile.mkdtemp()) / "labels"
    src.write_bytes(SHARED)
    n, end = 0, time.time() + seconds
    while time.time() < end:
        n += 1
        c.put(B, src, {"n": n}, {"task": "cfg7", "computed": time.time()})
        time.sleep(0.05)
    out.put(("published B", n))


def deleter(url, seconds, out):
    c = cache(url, "del-")
    src = Path(tempfile.mkdtemp()) / "labels"
    src.write_bytes(SHARED)
    n, waited, end = 0, 0.0, time.time() + seconds
    while time.time() < end:
        c.put(A, src, {"a": n}, {"task": "cfg7", "computed": time.time()})
        t = time.perf_counter()
        c.delete(A)                            # purges A's bytes - which B shares
        waited += time.perf_counter() - t
        n += 1
    out.put(("deleted A", n))
    out.put(("seconds spent in delete", round(waited, 1)))


def reader(url, seconds, out):
    c = cache(url, "read-")
    hits = misses = torn = 0
    end = time.time() + seconds
    while time.time() < end:
        hit = c.get(B)
        if hit is None:
            misses += 1
        else:
            hits += 1
            if Path(hit[0]).read_bytes() != SHARED:
                torn += 1
        time.sleep(0.05)
    out.put(("read B", {"hits": hits, "misses": misses, "torn": torn}))


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    url = store_url("cfg7")
    import obstore

    from haversack.objectcache import open_store
    store, prefix = open_store(url)
    try:
        # what resolution does this store actually date objects to?
        probe = cache(url, "probe-")
        src = Path(tempfile.mkdtemp()) / "l"
        src.write_bytes(SHARED)
        blob = probe.blobs.put_file(src)
        stamp = probe.blobs.entries()[0]["modified"]
        print(f"  store timestamps: {stamp} -> "
              f"{'WHOLE SECONDS (the waiting path)' if stamp == int(stamp) else 'sub-second'}")
        probe.blobs.sweep(keep=set(), allow_empty=True, grace_s=0,
                          candidates=probe.blobs.entries())

        q = mp.Queue()
        procs = [mp.Process(target=f, args=(url, seconds, q))
                 for f in (publisher, deleter, reader, reader)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(seconds + 180)
        results = {}
        while not q.empty():
            k, v = q.get()
            results[k] = v
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")
        cold = cache(url, "cold-")
        final = cold.get(B)
        ok = final is not None and Path(final[0]).read_bytes() == SHARED
        torn = sum(v["torn"] for k, v in results.items() if k == "read B"
                   ) if "read B" in results else 0
        print("  B readable on a cold host at the end:", ok)
        print("CFG7", "PASSED" if (ok and not torn) else "FAILED")
    finally:
        n = 0
        for batch in obstore.list(store, prefix):
            for obj in batch:
                obstore.delete(store, obj["path"])
                n += 1
        print(f"  cleaned {n} object(s)")
