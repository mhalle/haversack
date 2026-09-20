"""Two haversack versions against ONE bucket.

The newer writes a pointer format the older cannot read. The rules under test:
  - the old host reads it as a miss and must NOT publish over it;
  - the new host keeps working;
  - the old host's sweep must delete nothing (it cannot account for those blobs);
  - the old host's delete must not claim it purged bytes it could not see.
The "newer version" is this code with POINTER_FORMAT bumped in its own process.
"""
import hashlib
import multiprocessing as mp
import sys
import tempfile
import time
import uuid
from pathlib import Path

NEW_KEY = "ab" * 32
OLD_KEY = "cd" * 32


def cache(url, tag, *, newer=False):
    from haversack import objectcache
    from haversack.objectcache import SharedResultCache, open_store
    from haversack.serve import ResultCache
    if newer:
        objectcache.POINTER_FORMAT += 1        # a haversack from the future
    store, prefix = open_store(url)
    return SharedResultCache(store, ResultCache(Path(tempfile.mkdtemp(prefix=tag)) / "c"),
                             prefix=prefix, check=False)


def newer_host(url, out):
    c = cache(url, "new-", newer=True)
    src = Path(tempfile.mkdtemp()) / "labels"
    src.write_bytes(b"written by the newer version")
    c.put(NEW_KEY, src, {"by": "newer"}, {"task": "cfg2", "computed": time.time()})
    hit = c.get(NEW_KEY)
    out.put(("newer publishes and reads its own", hit is not None
             and Path(hit[0]).read_bytes() == b"written by the newer version"))
    time.sleep(6)                              # while the old host does its worst
    hit = c.get(NEW_KEY)
    out.put(("newer still reads its own afterwards", hit is not None
             and Path(hit[0]).read_bytes() == b"written by the newer version"))


def older_host(url, out):
    time.sleep(2)
    c = cache(url, "old-")
    src = Path(tempfile.mkdtemp()) / "labels"
    src.write_bytes(b"written by the older version")
    out.put(("older reads the newer entry as a miss", c.get(NEW_KEY) is None))
    try:
        c.put(NEW_KEY, src, {"by": "older"}, {"task": "cfg2", "computed": time.time()})
        out.put(("older REFUSES to publish over it", False))
    except Exception as e:                     # noqa: BLE001
        out.put(("older REFUSES to publish over it", "newer haversack" in str(e)))
    # its own key still works
    c.put(OLD_KEY, src, {"by": "older"}, {"task": "cfg2", "computed": time.time()})
    out.put(("older still publishes its own keys", c.get(OLD_KEY) is not None))
    swept = c.sweep(grace_s=0)
    out.put(("older's sweep deletes nothing it cannot account for",
             swept["deleted_blobs"] == 0 and swept["unreadable_pointers"] == 1))
    seen = []
    c.delete(NEW_KEY, report=seen.append)
    out.put(("older's delete does not claim a purge it could not do",
             seen and seen[0]["purged"] is False))


if __name__ == "__main__":
    url = f"s3://haversack-backing/cfg2/run-{uuid.uuid4().hex[:8]}"
    import obstore

    from haversack.objectcache import open_store
    store, prefix = open_store(url)
    q = mp.Queue()
    procs = [mp.Process(target=newer_host, args=(url, q)),
             mp.Process(target=older_host, args=(url, q))]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join(180)
        results = []
        while not q.empty():
            results.append(q.get())
        for name, ok in sorted(results):
            print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        print("CFG2", "PASSED" if all(ok for _, ok in results) else "FAILED",
              f"({len(results)} checks)")
    finally:
        n = 0
        for batch in obstore.list(store, prefix):
            for obj in batch:
                obstore.delete(store, obj["path"])
                n += 1
        print(f"cleaned {n} object(s)")
