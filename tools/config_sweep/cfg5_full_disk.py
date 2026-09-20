"""A cache directory that runs out of space, on a real filesystem."""
import hashlib
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import obstore

from haversack.objectcache import SharedResultCache, open_store
from haversack.serve import ResultCache

KEY = "ab" * 32
root = Path(sys.argv[1]) / "cache"
root.mkdir(parents=True, exist_ok=True)
url = store_url("cfg5")
store, prefix = open_store(url)


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


def host(dirname):
    return SharedResultCache(store, ResultCache(Path(dirname)), prefix=prefix, check=False)


try:
    publisher = host(tempfile.mkdtemp(prefix="cfg5-pub-"))
    src = Path(tempfile.mkdtemp()) / "labels"
    body = os.urandom(6 << 20)                  # 6 MB: two of these will not fit in 12 MB
    src.write_bytes(body)
    publisher.put(KEY, src, {"sha": hashlib.sha256(body).hexdigest()},
                  {"task": "cfg5", "computed": time.time()})
    print("  published 6 MB to the store")

    starved = host(root)
    # fill the little disk almost completely, so the fill cannot land
    ballast = root.parent / "ballast"
    with open(ballast, "wb") as f:
        try:
            f.write(os.urandom(11 << 20))
        except OSError:
            pass
    free = os.statvfs(root)
    print(f"  free space on the cache disk: {free.f_bavail * free.f_frsize >> 10} KB")

    try:
        hit = starved.get(KEY)
        print("  get on the full disk:", "MISS (correct)" if hit is None else "served")
        ok_read = hit is None
    except Exception as e:                      # noqa: BLE001
        print(f"  get on the full disk RAISED {type(e).__name__}: {e}")
        ok_read = False

    # and a PUBLICATION from the starved host: the store must still get it
    key2 = "cd" * 32
    try:
        starved.put(key2, src, {"sha": hashlib.sha256(body).hexdigest()},
                    {"task": "cfg5", "computed": time.time()})
        published = host(tempfile.mkdtemp(prefix="cfg5-check-")).get(key2) is not None
        print("  publication from the starved host reached the store:", published)
    except Exception as e:                      # noqa: BLE001
        published = False
        print(f"  publication RAISED {type(e).__name__}: {e}")

    ballast.unlink(missing_ok=True)
    healed = starved.get(KEY)
    ok_heal = healed is not None and Path(healed[0]).read_bytes() == body
    print("  once there is room again:", "HIT and correct" if ok_heal else "FAILED")
    print("CFG5", "PASSED" if (ok_read and published and ok_heal) else "FAILED")
finally:
    n = 0
    for batch in obstore.list(store, prefix):
        for obj in batch:
            obstore.delete(store, obj["path"])
            n += 1
    print(f"  cleaned {n} object(s)")
