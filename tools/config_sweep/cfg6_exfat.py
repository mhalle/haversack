"""A cache root on exFAT: no hard links, case-INSENSITIVE, no POSIX permissions.

AGENTS.md records that a cache on a USB stick is an ordinary way to run this, and that
FAT32/exFAT answer ENOTSUP to os.link. The whole shared-store protocol has only ever run on
APFS; this is the first time the local tier of it runs where hard links do not exist.
"""
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


root = Path(sys.argv[1]) / "cache"
root.mkdir(parents=True, exist_ok=True)

# the filesystem's own facts, measured rather than assumed
a, b = root / "LinkProbe", root / "linkprobe"
a.write_bytes(b"x")
links = True
try:
    os.link(a, root / "LinkProbe2")
except OSError as e:
    links = False
    print(f"  os.link: {e.strerror}")
case_sensitive = not b.exists()
print(f"  hard links: {links}; case-sensitive: {case_sensitive}")
for p in (a, b, root / "LinkProbe2"):
    p.unlink(missing_ok=True)

url = store_url("cfg6")
store, prefix = open_store(url)
KEY = "ab" * 32
checks = []
try:
    here = SharedResultCache(store, ResultCache(root), prefix=prefix, check=False)
    elsewhere = SharedResultCache(store, ResultCache(Path(tempfile.mkdtemp())),
                                  prefix=prefix, check=False)
    src = Path(tempfile.mkdtemp()) / "labels"
    body = os.urandom(512 << 10)
    src.write_bytes(body)

    gen = here.put(KEY, src, {"sha": hashlib.sha256(body).hexdigest()},
                   {"task": "cfg6", "computed": time.time()})
    hit = here.get(KEY)
    checks.append(("publish and read on exFAT",
                   hit is not None and Path(hit[0]).read_bytes() == body))
    checks.append(("the local copy kept the store's generation",
                   here.local.generation(KEY) == gen))

    pulled = elsewhere.pull()
    checks.append(("another host pulls it", pulled["pulled"] == 1))

    body2 = os.urandom(512 << 10)
    src.write_bytes(body2)
    elsewhere.put(KEY, src, {"sha": hashlib.sha256(body2).hexdigest()},
                  {"task": "cfg6", "computed": time.time()})
    hit = here.get(KEY)
    checks.append(("a republication elsewhere replaces the exFAT copy",
                   hit is not None and Path(hit[0]).read_bytes() == body2))

    # push from the exFAT host: a legacy-free cache, but the local read path is exercised
    pushed = here.push()
    checks.append(("push from exFAT reports honestly",
                   pushed["failed"] == 0 and pushed["unreadable"] == 0))

    checks.append(("history is listed", len(here.history(KEY)) == 2))
    swept = here.sweep(grace_s=0)
    checks.append(("sweep runs", swept["unreadable_pointers"] == 0))
    here.delete(KEY)
    checks.append(("delete removes it", here.get(KEY) is None))
    leftovers = sorted(p.name for p in root.glob(".*"))
    checks.append((f"no leavings in the cache root ({leftovers})", not leftovers))
finally:
    n = 0
    for batch in obstore.list(store, prefix):
        for obj in batch:
            obstore.delete(store, obj["path"])
            n += 1
    print(f"  cleaned {n} object(s)")
for name, ok in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
print("CFG6", "PASSED" if all(ok for _, ok in checks) else "FAILED")
