"""Check that an object store can carry the shared result cache, then round-trip one result.

    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
    AWS_ENDPOINT=https://s3.us-east-1.wasabisys.com AWS_REGION=us-east-1 \\
    uv run --no-sync python tools/probe_result_store.py s3://BUCKET/haversack-probe

Written for the first real-bucket test (2026-09-19, Wasabi): the suite runs only against
obstore's in-memory store, and whether an S3-compatible service honors If-Match and
If-None-Match on PUT is exactly what its marketing does not say. Everything goes under a
fresh ``<prefix>/run-<id>/`` and is deleted at the end, whatever happened.
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path


def main(url: str) -> int:
    import obstore

    from haversack.objectcache import (SharedResultCache, check_conditional_writes,
                                       open_store)
    from haversack.serve import ResultCache

    store, prefix = open_store(url)
    prefix = f"{prefix}run-{uuid.uuid4().hex[:8]}/"
    print(f"store {type(store).__name__}, prefix {prefix}")
    try:
        check_conditional_writes(store, prefix)
        print("ok   conditional writes: create-if-absent and replace-if-unchanged honored")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a = SharedResultCache(store, ResultCache(d / "a"), prefix=prefix, check=False)
            b = SharedResultCache(store, ResultCache(d / "b"), prefix=prefix, check=False)
            key = uuid.uuid4().hex * 2
            src = d / "labels"
            src.write_bytes(b"probe labels " + key.encode())
            gen = a.put(key, src, {"probe": True}, {"task": "probe", "computed": 0})
            hit = b.get(key)
            assert hit is not None and Path(hit[0]).read_bytes() == src.read_bytes(), hit
            assert b.local.generation(key) == gen
            print("ok   a result published by one host is served by another")
            png = d / "p.png"
            png.write_bytes(b"png")
            assert a.add_artifact(key, "preview.png", png, generation=gen)
            assert not a.add_artifact(key, "preview.png", png, generation="stale")
            print("ok   artifacts: current generation accepted, stale refused")
            a.put(key, src, {"probe": 2}, {"task": "probe", "computed": 1})
            assert b.get(key)[1] == {"probe": 2}
            print("ok   a republication replaces the other host's copy")
            rows, position = b.list()          # main's listing contract, 2026-09-21
            assert [e["key"] for e in rows] == [key], rows
            assert position is None, position
            page, _ = b.list(keys=[key])       # computed names: no bucket listing
            assert [e["key"] for e in page] == [key], page
            print("ok   list, whole and by computed key")
        return 0
    finally:
        n = 0
        for batch in obstore.list(store, prefix):
            for obj in batch:
                obstore.delete(store, obj["path"])
                n += 1
        print(f"cleaned {n} object(s) under {prefix}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
