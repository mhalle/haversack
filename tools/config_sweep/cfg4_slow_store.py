"""A store that is SLOW rather than broken.

The failure everybody designs for is an outage; the one that actually happens is latency.
What must hold: a slow store must not stall requests that never touch it, because the
blocking calls are handed to the threadpool - that is the whole point of `_offload`.
"""
import sys
import tempfile
from pathlib import Path
import threading
import time
from unittest import mock

# the suite's fakes stand in for a segmenter: no weights, no GPU, no bucket
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import obstore
from fastapi.testclient import TestClient
from obstore.store import MemoryStore

from haversack.serve import LocalExecutor, create_app
from test_job_result_cache import _Segmenter
from test_serve import submit, wait_state

LATENCY = 1.5
tmp = Path(tempfile.mkdtemp(prefix="cfg4-"))
store = MemoryStore()
ex = LocalExecutor(_Segmenter(steps=1), workdir=tmp / "w", cache_dir=tmp / "c",
                   result_store=store)
client = TestClient(create_app(ex))
try:
    jid = submit(client)
    s = wait_state(client, jid, ("done",))
    key = s["key"]
    print("published; the store now answers every read after "
          f"{LATENCY}s")

    real_get = obstore.get

    def slow(st, path, *a, **kw):
        if "results/" in str(path) or "blobs/" in str(path):
            time.sleep(LATENCY)
        return real_get(st, path, *a, **kw)

    timings = {}

    def hit_the_store():
        t = time.perf_counter()
        r = client.get(f"/v1/jobs/{jid}/result")
        timings["store route"] = (time.perf_counter() - t, r.status_code)

    def touch_nothing():
        time.sleep(0.2)                        # while the slow one is in flight
        t = time.perf_counter()
        r = client.get("/v1/health")
        timings["health"] = (time.perf_counter() - t, r.status_code)
        t = time.perf_counter()
        r = client.get("/v1/tasks")
        timings["tasks"] = (time.perf_counter() - t, r.status_code)

    with mock.patch.object(obstore, "get", slow):
        threads = [threading.Thread(target=hit_the_store),
                   threading.Thread(target=touch_nothing)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(60)

    for name, (secs, code) in sorted(timings.items()):
        print(f"  {name:12s} {secs:5.2f}s  HTTP {code}")
    stalled = [n for n, (secs, _) in timings.items()
               if n != "store route" and secs > LATENCY * 0.6]
    ok = (not stalled
          and timings["store route"][1] == 200
          and all(code == 200 for _, code in timings.values()))
    print("  routes stalled behind the slow store:", stalled or "none")
    print("CFG4", "PASSED" if ok else "FAILED")
finally:
    ex.close()
