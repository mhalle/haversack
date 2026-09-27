"""Soak a real `haversack serve` on the blob input store (step 6d of docs/cache-consolidation.md).

    uv run --no-sync python tools/soak_input_store.py [--image CT.nii.gz] [--idc UUID ...] \\
        --minutes 15 --budget-gb 0.05 --kill-at 5 --root DIR

One server with HAVERSACK_INPUT_STORE=blobs and an input budget small enough that inputs are
evicted all the time, and client threads that, at random:

- upload a new variant of the image with a job (POST /v1/jobs, a file part);
- store a new variant (PUT /v1/inputs/<digest>) and submit a job naming it by digest;
- submit a job naming an OLDER stored variant by digest - done, or 410 input_gone once it was
  evicted: both are right answers, and which one is counted;
- submit a job on a hosted series (--idc), each with another option so it is another result -
  fetched again whenever eviction took it;
- now and then say `Cache-Control: no-cache`.

At --kill-at minutes the server is SIGKILLed - mid-fetch or mid-job, whatever it was doing -
and restarted on the same directories. What must hold:

1. No 5xx, ever.
2. Every job ends `done` or, for a digest reference, 410 at submit - except jobs that were
   running on the killed server.
3. Every result downloaded hashes to the digest its job reported.
4. After the run: no staging and no view files left (the killed process's are reaped by the
   restarted one), no temporary file of an interrupted blob write once the store is reopened,
   every ref's blobs present, and no unreferenced blob once swept.

Inputs are variants of --image with one voxel changed, so every upload is new content.
Writes report.json and the server logs under --root. Nothing leaves the machine except the
--idc fetches.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx


def variant_bytes(image, i: int) -> bytes:
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(image)
    arr.flat[i % arr.size] += 1 + i // arr.size
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v.nii.gz"
        sitk.WriteImage(out, str(p))
        return p.read_bytes()


class Soak:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(24)
        self.base = f"http://127.0.0.1:{args.port}"
        self.H = {"Authorization": f"Bearer {self.token}"}
        self.lock = threading.Lock()
        self.stats = {"jobs": 0, "done": 0, "input_gone": 0, "failed": [], "lost_on_kill": 0,
                      "5xx": [], "digest_mismatch": [], "server_down_retries": 0,
                      "idc_jobs": 0, "no_cache": 0}
        self.stored: list[str] = []
        self.counter = iter(range(10**9))
        self.proc = None
        self.generation = 0
        self.killed_at = None
        import SimpleITK as sitk
        self.image = sitk.ReadImage(str(args.image))

    # -- the server -----------------------------------------------------------------------------

    def start(self):
        self.generation += 1
        env = {**os.environ, "HAVERSACK_INPUT_STORE": "blobs",
               "HAVERSACK_SERVER_TOKEN": self.token,
               "HAVERSACK_CACHE_DIR": str(self.root / "hcache")}
        log = open(self.root / f"server-{self.generation}.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "haversack.cli", "serve", "--port", str(self.args.port),
             "--device", self.args.device, "--workdir", str(self.root / "work"),
             "--cache-dir", str(self.root / "cache"),
             "--input-cache-gb", str(self.args.budget_gb)],
            stdout=log, stderr=subprocess.STDOUT, env=env)
        for _ in range(120):
            try:
                if httpx.get(f"{self.base}/v1/version", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise SystemExit("the server did not come up; see server log")

    def kill(self):
        self.killed_at = time.time()
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()
        time.sleep(1)
        self.start()

    # -- a client ---------------------------------------------------------------------------------

    def _note_5xx(self, r, what):
        if r.status_code >= 500:
            with self.lock:
                self.stats["5xx"].append(f"{what}: {r.status_code} {r.text[:200]}")

    def _wait(self, jid, submitted_gen):
        while True:
            try:
                r = httpx.get(f"{self.base}/v1/jobs/{jid}", headers=self.H, timeout=30)
            except httpx.HTTPError:
                with self.lock:
                    self.stats["server_down_retries"] += 1
                time.sleep(1)
                continue
            if r.status_code == 404 and self.generation != submitted_gen:
                return None                        # the killed server's job, gone with it
            self._note_5xx(r, f"status {jid}")
            s = r.json()
            if s.get("state") in ("done", "failed", "cancelled"):
                return s
            time.sleep(0.3)

    def _check_result(self, s):
        want = ((s.get("result") or {}).get("outputs") or [{}])[0].get("sha256")
        r = httpx.get(f"{self.base}/v1/jobs/{s['id']}/result", headers=self.H, timeout=120)
        self._note_5xx(r, f"result {s['id']}")
        if r.status_code == 200 and want:
            got = "sha256:" + hashlib.sha256(r.content).hexdigest()
            if got != want:
                with self.lock:
                    self.stats["digest_mismatch"].append((s["id"], want, got))

    def _run(self, data, files=None, headers=None, kind="job"):
        gen = self.generation
        try:
            r = httpx.post(f"{self.base}/v1/jobs", headers={**self.H, **(headers or {})},
                           data=data, files=files, timeout=300)
        except httpx.HTTPError:
            with self.lock:
                self.stats["server_down_retries"] += 1
            return
        self._note_5xx(r, f"submit {kind}")
        if r.status_code == 410:
            with self.lock:
                self.stats["input_gone"] += 1
            return
        if r.status_code != 202:
            with self.lock:
                self.stats["failed"].append(f"submit {kind}: {r.status_code} {r.text[:200]}")
            return
        with self.lock:
            self.stats["jobs"] += 1
        s = self._wait(r.json()["id"], gen)
        if s is None:
            with self.lock:
                self.stats["lost_on_kill"] += 1
            return
        if s["state"] == "done":
            with self.lock:
                self.stats["done"] += 1
            self._check_result(s)
        elif s.get("started") and self.killed_at and gen != self.generation:
            with self.lock:
                self.stats["lost_on_kill"] += 1
        else:
            with self.lock:
                self.stats["failed"].append(f"{kind} {s['id']}: {s['state']} {s.get('error')}")

    def client(self, deadline, rng):
        task = self.args.task
        while time.time() < deadline:
            action = rng.random()
            nc = {"Cache-Control": "no-cache"} if rng.random() < 0.1 else None
            if nc:
                with self.lock:
                    self.stats["no_cache"] += 1
            if action < 0.3:
                raw = variant_bytes(self.image, next(self.counter))
                self._run({"task": task}, files={"file": ("scan.nii.gz", raw)}, headers=nc,
                          kind="upload")
            elif action < 0.55:
                raw = variant_bytes(self.image, next(self.counter))
                d = "sha256:" + hashlib.sha256(raw).hexdigest()
                try:
                    r = httpx.put(f"{self.base}/v1/inputs/{d}", content=raw, headers=self.H,
                                  timeout=300)
                except httpx.HTTPError:
                    continue
                self._note_5xx(r, "put input")
                if r.status_code == 200:
                    with self.lock:
                        self.stored.append(d)
                    self._run({"task": task, "source": json.dumps([{"kind": "input", "sha256": d}])},
                              headers=nc, kind="stored")
            elif action < 0.8 and self.stored:
                with self.lock:
                    d = rng.choice(self.stored)
                self._run({"task": task, "source": json.dumps([{"kind": "input", "sha256": d}])},
                          headers=nc, kind="reference")
            elif self.args.idc:
                uuid = rng.choice(self.args.idc)
                grid = rng.choice(["input", "model", 2.0, 3.0, 4.0])
                interp = rng.choice(["linear", "nearest"])
                with self.lock:
                    self.stats["idc_jobs"] += 1
                self._run({"task": task, "options": json.dumps({"grid": grid, "interp": interp}),
                           "source": json.dumps([{"kind": "idc", "crdc_series_uuid": uuid}])},
                          headers=nc, kind="idc")

    # -- the run ----------------------------------------------------------------------------------

    def run(self):
        self.start()
        t0 = time.time()
        deadline = t0 + 60 * self.args.minutes
        threads = [threading.Thread(target=self.client, args=(deadline, random.Random(i)), daemon=True)
                   for i in range(self.args.workers)]
        [t.start() for t in threads]
        if self.args.kill_at:
            while time.time() < t0 + 60 * self.args.kill_at:
                time.sleep(1)
            print(f"[{time.time() - t0:5.0f}s] SIGKILL the server and restart it", flush=True)
            self.kill()
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with self.lock:
                print(f"[{time.time() - t0:5.0f}s] jobs {self.stats['jobs']} done {self.stats['done']} "
                      f"gone {self.stats['input_gone']} lost {self.stats['lost_on_kill']} "
                      f"failed {len(self.stats['failed'])} 5xx {len(self.stats['5xx'])}", flush=True)
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait(timeout=60)
        return self.after()

    def after(self):
        """What the store looks like once the server is gone."""
        from haversack.inputstore import InputStore
        root = self.root / "work" / "input_store"
        leftovers = {
            "staging_files": [str(p) for p in (root / "staging").rglob("*") if p.is_file()],
            "view_files": [str(p) for p in (root / "views").rglob("*") if p.is_file()],
        }
        store = InputStore(root, None, grace_s=0)   # reaps the killed process's staging, too
        leftovers["staging_after_reopen"] = [str(p) for p in (root / "staging").rglob("*")
                                             if p.is_file()]
        # ... and a blob write the kill interrupted (its provender temporary file)
        leftovers["temp_writes_after_reopen"] = [str(p) for p in
                                                 (root / "store").rglob("*.provender-tmp")]
        refs = store._refs()
        missing = [key for _, key, doc in refs
                   if not all((store.store.root / store.blobs.path(b["digest"])).is_file()
                              for b in doc["files"].values())]
        store.budget = 1 << 62
        store.evict()
        live = {b["digest"] for _, _, d in store._refs() for b in d["files"].values()}
        orphans = [b["digest"] for b in store.blobs.entries() if b["digest"] not in live]
        report = {**self.stats, "refs": len(refs), "refs_missing_blobs": missing,
                  "orphan_blobs_after_sweep": orphans, **leftovers}
        (self.root / "report.json").write_text(json.dumps(report, indent=1))
        ok = (not report["5xx"] and not report["failed"] and not report["digest_mismatch"]
              and not missing and not orphans and not leftovers["staging_after_reopen"]
              and not leftovers["temp_writes_after_reopen"])
        print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()},
                         indent=1))
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", default=os.path.expanduser("~/tmp/data/CT_Abdo.nii.gz"))
    ap.add_argument("--idc", action="append", default=[], help="crdc_series_uuid (repeatable)")
    ap.add_argument("--task", default="ts.v2:total_fast")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--kill-at", type=float, default=5)
    ap.add_argument("--budget-gb", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--root", required=True)
    raise SystemExit(Soak(ap.parse_args()).run())


if __name__ == "__main__":
    main()
