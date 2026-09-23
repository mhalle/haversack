"""Soak real `haversack serve` processes on ONE directory result store (step 5).

    uv run --no-sync python tools/soak_server_store.py IMAGE.nii.gz [...] --minutes 20 \\
        [--sync-url s3://bucket/prefix]

Step 5 of docs/cache-consolidation.md: the store protocol on provender's DiskStore
(`--result-store file:///path`), run as it would be - real server processes, a real engine,
real HTTP - with everything else the design adds running around it:

- two writer servers share one directory store, each with its own local cache, and clients
  submit the same inputs to either, so a result computed by one must be a cache HIT on the
  other; some submits say `Cache-Control: no-cache`, which republishes (history);
- `haversack cache sweep` of that store and, with --sync-url, `haversack cache sync` of it
  into a bucket, both on a schedule while jobs run;
- a read-only `haversack serve-store` over the directory store, whose listing must come to
  hold every result the writers published;
- one writer killed with SIGKILL mid-run and restarted on the same directories.

What must hold, checked as it runs and at the end:

1. No 5xx from any server, and every job that was not on the killed writer ends `done`.
2. A job's downloaded result hashes to the digest the job reported.
3. A cache hit serves bytes some publication of that key actually produced.
4. The reader lists every key the writers published.
5. After a final sweep and sync, every key reads back from the directory store AND the
   bucket with the same current bytes.

Inputs are variants of the given images, each with one voxel changed, so there are more
distinct results than images. The bucket's credentials are taken from SOAK_AWS_ACCESS_KEY_ID,
SOAK_AWS_SECRET_ACCESS_KEY, SOAK_AWS_ENDPOINT and SOAK_AWS_REGION and handed to the sync
subprocess ONLY: the writer servers never see them, so no data source of theirs can be
pointed at the bucket by accident. Writes a report and the servers' logs under --root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import uuid
from pathlib import Path

TOKEN = "soak-" + uuid.uuid4().hex


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _haversack() -> str:
    return str(Path(sys.executable).with_name("haversack"))


def _variants(images, per_image: int, into: Path) -> list:
    """``per_image`` copies of each image, the n-th with one voxel raised by n."""
    import nibabel as nib
    import numpy as np
    out = []
    into.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        src = nib.load(str(img))
        base = np.asanyarray(src.dataobj)
        for n in range(per_image):
            data = base.copy()
            data.flat[0] = data.flat[0] + n
            p = into / f"in{i}-v{n}.nii.gz"
            nib.save(nib.Nifti1Image(data, src.affine, src.header), str(p))
            out.append(p)
    return out


class Server:
    """One `haversack` server process, restartable on the same port and directories."""

    def __init__(self, name, argv, port, root: Path, env):
        self.name, self.argv, self.port, self.env = name, argv, port, env
        self.log = open(root / f"{name}.log", "a")
        self.proc = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout=180.0):
        self.proc = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.env)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited at startup; see {self.log.name}")
            try:
                _request(f"{self.base}/v1/health", timeout=2)
                return
            except OSError:
                time.sleep(0.3)
        raise RuntimeError(f"{self.name} did not answer within {timeout:.0f}s")

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait(30)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.kill()


def _request(url, *, data=None, headers=None, method=None, timeout=60):
    """Through haversack's one door for URLs (tests/test_fetchlib.py holds every tool to
    one), though every URL here is this machine's own servers."""
    from haversack import fetchlib
    with fetchlib.open(url, data=data, headers=headers or {}, method=method,
                       timeout=timeout) as r:
        return r.status, r.read(), dict(r.headers)


def _multipart(path: Path, fields: dict):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                     f"{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                 .encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class Soak:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root or tempfile.mkdtemp(prefix="haversack-soak-server-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_url = f"file://{self.root / 'store'}"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.published: dict = {}              # key -> set of digests a publication produced
        self.hits: list = []                   # (server, key, digest) of every cache hit
        self.notes = None
        self.counts = {"jobs": 0, "done": 0, "hits": 0, "republished": 0, "failed": 0,
                       "cancelled": 0, "lost_to_kill": 0, "syncs": 0, "sweeps": 0}
        self.errors: list = []
        self.latency: list = []
        self.killing = threading.Event()       # while set, writer 2 is expected to be gone

    # -- the pieces -----------------------------------------------------------------------

    def env(self, *, bucket: bool = False) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "SOAK_AWS_"))}
        env["HAVERSACK_CACHE_DIR"] = str(self.root / "hvcache")
        if bucket:
            for name in ("ACCESS_KEY_ID", "SECRET_ACCESS_KEY", "ENDPOINT", "REGION"):
                if os.environ.get(f"SOAK_AWS_{name}"):
                    env[f"AWS_{name}"] = os.environ[f"SOAK_AWS_{name}"]
        return env

    def error(self, what: str):
        with self.lock:
            self.errors.append(f"{time.strftime('%H:%M:%S')} {what}")
        print(f"  ERROR {what}", flush=True)

    def writer(self, i: int) -> Server:
        port = _free_port()
        argv = [_haversack(), "serve", "--port", str(port), "--token", TOKEN,
                "--result-store", self.store_url, "--cache-dir", str(self.root / f"cache-{i}"),
                "--workdir", str(self.root / f"work-{i}"), "--max-pending", "64",
                "--keep-finished", "400", "--sweep-interval-hours", "0"]
        return Server(f"writer{i}", argv, port, self.root, self.env())

    def reader(self) -> Server:
        port = _free_port()
        argv = [_haversack(), "serve-store", self.store_url, "--port", str(port),
                "--cache-dir", str(self.root / "reader-cache")]
        return Server("reader", argv, port, self.root, self.env())

    # -- the load -------------------------------------------------------------------------

    def client(self, n: int):
        rng = random.Random(n)
        while not self.stop.is_set():
            w = rng.randrange(len(self.writers))
            server = self.writers[w]
            path = rng.choice(self.inputs)
            recompute = rng.random() < self.args.recompute
            headers = {"Authorization": f"Bearer {TOKEN}"}
            if recompute:
                headers["Cache-Control"] = "no-cache"
            body, ctype = _multipart(path, {"task": self.args.task, "options": "{}"})
            headers["Content-Type"] = ctype
            t0 = time.time()
            try:
                status, raw, _ = _request(f"{server.base}/v1/jobs", data=body,
                                          headers=headers, method="POST")
                jid = json.loads(raw)["id"]
                s = self.wait(server, jid)
            except (OSError, ValueError) as e:
                if w == 1 and self.killing.is_set() or self.killed_recently(w):
                    with self.lock:
                        self.counts["lost_to_kill"] += 1
                    time.sleep(1)
                    continue
                if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                    self.error(f"{server.name} refused a submit: {e.code} {e.read()[:200]!r}")
                elif isinstance(e, urllib.error.HTTPError) and e.code == 429:
                    time.sleep(2)
                    continue
                else:
                    self.error(f"{server.name}: {type(e).__name__}: {e}")
                continue
            if s is None:
                continue
            self.record(server, s, recompute, time.time() - t0)

    def killed_recently(self, w: int) -> bool:
        return w == 1 and time.time() - getattr(self, "killed_at", 0) < 120

    def wait(self, server, jid, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            _, raw, _ = _request(f"{server.base}/v1/jobs/{jid}",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
            s = json.loads(raw)
            if s["state"] in ("done", "failed", "cancelled"):
                return s
            time.sleep(0.5)
        self.error(f"{server.name} job {jid} not finished after {timeout}s")
        return None

    def record(self, server, s, recompute, seconds):
        with self.lock:
            self.counts["jobs"] += 1
        if s["state"] != "done" and self.killed_recently(self.writers.index(server)):
            with self.lock:                    # a restarted writer reports what the kill
                self.counts["lost_to_kill"] += 1   # interrupted: expected, not a defect
            return
        if s["state"] != "done":
            with self.lock:
                self.counts[s["state"]] += 1
            self.error(f"{server.name} job {s['id']} ended {s['state']}: {s.get('error')}")
            return
        out = (s.get("result") or {}).get("outputs") or [{}]
        want = out[0].get("sha256")
        try:
            _, body, _ = _request(server.base + s["links"]["result"],
                                  headers={"Authorization": f"Bearer {TOKEN}"}, timeout=120)
        except OSError as e:
            self.error(f"{server.name} result of {s['id']}: {e}")
            return
        got = "sha256:" + hashlib.sha256(body).hexdigest()
        if want and got != want:
            self.error(f"{server.name} job {s['id']}: served {got}, reported {want}")
        key = s.get("key")
        with self.lock:
            self.counts["done"] += 1
            self.latency.append(seconds)
            self.published.setdefault(key, set())
            if s.get("cached"):
                # judged at the END, against every publication any client saw: a hit on
                # one writer can serve bytes another client's job published a moment ago
                # and has not recorded yet (and `error` takes this lock - calling it from
                # here deadlocked the second run of this soak)
                self.counts["hits"] += 1
                self.hits.append((server.name, key, got))
            else:
                if recompute:
                    self.counts["republished"] += 1
                self.published[key].add(got)

    # -- the background -------------------------------------------------------------------

    def every(self, seconds, fn):
        while not self.stop.wait(seconds):
            fn()

    def sweep(self):
        r = subprocess.run([_haversack(), "cache", "sweep", self.store_url, "--grace-hours",
                            str(self.args.sweep_grace_hours), "--quiet"],
                           capture_output=True, text=True, env=self.env())
        with self.lock:
            self.counts["sweeps"] += 1
        if r.returncode != 0:
            self.error(f"sweep exited {r.returncode}: {r.stderr.strip()[-300:]}")

    def sync(self):
        t0 = time.time()
        r = subprocess.run([_haversack(), "cache", "sync", self.store_url,
                            self.args.sync_url, "--quiet"],
                           capture_output=True, text=True, env=self.env(bucket=True))
        with self.lock:
            self.counts["syncs"] += 1
        line = (r.stderr.strip().splitlines() or [""])[-1]
        print(f"  sync {time.time() - t0:.1f}s: {line}", flush=True)
        if r.returncode != 0:
            self.error(f"sync exited {r.returncode}: {r.stderr.strip()[-300:]}")

    def reader_keys(self) -> set:
        from urllib.parse import quote
        keys, cursor = set(), None
        while True:
            url = f"{self.reader_server.base}/v1/segmentations?limit=1000"
            if cursor:
                url += f"&cursor={quote(cursor)}"
            _, raw, _ = _request(url)
            page = json.loads(raw)
            keys |= {r["key"] for r in page.get("segmentations", [])}
            cursor = page.get("next_cursor")
            if not cursor:
                return keys

    def kill_and_restart(self):
        if self.stop.wait(self.args.kill_after):
            return
        print("  SIGKILL writer1 (it restarts in 5 s)", flush=True)
        self.killing.set()
        self.killed_at = time.time()
        self.writers[1].kill()
        time.sleep(5)
        self.writers[1].start()
        self.killing.clear()
        print("  writer1 is back", flush=True)

    # -- the run --------------------------------------------------------------------------

    def run(self) -> int:
        a = self.args
        print(f"soak root {self.root}", flush=True)
        self.inputs = _variants(a.images, a.variants, self.root / "inputs")
        self.writers = [self.writer(i) for i in range(2)]
        self.reader_server = self.reader()
        for s in (*self.writers, self.reader_server):
            s.start()
        print(f"{len(self.inputs)} inputs, task {a.task}, {a.clients} clients, "
              f"{a.minutes} min; writers on {[w.port for w in self.writers]}, reader on "
              f"{self.reader_server.port}", flush=True)
        threads = [threading.Thread(target=self.client, args=(n,), daemon=True)
                   for n in range(a.clients)]
        threads.append(threading.Thread(target=self.every, args=(a.sweep_every, self.sweep),
                                        daemon=True))
        if a.sync_url:
            threads.append(threading.Thread(target=self.every, args=(a.sync_every, self.sync),
                                            daemon=True))
        if a.kill_after:
            threads.append(threading.Thread(target=self.kill_and_restart, daemon=True))
        for t in threads:
            t.start()
        deadline = time.time() + a.minutes * 60
        try:
            while time.time() < deadline:
                time.sleep(min(60, max(1, deadline - time.time())))
                with self.lock:
                    c = dict(self.counts)
                print(f"  {time.strftime('%H:%M:%S')} {c}  errors {len(self.errors)}",
                      flush=True)
        finally:
            self.stop.set()
            for t in threads[:a.clients]:
                t.join(timeout=900)
        return self.finish()

    def finish(self) -> int:
        from haversack.objectcache import SharedResultCache
        from haversack.serve import ResultCache
        a = self.args
        self.sweep()
        if a.sync_url:
            self.sync()
        # a hit is a publication's bytes: some job computed them. A job lost to the kill can
        # have published without its client recording it, so its bytes are the one
        # legitimate source a hit can have that no recorded job accounts for
        strays = [(n, k, d) for n, k, d in self.hits if d not in self.published.get(k, ())]
        if strays:
            n, k, d = strays[0]
            what = (f"{len(strays)} cache hit(s) served bytes no recorded job published; first: "
                    f"{n} served {d[:19]} for {k[:12]}")
            if self.counts["lost_to_kill"]:
                print(f"  note: {what} - possible from a job the kill interrupted after it "
                      "published", flush=True)
                self.notes = what
            else:
                self.error(what)
        published = set(self.published)
        listed = self.reader_keys()
        missing = published - listed
        if missing:
            self.error(f"the reader does not list {len(missing)} published key(s)")
        disk = SharedResultCache.open(self.store_url, ResultCache(self.root / "check-disk"))
        stores = [("directory", disk)]
        if a.sync_url:
            os.environ.update({k: v for k, v in self.env(bucket=True).items()
                               if k.startswith("AWS_")})
            bucket = SharedResultCache.open(a.sync_url, ResultCache(self.root / "check-bucket"))
            stores.append(("bucket", bucket))
        for key in sorted(published):
            current = {}
            for name, cache in stores:
                hit = cache.get(key)
                if hit is None:
                    self.error(f"{key[:12]} does not read back from the {name}")
                    continue
                current[name] = "sha256:" + hashlib.sha256(Path(hit[0]).read_bytes()).hexdigest()
                if current[name] not in self.published[key]:
                    self.error(f"{key[:12]} reads back from the {name} as bytes no "
                               "publication produced")
            if len(set(current.values())) > 1:
                self.error(f"{key[:12]}: the directory and the bucket disagree")
        for s in (*self.writers, self.reader_server):
            s.stop()
        lat = sorted(self.latency)
        report = {"counts": self.counts, "keys": len(published), "listed": len(listed),
                  "latency_s": {"p50": lat[len(lat) // 2] if lat else None,
                                "p95": lat[int(len(lat) * 0.95)] if lat else None},
                  "notes": self.notes, "errors": self.errors}
        (self.root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        ok = not self.errors and self.counts["done"] > 0
        print("SOAK", "PASSED" if ok else "FAILED", f"(report and logs in {self.root})")
        if ok and not a.keep:
            shutil.rmtree(self.root / "store", ignore_errors=True)
            if a.sync_url:
                from provender import ops
                gone = 0
                for batch in ops.list(bucket.store, bucket.prefix):
                    for o in batch:
                        ops.delete(bucket.store, o["path"])
                        gone += 1
                print(f"cleaned {gone} object(s) under {bucket.prefix} in the bucket")
        return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--task", default="ts.v2:total_fast")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--variants", type=int, default=6, help="inputs made from each image")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--recompute", type=float, default=0.1,
                    help="fraction of submits sent with Cache-Control: no-cache")
    ap.add_argument("--sync-url", help="a bucket to `cache sync` the store into")
    ap.add_argument("--sync-every", type=float, default=90.0)
    ap.add_argument("--sweep-every", type=float, default=120.0)
    ap.add_argument("--sweep-grace-hours", type=float, default=0.05,
                    help="3 minutes: far longer than a publication takes, so a sweep must "
                         "never take a live blob")
    ap.add_argument("--kill-after", type=float, default=300.0,
                    help="SIGKILL the second writer this many seconds in (0: never)")
    ap.add_argument("--root", help="where the store, logs and report go (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the store after a pass")
    return Soak(ap.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
