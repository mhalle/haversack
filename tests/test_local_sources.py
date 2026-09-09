"""Remote inputs on the local CLI (2026-09-03): `haversack segment idc:<uuid>` and
`haversack segment https://.../scan.nii.gz` fetch once into a local cache and segment the
result. Hosted sources are the server's registry; http(s) is a local-only source that no
server carries. Fakes and a loopback HTTP server - nothing leaves the machine.
"""
import http.server
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from haversack import sources
from haversack.errors import InputError


class FakeSource(sources.DataSource):
    prefix = "fake"
    id_pattern = r"[a-z0-9]+"
    description = "test double"

    def __init__(self):
        self.calls = 0

    def fetch(self, identifier, dest_dir, *, credentials=None):
        self.calls += 1
        d = Path(dest_dir) / "series"
        d.mkdir()
        (d / f"{identifier}.nii.gz").write_bytes(b"not really nifti")
        return d


def test_parse_input_tells_remote_from_local():
    assert sources.parse_input("idc:0123abcd-0000-0000-0000-000000000000") == ("idc", "0123abcd-0000-0000-0000-000000000000")
    assert sources.parse_input("zenodo:7262581/amos22.zip!amos22/imagesVa/amos_0575.nii.gz")[0] == "zenodo"
    assert sources.parse_input("https://example.org/a.nii.gz") == ("http", "https://example.org/a.nii.gz")
    # a registered prefix is recognized even when it does not fit the "three or
    # more letters" shape a local path is told apart by: s3 is two characters
    # and carries a digit
    assert sources.parse_input("s3:fcp-indi/data/sub-01_T1w.nii.gz") == (
        "s3", "fcp-indi/data/sub-01_T1w.nii.gz")
    assert sources.parse_input("github:o/r@v1.0.0/a.zip!b.nii.gz")[0] == "github"
    for local in ("scan.nii.gz", "/abs/path/ct.nrrd", "C:\\data\\ct.nii", "./series", "a:b"):
        assert sources.parse_input(local) is None, local


def test_materialize_fetches_once_and_returns_the_file(tmp_path):
    fake = FakeSource()
    seen = []
    p1 = sources.materialize("fake:abc", cache_dir=tmp_path, sources=[fake], progress=seen.append)
    p2 = sources.materialize("fake:abc", cache_dir=tmp_path, sources=[fake])
    assert p1 == p2 and p1.name == "abc.nii.gz" and p1.read_bytes() == b"not really nifti"
    assert fake.calls == 1 and seen == ["fetching fake:abc"]
    assert sources.materialize(tmp_path / "local.nii.gz", cache_dir=tmp_path, sources=[fake]) == tmp_path / "local.nii.gz"


def test_materialize_refuses_unknown_and_invalid(tmp_path):
    with pytest.raises(InputError, match="unknown input source"):
        sources.materialize("nope:x", cache_dir=tmp_path, sources=[FakeSource()])
    with pytest.raises(InputError, match="not a valid fake identifier"):
        sources.materialize("fake:NOT-VALID", cache_dir=tmp_path, sources=[FakeSource()])


def test_idc_without_obstore_names_the_extra(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_obstore(name, *a, **k):
        if name == "obstore":
            raise ImportError("no obstore")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_obstore)
    with pytest.raises(InputError, match="obstore"):
        sources.materialize("idc:0123abcd-0000-0000-0000-000000000000", cache_dir=tmp_path)


def test_http_source_is_local_only():
    assert "http" not in sources.registry()            # a server never carries it


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    (root / "scan.nii.gz").write_bytes(b"NIFTI-ish bytes")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("amos22/imagesVa/amos_0575.nii.gz", b"member bytes " * 1000)
        z.writestr("amos22/imagesVa/amos_0576.nii.gz", b"other")
    (root / "amos22.zip").write_bytes(buf.getvalue())

    class Quiet(http.server.SimpleHTTPRequestHandler):
        """SimpleHTTPRequestHandler plus HTTP Range (the stdlib one answers 200 to a
        Range request, which the zip reader rightly refuses)."""

        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            if not rng or not path.is_file():
                return super().do_GET()
            data = path.read_bytes()
            lo, _, hi = rng.removeprefix("bytes=").partition("-")
            lo = int(lo or 0)
            hi = int(hi) if hi else len(data) - 1
            chunk = data[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)

    handler = lambda *a, **k: Quiet(*a, directory=str(root), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_http_whole_file_and_zip_member(served, tmp_path):
    cache = tmp_path / "cache"
    whole = sources.materialize(f"{served}/scan.nii.gz", cache_dir=cache)
    assert whole.name == "scan.nii.gz" and whole.read_bytes() == b"NIFTI-ish bytes"
    member = sources.materialize(f"{served}/amos22.zip!amos22/imagesVa/amos_0575.nii.gz", cache_dir=cache)
    assert member.name == "amos_0575.nii.gz" and member.read_bytes() == b"member bytes " * 1000
    assert not (member.parent / "amos_0576.nii.gz").exists()       # only the named member
    assert whole.parent != member.parent                            # separate cache entries


def test_cli_segment_takes_a_remote_input(monkeypatch, tmp_path, capsys):
    from haversack import cli, pipeline
    fake = FakeSource()
    monkeypatch.setattr(sources, "default_sources", lambda: [fake])
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    got = {}
    monkeypatch.setattr(pipeline, "segment", lambda image, task, **kw: got.update(image=image) or _Saved())
    rc = cli.main(["segment", "fake:abc", "--task", "total_fast", "-o", str(tmp_path / "out.nii.gz")])
    assert rc == 0, capsys.readouterr().err
    assert Path(got["image"]).name == "abc.nii.gz" and Path(got["image"]).exists()
    assert "fetching fake:abc" in capsys.readouterr().err


class _Saved:
    timings = {}
    provenance = {}
    grid = type("G", (), {"shape": (1, 1, 1)})()
    schema = type("S", (), {"names": []})()

    def save(self, path):
        return path

    def present(self):
        return {}


def _probe(*args, cache=None):
    """Run one materialize in a real subprocess.

    The child must import haversack the way the PARENT does, and the parent's way
    differs between environments: here it comes from the venv, in CI it comes from
    `PYTHONPATH=src` with the project deliberately not installed (`uv run
    --no-project`, so the hand-installed CPU torch survives). Replacing PYTHONPATH
    with the repo root passed locally and failed CI with `No module named
    'haversack'` - the inherited value is prepended to, never overwritten."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    inherited = os.environ.get("PYTHONPATH", "")
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(
               [p for p in (str(root), str(root / "src"), inherited) if p])}
    probe = str(Path(__file__).resolve().parent / "slow_source_probe.py")
    return subprocess.Popen([sys.executable, probe, *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)


def _sole_entry(cache):
    kinds = [p for p in (cache / "slow").iterdir() if not p.name.startswith(".")]
    assert len(kinds) == 1, [p.name for p in kinds]
    return kinds[0]


def test_two_processes_materializing_one_input_EXCLUDE_EACH_OTHER(tmp_path):
    """Two `haversack` commands on one input is ordinary - the input cache is a single
    shared user-level root - and it used to be destructive.

    The old protocol read an absent `.done` as "the writer is dead", so the second
    caller deleted the first's in-progress tree and fetched into a directory the first
    still believed it owned. Both could then write `.done` over contents assembled from
    two fetches, and `.done` records the identifier rather than a size or a digest, so
    nothing downstream could tell.

    What is asserted is MUTUAL EXCLUSION, not an outcome. An earlier version compared
    the published bytes and caught only the interleavings that happened to corrupt -
    a review measured that at 4 runs in 10, and it could not tell the shipped protocol
    from one with no lock at all. The two fetches' intervals are recorded by the source
    itself, and they must not overlap; with the lock removed the second fetch starts
    while the first is still writing, every time, because the hand-off below guarantees
    it. Real subprocesses, not threads: the claim is an advisory FILE lock, which
    threads in one process share by definition.
    """
    cache, started, events = tmp_path / "cache", tmp_path / "A-began", tmp_path / "events"
    procs = [
        _probe("A", str(cache), "case1", "--signal", str(started), "--events", str(events)),
        # B enters only once A is demonstrably mid-fetch: the window that used to
        # destroy A's tree. Leaving it to start-up jitter reproduced nothing.
        _probe("B", str(cache), "case1", "--await", str(started), "--events", str(events)),
    ]
    outs = [p.communicate(timeout=180) for p in procs]
    for i, (out, err) in enumerate(outs):
        assert procs[i].returncode == 0, f"writer {'AB'[i]} failed:\n{err}"

    log = [ln.split() for ln in events.read_text().splitlines() if ln.strip()]
    spans = {}
    for marker, event, when in log:
        spans.setdefault(marker, {})[event] = float(when)
    assert spans, "the probes recorded no fetch at all"
    for marker, span in spans.items():
        assert "fetch-end" in span, f"writer {marker} started a fetch it never finished"
    ordered = sorted(spans.values(), key=lambda s: s["fetch-start"])
    for earlier, later in zip(ordered, ordered[1:]):
        assert later["fetch-start"] >= earlier["fetch-end"], (
            "two fetches of one input overlapped in time: the second began "
            f"{earlier['fetch-end'] - later['fetch-start']:.2f}s before the first ended, "
            "so nothing was excluding them")

    entry = _sole_entry(cache)
    slices = sorted((entry / "series").glob("*.dcm"))
    markers = {p.read_text(encoding="utf-8") for p in slices}
    assert len(slices) == 6, f"the published entry is short: {[p.name for p in slices]}"
    assert len(markers) == 1, f"the published entry mixes two fetches: {markers}"
    assert (entry / ".done").exists()
    assert {out.strip() for out, _ in outs} == {str(entry / "series")}
    assert not [p for p in (cache / "slow").iterdir() if p.name.startswith(".staging")]


def test_a_writer_killed_mid_fetch_publishes_nothing_and_leaves_no_litter(tmp_path):
    """The half the exclusion test cannot see: what the staging directory is FOR.

    Under a lock alone, a second caller simply waits and then finds `.done`, so a
    protocol with no staging passes that test - which is what a review demonstrated.
    Staging earns its place when a writer dies: nothing half-fetched may be published,
    and what it abandoned must not become permanent invisible litter. It briefly was:
    `.staging-*` is a dotfile, and `cache_admin._entries` skips dotfiles, so `cache
    clean` could not see it while `cache usage` counted its bytes.
    """
    import signal

    cache, started = tmp_path / "cache", tmp_path / "A-began"
    doomed = _probe("A", str(cache), "case1", "--signal", str(started))
    deadline = time.monotonic() + 60
    while not started.exists():
        assert time.monotonic() < deadline, "the writer never began fetching"
        assert doomed.poll() is None, "the writer exited before it began"
        time.sleep(0.01)
    doomed.send_signal(signal.SIGKILL)          # no teardown runs: the harshest case
    doomed.communicate(timeout=60)

    published = [p for p in (cache / "slow").iterdir() if not p.name.startswith(".")]
    assert published == [], (
        f"a killed writer published something: {[p.name for p in published]} - the entry "
        "is only ever created by the final rename, so nothing half-fetched can appear "
        "under a name a reader looks up")
    abandoned = [p for p in (cache / "slow").iterdir() if p.name.startswith(".staging")]
    assert abandoned, "expected the killed writer's staging to be left behind"

    ok = _probe("B", str(cache), "case1")
    out, err = ok.communicate(timeout=180)
    assert ok.returncode == 0, err
    entry = _sole_entry(cache)
    assert (entry / ".done").exists()
    assert {p.read_text(encoding="utf-8") for p in (entry / "series").glob("*.dcm")} == {"B"}
    left = [p.name for p in (cache / "slow").iterdir() if p.name.startswith(".staging")]
    assert left == [], f"the dead writer's staging survived a later fetch: {left}"


@pytest.fixture
def mutable_www(tmp_path):
    """A loopback server whose files can be REPLACED while a test runs, with Range
    support (the stdlib handler answers 200 to a Range request, which the zip reader
    rightly refuses)."""
    root = tmp_path / "mut"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            if not rng or not path.is_file():
                return super().do_GET()
            data = path.read_bytes()
            lo, _, hi = rng.removeprefix("bytes=").partition("-")
            lo, hi = int(lo or 0), (int(hi) if hi else len(data) - 1)
            chunk = data[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)

    srv = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(root), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", root
    srv.shutdown()


def _archive(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("padding.txt", b"x" * 4096)          # move the member's offset
        z.writestr(member, payload)
    return buf.getvalue()


def test_a_forced_refresh_REALLY_REREADS_an_archive_that_changed_upstream(mutable_www, tmp_path):
    """`Cache-Control: no-cache` against an archive that was genuinely replaced.

    This is the case the refresh exists for: `s3:` keys and `github:` release assets
    can both be replaced under one identifier - the GitHub source's own docstring says
    so - and a result cached under such an identity can outlive the bytes it describes.

    The policy dropped the cached series and the read-ahead image and stopped there,
    while the SOURCE INSTANCE - which lives as long as the process on both deployments -
    still held the parsed archive: a ZipFile with its central directory and a RangeFile
    block cache. On a hit `resolve` is skipped, so the refetch reused the old resolved
    URL and the old member offsets. The padding member above exists to move the real
    member's offset between the two versions, so reading at the stale offsets cannot
    accidentally succeed.

    The unit tests either side of this one check that `forget` clears the cache and that
    the policy calls it. What only this can check is that the SERVER passes the source in
    at all - `_refresh_input(kind=..., ident=...)` looking it up in `self.sources` - and
    that a real replaced archive really is re-read end to end.
    """
    pytest.importorskip("fastapi")
    from test_serve import FakeSegmenter, LocalExecutor, TestClient, create_app, wait_state

    url, root = mutable_www
    (root / "a.zip").write_bytes(_archive("inner/scan.nii.gz", b"\x1f\x8bOLD" + b"o" * 900))

    class Mutable(sources.ArchiveReadingSource):
        """Shaped like the sources whose identity is not content-pinned."""

        prefix = "mut"
        id_pattern = r"[A-Za-z0-9._/-]+(?:![A-Za-z0-9._/-]+)?"
        description = "test double: an archive that can be replaced under one name"

        def enabled(self):
            return True

        def resolve(self, outer, credentials=None):
            return f"{url}/{outer}", len((root / outer).read_bytes())

    def digest_of(client, no_cache=False):
        headers = {"Cache-Control": "no-cache"} if no_cache else {}
        r = client.post("/v1/jobs",
                        data={"task": "total_fast",
                              "source": json.dumps([{"kind": "mut",
                                                     "id": "a.zip!inner/scan.nii.gz"}])},
                        headers=headers)
        assert r.status_code == 202, r.text
        s = wait_state(client, r.json()["id"], ("done", "failed"))
        assert s["state"] == "done", s.get("error")
        return s["result"]["provenance"]["inputs"][0]["content"]["digest"]

    ex = LocalExecutor(FakeSegmenter(), workdir=tmp_path / "w",
                       cache_dir=tmp_path / "rc", sources=[Mutable()])
    try:
        client = TestClient(create_app(ex))
        first = digest_of(client)

        # the same identifier, different bytes upstream, and a different layout
        (root / "a.zip").write_bytes(
            _archive("inner/scan.nii.gz", b"\x1f\x8bNEW" + b"n" * 1500))

        second = digest_of(client, no_cache=True)
        assert second != first, (
            "the forced refetch returned the bytes of the archive this source had "
            "already parsed; `resolve` was skipped on the cache hit, so it read the old "
            "member offsets from the old central directory")
    finally:
        ex.close()
