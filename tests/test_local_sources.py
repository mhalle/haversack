"""Remote inputs on the local CLI (2026-09-03): `haversack segment idc:<uuid>` and
`haversack segment https://.../scan.nii.gz` fetch once into a local cache and segment the
result. Hosted sources are the server's registry; http(s) is a local-only source that no
server carries. Fakes and a loopback HTTP server - nothing leaves the machine.
"""
import http.server
import io
import threading
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


def test_two_processes_materializing_one_input_do_not_corrupt_each_other(tmp_path):
    """Two `haversack` commands on one input is ordinary - the input cache is a single
    shared user-level root - and it used to be destructive.

    The old protocol read an absent `.done` as "the writer is dead", so the second
    caller deleted the first's in-progress tree and fetched into a directory the first
    still believed it owned. Both could then write `.done` over contents assembled from
    two fetches, and `.done` records the identifier rather than a size or a digest, so
    nothing downstream could tell. Every slice here carries its writer's marker, so a
    directory holding both markers is exactly that mixed entry.

    Real subprocesses, not threads: the claim is an advisory FILE lock, and threads in
    one process share it by definition and would prove nothing.
    """
    import subprocess
    import sys

    cache = tmp_path / "cache"
    env = {**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    probe = str(Path(__file__).resolve().parent / "slow_source_probe.py")
    started = tmp_path / "A-has-begun"
    procs = [
        subprocess.Popen([sys.executable, probe, "A", str(cache), "case1",
                          "--signal", str(started)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env),
        # B enters only once A is demonstrably mid-fetch, which is the window that
        # used to destroy A's tree; leaving it to start-up jitter reproduced nothing
        subprocess.Popen([sys.executable, probe, "B", str(cache), "case1",
                          "--await", str(started)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env),
    ]
    outs = [p.communicate(timeout=180) for p in procs]
    for (out, err), m in zip(outs, "AB"):
        assert procs["AB".index(m)].returncode == 0, f"writer {m} failed:\n{err}"

    entry = cache / "slow" / sorted(p.name for p in (cache / "slow").iterdir()
                                    if not p.name.startswith("."))[0]
    series = entry / "series"
    slices = sorted(series.glob("*.dcm"))
    markers = {p.read_text(encoding="utf-8") for p in slices}
    assert len(slices) == 6, f"the published entry is short: {[p.name for p in slices]}"
    assert len(markers) == 1, f"the published entry mixes two fetches: {markers}"
    assert (entry / ".done").exists()
    # both callers returned the SAME published directory, and neither left staging behind
    assert {out.strip() for out, _ in outs} == {str(series)}
    assert not [p for p in (cache / "slow").iterdir() if p.name.startswith(".staging")]
