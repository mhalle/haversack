"""The archive-reading source family, tested against a local in-process HTTP
server that serves Range requests - no network, real bytes."""
import http.server
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from haversack.errors import InputError
from haversack.sources import (ArchiveReadingSource, GCSSource, GitHubReleaseSource, HuggingFaceSource,
                           RangeFile, S3Source, ZenodoSource, registry)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        body = self.payload
        if rng:
            lo, hi = rng.split("=")[1].split("-")
            lo, hi = int(lo), min(int(hi), len(body) - 1)
            chunk = body[lo:hi + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(body)}")
        else:
            chunk = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def range_server():
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("case1/ct.nii.gz", b"\x1f\x8b" + b"CT" * 500)
        z.writestr("case1/seg.nii.gz", b"\x1f\x8b" + b"SG" * 300)
        z.writestr("case2/ct.nii.gz", b"\x1f\x8b" + b"XX" * 400)
        z.writestr("../evil.txt", b"nope")
    _RangeHandler.payload = zbuf.getvalue()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/archive.zip", len(_RangeHandler.payload)
    srv.shutdown()


def test_rangefile_reads_match_direct_bytes(range_server):
    url, size = range_server
    rf = RangeFile(url, size, block=256)
    rf.seek(-100, 2)
    tail = rf.read(100)
    assert tail == _RangeHandler.payload[-100:]
    rf.seek(10)
    assert rf.read(50) == _RangeHandler.payload[10:60]
    assert rf.requests >= 1


class _Src(ArchiveReadingSource):
    prefix = "tst"
    id_pattern = r"[a-z.]+(?:![A-Za-z0-9._/]+)?"

    def __init__(self, url, size):
        self._url, self._size = url, size
        self.seen_credentials = []

    def resolve(self, outer, credentials=None):
        self.seen_credentials.append(credentials)
        return self._url, self._size


def test_member_extraction_fetches_member_only(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!case1/ct.nii.gz", tmp_path)
    assert (d / "ct.nii.gz").read_bytes().startswith(b"\x1f\x8bCT")
    assert sorted(p.name for p in d.iterdir()) == ["ct.nii.gz"]


def test_folder_prefix_extracts_the_case(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!case1/", tmp_path)
    assert sorted(p.name for p in d.iterdir()) == ["ct.nii.gz", "seg.nii.gz"]


def test_zip_slip_member_lands_flat(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip!../evil.txt", tmp_path)
    assert (d / "evil.txt").exists()                 # flattened by basename
    assert not (tmp_path.parent / "evil.txt").exists()


def test_plain_file_downloads_whole(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    d = src.fetch("archive.zip", tmp_path)
    assert (d / "archive.zip").stat().st_size == size


def test_credentials_reach_the_resolver(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    src.fetch("archive.zip!case2/ct.nii.gz", tmp_path, credentials="tok123")
    assert "tok123" in src.seen_credentials


def test_missing_member_is_a_clear_error(range_server, tmp_path):
    url, size = range_server
    src = _Src(url, size)
    with pytest.raises(InputError, match="no such member"):
        src.fetch("archive.zip!nope/", tmp_path)


def test_grammars_and_registry():
    import re
    reg = registry(None)
    assert set(reg) >= {"idc", "tcia", "openneuro", "zenodo", "hf"}
    z = reg["zenodo"].id_pattern
    assert re.fullmatch(z, "6802614/Totalsegmentator_dataset.zip!Totalsegmentator_dataset/s0001/ct.nii.gz")
    assert re.fullmatch(z, "6802614/labels.json")
    h = reg["hf"].id_pattern
    sha = "a" * 40
    assert re.fullmatch(h, f"org-x/data.set@{sha}/images/a.nii.gz!inner/file.nii.gz")
    assert not re.fullmatch(h, "org/data@main/images/a.nii.gz")   # floating refs refused
    assert not re.fullmatch(h, f"org/data@{'a' * 39}/x.nii.gz")


def test_token_header_reaches_fetch_and_never_leaks(tmp_path):
    """Haversack-Source-Token flows to the source's fetch and appears nowhere a
    client can read: not in status, not in the jobs listing, not in cache
    meta. The prefetch (which has no request context) fails without the
    token and the dispatcher's inline fetch with it succeeds - the existing
    fallback covers restricted prefetches by design."""
    import json as _json

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_serve import FakeSegmenter, wait_state

    seen = []

    class TokSource:
        prefix = "tok"
        id_pattern = r"[a-z0-9]+"
        description = ""

        def enabled(self):
            return True

        def identity(self, i):
            return f"tok:{i}"

        def describe(self):
            return {"prefix": "tok", "id_pattern": self.id_pattern,
                    "enabled": True, "description": ""}

        def fetch(self, ident, dest_dir, *, credentials=None):
            seen.append(credentials)
            if credentials != "sekrit":
                raise InputError("token required")
            d = dest_dir / "series"
            d.mkdir(parents=True, exist_ok=True)
            (d / "x.nii.gz").write_bytes(b"\x1f\x8b" + ident.encode())
            return d

    seg = FakeSegmenter()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       sources=[TokSource()])
    client = TestClient(create_app(ex))
    r = client.post("/v1/jobs",
                    data={"task": "total_fast",
                          "source": _json.dumps([{"kind": "tok", "id": "abc123"}])},
                    headers={"Haversack-Source-Token": "tok=sekrit"})
    assert r.status_code == 202, r.text
    jid = r.json()["id"]
    st = wait_state(client, jid, ("done",))
    assert "sekrit" in seen                          # the fetch got the token
    assert "sekrit" not in _json.dumps(st)           # ...and no client surface does
    assert "sekrit" not in client.get("/v1/jobs").text
    assert "sekrit" not in client.get("/v1/segmentations").text
    assert "source_tokens" not in st


def test_slashed_identifiers_get_the_full_path_surface(tmp_path, monkeypatch):
    """Multi-segment identifiers (hf paths, zenodo recid/file!member) live in
    ordinary URLs via greedy right-to-left routes: blocking GET initiates,
    HEAD probes, meta and preview serve, DELETE evicts, and the listing links
    the slashed path."""
    import json as _json

    import numpy as np
    import SimpleITK as sitk
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_serve import FakeSegmenter

    class DeepSource:
        prefix = "deep"
        id_pattern = r"[a-z0-9]+/[a-z0-9@.]+/[a-z0-9._/!-]+"
        description = ""

        def enabled(self):
            return True

        def identity(self, i):
            return f"deep:{i}"

        def describe(self):
            return {"prefix": "deep", "id_pattern": self.id_pattern,
                    "enabled": True, "description": ""}

        def fetch(self, ident, dest_dir, *, credentials=None):
            d = dest_dir / "series"
            d.mkdir(parents=True, exist_ok=True)
            a = np.full((16, 24, 24), -1000, np.int16)
            a[4:12, 6:18, 6:18] = 40
            sitk.WriteImage(sitk.GetImageFromArray(a), str(d / "vol.nii.gz"))
            return d

    class SavingSeg(FakeSegmenter):
        def segment(self, image, task, *, progress=None, cancel=None, **options):
            self.calls.append((str(image), task, options))

            class R:
                def save(_, path):
                    a = np.zeros((16, 24, 24), np.uint8)
                    a[5:10, 8:16, 8:16] = 1
                    img = sitk.GetImageFromArray(a)
                    img.SetMetaData("Segment0_Name", "blob")
                    img.SetMetaData("Segment0_LabelValue", "1")
                    img.SetMetaData("Segment0_Color", "0.2 0.6 0.9")
                    sitk.WriteImage(img, str(path))
                    return path
                schema = type("S", (), {"names": {1: "blob"}})()
                def volumes_ml(_):
                    return {"blob": 1.0}
                provenance = {}
            return R()

    seg = SavingSeg()
    ex = LocalExecutor(seg, workdir=tmp_path, cache_dir=tmp_path / "rc",
                       sources=[DeepSource()])
    client = TestClient(create_app(ex))
    ident = "org1/repo@abc123.def/images/case7.nii.gz!inner/ct.nii.gz"
    url = f"/v1/deep/{ident}/total_fast/labels.seg.nrrd"

    r = client.get(url, headers={"Prefer": "wait=30"})   # blocking GET initiates
    assert r.status_code == 200, r.text
    assert client.head(url).status_code == 200           # probe sees it
    m = client.get(f"/v1/deep/{ident}/total_fast/meta.json")
    assert m.status_code == 200
    try:
        import matplotlib  # noqa: F401
        from test_serve import wait_artifact
        p = wait_artifact(client, f"/v1/deep/{ident}/total_fast/preview.png")
        assert p.status_code == 200 and p.content[:4] == b"\x89PNG"
    except ImportError:
        pass
    segs = client.get("/v1/segmentations").json()["segmentations"]
    e = next(x for x in segs if x["identity"] == [f"deep:{ident}"])
    assert e["links"]["labels"] == url                   # slashed path listed
    d = client.delete(url)
    assert d.status_code == 200 and d.json()["deleted"]
    assert client.head(url).status_code == 404           # gone
    assert len(seg.calls) == 1


def test_archive_cache_isolates_credentials(range_server, tmp_path):
    """Round 4 (HIGH): the archive cache keys on (outer, credentials), so a
    caller with no token cannot reuse an earlier caller's authenticated
    archive - resolve() (the access gate + token binding) runs for each."""
    url, size = range_server
    src = _Src(url, size)
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    src.fetch("archive.zip!case1/ct.nii.gz", tmp_path / "a", credentials="ALICE")
    src.fetch("archive.zip!case2/ct.nii.gz", tmp_path / "b", credentials=None)
    # both callers triggered a resolve(): the tokenless one was NOT served
    # from Alice's cached archive
    assert "ALICE" in src.seen_credentials and None in src.seen_credentials
    # same credential reuses the cached archive (one resolve for two members)
    before = len(src.seen_credentials)
    src.fetch("archive.zip!case1/seg.nii.gz", tmp_path / "c", credentials="ALICE")
    assert len(src.seen_credentials) == before        # cache hit, no re-resolve


def test_rangefile_refuses_status_200(tmp_path):
    """Round 4: a server that ignores Range and streams the whole body (200)
    is refused, not pulled into memory."""
    class _Ignore(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)                    # ignores Range
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"WHOLEBODY"[:8])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ignore)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rf = RangeFile(f"http://127.0.0.1:{srv.server_address[1]}/x", 8, block=4)
        with pytest.raises(InputError, match="Range"):
            rf.read(4)
    finally:
        srv.shutdown()


def test_dotdot_identifier_refused(range_server, tmp_path):
    """Round 4: a '..' segment in the OUTER identifier is refused (it would
    steer the resolve/template URL off the operator's prefix)."""
    url, size = range_server
    src = _Src(url, size)
    with pytest.raises(InputError, match=r"\.\."):
        src.fetch("../../secret/archive.zip!case1/ct.nii.gz", tmp_path)


def test_zenodo_access_fails_closed(monkeypatch):
    """Round 4: absent/empty access fields must read as restricted, not open."""
    import urllib.request

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=None):
        return FakeResp(json.dumps({"files": [{"key": "x.nii.gz", "size": 10,
                        "links": {"content": "http://h/x"}}]}).encode())

    from haversack import fetchlib
    monkeypatch.setattr(fetchlib, "urlopen", fake_open)
    z = ZenodoSource()                                 # allow_restricted=False
    with pytest.raises(InputError, match="undeclared access|restricted|only fetches"):
        z.resolve("1234567/x.nii.gz")


def test_redirect_strips_auth_on_scheme_downgrade():
    """Round-4 sign-off: the token is dropped on an https->http downgrade
    even on the same host (cleartext exposure), while an http->https upgrade
    keeps it - matching httpx."""
    from haversack import fetchlib
    opener = fetchlib._build_opener()
    (h,) = [x for x in opener.handlers
            if type(x).__name__ == "_StripCrossHostAuth"]
    import urllib.request

    def mk(url):
        r = urllib.request.Request(url, headers={"Authorization": "Bearer T"})
        return r

    # https -> http, same host: stripped
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "http://h/b")
    assert not any(k.lower() == "authorization" for k in new.headers)
    # http -> https, same host: kept (safe upgrade)
    new = h.redirect_request(mk("http://h/a"), None, 302, "", {},
                             "https://h/b")
    assert any(k.lower() == "authorization" for k in new.headers)
    # https -> https, same host: kept
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "https://h/b")
    assert any(k.lower() == "authorization" for k in new.headers)
    # cross host: stripped
    new = h.redirect_request(mk("https://h/a"), None, 302, "", {},
                             "https://evil/b")
    assert not any(k.lower() == "authorization" for k in new.headers)


# --- the two sources added 2026-09-05: a bucket allowlist and a release tag ---

class _FakeStore:
    """An obstore stand-in: objects by key, with the four calls the sources make."""

    def __init__(self, objects):
        self.objects = dict(objects)
        self.calls = []

    def list(self, prefix=None):
        self.calls.append(("list", prefix))
        yield [{"path": k, "size": len(v)} for k, v in self.objects.items()
               if k.startswith(prefix or "")]


class _FakeObstore:
    """The module-level obstore functions, against a _FakeStore."""

    class _Get:
        def __init__(self, data): self._d = data
        def bytes(self): return self._d
        def stream(self, min_chunk_size=1 << 20):
            for i in range(0, len(self._d), min_chunk_size):
                yield self._d[i:i + min_chunk_size]

    @staticmethod
    def head(store, key):
        if key not in store.objects:
            raise FileNotFoundError(key)
        return {"size": len(store.objects[key])}

    @staticmethod
    def get(store, key):
        store.calls.append(("get", key))
        return _FakeObstore._Get(store.objects[key])

    @staticmethod
    def get_range(store, key, start, end):
        store.calls.append(("get_range", key, start, end))
        return store.objects[key][start:end]


@pytest.fixture
def fake_cloud(monkeypatch):
    """Route _object_store to fakes and record (cloud, bucket, region) per call."""
    import sys
    import types
    from haversack import sources as srcmod
    built, stores = [], {}
    fake_mod = types.SimpleNamespace(head=_FakeObstore.head, get=_FakeObstore.get,
                                     get_range=_FakeObstore.get_range)
    monkeypatch.setitem(sys.modules, "obstore", fake_mod)

    def factory(cloud, bucket, region=None):
        built.append((cloud, bucket, region))
        return stores.setdefault((cloud, bucket), _FakeStore({}))

    monkeypatch.setattr(srcmod, "_object_store", factory)
    return built, stores


def test_s3_reads_only_from_the_allowlist_and_passes_each_buckets_region(fake_cloud):
    """The bucket is the operator's choice, the key is the client's. An
    unlisted bucket is refused by name before any store is built, and the
    per-bucket region reaches the store (msd-for-monai lives in us-west-2)."""
    built, stores = fake_cloud
    stores[("aws", "fcp-indi")] = _FakeStore({"data/x.nii.gz": b"x" * 123})
    stores[("aws", "msd-for-monai")] = _FakeStore({"Task09_Spleen.tar": b"t"})
    src = S3Source()
    assert src.resolve("fcp-indi/data/x.nii.gz") == ("s3://fcp-indi/data/x.nii.gz", 123)
    assert built[-1] == ("aws", "fcp-indi", None)
    src.resolve("msd-for-monai/Task09_Spleen.tar")
    assert built[-1] == ("aws", "msd-for-monai", "us-west-2")
    with pytest.raises(InputError, match="not one this server fetches from"):
        src.resolve("some-private-bucket/secret.nii.gz")
    assert not any(b == "some-private-bucket" for _, b, _ in built)
    # an operator may serve a different set; the identifier still cannot add one
    stores[("aws", "my-bucket")] = _FakeStore({"a.nii.gz": b"a"})
    assert S3Source({"my-bucket": None}).resolve("my-bucket/a.nii.gz")[0] == "s3://my-bucket/a.nii.gz"


def test_an_object_streams_down_through_the_store(fake_cloud, tmp_path):
    built, stores = fake_cloud
    stores[("aws", "fcp-indi")] = _FakeStore({"data/x.nii.gz": b"y" * (3 << 20)})
    got = S3Source().fetch("fcp-indi/data/x.nii.gz", tmp_path)
    assert (got / "x.nii.gz").read_bytes() == b"y" * (3 << 20)
    assert ("get", "data/x.nii.gz") in stores[("aws", "fcp-indi")].calls


def test_a_trailing_slash_fetches_every_object_under_the_prefix(fake_cloud, tmp_path):
    """The idc: mechanism for any allowlisted bucket: a DICOM series laid out in
    a bucket comes down in parallel, by basename, pseudo-directory keys skipped."""
    built, stores = fake_cloud
    stores[("gcp", "idc-open-data")] = _FakeStore({
        "abc/": b"", "abc/1.dcm": b"one", "abc/2.dcm": b"two", "abd/3.dcm": b"not mine"})
    got = GCSSource().fetch("idc-open-data/abc/", tmp_path)
    assert sorted(p.name for p in got.iterdir()) == ["1.dcm", "2.dcm"]
    assert (got / "2.dcm").read_bytes() == b"two"
    assert built[-1] == ("gcp", "idc-open-data", None)
    (tmp_path / "b").mkdir()
    with pytest.raises(InputError, match="no objects under"):
        GCSSource().fetch("idc-open-data/zzz/", tmp_path / "b")
    with pytest.raises(InputError, match="takes no !member"):
        GCSSource().fetch("idc-open-data/abc/!x", tmp_path / "b")


def test_a_prefix_over_the_cap_is_refused_before_any_object_moves(fake_cloud, tmp_path, monkeypatch):
    from haversack import sources as srcmod
    built, stores = fake_cloud
    store = stores[("aws", "fcp-indi")] = _FakeStore({"big/a": b"x" * 600, "big/b": b"x" * 600})
    monkeypatch.setattr(srcmod, "MAX_FETCH_BYTES", 1000)
    with pytest.raises(InputError, match="over the 1000-byte fetch cap"):
        S3Source().fetch("fcp-indi/big/", tmp_path)
    assert not any(c[0] == "get" for c in store.calls), "bytes moved before the cap was checked"


def test_a_zip_member_is_read_by_ranged_reads_on_the_store(fake_cloud, tmp_path):
    """The same remote-zip extraction as the HTTP sources, with the store's
    get_range under RangeFile instead of an HTTP Range request."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("inner/scan.nii.gz", b"\x1f\x8b" + b"z" * 5000)
        z.writestr("inner/other.txt", b"o")
    built, stores = fake_cloud
    store = stores[("aws", "fcp-indi")] = _FakeStore({"data/a.zip": buf.getvalue()})
    got = S3Source().fetch("fcp-indi/data/a.zip!inner/scan.nii.gz", tmp_path)
    assert (got / "scan.nii.gz").read_bytes() == b"\x1f\x8b" + b"z" * 5000
    ranged = [c for c in store.calls if c[0] == "get_range"]
    assert ranged and not any(c[0] == "get" for c in store.calls), "the whole zip was pulled"


def test_gs_is_a_served_source_with_its_own_allowlist_and_spelling_hint():
    from haversack.sources import check_identifier
    reg = registry()
    assert "gs" in reg and reg["gs"].enabled() in (True, False)
    assert reg["gs"].describe()["buckets"] == ["idc-open-data"]
    with pytest.raises(InputError, match="drop the slashes.*gs:idc-open-data/x"):
        check_identifier(GCSSource(), "//idc-open-data/x")
    with pytest.raises(InputError, match="not one this server fetches from"):
        GCSSource().check("fcp-indi/x.nii.gz")


def test_idc_prefers_the_mirror_when_asked_and_falls_back_to_aws(fake_cloud, tmp_path, monkeypatch):
    """The GCS mirror holds the main bucket only, so gcp means prefer, not only:
    a series that lives in -two is still found, on AWS, after the mirror miss."""
    from haversack.sources import IDCSource
    built, stores = fake_cloud
    uuid = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
    stores[("aws", "idc-open-data-two")] = _FakeStore({f"{uuid}/1.dcm": b"d"})
    monkeypatch.setenv("HAVERSACK_IDC_CLOUD", "gcp")
    assert IDCSource.probe_order() == [("gcp", "idc-open-data"), ("aws", "idc-open-data"),
                                       ("aws", "idc-open-data-two"), ("aws", "idc-open-data-cr")]
    got = IDCSource().fetch(uuid, tmp_path)
    assert (got / "1.dcm").read_bytes() == b"d"
    assert [(c, b) for c, b, _ in built] == [("gcp", "idc-open-data"), ("aws", "idc-open-data"),
                                             ("aws", "idc-open-data-two")]
    monkeypatch.setenv("HAVERSACK_IDC_CLOUD", "aws")
    assert IDCSource.probe_order()[0] == ("aws", "idc-open-data")
    monkeypatch.setenv("HAVERSACK_IDC_CLOUD", "azure")
    with pytest.raises(InputError, match="choose aws or gcp"):
        IDCSource.probe_order()
    monkeypatch.setenv("HAVERSACK_IDC_CLOUD", "gcp")
    (tmp_path / "x").mkdir()
    with pytest.raises(InputError, match="gs://idc-open-data.*s3://idc-open-data-cr"):
        IDCSource().fetch("0be27d1c-0000-0000-0000-000000000000", tmp_path / "x")


def test_github_source_reads_one_member_of_a_release_asset(range_server, tmp_path):
    url, _size = range_server
    port = url.rsplit(":", 1)[1].split("/")[0]

    class _Local(GitHubReleaseSource):
        HOST = f"http://127.0.0.1:{port}"

    src = _Local()
    # the URL is assembled from owner/repo@tag/asset, and stays the stable
    # github.com one - the signed CDN URL a HEAD redirects to expires
    resolved, size = src.resolve("owner/repo@v1.0.0/archive.zip")
    assert resolved == f"http://127.0.0.1:{port}/owner/repo/releases/download/v1.0.0/archive.zip"
    assert size > 0
    d = src.fetch("owner/repo@v1.0.0/archive.zip!case1/ct.nii.gz", tmp_path)
    assert (d / "ct.nii.gz").read_bytes().startswith(b"\x1f\x8bCT")
    with pytest.raises(InputError, match="no asset name"):
        src.resolve("owner/repo@v1.0.0")


def test_new_source_grammars_pin_the_identity():
    import re
    reg = registry(None)
    assert {"s3", "github"} <= set(reg)
    s3 = reg["s3"].id_pattern
    assert re.fullmatch(s3, "fcp-indi/data/Projects/ABIDE/sub-01_T1w.nii.gz")
    assert re.fullmatch(s3, "openneuro.org/ds000114/x.zip!inner/a.nii.gz")
    assert not re.fullmatch(s3, "UPPER/x.nii.gz")            # bucket syntax is AWS's
    assert not re.fullmatch(s3, "bucket")                    # a bucket alone is not an object
    g = reg["github"].id_pattern
    assert re.fullmatch(g, "robert-graf/VibeSegmentator@v1.0.0/100.zip")
    assert re.fullmatch(g, "o/r@v1.0.0/a.zip!inner/b.nii.gz")
    assert not re.fullmatch(g, "o/r/a.zip")                  # a tag is required
    # `..` must be refused by the dot-dot lookahead, not merely by the leading
    # character class - so the cases here are ones that START with a real
    # segment. `bucket/../x` would be rejected either way and proves nothing.
    for bad in ("fcp-indi/a/../../etc/passwd", "fcp-indi/a/..", "fcp-indi/../x"):
        assert not re.fullmatch(s3, bad), bad
    for bad in ("o/r@v1.0.0/a/../../x.zip", "o/../r@v1.0.0/a.zip"):
        assert not re.fullmatch(g, bad), bad


def test_the_anonymous_sources_refuse_a_credential_rather_than_sending_it():
    """`s3` and `github` read public data only. A token must not reach either:
    S3 answers one with 400, github.com redirects to a host that rejects it, and
    a private asset fetched with a caller's token would be cached where every
    reader of the cache can ask for it."""
    for src in (S3Source(), GitHubReleaseSource()):
        assert src._headers() == {}
        with pytest.raises(InputError, match="no credentials"):
            src._headers("SEKRET")
        with pytest.raises(InputError, match="no credentials"):
            src.resolve("fcp-indi/x.nii.gz" if src.prefix == "s3" else "o/r@v1/x.zip",
                        credentials="SEKRET")


def test_the_grammar_is_enforced_even_for_a_source_with_a_do_nothing_check():
    """`check_identifier` deferred to a source's own `check` when it had one -
    and a stand-in's attributes are all callable, so a Mock skipped validation
    entirely. The pattern is checked here, always, and the hook runs after."""
    from unittest import mock
    from haversack.sources import check_identifier

    stub = mock.Mock()
    stub.prefix = "tst"
    stub.id_pattern = r"[a-z]+"
    stub.description = "letters only"
    check_identifier(stub, "abc")                      # accepted, and the hook ran
    assert stub.check.called
    with pytest.raises(InputError, match="not a valid tst identifier"):
        check_identifier(stub, "NOT-LETTERS")

    class _Plain:                                      # no check at all
        prefix, id_pattern, description = "old", r"[0-9]+", "digits"

    check_identifier(_Plain(), "123")
    with pytest.raises(InputError):
        check_identifier(_Plain(), "abc")


def test_the_canonical_s3_url_spelling_is_answered_with_the_one_this_takes():
    """`s3://bucket/key` is what every AWS tool prints. The identifier IS the
    cache key, so accepting both spellings would split it - the refusal names the
    one this source takes instead of a bare grammar error."""
    from haversack.sources import S3Source, check_identifier
    with pytest.raises(InputError, match="drop the slashes.*s3:fcp-indi/x.nii.gz"):
        check_identifier(S3Source(), "//fcp-indi/x.nii.gz")


# ---------------------------------------------------------------------------
# Flattening: one rule, three call sites
# ---------------------------------------------------------------------------

#: Members that break a disambiguate-on-collision flattener. The first two share a
#: basename, so such a flattener invents `IM-1.dcm` for the second - and never
#: registers it, so the third member, genuinely named `IM-1.dcm`, overwrites it.
#: Three members, two files, one slice of a series silently gone. Order does not
#: save it: putting the real `IM-1.dcm` first only changes which file is lost.
COLLIDING = ["case/a/IM.dcm", "case/b/IM.dcm", "case/IM-1.dcm"]
COLLIDING_BYTES = [b"first", b"second", b"third"]


def _zip_of(names, payloads):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, p in zip(names, payloads):
            z.writestr(n, p)
    return buf.getvalue()


class TestOneFlatteningRule:
    """Every path that flattens names into one directory must keep every member.

    Three call sites had three answers, and two of them lost files: the object
    prefix fetch wrote every key to its basename with no dedup at all (through a
    32-thread pool, so two objects sharing a name interleaved into one inode and
    the survivor was neither), and the archive extractor disambiguated only on
    collision. `content.extract_zip` was the one that had it right, and its comment
    described precisely the bug in the other two. They share its helper now, and
    this exercises all three against the same adversarial member list.
    """

    def test_the_rule_itself_keeps_every_member_distinct(self):
        from haversack.content import flatten_names
        names = flatten_names(COLLIDING)
        assert len(set(names)) == len(COLLIDING), names
        # and indexing is all-or-nothing, so no synthesized name can equal a kept one
        assert all("_" in n for n in names), names

    def test_names_that_do_not_collide_are_left_alone(self):
        """A one-object prefix fetch should land as `scan.nii.gz`, not `0_scan.nii.gz`."""
        from haversack.content import flatten_names
        assert flatten_names(["a/scan.nii.gz", "b/mask.nii.gz"]) == ["scan.nii.gz", "mask.nii.gz"]

    def test_a_pseudo_directory_key_has_no_basename(self):
        """`abc/` is a bucket's directory marker and must not become a file called
        `abc`. pathlib normalizes the trailing slash away, which is why this is
        string slicing - caught by the prefix-fetch test when it did use pathlib."""
        from haversack.content import basename
        assert basename("abc/") == ""
        assert basename("a/b/c.dcm") == "c.dcm"
        assert basename("c.dcm") == "c.dcm"

    def test_the_content_store_keeps_every_member(self, tmp_path):
        from haversack.content import extract_zip
        src = tmp_path / "a.zip"
        src.write_bytes(_zip_of(COLLIDING, COLLIDING_BYTES))
        written = extract_zip(src, tmp_path / "out")
        assert len(written) == 3
        assert sorted(p.read_bytes() for p in written) == sorted(COLLIDING_BYTES)

    def test_the_archive_extractor_keeps_every_member(self, fake_cloud, tmp_path):
        built, stores = fake_cloud
        stores[("aws", "fcp-indi")] = _FakeStore(
            {"data/a.zip": _zip_of(COLLIDING, COLLIDING_BYTES)})
        got = S3Source().fetch("fcp-indi/data/a.zip!case/", tmp_path)
        files = sorted(got.iterdir())
        assert len(files) == 3, [p.name for p in files]
        assert sorted(p.read_bytes() for p in files) == sorted(COLLIDING_BYTES)

    def test_the_prefix_fetch_keeps_every_object(self, fake_cloud, tmp_path):
        built, stores = fake_cloud
        stores[("gcp", "idc-open-data")] = _FakeStore(
            {"case/": b"", **dict(zip(COLLIDING, COLLIDING_BYTES))})
        got = GCSSource().fetch("idc-open-data/case/", tmp_path)
        files = sorted(got.iterdir())
        assert len(files) == 3, [p.name for p in files]
        assert sorted(p.read_bytes() for p in files) == sorted(COLLIDING_BYTES)

    def test_a_prefix_fetch_counts_bytes_as_they_land(self, fake_cloud, tmp_path, monkeypatch):
        """The listing's sizes are checked first, but they are the listing's claim.
        A store that understates them must still hit the ceiling while streaming -
        and the ceiling is shared across the thread pool, or a prefix fetch is
        bounded at `threads * cap` rather than at `cap`."""
        from haversack import sources as S

        class _Liar(_FakeStore):
            def list(self, prefix=None):
                self.calls.append(("list", prefix))
                yield [{"path": k, "size": 1} for k in self.objects]   # every size a lie

        built, stores = fake_cloud
        stores[("gcp", "idc-open-data")] = _Liar(
            {f"case/{i}.dcm": b"x" * 4096 for i in range(8)})
        monkeypatch.setattr(S, "MAX_FETCH_BYTES", 5000)
        with pytest.raises(InputError, match="fetch cap"):
            GCSSource().fetch("idc-open-data/case/", tmp_path)


class TestTheArchiveCacheIsKeyedOnTheCredential:
    """Both `_zip` caches must key on `(outer, credentials)`, not on `outer`.

    On a cache hit the resolve/locate step is SKIPPED, and that step is the only
    place the access gate runs - the bucket allowlist here, the restricted-record
    refusal and the token binding in `ArchiveReadingSource`. Keying on `outer`
    alone let a tokenless caller reuse an earlier caller's authenticated archive.
    That was found once, fixed in `ArchiveReadingSource`, and written up in its
    docstring - while the same defect sat un-fixed in `ObjectStoreSource`, which
    keyed on `outer` until 2026-09-08. So this asserts it of BOTH, and of the
    cache key itself, because a class that grows a third archive cache will
    otherwise repeat it a third time.
    """

    def test_every_archive_cache_in_the_module_keys_on_the_credential(self):
        """Read as source, so a new `_zip` cannot quietly opt out."""
        import ast
        import inspect

        from haversack import sources as S
        tree = ast.parse(inspect.getsource(S))
        zips = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_zip"]
        assert len(zips) >= 2, "no _zip implementations found - has this been renamed?"
        caching = []
        for fn in zips:
            body = ast.unparse(fn)
            if "_archives" not in body:
                # an override that resolves and delegates (HttpSource) keeps no cache
                assert "super()._zip" in body, (
                    f"the _zip at line {fn.lineno} neither caches in `_archives` nor "
                    "delegates to super() - it has invented a third archive cache")
                continue
            caching.append(fn.lineno)
            assert "ck = (outer, credentials)" in body and "cache.get(ck)" in body, (
                f"the _zip at line {fn.lineno} does not key its archive cache on the "
                "credential; a cache hit skips the access gate")
        assert len(caching) >= 2, f"expected both archive caches, found {caching}"

    def test_a_cached_public_archive_does_not_admit_a_credentialed_request(
            self, fake_cloud, tmp_path):
        """The object store reads public buckets anonymously and REFUSES a
        credential. With the cache keyed on `outer` alone the refusal was skipped
        for any archive someone had already fetched."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("inner/scan.nii.gz", b"\x1f\x8b" + b"z" * 500)
        built, stores = fake_cloud
        stores[("aws", "fcp-indi")] = _FakeStore({"data/a.zip": buf.getvalue()})
        src = S3Source()
        src.fetch("fcp-indi/data/a.zip!inner/scan.nii.gz", tmp_path)       # warms the cache
        (tmp_path / "b").mkdir()
        with pytest.raises(InputError, match="takes no credentials"):
            src.fetch("fcp-indi/data/a.zip!inner/scan.nii.gz", tmp_path / "b",
                      credentials="someone-elses-token")
