"""A remote result is saved in the format its output name says (2026-09-23).

``RemoteClient.fetch`` asked the server for its native bytes whatever the name, so
``haversack remote submit ... -o labels.nii.gz`` wrote a ``.seg.nrrd`` under a NIfTI name and
every reader refused the file (found testing 0.13.0; the same in 0.12.4). The server converts
labels on request (``?format=nii.gz``), so the client now asks for it by name, and refuses a
``.nrrd`` / ``.zarr.zip`` name that does not match what the server sends, before writing
anything. A server double, driven through httpx's own mock transport: what is asked, what
lands on disk.
"""
from __future__ import annotations

import gzip

import pytest

httpx = pytest.importorskip("httpx")

from haversack.client import RemoteClient, RemoteError   # noqa: E402

NRRD = b"NRRD0004\n# a label map\n\n" + bytes(range(64))
NIFTI = b"\x5c\x01\x00\x00" + b"n+1\x00" + bytes(340)          # what a NIfTI header opens with
FIELD = b"PK\x03\x04" + bytes(60)                                 # a zip


def _client(kind="labels"):
    """A client whose server holds one finished job, ``j``, of ``kind``: labels or a field."""
    asked = []

    def handler(request):
        assert request.url.path == "/v1/jobs/j/result"
        fmt = request.url.params.get("format")
        asked.append(fmt)
        if kind == "field":
            if fmt is not None:
                return httpx.Response(422, text="an embedding field is served only as itself")
            return httpx.Response(200, content=FIELD, headers={"Content-Type": "application/zip"})
        if fmt in ("nii.gz", "nii"):     # the server gzips whichever of the two is asked
            return httpx.Response(200, content=gzip.compress(NIFTI),
                                  headers={"Content-Type": "application/gzip"})
        return httpx.Response(200, content=NRRD,
                              headers={"Content-Type": "application/octet-stream"})
    rc = RemoteClient("http://testserver")
    rc._http = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    return rc, asked


def test_a_nifti_gz_name_gets_nifti(tmp_path):
    rc, asked = _client()
    out = rc.fetch("j", tmp_path / "labels.nii.gz")
    assert asked == ["nii.gz"]
    assert gzip.decompress(out.read_bytes()) == NIFTI


def test_a_plain_nii_name_gets_it_uncompressed(tmp_path):
    rc, asked = _client()
    out = rc.fetch("j", tmp_path / "labels.nii")
    assert asked == ["nii.gz"] and out.read_bytes() == NIFTI


def test_the_suffix_is_read_without_regard_to_case(tmp_path):
    rc, asked = _client()
    rc.fetch("j", tmp_path / "LABELS.NII.GZ")
    assert asked == ["nii.gz"]


@pytest.mark.parametrize("name", ["labels.seg.nrrd", "labels.nrrd", "labels"])
def test_every_other_name_gets_the_servers_own_bytes(tmp_path, name):
    rc, asked = _client()
    out = rc.fetch("j", tmp_path / name)
    assert asked == [None] and out.read_bytes() == NRRD


def test_a_field_is_saved_as_a_field(tmp_path):
    rc, asked = _client("field")
    out = rc.fetch("j", tmp_path / "f.zarr.zip")
    assert asked == [None] and out.read_bytes() == FIELD


@pytest.mark.parametrize("kind,name,says", [
    ("field", "f.seg.nrrd", ".zarr.zip"),       # a field under a label map's name
    ("labels", "f.zarr.zip", ".seg.nrrd"),      # a label map under a field's name
    ("field", "f.nii.gz", "only as itself"),    # the server refuses to convert a field
])
def test_a_name_that_misstates_the_result_writes_nothing(tmp_path, kind, name, says):
    rc, _asked = _client(kind)
    with pytest.raises(RemoteError, match=says.replace(".", r"\.")):
        rc.fetch("j", tmp_path / name)
    assert list(tmp_path.iterdir()) == []            # no file, no .part


def test_a_nii_that_is_not_gzip_leaves_no_file(tmp_path):
    """The .nii path decompresses after the download; a body that is not gzip fails there,
    and leaves neither the download nor a half-written file behind."""
    def handler(request):
        return httpx.Response(200, content=NRRD, headers={"Content-Type": "application/gzip"})
    rc = RemoteClient("http://testserver")
    rc._http = httpx.Client(base_url="http://testserver", transport=httpx.MockTransport(handler))
    with pytest.raises(Exception):
        rc.fetch("j", tmp_path / "labels.nii")
    assert list(tmp_path.iterdir()) == []
