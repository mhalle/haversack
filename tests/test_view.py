"""`haversack view`: the preview page served beside a store, with Range for zips."""
from __future__ import annotations

import urllib.request
import zipfile

import pytest

from haversack.errors import InputError
from haversack.view import preview_page, serve_stores


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@pytest.fixture
def stores(tmp_path):
    d = tmp_path / "a.duckn"
    (d / "parts").mkdir(parents=True)
    (d / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    (d / "parts" / "c0").write_bytes(bytes(range(256)))
    z = tmp_path / "b.duckn.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("zarr.json", '{"zarr_format": 3, "node_type": "group"}')
        zf.writestr("blob", bytes(range(256)) * 4)
    return d, z


def test_the_page_ships_and_names_itself():
    assert "ranked store preview" in preview_page()


def test_stores_are_served_and_a_zip_takes_ranges(stores):
    d, z = stores
    server, url = serve_stores([d, z], port=0)
    try:
        code, h, body = _get(url)                              # redirect to the page + first store
        assert code == 200 and b"ranked store preview" in body
        code, h, body = _get(url + "stores/a.duckn/zarr.json")
        assert code == 200 and b'"zarr_format"' in body
        code, h, body = _get(url + "stores/a.duckn/parts/c0")
        assert code == 200 and body == bytes(range(256))
        assert _get(url + "stores/a.duckn/parts/missing")[0] == 404   # a missing chunk is a 404, never a page
        assert _get(url + "stores/a.duckn/../../etc/passwd")[0] == 404
        size = z.stat().st_size
        code, h, body = _get(url + "stores/b.duckn.zip", {"Range": "bytes=10-19"})
        assert code == 206 and body == z.read_bytes()[10:20]
        assert h["Content-Range"] == f"bytes 10-19/{size}" and h["Accept-Ranges"] == "bytes"
        code, h, body = _get(url + "stores/b.duckn.zip", {"Range": "bytes=-8"})
        assert code == 206 and body == z.read_bytes()[-8:]
        assert _get(url + "stores/b.duckn.zip", {"Range": f"bytes={size + 5}-"})[0] == 416
        code, h, body = _get(url + "stores/b.duckn.zip")
        assert code == 200 and len(body) == size
        assert _get(url + "stores/nope")[0] == 404 and _get(url + "anything")[0] == 404
    finally:
        server.shutdown()


def test_only_stores_are_served(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(InputError, match="not a ranked store"):
        serve_stores([tmp_path / "notes.txt"], port=0)
    with pytest.raises(InputError, match="not a ranked store"):
        serve_stores([tmp_path], port=0)


def test_view_is_a_command_but_not_in_the_help(capsys):
    from haversack import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["view", "--help"])
    assert e.value.code == 0 and "ranked stores" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert " view " not in capsys.readouterr().out
