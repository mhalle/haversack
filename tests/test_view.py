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
        code, h, body = _get(url + "stores")                    # the index: every store, linked
        assert code == 200 and b"a.duckn" in body and b"b.duckn.zip" in body
        # a Range the server does not speak is ignored, not answered as partial content
        code, h, body = _get(url + "stores/b.duckn.zip", {"Range": "items=0-1"})
        assert code == 200 and len(body) == size and "Content-Range" not in h
        code, h, body = _get(url + "stores/b.duckn.zip", {"Range": "bytes=5-3"})
        assert code == 200 and len(body) == size
        code, h, body = _get(url + "stores/b.duckn.zip", {"Range": "bytes=0-1,5-6"})
        assert code == 200 and len(body) == size
        assert _get(url + "stores/b.duckn.zip", {"Range": "bytes=-0"})[0] == 416
        # paths the OS refuses are 404s, not dropped connections
        assert _get(url + "stores/a.duckn/%00")[0] == 404
        assert _get(url + "stores/a.duckn/" + "x" * 5000)[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_store_given_as_dot_gets_its_real_name(stores, monkeypatch):
    d, _ = stores
    monkeypatch.chdir(d)
    server, url = serve_stores(["."], port=0)
    try:
        assert list(server.RequestHandlerClass.stores) == ["a.duckn"]
        assert _get(url + "stores/a.duckn/zarr.json")[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_the_port_is_free_again_after_close(stores):
    d, _ = stores
    server, url = serve_stores([d], port=0)
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    again, _ = serve_stores([d], port=port)                    # would fail while the socket lingers
    again.shutdown()
    again.server_close()


def test_display_hosts():
    from haversack.view import _byte_range, _display_host
    assert _display_host("0.0.0.0") == "127.0.0.1" and _display_host("::") == "[::1]"
    assert _display_host("::1") == "[::1]" and _display_host("gpu-box") == "gpu-box"
    assert _byte_range("bytes=0-0", 10) == (0, 0) and _byte_range("bytes=-3", 10) == (7, 9)
    assert _byte_range("bytes=5-", 10) == (5, 9) and _byte_range("bytes=5-99", 10) == (5, 9)
    assert _byte_range("bytes=10-", 10) == "unsatisfiable" and _byte_range("bytes=-0", 10) == "unsatisfiable"
    assert _byte_range("bytes=5-3", 10) is None and _byte_range("items=0-1", 10) is None
    assert _byte_range("bytes=a-b", 10) is None and _byte_range(None, 10) is None


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
