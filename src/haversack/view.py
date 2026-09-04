"""``haversack view``: the slice preview of a ranked store, served locally.

UNDOCUMENTED, like the ranked store itself (see ranked_store). The page is sdfview's
``src/preview`` built into one file (``npm run build:preview`` there, copied to
``data/preview.html`` here): three orthogonal canvases drawn from the store's winner /
runner-up planes, with the zero level of the margin field as the outline. It reads the
store over HTTP, a directory store file by file or a zip by Range request - so this module
is a small static server that knows how to hand out both, and nothing else.
"""
from __future__ import annotations

import mimetypes
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from .errors import InputError

STORE_PREFIX = "/stores/"


def preview_page() -> str:
    """The single-file preview page shipped in the package."""
    from importlib.resources import files
    p = files("haversack").joinpath("data/preview.html")
    if not p.is_file():
        raise InputError("preview.html is missing from this installation "
                         "(sdfview: npm run build:preview, copied to haversack/src/haversack/data)")
    return p.read_text(encoding="utf-8")


def _is_store(p: Path) -> bool:
    from .ranked_store import is_zip
    if is_zip(p):
        return p.is_file()
    return p.is_dir() and (p / "zarr.json").is_file()


def _index_page(stores: dict) -> bytes:
    items = "".join(
        f'<li><a href="/preview.html?store={quote(STORE_PREFIX + n)}">{n}</a> '
        f'<small>{p}</small></li>' for n, p in stores.items())
    return (f"<!doctype html><meta charset=utf-8><title>haversack view</title>"
            f"<body style='font:14px system-ui;padding:1em 2em'><h2>ranked stores</h2><ul>{items}</ul>"
            ).encode("utf-8")


class _Handler(SimpleHTTPRequestHandler):
    """``/`` and ``/preview.html`` hand out the page (``/`` redirects to it with the first
    store filled in); ``/stores/<name>`` is a zip, with Range; ``/stores/<name>/<path>`` is a
    file inside a directory store. Anything else is 404 - a preview server has no fallback
    page, because a zarr reader probes for chunks that may not exist and must see the 404."""
    stores: dict[str, Path] = {}
    page: bytes = b""
    quiet = True
    protocol_version = "HTTP/1.1"       # keep-alive: a directory store is a few hundred files

    def log_message(self, *a):  # noqa: D102 - stdlib hook
        if not self.quiet:
            super().log_message(*a)

    def do_GET(self):  # noqa: N802 - stdlib hook
        self._serve(head=False)

    def do_HEAD(self):  # noqa: N802
        self._serve(head=True)

    def _send(self, code, body: bytes, ctype: str, extra=None, head=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _serve(self, *, head: bool):
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            first = next(iter(self.stores), None)
            target = "/preview.html" + (f"?store={quote(STORE_PREFIX + first)}" if first else "")
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/preview.html":
            return self._send(200, self.page, "text/html; charset=utf-8", head=head)
        if path in ("/stores", "/stores/"):
            return self._send(200, _index_page(self.stores), "text/html; charset=utf-8", head=head)
        if not path.startswith(STORE_PREFIX):
            return self._send(404, b"not found", "text/plain", head=head)
        rest = path[len(STORE_PREFIX):]
        name, _, inner = rest.partition("/")
        store = self.stores.get(name)
        if store is None:
            return self._send(404, b"no such store", "text/plain", head=head)
        if store.is_file():
            if inner:
                return self._send(404, b"a zip store has no members to serve", "text/plain", head=head)
            return self._file(store, "application/zip", head=head)
        # a directory store: files inside it, and nothing outside it. A path the OS
        # refuses (a NUL byte, a name past the length limit) is a 404 like any other,
        # not a dead connection.
        try:
            target = (store / inner).resolve() if inner else None
            ok = (target is not None and str(target).startswith(str(store.resolve()) + os.sep)
                  and target.is_file())
        except (OSError, ValueError):
            ok = False
        if not ok:
            return self._send(404, b"not found", "text/plain", head=head)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return self._file(target, ctype, head=head)

    def _file(self, p: Path, ctype: str, *, head: bool):
        size = p.stat().st_size
        rng = _byte_range(self.headers.get("Range"), size)
        if rng == "unsatisfiable":
            return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"}, head=head)
        start, end = rng if rng else (0, size - 1)
        with open(p, "rb") as f:
            f.seek(start)
            body = b"" if head else f.read(end - start + 1)
        extra = {"Accept-Ranges": "bytes"}
        if rng:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)


def _byte_range(header, size: int):
    """RFC 9110 §14: ``(start, end)`` for one satisfiable byte range; ``None`` when there
    is no Range header or it must be ignored (another unit, malformed, several ranges,
    ``start > end``); ``"unsatisfiable"`` for a well-formed range the file cannot serve."""
    if not header or not header.strip().lower().startswith("bytes="):
        return None
    spec = header.strip()[6:].strip()
    if "," in spec:
        return None                          # several ranges: serve the whole thing
    a, sep, b = spec.partition("-")
    if not sep:
        return None
    try:
        if a.strip():
            start = int(a)
            end = int(b) if b.strip() else None
            if start < 0 or (end is not None and start > end):
                return None                  # invalid byte-range-spec: the field is ignored
            if start >= size:
                return "unsatisfiable"
            return start, size - 1 if end is None else min(end, size - 1)
        if b.strip():                        # suffix range: the last N bytes
            n = int(b)
            if n <= 0 or size == 0:
                return "unsatisfiable"
            return max(0, size - n), size - 1
    except ValueError:
        return None
    return None


def _display_host(host: str) -> str:
    """The host to put in a URL: a wildcard bind is reached on loopback, IPv6 in brackets."""
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if ":" in host else host


def serve_stores(paths, *, host: str = "127.0.0.1", port: int = 0, quiet: bool = True):
    """Start the preview server for ``paths`` (ranked stores: directories or zips) on a
    background thread; returns ``(server, url)``. ``port=0`` picks a free one."""
    stores: dict[str, Path] = {}
    for raw in paths:
        p = Path(raw).expanduser().resolve()     # `.` and `..` have no name until resolved
        if not _is_store(p):
            raise InputError(f"{p}: not a ranked store (a .duckn directory with zarr.json, or a .duckn.zip)")
        name = p.name
        if not name:
            raise InputError(f"{p}: a store needs a name (the filesystem root has none)")
        n = 2
        while name in stores:                    # two stores with one name: number the second
            name, n = f"{p.name}~{n}", n + 1
        stores[name] = p
    handler = type("PreviewHandler", (_Handler,),
                   {"stores": stores, "page": preview_page().encode("utf-8"), "quiet": quiet})
    import socket

    class _Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        def handle_error(self, request, client_address):
            # a browser that navigates away mid-load resets its connections; that is
            # not an error of ours, and a traceback per abandoned chunk is noise
            import sys as _sys
            exc = _sys.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                return
            if not quiet:
                super().handle_error(request, client_address)

    server_cls = _Server
    try:
        server = server_cls((host, port), handler)
    except OSError as e:
        raise InputError(f"cannot listen on {host}:{port}: {e}") from None
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="haversack-view", daemon=True).start()
    url = f"http://{_display_host(host)}:{server.server_address[1]}/"
    return server, url


def main_view(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="haversack view", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Serve the slice preview of one or more ranked stores (.duckn directories or "
                    ".duckn.zip files) on this machine and open it in the browser.")
    ap.add_argument("store", nargs="+", help="ranked store(s)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on. Anything but loopback serves the stores to that "
                         "network with NO authentication")
    ap.add_argument("--port", type=int, default=8795, help="0 picks a free port")
    ap.add_argument("--no-browser", action="store_true", help="print the URL only")
    ap.add_argument("--quiet", action="store_true", help="no request log")
    args = ap.parse_args(argv)
    server, url = serve_stores(args.store, host=args.host, port=args.port, quiet=args.quiet)
    names = list(server.RequestHandlerClass.stores)
    print(f"haversack view: {url}  ({len(names)} store{'s' if len(names) != 1 else ''}; "
          "Ctrl-C to stop)", flush=True)
    for n in names:
        print(f"  {url}preview.html?store={quote(STORE_PREFIX + n)}", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  note: bound to {args.host} - reachable from that network, with no authentication",
              file=sys.stderr, flush=True)
    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()               # shutdown alone leaves the port bound
    return 0
