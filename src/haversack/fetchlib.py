"""One door for every HTTP request haversack makes. Stdlib only.

There were three: a hand-built opener in ``sources`` that dropped
``Authorization`` on cross-host redirects, bare ``urlopen`` calls in the weights
installers that had just learned to name themselves, and a fifth spelling in the
engine checkpoint fetch that did neither. A fix to one was invisible to the
others - the User-Agent fix (``model.s.mdforge.com`` answers Python's default
agent with 403) landed on the weights path and left every hosted *source* still
sending ``Python-urllib/3.x``, which is the path a server takes on a client's
behalf. This module is where both properties live now, and
``tests/test_fetchlib.py`` fails if a ``urlopen`` grows anywhere else.

Why urllib and not httpx: the traffic is a dozen call sites of single large
streams, HEADs and Range reads - no pooling, no retries, no per-host
concurrency - and stdlib is what lets the lean install and the ``--no-project``
manifest generators run with nothing added. The generators cannot import this
package, so ``tools/zippeek.py`` carries the same two properties by hand.

The seam is :func:`urlopen`: tests replace it, and everything above it (the
named agent, the redirect rule) is on the ``Request`` or in the opener it wraps.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from urllib.parse import urlparse

from .errors import InputError


def user_agent() -> str:
    """What every request identifies itself as: ``haversack/<version>``.

    Python's default, ``Python-urllib/3.x``, is answered 403 by the Cloudflare
    rule in front of ``model.s.mdforge.com`` - the host of MOOSE's one asset that
    is not a GitHub release (``clin_ct_dental``, 2026-09-06) - while any name
    that says who is asking gets the bytes. Read at call time so the version is
    the package's own without a circular import.
    """
    from . import __version__
    return f"haversack/{__version__}"


def _origin_change(old_url: str, new_url: str) -> bool:
    a, b = urlparse(old_url), urlparse(new_url)
    if a.netloc != b.netloc:
        return True
    # an https->http downgrade on the SAME host still exposes a token in
    # cleartext; strip it (an http->https upgrade is safe and kept, which is
    # exactly what httpx does)
    return a.scheme == "https" and b.scheme == "http"


class _StripCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """urllib copies every header across a redirect except ``content-*``, so a
    per-request source token would otherwise follow a first-hop redirect to any
    host the response names - Zenodo and Hugging Face both redirect downloads
    to a CDN. Dropped on a change of origin, as httpx does."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin_change(req.full_url, newurl):
            for k in [h for h in new.headers if h.lower() == "authorization"]:
                del new.headers[k]
        return new


def _build_opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener(_StripCrossHostAuth)
    opener.addheaders = [("User-Agent", user_agent())]     # for a Request built elsewhere
    return opener


_opener: urllib.request.OpenerDirector | None = None


def request(url: str, *, method: str | None = None, headers: dict | None = None
            ) -> urllib.request.Request:
    """A ``Request`` that already names the client. Caller headers win, so a
    source that must send a different agent still can."""
    return urllib.request.Request(url, method=method,
                                  headers={"User-Agent": user_agent(), **(headers or {})})


def urlopen(req, timeout: float | None = None):
    """THE seam. Every request passes through here; tests replace this name."""
    global _opener
    if _opener is None:
        _opener = _build_opener()
    return _opener.open(req, timeout=timeout)


def open(url, *, timeout: float = 60, method: str | None = None,   # noqa: A001 - reads as fetchlib.open
         headers: dict | None = None):
    """Open ``url`` (a string, or a ready ``Request``) and return the response,
    a context manager. ``timeout`` is per call and deliberately has no "forever"
    default: a download that hangs must fail, not hold a claim on a series
    cache until someone notices."""
    req = url if isinstance(url, urllib.request.Request) else request(
        url, method=method, headers=headers)
    return urlopen(req, timeout=timeout)


def head_status(url: str, *, timeout: float = 60, headers: dict | None = None) -> int:
    """The HTTP status a HEAD gets. A 4xx/5xx is returned, not raised; a failure
    below HTTP (DNS, refused, timeout) still raises."""
    try:
        with open(url, method="HEAD", timeout=timeout, headers=headers) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def get_json(url: str, *, timeout: float = 60, headers: dict | None = None):
    import json
    with open(url, timeout=timeout, headers=headers) as r:
        return json.load(r)


def content_length(response) -> int:
    """The byte size a download will be, or 0 when the server did not say."""
    headers = getattr(response, "headers", None)
    try:
        return int(headers.get("Content-Length") or 0) if headers is not None else 0
    except (TypeError, ValueError, AttributeError):
        return 0


def copy_capped(src, dst, cap: int, what: str) -> int:
    """copyfileobj with a hard byte ceiling - a server that lies about
    Content-Length (or a bomb) cannot fill the disk."""
    n = 0
    while True:
        chunk = src.read(1 << 20)
        if not chunk:
            return n
        n += len(chunk)
        if n > cap:
            raise InputError(f"{what}: exceeded the {cap}-byte fetch cap")
        dst.write(chunk)
