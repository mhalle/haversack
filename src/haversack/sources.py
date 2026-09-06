"""Pluggable remote data repositories the server can fetch inputs from.

The IDC methodology - a URL path prefix, a strict identifier pattern, and a
fetch that materializes the identified series locally - generalizes to any
repository. A :class:`DataSource` packages those three things, and everything
else (the series cache, the prefetch pipeline, the path surface, result-cache
identity) is source-agnostic plumbing over the registry.

Two ways to add a repository:

- **Programmatically**: subclass :class:`DataSource` when fetching needs code
  (bucket probing, manifest walks, auth handshakes).
- **Data only**: instantiate :class:`UrlTemplateSource` for the simplest case -
  one anonymously fetchable file per identifier at a templated URL.

Identifiers are the *whole* reference: every source declares a fullmatch
regex, and nothing here accepts a client-supplied URL. That is what keeps the
fetch path SSRF-free, so keep patterns strict when adding sources.

Identity strings are ``"<prefix>:<identifier>"`` - the result-cache key
component - so the ``idc`` source reproduces the established ``idc:<uuid>``
identities byte for byte.
"""
import os
import re
import urllib.error
from pathlib import Path

from . import fetchlib
from .errors import InputError

__all__ = ["DataSource", "UrlTemplateSource", "IDCSource", "GitHubReleaseSource",
           "S3Source", "default_sources", "IDC_BUCKETS", "PUBLIC_S3_BUCKETS", "CRDC_RE"]

CRDC_RE = r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"

# The three public IDC buckets, probed in order (idc-open-data holds 99.5 % of
# series; -two and -cr the rest - found the hard way 2026-08-24). The clean
# upgrade path is resolving per series via idc-index (`series_aws_url`), which we
# take when /v1/resolve lands.
IDC_BUCKETS = ("idc-open-data", "idc-open-data-two", "idc-open-data-cr")


# The S3 buckets a server may be pointed at, and the region each answers in
# (None = the global path-style endpoint). An ALLOWLIST, not a parameter the
# identifier carries: "fetch s3://<whatever the client says>" is the SSRF hole
# this module exists to avoid, so the bucket is chosen by the operator and the
# identifier only picks a key inside one. Buckets here are public, requester-pays
# free, and anonymously readable - checked by a range GET, 2026-09-05.
PUBLIC_S3_BUCKETS = {
    "fcp-indi": None,               # INDI / Preprocessed Connectomes (ABIDE, ADHD-200, CoRR, NKI-RS)
    "openneuro.org": None,          # OpenNeuro's own bucket (also the openneuro: source's backing)
    "msd-for-monai": "us-west-2",   # the Medical Segmentation Decathlon mirror
    "idc-open-data": None,          # IDC's buckets, for reaching one object rather than a series
    "idc-open-data-two": None,
    "idc-open-data-cr": None,
}


class DataSource:
    """One remote repository: a path prefix, an identifier pattern, a fetch.

    ``prefix`` names the source everywhere: the ``source`` kind at submit,
    the path surface (``/v1/<prefix>/<id>/<task>/labels.seg.nrrd``), the
    series-cache key namespace, and the identity string. Lowercase letters,
    digits, and ``_`` only, so it stays a clean URL segment.

    ``id_pattern`` is fullmatched against every identifier before anything
    else happens - it is the input-validation boundary, keep it strict.

    ``fetch(identifier, dest_dir)`` materializes the input under ``dest_dir``
    and returns the path the pipeline should read (a file, or a directory for
    a DICOM series). It runs on the worker, possibly on a prefetch thread.
    """

    prefix: str = ""
    id_pattern: str = ""
    description: str = ""

    def enabled(self) -> bool:
        """Whether this source can fetch on this install (dependencies etc.)."""
        return True

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        """Materialize the input. ``credentials`` is an optional per-request
        secret (e.g. a bearer token) - a credential in transit: never store,
        log, or record it anywhere durable."""
        raise NotImplementedError

    def check(self, identifier: str, credentials=None) -> None:
        """Refuse what this source cannot serve, doing NO I/O.

        Every door calls this - the CLI before it fetches, the server at submit -
        so a request that was always going to fail is refused where the caller is
        still listening, instead of minutes later inside a worker (which on Modal
        means a GPU container was started for it). Anything decidable without a
        network round trip belongs here: the identifier grammar, an allowlist, a
        credential this source does not accept. Anything needing the network
        (does the object exist?) belongs in :meth:`fetch`.
        """
        if self.id_pattern and not re.fullmatch(self.id_pattern, identifier):
            raise InputError(f"{self.prefix}:{identifier} is not a valid {self.prefix} "
                             f"identifier - {self.description or self.id_pattern}")

    def identity(self, identifier: str) -> str:
        """The result-cache identity token for one identifier."""
        return f"{self.prefix}:{identifier}"

    def describe(self) -> dict:
        return {"prefix": self.prefix, "id_pattern": self.id_pattern,
                "enabled": self.enabled(), "description": self.description}


class UrlTemplateSource(DataSource):
    """The simplest possible source, defined by data alone: one anonymously
    fetchable file per identifier at ``url_template.format(id=identifier)``.

    The template is fixed at construction and the identifier is validated
    against ``id_pattern`` before substitution, so clients still cannot steer
    the URL anywhere the operator did not choose.
    """

    def __init__(self, prefix: str, id_pattern: str, url_template: str, *,
                 filename: str | None = None, description: str = ""):
        if "{id}" not in url_template:
            raise ValueError("url_template needs an {id} placeholder")
        self.prefix, self.id_pattern = prefix, id_pattern
        self.url_template, self.filename = url_template, filename
        self.description = description or f"single-file fetch from {url_template}"

    def _filename(self, identifier: str) -> str:
        """The saved name defaults to the identifier's basename so the format
        stays detectable (.nii.gz etc.); explicit ``filename`` overrides."""
        if self.filename:
            return self.filename
        name = Path(identifier).name
        return name if name and name not in (".", "..") else "image"

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        if _has_dotdot(identifier):
            raise InputError(f"{self.prefix}:{identifier}: '..' path segment refused")
        url = self.url_template.format(id=identifier)
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        out = dest / self._filename(identifier)
        try:
            with fetchlib.open(url, timeout=300) as r, open(out, "wb") as f:
                fetchlib.copy_capped(r, f, MAX_FETCH_BYTES, f"{self.prefix}:{identifier}")
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
        return dest


class IDCSource(DataSource):
    """NCI Imaging Data Commons: DICOM series by ``crdc_series_uuid`` from the
    public open-data buckets, anonymously, 32 threads (obstore beat s5cmd in
    every measured quadrant). The uuid names a version-pinned series; the
    bucket prefix is probed across the three known buckets rather than
    assumed."""

    prefix = "idc"
    id_pattern = CRDC_RE
    description = "NCI Imaging Data Commons, by crdc_series_uuid"

    def enabled(self) -> bool:
        try:
            import obstore  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        from concurrent.futures import ThreadPoolExecutor

        from obstore.store import S3Store
        keys, store = [], None
        for bucket in IDC_BUCKETS:
            store = S3Store.from_url(f"s3://{bucket}", config={"aws_skip_signature": "true"})
            keys = [(o.get("path") if isinstance(o, dict) else str(o))
                    for b in store.list(prefix=f"{identifier}/") for o in b]
            if keys:
                break
        if not keys:
            raise InputError(f"no objects under {identifier!r}/ in any probed IDC bucket "
                             f"({', '.join(IDC_BUCKETS)}); if the series exists, IDC may "
                             "have added a bucket this server does not know")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)

        def one(k):
            base = k.rsplit("/", 1)[-1]
            if not base or base in (".", ".."):
                return                     # bucket pseudo-dir key; skip
            with open(dest / base, "wb") as f:
                f.write(bytes(store.get(k).bytes()))

        try:
            with ThreadPoolExecutor(32) as ex:
                list(ex.map(one, keys))
        except Exception as e:
            raise InputError(f"fetch of idc:{identifier} failed: {e}") from e
        return dest


class TCIASource(DataSource):
    """The Cancer Imaging Archive via its NBIA REST API: a DICOM series by
    SeriesInstanceUID, anonymously, for fully public collections. The API
    returns one zip per series; entries are flattened by basename on
    extraction, which also makes zip-slip impossible by construction.

    NOT version-pinned: TCIA serves the collection's current revision, so the
    same SeriesInstanceUID can resolve to different bytes across data
    releases. For version-pinned identity use the idc source - the two doors
    deliberately have distinct identities (tcia:<uid> vs idc:<uuid>)."""

    prefix = "tcia"
    id_pattern = r"(?=.{10,64}$)[0-9]+(?:\.[0-9]+)+"   # DICOM UID: digits, dots, <=64
    description = "The Cancer Imaging Archive (NBIA), by SeriesInstanceUID"
    API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        import shutil
        import zipfile
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        tmp = Path(dest_dir) / "series.zip"
        try:
            with fetchlib.open(f"{self.API}?SeriesInstanceUID={identifier}",
                               timeout=600) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            n = 0
            with zipfile.ZipFile(tmp) as z:
                if sum(zi.file_size for zi in z.infolist()) > MAX_FETCH_BYTES:
                    raise InputError(f"tcia:{identifier}: series exceeds the "
                                     f"{MAX_FETCH_BYTES}-byte cap")
                seen: dict = {}
                for m in z.infolist():
                    name = Path(m.filename).name   # flatten: zip paths never touch disk
                    if m.is_dir() or not name or name.startswith("."):
                        continue
                    if name in seen:
                        seen[name] += 1
                        stem, dot, ext = name.partition(".")
                        name = f"{stem}-{seen[name]}{dot}{ext}"
                    else:
                        seen[name] = 0
                    with z.open(m) as src, open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    n += 1
            if n == 0:
                raise InputError(f"TCIA returned an empty series for {identifier!r}")
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of tcia:{identifier} failed: {e}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return dest


def openneuro_source() -> UrlTemplateSource:
    """OpenNeuro (CC0 neuroimaging, BIDS layout) straight off its public S3
    bucket over HTTPS - the data-only case: identifiers are
    ``ds<number>/<path-to-file>`` (slashed, so cache keys hash; the path
    surface's greedy routes carry the slashes in ordinary URLs)."""
    return UrlTemplateSource(
        "openneuro",
        r"(?!.*(?:^|/)\.\.(?:/|$))ds[0-9]{6}/[A-Za-z0-9][A-Za-z0-9._/-]{0,200}",
        "https://s3.amazonaws.com/openneuro.org/{id}",
        description="OpenNeuro (CC0), by ds<number>/<file path>")


MAX_FETCH_BYTES = int(float(os.environ.get("HAVERSACK_MAX_FETCH_GB", "16")) * (1 << 30))


def _has_dotdot(identifier: str) -> bool:
    """True if any path segment is '..' - such an identifier steers a
    template/resolve URL off the operator's prefix (the remote or a proxy may
    apply remove_dot_segments). The id_patterns also exclude it; this is the
    belt to that suspenders, at the one place every fetch passes through."""
    return any(seg == ".." for seg in re.split(r"[/\\]", identifier))


class RangeFile:
    """A seekable read-only file over HTTP Range requests, with an LRU block
    cache. Handing one to :class:`zipfile.ZipFile` gives remote archives
    random access - zip64, deflate, and per-member CRC verification all come
    from the stdlib. Redirects are followed per request (CDN URLs expire)."""

    def __init__(self, url: str, size: int, *, headers=None, block: int = 1 << 22,
                 max_blocks: int = 64):
        import collections
        self.url, self.size, self.pos = url, int(size), 0
        self.headers = dict(headers or {})
        self.block_size, self.max_blocks = int(block), int(max_blocks)
        self._blocks = collections.OrderedDict()
        self.requests = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self._blocks:
            self._blocks.move_to_end(i)
            return self._blocks[i]
        lo = i * self.block_size
        hi = min(self.size, lo + self.block_size) - 1
        with fetchlib.open(self.url, timeout=300,
                           headers={**self.headers, "Range": f"bytes={lo}-{hi}"}) as r:
            if r.status != 206:
                # a 200 means the server ignored Range and is streaming the
                # WHOLE body - on a multi-GB archive that is an unbounded read
                # into memory; refuse rather than pull it
                raise InputError(
                    f"range request not honored (status {r.status}) by "
                    f"{self.url}; server does not support HTTP Range")
            data = r.read()
        want = hi - lo + 1
        if len(data) != want:
            raise InputError(f"short/over range read from {self.url}: got "
                             f"{len(data)} bytes, asked {want}")
        self._blocks[i] = data
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        self.requests += 1
        self.fetched += len(data)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block_size)
            chunk = self._block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


class ArchiveReadingSource(DataSource):
    """Base for hosts that serve files - and, via HTTP Range, individual
    members of zip archives - by identifier.

    The identifier grammar is ``<outer>[!<member>]``: without ``!`` the outer
    file downloads whole; with it, the named member (or every member under a
    trailing-slash prefix) is extracted from the remote zip while fetching
    only the bytes it occupies. Subclasses implement one method:
    ``resolve(outer, credentials) -> (url, size)``. Tar and parquet archives
    are out of scope - only zip has the trailing central directory that makes
    remote random access possible."""

    def resolve(self, outer: str, credentials=None) -> tuple:
        raise NotImplementedError

    def _headers(self, credentials=None) -> dict:
        return {"Authorization": f"Bearer {credentials}"} if credentials else {}

    def _zip(self, outer: str, credentials=None):
        """The parsed archive, cached per ``(outer, credentials)``. Keying on
        the credential is load-bearing, not an optimization: on a cache hit
        ``resolve`` is skipped, and ``resolve`` is the ONLY place the
        restricted-access gate runs and the token is bound to the archive's
        RangeFile. Caching by ``outer`` alone let a tokenless caller reuse an
        earlier caller's authenticated archive - reading gated content and
        replaying that caller's token upstream (round-4 finding)."""
        import zipfile
        cache = self.__dict__.setdefault("_archives", {})
        ck = (outer, credentials)
        z = cache.get(ck)
        if z is None:
            url, size = self.resolve(outer, credentials)
            z = zipfile.ZipFile(RangeFile(url, size, headers=self._headers(credentials)))
            cache[ck] = z
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return z

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        import shutil
        outer, _, member = identifier.partition("!")
        if _has_dotdot(outer):             # the outer id steers the URL; the
                                           # member is neutralized by flatten
            raise InputError(f"{self.prefix}:{identifier}: '..' path segment refused")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        try:
            if not member:                 # plain file: stream it down whole
                url, _size = self.resolve(outer, credentials)
                name = Path(outer).name
                if not name or name in (".", ".."):
                    name = "image"
                with fetchlib.open(url, timeout=1800, headers=self._headers(credentials)) as r, \
                        open(dest / name, "wb") as f:
                    fetchlib.copy_capped(r, f, MAX_FETCH_BYTES, f"{self.prefix}:{identifier}")
                return dest
            z = self._zip(outer, credentials)
            members = ([m for m in z.namelist()
                        if m.startswith(member) and not m.endswith("/")]
                       if member.endswith("/") else [member])
            if not members:
                raise InputError(f"{self.prefix}:{identifier}: no such member in archive")
            # refuse a decompression bomb by declared uncompressed size BEFORE
            # opening anything (the central directory is already parsed)
            want = set(members)
            total = sum(zi.file_size for zi in z.infolist() if zi.filename in want)
            if total > MAX_FETCH_BYTES:
                raise InputError(f"{self.prefix}:{identifier}: members total "
                                 f"{total} bytes, over the {MAX_FETCH_BYTES} cap")
            seen: dict = {}
            for m in members:
                name = Path(m).name        # flatten: archive paths never touch disk
                if not name or name.startswith("."):
                    continue
                if name in seen:           # basename collision: disambiguate
                    seen[name] += 1        # rather than silently overwrite
                    stem, dot, ext = name.partition(".")
                    name = f"{stem}-{seen[name]}{dot}{ext}"
                else:
                    seen[name] = 0
                with z.open(m) as src, open(dest / name, "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
        return dest


class ZenodoSource(ArchiveReadingSource):
    """Zenodo records: ``<recid>/<filename>[!member]``. A record id pins one
    published version (new versions mint new record ids; only the concept DOI
    floats), so identities are cache-grade. A personal access token raises
    rate limits and - when the operator opts in - unlocks restricted records;
    by default restricted records are refused, because cached results are
    readable by every cache reader regardless of who fetched the input."""

    prefix = "zenodo"
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[0-9]{4,9}/[A-Za-z0-9._-]+(?:![A-Za-z0-9._ /-]+)?")
    description = "Zenodo records, by record id / filename (!member for zip contents)"

    def __init__(self, *, allow_restricted: bool = False):
        self.allow_restricted = allow_restricted

    def resolve(self, outer: str, credentials=None) -> tuple:
        import json as _json
        recid, _, filename = outer.partition("/")
        with fetchlib.open(f"https://zenodo.org/api/records/{recid}", timeout=60,
                           headers=self._headers(credentials)) as r:
            rec = _json.load(r)
        access = ((rec.get("metadata") or {}).get("access_right")
                  or (rec.get("access") or {}).get("files") or "")
        # fail CLOSED: absent/empty access fields (a narrowed API response, a
        # schema move) must not read as "open"
        if access not in ("open", "public") and not self.allow_restricted:
            raise InputError(
                f"zenodo record {recid} is {access or 'of undeclared access'!r}; "
                "this server only fetches open "
                "records (operator opt-in required for restricted data - cached "
                "results are readable by every cache reader)")
        for f in rec.get("files", []):
            if f.get("key") == filename or f.get("filename") == filename:
                size = f.get("size") or f.get("filesize")
                url = (f.get("links", {}).get("content")
                       or f"https://zenodo.org/records/{recid}/files/{filename}?download=1")
                return url, int(size)
        raise InputError(f"zenodo record {recid} has no file {filename!r}")


class HuggingFaceSource(ArchiveReadingSource):
    """Hugging Face dataset repos: ``<org>/<name>@<commit-sha>/<path>[!member]``.

    The 40-hex commit sha is REQUIRED: branch names float (``main`` is a
    moving target), and a floating reference under a content-addressed cache
    is the stale-identity bug in waiting - the same doctrine as IDC's
    crdc_series_uuid and the task grammar's @version. Gated/private repos
    follow the same operator opt-in rule as Zenodo restricted records."""

    prefix = "hf"
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
                  r"@[0-9a-f]{40}/[A-Za-z0-9._/-]+(?:![A-Za-z0-9._ /-]+)?")
    description = "Hugging Face datasets, by org/name@commit-sha/path (!member for zips)"

    def __init__(self, *, allow_restricted: bool = False):
        self.allow_restricted = allow_restricted

    def resolve(self, outer: str, credentials=None) -> tuple:
        repo, _, rest = outer.partition("@")
        sha, _, path = rest.partition("/")
        url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}"
        try:
            with fetchlib.open(url, method="HEAD", timeout=60,
                               headers=self._headers(credentials)) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise InputError(
                    f"hf dataset {repo} is gated or private; "
                    + ("a valid token is required"
                       if self.allow_restricted else
                       "this server only fetches public repos (operator opt-in "
                       "required for gated data)")) from e
            raise
        if size <= 0:
            raise InputError(f"hf: could not size {url}")
        return url, size


class S3Source(ArchiveReadingSource):
    """Public S3 buckets, one object per identifier: ``<bucket>/<key>[!member]``.

    The bucket is checked against :data:`PUBLIC_S3_BUCKETS` before anything is
    fetched, so a client picks a key inside a bucket the *operator* chose and
    never the bucket itself - the same containment the other sources get from a
    fixed host. Reads go over anonymous HTTPS (path-style, which is what a
    dotted bucket name like ``openneuro.org`` needs: it cannot appear in a
    virtual-hosted name without breaking TLS), so this needs no obstore and no
    credentials, and inherits ``!member`` zip reading from the base.

    NOT version-pinned, like ``tcia`` and unlike ``idc``: a bucket key can be
    overwritten in place, so the same identity can resolve to different bytes
    across dataset releases. Whole DICOM *series* have their own doors (``idc``,
    ``tcia``) - this one addresses a single object.
    """

    prefix = "s3"
    # <bucket>/<key>[!member]. Bucket syntax is AWS's own (3-63 chars, lowercase
    # alphanumerics, dots and hyphens); membership in the allowlist is checked in
    # resolve(), because a rejected bucket deserves a message naming the ones served.
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[A-Za-z0-9][A-Za-z0-9._/-]{0,300}"
                  r"(?:![A-Za-z0-9._ /-]+)?")
    description = "public S3 buckets, by bucket/key (!member for zip contents)"

    def __init__(self, buckets=None):
        self.buckets = dict(PUBLIC_S3_BUCKETS if buckets is None else buckets)

    def describe(self) -> dict:
        return {**super().describe(), "buckets": sorted(self.buckets)}

    def _headers(self, credentials=None) -> dict:
        """Anonymous only. The allowlisted buckets are public, and S3 answers a
        bearer token with ``400 InvalidArgument`` - so a token could only turn a
        working fetch into a failing one. Refused rather than dropped, because a
        caller who set one meant it to be used."""
        if credentials:
            raise InputError("s3: this source reads public buckets anonymously and takes "
                             "no credentials; a bearer token is refused by S3 itself")
        return {}

    def explain_refusal(self, identifier: str) -> None:
        """`s3://bucket/key` is the spelling every AWS tool uses and the first
        thing anyone types here. The identifier IS the cache key, so accepting
        both spellings would split it - name the one this takes instead."""
        if identifier.startswith("//"):
            bare = identifier.lstrip("/")
            raise InputError(f"s3://{bare}: drop the slashes - this source takes "
                             f"s3:{bare}, so that one object has one identity")

    def check(self, identifier: str, credentials=None) -> None:
        self.explain_refusal(identifier)
        super().check(identifier, credentials)
        self._headers(credentials)
        bucket = identifier.partition("!")[0].partition("/")[0]
        if bucket not in self.buckets:
            raise InputError(
                f"s3 bucket {bucket!r} is not one this server fetches from; "
                f"served buckets: {', '.join(sorted(self.buckets))}")

    def resolve(self, outer: str, credentials=None) -> tuple:
        self.check(outer, credentials)         # allowlist and token, before any request
        bucket, _, key = outer.partition("/")
        region = self.buckets[bucket]
        host = "s3.amazonaws.com" if region is None else f"s3.{region}.amazonaws.com"
        url = f"https://{host}/{bucket}/{key}"
        try:
            with fetchlib.open(url, method="HEAD", timeout=60) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except Exception as e:
            raise InputError(f"s3:{outer}: HEAD failed: {e}") from e
        if size <= 0:
            raise InputError(f"s3:{outer}: the bucket gives no Content-Length")
        return url, size


class GitHubReleaseSource(ArchiveReadingSource):
    """GitHub release assets: ``<owner>/<repo>@<tag>/<asset>[!member]``.

    The tag is REQUIRED - a branch or ``latest`` floats, and reaching a
    repository's *source* is deliberately not offered; only the assets attached
    to a release, which is where datasets and model zips are published.

    **A tag is readable, not immutable.** Unlike ``hf``'s commit sha, ``zenodo``'s
    record id or ``idc``'s version-pinned uuid, a release asset can be replaced
    under a published tag (``gh release upload --clobber``, or delete and
    re-upload through the API), and the tag itself is a git ref that can be
    moved. So this identity is the ``tcia`` kind: stable only as far as the
    publisher's discipline goes, and a result cached under it can outlive the
    bytes it describes. Where that matters, address an asset the publisher names
    by digest (``Slicer/SlicerTestingData`` names every asset by its sha256), or
    use one of the pinning sources.

    Credentials are not accepted, so a private repository's asset can never be
    fetched with a caller's token and then served from a cache every reader can
    ask - the rule ``zenodo`` and ``hf`` state for gated content. A token would
    not work anyway: github.com redirects an authenticated asset download to a
    host that rejects the token, which :func:`_safe_opener` correctly strips on
    the cross-host hop.

    ``resolve`` returns the stable ``github.com/.../releases/download/...`` URL
    rather than the signed CDN URL a HEAD redirects to: those expire in the hour,
    and :class:`RangeFile` re-requests (and so re-follows) per block, which is
    exactly the case its per-request redirect handling exists for.
    """

    prefix = "github"
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
                  r"@[A-Za-z0-9][A-Za-z0-9._+-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
                  r"(?:![A-Za-z0-9._ /-]+)?")
    description = ("GitHub release assets, by owner/repo@tag/asset "
                   "(!member for zip contents)")
    HOST = "https://github.com"

    def _headers(self, credentials=None) -> dict:
        if credentials:
            raise InputError(
                "github: this source fetches public release assets only and takes no "
                "credentials - a private asset fetched with your token would be cached "
                "where every reader of the cache can ask for it, and github.com "
                "redirects an authenticated download to a host that rejects the token "
                "in any case")
        return {}

    def check(self, identifier: str, credentials=None) -> None:
        super().check(identifier, credentials)
        self._headers(credentials)

    def resolve(self, outer: str, credentials=None) -> tuple:
        headers = self._headers(credentials)       # refuse a token before any request
        repo, _, rest = outer.partition("@")
        tag, _, asset = rest.partition("/")
        if not asset:
            raise InputError(f"github:{outer}: no asset name after the tag")
        url = f"{self.HOST}/{repo}/releases/download/{tag}/{asset}"
        try:
            with fetchlib.open(url, method="HEAD", timeout=60, headers=headers) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise InputError(
                    f"github:{outer}: {e.code} from github.com - this source reads public "
                    f"release assets only, and {repo} is private or does not exist") from e
            if e.code == 404:
                raise InputError(
                    f"github:{outer}: no such release asset. Check that {repo} has a RELEASE "
                    f"tagged {tag!r} with an asset named {asset!r} - a branch name or 'latest' "
                    "will not do, and a release's assets are not its source archive") from e
            raise InputError(f"github:{outer}: HEAD failed: {e}") from e
        except Exception as e:
            raise InputError(f"github:{outer}: HEAD failed: {e}") from e
        if size <= 0:
            raise InputError(
                f"github:{outer}: no Content-Length for the asset - check that "
                f"release {tag!r} of {repo} has an asset named {asset!r}")
        return url, size


def check_identifier(src, identifier: str, credentials=None) -> None:
    """``src.check(...)``, for a source that has one.

    Sources are duck-typed here - :func:`registry` accepts any object with a
    prefix, an id_pattern and a fetch - so a source predating :meth:`DataSource.check`
    (or a test stand-in) still gets the grammar enforced, which is the part that
    was always the boundary."""
    # The pattern is checked HERE, always, and the hook runs after. Deferring to
    # the hook when one exists let a stand-in with a do-nothing `check` (a Mock's
    # attributes are all callable) skip validation entirely - the opposite of what
    # a fallback for duck-typed sources is for.
    # A source may recognize a shape the pattern will reject and say something
    # better about it (`s3://bucket/key` is the spelling every AWS tool uses).
    # Give it that chance before the generic refusal, then check the pattern
    # anyway, so a do-nothing hook cannot skip validation.
    explain = getattr(src, "explain_refusal", None)
    if callable(explain):
        explain(identifier)
    pattern = getattr(src, "id_pattern", "")
    if pattern and not re.fullmatch(pattern, identifier):
        raise InputError(f"{getattr(src, 'prefix', '?')}:{identifier} is not a valid "
                         f"{getattr(src, 'prefix', '?')} identifier - "
                         f"{getattr(src, 'description', '') or pattern}")
    hook = getattr(src, "check", None)
    if callable(hook):
        hook(identifier, credentials)


def default_sources() -> list:
    """The sources a server carries unless told otherwise."""
    return [IDCSource(), TCIASource(), openneuro_source(), ZenodoSource(),
            HuggingFaceSource(), S3Source(), GitHubReleaseSource()]


#: The prefixes the built-in sources declare. Read from the sources themselves so
#: it cannot drift from them, and frozen at import rather than per call: a caller
#: may swap ``default_sources`` for its own registry, and whether a string is
#: shaped like a remote input must not depend on which sources a given server
#: happens to serve. Only :func:`parse_input` uses it, and only to recognize a
#: prefix the local-path heuristic would otherwise reject.
_BUILTIN_PREFIXES: frozenset = frozenset()


def registry(sources=None) -> dict:
    """Normalize a list of sources into an ordered ``{prefix: source}`` map."""
    out = {}
    for s in (default_sources() if sources is None else list(sources)):
        if not s.prefix or not s.prefix.replace("_", "").isalnum() or not s.prefix.islower():
            raise ValueError(f"bad source prefix {s.prefix!r}")
        if s.prefix in out or s.prefix in ("jobs", "tasks", "health", "upload", "segmentations", "sources"):
            raise ValueError(f"source prefix {s.prefix!r} collides")
        if not s.id_pattern:
            raise ValueError(f"source {s.prefix!r} declares no id_pattern")
        # identifiers may contain slashes (DOIs, org/name ids): the series
        # cache hashes filesystem-unsafe keys, and the path surface's greedy
        # routes ({ident:path}, parsed right-to-left in serve._mount_source)
        # address them in ordinary URLs - so no pattern restriction is needed.
        out[s.prefix] = s
    return out


_BUILTIN_PREFIXES = frozenset(registry(default_sources()))


class HttpSource(ArchiveReadingSource):
    """A bare ``http://`` / ``https://`` URL, with the same ``!member`` zip-member
    reading as the hosted sources. **Local only**: it is never in
    :func:`default_sources`, so no server carries it - a server fetching from
    client-chosen URLs could be steered at anything it can reach, and a URL is not
    a cache-grade identity (the bytes behind it can change). On the user's own
    machine both objections vanish: it is their URL and their cache."""

    prefix = "http"
    id_pattern = r"(?:s?://)?[^\s!]+(?:![A-Za-z0-9._ /-]+)?"
    description = "any http(s) URL (local CLI only; !member for zip contents)"

    def resolve(self, outer: str, credentials=None) -> tuple:
        """``(url, size)`` from a HEAD request; a host that will not say its size
        still works for whole-file downloads (size 0 means unknown), but a zip
        member needs the size for Range reads and fails clearly without it."""
        url = outer if "://" in outer else f"http://{outer}"
        try:
            with fetchlib.open(url, method="HEAD", timeout=60,
                               headers=self._headers(credentials)) as r:
                size = int(r.headers.get("Content-Length") or 0)
                url = r.url or url                  # follow the redirect once here
        except Exception as e:
            raise InputError(f"{url}: HEAD failed: {e}") from e
        return url, size

    def _zip(self, outer: str, credentials=None):
        url, size = self.resolve(outer, credentials)
        if not size:
            raise InputError(f"{url}: the host gives no Content-Length, so a zip "
                             "member cannot be read by Range; download the archive instead")
        return super()._zip(outer, credentials)


def parse_input(spec, known=None) -> tuple:
    """``(kind, identifier)`` when ``spec`` names a remote input, else ``None``.

    Remote inputs are ``<kind>:<identifier>`` for a registered source (``idc:``,
    ``zenodo:``, ...) or a bare ``http(s)://`` URL. A local path is never a remote
    input, whatever it contains: Windows drive letters and paths with colons in
    their names are one or two characters before the colon, and every source
    prefix is longer *or* is one of the built-in prefixes - ``s3`` is two
    characters and carries a digit, so the shape heuristic alone would read
    ``s3:fcp-indi/...`` as a local path and try to open it as a file. ``known``
    adds the prefixes a caller's own registry serves, which is how a source with
    a short or digit-bearing prefix becomes reachable without loosening the
    shape test for everyone."""
    text = str(spec)
    if text.startswith(("http://", "https://")):
        return "http", text
    kind, sep, ident = text.partition(":")
    if not sep or not ident or "/" in kind:
        return None
    if kind in _BUILTIN_PREFIXES or kind in (known or ()):
        return kind, ident
    # Anything else has to LOOK like a source rather than a path. Letters only,
    # three or more: `sub_01:ses1`, `data_2024:merged.nii` and `study2:v1.nii.gz`
    # are ordinary names in this field, and reading them as remote specs would be
    # worse than making a caller name their own source (which `known` is for).
    if len(kind) >= 3 and kind.isalpha() and kind.islower():
        return kind, ident
    return None


def default_input_cache() -> Path:
    from .cache_admin import cache_root      # ONE root: HAVERSACK_CACHE_DIR, expanded, else XDG
    return cache_root() / "inputs"


def source_stem(spec) -> str:
    """A filename stem for a source, for naming a converted output into a directory: the
    member/file basename for zenodo/http/etc. (minus any image extension), or the bare
    identifier for IDC (only a UUID exists). Falls back to the whole spec for a local path."""
    from .io import image_suffix
    parsed = parse_input(spec)
    text = parsed[1] if parsed else str(spec)
    text = text.split("!")[-1] if "!" in text else text
    name = Path(text).name or text
    suf = image_suffix(name)
    return name[: -len(suf)] if suf else name


def materialize(spec, *, cache_dir=None, sources=None, progress=None, credentials=None) -> Path:
    """Turn a remote input spec into a local path, fetching once.

    The local counterpart of the server's series cache: one directory per
    ``(kind, identifier)`` under ``cache_dir`` (default
    ``~/.cache/haversack/inputs``), fetched by the source and marked complete, so
    a second task on the same series does not download it again. Returns the
    fetched file when the fetch produced exactly one, else the directory (a DICOM
    series). A local path passes through untouched. ``sources`` defaults to the
    server's registry plus :class:`HttpSource`; a source whose runtime is missing
    (IDC without obstore) refuses with the extra to install.
    """
    import hashlib
    reg = registry(sources) if sources is not None else registry(default_sources() + [HttpSource()])
    parsed = parse_input(spec, known=reg)
    if parsed is None:
        return Path(spec)
    kind, ident = parsed
    src = reg.get("http" if kind == "https" else kind)
    if src is None:
        raise InputError(f"unknown input source {kind!r}; known: {', '.join(sorted(reg))} and http(s) URLs")
    if hasattr(src, "enabled") and not src.enabled():
        raise InputError(f"the {kind} source's runtime is not installed in this environment "
                         f"(a lean install?): uv pip install {'obstore' if kind == 'idc' else kind}")
    check_identifier(src, ident, credentials)
    root = Path(cache_dir) if cache_dir else default_input_cache()
    entry = root / kind / hashlib.sha1(ident.encode()).hexdigest()[:20]
    done = entry / ".done"
    if not done.exists():
        import shutil
        if entry.exists():
            shutil.rmtree(entry)             # a partial fetch: start over
        entry.mkdir(parents=True)
        if progress:
            progress(f"fetching {ident if kind == 'http' else f'{kind}:{ident}'}")
        src.fetch(ident, entry, credentials=credentials)
        done.write_text(f"{kind}:{ident}\n")
    content = entry / "series"
    if not content.is_dir():
        content = entry                      # a source that wrote directly under the entry
    files = [f for f in content.iterdir() if f.is_file() and not f.name.startswith(".")]
    return files[0] if len(files) == 1 else content
