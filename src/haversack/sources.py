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

from . import content, fetchlib
from .errors import InputError

__all__ = ["DataSource", "UrlTemplateSource", "IDCSource", "GitHubReleaseSource",
           "ObjectStoreSource", "S3Source", "GCSSource", "default_sources", "IDC_BUCKETS",
           "IDC_GCS_BUCKETS", "PUBLIC_S3_BUCKETS", "PUBLIC_GCS_BUCKETS", "CRDC_RE"]

CRDC_RE = r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"

# The three public IDC buckets, probed in order (idc-open-data holds 99.5 % of
# series; -two and -cr the rest - found the hard way 2026-08-24). The clean
# upgrade path is resolving per series via idc-index (`series_aws_url`), which we
# take when /v1/resolve lands.
IDC_BUCKETS = ("idc-open-data", "idc-open-data-two", "idc-open-data-cr")
# IDC's Google Cloud mirror holds the main bucket only: `-two` and `-cr` answer
# NoSuchBucket on GCS (checked 2026-09-06). So HAVERSACK_IDC_CLOUD=gcp means
# "prefer GCS", and a series that lives only in the other two is still fetched
# from AWS - see IDCSource.probe_order.
IDC_GCS_BUCKETS = ("idc-open-data",)


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

# The same allowlist for Google Cloud Storage; `gs:` reads these anonymously.
PUBLIC_GCS_BUCKETS = {
    "idc-open-data": None,          # IDC's GCS mirror of its main bucket
}


def idc_cloud() -> str:
    """Which cloud the ``idc:`` source fetches from first: ``aws`` (default) or
    ``gcp`` (``HAVERSACK_IDC_CLOUD``). A deployment that runs in Google Cloud
    pays egress and latency for the AWS buckets and none for the mirror. The
    identity is the same either way - ``idc:<uuid>`` names the same bytes on
    both clouds - so the result cache does not care which served them."""
    cloud = (os.environ.get("HAVERSACK_IDC_CLOUD") or "aws").strip().lower()
    if cloud not in ("aws", "gcp"):
        raise InputError(f"HAVERSACK_IDC_CLOUD={cloud!r}: choose aws or gcp")
    return cloud


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

    def forget(self, identifier: str) -> None:
        """Drop whatever this source has cached about ``identifier``.

        Called when a job asks for fresh bytes (``Cache-Control: no-cache``), so
        the refusal has to reach EVERY cache that stands between the request and
        the repository - not just the series cache and the read-ahead image.

        The one that was missed is the parsed archive. A source instance lives as
        long as the process on both deployments (``LocalExecutor.__init__``, and
        Modal's ``@modal.enter``), and it holds up to four ``ZipFile`` objects
        with their central directories and a ``RangeFile`` block cache of up to
        256 MiB each. On a hit ``resolve``/``locate`` is skipped, so a forced
        refetch of a replaced ``s3:`` object or ``github:`` asset reused not only
        old bytes but the OLD RESOLVED URL and the old central-directory offsets -
        against an object that really had changed, that is a CRC failure or
        garbage rather than a refresh (2026-09-08).

        Defined here rather than on the two archive classes because both keep
        their cache in the same attribute, and a third would inherit the fix.
        """
        outer = str(identifier).partition("!")[0]
        cache = self.__dict__.get("_archives")
        if not cache:
            return
        # every credential's parse of that object is equally stale
        for ck in [k for k in list(cache) if (k[0] if isinstance(k, tuple) else k) == outer]:
            cache.pop(ck, None)

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """What this repository says about one input: where it came from, under
        what license, and what to cite. ``None`` means *not determined* - never
        a guess. The shape is the one every result's ``provenance.inputs``
        carries::

            {"origin":  {collection or dataset, uid, doi, url, creators, version,
                         determined_by},
             "license": {"name", "url"} or None,
             "cite":    [{"text", "doi", "for"}, ...]}

        Metadata only: it must not download the data (``haversack rights``
        calls it without fetching), and it may use ``fetched`` - the path a
        fetch just produced - when the answer is in the files. Called once per
        fetch and recorded beside the bytes (:func:`fetch_recording_origin`),
        with the content's own digest, so a result computed from a CC BY-NC
        series can say so, and can say which bytes it saw. Best effort: a
        failure here is recorded as undetermined, never raised into a fetch that
        has already succeeded.
        """
        return None
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

    def __init__(self, prefix: str, id_pattern: str, url_template: str, *, policy_record=None,
                 filename: str | None = None, description: str = ""):
        if "{id}" not in url_template:
            raise ValueError("url_template needs an {id} placeholder")
        self.prefix, self.id_pattern = prefix, id_pattern
        self.url_template, self.filename = url_template, filename
        self.description = description or f"single-file fetch from {url_template}"
        self.policy_record = policy_record


    #: The record that holds for every identifier of this source by the
    #: publisher's policy (OpenNeuro publishes everything CC0), with ``{id}`` and
    #: ``{dataset}`` substituted. None means the source cannot say.
    policy_record: dict | None = None

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        if not self.policy_record:
            return None
        dataset = identifier.split("/", 1)[0]

        def fill(v):
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            return v.format(id=identifier, dataset=dataset) if isinstance(v, str) else v
        return fill(self.policy_record)
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


#: The Creative Commons names the repositories use, to the license text. A name
#: not here is passed through with no URL rather than mapped to a wrong one.
CC_LICENSE_URLS = {
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC 3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


def _license(name, url=None) -> dict | None:
    if not name:
        return None
    key = str(name).strip()
    spdx = key.lower().replace(" ", "-")            # "cc-by-4.0" as Zenodo spells it
    out = {"name": key}
    url = url or CC_LICENSE_URLS.get(key) or next(
        (u for n, u in CC_LICENSE_URLS.items() if n.lower().replace(" ", "-") == spdx), None)
    if url:
        out["url"] = url
    return out


INPUT_SIDECAR = ".input.json"


def sole_file(directory) -> Path | None:
    """The one file an entry holds, or None when it holds several.

    A fetch always produces a DIRECTORY - ``<entry>/series/`` - even for a single
    object, and :func:`materialize` hands the pipeline that one file rather than
    the directory when there is exactly one. So a one-file fetch IS a file, and
    its digest has to say so: it read ``sha256-tree:`` over a directory of one
    while the same bytes uploaded read ``sha256:``, and the two never matched.
    Dotfiles are not content (the fetch's own ``.input.json`` lives beside them).
    """
    try:
        files = [f for f in Path(directory).iterdir()
                 if f.is_file() and not f.name.startswith(".")]
    except OSError:
        return None
    return files[0] if len(files) == 1 else None


def _dicom_facts(directory) -> dict | None:
    """The identifiers a DICOM series carries in its own files, or None.

    Read from the DIRECTORY the files sit in, whichever number of them there is.
    This lived inside the many-files branch, so a series stored as ONE file - an
    enhanced multiframe volume, or any of IDC's 85k single-instance series - lost
    exactly the identifiers that pin a result whose source identity is not
    cache-grade (`tcia:` is not version-pinned; its SeriesInstanceUID is the
    durable fact). Found 2026-09-07 by review.
    """
    try:
        from . import io as nio
        d = Path(directory)
        uids = nio.dicom_series_ids(d)
        if not uids:
            return None
        files = sorted(f for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
        out = {"series_instance_uid": uids[0] if len(uids) == 1 else uids}
        reader = nio._sitk().ImageFileReader()
        reader.SetFileName(str(files[0]))
        reader.ReadImageInformation()
        for key, tag in (("study_instance_uid", "0020|000d"), ("modality", "0008|0060"),
                         ("series_description", "0008|103e")):
            if reader.HasMetaDataKey(tag) and reader.GetMetaData(tag).strip():
                out[key] = reader.GetMetaData(tag).strip()
        return out
    except Exception:                           # not DICOM, or unreadable: the digest stands
        return None


def _content_facts(fetched) -> dict | None:
    """What the fetched bytes ARE: their digest (the content store's own
    function, so a fetched series and a stored one hash alike), size, file
    count - and for a DICOM series the identifiers the files themselves carry.
    This is what makes a result reproducible from an identity that is not
    cache-grade (``s3:``, ``github:``, ``tcia:``): the record pins the bytes."""
    if fetched is None:
        return None
    from .content import digest_dir, digest_file
    p = Path(fetched)
    # The directory the bytes sit in, kept whatever the digest turns out to
    # describe: the DICOM identifiers are read from it either way.
    holder = p if p.is_dir() else p.parent
    one = sole_file(p) if p.is_dir() else (p if p.is_file() else None)
    if one is not None:                         # one file is a file: see sole_file
        out = {"digest": digest_file(one), "bytes": one.stat().st_size, "files": 1}
    elif p.is_dir():
        files = [f for f in p.rglob("*") if f.is_file()]
        out = {"digest": digest_dir(p), "bytes": sum(f.stat().st_size for f in files),
               "files": len(files)}
    else:
        return None
    dicom = _dicom_facts(holder)
    if dicom:
        out["dicom"] = dicom
    return out


def fetch_recording_origin(src, identifier: str, entry, credentials=None):
    """``src.fetch(...)``, then what the bytes are and where they came from,
    recorded beside them as ``.input.json`` - one door for every fetch that
    lands in a series cache (the local server, the Modal worker, ``haversack
    get``), so the record exists wherever the bytes do. The lookup is best
    effort: an input whose origin could not be determined says so, and a
    failure never undoes a fetch that succeeded."""
    import json
    import time
    entry = Path(entry)
    fetched = (src.fetch(identifier, entry, credentials=credentials) if credentials is not None
               else src.fetch(identifier, entry))
    identity = getattr(src, "identity", None)   # sources are duck-typed (registry())
    record = {"kind": src.prefix,
              "identity": identity(identifier) if callable(identity) else f"{src.prefix}:{identifier}",
              "content": None, "origin": None, "license": None, "cite": []}
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        record["content"] = _content_facts(fetched)
    except Exception as e:                      # noqa: BLE001 - a digest is a courtesy
        record["content_error"] = f"{type(e).__name__}: {e}"
    try:
        describe = getattr(src, "describe_input", None)
        said = describe(identifier, fetched, credentials) if callable(describe) else None
        if said:
            record["origin"] = {**(said.get("origin") or {}), "fetched": fetched_at}
            record["license"] = said.get("license")
            record["cite"] = list(said.get("cite") or [])
        else:
            record["origin"] = {"fetched": fetched_at}
            record["note"] = "the source could not determine this input's origin or license"
    except Exception as e:                      # noqa: BLE001 - undetermined, not fatal
        record["origin"] = {"fetched": fetched_at}
        record["error"] = f"{type(e).__name__}: {e}"
    try:
        (entry / INPUT_SIDECAR).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass                                    # a read-only cache: the fetch still stands
    return fetched


def read_input_record(entry) -> dict | None:
    """The record :func:`fetch_recording_origin` left, or None when there is none."""
    import json
    try:
        return json.loads((Path(entry) / INPUT_SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


IDC_API = "https://api.imaging.datacommons.cancer.gov/v3"


def _idc_record(crdc_series_uuid: str) -> dict | None:
    """One series' collection, IDC data release, license and citation from
    IDC's own API.

    The license belongs to the SERIES in IDC - 39 of its collections carry more
    than one - so it is asked per series, by the uuid, through the API's SQL
    endpoint over the same index idc-index ships. The citation endpoint answers
    the dataset's own citation and IDC's acknowledgment (Fedorov et al. 2023),
    which IDC asks for on every use. Read 2026-09-06 against API 3.0.0b3; no
    authentication."""
    rows = fetchlib.post_json(f"{IDC_API}/sql", {
        "sql": "SELECT collection_id, license_short_name, SeriesInstanceUID, "
               "series_revised_idc_version FROM index "
               f"WHERE crdc_series_uuid = '{crdc_series_uuid}'"}, timeout=30).get("rows") or []
    if not rows:
        return None
    row = rows[0]
    origin = {"collection": row.get("collection_id"), "uid": row.get("SeriesInstanceUID"),
              "url": f"https://portal.imaging.datacommons.cancer.gov/explore/filters/"
                     f"?collection_id={row.get('collection_id')}",
              "determined_by": "IDC API v3 (index by crdc_series_uuid)"}
    if row.get("series_revised_idc_version") is not None:
        origin["version"] = f"IDC data release v{row['series_revised_idc_version']}"
    out = {"origin": origin, "license": _license(row.get("license_short_name")), "cite": []}
    try:
        cit = fetchlib.post_json(f"{IDC_API}/citations", {
            "filters": {"terms": {"SeriesInstanceUID": [row.get("SeriesInstanceUID")]}}}, timeout=30)
        for text in [re.sub(r"<[^>]+>", "", c) for c in cit.get("citations") or []]:
            ref = {"text": text, "for": "dataset"}
            doi = re.search(r"10\.\d{4,9}/\S+", text)
            if doi:
                ref["doi"] = doi.group(0).rstrip(".")
                origin.setdefault("doi", ref["doi"])
            out["cite"].append(ref)
        if cit.get("idc_acknowledgment"):
            out["cite"].append({"text": re.sub(r"\s+", " ", cit["idc_acknowledgment"]),
                                "doi": "10.1148/rg.230180", "for": "IDC"})
    except Exception:                           # the license is the answer; the citation is a courtesy
        pass
    return out


def _object_store(cloud: str, bucket: str, region: str | None = None):
    """An anonymous obstore store on a public bucket: the one place the two
    clouds' anonymous configurations are spelled. Imported at call time so a
    lean install without obstore still imports this module, and so tests can
    stand a fake in for the store classes."""
    if cloud == "aws":
        from obstore.store import S3Store
        config = {"aws_skip_signature": "true",
                  # path-style: a dotted bucket name (openneuro.org) cannot be
                  # virtual-hosted without breaking TLS
                  "aws_virtual_hosted_style_request": "false"}
        if region:
            config["aws_region"] = region
        return S3Store.from_url(f"s3://{bucket}", config=config)
    if cloud == "gcp":
        from obstore.store import GCSStore
        return GCSStore.from_url(f"gs://{bucket}", skip_signature=True)
    raise ValueError(f"unknown cloud {cloud!r}")


def _list_objects(store, prefix: str) -> list:
    """``[(key, size)]`` under ``prefix``, in listing order."""
    out = []
    for page in store.list(prefix=prefix):
        for o in page:
            if isinstance(o, dict):
                out.append((str(o.get("path")), int(o.get("size") or 0)))
            else:
                out.append((str(o), 0))
    return out


class _Budget:
    """A byte ceiling shared by concurrent downloads.

    One object's stream cannot see what the other thirty-one are doing, so a
    per-object cap bounds a prefix fetch at ``threads * cap`` rather than at
    ``cap``. This is the shared counter, and it is spent as the bytes land.
    """

    def __init__(self, cap: int, what: str):
        import threading
        self.cap, self._what = cap, what
        self._left = cap
        self._lock = threading.Lock()

    def spend(self, n: int) -> None:
        with self._lock:
            self._left -= n
            if self._left < 0:
                raise InputError(f"{self._what}: exceeded the {self.cap}-byte fetch cap")


def _fetch_objects(store, keys: list, dest: Path, *, what: str, cap: int,
                   threads: int = 32) -> int:
    """Download every listed object into ``dest``, flattened, in parallel - the
    IDC mechanism (obstore beat s5cmd in every measured quadrant), shared by
    every prefix fetch. Returns how many files were written.

    Two objects under one prefix can share a basename (``case/a/IM.dcm`` and
    ``case/b/IM.dcm``), and until 2026-09-08 both were opened at the same
    destination by different threads - so the survivor was not the later object
    but an interleaving of the two, and a slice of the series was simply gone.
    The flattening rule is :func:`haversack.content.flatten_names`, shared with
    the content store and the archive extractor. It matters more here than it
    looks: a series that collapses onto one name stops being a series, because
    :func:`sole_file` then hands the pipeline a single file, and the input's
    provenance digest turns from ``sha256-tree:`` into ``sha256:``.

    The listing's own sizes are checked before any byte moves, and then the
    bytes are counted as they arrive - a listing that understated a size cannot
    fill the disk.
    """
    from concurrent.futures import ThreadPoolExecutor
    total = sum(size for _, size in keys)
    if total > cap:
        raise InputError(f"{what}: {len(keys)} objects total {total} bytes, over the "
                         f"{cap}-byte fetch cap (HAVERSACK_MAX_FETCH_GB)")
    # a bucket pseudo-directory key has no basename of its own
    wanted = [k for k, _ in keys if content.basename(k) not in ("", ".", "..")]
    budget = _Budget(cap, what)
    names = content.flatten_names(wanted)

    def one(pair):
        key, name = pair
        _stream_object(store, key, dest / name, what=what, cap=cap, budget=budget)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, zip(wanted, names)))
    return len(wanted)


def _stream_object(store, key: str, out: Path, *, what: str, cap: int, budget=None) -> int:
    """One object to disk in 1 MiB chunks, with the same hard ceiling the HTTP
    downloads have: a listing that lied about a size cannot fill the disk.

    ``budget``, when given, is a ceiling shared with every other object in the
    same fetch; the per-object ``cap`` still applies on its own."""
    import obstore
    n = 0
    with open(out, "wb") as f:
        for chunk in obstore.get(store, key).stream(min_chunk_size=1 << 20):
            n += len(chunk)
            if n > cap:
                raise InputError(f"{what}: exceeded the {cap}-byte fetch cap")
            if budget is not None:
                budget.spend(len(chunk))
            f.write(chunk)
    return n


class IDCSource(DataSource):
    """NCI Imaging Data Commons: DICOM series by ``crdc_series_uuid`` from the
    public open-data buckets, anonymously, 32 threads. The uuid names a
    version-pinned series; the bucket is probed across the known ones rather
    than assumed, on the cloud ``HAVERSACK_IDC_CLOUD`` prefers first."""

    prefix = "idc"
    id_pattern = CRDC_RE
    description = "NCI Imaging Data Commons, by crdc_series_uuid"

    def enabled(self) -> bool:
        try:
            import obstore  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def probe_order(cloud: str | None = None) -> list:
        """``[(cloud, bucket)]`` in the order a series is looked for. GCS first
        when asked, then every AWS bucket - the mirror is partial, and a series
        that lives only in ``-two`` or ``-cr`` must still be found."""
        aws = [("aws", b) for b in IDC_BUCKETS]
        if (cloud or idc_cloud()) == "gcp":
            return [("gcp", b) for b in IDC_GCS_BUCKETS] + aws
        return aws

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        return _idc_record(identifier)
    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        keys, store, probed = [], None, []
        for cloud, bucket in self.probe_order():
            probed.append(f"{'gs' if cloud == 'gcp' else 's3'}://{bucket}")
            store = _object_store(cloud, bucket)
            keys = _list_objects(store, f"{identifier}/")
            if keys:
                break
        if not keys:
            raise InputError(f"no objects under {identifier!r}/ in any probed IDC bucket "
                             f"({', '.join(probed)}); if the series exists, IDC may "
                             "have added a bucket this server does not know")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        try:
            _fetch_objects(store, keys, dest, what=f"idc:{identifier}", cap=MAX_FETCH_BYTES)
        except InputError:
            raise
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
    SERIES_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """NBIA's ``getSeries`` answers the collection and its license per series
        (``LicenseName``, ``LicenseURI``, ``Collection``, ``CollectionURI``; read
        2026-09-06). A series on a separate NBIA instance (NLST) answers nothing
        here, and that is reported as undetermined rather than assumed."""
        rows = fetchlib.get_json(f"{self.SERIES_API}?SeriesInstanceUID={identifier}", timeout=30)
        if not rows:
            return None
        row = rows[0]
        origin = {"collection": row.get("Collection"), "uid": identifier,
                  "url": row.get("CollectionURI"), "released": row.get("DateReleased"),
                  "determined_by": "TCIA NBIA API getSeries"}
        cite = []
        doi = re.search(r"10\.\d{4,9}/\S+", str(row.get("CollectionURI") or ""))
        if doi:
            origin["doi"] = doi.group(0)
            cite.append({"text": f"{row.get('Collection')} (The Cancer Imaging Archive). "
                                 f"https://doi.org/{doi.group(0)}", "doi": doi.group(0), "for": "dataset"})
        return {"origin": origin, "license": _license(row.get("LicenseName"), row.get("LicenseURI")),
                "cite": cite}
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
        description="OpenNeuro (CC0), by ds<number>/<file path>",
        # by policy, not per dataset: every published OpenNeuro dataset is CC0
        policy_record={
            "origin": {"dataset": "{dataset}", "url": "https://openneuro.org/datasets/{dataset}",
                       "determined_by": "OpenNeuro policy: every published dataset is released "
                                        "under CC0 (docs.openneuro.org/faq, read 2026-09-06)"},
            "license": {"name": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
            "cite": []})


#: The version of the FETCHED-INPUT contract: what a download of one identifier is
#: expected to produce on disk. It is part of a fetched entry's cache key, so an entry
#: written by an older build is simply not found and is downloaded again.
#:
#: NOT the same thing as `serve.CACHE_EPOCH`, which versions computed RESULTS, and the
#: distinction is the whole point: a result epoch throws away segmentations while
#: leaving the inputs they were computed from marked complete, so a build that fixes
#: HOW AN INPUT IS DOWNLOADED needs this one too or the corrected code never runs.
#:
#: 1 (implicit): everything before 2026-09-09.
#: 2: the flattener that could write two objects of one prefix to a single path, so a
#:    DICOM series could be committed a slice short - and, being one file, could stop
#:    reading as a series at all.
FETCH_EPOCH = "2"

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
                 max_blocks: int = 64, reader=None):
        import collections
        self.url, self.size, self.pos = url, int(size), 0
        self.headers = dict(headers or {})
        #: ``reader(lo, hi) -> bytes`` for the inclusive byte range. HTTP Range
        #: by default; an object store supplies its own (``get_range``), and the
        #: block cache, the short-read check and zipfile's random access are
        #: the same over either.
        self.reader = reader or self._http_range
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

    @classmethod
    def over(cls, reader, size: int, **kw) -> "RangeFile":
        """A RangeFile with no URL at all: every read goes to ``reader``."""
        return cls("", size, reader=reader, **kw)

    def _http_range(self, lo: int, hi: int) -> bytes:
        with fetchlib.open(self.url, timeout=300,
                           headers={**self.headers, "Range": f"bytes={lo}-{hi}"}) as r:
            if r.status != 206:
                # a 200 means the server ignored Range and is streaming the
                # WHOLE body - on a multi-GB archive that is an unbounded read
                # into memory; refuse rather than pull it
                raise InputError(
                    f"range request not honored (status {r.status}) by "
                    f"{self.url}; server does not support HTTP Range")
            return r.read()

    def _block(self, i: int) -> bytes:
        if i in self._blocks:
            self._blocks.move_to_end(i)
            return self._blocks[i]
        lo = i * self.block_size
        hi = min(self.size, lo + self.block_size) - 1
        data = self.reader(lo, hi)
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
            # flatten: archive paths never touch disk. The naming rule is shared
            # with the content store and the prefix fetch, because this site had
            # its own and it lost a member - it disambiguated only on collision,
            # so the `IM-1.dcm` it invented for a second `IM.dcm` was overwritten
            # by a third member genuinely named `IM-1.dcm`.
            keep = [m for m in members
                    if content.basename(m) and not content.basename(m).startswith(".")]
            for m, name in zip(keep, content.flatten_names(keep)):
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

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """The record's own license, creators and DOI (every Zenodo record has
        all three), and a citation in the form Zenodo suggests. A record id IS a
        version: new versions mint new ids."""
        recid = identifier.partition("/")[0]
        rec = fetchlib.get_json(f"https://zenodo.org/api/records/{recid}", timeout=30,
                                headers=self._headers(credentials))
        meta = rec.get("metadata") or {}
        lic = meta.get("license") or {}
        creators = [c.get("name") for c in meta.get("creators") or [] if c.get("name")]
        doi = rec.get("doi") or meta.get("doi")
        origin = {"dataset": meta.get("title"), "url": f"https://zenodo.org/records/{recid}",
                  "creators": creators, "doi": doi,
                  "version": meta.get("version") or f"record {recid}",
                  "determined_by": "Zenodo record metadata"}
        cite = []
        if creators and meta.get("title") and doi:
            cite.append({"text": f"{', '.join(creators[:3])}{' et al.' if len(creators) > 3 else ''}. "
                                 f"{meta['title']}. Zenodo. https://doi.org/{doi}",
                         "doi": doi, "for": "dataset"})
        return {"origin": origin, "license": _license(lic.get("id") or lic.get("title")), "cite": cite}
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

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """The dataset card's license (``cardData.license`` / a ``license:`` tag),
        which is whatever the uploader declared - reported as such. The commit
        sha in the identifier is the version."""
        repo, _, rest = identifier.partition("@")
        card = fetchlib.get_json(f"https://huggingface.co/api/datasets/{repo}", timeout=30,
                                 headers=self._headers(credentials))
        lic = (card.get("cardData") or {}).get("license")
        if isinstance(lic, list):
            lic = ", ".join(str(x) for x in lic)
        if not lic:
            tags = [t.partition(":")[2] for t in card.get("tags") or [] if str(t).startswith("license:")]
            lic = ", ".join(tags) or None
        return {"origin": {"dataset": repo, "url": f"https://huggingface.co/datasets/{repo}",
                           "author": card.get("author"), "version": rest.partition("/")[0] or None,
                           "determined_by": "Hugging Face dataset card (as declared by the uploader)"},
                "license": _license(lic), "cite": []}
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


class ObjectStoreSource(ArchiveReadingSource):
    """Public object-store buckets through obstore, anonymously.

    Three identifier shapes: ``<bucket>/<key>`` downloads one object;
    ``<bucket>/<key>!<member>`` reads a member of a remote zip by ranged reads;
    ``<bucket>/<prefix>/`` (trailing slash) downloads every object under the
    prefix in parallel, by basename - a DICOM series laid out in a bucket, which
    is exactly what ``idc:`` does for its own buckets and what no HTTP source
    can do at all. The bucket is chosen by the operator (an ALLOWLIST, per
    cloud); the identifier only picks inside one. That is the SSRF boundary.

    Anonymous only, on purpose. The buckets are public; a token could only
    turn a working fetch into a failing one, and a private object fetched with
    a caller's credential would be cached where every cache reader can ask
    for it. A credential is refused rather than dropped, because a caller who
    set one meant it to be used.

    NOT version-pinned: an object can be overwritten in place, so the same
    identity can resolve to different bytes across dataset releases (which is
    what ``Cache-Control: no-cache`` is for).
    """

    #: ``aws`` or ``gcp`` - what :func:`_object_store` builds.
    cloud: str = ""
    #: ``{bucket: region-or-None}``; subclasses set the default allowlist.
    default_buckets: dict = {}
    #: The URL scheme people type by habit (``s3://``), refused with a hint.
    scheme: str = ""

    def __init__(self, buckets=None):
        self.buckets = dict(self.default_buckets if buckets is None else buckets)

    def enabled(self) -> bool:
        try:
            import obstore  # noqa: F401
            return True
        except ImportError:
            return False

    def describe(self) -> dict:
        return {**super().describe(), "buckets": sorted(self.buckets)}

    def _headers(self, credentials=None) -> dict:
        if credentials:
            raise InputError(f"{self.prefix}: this source reads public buckets anonymously "
                             "and takes no credentials")
        return {}

    def explain_refusal(self, identifier: str) -> None:
        """``s3://bucket/key`` is the spelling every cloud tool prints and the
        first thing anyone types here. The identifier IS the cache key, so
        accepting both spellings would split it - name the one this takes."""
        if identifier.startswith("//"):
            bare = identifier.lstrip("/")
            raise InputError(f"{self.scheme}://{bare}: drop the slashes - this source takes "
                             f"{self.prefix}:{bare}, so that one object has one identity")

    def check(self, identifier: str, credentials=None) -> None:
        self.explain_refusal(identifier)
        super().check(identifier, credentials)
        self._headers(credentials)
        bucket = identifier.partition("!")[0].partition("/")[0]
        if bucket not in self.buckets:
            raise InputError(
                f"{self.prefix} bucket {bucket!r} is not one this server fetches from; "
                f"served buckets: {', '.join(sorted(self.buckets))}")

    def _store(self, bucket: str):
        return _object_store(self.cloud, bucket, self.buckets[bucket])

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """A bucket carries no license of its own ("the bucket name is not a
        license label", IDC's own words). An IDC bucket's series prefix IS an
        IDC series, and is asked about as one; anything else is undetermined."""
        bucket, _, key = identifier.partition("!")[0].partition("/")
        m = re.fullmatch(rf"({CRDC_RE})/?", key)
        if bucket in IDC_BUCKETS + IDC_GCS_BUCKETS and m:
            return _idc_record(m.group(1))
        return None
    def locate(self, outer: str, credentials=None) -> tuple:
        """``(store, key, size)`` for one object, after the allowlist."""
        import obstore
        self.check(outer, credentials)         # allowlist and token, before any request
        bucket, _, key = outer.partition("/")
        store = self._store(bucket)
        try:
            size = int(obstore.head(store, key)["size"])
        except Exception as e:
            raise InputError(f"{self.prefix}:{outer}: not found or unreadable: {e}") from e
        return store, key, size

    def resolve(self, outer: str, credentials=None) -> tuple:
        """``(url, size)`` in the cloud's own spelling, for anyone who asks;
        the fetch itself goes through the store, not this URL."""
        _store, key, size = self.locate(outer, credentials)
        return f"{self.scheme}://{outer.partition('/')[0]}/{key}", size

    def _zip(self, outer: str, credentials=None):
        """The remote zip over the store's ranged reads instead of HTTP Range.

        Cached per ``(outer, credentials)``, for the reason
        :meth:`ArchiveReadingSource._zip` gives at length: on a cache hit ``locate`` is
        skipped, and ``locate`` is where ``check`` runs the bucket allowlist and
        ``_headers`` refuses a credential. This cache keyed on ``outer`` alone until
        2026-09-08 - the exact defect its sibling documents as a round-4 finding, sitting
        un-fixed in the parallel class. What it bypassed here is the allowlist rather than
        a token, because this source refuses credentials outright rather than forwarding
        them; that mitigation is incidental, and the next object-store subclass to accept
        one would inherit the replay.
        """
        import obstore
        import zipfile
        cache = self.__dict__.setdefault("_archives", {})
        ck = (outer, credentials)
        z = cache.get(ck)
        if z is None:
            store, key, size = self.locate(outer, credentials)

            def read_range(lo, hi):
                return bytes(obstore.get_range(store, key, start=lo, end=hi + 1))

            z = zipfile.ZipFile(RangeFile.over(read_range, size))
            cache[ck] = z
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return z

    def fetch(self, identifier: str, dest_dir: Path, *, credentials=None) -> Path:
        outer, _, member = identifier.partition("!")
        if _has_dotdot(outer):
            raise InputError(f"{self.prefix}:{identifier}: '..' path segment refused")
        if outer.endswith("/"):                    # every object under a prefix
            if member:
                raise InputError(f"{self.prefix}:{identifier}: a prefix fetch takes no !member")
            self.check(outer, credentials)
            bucket, _, prefix = outer.partition("/")
            store = self._store(bucket)
            dest = Path(dest_dir) / "series"
            dest.mkdir(exist_ok=True)
            try:
                keys = _list_objects(store, prefix)
                if not keys:
                    raise InputError(f"{self.prefix}:{identifier}: no objects under that prefix")
                _fetch_objects(store, keys, dest, what=f"{self.prefix}:{identifier}",
                               cap=MAX_FETCH_BYTES)
            except InputError:
                raise
            except Exception as e:
                raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
            return dest
        if member:                                 # a zip member: the shared extractor
            return super().fetch(identifier, dest_dir, credentials=credentials)
        store, key, size = self.locate(outer, credentials)
        if size > MAX_FETCH_BYTES:
            raise InputError(f"{self.prefix}:{identifier}: {size} bytes, over the "
                             f"{MAX_FETCH_BYTES}-byte fetch cap")
        dest = Path(dest_dir) / "series"
        dest.mkdir(exist_ok=True)
        name = Path(key).name or "image"
        try:
            _stream_object(store, key, dest / name, what=f"{self.prefix}:{identifier}",
                           cap=MAX_FETCH_BYTES)
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"fetch of {self.prefix}:{identifier} failed: {e}") from e
        return dest


class S3Source(ObjectStoreSource):
    """Public S3 buckets: ``<bucket>/<key>[!member]`` or ``<bucket>/<prefix>/``.
    See :class:`ObjectStoreSource`; the buckets are :data:`PUBLIC_S3_BUCKETS`."""

    prefix = "s3"
    cloud = "aws"
    scheme = "s3"
    default_buckets = PUBLIC_S3_BUCKETS
    # <bucket>/<key>[!member]. Bucket syntax is AWS's own (3-63 chars, lowercase
    # alphanumerics, dots and hyphens); membership in the allowlist is checked in
    # check(), because a rejected bucket deserves a message naming the ones served.
    id_pattern = (r"(?!.*(?:^|/)\.\.(?:/|!|$))"
                  r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[A-Za-z0-9][A-Za-z0-9._/-]{0,300}"
                  r"(?:![A-Za-z0-9._ /-]+)?")
    description = ("public S3 buckets, by bucket/key (!member for zip contents; "
                   "a trailing / fetches every object under a prefix)")


class GCSSource(ObjectStoreSource):
    """Public Google Cloud Storage buckets, the same way: ``gs:<bucket>/<key>``.
    The buckets are :data:`PUBLIC_GCS_BUCKETS` - today IDC's mirror, so a
    deployment in Google Cloud can address one object or a whole series prefix
    (``gs:idc-open-data/<crdc_series_uuid>/``) without leaving the cloud."""

    prefix = "gs"
    cloud = "gcp"
    scheme = "gs"
    default_buckets = PUBLIC_GCS_BUCKETS
    id_pattern = S3Source.id_pattern            # GCS bucket names follow the same rules
    description = ("public Google Cloud Storage buckets, by bucket/key (!member for zip "
                   "contents; a trailing / fetches every object under a prefix)")


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

    def describe_input(self, identifier: str, fetched=None, credentials=None) -> dict | None:
        """The repository's declared license (GitHub's own detection of its
        LICENSE file) and the release tag. A release asset need not be under the
        repository's license, and a repository without one answers null - both
        reported as undetermined."""
        repo, _, rest = identifier.partition("@")
        info = fetchlib.get_json(f"https://api.github.com/repos/{repo}", timeout=30,
                                 headers={"Accept": "application/vnd.github+json"})
        lic = info.get("license") or {}
        if not lic.get("spdx_id") or lic.get("spdx_id") == "NOASSERTION":
            return None
        return {"origin": {"repository": f"https://github.com/{repo}", "url": f"https://github.com/{repo}",
                           "version": rest.partition("/")[0] or None,
                           "determined_by": "GitHub API: the repository's detected license, which may "
                                            "not cover a release asset"},
                "license": {"name": lic.get("spdx_id"), "url": lic.get("url")}, "cite": []}
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
            HuggingFaceSource(), S3Source(), GCSSource(), GitHubReleaseSource()]


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


def input_record(spec, *, cache_dir=None, sources=None) -> dict:
    """What a result's provenance says about one input given on the command
    line. A remote input: the record its fetch left beside the bytes. A local
    file: its path and its digest - haversack cannot know where it came from,
    but it can pin what it was."""
    reg = registry(sources) if sources is not None else registry(default_sources() + [HttpSource()])
    parsed = parse_input(spec, known=reg)
    if parsed is None:
        rec = {"kind": "file", "identity": str(spec), "content": None, "origin": None,
               "license": None, "cite": [],
               "note": "a local file; its origin and license are not known to haversack"}
        try:
            rec["content"] = _content_facts(Path(spec))
        except Exception:                       # unreadable, or gone: the path still names it
            pass
        return rec
    kind, ident = parsed
    root = Path(cache_dir) if cache_dir else default_input_cache()
    # `input_entry_dir`, not a third copy of it: this one was missed when FETCH_EPOCH
    # went into the key, so `haversack` would have read provenance out of the entry a
    # PREVIOUS build downloaded while segmenting the one this build did - the two
    # disagreeing about the same identifier, silently
    rec = read_input_record(input_entry_dir(root, kind, ident))
    if rec is None:
        return {"kind": kind, "identity": f"{kind}:{ident}", "content": None, "origin": None,
                "license": None, "cite": [], "note": "no record of this fetch"}
    rec["identity"] = f"{kind}:{ident}"
    return rec


def input_entry_dir(root, kind: str, ident: str) -> Path:
    """Where the local input cache keeps one fetched input.

    THE derivation, called by `materialize` and by `cache_admin.input_entry` - which
    had its own copy, so putting FETCH_EPOCH into the key here made `cache clean
    inputs <spec>` silently match nothing while reporting success.

    FETCH_EPOCH is part of the key for the reason it exists: an entry a previous build
    downloaded is not found by this one, and is fetched again. The server's cache does
    the same at `SeriesCache._entry`.
    """
    import hashlib
    return Path(root) / kind / hashlib.sha1(
        f"e{FETCH_EPOCH}!{ident}".encode()).hexdigest()[:20]


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
    entry = input_entry_dir(root, kind, ident)
    done = entry / ".done"
    if not done.exists():
        import shutil
        import uuid

        from . import filelock
        # Fetch into a directory THIS caller owns and publish it by rename, under the
        # same advisory lock the model installs take.
        #
        # The protocol used to be: no `.done`, so delete the entry and start fetching.
        # That reads an absent marker as "the writer is dead", and a live download of a
        # large series looks exactly like abandoned work - so a second `haversack`
        # process (the input cache is one shared user-level root; running two commands
        # at once is ordinary) deleted the first one's tree from under it and fetched
        # into a directory the first still believed it owned. Both could then write
        # `.done` over contents assembled from two fetches, and `.done` holds the
        # identifier rather than a size or a digest, so nothing downstream could tell.
        # The loud variants were no better: `entry.mkdir(parents=True)` without
        # `exist_ok` raised, and two simultaneous `rmtree`s raced mid-walk.
        #
        # The staging name carries the pid and a random suffix, so even where no lock
        # facility exists (`filelock.SUPPORTED` False) two callers cannot share a
        # staging directory: the last rename wins and every published entry is still
        # internally consistent. Dotfile names keep both invisible to `cache clean`.
        entry.parent.mkdir(parents=True, exist_ok=True)
        what = ident if kind == "http" else f"{kind}:{ident}"
        if progress and (entry.parent / f".lock-{entry.name}").exists():
            # said BEFORE blocking: a second caller used to sit silent for the whole of
            # someone else's download, since the only progress line was inside the lock
            progress(f"waiting for another process to finish fetching {what}")
        with filelock.held(entry.parent / f".lock-{entry.name}") as locked:
            # ONLY under a real lock: nobody else owns this entry then, so any staging
            # left here is abandoned - a writer killed mid-fetch, which its own teardown
            # cannot catch. Swept here because these are dotfiles: `cache_admin._entries`
            # skips them, so `cache clean` cannot see the litter and `cache usage` counts
            # its bytes without counting it as an item. Before staging existed the
            # abandoned partial sat at `entry/` and both swept it; this restores "the next
            # fetch of the same input repairs it".
            #
            # Unlocked, this sweep is the opposite of a repair: `held` used to claim a
            # lock it did not have, and two callers on that path deleted each other's LIVE
            # downloads - the unique staging name is isolation only while nothing removes
            # every name that matches. Litter is left for a locked run to collect.
            if locked:
                for old_staging in entry.parent.glob(f".staging-{entry.name}-*"):
                    shutil.rmtree(old_staging, ignore_errors=True)
            if not done.exists():            # another process published while we waited
                staging = entry.parent / f".staging-{entry.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                try:
                    staging.mkdir(parents=True)
                    if progress:
                        progress(f"fetching {what}")
                    fetch_recording_origin(src, ident, staging, credentials)
                    (staging / ".done").write_text(f"{kind}:{ident}\n", encoding="utf-8")
                    if done.exists():
                        # published while we fetched - only reachable unlocked, and the
                        # reason the swap below re-checks rather than trusting the
                        # earlier one: theirs is complete, ours is redundant
                        shutil.rmtree(staging, ignore_errors=True)
                    else:
                        shutil.rmtree(entry, ignore_errors=True)  # a partial fetch: start over
                        os.replace(staging, entry)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
    got = entry / "series"
    if not got.is_dir():
        got = entry                          # a source that wrote directly under the entry
    return sole_file(got) or got
