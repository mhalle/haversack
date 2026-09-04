"""Provision model weights on demand from their official source.

TotalSegmentator publishes each dataset as a zip on its own GitHub releases; the manifest
(``data/ts_weights.json``, id -> url + optional sha256) maps a weights id to its asset. This
module downloads and unpacks the ones a task needs into a weights root, skipping any already
present - so a fresh machine (or a cloud volume) fills its cache on first run and never
re-downloads. Nothing is redistributed by us: the URLs point at wasserth's releases.

Stdlib only (urllib + zipfile), so importing haversack never pulls a download stack it will not use.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MANIFEST = Path(__file__).parent / "data" / "ts_weights.json"


def _sort_key(item):
    """Numeric ids sort numerically; any non-numeric id sorts after them, alphabetically."""
    k = item[0]
    return (0, int(k), "") if str(k).isdigit() else (1, 0, str(k))


# Written into each unpacked model folder by fetch_one. A sidecar rather than one index at the
# weights root: only the fetch that created the folder writes it, so concurrent fetches into a
# shared volume cannot race, and the record travels with the folder if it is copied elsewhere.
SIDECAR = ".haversack-version.json"
# Written by nnseg before the rename (2026-09-02). Read, never written: an installed model
# folder keeps its record without a re-download.
LEGACY_SIDECARS = (".nnseg-version.json",)


def installed_version(folder) -> dict | None:
    """What :func:`fetch_one` recorded when it installed this model folder, if anything.

    ``None`` means haversack did not install it - TotalSegmentator may have, or it was copied in by
    hand. That is reported as unknown rather than guessed at from the manifest: guessing would
    be wrong in exactly the case versioning exists for, where an older version is on disk and
    the manifest has since moved on.
    """
    f = Path(folder)
    for cand in [d / name for d in (f, f.parent)        # accept a model folder or its dataset dir
                 for name in (SIDECAR, *LEGACY_SIDECARS)]:
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                return None
    return None


def _write_sidecar(dest: Path, weights_id, tag: str, entry: dict, sha256: str | None) -> None:
    import datetime
    rec = {"id": dataset_key(weights_id), "tag": tag, "sha256": sha256 or entry.get("sha256"),
           "url": entry.get("url"), "name": entry.get("name"),
           "installed": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "by": "haversack"}
    try:
        (dest / SIDECAR).write_text(json.dumps(rec, indent=2) + "\n")
    except OSError:                                   # a read-only weights root must not fail a fetch
        pass


def dataset_key(weights_id) -> str:
    """Canonical manifest key for a dataset id: unpadded decimal.

    TotalSegmentator publishes both ``Dataset008_HepaticVessel`` and ``Dataset297_...``, so the
    same dataset can be written 8 or 008. Without canonicalizing, those become two entries for
    one model.
    """
    t = str(weights_id).strip()
    return str(int(t)) if t.isdigit() else t


def user_manifest_path() -> Path:
    """The user's own manifest: ``HAVERSACK_TS_MANIFEST``, else
    ``$XDG_CONFIG_HOME/haversack/ts_weights.json`` (``~/.config/haversack/...``).

    A user should not wait for a release to pick up weights TotalSegmentator has published:
    ``haversack weights refresh`` from an installed package writes here, and the packaged
    manifest is read with this one laid over it (2026-09-03). It survives upgrades, which a
    refresh into site-packages did not."""
    env = os.environ.get("HAVERSACK_TS_MANIFEST")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "haversack" / "ts_weights.json"


def _read_manifest_file(path) -> dict:
    raw = json.loads(Path(path).read_text())
    # `raw.get("weights") or raw` would fall through to the wrapper for an EMPTY manifest,
    # leaking the key "weights" in as a dataset id. Test for the key, not its truthiness.
    return _normalize(raw["weights"] if "weights" in raw else raw)


def _manifest(path=None) -> dict:
    """A named file alone, or (``None``) the packaged manifest with the user's laid over it:
    a dataset the user's file names replaces the packaged entry for that id."""
    if path is not None:
        return _read_manifest_file(path)
    merged = _read_manifest_file(MANIFEST)
    user = user_manifest_path()
    if user.is_file():
        merged.update(_read_manifest_file(user))
    return merged


def manifest_sources() -> dict:
    """Where the effective manifest's entries come from, for `weights coverage`."""
    packaged = _read_manifest_file(MANIFEST)
    user = user_manifest_path()
    mine = _read_manifest_file(user) if user.is_file() else {}
    return {"package": len(packaged), "user": len(mine), "user_path": str(user),
            "user_overrides": sorted(w for w in mine if w in packaged)}


def _is_checkout() -> bool:
    """Whether this package is a source checkout (a refresh then edits the repository's
    manifest, to be committed) rather than an installed copy (a refresh goes to the user's)."""
    here = MANIFEST.resolve()
    if "site-packages" in here.parts or "dist-packages" in here.parts:
        return False
    return any((parent / ".git").exists() for parent in here.parents)


def refresh_target() -> Path:
    return MANIFEST if _is_checkout() else user_manifest_path()


def _normalize(entries: dict) -> dict:
    """Accept both manifest shapes and return the current one.

    An entry is ``{"default": tag, "versions": {tag: {...}}}``: what upstream published is a
    *fact* that refresh may rewrite freely, while which one to install is a *decision* that only
    a human changes. Keeping them parallel means switching versions is a one-token edit and no
    version is a special case.

    ``default``, not ``current``: in a file listing several versions "current" reads as "the
    newest", which is wrong here - Dataset297's default is v2.0.0 while v2.0.4 exists, because
    v2.0.0 is what TotalSegmentator installs. It is the version you get without asking for one.

    Legacy shapes - a flat ``{"url": ...}`` entry, or the earlier ``current`` key - are lifted
    into this shape on read, so an old manifest still works.
    """
    out: dict = {}
    for raw, e in entries.items():
        wid = dataset_key(raw)
        if isinstance(e, dict) and "versions" in e:
            if "current" in e and "default" not in e:      # the key was named `current` before
                e = {"default": e["current"], "versions": e["versions"]}
        else:
            e = e if isinstance(e, dict) else {"url": e}
            tag = e.get("tag") or "unversioned"
            e = {"default": tag, "versions": {tag: e}}
        if wid in out:                                # 8 and 008 are the same dataset
            merged = dict(out[wid]["versions"]); merged.update(e["versions"])
            keep = out[wid]["default"] if out[wid]["default"] != "unversioned" else e["default"]
            out[wid] = {"default": keep if keep in merged else next(iter(merged)), "versions": merged}
        else:
            out[wid] = e
    return out


def selected(entry: dict, tag: str | None = None) -> dict:
    """The version of an entry that would be installed - ``tag`` if given, else the default."""
    versions = entry["versions"]
    want = tag or entry.get("default")
    if want not in versions:
        raise KeyError(f"version {want!r} not in manifest; have {sorted(versions)}")
    return versions[want]


def _content_length(response) -> int:
    """The byte size a download will be, or 0 when the server did not say."""
    headers = getattr(response, "headers", None)
    try:
        return int(headers.get("Content-Length") or 0) if headers is not None else 0
    except (TypeError, ValueError, AttributeError):
        return 0


def is_present(weights_id, root) -> bool:
    from .tasks import _dataset_dirs
    return bool(_dataset_dirs(Path(root), weights_id))


def _no_entry(weights_id) -> "ModelNotFound":
    from .errors import ModelNotFound
    wid = str(weights_id)
    if wid in LICENSE_GATED:
        return ModelNotFound(
            f"Dataset{wid} ({LICENSE_GATED[wid]}) is not a public TotalSegmentator release asset - "
            f"it is served from TotalSegmentator's licensed backend. Obtain a license from "
            f"totalsegmentator.com, run `totalseg_set_license`, and let TotalSegmentator download "
            f"it into the weights root; haversack will then find it.")
    return ModelNotFound(
        f"no manifest entry for Dataset{wid}; try `haversack weights refresh` to pick up newly "
        f"published weights, or place the model folder under the weights root yourself")


def fetch_one(weights_id, root, *, tag: str | None = None, progress=None) -> Path:
    """Download and unpack one dataset into ``root`` if it is not already there.

    ``tag`` installs a specific published version instead of the manifest's ``current`` one.

    Returns the ``Dataset{id}_*`` directory. Verifies sha256 when the manifest gives one.
    Unpacks to a temp dir and moves into place, so an interrupted download never leaves a
    half-populated model folder that ``is_present`` would accept.
    """
    root = Path(root)
    from .progress import InstallProgress
    from .tasks import _dataset_dirs
    say = InstallProgress.of(progress)
    existing = _dataset_dirs(root, weights_id)
    if existing:
        say.finished(f"Dataset{weights_id} present")
        return existing[0]
    entry = _manifest().get(dataset_key(weights_id))
    if entry is None:
        raise _no_entry(weights_id)
    chosen_tag = tag or entry.get("default")
    chosen = selected(entry, tag)
    if not chosen.get("url"):                        # a placeholder, e.g. a license-gated dataset
        raise _no_entry(weights_id)
    url, expected = chosen["url"], chosen.get("sha256")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        archive = tmp / f"Dataset{weights_id}.zip"
        what = f"downloading Dataset{weights_id} from {url.rsplit('/', 1)[-1]}"
        say(what)
        h = hashlib.sha256()
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
            total = _content_length(r) or int(chosen.get("size") or 0)
            done = 0
            say.download(done, total, what)
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                say.download(done, total, what)
            say.download(done, done, what)            # closes the bar when the size was unknown
        if expected and h.hexdigest() != expected:
            raise ValueError(f"sha256 mismatch for Dataset{weights_id}: {h.hexdigest()} != {expected}")
        say.unpack(f"unpacking Dataset{weights_id}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        prefix = f"Dataset{int(str(weights_id)):03d}" if str(weights_id).isdigit() else f"Dataset{weights_id}"
        unpacked = next(p for p in tmp.iterdir() if p.is_dir()
                        and (p.name.startswith(f"Dataset{weights_id}") or p.name.startswith(prefix)))
        _write_sidecar(unpacked, weights_id, chosen_tag, chosen, h.hexdigest())
        dest = root / unpacked.name
        say.finished(f"Dataset{weights_id} installed")
        os.replace(unpacked, dest)                    # sidecar is inside, so the move stays atomic
    return dest


def ensure_task_weights(task, root, *, catalog=None, progress=None, tag=None,
                        _seen=None) -> list[Path]:
    """Fetch every model a task needs. Recurses through cascade ``crop_from_task`` stages, so a
    task that crops from another task (teeth <- craniofacial_structures) pulls that chain too.
    Idempotent."""
    from .progress import InstallProgress
    from .tasks import TaskCatalog
    cat = catalog or TaskCatalog("ts")
    spec = cat.get(task) if isinstance(task, str) else task
    chain = _weights_chain(spec, cat, tag, _seen if _seen is not None else set())
    say = InstallProgress.of(progress)
    say.begin(len(chain))                 # the parts a job reports: one per model
    paths = []
    for i, (wid, wtag) in enumerate(chain):
        say.item(i, f"Dataset{wid}")
        paths.append(fetch_one(wid, root, tag=wtag, progress=say))
    return paths


def _weights_chain(spec, cat, tag, seen) -> list[tuple]:
    """Every ``(weights_id, tag)`` a task needs, in install order: the task's own models, then
    each cascade crop-from task's chain. A version pin applies to the task's own models only;
    a crop-from task installs its default, as it always did."""
    chain = [(wid, tag) for wid in spec.weights_ids]
    for st in spec.cascade:
        if st.crop_from_task and st.crop_from_task not in seen:
            seen.add(st.crop_from_task)
            chain += _weights_chain(cat.get(st.crop_from_task), cat, None, seen)
    return chain


# -- keeping the manifest current ------------------------------------------------------------
# The manifest is the provisioning mechanism for any machine that does not already have weights
# on disk - a cloud volume, a fresh container, someone else's laptop - so a gap in it means a
# task simply cannot run there. TotalSegmentator publishes weights as release assets, and adds
# to them over time, so the manifest has to be refreshable rather than hand-maintained.
TS_REPO = "wasserth/TotalSegmentator"

# TotalSegmentator serves some models from its own licensed backend rather than as public
# release assets (``commercial_models`` in totalsegmentator/map_to_binary.py): a POST to
# backend.totalsegmentator.com carrying a license number, not a URL anyone can fetch. A URL
# manifest structurally cannot cover them, so they are recorded here and reported as
# "needs a license" rather than "missing" - the difference between an actionable message and
# what looks like a broken manifest.
LICENSE_GATED = {
    "301": "heartchambers_highres", "303": "face", "304": "appendicular_bones",
    "409": "brain_structures", "481": "tissue_types", "485": "tissue_4_types",
    "507": "coronary_arteries_LEGACY", "509": "coronary_arteries", "514": "pulmonary_artery_landmarks",
    "710": "renal_arteries", "713": "aorta_annulus", "716": "aortic_dissection",
    "855": "appendicular_bones_mr", "856": "face_mr", "857": "thigh_shoulder_muscles",
    "920": "aortic_sinuses", "925": "tissue_types_mr",
}
ASSET_RE = re.compile(r"^Dataset(\d+)_.*\.zip$")


def _api(url: str, token: str | None = None) -> list:
    """One GitHub API GET, following pagination. Stdlib only, like the rest of this module."""
    out, page = [], 1
    while True:
        req = urllib.request.Request(f"{url}?per_page=100&page={page}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "haversack"})
        tok = token or os.environ.get("GITHUB_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=60) as r:
            batch = json.loads(r.read())
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def discover_release_assets(repo: str = TS_REPO, *, token: str | None = None,
                            progress=None) -> dict[str, dict]:
    """Every ``Dataset<id>_*.zip`` published as a release asset, newest release first.

    Returns ``{weights id: {url, name, tag, size, sha256?}}``. When a dataset appears in more
    than one release the newest wins, which is what TotalSegmentator itself would install.
    Unauthenticated GitHub allows 60 requests/hour; set ``GITHUB_TOKEN`` to lift that.
    """
    say = progress or (lambda s: None)
    say(f"listing releases of {repo}")
    releases = _api(f"https://api.github.com/repos/{repo}/releases", token)
    releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    found: dict[str, dict] = {}
    for rel in releases:                              # newest first, so current defaults to newest
        tag = rel.get("tag_name", "")
        for a in rel.get("assets") or ():
            m = ASSET_RE.match(a.get("name", ""))
            if not m:
                continue
            wid = m.group(1)
            entry = {"url": a["browser_download_url"], "name": a["name"], "size": a.get("size")}
            digest = a.get("digest") or ""            # "sha256:..." on newer GitHub API responses
            if digest.startswith("sha256:"):
                entry["sha256"] = digest.split(":", 1)[1]
            slot = found.setdefault(dataset_key(wid), {"default": tag, "versions": {}})
            slot["versions"].setdefault(tag, entry)
    n_ver = sum(len(v["versions"]) for v in found.values())
    say(f"found {len(found)} datasets ({n_ver} published versions) across {len(releases)} releases")
    return found


PINS_URL = "https://raw.githubusercontent.com/{repo}/master/totalsegmentator/map_tasks_config.py"
PIN_RE = re.compile(r'(\d+):\s*\{[^}]*?"version":\s*"([^"]+)"', re.S)


def upstream_pins(repo: str = TS_REPO, *, progress=None) -> dict[str, str]:
    """Which release TotalSegmentator itself installs for each dataset.

    TS records this in ``map_tasks_config.py`` and it is the right meaning for ``current``:
    the version whose behavior their documentation and published numbers describe. It is not
    always the newest asset - Dataset297 is published as v2.0.0 and v2.0.4 while TS installs
    v2.0.0 - so "newest wins" would silently diverge from the ecosystem haversack exists to match.

    Read over HTTP rather than by importing totalsegmentator, so haversack keeps no dependency on
    it. Best effort: an unreachable file yields ``{}`` and the caller falls back to newest.
    """
    say = progress or (lambda s: None)
    try:
        req = urllib.request.Request(PINS_URL.format(repo=repo), headers={"User-Agent": "haversack"})
        with urllib.request.urlopen(req, timeout=60) as r:
            src = r.read().decode("utf-8", "replace")
    except Exception as e:                            # noqa: BLE001 - advisory, never fatal
        say(f"  ! could not read {repo}'s version pins ({type(e).__name__}); falling back to newest")
        return {}
    pins = {str(int(m.group(1))): m.group(2) for m in PIN_RE.finditer(src)}
    say(f"read {len(pins)} version pins from {repo}")
    return pins


def refresh_manifest(path=None, *, repo: str = TS_REPO, token: str | None = None,
                     add_missing: bool = True, update_existing: bool = False,
                     write: bool = True, progress=None) -> dict:
    """Merge newly published weights into the manifest.

    Adds entries that are absent. Existing entries are **left alone** unless
    ``update_existing``: repointing a dataset at a newer release changes which weights get
    downloaded, and therefore the segmentations - not something to do silently. The return value
    reports what changed either way, so ``write=False`` is a dry run.

    ``path=None`` picks the target by where the package lives (:func:`refresh_target`): the
    repository's manifest in a checkout, the user's own file from an installed copy. The
    user's file is written whole (the effective manifest plus the news), so it stands alone.
    """
    say = progress or (lambda s: None)
    if path is None:
        path = refresh_target()
    current = _manifest() if Path(path) == user_manifest_path() else _manifest(path)
    upstream = discover_release_assets(repo, token=token, progress=progress)
    pins = upstream_pins(repo, progress=progress)
    for wid, up in upstream.items():                  # prefer TS's own pin over "newest asset"
        if pins.get(wid) in up["versions"]:
            up["default"] = pins[wid]

    merged = {w: {"default": e["default"], "versions": dict(e["versions"])} for w, e in current.items()}
    added, new_versions, repointed, migrated = {}, {}, {}, {}
    for wid, up in upstream.items():
        if wid not in merged:
            if add_missing:
                added[wid] = up
                merged[wid] = {"default": up["default"], "versions": dict(up["versions"])}
            continue
        slot = merged[wid]
        fresh = {t: v for t, v in up["versions"].items() if t not in slot["versions"]}
        if fresh:
            new_versions[wid] = sorted(fresh)
            slot["versions"].update(fresh)            # facts: always kept up to date
        # A legacy flat entry carries no tag. Name it by matching its URL against what upstream
        # published - NOT by adopting upstream's newest, which would silently repoint it (297 is
        # published as both v2.0.0 and v2.0.4, and TotalSegmentator itself pins v2.0.0).
        if slot.get("default") == "unversioned":
            was = (slot["versions"].get("unversioned") or {}).get("url")
            match = next((t for t, v in slot["versions"].items()
                          if t != "unversioned" and v.get("url") == was), None)
            if match:
                slot["versions"].pop("unversioned")
                slot["default"] = match
                migrated[wid] = match
            else:
                say(f"  ! Dataset{wid}: current URL matches no published asset; leaving as-is")
        elif update_existing and slot["default"] != up["default"]:
            slot["default"] = up["default"]              # a decision, never made silently
            repointed[wid] = up["default"]

    behind = {w: (merged[w]["default"], upstream[w]["default"])
              for w in upstream if w in merged and merged[w]["default"] != upstream[w]["default"]}
    say(f"manifest: {len(current)} -> {len(merged)} datasets; {len(added)} added, "
        f"{len(new_versions)} gained versions, {len(behind)} differ from "
        f"{'TotalSegmentator' if pins else 'upstream newest'}"
        f"{' (repointed)' if update_existing else ' (left alone)'}")
    if write and merged != current:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"weights": dict(sorted(merged.items(), key=_sort_key))}, indent=2) + "\n")
        say(f"wrote {path}")
    elif write:
        say(f"nothing to write ({path} is current)")
    return {"added": added, "new_versions": new_versions, "behind_upstream": behind,
            "migrated": migrated,
            "total": len(merged), "path": str(path)}


def coverage(catalog=None) -> dict:
    """Which catalog tasks the manifest can provision, which need a license, which are missing.

    Three outcomes, not two: a task haversack cannot download because TotalSegmentator gates it
    behind a license is a different situation from one whose URL we simply do not have, and a
    caller (or a UI) should be able to tell them apart.
    """
    from .tasks import TaskCatalog
    cat = catalog or TaskCatalog("ts")
    have = _manifest()
    ok, licensed, missing = [], {}, {}
    for name in cat.names():
        absent = [dataset_key(w) for w in cat.get(name).weights_ids if dataset_key(w) not in have]
        if not absent:
            ok.append(name)
        elif all(w in LICENSE_GATED for w in absent):
            licensed[name] = absent
        else:
            missing[name] = absent
    return {"covered": ok, "license_required": licensed, "missing": missing,
            "n_weights": len(have), "n_tasks": len(cat), "sources": manifest_sources()}
