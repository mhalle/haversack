"""The segments every task produces - each one's label value, its layer where the output
overlaps, and the id its model gives it - mined from where the model states them into one JSON
index, ``data/segments.json``.

A **segment** is one item of a task's segment table, as in a ``.seg.nrrd``, a DICOM
segmentation or a duckn ``seg`` leaf: the label ``value`` it is written with, the ``layer`` it
lives in when the output overlaps (absent for the single labelmap most tasks write), and its
``id`` - the model's own token for it: ``kidney_left``, ``Left-Cerebral-White-Matter``, and for
two TotalVibe models nothing but numbers. The id is a code in that model's class list at its
pinned version, not a display name and not an identity across models: ``kidney_left`` in two
catalogs is two ids in two coding systems. Search groups ids that fold alike, which is a match
on spelling and never a claim that two models mean one thing. Segments need not be disjoint -
an overlapping output is segments in different layers, not a different kind of thing.

What the record leaves room for, in duckn's own shape (2026-09-13): a group segment with
``members`` (and ``disjoint``/``exhaustive`` only where a maker states them), a display
``name``, coded ``designations``. Every id mined so far is a valid duckn id.

Why a second copy of labels, in a package whose rule is that the checkpoint is the spec: that
rule governs what RUNS. A manifest still never holds labels, and an installed model's own
``dataset.json`` still decides every result. This index answers the questions asked before
anything is installed - which tasks produce a pancreas, and with what label value - which the
catalog cannot answer for most of its tasks without downloading their weights. So it is
derived and never consulted to run anything, and every record carries what it was read from
and the version that pins it (:meth:`haversack.ecosystems.ModelEcosystem.label_version`). The
moment a catalog's answer differs, the record is stale, and ``haversack catalog check`` - and
the suite - say so. Searching it is ``haversack tasks --find`` and ``GET /v1/segments``; the
CLI keeps the noun off its top level, where ``segments`` sat one letter from ``segment``, the
verb that runs a model (2026-09-13).

Mining belongs to the ecosystem (:meth:`~haversack.ecosystems.ModelEcosystem.label_listing`),
not to this module (2026-09-12). A catalog on a shared shape - a zip manifest, an image-baked
engine - inherits it, so a new catalog needs no entry here, and one that answers nothing fails
``tests/test_segments.py`` naming the method it has to write. This module chooses targets,
merges, writes and searches.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import InputError

SCHEMA_VERSION = 1
PACKAGED = Path(__file__).parent / "data" / "segments.json"
KINDS = ("segments", "open")
_META = {
    "schema_version": SCHEMA_VERSION,
    "generator": "haversack catalog mine",
    "note": ("Derived, and never used to run anything: each record lists the segments a task "
             "produces - label value, layer where its output overlaps, and the id its model gives "
             "each - read from where the model states them (a checkpoint's dataset.json, a "
             "bundle's metadata.json, an engine's own table), and `version` is what pins them. An "
             "installed model's own labels always decide a result. Regenerate with `haversack "
             "catalog mine --all`; `haversack catalog check` finds stale records."),
}


def user_path() -> Path:
    """``HAVERSACK_SEGMENTS``, else ``$XDG_CONFIG_HOME/haversack/segments.json`` - where an
    installed package's miner writes, as ``weights refresh`` writes its manifest."""
    env = os.environ.get("HAVERSACK_SEGMENTS")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "haversack" / "segments.json"


def target() -> Path:
    """The packaged index in a source checkout (to be committed), the user's otherwise - the
    rule ``weights refresh`` follows, for its reason: site-packages is not the user's to edit
    and does not survive an upgrade."""
    from .weights_fetch import _is_checkout
    return PACKAGED if _is_checkout() else user_path()


class HttpReader:
    """The network half of mining, through :mod:`haversack.fetchlib`: a remote zip as a
    random-access file (:class:`haversack.sources.RangeFile`, so a checkpoint's dataset.json
    costs a few small Range reads rather than its gigabytes), and a JSON document with its
    response headers. Ecosystems see only this interface, which is what lets a test walk every
    catalog offline."""

    #: Small blocks: what is read is a central directory and one small member, and the
    #: RangeFile default (4 MB) would pull ~8 MB per archive to read a few kilobytes.
    block = 1 << 16

    def zip(self, url: str):
        import zipfile
        from .sources import RangeFile
        return zipfile.ZipFile(RangeFile(url, self._size(url), block=self.block, max_blocks=256))

    def json(self, url: str) -> tuple:
        from . import fetchlib
        with fetchlib.open(url, timeout=60) as r:
            return json.load(r), {k.lower(): v for k, v in r.headers.items()}

    @staticmethod
    def _size(url: str) -> int:
        """The archive's size, from a one-byte Range read - which also proves the host
        honors Range before anything larger is asked of it."""
        from . import fetchlib
        with fetchlib.open(url, timeout=120, headers={"Range": "bytes=0-0"}) as r:
            total = (r.headers.get("Content-Range") or "").rpartition("/")[2]
            honored = r.status == 206
        if not honored or not total.isdigit():
            raise InputError(f"{url}: the host did not honor a Range request, so its archive "
                             "cannot be read without downloading it whole")
        return int(total)


def load(path) -> dict:
    """The index at ``path``, or an empty one. A file of another schema is refused rather than
    merged into, since a merge would mix two shapes in one file."""
    p = Path(path)
    if not p.is_file():
        return {"_meta": dict(_META), "tasks": {}}
    raw = json.loads(p.read_text(encoding="utf-8"))
    have = (raw.get("_meta") or {}).get("schema_version") if isinstance(raw, dict) else None
    if have != SCHEMA_VERSION or not isinstance(raw.get("tasks"), dict):
        raise InputError(f"{p}: a segments index of schema {have!r}; this haversack reads "
                         f"{SCHEMA_VERSION} - regenerate it with `haversack catalog mine --all`")
    return raw


def dump(records: dict, path) -> None:
    """Write the index atomically, records sorted by task, so a diff of the file is a diff of
    the catalog."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"_meta": dict(_META), "tasks": {k: records[k] for k in sorted(records)}}
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def plan(targets, *, all_: bool = False, ecosystems=None) -> list:
    """``[(ecosystem, [task, ...], whole)]`` for what the command line named. ``whole`` means
    a catalog was named, so its records of tasks it no longer offers are dropped."""
    from .ecosystems import RENAMED_ECOSYSTEMS, RENAMED_TASKS, known_ecosystems
    ecos = {e.name: e for e in (known_ecosystems() if ecosystems is None else ecosystems)}
    targets = list(targets or ())
    if all_ == bool(targets):
        raise InputError("name what to mine: --all, or one or more catalogs or tasks "
                         "(ts.v2, moose:clin_ct_organs) - not both")
    if all_:
        return [(e, list(e.tasks()), True) for e in ecos.values()]
    chosen: dict[str, tuple] = {}
    for t in targets:
        if "@" in t:
            raise InputError(f"{t}: the miner reads the version each catalog offers now and "
                             "records it with the list; a pinned @version is not mined")
        name, sep, task = t.partition(":")
        if name in RENAMED_ECOSYSTEMS:
            new = RENAMED_ECOSYSTEMS[name]
            raise InputError(f"catalog {name!r} is now {new!r}: use {new}{sep}{task}")
        if t in RENAMED_TASKS:
            raise InputError(f"task {t!r} is now {RENAMED_TASKS[t]!r}")
        eco = ecos.get(name)
        if eco is None:
            raise InputError(f"unknown catalog {name!r}; known: {', '.join(sorted(ecos))}")
        offered = list(eco.tasks())
        if not sep:
            chosen[name] = (eco, offered, True)
            continue
        if task not in offered:
            raise InputError(f"unknown task {t!r}; {name} offers {len(offered)}, e.g. "
                             f"{', '.join(offered[:5])}")
        prev = chosen.get(name)
        if prev is None:
            chosen[name] = (eco, [task], False)
        elif not prev[2] and task not in prev[1]:
            prev[1].append(task)
    return list(chosen.values())


def mine(plan, path, *, root=None, reader=None, write: bool = True, prune: bool = False,
         workers: int = 8, today: str | None = None) -> dict:
    """Mine every task in ``plan`` and merge the records into the index at ``path``.

    Records the run did not ask for are left exactly as they are. A task whose segments are
    unchanged keeps its record, stamp included, so re-mining everything rewrites nothing that
    did not change. A task that fails keeps its previous record, if it had one, and fails the
    run: a transient 503 must not delete a good list, and a failure must not look like
    success. ``prune`` (``--all``) also drops records of catalogs this build no longer has.
    ``root`` is the weights root whose installed copies are held against their archives.
    """
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date
    from . import __version__
    reader = reader or HttpReader()
    records = dict(load(path)["tasks"])
    jobs = [(eco, t) for eco, tasks, _ in plan for t in tasks]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs) or 1))) as pool:
        outcomes = list(pool.map(lambda job: _mine_one(job[0], job[1], root, reader), jobs))
    stamp = {"at": today or date.today().isoformat(), "by": f"haversack {__version__}"}
    results = []
    for (eco, t), (record, problem, note) in zip(jobs, outcomes):
        name = f"{eco.name}:{t}"
        old = records.get(name)
        if problem:
            kept = f" - kept the record mined {(old.get('mined') or {}).get('at')}" if old else ""
            results.append((name, "failed", problem + kept))
            continue
        said = f" ({note})" if note else ""
        if old is not None and _content(old) == _content(record):
            results.append((name, "unchanged", _summary(old) + said))
            continue
        records[name] = {**record, "mined": stamp}
        results.append((name, "added" if old is None else "changed",
                        (_summary(record) if old is None else _diff(old, record)) + said))
    whole = {eco.name: set(tasks) for eco, tasks, is_whole in plan if is_whole}
    planned = {eco.name for eco, _, _ in plan}
    for name in sorted(records):
        eco_name, _, short = name.partition(":")
        if eco_name in whole and short not in whole[eco_name]:
            del records[name]
            results.append((name, "removed", "no longer in its catalog"))
        elif prune and eco_name not in planned:
            del records[name]
            results.append((name, "removed", "its catalog is no longer in this build"))
    changed = any(s in ("added", "changed", "removed") for _, s, _ in results)
    if write and changed:
        dump(records, path)
    return {"path": str(path), "results": results, "written": bool(write and changed),
            "failed": any(s == "failed" for _, s, _ in results)}


def check(path, *, ecosystems=None, targets=None) -> list:
    """``[(task, status, detail)]`` comparing the index at ``path`` with this build's catalogs,
    offline: ``ok``, ``stale`` (the version that pins it moved), ``missing`` (never mined),
    ``orphan`` (no such task any more) or ``error`` (its catalog cannot say).

    Everything by default - it is offline and instant, which is what CI wants. ``targets``
    narrows it, in the grammar ``mine`` takes: a named catalog's orphans are still reported,
    since the catalog was asked about whole; a named task has none."""
    from .ecosystems import known_ecosystems
    records = load(path)["tasks"]
    ecos = known_ecosystems() if ecosystems is None else list(ecosystems)
    chosen = (plan(targets, ecosystems=ecos) if targets
              else [(e, list(e.tasks()), True) for e in ecos])
    whole = {eco.name for eco, _, is_whole in chosen if is_whole}
    out, live = [], set()
    for eco, tasks, _ in chosen:
        for t in tasks:
            name = f"{eco.name}:{t}"
            live.add(name)
            rec = records.get(name)
            try:
                version = json.loads(json.dumps(eco.label_version(t)))
            except Exception as e:                  # noqa: BLE001 - one row of the report
                out.append((name, "error", f"{type(e).__name__}: {e}"))
                continue
            if rec is None:
                out.append((name, "missing", "never mined"))
            elif rec.get("version") != version:
                out.append((name, "stale", _version_changes(rec.get("version") or {}, version)))
            else:
                out.append((name, "ok", _summary(rec)))
    out += [(n, "orphan", "no such task in this build's catalogs")
            for n in sorted(set(records) - live)
            if not targets or n.partition(":")[0] in whole]
    return out


def _mine_one(eco, task: str, root, reader) -> tuple:
    """``(record, problem, note)`` - never raises: one task's failure is one line of the
    report, not the end of the run."""
    try:
        version = eco.label_version(task)
        listing = dict(eco.label_listing(task, root, reader))
        installed = listing.pop("installed", None)
        record = _record(eco, task, version, listing)
    except Exception as e:                          # noqa: BLE001 - reported, and the run fails
        return None, f"{type(e).__name__}: {e}", None
    note = None
    if installed:
        if installed.get("compared") and not installed.get("agrees"):
            return None, (f"the installed copy (version {installed.get('version')}) names its "
                          "segments differently from its published archive at the same version - "
                          "one of them changed under that version; not recorded"), None
        if installed.get("compared"):
            note = "installed copy agrees"
        elif installed.get("note"):
            note = installed["note"]
        else:
            note = f"installed copy is version {installed.get('version') or 'unknown'}, not compared"
    return record, None, note


def _record(eco, task: str, version: dict, listing: dict) -> dict:
    kind = listing.get("kind")
    if kind not in KINDS:
        raise ValueError(f"listing kind {kind!r} is not one of {KINDS}")
    rec = {"task": f"{eco.name}:{task}", "ecosystem": eco.name, "engine": eco.engine,
           "kind": kind, "modality": listing.get("modality"), "version": dict(version),
           "source": dict(listing.get("source") or {})}
    if kind == "segments":
        rec["segments"] = _segment_table(listing.get("segments"))
    if listing.get("note"):
        rec["note"] = str(listing["note"])
    return json.loads(json.dumps(rec))              # the shape it has once read back


def _segment_table(listed) -> list:
    """The listed segments, checked and ordered: each an ``id`` and a label ``value``, with a
    ``layer`` only where it is not 0 (duckn's default, and the single labelmap most tasks
    write). A layer and a value name one segment - duckn's leaves never share a value within
    a layer - so two that do are refused rather than one silently shadowing the other."""
    table, seen = [], set()
    for s in listed or ():
        if not isinstance(s, dict) or not str(s.get("id") or "").strip() or s.get("value") is None:
            raise ValueError(f"a segment needs an id and a label value, not {s!r}")
        seg = {"id": str(s["id"]), "value": int(s["value"])}
        if int(s.get("layer") or 0):
            seg["layer"] = int(s["layer"])
        where = (seg.get("layer", 0), seg["value"])
        if where in seen:
            raise ValueError(f"two segments share layer {where[0]} value {where[1]}")
        seen.add(where)
        table.append(seg)
    if not table:
        raise ValueError("its source names no segments")
    return sorted(table, key=lambda s: (s.get("layer", 0), s["value"]))


def _content(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "mined"}


def _summary(rec: dict) -> str:
    what = (f"{len(rec.get('segments') or ())} segments" if rec.get("kind") == "segments"
            else "open vocabulary")
    return f"{what}, {_version_text(rec.get('version') or {})}"


def _version_text(v: dict) -> str:
    for key in ("tag", "bundle_version", "ts_version"):
        if v.get(key):
            return f"{key.replace('_', ' ')} {v[key]}"
    engine = v.get("engine") or []
    if engine:
        return "engine " + ", ".join(f"{e.get('id')} {e.get('version')}" for e in engine)
    return "unversioned"


def _short(value) -> str:
    text = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    return text if len(text) <= 24 else text[:21] + "..."


def _version_changes(old: dict, new: dict) -> str:
    return "; ".join(f"{k} {_short(old.get(k))} -> {_short(new.get(k))}"
                     for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k))


def _diff(old: dict, new: dict) -> str:
    def table(rec):
        return {(s.get("layer", 0), s["value"]): s["id"] for s in rec.get("segments") or ()}

    parts = [p for p in [_version_changes(old.get("version") or {}, new.get("version") or {})] if p]
    ol, nl = table(old), table(new)
    added, gone = set(nl) - set(ol), set(ol) - set(nl)
    renamed = [k for k in set(ol) & set(nl) if ol[k] != nl[k]]
    if added:
        parts.append(f"+{len(added)} segments")
    if gone:
        parts.append(f"-{len(gone)} segments")
    if renamed:
        parts.append(f"{len(renamed)} renamed")
    parts += [f"{k} changed" for k in ("kind", "modality", "source", "note") if old.get(k) != new.get(k)]
    return "; ".join(parts) or "changed"


# -- search -------------------------------------------------------------------------------
#: How a query is read. ``words`` (the default) matches word prefixes in any order, ``glob``
#: a shell pattern over the whole key, ``regex`` Python's syntax anywhere in the key.
MODES = ("words", "glob", "regex")
#: What the server offers. A regular expression from an anonymous caller can take unbounded
#: time to evaluate (catastrophic backtracking) and Python's ``re`` cannot be interrupted;
#: word and glob matching cannot go that way (2026-09-12).
WIRE_MODES = ("words", "glob")
MAX_QUERY = 200
_OPEN_NOTE = ("these tasks have no fixed segment list - they segment what a caller names in a "
              "prompt - so they may segment this too")


def fold(text) -> str:
    """An id's search key: NFC, casefolded, spaces and hyphens to underscores, runs of
    underscores collapsed. ``Left-Cerebral-White-Matter`` and ``left cerebral white matter``
    are one key; the model's own spelling is always kept beside it."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFC", str(text)).casefold()
    return re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", t)).strip("_")


def words(key: str) -> list:
    """The words of a key. Letter-digit tokens stay whole - ``vertebrae_l1`` is ``vertebrae``
    and ``l1`` - because split further, ``l`` would be a word and a prefix of every id that
    starts with ``left``."""
    return [w for w in str(key).split("_") if w]


def _fold_glob(pattern: str) -> str:
    """A glob folded like a key, outside brackets only: in ``[s-u]`` the hyphen is a range."""
    import unicodedata
    out, depth = [], 0
    for ch in unicodedata.normalize("NFC", pattern).casefold():
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif depth == 0 and (ch.isspace() or ch == "-"):
            ch = "_"
        out.append(ch)
    return "".join(out)


class Index:
    """Every segment of every record, searchable. A few thousand short strings: a sorted
    vocabulary answers word prefixes and a linear scan answers globs, both in milliseconds,
    so no search engine is warranted. Not an ontology - an abbreviation or a synonym is not
    found, and nothing here pretends otherwise."""

    def __init__(self, records: dict):
        self.entries: list[dict] = []
        self.open: list[dict] = []
        postings: dict[str, set] = {}
        for task, rec in sorted(records.items()):
            base = {"task": task, "ecosystem": rec.get("ecosystem") or task.partition(":")[0],
                    "modality": rec.get("modality"), "kind": rec.get("kind")}
            if rec.get("kind") == "open":
                self.open.append(base)
                continue
            for s in rec.get("segments") or ():
                entry = {**base, "value": s["value"], "id": str(s["id"]), "key": fold(s["id"])}
                if s.get("layer"):
                    entry["layer"] = s["layer"]
                for w in words(entry["key"]):
                    postings.setdefault(w, set()).add(len(self.entries))
                self.entries.append(entry)
        self._postings = postings
        self._vocabulary = sorted(postings)
        self.catalogs = sorted({e["ecosystem"] for e in self.entries} |
                               {o["ecosystem"] for o in self.open})

    def search(self, query, *, mode: str = "words", field: str = "key", catalog=None,
               modality=None, tasks=None, limit: int | None = None,
               allowed_modes=MODES) -> dict:
        """Segments matching ``query``, grouped by folded id, each group with every task and
        label value that has it. ``tasks`` limits the answer to those tasks (a server passes
        the ones it serves); ``allowed_modes`` is what the caller may use (the server: no
        regex). ``field="id"`` matches a glob or regex against the model's own spelling."""
        import bisect
        import fnmatch
        import re
        query = str(query or "").strip()
        if mode not in allowed_modes:
            if mode == "regex":
                raise InputError("regex is not offered here - a pattern can take unbounded time "
                                 "to evaluate; use mode=words or mode=glob, or run `haversack "
                                 "tasks --find PATTERN --regex` locally")
            raise InputError(f"mode {mode!r}: use {' or '.join(allowed_modes)}")
        if field not in ("key", "id"):
            raise InputError(f"field {field!r}: use key (folded) or id (as the model writes it)")
        if mode == "words" and field != "key":
            raise InputError("word search reads the folded key; field=id is for glob and regex")
        if not query:
            raise InputError("an empty query matches nothing: give words, a glob or a pattern")
        if len(query) > MAX_QUERY:
            raise InputError(f"the query is {len(query)} characters; the limit is {MAX_QUERY}")
        if limit is not None and limit < 1:
            raise InputError("limit must be at least 1")
        if catalog is not None and catalog not in self.catalogs:
            raise InputError(f"unknown catalog {catalog!r}; the index has "
                             f"{', '.join(self.catalogs)}")

        def keep(e) -> bool:
            return ((catalog is None or e["ecosystem"] == catalog)
                    and (modality is None or str(modality).casefold()
                         in str(e.get("modality") or "").casefold())
                    and (tasks is None or e["task"] in tasks))

        if mode == "words":
            terms = words(fold(query))
            if not terms:
                raise InputError(f"{query!r} holds no words to search for")
            hit = None
            for term in terms:
                found: set = set()
                i = bisect.bisect_left(self._vocabulary, term)
                while i < len(self._vocabulary) and self._vocabulary[i].startswith(term):
                    found |= self._postings[self._vocabulary[i]]
                    i += 1
                hit = found if hit is None else hit & found
            matched = [i for i in sorted(hit or ()) if keep(self.entries[i])]
        elif mode == "glob":
            pattern = _fold_glob(query) if field == "key" else query
            matched = [i for i, e in enumerate(self.entries)
                       if keep(e) and fnmatch.fnmatchcase(e[field], pattern)]
        else:
            try:
                rx = re.compile(query, re.IGNORECASE)
            except re.error as e:
                raise InputError(f"regex {query!r}: {e}") from None
            matched = [i for i, e in enumerate(self.entries) if keep(e) and rx.search(e[field])]

        groups: dict[str, dict] = {}
        for i in matched:
            e = self.entries[i]
            g = groups.setdefault(e["key"], {"key": e["key"], "ids": [], "segments": []})
            if e["id"] not in g["ids"]:
                g["ids"].append(e["id"])
            seg = {k: e[k] for k in ("task", "value", "id", "modality")}
            if "layer" in e:
                seg["layer"] = e["layer"]
            g["segments"].append(seg)
        exact = fold(query)
        ordered = sorted(groups.values(), key=lambda g: (g["key"] != exact, g["key"]))
        truncated = limit is not None and len(ordered) > limit
        opened = [o["task"] for o in self.open if keep(o)]
        out = {"query": query, "mode": mode, "field": field, "keys": len(groups),
               "segments": len(matched), "truncated": truncated,
               "results": ordered[:limit] if truncated else ordered,
               "open_vocabulary": opened}
        if opened:
            out["open_vocabulary_note"] = _OPEN_NOTE
        return out


_INDEX_CACHE: dict = {}


def index(paths=None) -> Index:
    """The searchable index: the packaged file with the user's laid over it - a user record
    replaces the packaged one for its task, as the user's weights manifest overlays the
    packaged one. Rebuilt only when one of the files changes, so a server pays for it once."""
    import threading
    lock = _INDEX_CACHE.setdefault("_lock", threading.Lock())
    chosen, seen = [], set()
    for p in ([PACKAGED, user_path()] if paths is None else paths):
        p = Path(p)
        if p.resolve() not in seen:
            seen.add(p.resolve())
            chosen.append(p)

    def state(p: Path):
        try:
            st = p.stat()
            return (str(p), st.st_mtime_ns, st.st_size)
        except OSError:
            return (str(p), None, None)

    stamp = tuple(state(p) for p in chosen)
    with lock:
        built = _INDEX_CACHE.get(stamp)
        if built is None:
            records: dict = {}
            for p in chosen:
                if p.is_file():
                    records.update(load(p)["tasks"])
            built = Index(records)
            for k in [k for k in _INDEX_CACHE if k != "_lock"]:
                del _INDEX_CACHE[k]
            _INDEX_CACHE[stamp] = built
    return built
