"""Decisions both executors have to make the same way.

``LocalExecutor`` (``serve``) and the Modal worker (``modal_app``) already share
their *data structures* - ``SeriesCache``, ``ReadAhead``, ``ResultCache``,
``result_key``, ``publish_completion`` all come from ``serve``. What they did not
share was the *policy* around them: the same rule written twice, in different
vocabulary, so a fix to one read as unrelated code in the other.

That duplication does not show up in a text search, which is what makes it
dangerous - a shingle comparison of the two modules finds exactly one shared
five-line window. The rules matched only because someone remembered.

This module is where such a rule goes: no HTTP, no Modal, no job store, just the
decision, with its collaborators passed in. Both callers stay thin, and
``tests/test_jobpolicy.py`` fails if either grows its own copy again.
"""
from __future__ import annotations

from typing import NamedTuple

#: The job protocol's terminal states - the one place they are written.
#:
#: They had five independent definitions: serve.TERMINAL, jobstore.TERMINAL,
#: modal_app._TERMINAL, an inline literal inside modal_app._purgeable, and a
#: local in client.wait. Adding a sixth state would have needed five edits, and
#: missing one fails silently and asymmetrically - a client that polls forever,
#: a purge that never collects, a cancel that a progress update resurrects.
TERMINAL = ("done", "failed", "cancelled")


def purgeable(meta, now: float, ttl_s: float) -> bool:
    """May this job record be dropped?

    Terminal and older than the TTL. Queued and running records are NEVER purged
    by age: a stale active record is a symptom to surface, not to tidy away.
    Garbage that is not a record at all is purgeable.

    The sqlite jobstore cannot call this - it decides in SQL, over rows it never
    loads - so ``reap`` states the same rule as a WHERE clause built from the
    same TERMINAL tuple, and tests/test_jobpolicy.py checks the two agree case
    by case rather than trusting that they read alike.
    """
    if not isinstance(meta, dict):
        return True
    if meta.get("state") not in TERMINAL:
        return False
    return (now - float(meta.get("finished") or meta.get("created") or now)) > ttl_s


def refresh_cached_input(key: str, *, wanted: bool, cache, reporter, on_skipped,
                         read_ahead=None, already: set | None = None) -> bool:
    """Drop a cached input - and any image already read from it - when the caller
    asked for a recompute. Returns whether the input is now fresh.

    Both halves are needed. Dropping the cached bytes alone re-downloads the
    series and then segments the pre-read image anyway, because the read-ahead is
    keyed by series and nothing else invalidates it: ``no-cache`` would pay for a
    download and return the stale answer regardless.

    This matters most for the sources whose identity is not content-pinned.
    ``s3:`` keys and ``github:`` release assets can both be replaced under one
    identifier, so a forced recompute has to re-read the source or it answers the
    same wrong thing again.

    Call it BEFORE the job takes its own pin, since a pin is what makes a discard
    refuse.

    :param wanted: whether this job asked for fresh bytes at all.
    :param cache: a ``SeriesCache`` - needs ``has`` and ``discard``.
    :param reporter: gets ``stage("fetch", ...)`` so the caller can see which way
        it went; the deployments differ in how they *carry* that, not in whether.
    :param on_skipped: called when the discard was refused, to record it wherever
        this deployment keeps job state. A caller that asked for fresh bytes and
        did not get them must be able to see that from either deployment.
    :param read_ahead: optional; the Modal worker may have none.
    :param already: identifiers refreshed by this same job. A multi-input task may
        bind two roles to ONE identifier; without this the second pass sees the
        first pass's fresh bytes cached under its own pin and reports "could not
        refresh" for a key this very job just refreshed.
    """
    if not wanted:
        return True
    if already is not None:
        if key in already:
            return True
        already.add(key)
    if not cache.has(key):
        if read_ahead is not None:
            read_ahead.pop(key)
        return True                    # nothing cached: the fetch itself IS the refresh
    if cache.discard(key):
        if read_ahead is not None:
            read_ahead.pop(key)        # the image read from those bytes is stale too
        reporter.stage("fetch", "refetching (no-cache)")
        return True
    # Refused: another job is reading those bytes, or a writer holds the claim.
    # Say so rather than publish a result computed from bytes the caller
    # explicitly asked not to reuse - and leave the read-ahead alone, because it
    # belongs to the job that is still using it.
    # Record the skip BEFORE the stage message. `stage` publishes a status
    # snapshot, and on the local server the flag is only carried once it is set -
    # this order round the other way emitted the message without the flag, so a
    # client acting on that one snapshot could not see the refresh had failed.
    on_skipped()
    reporter.stage("fetch", "cached (no-cache could not refresh: input in use)")
    return False


def fill_read_ahead(key: str, *, read_ahead, cache=None, path=None) -> bool:
    """Pre-read an input into the read-ahead, holding a pin for the whole read.

    The pin is the point. A committed cache entry is unpinned between the writer
    releasing its claim and the job that wants it taking one, and the read-ahead
    reads in exactly that window - so without this a concurrent ``discard`` (a
    ``no-cache`` on the same series) or an LRU ``_evict`` under budget pressure
    renames the entry into the graveyard and deletes it *while this read is
    walking it*.

    A DICOM series is a directory that ``ImageSeriesReader`` opens file by file,
    so the failure is not a clean error on one handle: it is a partial read of a
    series that no longer exists. Both deployments had this window, which is why
    it is fixed here rather than twice.

    ``path`` is for bytes that are already local (an upload): nothing to pin,
    because nothing else can evict them.
    """
    if path is not None:
        return read_ahead.fill(key, path)
    if cache is None:
        raise ValueError(f"fill_read_ahead({key!r}) needs either a cache to pin "
                         "or a local path; got neither")
    cache.pin(key)
    try:
        return read_ahead.fill(key, cache.path(key))
    finally:
        cache.unpin(key)


class SourceKey(NamedTuple):
    """How one job input is addressed in the series cache."""
    kind: str
    ident: str
    key: str


def source_cache_key(source) -> SourceKey | None:
    """The series-cache identity of one job input, or None when it has none.

    Six places built this key by hand - three in serve, three in modal_app - and
    the prefetcher had to agree with the dispatcher on every one of them or it
    warmed a slot nothing ever looked up. That failure is silent: no error, just
    a download paid for twice and a cache that never hits.

    They also did not quite agree. The multi-input paths coerced a missing
    identifier with ``str(... or "")`` and the single-input paths did not, so the
    same absent id produced ``"idc:"`` on one path and the literal ``"idc:None"``
    on the other. This settles on the coercing spelling.

    Returns None for inputs that are not series-cached at all: an ``upload`` is
    local bytes, and an ``input`` is resolved through the content store.
    """
    src = source or {}
    kind = src.get("kind", "upload")
    if kind in ("upload", "input"):
        return None
    ident = str(src.get("id") or src.get("crdc_series_uuid") or "")
    return SourceKey(kind, ident, f"{kind}:{ident}")


def prefetchable(*, state, kind, refresh_input, sources) -> bool:
    """May this job's input be staged ahead of its turn?

    The prefetcher runs beside the current job and warms the NEXT one's input:
    downloads its series into the series cache, then reads the image into the
    read-ahead. Four kinds of job must be left alone, and both deployments had
    to know all four - the local server selected the head of its queue, the
    Modal worker scanned its whole jobs dict, and the two lists of exclusions
    had drifted apart by two entries before this function existed.

    - Not queued, or a ``prepare``: nothing to stage.
    - A job that asked for fresh bytes (``refresh_input``). Pre-reading them
      warms an image the job is about to discard, and worse: the pre-read PINS
      for the duration, ``discard`` refuses on any pin without being able to say
      whose, so the job's own refresh failed and it served the stale answer -
      on ``s3:`` and ``github:``, whose bytes can change under one identifier,
      which is the case ``no-cache`` exists for.
    - A multi-input job. The read-ahead holds ONE volume, so pre-reading a
      fraction of this job's inputs evicts a useful entry to save a fraction of
      one read, and the multi-input stager deliberately never pops it.
    - An ``input`` source: resolved through the content store, so there is no
      series to stage and no local file to pre-read either. Treating it as an
      upload once killed the prefetch thread on ``cache.pin(None)``.
    """
    if state != "queued" or kind == "prepare":
        return False
    if refresh_input:
        return False
    entries = list(sources or [{"kind": "upload"}])
    if len(entries) > 1:
        return False
    return entries[0].get("kind", "upload") != "input"


def take_pre_read(read_ahead, key: str, *, fresh_bytes_wanted: bool):
    """Claim the pre-read image for ``key`` - or None when this job must not use one.

    A job that asked for fresh bytes gets None even if an image is there, and
    the slot is emptied either way. Popping in the refresh step alone was not
    enough: the prefetch thread refills the slot with no synchronization, and a
    refill landing after that pop reinstalls the stale image - the fetch is paid
    for and the stale answer returned anyway. The local server had this guard
    and the Modal worker did not, which is why it lives here now.
    """
    image = read_ahead.pop(key)
    return None if fresh_bytes_wanted else image


def input_rights(sources, identities, series_cache) -> list:
    """What a result was computed from, and under what terms: one record per
    input, in the order the job bound them.

    A remote input's rights were looked up when it was fetched and recorded
    beside its bytes (``sources.fetch_recording_rights``); this reads them back
    from the series cache. An upload has no origin haversack can know, and says
    so. Written once because both deployments must answer identically - a
    result cached from either is read by both.

    :param sources: the job's ``source`` entries.
    :param identities: the job's ``input_identity`` list, in the same order
        (``ROLE=identity`` strings for a multi-input job).
    """
    from .sources import read_rights
    out = []
    entries = list(sources or [{"kind": "upload"}])
    ids = list(identities or [])
    for i, entry in enumerate(entries):
        ident = ids[i] if i < len(ids) else None
        ident = ident.partition("=")[2] if ident and "=" in ident else ident
        kind = entry.get("kind", "upload")
        rec = {"kind": kind, "identity": ident, "rights": None}
        if entry.get("role"):
            rec["role"] = entry["role"]
        sk = source_cache_key(entry)
        if sk is None:
            rec["note"] = ("uploaded by the caller; its origin and license are not known "
                           "to haversack" if kind == "upload"
                           else "content the caller stored; its origin is not known to haversack")
        else:
            side = read_rights(series_cache.entry(sk.key)) or {}
            rec["rights"] = side.get("rights")
            if side.get("error"):
                rec["rights_error"] = side["error"]
            if rec["rights"] is None and "rights_error" not in rec:
                rec["note"] = "the source could not determine this input's license"
        out.append(rec)
    return out


def record_inputs(seg, sources, identities, series_cache) -> None:
    """Write :func:`input_rights` into a finished result's provenance, in place.

    In place, because ``Segmentation`` is a frozen dataclass: assigning the
    field raises, and the test doubles that stand in for it are not frozen, so
    that slip passed every server test and failed on the first real run. The
    dict inside is mutable, and every engine builds one.
    """
    prov = getattr(seg, "provenance", None)
    if isinstance(prov, dict):
        prov["inputs"] = input_rights(sources, identities, series_cache)
