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
    reporter.stage("fetch", "cached (no-cache could not refresh: input in use)")
    on_skipped()
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
    cache.pin(key)
    try:
        return read_ahead.fill(key, cache.path(key))
    finally:
        cache.unpin(key)
