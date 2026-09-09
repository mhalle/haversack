"""Advisory exclusive file locks on POSIX and Windows.

Two places take one - the per-model install lock in :mod:`haversack.ecosystems`
and the one-writer lock on a ranked store - and both used to call ``fcntl``
directly, falling back to *no lock at all* where it could not be imported. That
fallback is exactly what runs on Windows, so two installs of one model there
could still delete each other's weights, which is the bug the lock exists to
stop. ``msvcrt.locking`` gives the same guarantee on Windows; this module hides
which one is underneath.

Locks are on a file descriptor and released by :func:`unlock` or by closing it.
On Windows the lock covers the first byte of the file (a region past EOF is
lockable, so an empty lock file works).
"""
from __future__ import annotations

import contextlib as _contextlib
import time

try:
    import fcntl as _fcntl
except ImportError:                          # Windows
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:                          # POSIX
    _msvcrt = None

#: Whether this platform can lock at all. When False, :func:`lock` returns True
#: without locking, so a caller degrades to unlocked rather than refusing -
#: which is no worse than before, and is reported here so a caller can say so.
SUPPORTED = _fcntl is not None or _msvcrt is not None


def _fileno(handle) -> int:
    return handle if isinstance(handle, int) else handle.fileno()


def lock(handle, *, blocking: bool = True, poll_s: float = 0.1) -> bool:
    """Take the exclusive lock. Returns True once held; False only when
    ``blocking`` is False and another holder has it."""
    fd = _fileno(handle)
    if _fcntl is not None:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | (0 if blocking else _fcntl.LOCK_NB))
        except BlockingIOError:
            return False
        return True
    if _msvcrt is not None:
        # LK_LOCK retries ten times over ~10 s and then raises, which is the
        # wrong shape for waiting out a multi-minute install; poll LK_NBLCK
        # instead so the blocking form waits as long as it must.
        while True:
            try:
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if not blocking:
                    return False
                time.sleep(poll_s)
    return True                              # no lock facility: proceed unlocked


def unlock(handle) -> None:
    fd = _fileno(handle)
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        elif _msvcrt is not None:
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    except OSError:
        pass                                 # closing the descriptor releases it anyway


@_contextlib.contextmanager
def held(path, *, mode: int = 0o644):
    """Hold an exclusive advisory lock on ``path`` for the block.

    Yields False only when the lock FILE could not be opened - a read-only cache
    directory - and True otherwise, so a caller degrades to unlocked rather than
    refusing, which is no worse than having no lock. Note the asymmetry: on a
    platform with no lock facility at all (:data:`SUPPORTED` False) this yields
    True while holding nothing, because :func:`lock` returns True unlocked. Read
    :data:`SUPPORTED` if you need to tell those apart; the yielded value cannot.
    The lock file is created if missing and never removed: unlinking it would let
    the next process create a different inode and lock that instead, which is a
    lock that locks nothing.

    Written once here because two callers had the same twenty lines - the
    per-model install lock in :mod:`haversack.ecosystems` and, from 2026-09-08,
    the input cache's fetch claim in :mod:`haversack.sources`.
    """
    import os
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, mode)
        lock(fd)
    except OSError:
        if fd is not None:
            os.close(fd)                     # opened but not locked: do not leak the fd
        fd = None
    try:
        yield fd is not None
    finally:
        if fd is not None:
            try:
                unlock(fd)
            finally:
                os.close(fd)
