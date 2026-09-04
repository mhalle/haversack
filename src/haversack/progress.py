"""Progress reporting and cooperative cancellation.

A caller driving a UI needs two things a plain function call cannot give: to know how far along
the work is, and to stop it. Both have to be *cooperative* - you cannot interrupt a blocking
torch call from outside - so the run checks a token and emits a snapshot at the one granularity
that already exists in the pipeline: between sliding-window patches.

:class:`Progress` is a frozen snapshot, so it is safe to hand to another thread and safe to
stash for a UI to poll. It prints readably, so an existing ``progress=lambda m: print(m)``
callback keeps working.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .errors import Cancelled

# Rough share of a run spent before/after the network, used only to make the fraction move
# sensibly during load and preprocess rather than sitting at 0 until the first patch.
# Where each stage starts, as a fraction of one model part. Sized from measured warm runs
# (read ~15 %, load ~10 %, preprocess ~5 %, network ~60 %, restore ~10 %) so the number
# moves with the work: it used to sit at 5 % through loading, preprocess and the start of
# predict and then jump to 95 % - on a one-patch volume that was the whole readout.
_STAGE_START = {"starting": 0.0, "queued": 0.0, "read": 0.0, "loading": 0.12, "preprocess": 0.22,
                "cascade": 0.22, "predict": 0.27, "restore": 0.90, "finalize": 0.97,
                "weights": 0.0}
_PREDICT_SPAN = _STAGE_START["restore"] - _STAGE_START["predict"]


@dataclass(frozen=True)
class Progress:
    """Where a run has got to. Immutable, so it crosses threads safely."""

    stage: str                       # loading | preprocess | predict | restore | finalize | queued
    detail: str = ""
    part: int = 0                    # 0-based index of the current model part
    n_parts: int = 1
    step: int = 0                    # patches done within this part
    n_steps: int = 0
    fraction: float = 0.0            # 0..1 over the whole run, best effort
    elapsed: float = 0.0

    def __str__(self) -> str:
        pct = f"{self.fraction * 100:3.0f}%"
        where = f" [{self.part + 1}/{self.n_parts}]" if self.n_parts > 1 else ""
        steps = f" {self.step}/{self.n_steps}" if self.n_steps else ""
        return f"{pct} {self.stage}{where}{steps} {self.detail}".rstrip()


class CancelToken:
    """A flag one thread sets and the running job polls. Idempotent and thread-safe."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise :class:`~haversack.errors.Cancelled` if the token has fired."""
        if self._event.is_set():
            raise Cancelled("segmentation cancelled")

    def __repr__(self) -> str:
        return f"CancelToken({'cancelled' if self.cancelled else 'active'})"


class Reporter:
    """Threads progress and cancellation through one run.

    Deliberately *not* stored on a :class:`~haversack.network.TorchModel`: models are shared through
    the cache, and per-run state on a shared object is how two jobs corrupt each other. A
    Reporter belongs to the call, and is passed down.
    """

    def __init__(self, progress=None, cancel: CancelToken | None = None, n_parts: int = 1):
        self._cb = progress
        self.cancel = cancel
        self.n_parts = max(1, int(n_parts))
        self.part = 0
        self.stage_name = "starting"
        self.t0 = time.perf_counter()
        self.last: Progress | None = None

    # -- the run calls these ------------------------------------------------
    def check(self) -> None:
        if self.cancel is not None:
            self.cancel.check()

    def enter_part(self, index: int, name: str = "") -> None:
        self.part = int(index)
        self.stage("loading", name)

    def stage(self, name: str, detail: str = "") -> None:
        self.stage_name = name
        self._emit(detail=detail, step=0, n_steps=0,
                   within=_STAGE_START.get(name, _STAGE_START["predict"]))

    def tick(self, step: int, n_steps: int, detail: str = "") -> None:
        """One patch done. Checks cancellation first - this is the interrupt point."""
        self.check()
        within = _STAGE_START["predict"] + (_PREDICT_SPAN * step / n_steps if n_steps else 0.0)
        self._emit(detail=detail, step=step, n_steps=n_steps, within=within)

    def advance(self, within: float, *, step: int = 0, n_steps: int = 0, detail: str = "") -> None:
        """Report a position inside the current part directly (``within`` in 0..1), for work
        whose stages are not the segmentation pipeline's - a weights install reports bytes
        received this way. Checks cancellation, like :meth:`tick`."""
        self.check()
        self._emit(detail=detail, step=step, n_steps=n_steps, within=within)

    # -- plumbing -----------------------------------------------------------
    def _emit(self, *, detail: str, step: int, n_steps: int, within: float) -> None:
        frac = (self.part + min(max(within, 0.0), 1.0)) / self.n_parts
        p = Progress(stage=self.stage_name, detail=detail, part=self.part, n_parts=self.n_parts,
                     step=step, n_steps=n_steps, fraction=min(max(frac, 0.0), 1.0),
                     elapsed=time.perf_counter() - self.t0)
        self.last = p
        if self._cb is not None:
            self._cb(p)

    @staticmethod
    def of(progress=None, cancel=None, n_parts: int = 1) -> "Reporter":
        """Accept a Reporter, a callback, or nothing, and always get a Reporter."""
        if isinstance(progress, Reporter):
            return progress
        return Reporter(progress=progress, cancel=cancel, n_parts=n_parts)


# Where a single weights install's phases sit within its part: the download is most of the
# wall time, unpacking is a short tail. A job's fraction is then bytes-driven, not a guess.
_DOWNLOAD_SPAN = 0.9
_UNPACK_AT = 0.95


class InstallProgress:
    """What a weights installer reports through.

    Installers used to take ``progress`` as a message callback, so a served install could say
    nothing finer than "weights" until it was done - a 230 MB fetch showed a client one static
    stage. This wraps whichever the caller passed - a job's :class:`Reporter`, a message
    callback, or nothing - so an installer writes one thing and a job gets stage / part /
    bytes / fraction snapshots while the CLI still gets its lines.

    It is callable, so a call site that still does ``progress("downloading x")`` keeps working.
    Messages go to the callback and the reporter's ``detail``; positions (``item``,
    ``download``, ``finished``) go to the reporter only, so CLI output is unchanged.
    """

    def __init__(self, progress=None):
        self.reporter = progress if isinstance(progress, Reporter) else None
        self._say = progress if (self.reporter is None and callable(progress)) else None
        self._within = 0.0
        self._step = 0
        self._n_steps = 0
        self._last_bytes = -1

    @classmethod
    def of(cls, progress=None) -> "InstallProgress":
        return progress if isinstance(progress, InstallProgress) else cls(progress)

    def __call__(self, message: str) -> None:
        self.say(message)

    def begin(self, n_items: int) -> None:
        """How many models this install covers - the reporter's parts."""
        if self.reporter is not None:
            self.reporter.n_parts = max(1, int(n_items))

    def item(self, index: int, detail: str = "") -> None:
        """Start model ``index`` of the install."""
        self._within, self._step, self._n_steps, self._last_bytes = 0.0, 0, 0, -1
        if self.reporter is not None:
            self.reporter.part = int(index)
            self.reporter.stage("weights", detail)

    def _advance(self, detail: str) -> None:
        r = self.reporter
        if r is not None:
            r.stage_name = "weights"          # whatever the job was doing, this is an install
            r.advance(self._within, step=self._step, n_steps=self._n_steps, detail=detail)

    def say(self, message: str) -> None:
        if self._say is not None:
            self._say(message)
        self._advance(message)

    def download(self, done: int, total: int, detail: str = "") -> None:
        """Bytes received so far of ``total`` (0 when the size is unknown). Emits on the first
        call, at completion, and otherwise every 2 % (at least 4 MiB) so a job's stream is
        not a snapshot per chunk."""
        step, n_steps = int(done), int(total)
        if 0 <= self._last_bytes and step < n_steps and \
                step - self._last_bytes < max(n_steps // 50, 4 << 20):
            return
        self._last_bytes = step
        self._step, self._n_steps = step, n_steps
        self._within = _DOWNLOAD_SPAN * (step / n_steps if n_steps else 0.0)
        self._advance(detail)

    def unpack(self, message: str) -> None:
        self._within = _UNPACK_AT
        self.say(message)

    def finished(self, detail: str = "") -> None:
        self._within = 1.0
        self._advance(detail)
