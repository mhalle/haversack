"""One `materialize` of a deliberately slow source, for the input-cache race tests.

Run as a script:
    probe.py MARKER CACHE_DIR IDENT [--signal P] [--await P] [--events P]

`--signal` is touched once the fetch has begun; `--await` blocks until that path
exists before `materialize` is called. Together they make the interleaving that
matters DETERMINISTIC - the second caller enters while the first is mid-fetch -
rather than leaving it to process start-up jitter, which is what a first version
of this did, and it never reproduced the race it was written for.

`--events` appends `MARKER EVENT TIME` for each fetch's start and end. That is what
lets the test assert MUTUAL EXCLUSION rather than an outcome: comparing published
bytes only catches the corrupt interleavings that actually happened, which a review
measured at 4 runs in 10, and could not tell the shipped protocol apart from one
with no lock at all.
"""
import os
import sys
import time
from pathlib import Path

from haversack import sources


class SlowSource(sources.DataSource):
    """Writes SLICES slices with a pause between them, so two fetches overlap.

    Every slice carries the writer's marker, so a directory holding two markers is
    a MIXED entry - the outcome the claim protocol exists to prevent.
    """

    prefix = "slow"
    id_pattern = r"[a-z0-9]+"
    description = "test double: a fetch slow enough to overlap another"
    SLICES = 6

    def __init__(self, marker="?", pause=0.25, signal=None, events=None):
        self.marker, self.pause = marker, pause
        self.signal, self.events = signal, events

    def _note(self, event):
        if not self.events:
            return
        line = f"{self.marker} {event} {time.time():.6f}\n"
        fd = os.open(self.events, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode())          # O_APPEND: no interleaving to worry about
        finally:
            os.close(fd)

    def fetch(self, identifier, dest_dir, *, credentials=None):
        d = Path(dest_dir) / "series"
        d.mkdir()
        self._note("fetch-start")
        for i in range(self.SLICES):
            (d / f"{i}.dcm").write_text(self.marker, encoding="utf-8")
            if i == 0 and self.signal:
                Path(self.signal).write_text("fetching\n", encoding="utf-8")
            time.sleep(self.pause)
        self._note("fetch-end")
        return d


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def main(argv):
    marker, cache_dir, ident = argv[1], argv[2], argv[3]
    awaited = _opt(argv, "--await")
    if awaited:
        deadline = time.monotonic() + 60
        while not Path(awaited).exists():
            if time.monotonic() > deadline:
                raise SystemExit("timed out waiting for the other writer to start")
            time.sleep(0.01)
    got = sources.materialize(
        f"slow:{ident}", cache_dir=cache_dir,
        sources=[SlowSource(marker=marker, signal=_opt(argv, "--signal"),
                            events=_opt(argv, "--events"))])
    print(got)


if __name__ == "__main__":
    main(sys.argv)
