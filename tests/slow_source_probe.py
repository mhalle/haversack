"""One `materialize` of a deliberately slow source, for the two-process race test.

Run as a script: `probe.py MARKER CACHE_DIR IDENT [--signal PATH] [--await PATH]`.
`--signal` is touched once the fetch has begun; `--await` blocks until that path
exists before `materialize` is called. Together they make the interleaving that
matters DETERMINISTIC - the second caller enters while the first is mid-fetch -
rather than leaving it to process start-up jitter, which is what a first version
of this did, and it never reproduced the race it was written for.
"""
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

    def __init__(self, marker="?", pause=0.25, signal=None):
        self.marker, self.pause, self.signal = marker, pause, signal

    def fetch(self, identifier, dest_dir, *, credentials=None):
        d = Path(dest_dir) / "series"
        d.mkdir()
        for i in range(self.SLICES):
            (d / f"{i}.dcm").write_text(self.marker, encoding="utf-8")
            if i == 0 and self.signal:
                Path(self.signal).write_text("fetching\n", encoding="utf-8")
            time.sleep(self.pause)
        return d


def main(argv):
    marker, cache_dir, ident = argv[1], argv[2], argv[3]
    signal = awaited = None
    if "--signal" in argv:
        signal = argv[argv.index("--signal") + 1]
    if "--await" in argv:
        awaited = Path(argv[argv.index("--await") + 1])
        deadline = time.monotonic() + 60
        while not awaited.exists():
            if time.monotonic() > deadline:
                raise SystemExit("timed out waiting for the other writer to start")
            time.sleep(0.01)
    got = sources.materialize(f"slow:{ident}", cache_dir=cache_dir,
                              sources=[SlowSource(marker=marker, signal=signal)])
    print(got)


if __name__ == "__main__":
    main(sys.argv)
