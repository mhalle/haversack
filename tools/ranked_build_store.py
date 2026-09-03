"""Build a duckn store from a `ranked_emit.py` output directory.

The implementation lives in the package as ``haversack.ranked_build`` (moved 2026-09-03 so the
CLI can build stores itself); this file is only its command line.

usage: uv run python tools/ranked_build_store.py RANKED_DIR OUT.duckn[.zip] CASE [all|last]
"""
from haversack.ranked_build import main

if __name__ == "__main__":
    main()
