"""Run a task through the PRODUCT path and keep its ranked output distribution.

The implementation lives in the package as ``haversack.ranked_output`` (moved 2026-09-03);
this file is only its command line.

usage: uv run python tools/ranked_emit.py IMAGE TASK OUTDIR [--depth N] [--clip C] [--envelope-mm MM|none]
"""
from haversack.ranked_output import main_cli

if __name__ == "__main__":
    main_cli()
