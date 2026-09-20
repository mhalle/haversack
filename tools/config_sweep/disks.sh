#!/bin/zsh
# cfg5: a FULL cache disk.  cfg6: a cache root on exFAT (no hard links, case-insensitive).
# Both on real filesystems via hdiutil, the recipe AGENTS.md already uses for the no-link
# fallback - a mocked ENOSPC is not the same thing.
set -e
WT=/Users/halazar/Dropbox/development/haversack/.claude/worktrees/object-store-cache
PY=$WT/.venv/bin/python
SCRATCH=/private/tmp/claude-501/-Users-halazar-Dropbox-development-haversack/ae9ea27e-04b1-43f2-a1bf-37aa46ae9cbf/scratchpad
export AWS_ACCESS_KEY_ID="$(security find-generic-password -s haversack-r2 -a AWS_ACCESS_KEY_ID -w)"
export AWS_SECRET_ACCESS_KEY="$(security find-generic-password -s haversack-r2 -a AWS_SECRET_ACCESS_KEY -w)"
export AWS_ENDPOINT=https://cfe95016c86e1beb8275e81704948996.r2.cloudflarestorage.com
export AWS_REGION=auto
cd $WT

make_disk() {  # name fs size
  local img=$SCRATCH/$1.dmg
  rm -f $img
  hdiutil create -quiet -size $3 -fs $2 -volname $1 $img >/dev/null
  hdiutil attach -quiet -nobrowse $img >/dev/null
  echo /Volumes/$1
}

cleanup() {
  hdiutil detach -quiet /Volumes/CFG5 2>/dev/null || true
  hdiutil detach -quiet /Volumes/CFG6 2>/dev/null || true
}
trap cleanup EXIT

echo "=== cfg5: a full cache disk (HFS+, 12 MB)"
FULL=$(make_disk CFG5 "HFS+" 12m)
$PY $SCRATCH/cfg5_full_disk.py "$FULL"

echo "=== cfg6: a cache root on exFAT (no hard links, case-insensitive)"
EXFAT=$(make_disk CFG6 ExFAT 40m)
$PY $SCRATCH/cfg6_exfat.py "$EXFAT"
