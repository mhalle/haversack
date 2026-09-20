#!/bin/zsh
# cfg5: a cache disk that runs out of room.  cfg6: a cache root on exFAT (no hard links,
# case-insensitive).  Both on real filesystems via hdiutil - the recipe AGENTS.md already
# uses for the no-hard-link tests, because a mocked ENOSPC is not the same thing.
#
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_ENDPOINT=... AWS_REGION=auto
#   export HAVERSACK_TEST_STORE=s3://your-bucket/haversack-sweep
#   zsh tools/config_sweep/disks.sh
#
# On a Mac the credentials are conveniently kept in the keychain and read inline:
#   export AWS_SECRET_ACCESS_KEY="$(security find-generic-password -s my-store \
#                                   -a AWS_SECRET_ACCESS_KEY -w)"
set -e
: ${HAVERSACK_TEST_STORE:?set it, e.g. s3://your-bucket/haversack-sweep}

HERE=${0:A:h}
ROOT=${HERE:h:h}                      # tools/config_sweep -> the repository
PY=${PYTHON:-$ROOT/.venv/bin/python}
IMAGES=$(mktemp -d)
cd $ROOT

make_disk() {                          # name fs size
  hdiutil create -quiet -size $3 -fs $2 -volname $1 $IMAGES/$1.dmg >/dev/null
  hdiutil attach -quiet -nobrowse $IMAGES/$1.dmg >/dev/null
  echo /Volumes/$1
}

cleanup() {
  hdiutil detach -quiet /Volumes/HVCFG5 2>/dev/null || true
  hdiutil detach -quiet /Volumes/HVCFG6 2>/dev/null || true
  rm -rf $IMAGES
}
trap cleanup EXIT

echo "=== cfg5: a full cache disk (HFS+, 12 MB)"
$PY $HERE/cfg5_full_disk.py "$(make_disk HVCFG5 "HFS+" 12m)"

echo "=== cfg6: a cache root on exFAT (no hard links, case-insensitive)"
$PY $HERE/cfg6_exfat.py "$(make_disk HVCFG6 ExFAT 40m)"
