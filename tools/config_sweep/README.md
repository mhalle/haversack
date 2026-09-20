# The configuration sweep

Six environments the shared result store has to survive, each as a script you can re-run.
They exist because four rounds of review could attack the CODE and not the ENVIRONMENT, and
three of these six found defects the reviewers could not have (2026-09-20). A reviewer then
pointed out that results nobody can reproduce are not evidence, which is why they are here
rather than in a scratch directory.

Run them against a bucket you do not mind writing to; each cleans up its own prefix.

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT=https://<account>.r2.cloudflarestorage.com AWS_REGION=auto
uv run --no-sync python tools/config_sweep/cfg1_shared_cachedir.py 30
uv run --no-sync python tools/config_sweep/cfg2_version_skew.py
uv run --no-sync python tools/config_sweep/cfg3_kill_reader.py
uv run --no-sync python tools/config_sweep/cfg4_slow_store.py      # no bucket needed
zsh tools/config_sweep/disks.sh                                    # cfg5 + cfg6, hdiutil
```

| script | the configuration | what it found first time |
|---|---|---|
| `cfg1_shared_cachedir.py` | two servers over ONE `--cache-dir` (the per-GPU deployment) | nothing |
| `cfg2_version_skew.py` | two haversack versions against one bucket | an older host DELETED a newer one's entry |
| `cfg3_kill_reader.py` | a reader SIGKILLed mid-fill | its work directory survived an hour, invisible to `cache clean` |
| `cfg4_slow_store.py` | a store that is slow, not broken | nothing (health 0.01 s vs a 1.5 s store route) |
| `cfg5_full_disk.py` | a cache disk with no room left (12 MB image) | a fill needed TWICE the result's size free |
| `cfg6_exfat.py` | a cache root on exFAT: no hard links, case-insensitive | nothing |
| `cfg7_purge_race.py` | a delete loop against a deduplicating publisher, one shared blob | nothing - and it measured R2's timestamps as sub-second, so the whole-second waiting path is dormant there |

`disks.sh` builds and mounts the two disk images with `hdiutil`, the same recipe
`AGENTS.md` uses for the no-hard-link tests, and detaches them on the way out.

**What none of these cover:** a cache root on a network filesystem, Modal, a store with
object versioning, and a bucket big enough for `list` and `sweep` to cost real money.
