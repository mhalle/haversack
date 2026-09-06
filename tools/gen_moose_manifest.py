"""Regenerate src/haversack/data/moose_weights.json from a moosez checkout.

The checkpoint is the spec (labels come from each model's own dataset.json at
install time); this manifest holds only what the checkpoint cannot know -
the task name, where to download it, and the release tag parsed from the
asset filename. Run after updating upstream/MOOSE.

**Every entry in the registry is accounted for.** The registry is parsed with a
pattern, and a pattern matches what it was written against: the name group was
``[a-z0-9_]+`` and upstream's ``clin_ct_ALPACA``, ``clin_ct_PUMA``,
``clin_ct_PUMA4`` and ``clin_mr_FVM`` fell through it without a word, so the
manifest offered 21 of 25 models and nothing said so. Now every ``KEY_URL:``
block in the file must parse as an entry, and one that does not stops the run -
a future layout change (a swapped key order, a name character the pattern does
not admit, a value that is not a string literal) is an error here rather than a
model that quietly goes missing.

**A model that is not offered is recorded with a reason, and the reason is
checked against the asset**, the way tools/gen_totalvibe_manifest.py records its
exclusions: `NOT_OFFERED` names the entry and a category, the generator reads the
archive's layout by Range to confirm the category is true, and a reason that has
stopped being true is an error rather than prose that quietly rots. An entry
that leaves the registry while still listed here is an error too.

Every offered URL is checked with a HEAD before the manifest is written, and one
that does not answer 200 stops the run. The offered assets are never read - the
other generators read each zip's layout by Range, which fails on a dead URL by
itself, and this one has no such reason to touch them - so without the check a
URL that had stopped serving would be recorded and found by a user mid-install.
The HEAD is sent with the same kind of User-Agent the installer sends
(tools/zippeek.py says why that matters).

Usage (stdlib only):
    uv run --no-project python tools/gen_moose_manifest.py [path/to/MOOSE]
"""
import json
import re
import sys
from pathlib import Path

import zippeek

DEST = Path(__file__).parent.parent / "src/haversack/data/moose_weights.json"

#: Registry entries not offered as tasks, and why. The reason is a category the
#: generator checks against the asset; see REASONS.
NOT_OFFERED = {
    # The zip's top level is ``clin_mr_FVM/``, with ``Dataset501_FVM`` one
    # level down. haversack's installer refuses an archive whose top-level name
    # is not the manifest folder (moving it into place would replace whatever
    # owns that name), and moosez's own extractor unpacks into its models
    # directory and then looks for ``Dataset501_FVM`` there, so the model is
    # unusable as packaged upstream too. Checked 2026-09-06 against
    # moosez-v.3.2.0's asset.
    "clin_mr_FVM": "mispackaged",
}
REASONS = {
    "mispackaged": "the archive does not unpack to the folder the registry names, so "
                   "haversack's installer would refuse it and moosez's own extractor "
                   "would not find a model where it looks",
}

#: One registry entry: ``"<name>": {KEY_URL: "<url>", KEY_FOLDER_NAME: "<folder>"``.
#: Names carry uppercase (``clin_ct_PUMA4``); ``\w`` is not used because it
#: also admits non-ASCII letters that no task name here should have.
ENTRY = re.compile(
    r'"(?P<name>[A-Za-z0-9_]+)"\s*:\s*\{\s*'
    r'KEY_URL\s*:\s*"(?P<url>[^"]+)"\s*,\s*'
    r'KEY_FOLDER_NAME\s*:\s*"(?P<folder>[^"]+)"')
#: Every place the registry starts a URL, whatever follows. ``KEY_URL`` alone
#: also appears in the import and in attribute lookups; the colon is what makes
#: it a dict key.
URL_KEY = re.compile(r'KEY_URL\s*:\s*(?P<value>"[^"]*"|[^,\s}]*)')


def release_tag(url: str) -> str:
    tag = (re.search(r"_(\d{8})\.zip$", url) or re.search(r"v[\d.]+", url))
    return tag.group(0).strip("_.zip") if tag else "unknown"


def parse(models_py: Path) -> dict:
    """``{name: {url, folder, tag}}`` for every entry in the registry, or a
    clear error naming the ``KEY_URL`` blocks that did not parse as one."""
    src = models_py.read_text()
    out = {}
    for m in ENTRY.finditer(src):
        name, url, folder = m.group("name", "url", "folder")
        out[name] = {"url": url, "folder": folder, "tag": release_tag(url)}
    blocks = [m.group("value") for m in URL_KEY.finditer(src)]
    if len(blocks) != len(out):
        parsed = {e["url"] for e in out.values()}
        stray = [v for v in blocks if v.strip('"') not in parsed] or blocks
        raise SystemExit(
            f"{models_py}: {len(blocks)} KEY_URL blocks but {len(out)} parsed as model "
            f"entries - the registry's layout has changed and this parser would drop the "
            f"rest silently. Not matched: {stray}")
    return out


def unreachable(entries: dict, status=zippeek.head_status) -> dict:
    """``{task: what a HEAD of its URL answered}`` for every entry that did not
    answer 200 - an HTTP status, or the reason when no HTTP answer came."""
    out = {}
    for name, e in entries.items():
        try:
            code = status(e["url"])
        except OSError as err:              # URLError, and a bare socket timeout
            out[name] = str(getattr(err, "reason", err))
            continue
        if code != 200:
            out[name] = code
    return out


def excluded_with_reasons(entries: dict, not_offered: dict, top_level=zippeek.top_level) -> dict:
    """``{name: reason}`` for every entry in ``not_offered``, each reason checked
    against the asset it describes."""
    unknown = sorted(set(not_offered) - set(entries))
    if unknown:
        raise SystemExit(f"{unknown} are listed in NOT_OFFERED but are not in the registry - "
                         "upstream dropped or renamed them; update the list")
    out = {}
    for name, reason in not_offered.items():
        e = entries[name]
        tops = top_level(e["url"])
        if reason == "mispackaged":
            if tops == [e["folder"]]:
                raise SystemExit(
                    f"{name}: {e['url']} unpacks to {e['folder']!r}, exactly the folder the "
                    "registry names - it is not mispackaged; offer it")
            out[name] = (f"{REASONS[reason]} (the archive's top level is {tops}, the "
                         f"registry names {e['folder']!r}; {e['url']})")
        else:
            raise SystemExit(f"{name}: {reason!r} is not a reason this generator checks")
    return out


def generate(models_py: Path, dest: Path = DEST, status=zippeek.head_status,
             top_level=zippeek.top_level, not_offered: dict = NOT_OFFERED) -> dict:
    entries = parse(models_py)
    if not entries:
        raise SystemExit(f"{models_py}: no models found - not the moosez registry?")
    dead = unreachable(entries, status)
    if dead:
        lines = [f"  {name}: {entries[name]['url']} -> {answer}" for name, answer in dead.items()]
        raise SystemExit(f"{len(dead)} of {len(entries)} URLs do not answer a HEAD with 200; "
                         "refusing to write a manifest that names them:\n" + "\n".join(lines))
    excluded = excluded_with_reasons(entries, not_offered, top_level)
    tasks = {name: e for name, e in entries.items() if name not in excluded}
    dest.write_text(json.dumps({"source": "moosez/models.py",
                                "excluded": excluded,
                                "tasks": tasks}, indent=1) + "\n")
    return tasks


if __name__ == "__main__":
    moose = Path(sys.argv[1] if len(sys.argv) > 1 else
                 "../../upstream/MOOSE") / "moosez/models.py"
    tasks = generate(moose)
    print(f"{len(tasks)} models, {len(NOT_OFFERED)} excluded, every URL answering -> {DEST}")
