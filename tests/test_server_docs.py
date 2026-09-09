"""The server guide (SERVER.md, `haversack docs --server`) and the OpenAPI document say the
same things: the guide's route table matches the app's routes, and the app's description is
the guide's rules. A route added without a row here, or a row without a route, fails."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from haversack import cli                                          # noqa: E402

from test_serve import make                                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _guide() -> str:
    return cli.guide_text("server")


def test_docs_server_prints_the_guide_whole_and_by_section(capsys):
    assert cli.main(["docs", "--server"]) == 0
    whole = capsys.readouterr().out
    assert whole.startswith("# haversack server") and "## Reads and computes" in whole
    assert cli.main(["docs", "--server", "jobs"]) == 0
    part = capsys.readouterr().out
    assert part.startswith("## Jobs") and "## Routes" not in part
    assert cli.main(["docs", "--server", "--sections"]) == 0
    heads = capsys.readouterr().out.splitlines()
    assert "Routes" in heads and "Deploying to Modal" in heads
    assert cli.main(["docs"]) == 0                     # the user guide is still the default
    assert capsys.readouterr().out.startswith("# haversack\n")


def _doc_routes() -> set:
    """(METHOD, path) from the guide's route table, in the app's own placeholder spelling."""
    text = _guide()
    table = text.split("## Routes", 1)[1].split("## Reference", 1)[0]
    out = set()
    for m in re.finditer(r"^\| (GET|POST|PUT|DELETE|HEAD) \| `([^`]+)` \|", table, re.M):
        method, path = m.groups()
        out.add((method, path))
    return out


def _app_routes(tmp_path) -> set:
    """(METHOD, path) for every /v1 route, normalized: the mounted source prefixes fold to
    `<source>`, path parameters to the guide's `<...>` names, grid tokens dropped."""
    _, _, client = make(tmp_path)
    prefixes = {s["prefix"] for s in client.get("/v1/sources").json()["sources"]}
    assert prefixes, "the app advertises no sources"
    out = set()
    for r in client.app.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/v1/"):
            continue
        for method in getattr(r, "methods", ()) or ():
            p = path
            parts = p.split("/")
            if len(parts) > 2 and parts[2] in prefixes:
                parts[2] = "<source>"
                p = "/".join(parts)
            p = (p.replace("{ident:path}", "<identifier>").replace("{task}", "<task>")
                  .replace("{jid}", "<id>").replace("{digest}", "<digest>")
                  .replace("_res-1mm", ""))
            out.add((method, p))
    return out


def test_the_route_table_matches_the_app(tmp_path):
    doc, app = _doc_routes(), _app_routes(tmp_path)
    assert doc, "no route table found in the guide"
    assert app - doc == set(), f"routes the guide does not list: {sorted(app - doc)}"
    assert doc - app == set(), f"rows with no route behind them: {sorted(doc - app)}"


def test_openapi_carries_the_guides_rules_and_tags(tmp_path):
    _, _, client = make(tmp_path)
    doc = client.app.openapi()
    desc = doc["info"]["description"]
    assert desc.startswith("## What it is") and "A token computes; anonymous reads." in desc
    assert "Ask twice, compute once." in desc
    names = {t["name"] for t in doc.get("tags", [])}
    assert names == {"service", "tasks", "jobs", "inputs", "results"}
    untagged = [(m, p) for p, ops in doc["paths"].items() if p.startswith("/v1/")
                for m, op in ops.items() if not op.get("tags")]
    assert untagged == [], untagged
    r = client.get("/openapi.json")
    assert r.status_code == 200 and r.json()["info"]["title"] == "haversack"


def test_the_guide_documents_every_engine_enable_flag():
    """The guide hand-lists the engine flags; the registry derives them.

    One fact in two places, and only one of them moves when an engine is added - the same
    shape as the `/v1/version` package list that went two engines stale without anyone
    noticing (fixed 2026-09-08 by deriving it from `Engine.dist`). An operator reads this
    guide to find out what to set at deploy, so a flag missing here is an engine nobody
    can turn on.
    """
    from haversack.engines.registry import engine_env_vars
    text = _guide()
    missing = [v for v in engine_env_vars() if v not in text]
    assert missing == [], (f"the server guide never mentions {missing} - add the flag where "
                           "the other engine flags are listed, or an operator cannot enable it")


def test_the_readme_DESCRIBES_every_engine_where_it_describes_engines():
    """The README is where a user decides whether this runs what they have.

    Scoped to the section that describes the engines, and matched on word boundaries,
    because the first version did neither and was hollow both ways. Deleting the whole
    `## Engines` section - every install command and both worked examples - left it
    green, since an extras list, a cache-path table cell and a deviations example still
    contained the words. And `monai` was already satisfied vacuously by `msd-for-monai`,
    the DATA SOURCE for the Medical Segmentation Decathlon mirror, which has nothing to
    do with the MONAI engine: one of the four engines it covered was not covered at all.
    """
    import pathlib
    import re

    import pytest

    from haversack.engines.registry import ENGINES, NNUNETV2
    readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    if not readme.exists():
        pytest.skip("running against an installed copy, not the repository")
    text = readme.read_text(encoding="utf-8")

    heads = [(m.start(), m.group(0)) for m in re.finditer(r"^## .*$", text, re.M)]
    engine_heads = [i for i, (_, h) in enumerate(heads) if "engine" in h.lower()]
    assert engine_heads, ("README.md has no `## ...Engines...` section - that section is "
                          "where the engines are described and this guard reads it")
    i = engine_heads[0]
    start = heads[i][0]
    end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
    section = text[start:end]
    assert len(section.split()) > 100, (
        f"the engines section is {len(section.split())} words - it no longer describes "
        "anything, whatever words survive elsewhere in the file")

    missing = [n for n in ENGINES if n != NNUNETV2
               and not re.search(rf"(?<![-\w]){re.escape(n)}(?![-\w])", section, re.I)]
    assert missing == [], (
        f"the engines section of README.md never names {missing} - add them where the "
        "other engines are described. A mention elsewhere in the file does not count, "
        "and neither does one inside a longer word: `msd-for-monai` is a data source.")
