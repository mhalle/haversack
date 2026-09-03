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
