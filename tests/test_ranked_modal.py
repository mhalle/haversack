"""Ranked jobs on Modal (2026-09-24): the api side, the routing and the worker hook, as plain
Python with the jobs Dict and the volumes replaced by doubles (tests/test_modal_app.py's way).
What a deployed worker writes is the smoke's business (AGENTS.md); what is held here:

- the kind reaches the job record and the status, and the key carries it and the store's versions;
- a ranked job is its TASK's worker's - the nnU-Net or FastSurfer worker - not a new one;
- the worker writes the store through its own warm runner, with the job's cancellation, and
  names the input by the job's identity;
- the api's scratch fallback reads the store by its own name;
- the images that write stores (and the api, which keys them) carry the duckn extra.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

pytest.importorskip("modal")
pytest.importorskip("duckn")

from test_modal_app import _swap_dict  # noqa: E402

import haversack  # noqa: E402

#: the source of the haversack actually imported - not the tree beside this file, which a
#: harness running a mutated copy on PYTHONPATH would leave these static checks reading
SRC = Path(haversack.__file__).resolve().parent


def test_a_ranked_job_is_its_tasks_workers():
    from haversack import modal_app as m
    assert m._worker_of({"task": "ts.v2:total_fast", "kind": "ranked"}) == "nnunetv2"
    assert m._worker_of({"task": "fastsurfer:asegdkt", "kind": "ranked"}) == "fastsurfer"


def _executor(monkeypatch):
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None, kind="segment":
                        spawned.append((task, jid, kind)) or types.SimpleNamespace(object_id="fc-1"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: None)
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions",
                        lambda task, kind="segment": [f"{kind}-versions"])
    monkeypatch.setattr(ex, "cache_get", lambda key: None)
    return m, fake, ex, spawned


def test_a_ranked_submit_records_keys_and_routes_as_ranked(monkeypatch, tmp_path):
    from haversack.serve import result_key
    m, fake, ex, spawned = _executor(monkeypatch)
    assert ex.ranked_stores is True
    ex.submit("r", tmp_path / "r", None, "ts.v2:total_fast", {}, identity=("idc:1",),
              kind="ranked", deliverables=["preview"])
    ex.submit("s", tmp_path / "s", None, "ts.v2:total_fast", {}, identity=("idc:1",))
    assert fake["r"]["kind"] == "ranked" and fake["r"]["deliverables"] == []
    assert "kind" not in fake["s"]
    assert fake["r"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {},
                                                ["ranked-versions"], kind="ranked")
    assert fake["s"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {},
                                                ["segment-versions"])
    assert spawned == [("ts.v2:total_fast", "r", "ranked"), ("ts.v2:total_fast", "s", "segment")]
    assert ex.status_of("r")["kind"] == "ranked" and "kind" not in ex.status_of("s")


def test_the_scratch_fallback_reads_the_store_by_its_name(monkeypatch, tmp_path):
    from haversack.serve import RANKED_NAME
    m, fake, ex, _ = _executor(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setattr(m, "MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setattr(m, "_reload_logged", lambda vol, name: True)
    (tmp_path / "scratch" / "r").mkdir(parents=True)
    (tmp_path / "scratch" / "r" / RANKED_NAME).write_bytes(b"PK")
    fake["r"] = {"id": "r", "task": "ts.v2:total_fast", "kind": "ranked", "state": "done"}
    state, path = ex.result_file("r")
    assert state == "done" and path.name == RANKED_NAME and path.read_bytes() == b"PK"


def test_the_nnunet_worker_writes_the_store_through_its_own_segmenter(monkeypatch, tmp_path):
    from haversack import modal_app as m
    from haversack import ranked_output
    seen = {}

    class Seg:
        def segment(self, image, task, **kw):
            seen["segment"] = (image, task, kw)
            return "the-seg"

    def fake_store(image, task, out, *, case=None, source=None, quiet=False, run=None,
                   progress=None, **kw):
        seen["store"] = {"task": task, "out": out, "case": case, "source": source}
        return run(image, task, probabilities="the-spec", progress=progress), out
    monkeypatch.setattr(ranked_output, "segment_to_store", fake_store)
    W = m.Worker._get_user_cls()
    w = W.__new__(W)
    w.seg = Seg()
    token = object()
    got = w._ranked("in.nii.gz", {"id": "r", "task": "ts.v2:total_fast", "version": None,
                                  "input_identity": ["idc:1"]}, tmp_path / "ranked.duckn.zip",
                    "reporter", token)
    assert got == "the-seg"
    assert seen["store"]["source"] == {"type": "image", "identifier": "idc:1"}
    assert seen["store"]["case"] == "r" and seen["store"]["task"] == "ts.v2:total_fast"
    image, task, kw = seen["segment"]
    assert kw["cancel"] is token and kw["probabilities"] == "the-spec" and kw["progress"] == "reporter"


def _extras_of_uv_syncs(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "uv_sync":
            for k in node.keywords:
                if k.arg == "extras":
                    out.append([e.value for e in k.value.elts])
    return out


def test_every_image_that_writes_or_keys_a_store_carries_the_duckn_extra():
    """The api keys a store on the formats it is written in (ranked_output.ranked_tag imports
    rankfield and duckn) and the nnU-Net worker writes it - one base image; FastSurfer's worker
    writes its own. A missing extra is a job that fails on the worker, or an api that cannot key."""
    base = _extras_of_uv_syncs(SRC / "modal_app.py")
    assert ["torch", "serve", "cuda", "duckn"] in base, base
    fs = _extras_of_uv_syncs(SRC / "engines" / "modal_fastsurfer.py")
    assert fs and all("duckn" in e for e in fs), fs


def test_the_worker_runs_the_ranked_branch_and_publishes_its_record():
    """``_execute_job`` runs only in a worker container, so the branch is held statically:
    it calls the worker's ``_ranked`` and builds the record with ``ranked_payload`` - parsed
    CALLS, not text, which a comment would satisfy."""
    tree = ast.parse((SRC / "modal_app.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_job")
    calls = {(getattr(c.func, "attr", None) or getattr(c.func, "id", None))
             for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "_ranked" in calls and "ranked_payload" in calls
