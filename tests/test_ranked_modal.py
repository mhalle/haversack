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
    assert m._worker_of({"task": "ts.v2:total_fast", "kind": "rankfield"}) == "nnunetv2"
    assert m._worker_of({"task": "fastsurfer:asegdkt", "kind": "rankfield"}) == "fastsurfer"


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
    monkeypatch.setattr(ex, "_cache_record", lambda key, wanted=(): None)
    return m, fake, ex, spawned


def test_a_ranked_submit_records_keys_and_routes_as_ranked(monkeypatch, tmp_path):
    from haversack.serve import result_key
    m, fake, ex, spawned = _executor(monkeypatch)
    assert ex.ranked_stores is True
    ex.submit("r", tmp_path / "r", None, "ts.v2:total_fast", {}, identity=("idc:1",),
              kind="rankfield", deliverables=["preview"])
    ex.submit("s", tmp_path / "s", None, "ts.v2:total_fast", {}, identity=("idc:1",))
    assert fake["r"]["kind"] == "rankfield" and fake["r"]["deliverables"] == []
    assert "kind" not in fake["s"]
    assert fake["r"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {},
                                                ["rankfield-versions"], kind="rankfield")
    assert fake["s"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {},
                                                ["segment-versions"])
    assert spawned == [("ts.v2:total_fast", "r", "rankfield"), ("ts.v2:total_fast", "s", "segment")]
    assert ex.status_of("r")["kind"] == "rankfield" and "kind" not in ex.status_of("s")


def test_the_scratch_fallback_reads_the_store_by_its_name(monkeypatch, tmp_path):
    from haversack.serve import RANKED_NAME
    m, fake, ex, _ = _executor(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setattr(m, "MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setattr(m, "_reload_logged", lambda vol, name: True)
    (tmp_path / "scratch" / "r").mkdir(parents=True)
    (tmp_path / "scratch" / "r" / RANKED_NAME).write_bytes(b"PK")
    fake["r"] = {"id": "r", "task": "ts.v2:total_fast", "kind": "rankfield", "state": "done"}
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
                                  "input_identity": ["idc:1"]}, tmp_path / "rankfield.duckn.zip",
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
    images = _extras_of_uv_syncs(SRC / "modal_app.py")
    # every image that answers a request (the api, its twin) or runs a worker carries `serve`,
    # and every one of them keys a store or writes one: duckn directly, or through `embed`.
    # The first version checked the worker's base image only, and the API's own image - a
    # second uv_sync - shipped without it: a 500 on every ranked submit, found by deploying.
    serving = [e for e in images if "serve" in e]
    assert len(serving) >= 2, images
    # the `duckn` extra itself, not `embed` standing in for it (review, 2026-09-25): `embed`
    # has no pydicom, so the embedding worker made no input copies
    assert all("duckn" in e for e in serving), serving
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


# -- the worker branch, driven (review 2026-09-25: a static "the call is somewhere" check was all
# -- that held it, and a ranked job running _compute, publishing under the labels' key or
# -- overwriting its store with labels all survived the suite) ----------------------------------

from test_worker_volume_view import _Ctx, _submit, worker  # noqa: E402,F401 - the fixture


class _RankedCtx(_Ctx):
    """A worker whose ranked hook writes a small zip; its compute and a label save must not run."""

    def __init__(self):
        super().__init__()
        self.ranked_calls = []
        from haversack.engines import registry
        # enough of a Segmenter for the key: an installed version and the engine's row
        self.seg = types.SimpleNamespace(
            describe=lambda t: {"weights_installed": [{"id": 297, "version": "v2.0.0"}]},
            engine_for=registry.engine_for_task)

    def _compute(self, *a, **k):
        raise AssertionError("a ranked job ran the segmentation compute")

    def _ranked(self, input_path, meta, out, on_progress, token):
        import zipfile
        self.ranked_calls.append(meta["id"])
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("zarr.json", "{}")

        class Seg:
            schema = types.SimpleNamespace(names={1: "liver"})
            provenance = {}

            def volumes_ml(self):
                return {"liver": 1.0}

            def save(self, path):
                raise AssertionError("a ranked job saved labels over its store")
        return Seg()


def test_a_ranked_job_on_the_worker_publishes_its_store_under_the_stores_key(worker):
    from haversack.serve import RANKED_NAME, ResultCache, result_key, versions_for
    m, jobs, scratch, cache = worker
    _submit(m, jobs, "rk")
    jobs["rk"]["kind"] = "rankfield"
    ctx = _RankedCtx()
    m._execute_job(ctx, "rk")
    rec = jobs["rk"]
    assert rec["state"] == "done", rec.get("error")
    assert ctx.ranked_calls == ["rk"]
    assert rec["result"]["outputs"][0]["kind"] == "rankfield"
    assert "inputs" in rec["result"]["provenance"]            # record_inputs ran, as locally
    want = result_key(("sha256:rk",), "ts.v2:total_fast", {},
                      versions_for(ctx.seg, "ts.v2:total_fast", "rankfield"), kind="rankfield")
    assert rec["cache_key"] == want                           # re-keyed as a STORE, not labels
    hit = ResultCache(m.CACHE_ROOT).get(want)
    assert hit is not None and Path(hit[0]).name == RANKED_NAME
    import json as _json
    meta = _json.loads((Path(hit[0]).parent / "meta.json").read_text())
    assert meta["kind"] == "rankfield"


def test_a_ranked_cache_hit_on_modal_says_its_kind(monkeypatch, tmp_path):
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(m, "_spawn_worker", lambda *a, **k: types.SimpleNamespace(object_id="x"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: None)
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task, kind="segment": [kind])
    stored = tmp_path / "k" / "g-1" / "rankfield.duckn.zip"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"PK")
    monkeypatch.setattr(ex, "_cache_record",
                        lambda key, wanted=(): (stored, {"outputs": [{"kind": "rankfield"}]}, ()))
    meta = ex.submit("h", tmp_path / "h", None, "ts.v2:total_fast", {}, identity=("idc:1",),
                     kind="rankfield")
    assert meta["cached"] and meta["kind"] == "rankfield" and meta["deliverables"] == []
    assert ex.status_of("h")["kind"] == "rankfield"


def test_versions_are_remembered_per_kind(monkeypatch):
    """One memo entry per (kind, task): a segmentation's versions answered for a store's key
    (or the reverse) would key the api's path and the worker's re-key apart for up to 30 s."""
    from haversack import modal_app as m
    from haversack import serve
    monkeypatch.setattr(m.ModalExecutor, "_wv_cache", {})
    monkeypatch.setattr(serve, "versions_for", lambda seg, task, kind="segment": [kind])
    ex = m.ModalExecutor()
    assert ex._fresh_weights_versions("ts.v2:total_fast") == ["segment"]
    assert ex._fresh_weights_versions("ts.v2:total_fast", "rankfield") == ["rankfield"]
    assert ex._fresh_weights_versions("ts.v2:total_fast", "embed") == ["embed"]
    assert ex._fresh_weights_versions("ts.v2:total_fast") == ["segment"]


def test_the_worker_hands_the_pin_to_the_store(monkeypatch, tmp_path):
    from haversack import modal_app as m
    from haversack import ranked_output
    seen = {}
    monkeypatch.setattr(ranked_output, "segment_to_store",
                        lambda image, task, out, **kw: seen.update(task=task, name=kw.get("image_name"))
                        or ("seg", out))
    W = m.Worker._get_user_cls()
    w = W.__new__(W)
    w.seg = types.SimpleNamespace(segment=lambda *a, **k: "seg")
    w._ranked("in.nii.gz", {"id": "r", "task": "ts.v2:total_fast", "version": "v2.0.0",
                            "input_identity": ["idc:1"]}, tmp_path / "s.duckn.zip", None, None)
    assert seen == {"task": "ts.v2:total_fast@v2.0.0", "name": "idc:1"}


def test_the_fastsurfer_worker_hands_the_ranked_sink_to_its_runner():
    """Driven in a subprocess: importing an adapter registers a Modal class, which this process
    must not carry. Without `probabilities=` every FastSurfer store job on Modal would fail with
    'produced no ranked output' - and no test touched this hook (review, 2026-09-25)."""
    import json
    import os
    import subprocess
    import sys
    code = "\n".join([
        "import json",
        "from haversack import modal_app",
        "from haversack.engines import modal_fastsurfer as mf, fastsurfer",
        "seen = []",
        "fastsurfer.segment = lambda image, **k: seen.append(sorted(k)) or 'S'",
        "U = mf.FastSurferWorker._get_user_cls()",
        "w = U.__new__(U)",
        "r = U._ranked_run(w, 'tok')('img', 'fastsurfer:asegdkt', probabilities='P', progress='R')",
        "print(json.dumps([r, seen, fastsurfer.segment.__name__]))"])
    code = code.replace("seen.append(sorted(k))", "seen.append({x: str(v) for x, v in sorted(k.items())})")
    env = {**os.environ, "HAVERSACK_FASTSURFER": "1",
           "PYTHONPATH": os.pathsep.join(p for p in (str(SRC.parent), os.environ.get("PYTHONPATH")) if p)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    out, seen, _ = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == "S" and seen == [{"device": "cuda", "probabilities": "P"}]



def test_the_twin_keys_each_kind_through_the_one_door(monkeypatch):
    """The Modal twin's versions function (``public``'s weights_fn): a store keys on the task's
    versions PLUS its format tag, as submit and the worker's re-key do - without the tag the twin
    404'd every store it held."""
    from haversack import modal_app as m
    from haversack import serve
    monkeypatch.setattr(serve, "weights_versions_of", lambda seg, task: ["297=v2"])
    got = {k: m._twin_weights_versions("seg", "ts.v2:total_fast", k) for k in ("segment", "rankfield")}
    assert got["segment"] == ["297=v2"]
    assert got["rankfield"] == serve.versions_for("seg", "ts.v2:total_fast", "rankfield")
    assert got["rankfield"][:-1] == ["297=v2"] and got["rankfield"][-1].startswith("rankfield=")
    # and the twin's endpoint uses it: parsed calls, not text
    tree = ast.parse((SRC / "modal_app.py").read_text(encoding="utf-8"))
    public = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "public")
    assert "_twin_weights_versions" in {getattr(c.func, "id", None) for c in ast.walk(public)
                                        if isinstance(c, ast.Call)}
