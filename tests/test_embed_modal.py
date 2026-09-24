"""Embedding jobs on Modal (2026-09-23): the api side and the routing, as plain Python with the
jobs Dict and the volumes replaced by doubles, the way tests/test_modal_app.py drives them.
What a deployed EmbedWorker does is the smoke's business (AGENTS.md); what is held here is
that the api keys, records and routes an embedding job as the local server does:

- the kind reaches the job record and the status, and the key carries it (``ts.v2:total_fast``
  is a task AND an encoder: without it a field and labels would share one key);
- the spawn and the prefetcher agree on whose job it is: an embedding job of a task-named encoder
  is the ENCODER worker's, never the nnU-Net worker's;
- a deployment without the encoder worker refuses the kind, at the executor too;
- the api's scratch fallback reads the job's field by its own name.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("modal")

from test_modal_app import _swap_dict  # noqa: E402


def test_whose_job_it_is_is_one_answer():
    from haversack import modal_app as m
    assert m._worker_of({"task": "ts.v2:total_fast", "kind": "embed"}) == m.EMBED_WORKER
    assert m._worker_of({"task": "ts.v2:total_fast"}) == "nnunetv2"
    assert m._worker_of({"task": "radar:pretrain", "kind": "embed"}) == m.EMBED_WORKER


def test_the_nnunet_worker_does_not_warm_an_embedding_job(monkeypatch):
    m, fake = _swap_dict(monkeypatch)
    fake["e"] = {"id": "e", "state": "queued", "created": 1, "task": "ts.v2:total_fast",
                 "kind": "embed", "source": [{"kind": "s3", "id": "b/e"}]}
    fake["s"] = {"id": "s", "state": "queued", "created": 2, "task": "ts.v2:total_fast",
                 "source": [{"kind": "s3", "id": "b/s"}]}
    assert m._prefetch_candidate("x", engine="nnunetv2")[-1] == "s"
    assert m._prefetch_candidate("x", engine=m.EMBED_WORKER)[-1] == "e"


def _executor(monkeypatch, *, embeds=True):
    m, fake = _swap_dict(monkeypatch)
    monkeypatch.setattr(m, "scratch_vol", types.SimpleNamespace(commit=lambda: None))
    spawned = []
    monkeypatch.setattr(m, "_spawn_worker", lambda task, jid, tokens=None, kind="segment":
                        spawned.append((task, jid, kind)) or types.SimpleNamespace(object_id="fc-1"))
    monkeypatch.setattr(m, "_emit", lambda jid, d: None)
    ex = m.ModalExecutor()
    monkeypatch.setattr(ex, "embeds", embeds)
    monkeypatch.setattr(ex, "_fresh_weights_versions",
                        lambda task, kind="segment": [f"{kind}-versions"])
    monkeypatch.setattr(ex, "cache_get", lambda key: None)
    return m, fake, ex, spawned


def test_an_embedding_submit_records_keys_and_routes_as_embed(monkeypatch, tmp_path):
    from haversack.serve import result_key
    m, fake, ex, spawned = _executor(monkeypatch)
    ex.submit("j1", tmp_path / "j1", None, "ts.v2:total_fast", {"int8": True},
              identity=("idc:1",), kind="embed", deliverables=["preview"])
    ex.submit("j2", tmp_path / "j2", None, "ts.v2:total_fast", {}, identity=("idc:1",))
    assert fake["j1"]["kind"] == "embed" and fake["j1"]["deliverables"] == []
    assert "kind" not in fake["j2"]
    assert fake["j1"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {"int8": True},
                                                 ["embed-versions"], kind="embed")
    assert fake["j2"]["cache_key"] == result_key(("idc:1",), "ts.v2:total_fast", {},
                                                 ["segment-versions"])
    assert spawned == [("ts.v2:total_fast", "j1", "embed"), ("ts.v2:total_fast", "j2", "segment")]
    assert ex.status_of("j1")["kind"] == "embed" and "kind" not in ex.status_of("j2")


def test_a_deployment_without_the_worker_refuses_the_kind(monkeypatch, tmp_path):
    m, fake, ex, spawned = _executor(monkeypatch, embeds=False)
    with pytest.raises(ValueError):
        ex.submit("j1", tmp_path / "j1", None, "radar:pretrain", {}, identity=("idc:1",), kind="embed")
    assert "j1" not in fake and spawned == []


def test_the_scratch_fallback_reads_the_field_by_its_name(monkeypatch, tmp_path):
    from haversack.serve import EMBEDDING_NAME
    m, fake, ex, _ = _executor(monkeypatch)
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setattr(m, "MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setattr(m, "_reload_logged", lambda vol, name: True)
    (tmp_path / "scratch" / "j1").mkdir(parents=True)
    (tmp_path / "scratch" / "j1" / EMBEDDING_NAME).write_bytes(b"PK")
    fake["j1"] = {"id": "j1", "task": "radar:pretrain", "kind": "embed", "state": "done"}
    state, path = ex.result_file("j1")
    assert state == "done" and path.name == EMBEDDING_NAME and path.read_bytes() == b"PK"


def test_the_spawn_refuses_an_embedding_without_the_worker(monkeypatch):
    from haversack import modal_app as m
    monkeypatch.setattr(m, "EMBED", False)
    with pytest.raises(RuntimeError, match="HAVERSACK_EMBED"):
        m._spawn_worker("radar:pretrain", "j", kind="embed")
