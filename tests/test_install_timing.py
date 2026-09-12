"""A weights install on first use is its own step in `segment`'s timings and progress.

`segment` resolved the task inside its `read+canonical` timer, and resolving a catalog task
whose weights are not in place downloads them - so on a fresh Modal container 29 s of a
`cads:headneck` run's "read+canonical" was its 760 MB weights, with no progress while they
came, where the read itself takes about half a second (2026-09-12). These drive the real
`segment()` with stub models and a catalog whose install sleeps.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("nnunetv2")
pytest.importorskip("SimpleITK")
from test_normalization_sharing import (ORGANS, _StubModel, _two_part_task,  # noqa: E402
                                        _write_ct)

from haversack import pipeline  # noqa: E402

INSTALL_S = 0.4


class _Catalog:
    """Just enough of EcosystemCatalog: a task that is installed or not, and an install that
    takes INSTALL_S and records what it was given to report to."""

    def __init__(self, spec, installed: bool):
        self.spec, self._installed, self.progress = spec, installed, []

    def installed(self, name) -> bool:
        return self._installed

    def get(self, name, progress=None):
        if not self._installed:
            self.progress.append(progress)
            time.sleep(INSTALL_S)
            self._installed = True
        return self.spec


def _run(tmp_path, monkeypatch, *, installed: bool):
    spec, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props) for _ in range(2)])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    cat, stages = _Catalog(spec, installed), []
    r = pipeline.segment(str(_write_ct(tmp_path, (12, 14, 16))), "fake:task", catalog=cat,
                         models=cache, device="cpu", convention="corner", folds=(0,),
                         progress=lambda p: stages.append(p.stage))
    return r, cat, stages


def test_a_first_use_install_is_timed_and_reported_as_its_own_step(tmp_path, monkeypatch):
    r, cat, stages = _run(tmp_path, monkeypatch, installed=False)
    weights = {k: v for k, v in r.timings.items() if k.startswith("weights:")}
    assert list(weights) == [f"weights:{cat.spec.name}"], r.timings
    assert next(iter(weights.values())) >= INSTALL_S
    assert r.timings["read+canonical"] < INSTALL_S, "the install is still inside the read's timer"
    assert r.timings["total"] >= INSTALL_S, "the total left the install out"
    assert "weights" in stages and stages.index("weights") < stages.index("read"), stages
    assert cat.progress and cat.progress[0] is not None, "the install had nothing to report to"


def test_an_installed_task_has_no_install_step(tmp_path, monkeypatch):
    r, cat, stages = _run(tmp_path, monkeypatch, installed=True)
    assert not [k for k in r.timings if k.startswith("weights:")], r.timings
    assert "weights" not in stages, stages


# -- the install reports inside the run's progress, not over it ------------------------------
#
# Handing the installer the job's own Reporter let it rewrite the run's part, part count and
# fraction: a job showed 100 % before any inference and then went backwards, and a cascade
# whose coarse task installed spent its whole fine stage at 100 % (review, 2026-09-12). These
# go through the REAL installer (weights_fetch.ensure_task_weights); only the network is faked,
# reporting the way fetch_one does.

def _fetch_one(wid, root, *, tag=None, progress=None):
    for done in (0, 50, 100):
        progress.download(done, 100, f"downloading Dataset{wid}")
    progress.unpack(f"unpacking Dataset{wid}")
    progress.finished(f"Dataset{wid} installed")
    return root


class _Installing:
    """A catalog whose listed tasks install through the real ensure_task_weights on first use."""

    def __init__(self, specs: dict, root, installed=()):
        self.specs, self.root, self.have = specs, root, set(installed)

    def installed(self, name) -> bool:
        return name in self.have

    def get(self, name, progress=None):
        from haversack import weights_fetch
        if name not in self.have:
            weights_fetch.ensure_task_weights(self.specs[name], self.root, catalog=self,
                                              progress=progress)
            self.have.add(name)
        return self.specs[name]


def test_a_first_use_install_never_moves_the_run_backwards(tmp_path, monkeypatch):
    from haversack import weights_fetch
    spec, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props) for _ in range(2)])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    monkeypatch.setattr(weights_fetch, "fetch_one", _fetch_one)
    seen = []
    pipeline.segment(str(_write_ct(tmp_path, (12, 14, 16))), "fake:task",
                     catalog=_Installing({"fake:task": spec}, tmp_path), models=cache,
                     device="cpu", convention="corner", folds=(0,), progress=seen.append)
    drops = [(a, b) for a, b in zip(seen, seen[1:]) if b.fraction < a.fraction - 1e-9]
    assert not drops, [(str(a), str(b)) for a, b in drops]
    assert any(p.stage == "weights" and p.n_steps == 100 for p in seen), "no download progress"
    # the run's own stages keep its two parts (before "loading" it has not counted them yet)
    run = [p for p in seen if p.stage in ("loading", "preprocess", "predict", "restore", "finalize")]
    assert run and all(p.n_parts == 2 for p in run), [str(p) for p in run]


def test_a_cascades_bare_coarse_task_is_looked_up_in_its_own_catalog(tmp_path, monkeypatch):
    """A registry names its own tasks bare (TotalSegmentator's `teeth` crops from
    `craniofacial_structures`); since bare names are refused, the pipeline qualifies the
    reference with the cascade's own catalog - `fine:coarse` for `fine:task`."""
    from haversack.tasks import CascadeStep, TaskSpec, UnionPart
    _, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props) for _ in range(2)])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    coarse = TaskSpec(name="fine:coarse", shape="union", label_map={1: "a"},
                      union=(UnionPart(weights_id=1, label_remap={1: 1}, name="c"),))
    fine = TaskSpec(name="fine:task", shape="cascade", label_map={1: "a"},
                    cascade=(CascadeStep(crop_from_task="coarse", crop_to_classes=(1,)),
                             CascadeStep(weights_id=2)))

    class _Qualified:
        asked = []

        def installed(self, name):
            return True

        def get(self, name, progress=None):
            self.asked.append(name)
            if ":" not in name:
                raise LookupError(f"task {name!r} needs its catalog")
            return coarse

    cat = _Qualified()
    pipeline.segment(str(_write_ct(tmp_path, (12, 14, 16))), fine, catalog=cat, models=cache,
                     device="cpu", convention="corner", folds=(0,))
    assert cat.asked == ["fine:coarse"], cat.asked


def test_an_install_inside_a_cascade_keeps_the_cascades_parts(tmp_path, monkeypatch):
    from haversack import weights_fetch
    from haversack.tasks import CascadeStep, TaskSpec, UnionPart
    _, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props) for _ in range(2)])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    monkeypatch.setattr(weights_fetch, "fetch_one", _fetch_one)
    coarse = TaskSpec(name="coarse:task", shape="union", label_map={1: "a"},
                      union=(UnionPart(weights_id=1, label_remap={1: 1}, name="c"),))
    fine = TaskSpec(name="fine:task", shape="cascade", label_map={1: "a"},
                    cascade=(CascadeStep(crop_from_task="coarse:task", crop_to_classes=(1,)),
                             CascadeStep(weights_id=2)))
    seen = []
    r = pipeline.segment(str(_write_ct(tmp_path, (12, 14, 16))), fine,
                         catalog=_Installing({"coarse:task": coarse}, tmp_path), models=cache,
                         device="cpu", convention="corner", folds=(0,), progress=seen.append)
    assert "weights:coarse:task" in r.timings, r.timings
    fine_stage = [p for p in seen if p.part == 1 and p.stage in ("loading", "predict", "restore")]
    assert fine_stage and all(p.n_parts == 2 and p.fraction < 1.0 for p in fine_stage), \
        [str(p) for p in fine_stage]
