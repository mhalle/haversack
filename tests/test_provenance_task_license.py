"""A license a catalog states for ONE task reaches the result, not only `describe()`.

`segment` wrote its result's attribution from the ecosystem's name and the modality alone, so
the license a manifest states per task never reached the seg.nrrd header: it fell back to the
ecosystem's. Every shipped manifest agreed with its ecosystem, so nothing was misstated - but
a catalog whose tasks differ in license would have been, in the one copy of the terms that
travels with a download (found 2026-09-20, reading the code for a license notice that was
deferred). The engine path always passed the catalog's record. These drive the real
`segment()` with stub models and a catalog double whose task is licensed unlike its catalog.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

pytest.importorskip("nnunetv2")
sitk = pytest.importorskip("SimpleITK")
from test_normalization_sharing import (ORGANS, _StubModel, _two_part_task,  # noqa: E402
                                        _write_ct)

from haversack import attribution, pipeline  # noqa: E402

#: A task in a real catalog, so the ecosystem's own license is a shipped fact (MOOSE's weights
#: are CC BY 4.0) that the double's per-task license has to win over.
TASK = "moose:licensed_unlike_its_catalog"
TASK_LICENSE = "CC-BY-NC-SA-4.0"


class _Catalog:
    """Just enough of EcosystemCatalog: an installed task, and the record `info()` gives."""

    def __init__(self, spec, info):
        self.spec, self._info, self.asked = spec, info, []

    def installed(self, name) -> bool:
        return True

    def get(self, name, progress=None):
        return self.spec

    def info(self, name):
        self.asked.append(name)
        if isinstance(self._info, Exception):
            raise self._info
        return dict(self._info)


class _NoInfo:
    """A plain TaskCatalog's shape: no `info` at all."""

    def __init__(self, spec):
        self.spec = spec

    def get(self, name):
        return self.spec


def _segment(tmp_path, monkeypatch, catalog_of, task=TASK):
    spec, store, cache = _two_part_task(tmp_path, [_StubModel(ORGANS._props) for _ in range(2)])
    spec = dataclasses.replace(spec, name=task)
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    cat = catalog_of(spec)
    r = pipeline.segment(str(_write_ct(tmp_path, (12, 14, 16))), task, catalog=cat,
                         models=cache, device="cpu", convention="corner", folds=(0,))
    return r, cat


def test_the_ecosystem_license_is_what_this_double_must_override():
    """The premise, from shipped data: without it the test below could pass by coincidence."""
    assert attribution.for_ecosystem("moose")["license"]["weights"] == "CC-BY-4.0"
    assert attribution.for_ecosystem("moose")["license"]["weights"] != TASK_LICENSE


def test_a_per_task_license_reaches_the_result_and_the_file(tmp_path, monkeypatch):
    r, cat = _segment(tmp_path, monkeypatch, lambda spec: _Catalog(
        spec, {"name": TASK, "ecosystem": "moose", "engine": "nnunetv2",
               "license": TASK_LICENSE}))
    assert cat.asked == [TASK], "the catalog was not asked for the task's own record"
    block = r.provenance["attribution"]
    assert block["license"] == {"weights": TASK_LICENSE}, block
    # ...and it is the ecosystem's papers still: only the license is the task's own
    assert "35772962" in [c.get("pmid") for c in block["cite"]], block["cite"]
    # the copy that travels: the seg.nrrd header
    out = r.save(tmp_path / "labels.seg.nrrd")
    header = json.loads(sitk.ReadImage(str(out)).GetMetaData("haversack_provenance"))
    assert header["attribution"]["license"] == {"weights": TASK_LICENSE}


def test_the_modality_is_the_specs_not_the_records(tmp_path, monkeypatch):
    """The modality decides which papers apply (TotalSegmentator's MRI paper), and the spec's
    is the one `describe()` uses; a record that misstates it must not change the list."""
    task = "ts.v2:a_ct_stub"
    r, _ = _segment(tmp_path, monkeypatch, lambda spec: _Catalog(
        spec, {"name": task, "ecosystem": "ts.v2", "modality": "MR"}), task=task)
    dois = [c.get("doi") for c in r.provenance["attribution"]["cite"]]
    assert "10.1148/ryai.230024" in dois, dois            # the CT paper
    assert "10.1148/radiol.241613" not in dois, "the record's MR won over the spec's CT"


@pytest.mark.parametrize("catalog_of", [
    _NoInfo,
    lambda spec: _Catalog(spec, LookupError("unknown task")),
    lambda spec: _Catalog(spec, {"name": TASK, "ecosystem": "moose", "engine": "nnunetv2"}),
], ids=["no info()", "info() raises", "no per-task license"])
def test_a_task_with_no_license_of_its_own_is_credited_exactly_as_before(
        tmp_path, monkeypatch, catalog_of):
    """Exactly what the grammar alone gave: such a task's header must not move, since a
    result's ETag is the digest of that file."""
    r, _ = _segment(tmp_path, monkeypatch, catalog_of)
    before = attribution.provenance_block(TASK, {"ecosystem": "moose", "modality": "CT"})
    assert r.provenance["attribution"] == before
    assert before["license"] == {"code": "Apache-2.0", "weights": "CC-BY-4.0"}


def test_no_shipped_tasks_header_moved_when_the_manifests_record_was_handed_over():
    """Every shipped manifest that states a per-task license repeats its ecosystem's, and the
    ecosystem's record is the fuller one (it names the code's license). Handing `segment` the
    manifest's record must therefore change no shipped task's block: before this rule, 17 did
    (cads and totalvibe lost `code`, dentalsegmentator's id changed case) for the same facts.
    Walks the real catalogs, so a future manifest that DOES differ shows up here by name -
    that is a fact to check against upstream, not a failure to silence."""
    import tempfile

    from haversack.ecosystems import EcosystemCatalog
    with tempfile.TemporaryDirectory() as td:
        cat = EcosystemCatalog(root=td)
        moved, stated = {}, 0
        for name in cat.names():
            info = cat.info(name)
            if info.get("engine") != "nnunetv2":
                continue
            stated += bool(info.get("license"))
            eco, mod = name.partition(":")[0], info.get("modality") or "CT"
            grammar = attribution.provenance_block(name, {"ecosystem": eco, "modality": mod})
            handed = attribution.provenance_block(name, {**info, "modality": mod})
            if grammar != handed:
                moved[name] = (grammar["license"], handed["license"])
    assert stated >= 17, f"only {stated} tasks state a license: this walk is looking at less"
    assert moved == {}, moved


def test_a_repeated_license_keeps_the_fuller_record_and_a_different_one_wins():
    same = attribution.provenance_block("cads:organs", {"ecosystem": "cads", "license": "cc-by-sa-4.0"})
    assert same["license"] == {"code": "Apache-2.0", "weights": "CC-BY-SA-4.0"}
    other = attribution.provenance_block("cads:organs", {"ecosystem": "cads", "license": TASK_LICENSE})
    assert other["license"] == {"weights": TASK_LICENSE}
    # a record that says MORE than the id (a note, a url) is never dropped
    noted = attribution.provenance_block(
        "cads:organs", {"ecosystem": "cads", "license": {"weights": "CC-BY-SA-4.0", "note": "x"}})
    assert noted["license"] == {"weights": "CC-BY-SA-4.0", "note": "x"}
