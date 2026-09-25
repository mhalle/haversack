"""A cascade's ranked store names each layer from its own model (2026-09-23).

A cascade's crop stage outputs its own model's classes: stage 0 of ``lung_vessels`` is
Dataset297, ``total_fast``'s 118 classes. The builder named every layer from the task's label
map, so layer 0's values 1-4 - spleen, kidney_right, kidney_left, gallbladder - were called
lung_airways, lung_airways_wall, lung_arteries and lung_veins, coded in ``ts.v2:lung_vessels``,
and 5-117 were ``label_<v>``. It meant to number the crop stages instead, but recognized them
by a part name (``<task>:s<i>`` for every stage) the pipeline had stopped writing: the final
stage is named ``<task>``.

The emit now records which task names each part's values (``labels_named_by``), the store's
part block keeps it, and a store whose stages follow two class lists declares both schemes.
Driven through ``segment_to_store`` with stub models - no weights, no network.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("SimpleITK")
pytest.importorskip("zarr")
pytest.importorskip("duckn")
pytest.importorskip("nnunetv2")

from haversack.ecosystems import EcosystemCatalog, known_ecosystems   # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
REGISTRY = Path(__file__).resolve().parent.parent / "src/haversack/data/ts_tasks.json"
CASCADES = [t for t in json.loads(REGISTRY.read_text(encoding="utf-8"))["tasks"]
            if t.get("shape") == "cascade"]


# ----------------------------------------------------------------------------------------
# the catalog: which task names a crop stage's classes
# ----------------------------------------------------------------------------------------

#: the crop models every shipped cascade uses, and the task that runs each alone
SOLO = {297: "ts.v2:total_fast", 298: "ts.v2:total_fastest", 300: "ts.v2:body_fast",
        852: "ts.v2:total_mr_fast"}


@pytest.mark.parametrize("entry", CASCADES, ids=[t["name"] for t in CASCADES])
def test_every_crop_stage_is_named_by_the_task_that_runs_its_model_alone(entry):
    cat = EcosystemCatalog(known_ecosystems())
    task = f"ts.v2:{entry['name']}"
    for i, st in enumerate(entry["cascade"][:-1]):
        got = cat.stage_task(task, i)
        if st.get("weights_id") is None:
            # teeth crops from craniofacial_structures' result: that run names its own parts
            assert st.get("crop_from_task") and got is None
            continue
        assert got == SOLO[st["weights_id"]], (task, i)
        solo = cat.get(got)
        assert solo.shape == "single" and int(solo.single) == int(st["weights_id"])
        # the classes it crops to are ones the stage task names
        assert set(st["crop_to_classes"]) <= set(solo.label_map), (task, st["crop_to_classes"])
    assert cat.stage_task(task, len(entry["cascade"]) - 1) is None     # the last is the task


def test_a_task_that_is_not_a_cascade_has_no_stages():
    cat = EcosystemCatalog(known_ecosystems())
    assert cat.stage_task("ts.v2:total", 0) is None
    assert cat.stage_task("ts.v2:total_fast", 0) is None


def _with_twin(tmp_path, label_map):
    """The shipped registry plus ``total_fast_twin``, a second task running Dataset297 alone."""
    from haversack.tasks import TaskCatalog
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    twin = dict(next(t for t in raw["tasks"] if t["name"] == "total_fast"),
                name="total_fast_twin", label_map=label_map)
    raw["tasks"].append(twin)
    path = tmp_path / "ts_tasks.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return TaskCatalog("ts", path=path)


def test_two_tasks_that_run_the_stage_model_alone_name_it_only_if_they_agree(tmp_path):
    """No single task names the stage when two run its model alone under different class
    lists: the stage is left unnamed, never named by whichever sorts first (review, 2026-09-23:
    no test reached this branch, and removing it survived)."""
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    same = next(t for t in raw["tasks"] if t["name"] == "total_fast")["label_map"]
    assert _with_twin(tmp_path, same).stage_task("lung_vessels", 0) == "total_fast"
    other = dict(same, **{"1": "not_the_spleen"})
    assert _with_twin(tmp_path, other).stage_task("lung_vessels", 0) is None


# ----------------------------------------------------------------------------------------
# the store, end to end
# ----------------------------------------------------------------------------------------

from test_cascade_union import BOX                      # noqa: E402
from test_normalization_sharing import ORGANS, _StubModel, _write_ct   # noqa: E402

from haversack import ranked_store as rs                     # noqa: E402


class _Found(_StubModel):
    """K channels; class ``cls`` wins in BOX, background elsewhere."""

    def __init__(self, K, cls):
        super().__init__(ORGANS._props, K=K)
        self.cls = cls

    def predict_logits(self, crop, report=None):
        import torch
        self.received = crop.clone()
        logits = torch.zeros((self.K, *crop.shape[1:]), dtype=torch.float32)
        logits[0] = 1.0
        z, y, x = crop.shape[1:]
        box = tuple(slice(min(s.start, n - 1), min(s.stop, n)) for s, n in zip(BOX, (z, y, x)))
        logits[self.cls][box] = 2.0
        return logits


def _verify(path):
    spec = importlib.util.spec_from_file_location("ranked_verify", TOOLS / "ranked_verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify(Path(path), deep=True, quiet=True)


def _store(tmp_path, monkeypatch, task, models, name="c.duckn", keep_stages=True, **kw):
    """``segment_to_store`` of ``task`` with the stub ``models`` handed out in stage order,
    stubbed as test_cascade_union stubs them: the store, its segmentation, its part blocks.

    ``keep_stages``: since 2026-09-24 the product path stores only the task's own field and
    leaves a cascade's crop stages out (ranked_output._task_field). The builder still builds
    every layer an emit directory holds - an older emit, or tools/ranked_emit.py's - and these
    tests are about how it NAMES them, so by default the reduction is bypassed."""
    from haversack import pipeline, ranked_output
    if keep_stages:
        monkeypatch.setattr(ranked_output, "_task_field", lambda out, metas, depth, **k: metas)
    out = tmp_path / name
    folder = tmp_path / "Dataset000_stub" / "trainer__plans__3d_fullres"
    (folder / "fold_0").mkdir(parents=True, exist_ok=True)

    class _Store:
        root = folder.parent.parent

        def resolve(self, weights_id, *, configuration=None, **k):
            return folder

        def describe(self, *a, **k):
            return {}

    class _Cache:
        def __init__(self):
            self.order = list(models)

        def get(self, folder, **k):
            return self.order.pop(0)

        def release(self, model):
            pass
    store = _Store()
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    image = _write_ct(tmp_path, (16, 20, 18))
    ranked_output.segment_to_store(str(image), task, out, depth=2, quiet=True, models=_Cache(),
                                   device="cpu", envelope_mm=None, convention="corner",
                                   folds=(0,), interp="nearest", distance_voxels=0, **kw)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
        blocks = [st.root[f"parts/{i}"].attrs.asdict()["duckn"]["extensions"]["ranked"]
                  for i in range(len(list(st.root["parts"].group_keys())))]
    return out, seg, blocks


def _by_layer_value(seg):
    return {((s.layer or 0), s.label_values[0]): s for s in seg.segments
            if len(s.label_values) == 1}


def test_a_cascade_store_names_its_crop_stage_from_the_stage_model(tmp_path, monkeypatch):
    # lung_vessels: Dataset297 (118 classes) crops to the lung lobes 10-14, Dataset117 (5)
    out, seg, blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                              [_Found(118, 10), _Found(5, 1)])
    by = _by_layer_value(seg)
    # layer 0 is total's classes, coded in total's scheme - not the fine task's
    assert [by[0, v].name for v in (1, 2, 3, 4)] == \
        ["spleen", "kidney_right", "kidney_left", "gallbladder"]
    assert [b["part"] for b in blocks] == ["ts.v2:lung_vessels:s0", "ts.v2:lung_vessels"]
    assert [b["labels_named_by"] for b in blocks] == ["ts.v2:total_fast", "ts.v2:lung_vessels"]
    assert by[0, 10].name == "lung_upper_lobe_left"
    assert not [s for (lay, _v), s in by.items() if lay == 0 and s.name.startswith("label_")]
    d = by[0, 1].designations[0]
    assert (d.scheme, d.code) == ("ts.v2:total", "spleen")
    # layer 1 is the task's
    assert [by[1, v].name for v in (1, 2, 3, 4)] == \
        ["lung_airways", "lung_airways_wall", "lung_arteries", "lung_veins"]
    assert by[1, 1].designations[0].scheme == "ts.v2:lung_vessels"
    # no name of the fine task is on the crop stage
    assert not {by[0, v].name for v in range(1, 118)} & {by[1, v].name for v in (1, 2, 3, 4)}

    # two class lists, two schemes: the task's own first, both registered
    assert seg.labeling_scheme == ["ts.v2:lung_vessels", "ts.v2:total"]
    assert seg.terminologies["ts.v2:total"].system_uri.endswith("#v2:total")
    assert _verify(out)


def test_the_last_part_alone_is_the_task_and_declares_one_scheme(tmp_path, monkeypatch):
    out, seg, blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                              [_Found(118, 10), _Found(5, 1)], parts="last")
    assert [b["part"] for b in blocks] == ["ts.v2:lung_vessels"]
    assert seg.labeling_scheme == "ts.v2:lung_vessels"
    assert _verify(out)


def test_the_product_store_of_a_cascade_is_its_final_stage(tmp_path, monkeypatch):
    """The store is the TASK's field (2026-09-24): the crop stage decided the final stage's box
    and is not the task's output, so the product path leaves it out - by the role the pipeline
    stated on its emit, never by its name - and the store is one layer, one scheme."""
    out, seg, blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                              [_Found(118, 10), _Found(5, 1)], keep_stages=False)
    assert [b["part"] for b in blocks] == ["ts.v2:lung_vessels"]
    assert "role" not in blocks[0] and blocks[0]["labels_named_by"] == "ts.v2:lung_vessels"
    assert seg.labeling_scheme == "ts.v2:lung_vessels"
    assert {s.name for s in seg.segments} == {"background", "lung_airways", "lung_airways_wall",
                                              "lung_arteries", "lung_veins"}
    assert _verify(out)


def test_an_emit_that_predates_labels_named_by_builds_the_same_segments(tmp_path, monkeypatch):
    """Emit directories written before the parts recorded ``labels_named_by`` (kept by
    tools/ranked_emit.py, rebuilt by tools/ranked_build_store.py) recognize a crop stage by its
    ``<task>:s<i>`` name, and build the store the product path does."""
    from haversack import ranked_output
    from haversack.ranked_build import build, label_task_of
    _out, seg, _blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                                [_Found(118, 10), _Found(5, 1)])
    # the same run, emitted into a directory that stays
    from haversack import pipeline
    emit = tmp_path / "emit"
    real = pipeline.segment
    models = iter([_Found(118, 10), _Found(5, 1)])

    class _Cache:
        def get(self, folder, **k):
            return next(models)

        def release(self, model):
            pass
    monkeypatch.setattr(pipeline, "segment",
                        lambda *a, **k: real(*a, **{**k, "models": _Cache()}))
    run = ranked_output.main(str(_write_ct(tmp_path, (16, 20, 18))), "ts.v2:lung_vessels", emit,
                             2, quiet=True, device="cpu", convention="corner", folds=(0,),
                             interp="nearest")
    meta = json.loads((emit / "meta.json").read_text(encoding="utf-8"))
    for part in meta["parts"].values():
        del part["labels_named_by"]
    (emit / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    old = build(emit, tmp_path / "old.duckn", "c", names=dict(run.schema.names), quiet=True,
                model_names=True, distance_voxels=0)
    with rs.open_store(old) as st:
        rebuilt = rs.read_segmentation(st.root)
    key = lambda g: [(s.id, s.name, s.layer, s.label_values,               # noqa: E731
                      [(d.scheme, d.code) for d in s.designations or ()]) for s in g.segments]
    assert key(rebuilt) == key(seg) and rebuilt.labeling_scheme == seg.labeling_scheme
    # the builder that once named every stage `<task>:s<i>` put the final stage last
    assert label_task_of("ts.v2:lung_vessels:s1", {"task": "ts.v2:lung_vessels"}, True) == \
        "ts.v2:lung_vessels"


def test_a_nested_crop_names_each_part_from_the_task_that_produced_it(tmp_path, monkeypatch):
    """teeth crops from craniofacial_structures' result, itself a cascade from Dataset298: three
    parts, three class lists, and no part name to parse for the middle one."""
    from haversack.tasks import TaskCatalog
    cranio = TaskCatalog("ts").get("craniofacial_structures")
    teeth = TaskCatalog("ts").get("teeth")
    out, seg, blocks = _store(
        tmp_path, monkeypatch, "ts.v2:teeth",
        [_Found(118, 91), _Found(max(cranio.label_map) + 1, 2), _Found(max(teeth.label_map) + 1, 1)])
    assert [b["labels_named_by"] for b in blocks] == \
        ["ts.v2:total_fastest", "ts.v2:craniofacial_structures", "ts.v2:teeth"]
    by = _by_layer_value(seg)
    assert by[0, 91].name == "skull"
    assert by[1, 2].name == cranio.label_map[2] and by[2, 1].name == teeth.label_map[1]
    assert by[1, 2].designations[0].scheme == "ts.v2:craniofacial_structures"
    assert seg.labeling_scheme == ["ts.v2:teeth", "ts.v2:total", "ts.v2:craniofacial_structures"]
    assert _verify(out)


def test_the_verifier_refuses_a_crop_stage_named_from_the_task(tmp_path, monkeypatch):
    """The store as the old builder wrote it: layer 0's values coded as the fine task's."""
    out, _seg, _blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                                [_Found(118, 10), _Found(5, 1)])
    with rs.open_store(out, "a") as st:
        attrs = st.root.attrs.asdict()
        segs = attrs["duckn"]["extensions"]["seg"]["segments"]
        fine = {s["label_values"][0]: s for s in segs if s.get("layer") == 1 and not s.get("role")}
        for s in segs:
            if not s.get("layer") and not s.get("role") and s["label_values"][0] in fine:
                f = fine[s["label_values"][0]]
                s["name"], s["designations"] = f["name"], f["designations"]
        st.root.attrs.put(attrs)
    assert not _verify(out)


def _set_direction(out, part, change):
    with rs.open_store(out, "a") as st:
        arr = st.root[f"parts/{part}/ranks"]
        attrs = arr.attrs.asdict()
        for a in attrs["duckn"]["axes"]:
            if a.get("kind") == "space":
                a["space_direction"] = change(a["space_direction"])
        arr.attrs.put(attrs)


def test_the_verifier_compares_orientations_not_spacings(tmp_path, monkeypatch):
    """A cascade's stages sit at different spacings (3 mm under 0.7 mm): the same orientation.
    The verifier compared whole direction vectors, spacing included, and failed every real
    cascade store; the stub stores share one spacing, so no test saw it (review, 2026-09-23).
    A flipped axis must still fail."""
    spec = importlib.util.spec_from_file_location("ranked_verify", TOOLS / "ranked_verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, _seg, _blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                                [_Found(118, 10), _Found(5, 1)])
    _set_direction(out, 0, lambda d: [4.0 * float(x) for x in d])
    assert mod.verify(out, quiet=True)
    _set_direction(out, 0, lambda d: [-float(x) for x in d])
    assert not mod.verify(out, quiet=True)


def test_upgrading_the_segment_metadata_keeps_every_scheme(tmp_path, monkeypatch):
    """tools/ranked_upgrade_seg.py rebuilt the seg extension without `labeling_scheme`."""
    out, seg, _blocks = _store(tmp_path, monkeypatch, "ts.v2:lung_vessels",
                               [_Found(118, 10), _Found(5, 1)])
    spec = importlib.util.spec_from_file_location("ranked_upgrade_seg",
                                                  TOOLS / "ranked_upgrade_seg.py")
    upgrader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upgrader)
    upgrader.upgrade(out)
    with rs.open_store(out) as st:
        assert rs.read_segmentation(st.root).labeling_scheme == seg.labeling_scheme


def test_a_crop_from_another_task_is_a_crop_stage_however_it_is_shaped(tmp_path, monkeypatch):
    """teeth crops from craniofacial_structures' RESULT - a cascade whose final stage was emitted
    through the ordinary path, with no role, and read as a second model of teeth: composed with
    teeth's own labels where the boxes agreed, refused as an --envelope problem where they did
    not (review, 2026-09-25). Everything a crop-from-task run emits is a crop stage now."""
    from haversack.tasks import TaskCatalog
    cranio = TaskCatalog("ts").get("craniofacial_structures")
    teeth = TaskCatalog("ts").get("teeth")
    out, seg, blocks = _store(
        tmp_path, monkeypatch, "ts.v2:teeth",
        [_Found(118, 91), _Found(max(cranio.label_map) + 1, 2), _Found(max(teeth.label_map) + 1, 1)],
        keep_stages=False)
    assert [b["part"] for b in blocks] == ["ts.v2:teeth"]
    assert blocks[0].get("scores") != "composed" and blocks[0]["labels_named_by"] == "ts.v2:teeth"
    assert _verify(out)


def test_a_cascade_whose_crop_finds_nothing_says_so(tmp_path, monkeypatch):
    """No lung in the scan: lung_vessels' crop finds none of its classes, its final model never
    runs, and only the crop stage was emitted. That left the build a bare StopIteration."""
    from haversack.errors import InputError
    with pytest.raises(InputError, match="no field to store"):
        _store(tmp_path, monkeypatch, "ts.v2:lung_vessels", [_Found(118, 1), _Found(5, 1)],
               keep_stages=False)
