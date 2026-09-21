"""The duckn labeling scheme each catalog declares (seg 0.8), and what it may code.

A scheme's ``uri`` identifies a CLASS LIST across stores and is compared byte for byte, so
it carries what changes the list and never the release; ``version`` is the release. A scheme
is declared only where the names a store holds ARE upstream's own - each claim of that kind
below was checked against upstream on 2026-09-21, and the tests here are what keep it true.
"""
import json
from pathlib import Path

import pytest

from haversack import ecosystems as ec

DATA = Path(ec.__file__).parent / "data"
SEGMENTS = json.loads((DATA / "segments.json").read_text(encoding="utf-8"))["tasks"]


def _classes(qualified):
    return {int(s["value"]): s["id"] for s in SEGMENTS[qualified]["segments"]}


def _every_scheme():
    for eco in ec.known_ecosystems():
        for task in eco.tasks():
            scheme = eco.labeling_scheme(task)
            if scheme is not None:
                yield eco, task, scheme


def test_every_declared_scheme_is_whole_and_keeps_the_release_out_of_its_uri():
    seen = list(_every_scheme())
    assert len(seen) > 60
    for eco, task, s in seen:
        assert set(s) == {"key", "name", "uri", "version", "url"}, (eco.name, task)
        assert all(isinstance(v, str) and v for v in s.values()), (eco.name, task, s)
        assert s["version"] != "unknown"
        assert s["version"] not in s["uri"], f"{eco.name}:{task} puts the release in its uri"
        assert s["uri"].split("#")[0].startswith("https://") and "#" in s["uri"]


def test_one_uri_is_one_class_list():
    """Two tasks that declare one scheme must hold the same classes; where the mined index
    knows both, check it - this is what `SAME_CLASSES`, `REPACKAGED` and the `_fast` folding
    each assert."""
    by_uri = {}
    for eco, task, s in _every_scheme():
        by_uri.setdefault(s["uri"], []).append(f"{eco.name}:{task}")
    shared = {u: ts for u, ts in by_uri.items() if len(ts) > 1}
    assert any("vibe" in u for u in shared) and any("zenodo" in u for u in shared)
    for uri, tasks in shared.items():
        known = [t for t in tasks if t in SEGMENTS]
        lists = [_classes(t) for t in known]
        assert all(one == lists[0] for one in lists), f"{uri}: {known} differ"


def test_unknown_tasks_raise_and_catalogs_without_a_class_list_declare_none():
    declaring = {eco.name for eco, _task, _s in _every_scheme()}
    assert declaring == {"ts.v2", "moose", "mrsegmentator", "dentalsegmentator", "totalvibe",
                         "cads", "fastsurfer", "monai"}
    for eco in ec.known_ecosystems():
        if eco.name in declaring:                      # the others answer None to anything
            with pytest.raises(LookupError):
                eco.labeling_scheme("no_such_task")
    by_name = {e.name: e for e in ec.known_ecosystems()}
    assert by_name["mrsegmentator"].labeling_scheme("body_comp") is None    # German names
    assert by_name["totalvibe"].labeling_scheme("body_regions") is None     # digit names
    assert by_name["totalvibe"].labeling_scheme("feet_bones") is None
    for name in ("synthstrip", "voxtell", "custom"):
        eco = by_name.get(name)
        if eco is not None:
            assert all(eco.labeling_scheme(t) is None for t in eco.tasks()), name


def test_the_reasons_for_declaring_none_still_hold():
    """If a future checkpoint names these classes properly, revisit the decision."""
    assert any(not n.isascii() or " " in n for n in _classes("mrsegmentator:body_comp").values())
    assert all(n.isdigit() for n in _classes("totalvibe:body_regions").values())


def test_moose():
    eco = ec.MooseEcosystem()
    s = eco.labeling_scheme("clin_ct_organs")
    assert s["key"] == "moose:clin_ct_organs"
    assert s["uri"] == "https://github.com/ENHANCE-PET/MOOSE#clin_ct_organs"
    assert s["url"].startswith("https://github.com/ENHANCE-PET/MOOSE/releases/tag/moosez-v.")
    # a `fast_*` task is its own upstream model with its own release: not folded, unlike ts.v2
    assert eco.labeling_scheme("clin_ct_fast_organs")["key"] == "moose:clin_ct_fast_organs"
    # a repackaged model declares the scheme of the model it is
    assert eco.labeling_scheme("clin_ct_dental") == ec.DentalSegmentatorEcosystem().labeling_scheme("base")


def test_a_moose_asset_with_no_release_stamp_declares_none(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"tasks": {"x": {
        "url": "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/x.zip",
        "folder": "Dataset1_x/t", "tag": "unknown"}}}))
    assert ec.MooseEcosystem(manifest=manifest).labeling_scheme("x") is None


def test_mrsegmentator_versions_by_the_weights_not_the_source_tag():
    s = ec.MRSegmentatorEcosystem().labeling_scheme("base")
    assert s["version"] == "1.2" and s["url"].endswith("/tree/v1.2.0")
    assert s["uri"] == "https://github.com/hhaentze/MRSegmentator#base"


def test_dentalsegmentator_is_identified_by_the_concept_doi():
    s = ec.DentalSegmentatorEcosystem().labeling_scheme("base")
    assert s["uri"] == "https://doi.org/10.5281/zenodo.10829674#base"
    assert s["version"] == "v100" and s["url"] == "https://zenodo.org/records/10829675"


def test_totalvibe_uses_the_repositorys_own_casing_and_shares_vibes_scheme():
    eco = ec.TotalVibeEcosystem()
    a, b = eco.labeling_scheme("vibe"), eco.labeling_scheme("vibe_sagittal")
    assert a == b and a["key"] == "totalvibe:vibe"
    assert a["uri"] == "https://github.com/robert-graf/VIBESegmentator#100:vibe"
    assert "VibeSegmentator" not in (DATA / "totalvibe_weights.json").read_text()


def test_cads_names_are_upstreams_value_for_value():
    """The claim behind the upstream uri, pinned: the nine class lists equal CADS's own
    `bodyparts_labelmaps.map_taskid_to_labelmaps` at the release, `0: background` aside."""
    upstream = json.loads((Path(__file__).parent / "fixtures" / "cads_labelmaps_v1.0.0.json")
                          .read_text(encoding="utf-8"))["labelmaps"]
    eco = ec.CADSEcosystem()
    assert len(eco.tasks()) == 9
    uris = set()
    for task in eco.tasks():
        s = eco.labeling_scheme(task)
        dsid = s["uri"].partition("#")[2].partition(":")[0]
        theirs = {int(v): n for v, n in upstream[dsid].items() if int(v) != 0}
        assert _classes(f"cads:{task}") == theirs, task
        assert s["version"] == "cads-model-open_v1.0.0" and s["url"].endswith(s["version"])
        uris.add(s["uri"])
    assert len(uris) == 9


def test_fastsurfer_codes_are_the_ids_and_a_bilateral_channel_has_none():
    from haversack.engines.fastsurfer import SPLIT_AFTER_THE_NETWORK, load_lut
    from haversack.engines.registry import ENGINES
    eco = ec.FastSurferEcosystem()
    s = eco.labeling_scheme("asegdkt")
    assert s["key"] == "fastsurfer:asegdkt"
    assert s["uri"] == "https://github.com/Deep-MI/FastSurfer#v2:asegdkt"
    assert s["version"] == ENGINES["fastsurfer"].weights_identity()[0]["version"]
    assert s["url"].endswith(f"/tree/v{s['version']}")
    lut = load_lut()
    assert len(SPLIT_AFTER_THE_NETWORK) == 19 and SPLIT_AFTER_THE_NETWORK <= set(lut)
    assert all(lut[v]["name"].startswith("ctx-lh-") for v in SPLIT_AFTER_THE_NETWORK)
    # the LUT's own asymmetry is the evidence: every lh id that is NOT split has an rh twin
    lh = {v for v in lut if 1000 <= v < 2000}
    assert {v + 1000 for v in lh - SPLIT_AFTER_THE_NETWORK} <= set(lut)
    coded = {v: eco.scheme_code("asegdkt", v, lut[v]["name"]) for v in lut}
    assert coded[17] == "17"
    assert {v for v, c in coded.items() if c is None} == SPLIT_AFTER_THE_NETWORK
    assert ec.TSEcosystem().scheme_code("total", 1, "spleen") == "spleen"     # names, as before


def test_monai_identifies_a_bundle_by_name_and_versions_it_by_release(tmp_path):
    eco = ec.MonaiEcosystem()
    task = sorted(eco.tasks())[0]
    s = eco.labeling_scheme(task)
    assert s["key"] == f"monai:{task}"
    assert s["uri"] == f"https://github.com/Project-MONAI/model-zoo#{task}"
    assert s["version"] in s["url"]
