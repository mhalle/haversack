"""A rank-field store under duckn convention 1.2 (2026-09-26, the owner's decisions after the
conformance audit): the store's groups carry duckn group metadata (duckn-spec §3.3), its `seg`
block is seg 0.10's group form naming each layer's part, every extension has a version, the
encoded arrays declare 1.2 (their absent value_transforms reads as "not stated", which is true
of codes), the distance array states its mapping in millimeters, and the provenance states only
what is known - including a cascade's crop model."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zarr")
duckn = pytest.importorskip("duckn")
if not hasattr(duckn, "DucknGroupMetadata"):
    pytest.skip("needs duckn with convention 1.2", allow_module_level=True)

from haversack import ranked_store as rs  # noqa: E402

from test_ranked_store import _synthetic_emit, _tool  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rf12")
    build = _tool("ranked_build_store")
    return build.build(_synthetic_emit(tmp), tmp / "s.duckn.zip", "s", quiet=True)


def _attrs(node):
    return node.attrs.asdict()["duckn"]


def test_the_groups_are_duckn_1_2_groups(built):
    from duckn import DucknGroupMetadata
    with rs.open_store(built) as st:
        for g in (st.root, st.root["parts/0"]):
            DucknGroupMetadata(**_attrs(g))                        # validates, or raises
        ext = _attrs(st.root)["extensions"]
        assert all("version" in e for e in ext.values())            # haversack's included


def test_the_seg_block_names_each_layers_part(built):
    with rs.open_store(built) as st:
        seg = rs.read_segmentation(st.root)
        assert seg.version == "0.10"
        assert [(r.path, r.labelmap_from) for r in seg.layers] == [("parts/0", "ranked")]


def test_encoded_arrays_state_no_calibration_and_distance_states_millimeters(built):
    from duckn import DucknMetadata
    with rs.open_store(built) as st:
        part = st.root["parts/0"]
        for name in part.array_keys():
            meta = DucknMetadata(**_attrs(part[name]))
            if name == "distance":
                (vt,) = meta.value_transforms
                assert vt.name == "linear" and meta.sample_units == "mm"
                block = _attrs(part)["extensions"]["ranked"]
                T, dmax = block["distance_truncation"], block["distance_max"]
                assert vt.parameters["intercept"] == pytest.approx(T)
                assert vt.parameters["slope"] == pytest.approx(-T / dmax)
            else:
                assert meta.values_stated() is False, name        # codes: not a quantity


def test_the_provenance_states_only_what_is_known(built):
    with rs.open_store(built) as st:
        steps = _attrs(st.root)["extensions"]["provenance"]["processing"]

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)
        else:
            yield x
    assert not [v for v in walk(steps) if v is None or v == "unknown"]
    assert steps[1]["software"]["name"] == "haversack.ranked_build"


def test_a_cascades_crop_model_is_recorded_with_its_role(tmp_path):
    from haversack.ranked_build import generator_steps
    from haversack.ranked_output import _task_field
    metas = {"crop": {"role": "crop", "softmax": {"model": "Dataset297", "version": "v2"}},
             "fine": {"softmax": {"model": "Dataset789", "version": "v2", "sha256": None}}}
    kept = _task_field(tmp_path, metas, 3, say=lambda *a, **k: None)
    steps = generator_steps({"task": "ts.v2:kidney_cysts"}, list(kept.items()), "nnunetv2")
    models = steps[0]["method"]["models"]
    assert {"model": "Dataset297", "version": "v2", "role": "crop"} in models
    assert {"model": "Dataset789", "version": "v2"} in models                # null left out


def test_a_fastsurfer_channel_that_holds_both_hemispheres_names_neither():
    """A store's channels come before FastSurfer's hemisphere split: 1035 (insula) covers both
    insulae there, so 'ctx-lh-insula' would state an untrue laterality. Lateralized channels
    (1025, which the network separates itself) keep their name."""
    from haversack.engines import fastsurfer
    names = fastsurfer.label_names()
    assert names[1035] == "ctx-insula"
    assert names[1025].startswith("ctx-lh-")
    assert not [n for i, n in names.items() if i in fastsurfer.SPLIT_AFTER_THE_NETWORK
                and ("ctx-lh-" in n or "ctx-rh-" in n)]
    assert fastsurfer.output_lut()[1035]["name"] == "ctx-lh-insula"   # a split labels output
