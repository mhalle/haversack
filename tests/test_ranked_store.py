"""The on-disk ranked store: one hierarchy in two standard containers, duckn metadata
through duckn's own models.

INTERNAL. The store module is deliberately undocumented (see its docstring); these tests
are its contract while it moves. They skip without zarr or duckn, the way the other
ranked-store tests do - the main test env installs both from the sibling checkout.
"""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr")
duckn = pytest.importorskip("duckn")

from haversack import ranked_store as rs                      # noqa: E402
from haversack.ranked import encode                           # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def _tool(name):
    if name == "ranked_build_store":                 # the builder lives in the package now
        return importlib.import_module("haversack.ranked_build")
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------
# the container
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".duckn", ".duckn.zip"])
def test_a_store_round_trips_through_either_container(tmp_path, suffix):
    p = tmp_path / f"s{suffix}"
    data = np.random.default_rng(0).integers(0, 4, (2, 70, 65, 66), dtype=np.uint8)
    with rs.open_store(p, "w") as st:
        assert st.is_zip == suffix.endswith(".zip")
        g = st.root.create_group("parts/0")
        z = g.create_array("ranks", shape=data.shape, dtype="uint8", chunks=(1, 64, 64, 64),
                           shards=(1, 128, 128, 128),
                           attributes=rs.grid_attrs(D, [1.5, 1.5, 1.5], [0, 0, 0],
                                                    list_axis=True, centering="node"))
        z[:] = data
        st.write_text("README.md", "# the format\n")
    with rs.open_store(p, "r") as st:
        assert st.exists("README.md") and st.read_text("README.md") == "# the format\n"
        np.testing.assert_array_equal(st.root["parts/0/ranks"][:], data)
        assert st.size_bytes() > 0


def test_the_zip_is_a_standard_zarr_zip_store(tmp_path):
    """No knowledge of this package is needed to read it: zarr's own ZipStore opens it,
    the entries are the zarr hierarchy, stored uncompressed (chunks are already zstd)."""
    p = tmp_path / "s.zip"
    with rs.open_store(p, "w") as st:
        st.root.create_group("parts/0").create_array("ranks", shape=(4, 4, 4), dtype="uint8",
                                                     chunks=(4, 4, 4))[:] = 3
        st.write_text("README.md", "x")
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        assert "zarr.json" in names and "parts/0/ranks/zarr.json" in names
        assert "README.md" in names
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
    from zarr.storage import ZipStore
    root = zarr.open_group(store=ZipStore(str(p), mode="r"), mode="r")
    assert int(root["parts/0/ranks"][0, 0, 0]) == 3


@pytest.mark.parametrize("suffix", [".duckn", ".duckn.zip"])
def test_a_store_can_be_amended_in_place_and_a_zip_stays_duplicate_free(tmp_path, suffix):
    """Amending rewrites group attributes, which a zip entry cannot do - so a zip is amended
    in a staging directory and repacked, and the archive holds every key exactly once."""
    p = tmp_path / f"s{suffix}"
    with rs.open_store(p, "w") as st:
        g = st.root.create_group("parts/0")
        g.create_array("ranks", shape=(4, 4, 4), dtype="uint8", chunks=(4, 4, 4))[:] = 1
        g.attrs.update(rs.part_attrs({"version": "0.1"}))
        st.write_text("README.md", "v1")
    with rs.open_store(p, "a") as st:
        g = st.root["parts/0"]
        g.create_array("distance", shape=(4, 4, 4), dtype="uint8", chunks=(4, 4, 4))[:] = 9
        g.attrs.update(rs.part_attrs({"version": "0.1", "distance_max": 255}))
        st.write_text("README.md", "v2")
    with rs.open_store(p, "r") as st:
        assert sorted(st.root["parts/0"].array_keys()) == ["distance", "ranks"]
        assert rs.read_metadata(st.root["parts/0"]).extensions["ranked"]["distance_max"] == 255
        assert st.read_text("README.md") == "v2"
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            assert len(names) == len(set(names))
        assert not (tmp_path / f"s{suffix}.staging").exists()


def test_an_exception_while_amending_a_zip_leaves_the_archive_untouched(tmp_path):
    p = tmp_path / "s.zip"
    with rs.open_store(p, "w") as st:
        st.write_text("README.md", "v1")
    before = p.read_bytes()
    with pytest.raises(RuntimeError):
        with rs.open_store(p, "a") as st:
            st.write_text("README.md", "v2")
            raise RuntimeError("midway")
    assert p.read_bytes() == before
    assert not [q for q in tmp_path.iterdir() if ".staging" in q.name or q.suffix == ".lock"]


def test_write_mode_replaces_a_store_and_nothing_else(tmp_path):
    from haversack.errors import InputError
    p = tmp_path / "s.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(InputError, match="not a ranked store"):
        rs.open_store(p, "w")
    assert p.read_bytes() == b"not a zip"
    p.unlink()
    with rs.open_store(p, "w") as st:
        st.write_text("README.md", "x")
    with rs.open_store(p, "w") as st:                          # a store may be replaced
        st.write_text("README.md", "y")
    with rs.open_store(p, "r") as st:
        assert st.read_text("README.md") == "y"


def test_writing_through_a_symlink_updates_the_real_store(tmp_path):
    real = tmp_path / "vol" / "real.duckn"
    with rs.open_store(real, "w") as st:
        st.write_text("README.md", "v1")
    link = tmp_path / "link.duckn"
    link.symlink_to(real)
    with rs.open_store(link, "w") as st:
        st.write_text("README.md", "v2")
    assert link.is_symlink() and (real / "README.md").read_text() == "v2"
    assert not [q for q in tmp_path.iterdir() if ".old-" in q.name or ".staging" in q.name]


def test_a_failure_after_staging_is_created_leaves_nothing_behind(tmp_path, monkeypatch):
    import zarr
    monkeypatch.setattr(zarr, "create_group", lambda **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        rs.open_store(tmp_path / "s.duckn.zip", "w")
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------------------
# the metadata: duckn's models in, duckn's validators out
# ----------------------------------------------------------------------------------------

def test_grid_attrs_are_duckn_metadata_that_validate_against_the_shape():
    from duckn import DucknMetadata, validate_against_shape
    a = rs.grid_attrs(D, [3.0, 1.5, 1.5], [10.0, -5.0, 2.0], list_axis=True, centering="node")
    m = DucknMetadata.model_validate(json.loads(json.dumps(a["duckn"])))
    assert [ax.kind for ax in m.axes] == ["list", "space", "space", "space"]
    assert m.axes[1].space_direction == [0.0, 0.0, 3.0]        # array Z is world z here
    assert all(ax.centering == "node" for ax in m.axes[1:])
    validate_against_shape(m, (4, 10, 20, 30))
    with pytest.raises(ValueError):
        validate_against_shape(m, (10, 20, 30))                 # a list axis is declared


def test_brick_attrs_place_the_first_brick_centre_and_scale_the_spacing():
    a = rs.brick_attrs(D, [3.0, 1.5, 1.5], [0.0, 0.0, 0.0], 32)["duckn"]
    assert a["space_origin"] == [23.25, 23.25, 46.5]            # (32-1)/2 * spacing, xyz
    assert a["axes"][1]["space_direction"] == [0.0, 0.0, 96.0]
    assert all(ax["centering"] == "cell" for ax in a["axes"][1:])


def test_a_group_of_one_is_a_valid_group():
    """A named union with one member coincides with its leaf and is still its own
    statement - identity is the id, not the voxels (duckn seg spec 0.7 §5)."""
    seg = rs.segmentation([rs.leaf("c5", "liver", 5, layer=0),
                           rs.group("g_one", "just the liver", ["c5"])])
    assert [s.id for s in seg.segments] == ["c5", "g_one"]


def test_the_standard_refuses_two_leaves_on_one_value_and_a_false_disjoint_claim():
    with pytest.raises(ValueError, match="all claim label value 5"):
        rs.segmentation([rs.leaf("a", "liver", 5), rs.leaf("b", "hepar", 5)])
    with pytest.raises(ValueError, match="claims disjoint members"):
        rs.segmentation([rs.leaf("a", "x", 1), rs.group("u", "union", ["a"]),
                         rs.group("v", "other union", ["a"]),
                         rs.group("p", "not a partition", ["u", "v"], disjoint=True)])


def test_root_attrs_read_back_through_the_standard(tmp_path):
    seg = rs.segmentation([rs.leaf("bg", "background", 0, layer=0, background=True),
                           rs.leaf("c5", "liver", 5, layer=0, extent=[0, 3, 0, 4, 0, 5]),
                           rs.leaf("c6", "spleen", 6, layer=0),
                           rs.group("g_ab", "abdomen", ["c5", "c6"], disjoint=True),
                           rs.group("all", "everything", ["bg", "c5", "c6"],
                                    disjoint=True, exhaustive=True)])
    with rs.open_store(tmp_path / "s.zip", "w") as st:
        st.root.attrs.update(rs.root_attrs(seg, haversack={"engine": "nnunetv2"},
                                           provenance={"version": "1.0", "processing": []}))
    with rs.open_store(tmp_path / "s.zip", "r") as st:
        back = rs.read_segmentation(st.root)
        assert [s.id for s in back.segments] == ["bg", "c5", "c6", "g_ab", "all"]
        assert back.segments[3].members == ["c5", "c6"] and back.segments[3].disjoint
        assert back.segments[0].background and back.segments[4].exhaustive
        assert (rs.read_metadata(st.root).extensions["haversack"]["engine"] == "nnunetv2")


def test_a_geometry_the_standard_rejects_never_reaches_the_store():
    with pytest.raises(ValueError, match="three spacings"):     # ours: an axis per array dim
        rs.grid_metadata(D, [1.0, 1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="space_origin"):       # duckn's: origin in 3-space
        rs.grid_metadata(D, [1.0, 1.0, 1.0], [0.0, 0.0])


# ----------------------------------------------------------------------------------------
# the builder and the verifier, end to end, in both containers
# ----------------------------------------------------------------------------------------

def _synthetic_emit(tmp_path):
    """A one-part emit directory of the shape ranked_emit.py writes, from random logits."""
    build = _tool("ranked_build_store")
    names = build.names_for("nnunetv2", "total_fast")
    labels = [0] + sorted(names)[:2]                          # background + two real ids
    torch.manual_seed(0)
    logits = torch.randn(len(labels), 20, 24, 28)
    logits[0] += 1.5                                          # background mostly wins
    code = encode(logits, depth=2, clip=8.0)
    src = tmp_path / "emit"
    src.mkdir()
    for nm, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
        if arr is not None:
            np.save(src / f"organs_{nm}.npy", arr)
    part = {**code.meta, "engine": "nnunetv2", "task": "total_fast", "part": "organs",
            "labels": labels, "convention": "corner", "spacing_zyx": [3.0, 3.0, 3.0],
            "frame": {"canonical": {"shape_zyx": [40, 48, 56], "spacing_zyx": [1.5, 1.5, 1.5],
                                    "origin_xyz": [-10.0, -20.0, 5.0], "direction_xyz": D}},
            "model_grid": [20, 24, 28], "envelope": {"start": [0, 0, 0], "stop": [20, 24, 28]},
            "softmax": {"classes": len(labels), "weights": "synthetic", "version": "0"},
            "haversack": "test"}
    (src / "meta.json").write_text(json.dumps(
        {"image": "synthetic.nii", "task": "total_fast", "depth": 2, "clip": 8.0,
         "envelope_mm": None, "parts": {"organs": part}}, default=str))
    return src


def _arrays(root):
    out = {}
    for name, g in root["parts"].groups():
        for an, a in g.arrays():
            out[f"{name}/{an}"] = (np.asarray(a[:]), json.dumps(a.attrs.asdict(), sort_keys=True))
        out[f"{name}/@"] = json.dumps(g.attrs.asdict(), sort_keys=True)
    out["@"] = json.dumps(root.attrs.asdict(), sort_keys=True)
    return out


def test_build_writes_the_same_store_into_a_directory_and_a_zip_and_both_verify(tmp_path):
    build, verify = _tool("ranked_build_store"), _tool("ranked_verify")
    src = _synthetic_emit(tmp_path)
    a = build.build(src, tmp_path / "case.duckn", "case")
    b = build.build(src, tmp_path / "case.duckn.zip", "case")
    assert b.is_file() and zipfile.is_zipfile(b)
    assert verify.verify(a, deep=True, quiet=True)
    assert verify.verify(b, deep=True, quiet=True)
    with rs.open_store(a) as sa, rs.open_store(b) as sb:
        xa, xb = _arrays(sa.root), _arrays(sb.root)
        assert xa.keys() == xb.keys()
        assert {"0/ranks", "0/support", "0/occupancy", "0/distance", "0/junction",
                "0/junction_pair"} <= xa.keys()
        for k in xa:
            if k.endswith("@"):
                assert xa[k] == xb[k], k
            else:
                np.testing.assert_array_equal(xa[k][0], xb[k][0], err_msg=k)
                assert xa[k][1] == xb[k][1], k
        assert sa.read_text("README.md") == sb.read_text("README.md")
        # the root went through duckn's model: the seg extension reads back validated,
        # with the part's partition (background included) stated as a group
        seg = rs.read_segmentation(sb.root)
        assert all(not s.name.startswith("label_") for s in seg.segments)
        by_id = {s.id: s for s in seg.segments}
        assert by_id["background_0"].background and by_id["background_0"].label_value == 0
        part = by_id["classes_0"]
        assert part.disjoint and part.exhaustive
        assert set(part.members) == {s.id for s in seg.segments if s.label_value is not None}


def test_the_junction_layer_can_be_appended_to_an_existing_zip(tmp_path):
    build, verify, junction = (_tool("ranked_build_store"), _tool("ranked_verify"),
                               _tool("ranked_add_junction"))
    src = _synthetic_emit(tmp_path)
    p = build.build(src, tmp_path / "case.duckn.zip", "case", distance_voxels=0)
    with rs.open_store(p) as st:
        assert "junction" not in st.root["parts/0"]
    junction.add(p)
    with rs.open_store(p) as st:
        g = st.root["parts/0"]
        assert "junction" in g and "junction_pair" in g
        block = rs.read_metadata(g).extensions["ranked"]
        assert block["junction_zero"] == 128 and block["junction_truncation"] > 0
        steps = rs.read_metadata(st.root).extensions["provenance"]["processing"]
        assert any(s["name"] == "Triple-line junction layer" for s in steps)
    assert verify.verify(p, quiet=True)
    junction.add(p, force=True)                     # a redo repacks; nothing is duplicated
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names))
    assert verify.verify(p, quiet=True)


# ----------------------------------------------------------------------------------------
# review fixes (2026-09-03): one writer, nothing replaced before it is complete
# ----------------------------------------------------------------------------------------

def test_a_directory_store_is_replaced_only_by_a_complete_build(tmp_path):
    p = tmp_path / "s.duckn"
    with rs.open_store(p, "w") as st:
        st.write_text("README.md", "v1")
    with pytest.raises(RuntimeError):
        with rs.open_store(p, "w") as st:
            st.write_text("README.md", "v2")
            raise RuntimeError("midway")
    with rs.open_store(p) as st:
        assert st.read_text("README.md") == "v1"
    assert not [q for q in tmp_path.iterdir() if ".staging" in q.name or ".old-" in q.name
                or q.suffix == ".lock"]


def test_a_directory_that_is_not_a_store_is_never_replaced(tmp_path):
    from haversack.errors import InputError
    p = tmp_path / "photos.duckn"
    p.mkdir()
    (p / "holiday.jpg").write_bytes(b"jpeg")
    with pytest.raises(InputError, match="not a ranked store"):
        rs.open_store(p, "w")
    with pytest.raises(InputError, match="not a ranked store"):
        rs.open_store(p, "a")
    assert (p / "holiday.jpg").exists() and not (p.with_name("photos.duckn.lock")).exists()


def test_two_writers_on_one_store_are_refused_at_open(tmp_path):
    from haversack.errors import InputError
    p = tmp_path / "s.duckn.zip"
    with rs.open_store(p, "w") as st:
        st.write_text("README.md", "v1")
        with pytest.raises(InputError, match="another process"):
            rs.open_store(p, "w")
    with rs.open_store(p, "a") as st:          # released with the handle
        st.write_text("README.md", "v2")
    assert not list(tmp_path.glob("*.lock"))


def _two_part_emit(tmp_path, names, part_names=("total_fast:s0", "total_fast:s1")):
    """A cascade-shaped emit: two parts that both emit values 1 and 2."""
    build = _tool("ranked_build_store")
    labels = [0, 1, 2]
    torch.manual_seed(1)
    src = tmp_path / "emit2"
    src.mkdir()
    parts = {}
    for pn in part_names:
        code = encode(torch.randn(3, 8, 8, 8), depth=2, clip=8.0)
        for nm, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
            if arr is not None:
                np.save(src / f"{pn}_{nm}.npy", arr)      # the emit names files by the part
        parts[pn] = {**code.meta, "engine": "nnunetv2", "task": "liver_segments", "part": pn,
                     "labels": labels, "convention": "corner", "spacing_zyx": [3.0] * 3,
                     "frame": {"canonical": {"shape_zyx": [16, 16, 16], "spacing_zyx": [1.5] * 3,
                                             "origin_xyz": [0.0, 0.0, 0.0], "direction_xyz": D}},
                     "model_grid": [8, 8, 8], "envelope": {"start": [0, 0, 0], "stop": [8, 8, 8]},
                     "softmax": {"classes": 3, "weights": "synthetic", "version": "0"},
                     "haversack": "test"}
    (src / "meta.json").write_text(json.dumps(
        {"image": "synthetic.nii", "task": "liver_segments", "depth": 2, "clip": 8.0,
         "envelope_mm": None, "parts": parts}, default=str))
    return build, src


def test_leaves_are_unique_per_layer_and_value_so_a_cascade_keeps_every_class(tmp_path):
    """Both stages of a cascade emit channel indices 1..K-1. A dedupe on the value alone gave
    stage 1 no leaves while `classes_1` still claimed to be exhaustive."""
    build, src = _two_part_emit(tmp_path, None)
    out = build.build(src, tmp_path / "c.duckn", "c", names={1: "segment_1", 2: "segment_2"},
                      quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    by = {s.id: s for s in seg.segments}
    assert {"c1_l0", "c2_l0", "c1_l1", "c2_l1"} <= by.keys()
    assert by["c1_l1"].name == "segment_1" and by["c1_l1"].layer == 1
    assert by["c1_l0"].name == "label_1"                        # stage 0 has its own classes
    assert set(by["classes_1"].members) == {"background_1", "c1_l1", "c2_l1"}
    assert set(by["classes_0"].members) == {"background_0", "c1_l0", "c2_l0"}
    verify = _tool("ranked_verify")
    assert verify.verify(out, deep=True, quiet=True)


def test_a_single_part_store_states_no_layer(tmp_path):
    build = _tool("ranked_build_store")
    out = build.build(_synthetic_emit(tmp_path), tmp_path / "one.duckn", "one", quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    assert all(s.layer is None for s in seg.segments)


def test_the_lungs_claim_needs_the_lobes_not_the_prefix():
    build = _tool("ranked_build_store")
    def leaves(names):
        return [rs.leaf(f"c{i + 1}", n, i + 1) for i, n in enumerate(names)]
    vessels = build.named_groups("nnunetv2", leaves(["lung_vessels", "lung_trachea_bronchia"]))
    assert not [g for g in vessels if g.id == "g_lungs"]
    lobes = build.named_groups("nnunetv2", leaves([
        "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
        "lung_middle_lobe_right", "lung_lower_lobe_right", "liver"]))
    g = {x.id: x for x in lobes}["g_lungs"]
    assert g.exhaustive and len(g.members) == 5
    pair = build.named_groups("nnunetv2", leaves(["lung_left", "lung_right"]))
    assert {x.id for x in pair} == {"g_lungs"}
    four = build.named_groups("nnunetv2", leaves([
        "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
        "lung_middle_lobe_right"]))
    assert not [x for x in four if x.id == "g_lungs"]           # a lobe short of the concept


def test_the_upgrade_tool_parses_arguments_and_names_a_store_that_is_not_one(tmp_path, capsys):
    up, build = _tool("ranked_upgrade_seg"), _tool("ranked_build_store")
    out = build.build(_synthetic_emit(tmp_path), tmp_path / "one.duckn.zip", "one", quiet=True)
    with pytest.raises(SystemExit):
        up.main(["--help"])
    with pytest.raises(SystemExit):
        up.main(["--no-such-flag", str(out)])
    up.main([str(out)])
    assert "seg 0.7 -> 0.7" in capsys.readouterr().out
    with rs.open_store(out) as st:
        assert all(s.layer is None for s in rs.read_segmentation(st.root).segments)
    bare = tmp_path / "bare.duckn"
    with rs.open_store(bare, "w") as st:
        st.write_text("README.md", "not a haversack store")
    with pytest.raises(SystemExit, match="no `haversack`"):
        up.main([str(bare)])
