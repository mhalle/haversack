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

from haversack.values import Geometry
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
    assert link.is_symlink() and (real / "README.md").read_text(encoding="utf-8") == "v2"
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


def test_two_segments_may_list_one_value_and_the_later_one_answers():
    """seg 0.8: a value may belong to several segments of a layer - identity is the id, not
    the voxels - and the topmost, the last listed, answers for it."""
    from duckn import topmost_for
    seg = rs.segmentation([rs.segment("c5", "liver", 5), rs.segment("again", "hepar", 5)])
    assert [s.id for s in seg.segments] == ["c5", "again"]
    assert topmost_for(seg, 5).id == "again"


def test_the_standard_refuses_what_a_reader_would_refuse():
    with pytest.raises(ValueError, match="rule-4a"):                  # one id, twice
        rs.segmentation([rs.segment("a", "liver", 5), rs.segment("a", "hepar", 6)])
    with pytest.raises(ValueError, match="rule-14"):                  # a structure on the background
        rs.segmentation([rs.segment("bg", "background", 0, role="background"),
                         rs.segment("a", "liver", 0)])
    with pytest.raises(ValueError, match="rule-12"):                  # the containing one goes first
        rs.segmentation([rs.segment("a", "x", 1), rs.segment("u", "union", [1, 2])])


def test_a_color_is_written_as_the_css_string_duckn_gives_it():
    seg = rs.segmentation([rs.segment("a", "x", 1, color=[1.0, 0.0, 0.0]),
                           rs.segment("b", "y", 2, color="#00ff00")])
    assert [s.color for s in seg.segments] == ["#ff0000", "#00ff00"]


def test_root_attrs_read_back_through_the_standard(tmp_path):
    seg = rs.segmentation([rs.segment("bg", "background", 0, role="background"),
                           rs.segment("c5", "liver", 5, extent=[0, 3, 0, 4, 0, 5]),
                           rs.segment("c6", "spleen", 6)])
    with rs.open_store(tmp_path / "s.zip", "w") as st:
        st.root.attrs.update(rs.root_attrs(seg, haversack={"engine": "nnunetv2"},
                                           provenance={"version": "1.0", "processing": []}))
    with rs.open_store(tmp_path / "s.zip", "r") as st:
        back = rs.read_segmentation(st.root)
        assert back.version == "0.9"
        assert [(s.id, s.label_values) for s in back.segments] == [
            ("bg", [0]), ("c5", [5]), ("c6", [6])]
        assert back.segments[0].role == "background" and back.segments[1].extent == [0, 3, 0, 4, 0, 5]
        assert (rs.read_metadata(st.root).extensions["haversack"]["engine"] == "nnunetv2")


def test_a_store_written_under_seg_0_7_still_reads(tmp_path):
    """Stores already delivered carry leaves, groups and `background: true`. duckn migrates
    them on read: a group becomes a segment listing its members' values, listed first."""
    with rs.open_store(tmp_path / "old.zip", "w") as st:
        st.root.attrs.update({"duckn": {"version": rs.DUCKN_VERSION, "extensions": {"seg": {
            "version": "0.7", "segments": [
                {"id": "bg", "name": "background", "label_value": 0, "background": True},
                {"id": "c5", "name": "liver", "label_value": 5},
                {"id": "c6", "name": "spleen", "label_value": 6},
                {"id": "g_ab", "name": "abdomen", "members": ["c5", "c6"], "disjoint": True}]}}}})
    with rs.open_store(tmp_path / "old.zip", "r") as st:
        back = rs.read_segmentation(st.root)
    # seg 0.9 keeps a union of structures as `members`; its values are its members'
    assert [(s.id, s.sorted_values, s.role) for s in back.segments] == [
        ("bg", [0], "background"), ("g_ab", [5, 6], None), ("c5", [5], None), ("c6", [6], None)]
    assert back.segments[1].members == ["c5", "c6"] and back.segments[1].label_values is None


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
    # a store records its task as it was named when written: a bare `total_fast` or `ts:` before
    # 0.11.0, `ts.v2:` since - the catalog refuses the first two, and the build still reads them
    assert build.names_for("nnunetv2", "ts:total_fast") == names
    assert build.names_for("nnunetv2", "ts.v2:total_fast") == names
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
            "frame": {"canonical": Geometry(shape_zyx=(40, 48, 56), spacing_zyx=(1.5, 1.5, 1.5),
                                            origin_xyz=(-10.0, -20.0, 5.0),
                                            direction_xyz=tuple(D)).to_record()},
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
        # the root went through duckn's model: the seg extension reads back validated -
        # the model's classes and a background segment, and nothing derived from them
        seg = rs.read_segmentation(sb.root)
        assert seg.version == "0.9"
        assert all(not s.name.startswith("label_") for s in seg.segments)
        by_id = {s.id: s for s in seg.segments}
        assert by_id["background_0"].role == "background"
        assert by_id["background_0"].label_values == [0]
        assert all(len(s.label_values) == 1 for s in seg.segments)      # no unions, no groups
        assert not [i for i in by_id if i.startswith(("classes_", "g_"))]


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
                     "frame": {"canonical": Geometry(shape_zyx=(16, 16, 16), spacing_zyx=(1.5,) * 3,
                                                     origin_xyz=(0.0, 0.0, 0.0),
                                                     direction_xyz=tuple(D)).to_record()},
                     "model_grid": [8, 8, 8], "envelope": {"start": [0, 0, 0], "stop": [8, 8, 8]},
                     "softmax": {"classes": 3, "weights": "synthetic", "version": "0"},
                     "haversack": "test"}
    (src / "meta.json").write_text(json.dumps(
        {"image": "synthetic.nii", "task": "liver_segments", "depth": 2, "clip": 8.0,
         "envelope_mm": None, "parts": parts}, default=str))
    return build, src


def test_leaves_are_unique_per_layer_and_value_so_a_cascade_keeps_every_class(tmp_path):
    """Both stages of a cascade emit channel indices 1..K-1. A dedupe on the value alone once
    gave stage 1 no segments at all."""
    build, src = _two_part_emit(tmp_path, None)
    out = build.build(src, tmp_path / "c.duckn", "c", names={1: "segment_1", 2: "segment_2"},
                      quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    by = {s.id: s for s in seg.segments}
    assert {"c1_l0", "c2_l0", "c1_l1", "c2_l1"} <= by.keys()
    assert by["c1_l1"].name == "segment_1" and by["c1_l1"].layer == 1
    assert by["c1_l0"].name == "label_1"                        # stage 0 has its own classes
    assert {s.id for s in seg.segments if s.layer == 1} == {"background_1", "c1_l1", "c2_l1"}
    assert {s.id for s in seg.segments if not s.layer} == {"background_0", "c1_l0", "c2_l0"}
    verify = _tool("ranked_verify")
    assert verify.verify(out, deep=True, quiet=True)


def test_a_single_part_store_states_no_layer(tmp_path):
    build = _tool("ranked_build_store")
    out = build.build(_synthetic_emit(tmp_path), tmp_path / "one.duckn", "one", quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    assert all(s.layer is None for s in seg.segments)


def test_a_build_writes_no_unions_even_over_the_five_lobes(tmp_path):
    """Until seg 0.8 the builder wrote `g_lungs` over TotalSegmentator's five lobes, claimed
    exhaustive. That is a fact about the labeling scheme, the same for every store the model
    produces, and it now lives outside the store."""
    build = _tool("ranked_build_store")
    assert not hasattr(build, "named_groups") and not hasattr(build, "GROUP_CLAIMS")
    assert "g_lungs" in build.GENERATED_GROUP_IDS          # still known, so old stores upgrade


def test_the_upgrade_tool_parses_arguments_and_names_a_store_that_is_not_one(tmp_path, capsys):
    up, build = _tool("ranked_upgrade_seg"), _tool("ranked_build_store")
    out = build.build(_synthetic_emit(tmp_path), tmp_path / "one.duckn.zip", "one", quiet=True)
    with pytest.raises(SystemExit):
        up.main(["--help"])
    with pytest.raises(SystemExit):
        up.main(["--no-such-flag", str(out)])
    up.main([str(out)])
    assert "seg 0.9 -> 0.9" in capsys.readouterr().out
    with rs.open_store(out) as st:
        assert all(s.layer is None for s in rs.read_segmentation(st.root).segments)
    bare = tmp_path / "bare.duckn"
    with rs.open_store(bare, "w") as st:
        st.write_text("README.md", "not a haversack store")
    with pytest.raises(SystemExit, match="no `haversack`"):
        up.main([str(bare)])


# ----------------------------------------------------------------------------------------
# the metadata upgrader and the claims it inherits
# ----------------------------------------------------------------------------------------

LOBES = ("lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
         "lung_middle_lobe_right", "lung_lower_lobe_right")


def _legacy_store(path, engine, extra=()):
    """A store as the builder wrote it under seg 0.7: five lung lobes, a generated
    `g_lungs` claimed exhaustive, and the part's partition group."""
    ids = [f"c{i}" for i in range(len(LOBES))]
    segs = [{"id": "bg", "name": "background", "label_value": 0, "background": True}]
    segs += [{"id": i, "name": n, "label_value": k + 1} for k, (i, n) in enumerate(zip(ids, LOBES))]
    segs.append({"id": "g_lungs", "name": "lungs", "members": ids,
                 "disjoint": True, "exhaustive": True})
    segs.append({"id": "classes_0", "name": "every class", "members": ["bg", *ids],
                 "disjoint": True, "exhaustive": True})
    segs += list(extra)
    with rs.open_store(path, "w") as st:
        st.root.attrs.update({"duckn": {"version": rs.DUCKN_VERSION, "extensions": {
            "seg": {"version": "0.7", "segments": segs},
            "haversack": {"engine": engine, "part_order": [{"name": "labels"}]},
            "provenance": {"version": "1.0", "processing": []}}}})


def _segments_after_upgrade(path):
    upgrader = _tool("ranked_upgrade_seg")
    upgrader.write_readme = lambda st: None            # the README is not what is under test
    upgrader.upgrade(Path(path))
    with rs.open_store(path, "r") as st:
        raw = st.root.attrs.asdict()["duckn"]["extensions"]["seg"]
        back = rs.read_segmentation(st.root)
    assert raw["version"] == "0.9"                     # rewritten, not merely readable
    return {s.id: s for s in back.segments}


@pytest.mark.parametrize("engine", ["monai", "nnunetv2"])
def test_the_upgrader_removes_every_group_the_builder_generated(tmp_path, engine):
    """Whatever engine wrote it: seg 0.8 stores carry the model's classes, and the unions
    the builder used to derive from them - `g_lungs`, the part's partition - are gone. The
    question is what the builder EVER generated, not what it generates for this engine."""
    store = tmp_path / "old.duckn.zip"
    _legacy_store(store, engine)
    after = _segments_after_upgrade(store)
    assert set(after) == {"bg", "c0", "c1", "c2", "c3", "c4"}
    assert after["bg"].role == "background"
    assert all(len(s.label_values) == 1 for s in after.values())


def test_the_upgrader_keeps_a_group_nothing_ever_generated(tmp_path):
    """A user-authored group is not the builder's to remove: it stays, as the union segment
    duckn's migration makes of it, listed before the classes it contains."""
    store = tmp_path / "mine.duckn.zip"
    _legacy_store(store, "monai", extra=[
        {"id": "my_own", "name": "what I care about", "members": ["c0", "c1"]}])
    after = _segments_after_upgrade(store)
    assert "my_own" in after and after["my_own"].sorted_values == [1, 2]
    assert list(after).index("my_own") < list(after).index("c0")
    assert "g_lungs" not in after and "classes_0" not in after


# ----------------------------------------------------------------------------------------
# the labeling scheme a store declares (duckn seg 0.8)
# ----------------------------------------------------------------------------------------

def test_the_fast_variants_declare_the_scheme_of_the_task_whose_classes_they_are():
    """`total_fast` is not a task upstream and has no class list of its own: it is `total`'s
    classes from a coarser model. One class list is one scheme, identified by a uri that
    carries the catalog's major version and the task, and not the release."""
    from haversack.ecosystems import TSEcosystem
    eco = TSEcosystem()
    fast, full = eco.labeling_scheme("total_fast"), eco.labeling_scheme("total")
    assert fast == full == eco.labeling_scheme("total_fastest")
    assert full["key"] == "ts.v2:total"
    assert full["system_uri"] == "https://github.com/wasserth/TotalSegmentator#v2:total"
    assert full["definition_url"].endswith(f"/tree/v{full['version']}") and "v" not in full["version"]
    assert eco.labeling_scheme("total_mr_fast")["key"] == "ts.v2:total_mr"
    assert eco.labeling_scheme("liver_segments")["key"] == "ts.v2:liver_segments"
    with pytest.raises(LookupError):
        eco.labeling_scheme("no_such_task")


def test_a_build_declares_its_scheme_and_codes_every_class_in_it(tmp_path, monkeypatch):
    build = _tool("ranked_build_store")
    import haversack.ranked_build as rb
    scheme = {"key": "ts.v2:total", "name": "n", "version": "2.13.0",
              "system_uri": "https://github.com/wasserth/TotalSegmentator#v2:total", "definition_url": "u"}
    monkeypatch.setattr(rb, "scheme_for", lambda engine, task: (scheme, lambda v, n: n))
    monkeypatch.setattr(rb, "names_for", lambda *a, **k: {1: "liver", 2: "spleen"})
    emit = _synthetic_emit(tmp_path)
    out = build.build(emit, tmp_path / "s.duckn", "s", quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    assert seg.labeling_scheme == "ts.v2:total"
    assert seg.terminologies["ts.v2:total"].system_uri == scheme["system_uri"]
    coded = {s.name: s.designations[0].code for s in seg.segments if s.designations}
    assert coded and all(name == code for name, code in coded.items())
    assert not [s for s in seg.segments if s.role and s.designations]   # the background has none

    # a caller's own names have left the scheme: none is declared
    (tmp_path / "again").mkdir()
    out2 = build.build(_synthetic_emit(tmp_path / "again"), tmp_path / "t.duckn", "t", quiet=True,
                       names={1: "mine", 2: "also mine"})
    with rs.open_store(out2) as st:
        assert rs.read_segmentation(st.root).labeling_scheme is None


def _fastsurfer_emit(tmp_path):
    """A FastSurfer-shaped emit: aparc+aseg ids as values, one of them a channel that
    `split_cortex_labels` lateralizes only after the network."""
    labels = [0, 17, 1003, 1012]          # background, Left-Hippocampus, a bilateral channel, an exact one
    torch.manual_seed(1)
    logits = torch.randn(len(labels), 12, 12, 12)
    code = encode(logits, depth=2, clip=8.0)
    src = tmp_path / "fs-emit"
    src.mkdir()
    for nm, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
        if arr is not None:
            np.save(src / f"asegdkt_{nm}.npy", arr)
    part = {**code.meta, "engine": "fastsurfer", "task": "fastsurfer:asegdkt", "part": "asegdkt",
            "labels": labels,
            "frame": {"canonical": Geometry(shape_zyx=(12, 12, 12), spacing_zyx=(1.0, 1.0, 1.0),
                                            origin_xyz=(0.0, 0.0, 0.0),
                                            direction_xyz=tuple(D)).to_record()},
            "model_grid": [12, 12, 12], "envelope": {"start": [0, 0, 0], "stop": [12, 12, 12]},
            "softmax": {"classes": len(labels), "weights": "fastsurfer", "version": "2.5.4"},
            "haversack": "test"}
    (src / "meta.json").write_text(json.dumps(
        {"image": "t1.nii", "task": "fastsurfer:asegdkt", "depth": 2, "clip": 8.0,
         "envelope_mm": None, "parts": {"asegdkt": part}}, default=str))
    return src


def test_a_fastsurfer_store_codes_by_id_and_leaves_a_bilateral_channel_uncoded(tmp_path):
    """A ranked store holds the network's channels BEFORE FastSurfer's spatial hemisphere
    split, so value 1003 is both caudal middle frontal cortices under a left-hemisphere
    name: not exactly the concept its id names, and a designation must be exact."""
    build = _tool("ranked_build_store")
    out = build.build(_fastsurfer_emit(tmp_path), tmp_path / "fs.duckn", "fs", quiet=True)
    with rs.open_store(out) as st:
        seg = rs.read_segmentation(st.root)
    assert seg.labeling_scheme == "fastsurfer:asegdkt"
    assert seg.terminologies["fastsurfer:asegdkt"].system_uri.endswith("#v2:asegdkt")
    by = {s.label_values[0]: s for s in seg.segments}
    d = by[17].designations[0]
    assert (d.scheme, d.code, d.meaning) == ("fastsurfer:asegdkt", "17", "Left-Hippocampus")
    assert by[1012].designations[0].code == "1012"
    assert by[1003].designations is None and by[1003].name.startswith("ctx-lh-")
    assert by[0].role == "background" and by[0].designations is None
    assert _tool("ranked_verify").verify(out, deep=True, quiet=True)


def test_names_the_run_reported_keep_the_scheme_and_a_callers_own_do_not(tmp_path, monkeypatch):
    """`segment_to_store` always hands names to the builder - the run's own - which used to
    switch the scheme off on the one path the product uses. `model_names` says whose they are."""
    build = _tool("ranked_build_store")
    emit = _synthetic_emit(tmp_path)
    names = build.names_for("nnunetv2", "total_fast")
    for flag, want in ((True, "ts.v2:total"), (False, None)):
        out = build.build(emit, tmp_path / f"m{flag}.duckn", "m", quiet=True, names=names,
                          model_names=flag)
        with rs.open_store(out) as st:
            assert rs.read_segmentation(st.root).labeling_scheme == want
