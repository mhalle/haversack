"""The store-backed restore against the pipeline's own restore, on the same run."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("rankfield")
if importlib.util.find_spec("duckn") is None:
    pytest.skip("duckn not installed", allow_module_level=True)

from test_normalization_sharing import ORGANS, RIBS, _StubModel, _two_part_task, _write_ct  # noqa: E402
from test_ranked_output import _ShapedStub  # noqa: E402

from haversack import ranked_restore as rr  # noqa: E402
from haversack.ranked_output import segment_to_store  # noqa: E402


@pytest.fixture(params=[None, 3.0], ids=["whole", "cropped"])
def run(tmp_path, monkeypatch, request):
    """One run of the two-part stub task at 1 mm, linear and nearest: its labels and its store,
    once on the whole grid and once cropped to an envelope (a frame WITH a crop is the
    composition the restore has to get right)."""
    from haversack import pipeline
    out = {}
    for interp in ("linear", "nearest"):
        d = tmp_path / interp
        d.mkdir()
        organs, ribs = _ShapedStub(ORGANS._props), _StubModel(RIBS._props)
        spec, store, cache = _two_part_task(d, [organs, ribs])     # the stubs are one-shot
        monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
        seg, path = segment_to_store(str(_write_ct(d)), spec, tmp_path / f"{interp}.duckn",
                                     case="rt", models=cache, device="cpu", envelope_mm=request.param,
                                     grid=1.0, interp=interp, convention="corner", folds=(0,),
                                     quiet=True)
        out[interp] = (seg, path)
    return out


def test_the_parts_carry_the_frame(run):
    from haversack import ranked_store as rs
    with rs.open_store(run["linear"][1]) as st:
        m = st.root["parts/0"].attrs["duckn"]["extensions"]["ranked"]
    assert "frame" in m and m["frame"]["convention"] == "corner"


@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_restoring_from_the_store_reproduces_the_runs_labels(run, interp):
    seg, path = run[interp]
    res = rr.restore(path, grid=1.0, interp=interp)
    assert res.frame is not None
    assert res.parts == ["first", "second"]
    img = res.image("input")
    a = np.asarray(__import__("SimpleITK").GetArrayFromImage(img))
    b = seg.array
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(img.GetOrigin(), seg.labels.GetOrigin(), atol=1e-6)
    assert np.allclose(img.GetDirection(), seg.labels.GetDirection(), atol=1e-9)
    mismatch = float((a != b).mean())
    # nearest: no interpolation, the argmax itself. Linear: the stub's field is clean enough
    # that no gap sits within a quantum of a tie, so this is exact too; a real case is not
    # (0.06 % on the torso, all within one quantum) - see the module header.
    assert mismatch == 0.0, mismatch


def test_a_roi_restore_equals_the_full_restore_on_that_box(run):
    _, path = run["linear"]
    full = rr.restore(path, grid=1.0)
    box = ((2, 9), (3, 12), (1, 14))
    part = rr.restore(path, grid=1.0, roi=box)
    sl = tuple(slice(a, b) for a, b in box)
    np.testing.assert_array_equal(part.labels, full.labels[sl])
    assert part.labels.shape == tuple(b - a for a, b in box)
    from haversack.values import Geometry          # rankfield 0.3 keeps one order; convert
    d = (np.asarray(Geometry.from_record(part.geometry).origin_xyz)
         - np.asarray(Geometry.from_record(full.geometry).origin_xyz))
    assert np.allclose(np.abs(d), np.asarray([box[2][0], box[1][0], box[0][0]]) * 1.0)


def test_a_structure_restores_inside_its_own_box(run):
    _, path = run["linear"]
    full = rr.restore(path, grid=1.0)
    present = [v for v in (1, 2) if (full.labels == v).any()]
    assert present, "the stub must put at least one structure on the grid"
    for value in present:
        box = rr.roi_of(path, [value], grid=1.0)
        assert any(b - a < n for (a, b), n in zip(box, full.labels.shape))    # a real sub-box
        part = rr.restore(path, grid=1.0, roi=box)
        sl = tuple(slice(a, b) for a, b in box)
        assert (part.labels == value).sum() == (full.labels == value).sum()   # nothing outside
        np.testing.assert_array_equal(part.labels, full.labels[sl])


def test_the_grid_can_be_named_without_the_frame(run):
    _, path = run["linear"]
    g, fr = rr.resolve_grid(path, 2.0)
    assert fr is not None and all(s == 2.0 for s in g.spacing)
    res = rr.restore(path, grid=2.0)
    assert res.labels.shape == g.shape


def test_the_metal_kernel_matches_the_torch_path_bit_for_bit(run):
    import torch
    from rankfield.backends import metal
    if not (torch.backends.mps.is_available() and metal.available()):
        pytest.skip("no MPS")
    _, path = run["linear"]
    cpu = rr.restore(path, grid=1.0, device="cpu")
    gpu = rr.restore(path, grid=1.0, device="mps")
    np.testing.assert_array_equal(gpu.labels, cpu.labels)
    box = ((2, 9), (3, 12), (1, 14))
    np.testing.assert_array_equal(rr.restore(path, grid=1.0, roi=box, device="mps").labels,
                                  cpu.labels[tuple(slice(a, b) for a, b in box)])
    fine = rr.restore(path, grid=0.5, device="cpu")
    np.testing.assert_array_equal(rr.restore(path, grid=0.5, device="mps").labels, fine.labels)




def test_the_store_is_read_as_library_parts(run):
    from haversack import ranked_store as rs
    _, path = run["linear"]
    with rs.open_store(path) as st:
        parts = rr.parts_of(st.root)
    assert [p.name for p in parts] == ["first", "second"]
    assert parts[0].field.frame and parts[0].field.labels and parts[0].field.geometry is not None
    # what the library calls current, never a literal: this line said "0.3" and went red
    # the day rankfield cut format 0.4, though nothing about the store was wrong
    import rankfield as rf
    assert parts[0].field.meta["version"] == rf.FORMAT_VERSION
    assert parts[0].field.meta["keep"] == "shell"


def test_a_frameless_store_takes_its_geometry_from_the_array(run, monkeypatch):
    _, path = run["linear"]
    from haversack import ranked_store as rs
    with rs.open_store(path) as st:
        from rankfield import store as rfstore
        geo = rfstore.array_geometry(st.root["parts/0/ranks"])
    real = rr.parts_of
    def frameless(root, **kw):
        out = real(root, **kw)
        for p in out:
            p.field.frame = None
        return out
    monkeypatch.setattr(rr, "parts_of", frameless)
    with pytest.raises(rr.InputError, match="no frame"):
        rr.restore(path, grid="input")
    res = rr.restore(path, grid=1.0, device="cpu")
    assert res.frame is None
    from haversack.values import Geometry          # rankfield 0.3 keeps one order; convert
    rg, sg = Geometry.from_record(res.geometry), Geometry.from_record(geo)
    assert np.allclose(rg.direction_xyz, sg.direction_xyz)
    D = np.asarray(sg.direction_xyz).reshape(3, 3)
    assert np.allclose(rg.origin_xyz, np.asarray(sg.origin_xyz) + D @ np.asarray(res.grid.origin)[::-1])


def test_the_command_writes_names_and_refuses_what_it_should(run, tmp_path):
    import SimpleITK as sitk
    _, path = run["linear"]
    out = tmp_path / "labels.seg.nrrd"
    assert rr.main_cli([str(path), "-o", str(out), "--quiet", "--device", "cpu"]) == 0
    img = sitk.ReadImage(str(out))
    assert [k for k in img.GetMetaDataKeys() if k.endswith("_Name")] and "haversack_provenance" in img.GetMetaDataKeys()
    with pytest.raises(rr.InputError, match="no such store"):
        rr.main_cli([str(tmp_path / "nope.duckn"), "-o", str(out)])
    (tmp_path / "notastore").mkdir()
    with pytest.raises(rr.InputError, match="not a ranked store"):
        rr.main_cli([str(tmp_path / "notastore"), "-o", str(out)])
    with pytest.raises(rr.InputError, match="labels take"):
        rr.main_cli([str(path), "-o", str(tmp_path / "x.duckn")])


def test_an_absurd_spacing_is_refused_before_anything_is_allocated(run):
    from haversack.errors import InputError
    _, path = run["linear"]
    with pytest.raises(InputError, match="2\\^31|coarser"):
        rr.restore(path, grid=0.0005, device="cpu")
    with pytest.raises(InputError, match="device"):
        rr.restore(path, grid=3.0, device="banana")


def test_no_module_restates_the_format_list_rankfield_owns():
    """rankfield owns the format version and the list of readable ones. Three places
    here used to restate one or the other, and when rankfield cut 0.4 haversack wrote
    stores its own reader refused - 25 failures across five files, none of them naming
    the cause. This asserts the three aliases still defer rather than restate.

    It does NOT catch a rankfield format bump (FORMAT_VERSION and KNOWN_VERSIONS move
    together, so the first two assertions hold by construction); the reader's behaviour
    against an unknown version is pinned by
    ``test_a_version_this_reader_does_not_know_is_refused_as_an_input_error``.
    """
    import rankfield as rf
    from rankfield.store import KNOWN_VERSIONS

    from haversack.ranked import RANKED_VERSION
    assert RANKED_VERSION == rf.FORMAT_VERSION          # the shim does not drift either
    assert RANKED_VERSION in KNOWN_VERSIONS, (
        f"haversack writes ranked format {RANKED_VERSION} but rankfield reads "
        f"{KNOWN_VERSIONS}")
    import importlib.util                               # the verifier agrees with the reader
    import pathlib
    tool = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ranked_verify.py"
    spec = importlib.util.spec_from_file_location("ranked_verify", tool)
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    assert rv.RANKED_VERSIONS == KNOWN_VERSIONS


def _meta_of(root, i=0):
    return dict(root[f"parts/{i}"].attrs.asdict()["duckn"]["extensions"]["ranked"])


def _rewrite_meta(root, i, meta):
    attrs = root[f"parts/{i}"].attrs.asdict()
    attrs["duckn"]["extensions"]["ranked"] = meta
    root[f"parts/{i}"].attrs.put(attrs)


def test_a_version_this_reader_does_not_know_is_refused_as_an_input_error(run):
    """The whole reason parts_of wraps rankfield's ValueError: haversack's callers
    catch InputError, and the CLI turns it into exit 2 rather than a traceback.
    Nothing exercised the wrapper - a store with an unreadable version was never built."""
    from haversack import ranked_store as rs
    _, path = run["linear"]
    with rs.open_store(path, "a") as st:
        m = _meta_of(st.root)
        m["version"] = "9.9"
        _rewrite_meta(st.root, 0, m)
        with pytest.raises(rr.InputError, match="9.9.*this reader knows"):
            rr.parts_of(st.root)


@pytest.mark.parametrize("bad", [None, 0, "1,11,0,10,0,13", {"lower": [0, 0, 0], "upper": [1, 1, 1]}],
                         ids=["none", "scalar", "string", "wrong-dict"])
def test_an_envelope_this_reader_cannot_place_is_refused_not_defaulted(run, bad):
    """rankfield's reader places a part with no usable envelope at the ORIGIN. This
    module raised on every such store before it delegated; afterwards `haversack
    restore` exited 0 on a legacy store and wrote misplaced labels. The oldest form
    carries `envelope_start_zyx` and no `envelope` at all, which tools/ranked_align_parts.py
    still reads - so these stores exist."""
    from haversack import ranked_store as rs
    _, path = run["linear"]
    with rs.open_store(path, "a") as st:
        m = _meta_of(st.root)
        if bad is None:
            m.pop("envelope", None)             # the legacy envelope_start_zyx shape
            m["envelope_start_zyx"] = [1, 0, 0]
        else:
            m["envelope"] = bad
        _rewrite_meta(st.root, 0, m)
        with pytest.raises(rr.InputError, match="not a form this reader can place"):
            rr.parts_of(st.root)


def test_the_envelope_forms_this_reader_does_place_still_work(run):
    """The half-open {"start": ...} dict is the form the delegation was FOR; it must
    still read, and place the part where it says."""
    from haversack import ranked_store as rs
    _, path = run["linear"]                 # the fixture is parameterized whole/cropped
    with rs.open_store(path, "a") as st:
        flat = _meta_of(st.root)["envelope"]
        want = tuple(int(v) for v in flat[0::2])
        assert rr.parts_of(st.root)[0].envelope_start == want
        m = _meta_of(st.root)
        m["envelope"] = {"start": list(want), "stop": [v + 1 for v in flat[1::2]]}
        _rewrite_meta(st.root, 0, m)
        assert rr.parts_of(st.root)[0].envelope_start == want
