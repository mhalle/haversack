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
    d = np.asarray(part.geometry.origin_xyz) - np.asarray(full.geometry.origin_xyz)
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
    assert parts[0].field.meta["version"] == "0.3" and parts[0].field.meta["keep"] == "shell"


def test_a_frameless_store_takes_its_geometry_from_the_array(run, monkeypatch):
    _, path = run["linear"]
    from haversack import ranked_store as rs
    with rs.open_store(path) as st:
        geo = rr._array_geometry(st.root["parts/0/ranks"])
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
    assert np.allclose(res.geometry.direction_xyz, geo.direction_xyz)
    D = np.asarray(geo.direction_xyz).reshape(3, 3)
    assert np.allclose(res.geometry.origin_xyz, np.asarray(geo.origin_xyz) + D @ np.asarray(res.grid.origin)[::-1])


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
