"""A store whose model grid was resampled in WORLD space restores onto its input (2026-09-25).

FastSurfer's field lives on its conformed 1 mm grid, rotated from an oblique input; a frame
cannot describe that, so its emit records the input grid (``target_grid``) and the conformed
grid is the part's own array geometry. ``haversack restore`` refused "input" for these stores
and could only restore onto the conformed grid. Now "input" is that recorded geometry, through
``rankfield``'s affine map. On the real ds000114 T1 store the result matches the served labels
(99.9965 % of voxels, every Dice >= 0.998) - that check needs a Modal run and lives in the
AGENTS.md record; these hold the adapter to it on synthetic stores: an OFF-CENTER ball, so a
wrong rotation, mirror or axis order cannot pass.
"""
from __future__ import annotations

import numpy as np
import pytest

rf = pytest.importorskip("rankfield")
torch = pytest.importorskip("torch")
pytest.importorskip("zarr")

from haversack import ranked_restore as R  # noqa: E402
from haversack.errors import InputError  # noqa: E402

CENTER = np.array([12.0, 6.0, 1.5])            # world LPS x, y, z


def _target(shape=(50, 70, 70), spacing=0.8, origin=(-24.0, -26.0, -22.0)):
    """An oblique input grid (tilted 8 degrees about x), in duckn's grid form."""
    t = np.deg2rad(8.0)
    rows = np.array([(0.0, np.sin(t), np.cos(t)), (0.0, np.cos(t), -np.sin(t)), (1.0, 0.0, 0.0)]) * spacing
    return {"space": "left-posterior-superior", "space_origin": list(origin),
            "axes": [{"kind": "space", "centering": "cell", "space_direction": list(r), "unit": "mm"}
                     for r in rows],
            "samples": list(shape)}


def _store(tmp_path, *, task="fastsurfer:asegdkt", target=True, frame=None):
    th = np.deg2rad(30.0)
    rows = ((0.0, 0.0, 1.0), (-np.sin(th), np.cos(th), 0.0), (np.cos(th), np.sin(th), 0.0))
    geo = rf.Geometry(shape=(40, 48, 48), directions=rows, origin=(-5.0, -30.0, -20.0))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in geo.shape], indexing="ij"), -1)
    d = np.linalg.norm(geo.world(idx) - CENTER, axis=-1)
    field = rf.encode(torch.from_numpy(np.stack([np.zeros_like(d), 9.0 - d]).astype(np.float32)), depth=2)
    field.labels = [0, 17]
    field.geometry = geo
    field.meta = {**field.meta, "labels_named_by": task,
                  **({"target_grid": _target()} if target else {})}
    if frame:
        field.frame = frame
    out = tmp_path / "s.duckn.zip"
    from rankfield.store import write_parts
    write_parts(out, [rf.Part(field=field, name="fastsurfer")])
    return out


def test_input_is_the_recorded_target_grid(tmp_path):
    r = R.restore(_store(tmp_path), device="cpu")
    t = _target()
    assert r.labels.shape == tuple(t["samples"])
    np.testing.assert_allclose(r.geometry.origin, t["space_origin"])
    np.testing.assert_allclose(r.geometry.directions, [a["space_direction"] for a in t["axes"]], atol=1e-12)
    ball = np.argwhere(r.labels == 17)
    assert np.abs(r.geometry.world(ball.mean(0)) - CENTER).max() < 0.25
    # a synthetic store has no white matter for FastSurfer's rule to anchor on: said, and
    # never an overflow of a narrow label array (the rule compares against 1003..2035)
    assert all("Overflow" not in n for n in r.notes), r.notes
    img = r.image("input")                          # written exactly on the input's geometry
    np.testing.assert_allclose(img.GetOrigin(), t["space_origin"])


def test_the_command_restores_onto_the_input(tmp_path):
    import SimpleITK as sitk
    out = tmp_path / "labels.seg.nrrd"
    assert R.main_cli([str(_store(tmp_path)), "-o", str(out), "--quiet", "--device", "cpu"]) == 0
    img = sitk.ReadImage(str(out))
    assert img.GetSize() == (70, 70, 50)
    arr = sitk.GetArrayViewFromImage(img)
    got = np.array(img.TransformContinuousIndexToPhysicalPoint(
        tuple(float(v) for v in np.argwhere(arr == 17).mean(0)[::-1])))
    assert np.abs(got - CENTER).max() < 0.25


def test_a_spacing_still_restores_on_the_stored_grid(tmp_path):
    r = R.restore(_store(tmp_path), grid=2.0, device="cpu")
    assert r.labels.shape == (20, 24, 24)


def test_a_store_without_a_target_still_needs_a_grid(tmp_path):
    with pytest.raises(InputError, match="no frame"):
        R.restore(_store(tmp_path, target=False), device="cpu")


def test_a_target_in_another_space_is_refused(tmp_path, monkeypatch):
    p = R.parts_of(R._open(_store(tmp_path))[1])[0]
    p.field.meta["target_grid"] = {**_target(), "space": "right-anterior-superior"}
    with pytest.raises(InputError, match="right-anterior-superior"):
        R.input_geometry(p)


def test_roi_of_finds_the_ball_on_the_input_grid(tmp_path, monkeypatch):
    s = _store(tmp_path)
    import zarr  # noqa: F401 - the store's segments block is the builder's; give roi_of one
    whole = R.restore(s, device="cpu").labels
    ball = np.argwhere(whole == 17)
    parts = R.parts_of(R._open(s)[1])
    grid = R.input_geometry(parts[0])
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in parts[0].field.geometry.shape], indexing="ij"), -1)
    inside = np.argwhere(np.linalg.norm(parts[0].field.geometry.world(idx) - CENTER, axis=-1) < 9.0)
    ext = [int(v) for pair in zip(inside.min(0), inside.max(0)) for v in pair]
    box = rf.roi_of(parts, [(0, ext)], rf.Grid(grid.shape, spacing=grid.spacing), world=grid)
    for ax, (a, b) in enumerate(box):
        assert a <= ball[:, ax].min() and ball[:, ax].max() < b


def test_fastsurfers_rule_lateralizes_the_cortex(monkeypatch, tmp_path):
    """The engine rule runs for a FastSurfer store, and only for one; without FastSurfer
    installed the restore says it did not lateralize, never silently."""
    import sys
    import types
    seen = []
    fake = types.ModuleType("FastSurferCNN.data_loader.data_utils")
    fake.split_cortex_labels = lambda a: (seen.append(a.shape), a + 1)[1]
    monkeypatch.setitem(sys.modules, "FastSurferCNN.data_loader.data_utils", fake)
    r = R.restore(_store(tmp_path), device="cpu")
    assert seen == [r.labels.shape] and r.notes == () and r.labels.max() == 18
    seen.clear()
    r = R.restore(_store(tmp_path / "ts", task="ts.v2:total_fast"), device="cpu")
    assert seen == [] and r.labels.max() == 17
    monkeypatch.setitem(sys.modules, "FastSurferCNN.data_loader.data_utils", None)
    r = R.restore(_store(tmp_path / "none"), device="cpu")
    assert r.labels.max() == 17 and "not lateralized" in r.notes[0]
