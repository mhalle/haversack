"""A ranked store straight out of segment(): the undocumented `.duckn` / `.duckn.zip` output.

Driven with the stub models of test_normalization_sharing - no weights, no network - so it
stays in the fast suite. What is under test is the seam: emit into a staging directory, build
the store from it, clean up, and hand back the Segmentation the run produced.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SimpleITK")
pytest.importorskip("zarr")
pytest.importorskip("duckn")

from test_normalization_sharing import ORGANS, RIBS, _StubModel, _two_part_task, _write_ct  # noqa: E402

from haversack import ranked_store as rs                       # noqa: E402
from haversack.ranked_output import is_store_output, segment_to_store   # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"


class _ShapedStub(_StubModel):
    """Puts class 1 in a block so the store has a structure with an extent and a surface."""

    def predict_logits(self, crop, report=None):
        logits = super().predict_logits(crop, report)
        z, y, x = logits.shape[1:]
        logits[1, z // 4: z // 2, y // 4: 3 * y // 4, x // 4: 3 * x // 4] = 3.0
        return logits


def _verify(path):
    spec = importlib.util.spec_from_file_location("ranked_verify", TOOLS / "ranked_verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify(Path(path), deep=True, quiet=True)


def test_store_output_is_detected_by_suffix_only():
    assert is_store_output("x.duckn") and is_store_output("x.duckn.zip")
    assert not is_store_output("x.seg.nrrd") and not is_store_output("x.zip")


@pytest.mark.parametrize("suffix", [".duckn.zip", ".duckn"])
def test_segment_to_store_builds_a_verified_store_and_cleans_up(tmp_path, monkeypatch, suffix):
    from haversack import pipeline
    organs, ribs = _ShapedStub(ORGANS._props), _StubModel(RIBS._props)
    spec, store, cache = _two_part_task(tmp_path, [organs, ribs])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    out = tmp_path / f"case{suffix}"

    seg, where = segment_to_store(str(_write_ct(tmp_path)), spec, out, case="stub",
                                  models=cache, device="cpu", envelope_mm=None,
                                  convention="corner", folds=(0,), quiet=True)
    assert where == out and out.exists()
    assert seg.grid.shape and "a" in seg.schema.names.values()
    assert not [p for p in tmp_path.iterdir() if ".emit-" in p.name], "staging left behind"
    assert _verify(out)

    with rs.open_store(out) as st:
        segs = rs.read_segmentation(st.root)
        by = {s.id: s for s in segs.segments}
        assert by["c1"].name == "a" and by["c2"].name == "b"     # the task's own label map
        assert by["c1"].extent is not None                        # class 1 has voxels
        assert by["classes_0"].exhaustive and by["classes_1"].disjoint
        assert sorted(st.root["parts"].group_keys()) == ["0", "1"]
        assert "distance" in st.root["parts/0"]
        rk = np.asarray(st.root["parts/0/ranks"][0])
        assert set(np.unique(rk)) == {1, 2}                       # background and class 1


def test_the_store_output_may_not_be_the_input(tmp_path):
    from haversack.errors import InputError
    src = tmp_path / "scan.duckn"
    src.mkdir()
    with pytest.raises(InputError, match="is the input"):
        segment_to_store(str(src), "total_fast", src)


def test_the_stores_argmax_is_the_runs_labels(tmp_path, monkeypatch):
    """The property the store is sold on: what a reader decodes is what the run wrote. On the
    model grid the restore is the identity, so ranks[0] through each part's label table,
    composited in paint order, must equal the labels voxel for voxel."""
    from haversack import pipeline
    organs, ribs = _ShapedStub(ORGANS._props), _StubModel(RIBS._props)
    spec, store, cache = _two_part_task(tmp_path, [organs, ribs])
    monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
    out = tmp_path / "rt.duckn"
    seg, _ = segment_to_store(str(_write_ct(tmp_path)), spec, out, case="rt",
                              models=cache, device="cpu", envelope_mm=None, grid="input",
                              convention="corner", folds=(0,), quiet=True)
    with rs.open_store(out) as st:
        order = st.root.attrs["duckn"]["extensions"]["haversack"]["part_order"]
        painted = None
        for p in order:
            g = st.root[f"parts/{p['index']}"]
            lut = np.asarray(g.attrs["duckn"]["extensions"]["ranked"]["labels"])
            win = np.asarray(g["ranks"][0]).astype(np.int64) - 1
            labels = lut[win]
            painted = labels if painted is None else np.where(labels > 0, labels, painted)
        store_dirs = [ax["space_direction"] for ax in g["ranks"].attrs["duckn"]["axes"]
                      if ax.get("space_direction")]                # store axes z,y,x in LPS
    # The store is on the model grid in canonical orientation; the labels are in the
    # input's. Align through the geometries - the world direction of each store axis
    # against the world direction of each label-array axis - never by assumed order.
    D = np.asarray(seg.labels.GetDirection()).reshape(3, 3)       # SimpleITK: LPS, columns x,y,z
    label_dirs = [D[:, 2 - k] for k in range(3)]                  # array axes z,y,x
    perm, flips = [], []
    for sd in store_dirs:
        dots = [float(np.dot(sd, ld)) / (np.linalg.norm(sd) * np.linalg.norm(ld)) for ld in label_dirs]
        k = int(np.argmax(np.abs(dots)))
        assert abs(dots[k]) > 0.999, "the stub grids are axis-aligned"
        perm.append(k)
        flips.append(dots[k] < 0)
    assert sorted(perm) == [0, 1, 2]
    aligned = np.transpose(seg.array, perm)                      # label axes into store order
    for axis, f in enumerate(flips):
        if f:
            aligned = np.flip(aligned, axis=axis)
    assert painted.shape == aligned.shape, (painted.shape, aligned.shape)
    assert (painted == aligned).all()
