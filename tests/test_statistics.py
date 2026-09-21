"""Round 4 differential + property tests for the statistics numerics (the
module had no direct test before)."""
import json

import numpy as np
import pytest

SimpleITK = pytest.importorskip("SimpleITK")
import SimpleITK as sitk

from haversack.statistics import compute_statistics, statistics_tsv


def _pair(gray, lab, names, tmp_path, spacing=(1.0, 1.0, 1.0)):
    gi = sitk.GetImageFromArray(gray)
    gi.SetSpacing(spacing)
    li = sitk.GetImageFromArray(lab.astype(np.uint16))
    li.SetSpacing(spacing)
    tmp_path = __import__("pathlib").Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    lp = tmp_path / "labels.seg.nrrd"
    for v, nm in names.items():
        li.SetMetaData(f"Segment{v}_Name", nm)
        li.SetMetaData(f"Segment{v}_LabelValue", str(v))
    sitk.WriteImage(li, str(lp))
    return gi, lp


def test_histogram_and_sort_paths_agree(tmp_path):
    """The int (histogram) and float (sort) paths must report identical
    percentiles - not 'within 1 unit'. numpy's linear interpolation invented
    values that disagreed by thousands of HU on multimodal structures."""
    rng = np.random.default_rng(20260825)
    worst = 0.0
    for case in range(120):
        shape = tuple(int(x) for x in rng.integers(12, 30, size=3))
        lab = np.zeros(shape, dtype=np.int64)
        K = int(rng.integers(2, 6))
        for v in range(1, K):
            mask = rng.random(shape) < 0.15
            lab[mask] = v
        # a distribution with heavy ties + negatives + bimodal structures
        gi = rng.integers(-1000, 1301, size=shape)
        if case % 3 == 0:                  # force a bimodal label
            gi = np.where(rng.random(shape) < 0.5, -1000, 1300)
        names = {v: f"s{v}" for v in range(1, K)}
        gint, lp = _pair(gi.astype(np.int16), lab, names, tmp_path / f"i{case}")
        gflt, _ = _pair(gi.astype(np.float32), lab, names, tmp_path / f"f{case}")
        ri = compute_statistics(gint, str(lp), tmp_path / f"ri{case}.json")
        rf = compute_statistics(gflt,
                                str((tmp_path / f"f{case}") / "labels.seg.nrrd"),
                                tmp_path / f"rf{case}.json")
        if ri is None and rf is None:
            continue
        a = {s["label"]: s for s in json.loads(ri.read_text(encoding="utf-8"))["structures"]}
        b = {s["label"]: s for s in json.loads(rf.read_text(encoding="utf-8"))["structures"]}
        assert a.keys() == b.keys()
        for v in a:
            for key in ("median", "p10", "p90", "min", "max"):
                d = abs(a[v][key] - b[v][key])
                worst = max(worst, d)
                assert d < 1e-6, (case, v, key, a[v][key], b[v][key])
    assert worst < 1e-6


def test_all_background_symmetric(tmp_path):
    """Empty foreground yields the same artifact (valid, empty structures)
    regardless of gray dtype - not None for int and JSON for float."""
    lab = np.zeros((8, 8, 8), dtype=np.int64)
    for dt in (np.int16, np.float32):
        gi, lp = _pair((np.zeros((8, 8, 8)) + 5).astype(dt), lab, {},
                       tmp_path / str(dt))
        out = compute_statistics(gi, str(lp), tmp_path / f"o{dt}.json")
        assert out is not None
        assert json.loads(out.read_text(encoding="utf-8"))["structures"] == []


def test_negative_labels_do_not_crash(tmp_path):
    """A signed labelmap with negative values must not crash bincount; the
    negatives are simply not structures."""
    lab = np.zeros((8, 8, 8), dtype=np.int64)
    lab[0, 0, 0] = -3
    lab[1, 1, 1] = 2
    gi, lp = _pair((np.zeros((8, 8, 8), dtype=np.int16) + 40), lab, {2: "kidney"},
                   tmp_path)
    out = compute_statistics(gi, str(lp), tmp_path / "o.json")
    labels = [s["label"] for s in json.loads(out.read_text(encoding="utf-8"))["structures"]]
    assert labels == [2]


def test_statistics_tsv_sanitizes_names():
    """A structure name with a tab/newline must not shift columns or split
    the row (moose/custom names are third-party dataset.json data)."""
    stats = {"units": {"intensity": "hu"}, "structures": [
        {"structure": "liver\twith\ttabs", "label": 1, "voxels": 3,
         "volume_ml": 1.0, "mean": 1, "std": 0, "min": 1, "max": 1,
         "median": 1, "p10": 1, "p90": 1, "centroid_ras_mm": [1, 2, 3]},
        {"structure": "spleen\nnewline", "label": 2, "voxels": 3,
         "volume_ml": 1.0, "mean": 1, "std": 0, "min": 1, "max": 1,
         "median": 1, "p10": 1, "p90": 1, "centroid_ras_mm": [4]},  # short centroid
    ]}
    tsv = statistics_tsv(stats)
    lines = tsv.split("\n")
    assert lines[-1] == "" and not lines[-2].endswith("\r")
    lines = [ln for ln in lines if ln]     # drop the trailing newline only
    assert len(lines) == 3                 # header + 2 rows, no split
    ncol = len(lines[0].split("\t"))
    assert all(len(ln.split("\t")) == ncol for ln in lines), [ln.count("\t") for ln in lines]


# -- the field measurement, reported alongside the counted one -------------------

def _phantom_case(tmp_path, radius=12.0, spacing=1.0):
    """One sphere, as both a labelmap (what statistics reads) and a ranked code (what
    the field path reads), on the SAME grid so the two are comparable here - which the
    server's are not, and which is why the output carries two spacings."""
    from haversack import phantoms as ph, ranked
    from haversack.grid import Grid
    n = 40
    g = Grid(shape=(n,) * 3, spacing=(spacing,) * 3, origin=(-(n - 1) * spacing / 2,) * 3)
    body = ph.sphere(radius)
    p = ph.Phantom((body,))
    lab = ph.labels(p, g).astype(np.uint16)
    gray = (lab * 100).astype(np.int16)
    gi, lp = _pair(gray, lab, {1: "sphere"}, tmp_path, spacing=(spacing,) * 3)
    code = ranked.encode(ph.logits(p, g), depth=2)
    code.meta.update(labels=[0, 1], spacing_zyx=[spacing] * 3)
    return gi, lp, code, body


def test_without_a_ranked_code_the_output_is_unchanged(tmp_path):
    """The compatibility contract: every existing number and key survives, and no field
    column appears. This is what makes it safe to ship the two side by side."""
    gi, lp, code, _ = _phantom_case(tmp_path)
    plain = json.loads(compute_statistics(gi, lp, tmp_path / "a.json").read_text(encoding="utf-8"))
    assert plain["structures"], "the fixture produced no structures"
    assert "field_grid_spacing_mm" not in plain
    for row in plain["structures"]:
        assert "volume_ml_field" not in row and "area_cm2_field" not in row
    assert "volume_ml_field" not in statistics_tsv(plain).splitlines()[0]


def test_the_field_columns_appear_and_beat_counting_on_a_known_sphere(tmp_path):
    """Truth is closed form here, so this is not 'the two agree' - it is which one is
    right. Counting has no area at all; the one the field reports is within a percent
    of 4 pi r^2, where the labelmap's own face count would be about half again too big."""
    gi, lp, code, body = _phantom_case(tmp_path)
    out = json.loads(compute_statistics(gi, lp, tmp_path / "b.json",
                                        ranked_code=code).read_text(encoding="utf-8"))
    row = next(r for r in out["structures"] if r["label"] == 1)
    assert out["field_grid_spacing_mm"] == [1.0, 1.0, 1.0]
    assert out["units"]["area"] == "cm2"

    counted, field = row["volume_ml"] * 1000.0, row["volume_ml_field"] * 1000.0
    assert abs(field / body.volume_mm3 - 1) < 0.01
    assert abs(field / body.volume_mm3 - 1) < abs(counted / body.volume_mm3 - 1)
    assert abs(row["area_cm2_field"] * 100.0 / body.area_mm2 - 1) < 0.02

    header, first = statistics_tsv(out).splitlines()[:2]
    assert header.split("\t")[-2:] == ["volume_ml_field", "area_cm2_field"]
    assert len(first.split("\t")) == len(header.split("\t"))


def test_a_code_that_cannot_be_read_leaves_the_counted_numbers_alone(tmp_path):
    """Best-effort by contract. A code with no channel->label map, or one that raises,
    must cost the caller nothing - statistics never fails a job, and a half-written
    artifact would be worse than an absent column."""
    gi, lp, code, _ = _phantom_case(tmp_path)
    ref = json.loads(compute_statistics(gi, lp, tmp_path / "c.json").read_text(encoding="utf-8"))
    for broken in (object(), _no_labels(code), _raising(code)):
        out = json.loads(compute_statistics(gi, lp, tmp_path / "d.json",
                                            ranked_code=broken).read_text(encoding="utf-8"))
        assert out["structures"] == ref["structures"]
        assert "field_grid_spacing_mm" not in out


def _no_labels(code):
    import copy
    c = copy.copy(code)
    c.meta = {k: v for k, v in code.meta.items() if k != "labels"}
    return c


def _raising(code):
    import copy
    c = copy.copy(code)
    c.meta = dict(code.meta, spacing_zyx=[0.0, 0.0, 0.0])   # measure() rejects this
    return c


def test_grid_spacing_is_ras_xyz_not_the_stored_axis_order(tmp_path):
    """`grid_spacing_mm` is (x, y, z) in RAS, because it is measured after the
    RAS reorientation - and the JSON says so in its `units`. Written 2026-09-19:
    a reader took it for the stored index order and read a coronal series'
    3 mm step as a rectangular pixel. The volumes pin the geometry beside it,
    so a mutant that reorders the spacing alone is still caught."""
    lab = np.zeros((4, 5, 6), np.uint16)      # (z, y, x) as SimpleITK hands arrays over
    lab[1:3, 1:4, 1:5] = 1
    gray = np.full(lab.shape, -1000, np.int16)
    # a coronal-like stored order: the 3 mm step sits on the SECOND stored axis
    gi, lp = _pair(gray, lab, {1: "s1"}, tmp_path / "cor", spacing=(0.8, 3.0, 0.5))
    li = sitk.ReadImage(str(lp))
    for v in (gi, li):                        # stored L, S, A -> RAS reorders to x, y, z
        v.SetDirection((-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0))
    sitk.WriteImage(li, str(lp))
    out = json.loads(compute_statistics(gi, str(lp), tmp_path / "s.json").read_text())
    assert out["units"]["grid_spacing_mm"] == "mm (x, y, z in RAS)"
    # stored (0.8, 3.0, 0.5) on (L, S, A) is (0.8, 0.5, 3.0) on (x, y, z) = (R, A, S)
    assert out["grid_spacing_mm"] == [0.8, 0.5, 3.0]
    assert out["structures"][0]["volume_ml"] == pytest.approx(2 * 3 * 4 * 0.8 * 3.0 * 0.5 / 1000.0,
                                                              abs=5e-4)   # the JSON rounds to 3 decimals
