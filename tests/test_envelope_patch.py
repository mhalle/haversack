"""The network never gets an envelope crop narrower than its patch, unless the grid is.

nnU-Net pads a crop narrower than a patch with 0 after normalization - the model's mean
foreground intensity, tissue where the image has air. On CADS that sat a margin's width outside
the skin of every head (2026-09-11). The pipeline now grows such a crop with real voxels
(`envelope.at_least`); these drive the real `segment()` with stub models and check what the
network was handed, including a model whose plans permute the axes, since the patch is stated
in the network's axis order and the crop is cut on the model grid.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("nnunetv2")
pytest.importorskip("SimpleITK")
from test_normalization_sharing import (ORGANS, _StubModel, _two_part_task,  # noqa: E402
                                        _write_ct)

from haversack import pipeline  # noqa: E402

GRID = (12, 14, 16)                                  # _write_ct's grid; the body is a block in it


@pytest.fixture
def run(tmp_path, monkeypatch):
    """segment() on the stub CT with a given patch / axis order; the network's input back."""
    n = itertools.count()

    def go(*, patch=(1, 1, 1), transpose=(0, 1, 2), envelope_mm=1.5, tiles=None):
        first, second = (_StubModel(ORGANS._props, patch=patch) for _ in range(2))
        for m in (first, second):
            m.transpose_forward = transpose
            if tiles is not None:
                m.tiles = tiles
        d = tmp_path / f"r{next(n)}"
        d.mkdir()
        spec, store, cache = _two_part_task(d, [first, second])
        monkeypatch.setattr(pipeline, "as_store", lambda *a, **k: store)
        pipeline.segment(str(_write_ct(d, GRID)), spec, models=cache, device="cpu",
                         envelope_mm=envelope_mm, convention="corner", folds=(0,))
        return first.received[0].numpy()
    return go


def _body_box(run):
    box = run().shape                                # a 1-voxel patch never grows anything
    assert all(b < g for b, g in zip(box, GRID)), f"the fixture must crop every axis: {box}"
    return box


def _is_a_window_of(crop, whole):
    """Whether ``crop`` is ``whole`` cut at some offset - real image, no invented voxel."""
    ranges = [range(w - c + 1) for c, w in zip(crop.shape, whole.shape)]
    return any(np.array_equal(whole[z:z + crop.shape[0], y:y + crop.shape[1], x:x + crop.shape[2]],
                              crop) for z, y, x in itertools.product(*ranges))


def test_a_crop_narrower_than_the_patch_is_grown_to_it_from_the_image(run):
    box = _body_box(run)
    want_y = box[1] + 2
    got = run(patch=(1, want_y, 1))
    assert got.shape == (box[0], want_y, box[2])
    assert _is_a_window_of(got, run(envelope_mm=None)), "the grown rows are not the image"


def test_the_patch_is_mapped_through_the_models_axis_order(run):
    """transpose_forward (2, 0, 1): network axis j is model axis (2, 0, 1)[j], so a patch of
    (1, 1, n) in the network's order asks for n along model axis 1 (y)."""
    box = _body_box(run)
    got = run(patch=(1, 1, box[1] + 2), transpose=(2, 0, 1))
    assert got.shape == (box[0], box[1] + 2, box[2])


def test_a_2d_models_patch_covers_the_last_two_network_axes(run):
    """A 2D configuration states two patch entries; nnU-Net runs every slice of the first
    network axis through them, so that axis is never grown. With an envelope asked for, the
    pipeline indexed the patch once per model axis and raised IndexError (2026-09-11)."""
    box = _body_box(run)
    got = run(patch=(box[1] + 2, 1))
    assert got.shape == (box[0], box[1] + 2, box[2])


def test_a_grid_narrower_than_the_patch_is_spanned_not_exceeded(run):
    box = _body_box(run)
    got = run(patch=(1, 99, 1))
    assert got.shape == (box[0], GRID[1], box[2])


def test_the_whole_volume_is_not_grown_or_cropped(run):
    assert run(patch=(99, 99, 99), envelope_mm=None).shape == GRID


# -- a crop has to save network work, counted in tiles --------------------------------------
#
# A crop re-tiles the window and labels move with the tiles, so it is worth it only if the
# network runs fewer patches. Until 2026-09-11 the pipeline judged by the box's volume, and on
# a chest-abdomen-pelvis CT the 10-40 mm crops ran every one of the whole volume's 54 tiles.

def test_a_crop_that_saves_no_tiles_runs_the_whole_grid(run):
    _body_box(run)
    assert run(tiles=lambda extent: 54).shape == GRID


def test_a_crop_that_saves_tiles_is_kept(run):
    box = _body_box(run)
    assert run(tiles=lambda extent: 54 if tuple(extent) == GRID else 36).shape == box


def _counting_model(patch, transpose):
    """A TorchModel with no weights: the real predict_logits and sliding window, nnU-Net's own
    slicers, and a pointwise stand-in network that counts its calls."""
    from functools import partial
    from types import SimpleNamespace

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from haversack import network as N
    calls = []

    def net(x):
        calls.append(tuple(x.shape[2:]))
        return x.expand(-1, 2, *(-1 for _ in x.shape[2:])).clone()     # 2D or 3D patches

    m = N.TorchModel.__new__(N.TorchModel)
    m.device, m.dtype, m.K, m.patch = torch.device("cpu"), torch.float32, 2, tuple(patch)
    m.transpose_forward = tuple(transpose)
    m.transpose_backward = N.inverse_perm(m.transpose_forward)
    m.accumulate, m.activation_reserve_gb, m.batch_size = "host", N.DEFAULT_ACTIVATION_RESERVE_GB, 1
    m.accumulate_choice = m.batch_choice = None
    m.fold_params, m._load_fold = [None], lambda i: None
    cfg = SimpleNamespace(configuration_manager=SimpleNamespace(patch_size=list(patch)),
                          tile_step_size=0.5, verbose=False)
    m.predictor = SimpleNamespace(
        tile_step_size=0.5,
        _internal_get_sliding_window_slicers=partial(nnUNetPredictor._internal_get_sliding_window_slicers, cfg))
    m.net, m._gaussian_cpu = net, torch.ones(patch)
    m.gaussian, m._on_device = m._gaussian_cpu, True
    return m, calls


@pytest.mark.parametrize("patch", [(8, 6, 4), (6, 4)])
@pytest.mark.parametrize("transpose", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("extent", [(5, 5, 5), (8, 6, 4), (4, 8, 6), (20, 13, 9), (9, 30, 17)])
def test_the_tile_count_is_what_the_sliding_window_runs(patch, transpose, extent):
    """`tiles` is a count of what predict_logits does, so it is checked against a run of it:
    the patch in the network's axis order, padding up to it, and nnU-Net's step rule - for a
    2D patch, every slice of the first network axis through a 2D window."""
    import torch
    m, calls = _counting_model(patch, transpose)
    m.predict_logits(torch.zeros((1, *extent)))
    assert len(calls) == m.tiles(extent), (len(calls), m.tiles(extent))
