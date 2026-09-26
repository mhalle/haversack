"""`haversack.inputs`: the public door to the standard form of input (2026-09-26)."""
import numpy as np
import pytest
import SimpleITK as sitk

from haversack import inputs, sources

from test_get_and_cache import FakeSource
from test_several_series import THREE, write_series


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(sources, "default_sources", lambda: [FakeSource()])
    from haversack import inputstore
    monkeypatch.setattr(inputstore, "_COMMAND", {})


def _ct(path, shape=(70, 9, 11)):
    arr = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.8, 1.5))
    img.SetOrigin((3.0, -2.0, 10.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return arr


@pytest.mark.parametrize("compression", ["zstd", "uncompressed"])
def test_whole_and_slab_reads_are_the_original_voxels(store, tmp_path, monkeypatch, compression):
    monkeypatch.setenv("HAVERSACK_INPUT_COPY_COMPRESSION", compression)
    want = _ct(tmp_path / "local" / f"ct-{compression}.nii.gz")
    x = inputs.open(tmp_path / "local" / f"ct-{compression}.nii.gz")
    assert x.is_copy and x.identity.startswith("sha256:")
    np.testing.assert_array_equal(x.array(), want)
    np.testing.assert_array_equal(x.array((33, 41)), want[33:41])     # across a 32-slice chunk edge
    np.testing.assert_array_equal(sitk.GetArrayFromImage(x.image()), want)
    assert x.image().GetOrigin() == pytest.approx((3.0, -2.0, 10.0))


def test_the_same_bytes_are_one_identity_and_the_record_names_the_file(store, tmp_path):
    _ct(tmp_path / "a" / "ct.nii.gz")
    import shutil
    (tmp_path / "b").mkdir()
    shutil.copyfile(tmp_path / "a" / "ct.nii.gz", tmp_path / "b" / "copy.nii.gz")
    a, b = inputs.open(tmp_path / "a" / "ct.nii.gz"), inputs.open(tmp_path / "b" / "copy.nii.gz")
    assert a.identity == b.identity and a.path == b.path
    assert a.record["content"]["digest"] == a.identity


def test_a_dicom_series_gives_its_tags_as_json(store, tmp_path):
    x = inputs.open(write_series(tmp_path / "dcm", 3, THREE, value=2))
    t = x.tags()
    assert t["series"]["Modality"] == "CT"
    assert [s["InstanceNumber"] for s in t["slices"]] == [1, 2, 3]     # per slice, in slice order


def test_a_hosted_input_and_a_held_digest(store, tmp_path):
    x = inputs.open("fake:case1")
    assert x.identity == "fake:case1" and x.array().shape == (8, 8, 8)
    local = inputs.open(_ct_path(tmp_path))
    again = inputs.open(local.identity)
    assert again.path == local.path and again.identity == local.identity
    from haversack.inputstore import InputGone
    with pytest.raises(InputGone):
        inputs.open("sha256:" + "0" * 64)


def _ct_path(tmp_path):
    p = tmp_path / "held" / "ct.nii.gz"
    _ct(p)
    return p
