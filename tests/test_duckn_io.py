"""duckn volumes as input: a zarr directory or zarr zip with duckn attributes reads into the
same SimpleITK image every other input becomes - geometry through duckn's own adapter, value
transforms applied. INTERNAL, undocumented (see haversack.duckn_io)."""
from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
duckn = pytest.importorskip("duckn")
from duckn.io import write as duckn_write            # noqa: E402
pytest.importorskip("zarr")

from haversack import io                                       # noqa: E402
from haversack.duckn_io import is_duckn_store                  # noqa: E402
from haversack.errors import InputError                        # noqa: E402


def _image():
    rng = np.random.default_rng(0)
    hu = rng.integers(-1000, 1500, (9, 11, 13)).astype(np.int16)
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((0.8, 0.9, 2.5))
    img.SetOrigin((-100.5, -80.25, 12.0))
    c, s = np.cos(0.3), np.sin(0.3)                           # a rotation about z
    img.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))
    return img


def _same(a, b):
    np.testing.assert_array_equal(sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b))
    np.testing.assert_allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-9)
    np.testing.assert_allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-6)
    np.testing.assert_allclose(a.GetDirection(), b.GetDirection(), atol=1e-9)


@pytest.mark.parametrize("suffix,fmt", [(".zarr.zip", "zarr.zip"), (".duckn.zip", "zarr.zip"),
                                        (".zarr", "zarr"), (".duckn", "zarr")])
def test_a_duckn_volume_reads_as_the_image_it_was_written_from(tmp_path, suffix, fmt):
    from duckn.sitk_adapter import from_sitk
    img = _image()
    p = tmp_path / f"ct{suffix}"
    duckn_write(from_sitk(img), p, format=fmt)
    assert is_duckn_store(p)
    back = io.read_image(p)
    assert back.GetDimension() == 3
    _same(back, img)


def test_a_zarr_directory_is_recognized_by_its_zarr_json_whatever_its_name(tmp_path):
    from duckn.sitk_adapter import from_sitk
    img = _image()
    p = tmp_path / "some_store"
    duckn_write(from_sitk(img), p, format="zarr")
    assert (p / "zarr.json").exists() and is_duckn_store(p)
    _same(io.read_image(p), img)


def test_value_transforms_are_applied_so_a_stored_uint16_ct_reads_in_hounsfield(tmp_path):
    """The classic: a CT stored as offset integers with a slope/intercept must come back
    calibrated, or every threshold in the pipeline (the -500 HU envelope cut) is wrong."""
    from duckn import DucknMetadata, ValueTransform
    from duckn.sitk_adapter import from_sitk
    from duckn.volume import Volume
    img = _image()
    vol = from_sitk(img)
    raw = (sitk.GetArrayFromImage(img).astype(np.int32) + 1024).astype(np.uint16)
    meta = DucknMetadata(**{**vol.metadata.model_dump(exclude_none=True),
                            "value_transforms": [ValueTransform(
                                name="linear", parameters={"slope": 1.0, "intercept": -1024.0})]})
    p = tmp_path / "ct.duckn.zip"
    duckn_write(Volume(raw=raw, metadata=meta), p, format="zarr.zip")
    back = io.read_image(p)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(back).astype(np.int32),
                                  sitk.GetArrayFromImage(img).astype(np.int32))
    np.testing.assert_allclose(back.GetOrigin(), img.GetOrigin(), atol=1e-6)


def test_a_ras_space_volume_lands_in_lps_like_every_other_input(tmp_path):
    """duckn may declare RAS; SimpleITK is LPS. The adapter flips x and y of the origin and
    the direction cosines - the left/right mirror that was a real bug three times."""
    from duckn import DucknMetadata
    from duckn.sitk_adapter import from_sitk
    from duckn.volume import Volume
    img = _image()
    vol = from_sitk(img)
    d = vol.metadata.model_dump(exclude_none=True)
    flip = np.array([-1.0, -1.0, 1.0])
    d["space"] = "right-anterior-superior"
    d["space_origin"] = (np.asarray(d["space_origin"]) * flip).tolist()
    for ax in d["axes"]:
        ax["space_direction"] = (np.asarray(ax["space_direction"]) * flip).tolist()
    p = tmp_path / "ras.zarr.zip"
    duckn_write(Volume(raw=vol.raw, metadata=DucknMetadata(**d)), p, format="zarr.zip")
    _same(io.read_image(p), img)


def test_a_four_dimensional_volume_is_refused(tmp_path):
    from duckn import AxisMetadata, DucknMetadata
    from duckn.volume import Volume
    meta = DucknMetadata(
        space="left-posterior-superior", space_origin=[0.0, 0.0, 0.0],
        axes=[AxisMetadata(kind="list")] + [
            AxisMetadata(kind="space", centering="cell", unit="mm", space_direction=v)
            for v in ([0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0])])
    p = tmp_path / "four.zarr.zip"
    duckn_write(Volume(raw=np.zeros((2, 3, 4, 5), np.uint8), metadata=meta), p, format="zarr.zip")
    with pytest.raises(InputError, match="3-D"):
        io.read_image(p)


def test_ordinary_inputs_are_untouched(tmp_path):
    p = tmp_path / "ct.nii.gz"
    sitk.WriteImage(_image(), str(p))
    assert not is_duckn_store(p)
    _same(io.read_image(p), _image())
