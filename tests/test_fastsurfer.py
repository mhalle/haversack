"""FastSurfer engine: the geometry (restore_logits) and the LUT, tested with
synthetic logits so no FastSurfer install or GPU is needed. The FastSurfer-
dependent compute (conform + inference) is validated by the live Modal smoke."""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from haversack.engines import fastsurfer as fs


def test_module_imports_without_fastsurfer():
    # importing the engine must not require FastSurfer (lazy inside segment())
    assert callable(fs.segment) and callable(fs.restore_logits)


def test_lut_has_canonical_freesurfer_labels():
    lut = fs.load_lut()
    assert lut[2]["name"] == "Left-Cerebral-White-Matter"
    assert lut[41]["name"] == "Right-Cerebral-White-Matter"
    assert lut[16]["name"] == "Brain-Stem"
    for v in (2, 41, 16):
        assert len(lut[v]["color"]) == 3 and all(0 <= c <= 255 for c in lut[v]["color"])
    assert len(lut) > 70          # the ~95 aparc+aseg structures (minus background)


def _upstream_lut_path():
    """FastSurfer's own colour table, wherever this machine has it.

    Engines get their own environments, so the checkout running the tests usually is NOT
    the one holding FastSurferCNN; look in the per-engine venvs too before giving up.
    """
    from pathlib import Path as _P
    try:
        import FastSurferCNN
        p = _P(FastSurferCNN.__file__).parent / "config" / "FastSurfer_ColorLUT.tsv"
        if p.exists():
            return p
    except ImportError:
        pass
    root = _P(__file__).resolve().parent.parent
    for p in sorted(root.glob(".venvs/*/lib/python*/site-packages/FastSurferCNN/config/"
                              "FastSurfer_ColorLUT.tsv")):
        return p
    return None


def test_the_shipped_lut_still_matches_the_one_fastsurfer_ships():
    """The shipped table is the ONLY source of FastSurfer label names now.

    Until 2026-09-08 the ranked store builder parsed upstream's `FastSurfer_ColorLUT.tsv`
    instead, so the two could disagree without anything noticing; they were verified equal
    by hand on all 78 ids and the builder was pointed at ours. A verification done once is
    a snapshot - a `fastsurfer-lean` bump could move upstream's table and nothing would say
    so, which is precisely the silent-staleness failure the builder's own docstring records
    (a changed LUT path once renamed all 78 segments to `label_<id>` while the build
    reported success). This pins it.

    Upstream carries id 0 `Background`; a label map excludes background by definition, so
    it is the one allowed difference and is asserted as such rather than merely ignored.
    """
    path = _upstream_lut_path()
    if path is None:
        pytest.skip("FastSurferCNN is not installed here and no .venvs copy was found")
    upstream = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 2 and f[0].strip().isdigit():
            upstream[int(f[0])] = f[1].strip()
    shipped = {i: v["name"] for i, v in fs.load_lut().items()}

    assert shipped, "the shipped LUT is empty"
    differing = {i: (shipped[i], upstream[i])
                 for i in set(shipped) & set(upstream) if shipped[i] != upstream[i]}
    assert differing == {}, f"shipped LUT disagrees with FastSurfer's own: {differing}"
    assert set(shipped) - set(upstream) == set(), \
        f"shipped LUT has ids upstream does not: {sorted(set(shipped) - set(upstream))}"
    assert set(upstream) - set(shipped) == {0}, \
        ("upstream should differ only by id 0 Background; it also has "
         f"{sorted(set(upstream) - set(shipped) - {0})}")


@pytest.mark.parametrize("override", [None, "/tmp/haversack-ckpt-probe"])
def test_the_engine_and_cache_admin_agree_where_checkpoints_live(monkeypatch, override):
    """Two modules answer "where are the checkpoints?" and both must say the same thing.

    They used to answer it from two hand-written copies of the convention, cache admin
    explaining in a comment that it was avoiding an engine import. The subdirectory and its
    override are `Engine.cache_store` now, so both read one fact - but they still reach it
    by different code, and a store that `cache clean` cannot find is one the disk keeps.
    """
    from haversack.cache_admin import stores
    if override is None:
        monkeypatch.delenv("HAVERSACK_FASTSURFER_CHECKPOINTS", raising=False)
    else:
        monkeypatch.setenv("HAVERSACK_FASTSURFER_CHECKPOINTS", override)
    admin = [s for s in stores() if s["name"] == "checkpoints"][0]["path"]
    assert str(admin) == str(fs.checkpoint_dir()), (
        f"cache admin says {admin}, the engine says {fs.checkpoint_dir()}")
    if override is not None:
        assert str(admin) == override


def _img(arr, spacing, origin=(0., 0., 0.)):
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    im.SetSpacing(spacing); im.SetOrigin(origin)
    return im


def test_restore_logits_places_boundary_at_physical_location():
    """A 2-class field whose boundary is the plane x=c: after restore to a finer
    grid, the argmax boundary must sit at the same physical x=c (sub-voxel), not
    snap to the coarse grid."""
    Zc = Yc = Xc = 24
    sp_c = 1.0
    xc_phys = 11.3                                   # boundary NOT on a coarse voxel center
    # sitk array is (z,y,x); build 2 channels of logits: ch1-ch0 = (x_phys - xc)
    xphys = (np.arange(Xc) * sp_c)[None, None, :] * np.ones((Zc, Yc, Xc))
    d = xphys - xc_phys
    logit = np.stack([-d, d], axis=-1).astype(np.float32)   # argmax=1 where x>xc
    source = _img(np.zeros((Zc, Yc, Xc)), (sp_c, sp_c, sp_c))

    # target: finer grid (0.25mm) over the same FOV -> upsampling
    sp_f = 0.25
    Xf = int(Xc * sp_c / sp_f)
    target = _img(np.zeros((Zc*4, Yc*4, Xf)), (sp_f, sp_f, sp_f))

    idx = fs.restore_logits(logit, source, target)
    # find the crossover column per row: first x where idx==1
    mid = idx[idx.shape[0]//2, idx.shape[1]//2, :]
    cross = np.argmax(mid == 1)                      # first index labeled 1
    cross_phys = cross * sp_f
    assert abs(cross_phys - xc_phys) <= sp_f + 1e-6, (cross_phys, xc_phys)


def test_restore_logits_beats_label_nn_on_a_slanted_boundary():
    """On upsampling, resampling the graded field + argmax matches the true
    boundary better than nearest-neighbor resampling the coarse labelmap."""
    from scipy import ndimage
    Z = Y = X = 20
    # slanted boundary: label 1 where (x + 0.5*y) > c, graded logit = signed dist
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    signed = (xx + 0.5*yy) - 15.0
    logit = np.stack([-signed, signed], axis=-1).astype(np.float32)
    lab_coarse = (signed > 0).astype(np.int32)
    source = _img(np.zeros((Z, Y, X)), (1., 1., 1.))
    f = 4
    target = _img(np.zeros((Z*f, Y*f, X*f)), (0.25, 0.25, 0.25))

    lg = fs.restore_logits(logit, source, target)
    # truth at fine res
    zf, yf, xf = np.meshgrid(*[(np.arange(n*f)*0.25) for n in (Z, Y, X)], indexing="ij")
    truth = ((xf + 0.5*yf) - 15.0 > 0).astype(np.int32)
    nn = ndimage.zoom(lab_coarse, f, order=0)
    # compare the interior only: target voxels near the FOV edge sample outside
    # the source support (default 0) - a test artifact, not a restore effect
    c = tuple(slice(8, -8) for _ in range(3))
    lg_err = float((lg[c] != truth[c]).mean())
    nn_err = float((nn[c] != truth[c]).mean())
    assert lg_err < nn_err, (lg_err, nn_err)        # logit-grade closer to truth
    assert lg_err < 0.005                            # near-exact on a linear field


def test_sitk_to_nibabel_roundtrip_preserves_geometry():
    """The SITK->nibabel bridge must preserve geometry: a marked voxel's RAS
    physical location and value must survive the conversion (LPS->RAS,
    zyx->xyz). A wrong axis/sign here is the silent-mirror class of bug."""
    nib = pytest.importorskip("nibabel")
    # a non-trivial axis-aligned geometry (anisotropic, flipped, offset)
    arr = np.zeros((6, 8, 10), dtype=np.float32)      # sitk (z,y,x)
    arr[1, 2, 3] = 7.0                                # a marked voxel, sitk index (x=3,y=2,z=1)
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing((1.0, 1.5, 2.0)); im.SetOrigin((10.0, -20.0, 5.0))
    im.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))   # LPS-ish flip

    nb = fs.sitk_to_nibabel(im)
    data = np.asanyarray(nb.dataobj)
    # value at nibabel index (i=3, j=2, k=1) == the marked voxel
    assert data[3, 2, 1] == 7.0
    # physical location must match: sitk physical point (RAS) of that voxel
    px, py, pz = im.TransformIndexToPhysicalPoint((3, 2, 1))    # LPS
    ras_sitk = np.array([-px, -py, pz])                          # LPS -> RAS
    ras_nib = (nb.affine @ np.array([3, 2, 1, 1.0]))[:3]
    assert np.allclose(ras_sitk, ras_nib, atol=1e-6), (ras_sitk, ras_nib)


def test_restore_gpu_matches_cpu_reference():
    """The GPU restore (grid_sample) must reproduce the SimpleITK CPU restore's
    argmax labels: same physical-space mapping and half-pixel (voxel-center)
    convention. Run on the CPU torch device so the geometry math is verified
    without a GPU. A flipped direction + anisotropic spacing + offset origin +
    upsampling exercises _resample_affine (the silent-bug-prone part)."""
    pytest.importorskip("torch")
    Z = Y = X = 16
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    # smooth per-class linear fields -> well-defined argmax planes, not tie noise
    feats = [xx, yy, zz, xx + yy, (X - 1 - xx)]
    logit = np.stack([f.astype(np.float32) for f in feats], axis=-1)     # (Z,Y,X,K)
    flip = (-1., 0., 0., 0., -1., 0., 0., 0., 1.)
    source = _img(np.zeros((Z, Y, X)), (1.5, 1.25, 1.0), origin=(10., -20., 5.))
    source.SetDirection(flip)
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.75, 0.6, 0.5), origin=(8., -18., 6.))
    target.SetDirection(flip)

    cpu = fs.restore_logits(logit, source, target)
    gpu = fs.restore_logits_gpu(logit, source, target, device="cpu")
    assert cpu.shape == gpu.shape == (Z * 2, Y * 2, X * 2)
    interior = tuple(slice(3, -3) for _ in range(3))     # edges differ by padding only
    agree = float((cpu[interior] == gpu[interior]).mean())
    assert agree > 0.99, agree


def test_restore_gpu_tensor_input_matches_numpy_input():
    """The on-GPU handoff passes a (K,Zs,Ys,Xs) torch tensor instead of the numpy
    (Zs,Ys,Xs,K) field; both must yield identical labels. Guards the permute
    layout used when the field is kept resident on the device."""
    torch = pytest.importorskip("torch")
    Z = Y = X = 12
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    logit = np.stack([xx, yy, zz, xx + yy], axis=-1).astype(np.float32)   # (Z,Y,X,K)
    source = _img(np.zeros((Z, Y, X)), (1.0, 1.25, 1.5), origin=(3., -4., 5.))
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.5, 0.625, 0.75), origin=(3., -4., 5.))

    from_numpy = fs.restore_logits_gpu(logit, source, target, device="cpu")
    tens = torch.from_numpy(logit).permute(3, 0, 1, 2).contiguous()       # (K,Z,Y,X)
    from_tensor = fs.restore_logits_gpu(tens, source, target, device="cpu")
    assert np.array_equal(from_numpy, from_tensor)


def test_restore_gpu_labels_do_not_depend_on_its_channel_groups_or_target_slabs():
    """restore_logits_gpu used to widen the whole source to fp32 and sample all K channels
    onto the whole target before its argmax - 17 GB and 42 GB for a 384^3 source and a
    512^3 target, past an A10G's 22 GB (2026-09-11). It now walks channel groups and target
    Z-slabs with a running argmax, and the labels must be exactly the one-piece labels
    whatever the sizes: ties (quantized fp16 values) exercise the lowest-index rule ACROSS
    groups, a flipped, anisotropic, offset geometry the per-slab grid. The source is the
    on-GPU path's form - an fp16 (K,Z,Y,X) view of the (X,Y,Z,K) field - and the numpy fp32
    copy of the same values must agree with it, since widening is exact."""
    torch = pytest.importorskip("torch")
    Z, Y, X, K = 9, 10, 11, 7
    rng = np.random.default_rng(2)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    feats = [xx, yy, zz, xx + yy - zz, X - 1 - xx, yy - xx, zz + 2]
    vals = np.stack([f + rng.normal(0, 1.5, f.shape) for f in feats], axis=0)   # (K,Z,Y,X)
    # Interpolated values almost never tie by chance, so make them: channels 5 and 6
    # duplicate 1 and 0 exactly - equal after sampling everywhere, so wherever either wins
    # the restore must name the LOWER one, even from a later group.
    vals[5], vals[6] = vals[1], vals[0]
    field = torch.from_numpy(np.round(vals * 2) / 2).permute(3, 2, 1, 0).contiguous()   # (X,Y,Z,K)
    field = field.to(torch.float16)
    view = field.permute(3, 2, 1, 0)                                   # (K,Z,Y,X), strided
    as_numpy = np.ascontiguousarray(np.transpose(field.float().numpy(), (2, 1, 0, 3)))   # (Z,Y,X,K)
    flip = (-1., 0., 0., 0., -1., 0., 0., 0., 1.)
    source = _img(np.zeros((Z, Y, X)), (1.5, 1.25, 1.0), origin=(10., -20., 5.))
    source.SetDirection(flip)
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.75, 0.6, 0.5), origin=(8., -18., 6.))
    target.SetDirection(flip)
    plane = (Y * 2) * (X * 2)

    one_piece = fs.restore_logits_gpu(view, source, target, device="cpu",
                                      group=K, slab_voxels=1 << 30)
    assert len(np.unique(one_piece)) > 2                               # not a trivial field
    for group, slab in ((1, 1), (2, plane * 3), (3, plane), (4, plane * 7), (K, 1)):
        got = fs.restore_logits_gpu(view, source, target, device="cpu",
                                    group=group, slab_voxels=slab)
        assert np.array_equal(got, one_piece), (group, slab)
    assert np.array_equal(fs.restore_logits_gpu(as_numpy, source, target, device="cpu",
                                                group=3, slab_voxels=plane * 2), one_piece)


def test_restore_cpu_takes_the_fp16_field_as_a_view_and_matches_the_fp32_copy():
    """Until 2026-09-11 the CPU path handed the restore an fp32, (Z,Y,X,K)-contiguous copy
    of FastSurfer's fp16 field, widened and transposed as two full copies - about 105 GB at
    peak for a 512^3 field on a Mac. It now hands over one fp16 (Z,Y,X,K) copy as a
    (K,Z,Y,X) view, each channel widened as the restore reads it. fp16 -> fp32 is exact, so
    the labels must be IDENTICAL, not close - including where classes tie, which fp16's
    coarse steps make common, and which the quantized values here force. The restore
    promises any layout, so an arbitrarily strided view must agree as well."""
    torch = pytest.importorskip("torch")
    X, Y, Z, K = 10, 12, 14, 5                    # distinct, so a swapped axis cannot pass
    rng = np.random.default_rng(0)
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    feats = [xx, yy, zz, xx + yy - zz, X - 1 - xx]
    vals = np.stack([f + rng.normal(0, 2, f.shape) for f in feats], axis=-1)
    field = torch.from_numpy(np.round(vals * 4) / 4).to(torch.float16)   # (X,Y,Z,K), ties
    handed = field.permute(2, 1, 0, 3).contiguous().permute(3, 0, 1, 2)   # _capture_logits'
    strided = field.permute(3, 2, 1, 0)                                   # any other layout
    old = np.ascontiguousarray(np.transpose(field.float().numpy(), (2, 1, 0, 3)))   # (Z,Y,X,K)

    source = _img(np.zeros((Z, Y, X)), (1.0, 1.25, 1.5), origin=(3., -4., 5.))
    source.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
    target = _img(np.zeros((Z * 2, Y * 2, X * 2)), (0.5, 0.6, 0.7), origin=(2., -3., 6.))
    target.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))

    expected = fs.restore_logits(old, source, target)
    for view in (handed, strided):
        assert not view.is_contiguous() and view.dtype == torch.float16
        assert np.array_equal(fs.restore_logits(view, source, target), expected)


def test_the_field_leaves_mps_exactly_when_one_buffer_cannot_hold_it():
    """A 384^3 FastSurfer run died on MPS allocating its (X,Y,Z,79) fp16 field: PyTorch's
    MPS allocator refuses a buffer past Metal's maxBufferLength, 8 GiB on the M2 it was
    measured on (2026-09-11) - 378^3 allocates, 379^3 does not. field_device moves such a
    field to the host (FastSurfer's own viewagg_device=cpu) and must not move smaller ones,
    which would cost speed for nothing. Only MPS has this limit."""
    K = 79
    assert fs.field_device("mps", (378, 378, 378, K)) == "mps"          # 7.95 GiB: allocates
    assert fs.field_device("mps", (379, 379, 379, K)) == "cpu"          # 8.01 GiB: refused
    assert fs.field_device("mps", (512, 512, 512, K)) == "cpu"
    assert fs.field_device("mps", (366, 366, 366, K)) == "mps"          # 256 mm FOV at 0.7 mm
    assert fs.field_device("cuda:0", (512, 512, 512, K)) == "cuda:0"    # no such buffer limit
    assert fs.field_device("cpu", (512, 512, 512, K)) == "cpu"


@pytest.mark.parametrize("zooms, want_vox, want_floored_from", [
    ((0.5, 0.5, 0.5), 0.7, 0.5),            # the case the floor exists for
    ((0.69, 0.69, 0.69), 0.7, 0.69),
    ((0.45, 0.45, 1.0), 0.7, 0.45),         # anisotropic: "min" takes the finest axis
    ((0.7, 0.7, 0.7), "min", None),         # at the floor: FastSurfer's own mode, untouched
    ((0.8, 0.8, 0.8), "min", None),
    ((0.9375, 0.9375, 1.2), "min", None),
    ((0.96, 0.96, 0.96), "min", None),      # FastSurfer snaps this to 1 mm itself
    ((1.0, 1.3, 1.0), "min", None),         # ds000114's T1
    ((1.2, 1.2, 1.2), "min", None),         # above the 1 mm cap
])
def test_the_floor_replaces_only_choices_finer_than_it(zooms, want_vox, want_floored_from):
    """FastSurfer's "min" rule has a 1 mm cap and no floor. The floor must raise ONLY a
    choice finer than 0.7 mm; anything else gets FastSurfer's own mode back verbatim, so
    its conform - and so its bytes - are literally what they were before the floor. The
    conform settings are FastSurfer's parser defaults, read rather than restated."""
    pytest.importorskip("FastSurferCNN")
    nib = pytest.importorskip("nibabel")
    from FastSurferCNN import run_prediction as rp
    a = rp.make_parser().parse_args(["--t1", "x", "--sd", "x"])
    ck = {"vox_size": a.vox_size, "img_size": a.image_size,
          "threshold_1mm": a.conform_to_1mm_threshold}
    assert ck["vox_size"] == "min"                        # what haversack runs today
    img = nib.Nifti1Image(np.zeros((16, 16, 16), np.uint8), np.diag([*zooms, 1.0]))
    vox, floored_from = fs.processing_vox_size(img, ck)
    assert vox == want_vox
    assert floored_from == (pytest.approx(want_floored_from) if want_floored_from else None)


@pytest.mark.slow
def test_the_largest_field_kept_on_mps_really_allocates():
    """The constant must never OVERestimate the device: a field field_device keeps on MPS
    has to allocate there. Allocates ~8 GB, so slow and MPS-only."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    K, n = 79, 378
    assert fs.field_device("mps", (n, n, n, K)) == "mps"
    t = torch.zeros((n, n, n, K), dtype=torch.float16, device="mps")
    t[:, :8].add_(torch.ones((n, 8, n, K), dtype=torch.float16, device="mps"), alpha=0.4)
    torch.mps.synchronize()
    del t
    torch.mps.empty_cache()


def test_sitk_nibabel_sitk_roundtrip_is_geometry_exact():
    """sitk -> nibabel -> sitk must recover size/spacing/origin/direction/data
    exactly. This is the bridge that recovers the conformed-orig geometry for the
    logit restore without a file round-trip; a silent error here misplaces every
    boundary."""
    arr = np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10)   # sitk (z,y,x)
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing((1.0, 1.5, 2.0)); im.SetOrigin((10.0, -20.0, 5.0))
    im.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))

    back = fs.nibabel_to_sitk(fs.sitk_to_nibabel(im))
    assert back.GetSize() == im.GetSize()
    assert np.allclose(back.GetSpacing(), im.GetSpacing(), atol=1e-9)
    assert np.allclose(back.GetOrigin(), im.GetOrigin(), atol=1e-6)
    assert np.allclose(back.GetDirection(), im.GetDirection(), atol=1e-9)
    assert np.array_equal(sitk.GetArrayFromImage(back), arr)


def test_emit_probabilities_hands_over_the_field_with_both_grids():
    """The engine hook: FastSurfer's 79-class field goes through the same encoder as
    nnU-Net's, carrying enough geometry to redo the restore later."""
    from haversack import ranked

    torch = pytest.importorskip("torch")
    K, Z, Y, X = 6, 4, 5, 6
    lg = torch.randn(K, Z, Y, X)
    conf = sitk.GetImageFromArray(np.zeros((Z, Y, X), np.float32))
    conf.SetSpacing((1.0, 1.0, 1.0)); conf.SetOrigin((-1.5, 2.0, 0.25))
    tgt = sitk.GetImageFromArray(np.zeros((Z + 1, Y, X), np.float32))
    tgt.SetSpacing((0.8, 0.9, 1.1)); tgt.SetOrigin((3.0, -4.0, 5.0))

    got = []
    spec = ranked.RankedSpec(sink=lambda part, code: got.append((part, code)), depth=3)
    fs.emit_probabilities(spec, lg, conf, tgt, list(range(K)))

    assert len(got) == 1
    part, code = got[0]
    assert part == "brain"
    assert code.meta["engine"] == "fastsurfer"          # a reader must know what made it
    assert code.meta["labels"] == list(range(K))
    assert code.meta["source_grid"]["origin_xyz"] == [-1.5, 2.0, 0.25]   # world: xyz
    assert code.meta["target_grid"]["spacing_zyx"] == [1.1, 0.9, 0.8]    # array: zyx
    assert code.meta["source_grid"]["shape_zyx"] != code.meta["target_grid"]["shape_zyx"]


def test_emit_probabilities_accepts_the_cpu_paths_axis_order():
    """_capture_logits returns (K,Z,Y,X) torch on every identity path (on the device, or an
    fp16 view on the host since 2026-09-11) but still (Z,Y,X,K) numpy after a non-identity
    reorder; the encoder only takes the former, so the hook must transpose."""
    from haversack import ranked

    pytest.importorskip("torch")
    K, Z, Y, X = 5, 3, 4, 5
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((Z, Y, X, K)).astype(np.float32)
    ref = sitk.GetImageFromArray(np.zeros((Z, Y, X), np.float32))

    got = []
    spec = ranked.RankedSpec(sink=lambda part, code: got.append(code), depth=3)
    fs.emit_probabilities(spec, lg, ref, ref, list(range(K)))

    assert got[0].meta["shape"] == [Z, Y, X]            # not the (Z,Y,X,K) misread
    assert got[0].meta["classes"] == K
    # and the winner survives the transpose
    np.testing.assert_array_equal(got[0].ranks[0] - 1, lg.argmax(axis=3).astype(np.uint8))


def test_emit_probabilities_stores_the_same_bytes_from_the_fp16_view_as_from_the_fp32_copy():
    """The host path used to hand the encoder an fp32 contiguous copy of the field; since
    2026-09-11 it hands over the fp16 field itself as a strided (K,Z,Y,X) view, and the
    encoder widens each slab. fp16 -> fp32 is exact, so every stored plane must be
    IDENTICAL - ranks, gap bytes and tail - including at the ties fp16 makes common, and
    the quantized values here force."""
    from haversack import ranked
    torch = pytest.importorskip("torch")
    X, Y, Z, K = 6, 5, 4, 7
    rng = np.random.default_rng(1)
    field = torch.from_numpy(np.round(rng.normal(0, 3, (X, Y, Z, K)) * 4) / 4).to(torch.float16)
    handed = field.permute(2, 1, 0, 3).contiguous().permute(3, 0, 1, 2)   # _capture_logits'
    old = np.ascontiguousarray(np.transpose(field.float().numpy(), (2, 1, 0, 3)))   # (Z,Y,X,K)
    ref = sitk.GetImageFromArray(np.zeros((Z, Y, X), np.float32))

    codes = []
    for lg in (old, handed):
        spec = ranked.RankedSpec(sink=lambda part, code: codes.append(code), depth=4)
        fs.emit_probabilities(spec, lg, ref, ref, list(range(K)))
    a, b = codes
    for name in ("ranks", "support", "tail"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.dtype == y.dtype and np.array_equal(x, y), name


def test_emit_probabilities_is_a_noop_without_a_spec():
    ref = sitk.GetImageFromArray(np.zeros((2, 2, 2), np.float32))
    fs.emit_probabilities(None, np.zeros((2, 2, 2, 3), np.float32), ref, ref, [0, 1, 2])


def test_checkpoint_dir_follows_env_then_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("HAVERSACK_FASTSURFER_CHECKPOINTS", str(tmp_path / "mine"))
    assert fs.checkpoint_dir() == tmp_path / "mine"
    monkeypatch.delenv("HAVERSACK_FASTSURFER_CHECKPOINTS")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert fs.checkpoint_dir() == tmp_path / "cache" / "haversack" / "fastsurfer-checkpoints"
    args = fs.checkpoint_args()
    assert args[::2] == ["--ckpt_ax", "--ckpt_cor", "--ckpt_sag"]
    assert [a.rsplit("/", 1)[1] for a in args[1::2]] == list(fs.CHECKPOINT_NAMES)


def test_ensure_checkpoints_fetches_verifies_and_is_idempotent(monkeypatch, tmp_path):
    """The Zenodo fetch (FastSurfer's b2share download fails cert verification, 2026-09-03):
    a present file with the right hash is left alone; a bad hash is refused."""
    import hashlib
    from haversack import fetchlib

    blobs = {n: f"weights-of-{n}".encode() for n in fs.CHECKPOINTS}
    monkeypatch.setattr(fs, "CHECKPOINTS", {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()})
    fetched = []

    class Resp:
        def __init__(self, data): self.data = data
        def read(self, n=-1):                 # sized reads, like a real HTTPResponse
            out, self.data = (self.data, b"") if n in (-1, None) else (self.data[:n], self.data[n:])
            return out
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        name = req.full_url.split("/files/")[1].split("?")[0]
        fetched.append(name)
        return Resp(blobs[name])

    monkeypatch.setattr(fetchlib, "urlopen", fake_urlopen)
    d = fs.ensure_checkpoints(tmp_path)
    assert sorted(fetched) == sorted(fs.CHECKPOINTS) and d == tmp_path
    fs.ensure_checkpoints(tmp_path)                        # second call: nothing refetched
    assert sorted(fetched) == sorted(fs.CHECKPOINTS)
    # a corrupted file is refetched, and a permanently-wrong hash is refused
    (tmp_path / next(iter(fs.CHECKPOINTS))).write_bytes(b"corrupt")
    fs.ensure_checkpoints(tmp_path)
    assert len(fetched) == len(fs.CHECKPOINTS) + 1
    monkeypatch.setitem(fs.CHECKPOINTS, next(iter(fs.CHECKPOINTS)), "0" * 64)
    (tmp_path / next(iter(fs.CHECKPOINTS))).unlink()
    from haversack.errors import ResourceError
    with pytest.raises(ResourceError, match="sha256"):
        fs.ensure_checkpoints(tmp_path)
