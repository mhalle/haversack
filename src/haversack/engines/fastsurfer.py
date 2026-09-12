"""FastSurfer whole-brain parcellation as a haversack engine.

FastSurferVINN is a 2.5D view-aggregation network, not an nnU-Net model, so it
is an *engine* (a different runner) rather than an ecosystem entry. This module
produces a :class:`haversack.result.Segmentation` on the input's grid so it flows
through the same cache / preview / statistics / client path as an nnU-Net run.

The value haversack adds over FastSurfer's own output: FastSurfer argmaxes at its
conformed 1 mm grid and only nearest-neighbor reorients the labelmap back to the
input orientation - it never restores to the input *grid*, and never at logit
grade. Here we capture the pre-argmax logit field, resample **it** to the input
grid (physical-space, so oblique acquisitions are handled), and argmax after -
sub-voxel boundary placement instead of a blocky label resample. Proven on Modal
2026-08-26 (1 mm self-check reproduces FastSurfer's own labels exactly; the
graded restore de-stairsteps when upsampling).

FastSurfer itself is imported lazily inside :func:`segment` - importing haversack,
or this module, never requires FastSurfer to be installed. The restore geometry
(:func:`restore_logits`) is dependency-light and unit-tested on its own.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import registry as _registry
from .geometry import resample_affine as _resample_affine

_LUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fastsurfer_lut.json"

ENGINE = "fastsurfer"

# The weights identity lives in the engine registry, which the API-side describe
# and the worker-side re-key both read - one literal, so the key a worker stores
# a result under and the key a plain GET computes cannot diverge. (They did once:
# worker keyed `fastsurfer=vinn-v2`, API keyed `unknown`, and every bare read
# 404'd a result that was actually cached. Bug found 2026-08-26.)
WEIGHTS_ID = ENGINE
WEIGHTS_VERSION = _registry.ENGINES[ENGINE].weights_identity()[0]["version"]


def weights_installed() -> list[dict]:
    """The engine's weights identity for the result-cache key (from the registry).

    FastSurfer bakes its checkpoints into the worker image rather tha haversack's
    weights volume, so this is a fixed version, not an install sidecar read."""
    return _registry.ENGINES[ENGINE].weights_identity()


def load_lut() -> dict[int, dict]:
    """FastSurfer output labels -> {name, color}. The segment table for the
    ``.seg.nrrd`` (names) and the canonical FreeSurfer colors."""
    raw = json.loads(_LUT_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def label_names() -> dict[int, str]:
    """FastSurfer output label id -> name, for consumers that need names without
    colors (the ranked store builder).

    Reads the LUT haversack SHIPS, not FastSurfer's ``FastSurfer_ColorLUT.tsv``.
    The builder used to hunt for that file - importing FastSurferCNN, then
    globbing ``.venvs/*/`` when the import failed, which it does in the ordinary
    case because engines get their own environments. Verified 2026-09-08:
    identical to upstream's table, name for name, on all 78 ids; upstream adds
    only id 0 ``Background``, which a label map excludes by definition. Reading
    what we ship removes an environment dependency from a build step that once
    degraded silently and named all 78 segments ``label_<id>``.
    """
    return {i: v["name"] for i, v in load_lut().items()}


def sitk_to_nibabel(img):
    """A SimpleITK image -> an in-memory nibabel Nifti1Image, so FastSurfer's
    ``conform`` (which is nibabel-coupled, and which we deliberately do not
    reimplement) can consume SimpleITK-decoded data without a file round-trip.

    The one geometry conversion on the way in: SimpleITK is LPS with array order
    (z, y, x); nibabel is RAS with array order (i, j, k) = (x, y, z). So the
    data is transposed to (x, y, z) and the affine is built from the direction/
    spacing/origin with the first two axes negated (LPS -> RAS). Round-trip
    tested."""
    import nibabel as nib
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img)                       # (z, y, x)
    data = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))   # (x, y, z)
    sp = np.asarray(img.GetSpacing(), dtype=np.float64)     # (sx, sy, sz)
    D = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = D * sp[np.newaxis, :]                     # columns scaled by spacing (LPS)
    aff[:3, 3] = np.asarray(img.GetOrigin(), dtype=np.float64)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff             # LPS -> RAS
    return nib.Nifti1Image(data, aff)


def nibabel_to_sitk(nb):
    """The inverse of :func:`sitk_to_nibabel`: an in-memory nibabel image back to
    a SimpleITK image. Used to recover the conformed-orig geometry from the
    conformed nibabel image FastSurfer produces, so the logit restore's source
    grid needs no file round-trip. Assumes an orthogonal affine (direction *
    spacing, no shear) - true for FastSurfer's conformed output and for medical
    acquisitions. Round-trip tested against sitk_to_nibabel."""
    import SimpleITK as sitk

    data = np.asanyarray(nb.dataobj)                        # (x, y, z)
    arr = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))   # (z, y, x) for sitk
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(nb.affine, dtype=np.float64)  # RAS -> LPS
    M = aff[:3, :3]
    sp = np.linalg.norm(M, axis=0)                          # column norms = spacing
    D = M / sp[np.newaxis, :]                               # unit columns = direction cosines
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(float(s) for s in sp))
    img.SetOrigin(tuple(float(o) for o in aff[:3, 3]))
    img.SetDirection(tuple(float(d) for d in D.flatten()))
    return img


def restore_logits(logits, source_ref, target_ref):
    """Resample a per-class logit field from ``source_ref``'s grid onto
    ``target_ref``'s grid (SimpleITK physical space, so any orientation/spacing
    difference is handled), and argmax over classes -> a class-index volume on
    the target grid.

    ``logits`` is a numpy ``(Z, Y, X, K)`` field (SimpleITK array order) or a torch
    ``(K, Z, Y, X)`` tensor of any float dtype and layout - the latter is what
    ``_capture_logits`` hands over on the host path since 2026-09-11, fp16 and strided -
    aligned with ``source_ref`` (a SimpleITK image carrying the source geometry). A tensor
    channel is widened to fp32 only as it is read: SimpleITK has no fp16, and widening is
    exact.
    ``target_ref`` is a SimpleITK image whose grid is the desired output.

    Streams one channel at a time with a running argmax, so peak memory is one
    source channel + one target channel + the O(target) accumulators - never the
    full resampled K-channel volume. This is the only interpolation in the
    output path; the coordinate change (LIA -> input orientation) is exact and
    lives in the caller's capture.
    """
    import SimpleITK as sitk

    tensor = not isinstance(logits, np.ndarray)       # torch (K,Z,Y,X), or numpy (Z,Y,X,K)
    if tensor:
        import torch
    K = int(logits.shape[0] if tensor else logits.shape[3])
    tgt_size = target_ref.GetSize()
    tgt_shape = (tgt_size[2], tgt_size[1], tgt_size[0])   # sitk size is (x,y,z)
    best = np.full(tgt_shape, -np.inf, dtype=np.float32)
    idx = np.zeros(tgt_shape, dtype=np.int32)
    for k in range(K):
        chan = (logits[k].to("cpu", torch.float32, memory_format=torch.contiguous_format).numpy()
                if tensor else np.ascontiguousarray(logits[..., k]))
        ch = sitk.GetImageFromArray(chan)
        ch.CopyInformation(source_ref)
        native = sitk.GetArrayFromImage(
            sitk.Resample(ch, target_ref, sitk.Transform(),
                          sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        up = native > best
        best[up] = native[up]
        idx[up] = k
    return idx


def restore_logits_gpu(logits_in, source_ref, target_ref, device="cuda", *,
                       group: int | None = None, slab_voxels: int = 1 << 24,
                       group_bytes: int = 1 << 30):
    """GPU equivalent of :func:`restore_logits`: trilinear-resample the whole
    K-channel logit field from ``source_ref``'s grid onto ``target_ref``'s grid
    and argmax over classes, in one batched ``grid_sample`` on ``device`` instead
    of 79 CPU SimpleITK resamples. Uses the FULL physical-space affine (via
    :func:`_resample_affine`, composing both grids' direction cosines), so
    orientation, spacing AND oblique rotation are handled uniformly; same
    half-pixel (voxel-center) convention as SimpleITK (``align_corners=False``,
    zero padding outside).

    ``logits_in`` is either a torch tensor ``(K, Zs, Ys, Xs)`` of any float dtype and
    layout (the on-GPU path hands over a view of the fp16 field - no host<->device copy)
    or a numpy ``(Zs, Ys, Xs, K)`` field, moved to ``device`` a channel group at a time.
    Returns a ``(Z, Y, X)`` int32 class-index volume (target array order).

    Memory is bounded by the groups, never the whole field in fp32 on either grid. Until
    2026-09-12 this widened the whole source to fp32 in one piece and sampled all K
    channels onto the whole target before its argmax: 17 GB and 42 GB for a 384^3 source
    and a 512^3 target, past an A10G's 22 GB. Now channels go ``group`` at a time (default:
    as many as fit ``group_bytes`` of fp32 source), each widened once, and within a group
    the target is walked in Z-slabs of about ``slab_voxels``; a running argmax on the
    device keeps the LOWEST index on ties, torch.argmax's rule, so the labels do not
    depend on either size. The per-slab grid is computed elementwise from global target
    indices, so a voxel's sample position does not depend on which slab it fell in."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    if isinstance(logits_in, torch.Tensor):              # (K,Zs,Ys,Xs), any layout/dtype/device
        src_all = logits_in
    else:                                                # numpy (Zs,Ys,Xs,K), host
        src_all = torch.from_numpy(np.asarray(logits_in)).permute(3, 0, 1, 2)
    K, Zs, Ys, Xs = (int(s) for s in src_all.shape)
    tgt = target_ref.GetSize()                            # (Xt, Yt, Zt)
    Xt, Yt, Zt = int(tgt[0]), int(tgt[1]), int(tgt[2])
    A, t = _resample_affine(source_ref, target_ref)
    A = [[float(a) for a in row] for row in np.asarray(A)]
    t = [float(v) for v in np.asarray(t)]
    N = torch.as_tensor([Xs, Ys, Zs], device=dev, dtype=torch.float64)
    xs = torch.arange(Xt, device=dev, dtype=torch.float64)[None, None, :]
    ys = torch.arange(Yt, device=dev, dtype=torch.float64)[None, :, None]

    def grid_for(z0, z1):
        """Normalized sample positions for target planes z0:z1, voxel-center convention
        (align_corners=False). Elementwise in the order x, y, z, then the offset, from
        GLOBAL indices - the same arithmetic for a voxel whichever slab holds it."""
        zz = torch.arange(z0, z1, device=dev, dtype=torch.float64)[:, None, None]
        src = torch.stack([xs * A[r][0] + ys * A[r][1] + zz * A[r][2] + t[r] for r in range(3)],
                          dim=-1)                                     # (zs,Yt,Xt,3) source (x,y,z)
        return ((src + 0.5) * 2.0 / N - 1.0).to(torch.float32)[None]  # (1,zs,Yt,Xt,3)

    g = group or max(1, min(K, group_bytes // (4 * Zs * Ys * Xs)))
    zs = max(1, min(Zt, slab_voxels // (Yt * Xt)))
    best = torch.full((Zt, Yt, Xt), float("-inf"), device=dev, dtype=torch.float32)
    bidx = torch.zeros((Zt, Yt, Xt), device=dev, dtype=torch.int32)
    for k0 in range(0, K, g):
        k1 = min(k0 + g, K)
        chan = src_all[k0:k1].to(dev).float().contiguous()[None]     # this group only, widened once
        for z0 in range(0, Zt, zs):
            z1 = min(z0 + zs, Zt)
            out = F.grid_sample(chan, grid_for(z0, z1), mode="bilinear",
                                padding_mode="zeros", align_corners=False)[0]   # (g,zs,Yt,Xt)
            i = out.argmax(dim=0)                                     # lowest index within the group
            v = out.gather(0, i[None])[0]
            b = best[z0:z1]
            up = v > b                                                # strict: an earlier group keeps a tie
            best[z0:z1] = torch.where(up, v, b)
            bidx[z0:z1] = torch.where(up, (i + k0).to(torch.int32), bidx[z0:z1])
            del out, i, v, up
        del chan
    return bidx.cpu().numpy()


_RUNNERS: dict = {}          # (device, batch_size) -> RunModelOnData, cached across jobs


def _host_memory_gb() -> float:
    """Physical memory of this host, in GB (0 when it cannot be read)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def local_viewagg(device: str, host_gb: float | None = None) -> str:
    """Where FastSurfer's view-aggregation field lives on a local host.

    The field is the whole 79-class probability volume, 2.6 GB fp16 at 256^3, and the
    three view passes accumulate into it. On CUDA FastSurfer's own ``"auto"`` puts it on
    the card when the card has 4 GB. MPS has no separate pool: the field would come out
    of the same unified memory as everything else, and on a 16 GB machine that is the
    difference between running and swapping - so it stays on the CPU unless the host
    is large. Memory is a policy, not a constant: the threshold reads the host."""
    dev = str(device)
    if dev.startswith("cuda"):
        return "auto"
    if dev == "mps":
        gb = _host_memory_gb() if host_gb is None else host_gb
        return "mps" if gb >= 32 else "cpu"
    return "cpu"


#: The largest single MPS buffer, in bytes. PyTorch's MPS allocator refuses anything past
#: Metal's ``maxBufferLength`` ("Invalid buffer size"), which is 8 GiB on the 16 GB M2 this
#: was measured on (2026-09-11): a (378^3, 79) fp16 field allocates, (379^3, 79) does not -
#: the crossover a 384^3 FastSurfer run first failed at. Larger machines report a larger
#: limit, but torch exposes no way to read it, so a field past this goes to the host even
#: where one buffer might have held it: slower, never wrong, and the same on every Mac.
MPS_MAX_BUFFER_BYTES = 8 << 30


def field_device(device, shape, itemsize: int = 2) -> str:
    """Where a view-aggregation field of ``shape`` can live, given where it was asked to.

    The host when no MPS buffer that size can exist, ``device`` otherwise. The move is
    FastSurfer's own ``--viewagg_device cpu``: the three view networks still run on the
    device and each batch is added into the field on the host, so the result is one
    FastSurfer computes itself - it is the speed that changes, not the method."""
    import math
    dev = str(device)
    if dev.startswith("mps") and math.prod(int(s) for s in shape) * itemsize >= MPS_MAX_BUFFER_BYTES:
        return "cpu"
    return dev


#: The finest voxel size FastSurfer is asked to process at, in mm. FastSurferVINN was
#: trained and validated on 0.7-1.0 mm, and its own --vox_size help calls anything below
#: 0.7 "experimental" (Henschel et al. 2022, doi:10.1016/j.neuroimage.2022.118933). Its
#: "min" rule has a 1 mm cap and no floor, so a 0.5 mm scan ran on a 512^3 grid: a 21 GB
#: field, past one MPS buffer, and 2.7x the network work of 0.7 mm. Finer inputs are
#: raised to this (2026-09-12, fastsurfer cache_epoch 1); the labels are still restored
#: onto the input's own grid.
VOX_FLOOR_MM = 0.7


def processing_vox_size(img, conform_kwargs):
    """FastSurfer's own voxel-size choice for ``img``, raised to :data:`VOX_FLOOR_MM`.

    Returns ``(vox_size, floored_from)``: the ``vox_size`` to conform with, and FastSurfer's
    own choice in mm when the floor replaced it (else None). An input at or above the floor
    gets FastSurfer's own mode back, not the number it resolves to, so its conform is
    literally the one it always had. The choice is FastSurfer's ``conformed_vox_img_size``
    itself - the 0.95 mm snap-to-1 threshold and the rounding are its rules, not copies."""
    from FastSurferCNN.data_loader.conform import conformed_vox_img_size
    vox, _ = conformed_vox_img_size(img, conform_kwargs["vox_size"], conform_kwargs["img_size"],
                                    threshold_1mm=conform_kwargs["threshold_1mm"])
    if vox is None:                                   # 'keep' mode: no size was chosen to floor
        return conform_kwargs["vox_size"], None
    # At FastSurfer's own 4-decimal precision (vox_eps): a header's float32 0.7 is
    # 0.699999988, which a float64 comparison calls finer than the floor - a genuine
    # 0.7 mm scan would have been "floored" to 0.7 and a deviation recorded for nothing.
    chosen = round(float(np.min(vox)), 4)
    if chosen >= VOX_FLOOR_MM:
        return conform_kwargs["vox_size"], None
    return VOX_FLOOR_MM, chosen


# The three FastSurferVINN v2.0.0 checkpoints, by filename and sha256. They are fixed,
# DOI-versioned files on Zenodo - once fetched they never change, so a local cache is
# permanent. haversack fetches them from Zenodo itself rather than through FastSurfer's
# downloader (2026-09-03): FastSurfer tries b2share.fz-juelich.de first, whose server omits
# an intermediate certificate, and its download helper catches only HTTPError - the SSLError
# propagates and kills model load before the Zenodo fallback is ever tried. Zenodo's chain
# is clean, so plain stdlib urllib verifies it.
CHECKPOINTS = {
    "aparc_vinn_axial_v2.0.0.pkl":    "81ab25ccbfc432cc41fb6089d11ff2a45b5a88e659092096cb1e067269f806b8",
    "aparc_vinn_coronal_v2.0.0.pkl":  "73813957e83ec99d2c4e22f7854308916ec276d66ab7ed8ef8bf3998ba05e289",
    "aparc_vinn_sagittal_v2.0.0.pkl": "edae6262d69526fea39a019f2dfd8df75eb705f8f025b9b97166e3e17b917414",
}
CHECKPOINT_NAMES = tuple(CHECKPOINTS)
ZENODO_BASE = "https://zenodo.org/records/10390573/files"


def checkpoint_dir() -> Path:
    """Where haversack keeps the FastSurfer checkpoints: ``HAVERSACK_FASTSURFER_CHECKPOINTS``,
    else ``fastsurfer-checkpoints`` under the cache root (``HAVERSACK_CACHE_DIR``, else
    ``$XDG_CACHE_HOME/haversack``, i.e. ``~/.cache/haversack``). Permanent,
    because the files are DOI-versioned - a download happens at most once per machine.

    Both facts come from this engine's `cache_store` row in the registry, through the
    same function `cache usage` and `cache clean` ask. They were written out here as
    well until 2026-09-08, so the registry was authoritative for cache admin and not for
    the downloads - a subdirectory changed there would have moved where `cache clean`
    swept without moving what it was sweeping."""
    from ..cache_admin import engine_store_dir
    return engine_store_dir("fastsurfer")


def ensure_checkpoints(directory=None, *, progress=None) -> Path:
    """Fetch any missing checkpoint from Zenodo into ``directory`` (default
    :func:`checkpoint_dir`), verify its sha256, and return the directory. A file already
    present with the right hash is left alone; a partial or wrong one is refetched. Atomic
    per file (temp + rename), so concurrent runs are safe."""
    import hashlib

    from .. import fetchlib
    from ..progress import InstallProgress

    d = Path(directory) if directory is not None else checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    say = InstallProgress.of(progress)
    missing = [(name, sha) for name, sha in CHECKPOINTS.items()
               if not ((d / name).is_file()
                       and hashlib.sha256((d / name).read_bytes()).hexdigest() == sha)]
    say.begin(len(missing))
    for i, (name, sha) in enumerate(missing):
        dest = d / name
        say.item(i, name)
        what = f"fetching {name} from Zenodo"
        say(what)
        tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
        with fetchlib.open(f"{ZENODO_BASE}/{name}?download=1", timeout=600) as r, \
                open(tmp, "wb") as f:
            total, done = fetchlib.content_length(r), 0
            say.download(done, total, what)
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                say.download(done, total, what)
            say.download(done, done, what)
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != sha:
            tmp.unlink(missing_ok=True)
            from ..errors import ResourceError
            raise ResourceError(f"checkpoint {name}: sha256 {got} != expected {sha}")
        tmp.replace(dest)              # rename() refuses an existing target on Windows
        say.finished(f"{name} installed")
    return d


def checkpoint_args(directory=None) -> list:
    """``--ckpt_*`` arguments pointing FastSurfer at :func:`checkpoint_dir` (or ``directory``)."""
    d = Path(directory) if directory is not None else checkpoint_dir()
    ax, cor, sag = (str(d / n) for n in CHECKPOINT_NAMES)
    return ["--ckpt_ax", ax, "--ckpt_cor", cor, "--ckpt_sag", sag]


def _get_runner(device: str, batch_size: int, viewagg_device: str = "auto"):
    """A FastSurfer ``RunModelOnData`` (the three view models + LUT), built ONCE
    per (device, batch_size) and reused across jobs. This is the expensive,
    input-independent setup - checkpoint load, arch build, device upload - so
    caching it turns per-job model reload (dominant in the warm case) into a
    one-time cost. Defaults (checkpoint/config/LUT paths, conform knobs) are
    taken from FastSurfer's own argument parser so we track upstream, not a
    hardcoded copy. FastSurfer is imported here and nowhere else."""
    key = (device, int(batch_size), viewagg_device)
    runner = _RUNNERS.get(key)
    if runner is not None:
        return runner
    from FastSurferCNN import run_prediction as rp

    ensure_checkpoints()                              # our own Zenodo fetch, sha256-verified
    args = rp.make_parser().parse_args(
        ["--t1", "x", "--sd", "x", "--device", device,
         "--batch_size", str(int(batch_size)), "--viewagg_device", viewagg_device,
         *checkpoint_args()])
    # Mirror main()'s constructor EXACTLY, every knob from the parsed args - the
    # init defaults are NOT the CLI defaults (e.g. image_size init=True but CLI
    # "auto"), and a wrong conform knob yields a degenerate segmentation.
    runner = rp.RunModelOnData(
        lut=args.lut, ckpt_ax=args.ckpt_ax, ckpt_sag=args.ckpt_sag,
        ckpt_cor=args.ckpt_cor, cfg_ax=args.cfg_ax, cfg_sag=args.cfg_sag,
        cfg_cor=args.cfg_cor, device=args.device, viewagg_device=args.viewagg_device,
        threads=args.threads, batch_size=args.batch_size, vox_size=args.vox_size,
        orientation=args.orientation, image_size=args.image_size,
        async_io=args.async_io, conform_to_1mm_threshold=args.conform_to_1mm_threshold)
    _RUNNERS[key] = runner
    return runner


def _capture_logits(t1_sitk, device: str, batch_size: int = 8, on_gpu: bool = True,
                    viewagg_device: str = "auto"):
    """Segment an in-memory SimpleITK image with a CACHED FastSurfer model,
    capturing the pre-argmax logit field in the conformed-orig frame. Drives
    conform + get_prediction directly (no ``rp.main``, no SubjectList): the input
    goes through nibabel in memory, conform runs on it, and NOTHING is written -
    the conformed orig, the segfile, brainmask/aseg/CC that ``main`` would emit
    are all skipped (we only need the logits).

    We conform to LIA (the default), so the LIA inference frame IS the conformed
    frame and ``n2l.inverse`` is the identity - the orientation change is deferred
    into the restore's physical-space resample. When ``on_gpu`` and that reorder
    is identity, the K-channel field stays on the GPU (returned as a torch tensor
    ``(K, Zs, Ys, Xs)``) - no host<->device copy, no per-channel reorder. Otherwise,
    with the identity reorder, it comes back to the host in the same ``(K, Zs, Ys, Xs)``
    shape, fp16, as a view of one ``(Zs, Ys, Xs, K)`` copy; only a non-identity reorder
    still returns numpy fp32 ``(Zs, Ys, Xs, K)``. Returns
    (logits, conf_orig_sitk, fs_labels_zyx, class_labels, capture) - the last says what
    the capture actually did, so the caller can record it: ``field_device`` (where the
    view-aggregation field lived, which :func:`field_device` may have moved to the host),
    ``vox_mm`` (the processing voxel size) and ``floored_from`` (FastSurfer's own choice
    when :data:`VOX_FLOOR_MM` replaced it, else None)."""
    import torch
    from FastSurferCNN.data_loader.conform import Reorientation, conform, is_conform
    import FastSurferCNN.data_loader.data_utils as du

    r = _get_runner(device, batch_size, viewagg_device)
    orig = sitk_to_nibabel(t1_sitk)                   # the SITK -> nibabel bridge
    orig_data = np.asanyarray(orig.dataobj)
    # conform in memory, no file writes (conform_and_save_orig minus the IO);
    # reuse FastSurfer's own conform kwargs so we match its trained-input contract
    ck = r._RunModelOnData__conform_kwargs()          # name-mangled: FastSurfer's exact knobs
    vox, floored_from = processing_vox_size(orig, ck)   # FastSurfer's choice, floored at 0.7
    ck = r._RunModelOnData__conform_kwargs(vox_size=vox)
    if not is_conform(orig, **r._RunModelOnData__conform_kwargs(vox_size=vox, verbose=False)):
        orig = conform(orig, **ck)
        orig_data = np.asanyarray(orig.dataobj)

    zoom = np.asarray(orig.header.get_zooms())
    n2l = Reorientation.from_target_orientation(
        orig.affine, "soft LIA", orig_data.shape, zoom)
    orig_in_lia = n2l(orig_data, order=1)
    shape = orig_in_lia.shape + (r.get_num_classes(),)
    field_dev = field_device(r.viewagg_device, shape)   # the host if MPS cannot hold it
    pred_prob = torch.zeros(shape, device=field_dev,
                            dtype=torch.float16, requires_grad=False)
    for plane, model in r.models.items():
        r.set_model(plane)
        pred_prob = model.run(pred_prob, "image", orig_in_lia,
                              n2l.reorder_axes(zoom), out=pred_prob)

    inv = n2l.inverse                                 # LIA -> conformed-orig frame
    identity = inv.is_identity()                      # true when conformed to LIA (default)

    # FastSurfer's own labels, for the self-check (single channel, cheap to move)
    pred_classes = inv(torch.argmax(pred_prob, 3), order=0)
    pred_classes = du.map_label2aparc_aseg(pred_classes, r.labels)
    fs_labels = du.split_cortex_labels(pred_classes.cpu().numpy())      # (X,Y,Z)
    fs_labels_zyx = np.ascontiguousarray(np.transpose(fs_labels, (2, 1, 0)))
    conf_orig = nibabel_to_sitk(orig)                 # conformed geometry, no file round-trip

    if on_gpu and identity:
        # keep the field on the device; (X,Y,Z,K) -> (K,Z,Y,X) for the resampler.
        # The orientation change is left to the restore's affine (no reorder here).
        # A view, not the contiguous copy it was until 2026-09-12: that copy was a second
        # whole field on the card beside the first (8.6 GiB each at 384^3, an A10G has 22),
        # and the grouped restore and the encoder each widen only what they read.
        logits = pred_prob.permute(3, 2, 1, 0)                         # (K,Zs,Ys,Xs) view, device
    elif identity:
        # The host path keeps the field in fp16. Until 2026-09-11 it was widened to fp32 and
        # then transposed contiguous - two fp32 copies beside the fp16 original, ~105 GB at
        # peak for a 512^3 field, and emit_probabilities made a third. Now one fp16 copy into
        # (Z,Y,X,K): the same transpose as before at half the bytes, so a channel read by the
        # CPU restore strides by K exactly as it did, and an encoder slab of Z-planes is one
        # contiguous block. Widening is exact and each consumer widens only what it reads, so
        # the bytes out are unchanged. A pure view of the (X,Y,Z,K) buffer would save this
        # copy too, but would make every one of the K channel reads a scattered 3D transpose.
        logits = pred_prob.cpu().permute(2, 1, 0, 3).contiguous().permute(3, 0, 1, 2)
    else:                                             # generic reorder (rare: non-LIA conform)
        pp = pred_prob.float().cpu().numpy()          # (X, Y, Z, K) nibabel order
        logit_conf = np.empty(inv(pp[..., 0], order=1).shape + (pp.shape[3],), np.float32)
        for k in range(pp.shape[3]):
            logit_conf[..., k] = np.asarray(inv(pp[..., k], order=1))
        logits = np.ascontiguousarray(np.transpose(logit_conf, (2, 1, 0, 3)))   # (Z,Y,X,K)
    del pred_prob
    capture = {"field_device": field_dev, "floored_from": floored_from,
               "vox_mm": round(float(np.min(zoom)), 4)}   # the header's float32, at vox_eps
    return logits, conf_orig, fs_labels_zyx, r.labels, capture


def _fs_version() -> str:
    try:
        import FastSurferCNN
        return getattr(FastSurferCNN, "__version__", "unknown")
    except Exception:
        return "unknown"


def emit_probabilities(spec, logits, source_ref, target_ref, class_labels) -> None:
    """Hand FastSurfer's pre-argmax logit field to a ranked sink.

    FastSurfer holds a 79-class softmax on its conformed 1 mm grid, which is structurally an
    nnU-Net part with a different K - so the encoder applies unchanged and this only has to
    supply the geometry a reader needs to redo the restore. Without both grids the arrays
    are a picture of the conformed grid and nothing else.

    ``logits`` arrives from ``_capture_logits`` as a torch tensor ``(K, Z, Y, X)`` - on the
    device when the GPU restore is in play, an fp16 view on the host otherwise - or, only
    after a non-identity reorder, as ``(Z, Y, X, K)`` numpy. The encoder wants the tensor,
    and promotes to fp32 per slab itself - so do not cast or copy the whole field here, which
    would materialize a second copy of the largest array in the run.
    """
    import torch

    from .. import __version__, ranked
    from .geometry import grid_record

    lg = logits if isinstance(logits, torch.Tensor) else torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(np.asarray(logits), 3, 0)))
    # which softmax produced these logits - see the nnU-Net path for why a reader needs it.
    # FastSurfer bakes its checkpoints into the image, so the identity is a fixed version
    # rather than an install sidecar.
    ident = weights_installed()
    ranked.emit(
        spec, "brain", lg,
        softmax={"engine": "fastsurfer", "classes": int(lg.shape[0]),
                 "weights": ident[0].get("id", "fastsurfer") if ident else "fastsurfer",
                 "version": ident[0].get("version") if ident else None},
        part="brain", engine="fastsurfer", haversack=__version__,
        labels=[int(v) for v in np.asarray(class_labels).reshape(-1)],
        labels_note="channel -> aparc+aseg id; segment() additionally applies "
                    "split_cortex_labels, which is spatial and not expressible as a LUT",
        source_grid=grid_record(source_ref),     # the conformed grid the logits live on
        target_grid=grid_record(target_ref))     # the input grid they restore onto


def segment(t1_input, *, out_dir=None, device: str = "cuda", batch_size: int = 8,
            logit_grade: bool = True, self_check: bool = True, restore: str = "auto",
            probabilities=None, viewagg_device: str = "auto"):
    """Segment a T1 with FastSurfer and return an :class:`haversack.result.Segmentation`
    on the input's grid.

    ``t1_input`` is a SimpleITK image (the memory-in path - what haversack's reader
    / read-ahead produces) or a path (read with haversack's ``io.read_image`` so its
    IPP/affine geometry fixes apply). Either way the data reaches FastSurfer
    through nibabel in memory; nothing is written to disk (``out_dir`` is accepted
    for call-site compatibility and unused - the engine writes no temp files).

    ``logit_grade`` restores the captured logit field to the input grid and
    argmaxes after (sub-voxel boundaries); ``False`` falls back to a
    nearest-neighbor resample of FastSurfer's own labelmap. ``self_check``
    verifies (loudly) that argmax at the conformed grid reproduces FastSurfer's
    own labels before trusting the restore.

    ``probabilities`` is a :class:`~haversack.ranked.RankedSpec`: when given, the captured
    79-class field is encoded and handed to its sink before the restore consumes it -
    the same hook the nnU-Net path uses, so the stored form is identical in kind. It runs
    *after* ``self_check``, so a distribution is never stored that we have just failed to
    reproduce FastSurfer's own labels from. Off by default; the field is otherwise
    discarded after the restore.

    ``restore`` selects the logit-restore backend: ``"gpu"`` (batched
    ``grid_sample``, fast, needs the whole field on the device), ``"cpu"``
    (per-channel SimpleITK, slow but memory-frugal - for limited local hosts), or
    ``"auto"`` (GPU on a CUDA device, CPU otherwise). The two are numerically
    equivalent (same physical-space mapping and half-pixel convention).
    """
    import SimpleITK as sitk
    import FastSurferCNN.data_loader.data_utils as du
    import torch

    from ..grid import Grid
    from ..result import Segmentation
    from ..values import LabelSchema

    import time

    if isinstance(t1_input, sitk.Image):
        t1_img = t1_input                             # memory-in (read-ahead / caller)
    else:
        from .. import io
        t1_img = io.read_image(str(t1_input))         # path: geometry-correct read
    use_gpu = restore == "gpu" or (restore == "auto" and str(device).startswith("cuda"))
    timings: dict[str, float] = {}
    _t = time.perf_counter()
    logits, conf_orig, fs_labels_zyx, class_labels, capture = _capture_logits(
        t1_img, device, batch_size, on_gpu=use_gpu, viewagg_device=viewagg_device)
    field_dev = capture["field_device"]
    timings["capture"] = time.perf_counter() - _t     # conform + VINN inference (model cached)

    def to_fs(idx_zyx):
        m = du.map_label2aparc_aseg(torch.from_numpy(idx_zyx.astype(np.int64)),
                                    class_labels)
        return du.split_cortex_labels(m.cpu().numpy()).astype(np.int32)

    def _source_argmax(lg):                           # argmax over classes -> (Zs,Ys,Xs)
        if isinstance(lg, torch.Tensor):              # (K,Zs,Ys,Xs) on device
            return lg.argmax(dim=0).to(torch.int32).cpu().numpy()
        return np.argmax(lg, axis=3).astype(np.int32)  # (Zs,Ys,Xs,K) numpy

    if self_check:                                    # argmax at conformed == FastSurfer's labels
        my = to_fs(_source_argmax(logits))
        agree = float((my == fs_labels_zyx).mean())
        if agree < 0.999:
            raise RuntimeError(f"FastSurfer logit self-check failed: {agree:.4%} "
                               "of conformed voxels match FastSurfer's own labels")

    if probabilities is not None:
        # After the self-check on purpose: never store a distribution we have just been
        # unable to reproduce FastSurfer's own labels from.
        _t = time.perf_counter()
        emit_probabilities(probabilities, logits, conf_orig, t1_img, class_labels)
        timings["probabilities"] = time.perf_counter() - _t

    def _extent(im):
        import numpy as _np
        lo = _np.array(im.TransformIndexToPhysicalPoint((0, 0, 0)))
        sz = im.GetSize()
        hi = _np.array(im.TransformIndexToPhysicalPoint((sz[0]-1, sz[1]-1, sz[2]-1)))
        return _np.minimum(lo, hi), _np.maximum(lo, hi)
    clo, chi = _extent(conf_orig); tlo, thi = _extent(t1_img)
    print(f"[fastsurfer] logits={tuple(logits.shape)} conf={conf_orig.GetSize()} "
          f"t1={t1_img.GetSize()} restore={'gpu' if use_gpu else 'cpu'} "
          f"conf_ext={clo.round(1)}..{chi.round(1)} "
          f"t1_ext={tlo.round(1)}..{thi.round(1)}", flush=True)

    if logit_grade:
        _t = time.perf_counter()
        if use_gpu:
            idx_native = restore_logits_gpu(logits, conf_orig, t1_img, device)
        else:
            idx_native = restore_logits(logits, conf_orig, t1_img)
        timings["restore"] = time.perf_counter() - _t   # physical-space resample + argmax
        labels_arr = to_fs(idx_native)                # (Z,Y,X) FreeSurfer ids on input grid
        nfg = int((labels_arr > 0).sum())
        print(f"[fastsurfer] restored foreground voxels={nfg}/{labels_arr.size}", flush=True)
        if nfg == 0:
            raise RuntimeError(
                f"logit-grade restore is empty: logits={tuple(logits.shape)}, "
                f"conf={conf_orig.GetSize()} ext {clo.round(1)}..{chi.round(1)}, "
                f"t1={t1_img.GetSize()} ext {tlo.round(1)}..{thi.round(1)}; "
                "conf/t1 physical extents likely do not overlap")
    else:
        conf_seg = sitk.GetImageFromArray(fs_labels_zyx.astype(np.uint16))
        conf_seg.CopyInformation(conf_orig)
        labels_arr = sitk.GetArrayFromImage(
            sitk.Resample(conf_seg, t1_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)).astype(np.int32)

    out_img = sitk.GetImageFromArray(labels_arr.astype(np.uint16))
    out_img.CopyInformation(t1_img)                   # input grid + orientation

    lut = load_lut()
    present = sorted(int(v) for v in np.unique(labels_arr) if v)
    names = {v: lut.get(v, {}).get("name", f"label_{v}") for v in present}
    grid = Grid(shape=tuple(int(s) for s in labels_arr.shape),
                spacing=tuple(float(s) for s in reversed(t1_img.GetSpacing())),
                origin=tuple(float(o) for o in reversed(t1_img.GetOrigin())))
    prov = {"engine": "fastsurfer", "fastsurfer_version": _fs_version(),
            "network": "FastSurferVINN (2.5D view aggregation)",
            "restore": (f"logit-grade (physical-space, {'gpu' if use_gpu else 'cpu'})"
                        if logit_grade else "label-nn"),
            "self_check": "reproduces FastSurfer labels at conformed grid" if self_check else "skipped",
            "device": device, "view_aggregation_device": field_dev,
            "processing_vox_mm": capture["vox_mm"], "deviations": []}
    if capture["floored_from"] is not None:
        from ..result import deviation
        prov["deviations"].append(deviation(
            "processing voxel size", f"{capture['floored_from']:g} mm (FastSurfer's 'min')",
            f"{capture['vox_mm']:g} mm",
            "FastSurferVINN is validated at 0.7-1.0 mm and calls finer experimental; the "
            "labels are still restored onto the input's own grid"))
    if field_dev == "cpu" and not str(device).startswith("cpu"):
        from ..result import deviation
        if viewagg_device == "cpu":                   # placed there by policy (local_viewagg)
            why = "the 79-class aggregation field would not fit the GPU's memory pool"
        else:                                         # moved there by field_device, for size
            gb = int(np.prod(logits.shape)) * 2 / 2**30
            why = (f"the 79-class aggregation field ({gb:.1f} GiB) is larger than one MPS "
                   f"buffer can be ({MPS_MAX_BUFFER_BYTES / 2**30:.0f} GiB)")
        prov["deviations"].append(deviation(
            "device (view-aggregation field)", device, "cpu",
            why + "; the three view networks still ran on the requested device"))
    seg = Segmentation(labels=out_img, schema=LabelSchema(names=names),
                       grid=grid, spec=None, timings=timings, provenance=prov)
    return seg


def run_local(image, *, device="auto", batch_size="auto", progress=None, cancel=None,
              probabilities=None, **_policy):
    """The in-process entry point (:attr:`Engine.compute`): what ``haversack segment`` and a
    local ``haversack serve`` call for ``fastsurfer:brain``. Resolves ``device`` the way the
    nnU-Net path does, places the view-aggregation field by :func:`local_viewagg`, reports
    one stage, and honors the cancel token before the model is built. nnU-Net policy keys
    that mean nothing here (grid, interp, ...) are accepted and ignored."""
    from ..progress import Reporter
    from ..resample import resolve_device
    report = Reporter.of(progress, cancel=cancel)
    dev = resolve_device(device).type
    bs = 8 if batch_size in (None, "auto") else int(batch_size)
    viewagg = local_viewagg(dev)
    report.check()
    report.stage("predict", f"fastsurfer:brain on {dev} (view aggregation on {viewagg})")
    seg = segment(image, device=dev, batch_size=bs, viewagg_device=viewagg,
                  probabilities=probabilities)
    for d in seg.provenance.get("deviations", ()):
        report.stage("note", f"{d['what']}: asked {d['requested']}, ran {d['effective']} - {d['why']}")
    report.check()
    return seg
