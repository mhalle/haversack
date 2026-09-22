"""segment(): the torch pipeline - task -> parts -> logits -> labels on the chosen grid."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from .envelope import (Envelope, at_least, body_mask, body_threshold, envelope_margin,
                       envelope_of, margin_in_voxels, worth_cropping)
from .frame import Frame
from .mapping import Mapping
from . import backends
from .restore import to_labels
from .network import TorchModel, available_folds
from .tasks import (ModelNotFound, TaskCatalog, TaskSpec, _resolve_spec,
                    _uses_nnunet_preprocessing, resolve_model_folder)
from .cache import ModelCache
from .result import Segmentation
from .progress import Reporter
from .weights import as_store
from .cache import ModelCache
from .values import LabelSchema
from .preprocess import normalization_fingerprint, normalize_for, to_model_grid


def _version() -> str:
    from . import __version__                    # deferred: __init__ imports this module
    return __version__


def _warm_restore_kernel(device: str) -> None:
    """Compile the fused CUDA kernel while the network runs, instead of inside the first restore."""
    if not str(device).startswith("cuda"):
        return
    from .backends import triton_gpu
    if not triton_gpu.available():
        return
    threading.Thread(target=triton_gpu.warmup, daemon=True).start()


def upstream_crop_box(labels, classes, margin_mm: float, spacing_zyx):
    """The box a TotalSegmentator cascade's final stage runs on: ``(lo, hi)`` source indices, end
    exclusive, or None when none of ``classes`` is present - upstream's ``crop_to_mask`` and
    ``get_bbox_from_mask`` (cropping.py), restated on the canonical input grid.

    Upstream takes the bounding box of the crop classes in the ORIGINAL image's index space and
    widens it by ``int(mm / zoom)`` voxels per axis - truncated, not rounded - with the upper
    end one past the last voxel, then clips to the image. The canonical grid is that index space
    with its axes permuted and flipped, so the same box, spacing axis for axis, is the same set
    of voxels (2026-09-22). An empty mask is upstream's "Crop is empty. Returning empty
    segmentation" - the caller returns background, never the whole volume.
    """
    mask = np.isin(np.asarray(labels), [int(c) for c in classes])
    if not mask.any():
        return None
    add = (float(margin_mm) / np.asarray(spacing_zyx, dtype=np.float64)).astype(int)
    idx = np.nonzero(mask)
    lo = tuple(max(0, int(i.min()) - int(a)) for i, a in zip(idx, add))
    hi = tuple(min(int(n), int(i.max()) + 1 + int(a)) for i, a, n in zip(idx, add, mask.shape))
    return lo, hi


def _lut(K: int, remap: dict | None) -> np.ndarray:
    lut = np.arange(K, dtype=np.int64)
    if remap:
        lut[:] = 0
        for local, global_ in remap.items():
            if not 0 <= int(local) < K:
                raise ModelNotFound(
                    f"catalog remap references local label {local} but the model "
                    f"emits {K} channels - the task catalog does not match the "
                    "installed weights (stale class map?)")
            lut[int(local)] = int(global_)
    return lut


def canonical_orientation_for(spec, store, *, configuration: str | None = None) -> str | None:
    """The orientation code the model must see, or None to keep the stored axis order.

    Orientation follows the model's own reader. TS canonicalizes to RAS; nnU-Net's default
    ``SimpleITKIO``/``NibabelIO`` do NOT - only the ``*WithReorient`` variants do - so a native
    model expects its acquisition orientation, and reorienting it anyway mirrors left/right.
    A spec may override the declared reader when the model's *packaging* reorients around it
    (MRSegmentator's reader forces LPS on top of plans that say ``SimpleITKIO``); that
    override is the ecosystem's statement about how the model was served, so it wins.
    Factored out so the decision is testable without a network.
    """
    from . import io as nio
    if spec.orientation is not None:
        return str(spec.orientation)
    if _uses_nnunet_preprocessing(spec) and spec.single is not None:
        folder = store.resolve(spec.single, configuration=configuration,
                               **spec.model_choice(spec.single))
        return nio.CANONICAL if nio.reader_reorients(folder) else None
    return nio.CANONICAL


def _attribution(spec, catalog=None) -> dict:
    """The license and citations a result carries, from the catalog's own record of the task.

    Until 2026-09-20 this was handed only the ecosystem's name and the modality, so a license
    the MANIFEST states for one task never reached the result: `describe()` named it, and the
    seg.nrrd header fell back to the ecosystem's. Every shipped manifest agreed with its
    ecosystem, so nothing was misstated - but a catalog whose tasks differ in license would
    have been, in the one copy that travels with a download. The engine path
    (`Segmenter._run_engine`) always passed the catalog's record; this is the same rule.
    `info()` never downloads. A TaskSpec or a model folder is in no catalog, and keeps the
    grammar's answer."""
    from . import attribution
    info = None
    if hasattr(catalog, "info"):
        try:
            info = catalog.info(spec.name)
        except Exception:                        # noqa: BLE001 - a spec or a name it does not know
            info = None
    return attribution.provenance_block(
        spec.name, {**(info or {}),
                    "ecosystem": (info or {}).get("ecosystem")
                                 or (spec.name.partition(":")[0] if ":" in spec.name else ""),
                    # the spec's, as describe() does: it decides which papers apply
                    "modality": spec.modality})


def segment(image, task: str, *, catalog=None, weights=None, device: str = "auto", dtype: str = "fp16",
            grid="input", interp="linear", outside: str = "background", convention: str = "auto",
            folds=(0,), accumulate: str = "auto", resampling_order: int = 3, batch_size="auto",
            envelope_mm: float | None = None, configuration: str | None = None,
            allow_transpose: bool = False,
            probabilities=None,
            models=None, cancel=None, progress=None):
    """Segment an image with a task from the toolkit's catalog.

    ``image`` is a path to anything SimpleITK reads - NIfTI, NRRD, MetaImage, a DICOM series
    directory - or a SimpleITK image the caller already holds.

    ``task`` is a catalog name, a ``TaskSpec``, or a path to a stock nnU-Net result folder
    (``.../Dataset<id>_<name>/<trainer>__<plans>__<config>/``, or the dataset folder - then
    ``configuration`` picks among 2d / 3d_lowres / 3d_fullres, preferring 3d_fullres).

    ``convention`` defaults to ``"auto"``: ``"center"`` (skimage half-pixel, plus crop-to-nonzero)
    for nnU-Net-native models, ``"corner"`` (TotalSegmentator's ``change_spacing``, no crop) for
    the TS catalog - each model's own training-time preprocessing.

    ``device`` defaults to ``"auto"``: CUDA, then MPS, then CPU.

    ``resampling_order`` is the spline order of the forward resample; 3 (cubic) matches what
    nnU-Net trained the models with - TotalSegmentator v2.18 defaults to 1 for speed, which is
    a mild train/test mismatch that grows with the downsampling factor.

    ``envelope_mm`` restricts inference to the patient's bounding box (air removed, largest
    connected component, plus this margin in mm): up to half the patches on a CT with air
    around the body, and NOT the same labels - cropping re-tiles the sliding window
    and the labels move with it (see :func:`~haversack.envelope.worth_cropping`), so it is off
    by default since 2026-09-11, when it was 20 mm. The air cut is the CT -500 HU threshold for
    CT models and a data-driven (Otsu) split for per-image-normalized MRI. ``None`` or ``0``
    runs the full volume, as ``--envelope 0`` does on the command line (before 2026-09-11, 0
    here meant a crop flush to the skin; see :func:`~haversack.envelope.envelope_margin`).

    ``batch_size`` is patches per forward pass: an int, or ``"auto"`` - 1 on Apple silicon
    (measured fastest), 4 on CUDA when the measured working set says it fits (18 % faster
    steady-state on an A10); it only applies when the accumulator is on the device.

    ``probabilities`` is a :class:`~haversack.ranked.RankedSpec`; when given, each part's output
    distribution is encoded from its logits (before the restore frees them) and handed to the
    spec's sink as a :class:`~haversack.ranked.RankedCode` on that model's own grid. Off by
    default: labels are a fraction of the size and most callers want only those.

    ``cancel`` is a :class:`~haversack.progress.CancelToken`; the run stops at the next patch
    boundary. ``progress`` is called with a :class:`~haversack.progress.Progress` snapshot (which
    prints readably, so a ``lambda p: print(p)`` callback works).

    ``models`` is a :class:`~haversack.cache.ModelCache`; pass one with ``capacity>=1`` (or use
    :class:`~haversack.segmenter.Segmenter`) to keep models warm between calls instead of rebuilding
    them every time - the difference between a script and a server.

    ``accumulate`` picks where the sliding-window accumulator lives: ``"auto"`` (from the
    device's free memory), ``"device"`` (fastest, needs headroom), ``"host"``.

    Returns a :class:`~haversack.result.Segmentation`: ``.labels`` is the label volume as a SimpleITK
    image in the *input's* orientation on the requested grid (``"input"`` = the input grid, a number =
    isotropic at that spacing, a ``Grid`` = as given). It also carries ``.array``, ``.mask(name)``,
    ``.present()``, ``.volumes_ml()``, ``.save(path)``, ``.timings`` and ``.provenance`` - what
    models, folds, device and preprocessing policy actually ran.
    Multi-model tasks composite at the label level in part order (later parts win).
    """
    from . import io as nio
    from .job import device_lock

    from .resample import resolve_device
    device = str(resolve_device(device))                  # "auto" -> cuda / mps / cpu, once
    envelope_mm = envelope_margin(envelope_mm)            # 0 -> None, before anything reads it
    report = Reporter.of(progress, cancel=cancel)
    _warm_restore_kernel(device)
    T: dict[str, float] = {}
    t_start = time.perf_counter()                         # `total` covers resolution and install too
    lock = device_lock(device)                            # reentrant: a Job already holds it
    if catalog is None:
        from .ecosystems import EcosystemCatalog
        catalog = EcosystemCatalog(root=as_store(weights).root)
    models = models if models is not None else ModelCache()   # no caching unless asked

    def resolve(name):
        """The task's spec, with an install on first use timed and reported as its own step.
        It used to run inside the `read+canonical` timer, unreported: on a fresh Modal
        container, 29 s of a `cads:headneck` run's read+canonical was its 760 MB weights
        (2026-09-12), where the read itself takes about half a second."""
        installed = getattr(catalog, "installed", None)
        if not (isinstance(name, str) and callable(installed)
                and not Path(name).expanduser().is_dir() and not installed(name)):
            return _resolve_spec(name, catalog)
        t = time.perf_counter()
        report.stage("weights", f"installing {name}")
        # the install reports INSIDE this run's position: handed `report` itself, the installer
        # rewrote its part, part count and fraction (Reporter.nested says what that did)
        spc = _resolve_spec(name, catalog, progress=report.nested("weights"))
        T[f"weights:{spc.name}"] = time.perf_counter() - t
        return spc

    spec = resolve(task)
    # nnU-Net-native models were trained on their own preprocessing: skimage's half-pixel
    # ("center") resample and crop-to-nonzero. TS bypasses both (corner-aligned change_spacing,
    # no crop). Getting this backwards is a silent geometry error, so it follows the task's
    # lineage unless the caller is explicit. See docs/resampler-parity-finding.md.
    nnunet_preproc = _uses_nnunet_preprocessing(spec)
    store = as_store(weights, layout="nnunetv2" if nnunet_preproc else "ts")
    if convention == "auto":
        convention = "center" if nnunet_preproc else "corner"
    crop_nonzero = nnunet_preproc
    canonical = canonical_orientation_for(spec, store, configuration=configuration)
    reorient = canonical is not None
    schema = LabelSchema(names={int(k): str(v) for k, v in spec.label_map.items()})
    prov = {"task": spec.name, "lineage": spec.lineage, "device": device, "dtype": dtype,
            "accumulate": accumulate, "deviations": [],
            "convention": convention, "reoriented_to_ras": canonical == nio.CANONICAL,
            "canonical_orientation": canonical, "interp": interp,
            "envelope_mm": envelope_mm, "resampling_order": resampling_order, "models": [],
            "weights_store": store.describe(),
            "haversack": _version(),
            # which license governs this output and what to cite for it - the
            # identifiers only; `describe()` has the full record
            "attribution": _attribution(spec, catalog)}

    t0 = time.perf_counter()                              # the read, and nothing before it
    report.stage("read", Path(image).name if isinstance(image, (str, Path)) else "in-memory image")
    if isinstance(image, (str, Path)):
        data_zyx, geometry, orientation = nio.read(image, reorient=reorient,
                                                   target=canonical or nio.CANONICAL)
    else:                                        # a SimpleITK image the caller already holds
        import SimpleITK as sitk
        orientation = nio.orientation_of(image)
        if reorient:
            image = sitk.DICOMOrient(image, canonical)
        data_zyx, geometry = sitk.GetArrayFromImage(image), nio.geometry_of(image)
    T["read+canonical"] = time.perf_counter() - t0

    labels = None
    out_grid = None
    frame: Frame | None = None
    cached = {}                                   # resample key -> ResampledGrid (NOT normalized)
    identity = {}                                 # weights id -> what actually ran, for the store

    def load(wid, spc):
        # the stage's own spec states which model folder a shared dataset means
        folder = store.resolve(wid, configuration=configuration, **spc.model_choice(wid))
        # a task's stated tile step (ts.v3: upstream's 0.8); stated nowhere else, so every
        # other task's call - and its provenance - is exactly what it was
        step = {} if spc.step_size is None else {"step_size": spc.step_size}
        m = models.get(folder, folds=folds, device=device, dtype=dtype,
                       accumulate=accumulate, batch_size=batch_size,
                       allow_transpose=allow_transpose, **step)
        # the folder name does NOT identify the weights version - Dataset297 ships as both
        # v2.0.0 and v2.0.4 and both unpack to the same name - so read what fetch_one recorded
        from .weights_fetch import installed_version
        rec = installed_version(folder) or {}
        identity[str(wid)] = {"weights": str(wid), "folder": folder.name,
                              "version": rec.get("tag", "unknown"),
                              "sha256": rec.get("sha256"), "classes": int(m.K)}
        prov["models"].append({"weights": str(wid), "folder": folder.name,
                               "version": rec.get("tag", "unknown"), "sha256": rec.get("sha256"),
                               "folds": list(available_folds(folder, folds)), "K": m.K,
                               "spacing": tuple(round(v, 4) for v in m.spacing_zyx),
                               **step,
                               **({"transpose_forward": list(m.transpose_forward),
                                   "transpose_validated": False}
                                  if m.transpose_forward != (0, 1, 2) else {})})
        return m

    def model_frame(model, box=None):
        """This model's network input, sharing the resample with other models at its spacing
        and, in a cascade's final stage, its crop box (``box``: see :func:`upstream_crop_box`).

        Only the crop+resample is cached. Normalization is nnU-Net's, and it is PER MODEL - each
        reads its own dataset's foreground statistics - so sharing a normalized array between the
        parts of a multi-model task silently runs parts 2..N on part 1's statistics. Caching the
        normalization-free grid and normalizing per model keeps the one resample per spacing that
        the cache is for, without that.
        """
        key = (tuple(model.spacing_zyx), convention, resampling_order, crop_nonzero, str(device), box)
        if key not in cached:
            cached[key] = to_model_grid(data_zyx, geometry, model.spacing_zyx, convention=convention,
                                        device=device, order=resampling_order,
                                        original_orientation=orientation, crop_to_nonzero=crop_nonzero,
                                        box=box)
        grid = cached[key]
        return normalize_for(grid, model), grid.frame

    def crop_on_model_grid(model, x, frame, *, use_body):
        """A voxel box on this model's grid: the body envelope, when asked for. None means run
        the whole grid. A cascade's crop is not this: it is applied to the source before the
        resample (``model_frame(model, box)``), as upstream applies it (2026-09-22)."""
        shape = tuple(int(s) for s in x.shape[1:])
        start = [0, 0, 0]
        stop = list(shape)
        if use_body and envelope_mm is not None:
            xnp = x[0].numpy()
            thr = body_threshold(xnp, normalization_schemes=model.normalization_schemes,
                                 intensity_properties=model.intensity_properties(0))
            e = envelope_of(body_mask(xnp, threshold=thr),
                            margin_voxels=margin_in_voxels(envelope_mm, model.spacing_zyx))
            start = [max(a, b) for a, b in zip(start, e.start)]
            stop = [min(a, b) for a, b in zip(stop, e.stop)]
        start = [max(0, v) for v in start]
        stop = [min(n, v) for n, v in zip(shape, stop)]
        if any(b <= a for a, b in zip(start, stop)):          # empty -> fall back to whole grid
            return Envelope((0, 0, 0), shape, shape)
        env = Envelope(tuple(start), tuple(stop), shape)
        if not env.is_whole():
            # never narrower than the patch: the window would pad the rest with normalized 0,
            # the model's mean tissue, where the image has air (at_least says why). The patch is
            # in the network's axis order; network axis j is model axis transpose_forward[j].
            # A 2D model's patch covers the last two network axes and nnU-Net runs every slice
            # of the first through it, so that axis asks for 1 voxel and is never grown.
            net_patch = (1,) * (3 - len(model.patch)) + tuple(int(p) for p in model.patch)
            patch = [0, 0, 0]
            for j, a in enumerate(model.transpose_forward):
                patch[a] = net_patch[j]
            env = at_least(env, patch)
            # A crop re-tiles the window and the labels move with the tiles, so it has to buy
            # network time to be worth it - counted in the tiles the network runs, not in the
            # box's volume: at half-patch steps a crop can remove a third of the volume and
            # still need every tile (worth_cropping says what that cost).
            return worth_cropping(env, saving=1.0 - model.tiles(env.extent) / model.tiles(shape))
        return env

    def emit_probabilities(model, logits, frame, env, *, lut, part, weights=None):
        """Encode this part's output distribution while the logits are still here.

        Between the network and the restore is the only moment they exist, so this costs
        one pass and no recomputation. The code is on the model's own (envelope-cropped)
        grid, which is where the distribution lives - restoring it to the output grid first
        would inflate it and bake in one interpolation choice.

        Which is exactly why ``frame`` and ``envelope`` go with it: argmax after
        interpolation depends only on logit DIFFERENCES, and those are what is stored, so a
        reader holding the spatial extent can redo the restore onto any grid - different
        spacing, nearest instead of linear, a confidence gate - without the network. Without
        the extent the same arrays are only a picture of the grid they were computed on.
        """
        from . import ranked
        t = time.perf_counter()
        # WHICH softmax these logits came from. Margins are comparable only within one
        # normalization, and a five-model task like `total` is five of them - so a reader
        # composing across parts has to be able to tell. The folder name does not identify the
        # weights (Dataset297 ships as both v2.0.0 and v2.0.4 and unpacks to the same name), so
        # the installed version and checksum travel with it.
        soft = dict(identity.get(str(weights), {"weights": str(weights)}))
        soft["engine"] = "nnunetv2"
        ranked.emit(
            probabilities, part, logits, softmax=soft,
            part=part, task=spec.name, haversack=_version(), engine="nnunetv2",
            spacing_zyx=[float(v) for v in model.spacing_zyx],
            envelope={"start": [int(v) for v in env.start],          # a range, both ends
                      "stop": [int(v) for v in env.stop]},
            model_grid=[int(v) for v in env.shape],
            labels=[int(v) for v in np.asarray(lut).reshape(-1)],   # channel -> global label
            convention=convention, reoriented_to_ras=canonical == nio.CANONICAL,
            canonical_orientation=canonical,
            input_orientation=orientation, frame=frame.to_meta())
        T[f"probabilities:{part}"] = time.perf_counter() - t

    def predict_into(model, x, frame, ogrid, env, *, lut, paint, out, part="", weights=None,
                     restore=None):
        # tripwire: `x` must carry THIS model's normalization. Several models share one resample,
        # and feeding one model's normalization to another is silent and severe - the organs
        # model's CT clip at +276 HU flattens all bone for the parts that follow it.
        stamped = getattr(x, "_haversack_normalization", None)
        if stamped != normalization_fingerprint(model):
            raise RuntimeError(
                "network input was normalized for a different model than the one consuming it "
                f"(input {stamped!r}, model {normalization_fingerprint(model)!r}). Normalization "
                "is per-model; share the resampled grid, not the normalized array.")
        crop = x[(slice(None), *env.slices)] if not env.is_whole() else x
        logits = model.predict_logits(crop, report=report).to(device)
        if probabilities is not None:
            emit_probabilities(model, logits, frame, env, lut=lut, part=part, weights=weights)
        mapping = frame.mapping(ogrid)
        if not env.is_whole():
            mapping = mapping >> Mapping((1.0, 1.0, 1.0), tuple(-float(v) for v in env.start))
        choice = backends.select("auto", logits.device, tuple(logits.shape), tuple(ogrid.shape))
        if choice.fallback:
            # the fused kernel cannot address this field, so the slower torch backend restores
            # it; before 2026-09-11 the Triton kernel's refusal failed the whole run instead
            report.stage("restore", f"torch backend: {choice.fallback}")
            from .result import deviation
            d = deviation("restore backend", "auto", "torch", choice.fallback)
            if d not in prov["deviations"]:                 # parts on one grid say it once
                prov["deviations"].append(d)
        to_labels(logits, ogrid, mapping, interp=restore or interp, outside="background", lut=lut, paint=paint,
                  out=out, backend=choice.name)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        del logits
        if device in ("cuda", "mps"):
            (torch.cuda if device == "cuda" else torch.mps).empty_cache()

    def run_single_or_union(spc, tag, *, parts=None, box=None, use_body=True, first=0,
                            out_grid=None, restore=None):
        """One model, or a union's models painted in order into one output. A cascade's final
        stage comes here too (2026-09-22) - one model, or a union like headneck_muscles' - with
        ``box``, the source box its crop stage found, which every part runs on; ``first``, its
        position in the task's progress; and, for a crop source, ``out_grid``/``restore``."""
        parts = spc.parts if parts is None else parts
        report.n_parts = max(report.n_parts, first + len(parts))
        og = None
        out = None
        fr = None
        # Timing keys: `load:<task>` for a one-model task, `load:<task>:<part>` for a union.
        # A single task's only part is named after the task, and the old unconditional suffix
        # printed `load:ts.v2:total_fast:ts.v2:total_fast`.
        sfx = "" if len(parts) == 1 else None
        for i, (wid, remap, pname) in enumerate(parts):
            key = f"{tag}{sfx if sfx is not None else ':' + pname}"
            t = time.perf_counter()
            report.enter_part(first + i, f"{pname} ({wid})" + ("" if box is None else " (cropped)"))
            model = load(wid, spc)
            T[f"load:{key}"] = time.perf_counter() - t
            t = time.perf_counter()
            x, fr = model_frame(model, box)
            env = crop_on_model_grid(model, x, fr, use_body=use_body and box is None)
            if not env.is_whole():
                report.stage("preprocess", f"envelope {env.fraction * 100:.0f} % of the model grid")
            T[f"preprocess:{key}"] = time.perf_counter() - t
            if og is None:
                og = fr.resolve_grid(grid if out_grid is None else out_grid)
                max_label = max((int(v) for v in spc.label_map), default=255)
                out = torch.zeros(og.shape, dtype=torch.uint8 if max_label <= 255 else torch.uint16, device=device)
            t = time.perf_counter()
            report.stage("predict", pname)
            predict_into(model, x, fr, og, env, lut=_lut(model.K, remap), paint=len(parts) > 1,
                         out=out, part=pname, weights=wid, restore=restore)
            where = "device" if model.accumulate_choice["on_device"] else "host"
            report.stage("restore", f"{where} accumulator")
            for m in prov["models"]:                 # the effective placement, per model
                if m["weights"] == str(wid):
                    m["accumulate"] = where
            if accumulate == "device" and where == "host":
                from .result import deviation
                prov["deviations"].append(deviation("accumulator placement", "device", "host",
                                                    model.accumulate_choice.get("why", "")))
            T[f"network:{key}"] = time.perf_counter() - t
            models.release(model)
        return out, fr, og

    def run_cascade(spc, tag, *, out_grid=None, restore=None):
        """TotalSegmentator's crop, as upstream runs it (2026-09-22). Each stage before the last
        labels the whole image (inside any box already found), restored nearest-neighbour onto
        the input grid as upstream restores its crop model; the box of its crop classes, widened
        by its margin (:func:`upstream_crop_box`), is cut out of the input BEFORE the final
        stage's resample, and the final stage runs on that cut alone - never grown to the patch,
        never dropped for saving too little - so nothing outside the box is labelled. An empty
        crop is an empty result, as upstream returns one.

        Until 2026-09-22 the crop was a speed approximation of whole-volume inference instead
        (grown by ``at_least``, collapsed by ``worth_cropping``), tuned against the model run on
        the whole volume (medseg docs/backend-decision.md, "Cascade mode-B") and never compared
        with upstream: on a neck CT headneck_bones_vessels scored mean Dice 0.738 against
        upstream, zygomatic arches labelled beyond upstream's box."""
        box = None
        stages = spc.cascade
        report.n_parts = max(report.n_parts, len(stages) - 1 + max(1, len(stages[-1].union)))
        for i, step in enumerate(stages):
            if i == len(stages) - 1:
                parts = ([(p.weights_id, dict(p.label_remap), p.name or str(p.weights_id))
                          for p in step.union] if step.union else [(step.weights_id, None, tag)])
                return run_single_or_union(spc, tag, parts=parts, box=box, use_body=False,
                                           first=i, out_grid=out_grid, restore=restore)
            if step.crop_from_task is not None:
                report.stage("cascade", f"{tag} stage {i + 1}/{len(stages)}: crop from {step.crop_from_task!r}")
                crop = step.crop_from_task
                if ":" not in crop and ":" in spc.name:       # a registry names its own tasks bare
                    crop = f"{spc.name.partition(':')[0]}:{crop}"
                # upstream runs the crop task whole, its labels on the input image
                labels_in, src, _ = run_task_canonical(resolve(crop), f"{tag}:{step.crop_from_task}",
                                                       out_grid="input", restore="nearest")
            else:
                t = time.perf_counter()
                report.enter_part(i, f"{tag} stage {i + 1}/{len(stages)}: model {step.weights_id}"
                                  + ("" if box is None else " (cropped)"))
                model = load(step.weights_id, spc)
                x, src = model_frame(model, box)
                T[f"load:{tag}:s{i}"] = time.perf_counter() - t
                env = Envelope((0, 0, 0), tuple(int(s) for s in x.shape[1:]), tuple(int(s) for s in x.shape[1:]))
                labels_in = torch.zeros(src.source.shape, dtype=torch.uint8, device=device)
                t = time.perf_counter()
                predict_into(model, x, src, src.source, env, lut=np.arange(model.K, dtype=np.int32),
                             paint=False, out=labels_in, part=f"{tag}:s{i}", weights=step.weights_id,
                             restore="nearest")
                T[f"network:{tag}:s{i}"] = time.perf_counter() - t
                models.release(model)
            box = upstream_crop_box(labels_in.cpu().numpy(), step.crop_to_classes, step.dilation_mm,
                                    src.source.spacing)
            prov.setdefault("crops", []).append({
                "task": tag, "stage": i + 1, "classes": [int(c) for c in step.crop_to_classes],
                "margin_mm": float(step.dilation_mm),
                "box": None if box is None else [list(box[0]), list(box[1])]})
            if box is None:
                report.stage("cascade", "crop classes absent -> empty result, as upstream returns")
                og = src.resolve_grid(grid if out_grid is None else out_grid)
                max_label = max((int(v) for v in spc.label_map), default=255)
                return (torch.zeros(og.shape, dtype=torch.uint8 if max_label <= 255 else torch.uint16,
                                    device=device), src, og)
            frac = float(np.prod([h - l for l, h in zip(*box)])) / float(np.prod(src.source.shape))
            report.stage("cascade", f"crop {frac * 100:.0f} % of the input, +{step.dilation_mm:g} mm")
        raise AssertionError("unreachable: a cascade ends in a model or a union stage")

    def run_task_canonical(spc, tag="", *, out_grid=None, restore=None):
        if spc.shape == "cascade":
            return run_cascade(spc, tag or spc.name, out_grid=out_grid, restore=restore)
        return run_single_or_union(spc, tag or spc.name, out_grid=out_grid, restore=restore)

    with lock:
        labels, frame, out_grid = run_task_canonical(spec)
    t = time.perf_counter()
    report.stage("finalize", "to input orientation")
    # back to the input's own orientation: a permute + flip where the labels already live,
    # not a single-threaded DICOMOrient over the host copy
    arr, geo = nio.reorient(labels, frame.output_geometry(out_grid), frame.original_orientation)
    out_img = nio.to_image(arr, geo)
    T["to input orientation"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t_start
    prov.update(input_orientation=orientation, output_grid=tuple(out_grid.shape),
                cropped_to_nonzero=bool(crop_nonzero) and frame.model_source is not None,
                probabilities=(None if probabilities is None else
                               {"depth": probabilities.depth, "clip": probabilities.clip}))
    return Segmentation(labels=out_img, schema=schema, grid=out_grid, spec=spec,
                        timings=T, provenance=prov)
