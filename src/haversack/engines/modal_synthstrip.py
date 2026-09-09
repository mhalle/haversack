"""The Modal deployment for the SynthStrip engine: its image and its worker.

Beside the runtime it deploys, rather than a thousand lines away in
:mod:`haversack.modal_app`. That module imports this one exactly when this engine is
enabled, AFTER defining everything named below - Modal resolves `@app.cls` at import,
so the class must be created then, and importing conditionally is what keeps a
deployment from building an image for an engine it does not run.

What used to live in modal_app and does not any more: a module-level `SYNTHSTRIP` flag
whose absence silently dropped the engine from every deploy, and an entry in a
name-string map that `globals()` was searched for, whose typo did the same. The
composer takes :data:`WORKER` directly, so neither failure mode exists to guard.
"""
import os

import modal

from haversack.modal_app import (CACHE_ROOT, GPU, GPU_SNAPSHOT, INPUTS_ROOT,
                                 MAX_CONTAINERS, SCALEDOWN, SCRATCH_ROOT, SNAPSHOT,
                                 WEIGHTS_ROOT, _WorkerBase, _cls_extra, _pkg_dir, _RUNTIME_KNOBS,
                                 app, cache_vol, inputs_vol, scratch_vol, weights_vol)

#: The engine this adapter deploys. Must equal the module suffix and a registry key;
#: `test_engine_completeness` reconciles all three.
ENGINE = "synthstrip"

# SynthStrip engine image (built only when enabled). uv-NATIVE, same shape as fs_image:
# `synthstrip` brings synthstrip-torch (from its git source in pyproject) + scipy (haversack
# mask cleanup); `idc` brings obstore (fetch); `preview` brings matplotlib (serve-core
# preview - synthstrip-torch doesn't carry it, it's a serve-tier concern). numpy<2 comes
# from synthstrip-torch (surfa's reorient breaks on numpy 2.x). Weights fetch from MGH at
# first use (cached warm), like FastSurfer's checkpoints.
def _synthstrip_image():
    synthstrip_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")                       # uv needs git for the git source in pyproject
        .uv_sync(extras=["synthstrip", "preview"], frozen=False)
        # Bake the 29 MB weights into the image at BUILD (to synthstrip-torch's default cache)
        # so cold containers don't re-download from MGH. Same rationale as FastSurfer above.
        .run_commands("python -c 'import synthstrip_torch; synthstrip_torch.fetch_weights()'")
        .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
        .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack")
    )
    return synthstrip_image


@app.cls(gpu=GPU, timeout=3600, memory=32768, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS, image=_synthstrip_image(),
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
              CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class SynthStripWorker(_WorkerBase):
    """The SynthStrip engine worker: the shared scheduler + serve-core, with
    the slim synthstrip image and the standalone synthstrip-torch package."""

    engine = "synthstrip"

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
        _pkg_dir()
        import torch  # noqa: F401
        import synthstrip_torch  # noqa: F401 - model class + torch
        import haversack  # noqa: F401
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if GPU_SNAPSHOT:
            from haversack.engines import synthstrip
            synthstrip._get_model("cuda")

    def _compute(self, input_path, meta, on_progress, token):
        from haversack.engines import synthstrip
        # input_path is a SimpleITK image (read-ahead memory-in) or a path;
        # segment() takes both and writes no temp files (model cached per worker).
        return synthstrip.segment(input_path, device="cuda")


#: What the composer in modal_app imports.
WORKER = SynthStripWorker
