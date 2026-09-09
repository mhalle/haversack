"""The Modal deployment for the VoxTell engine: its image and its worker.

Beside the runtime it deploys, rather than a thousand lines away in
:mod:`haversack.modal_app`. That module imports this one exactly when this engine is
enabled, AFTER defining everything named below - Modal resolves `@app.cls` at import,
so the class must be created then, and importing conditionally is what keeps a
deployment from building an image for an engine it does not run.

What used to live in modal_app and does not any more: a module-level `VOXTELL` flag
whose absence silently dropped the engine from every deploy, and an entry in a
name-string map that `globals()` was searched for, whose typo did the same. The
composer takes :data:`WORKER` directly, so neither failure mode exists to guard.
"""
import os
import sys

import modal

from haversack.modal_app import (CACHE_ROOT, GPU, GPU_SNAPSHOT, INPUTS_ROOT,
                                 MAX_CONTAINERS, SCALEDOWN, SCRATCH_ROOT, SNAPSHOT,
                                 WEIGHTS_ROOT, _WorkerBase, _cls_extra, _pkg_dir, _RUNTIME_KNOBS,
                                 app, cache_vol, inputs_vol, scratch_vol, weights_vol)

#: The engine this adapter deploys. Must equal the module suffix and a registry key;
#: `test_engine_completeness` reconciles all three.
ENGINE = "voxtell"

# VoxTell engine image (built only when enabled). The `voxtell` extra brings the package
# and its own tree (torch<2.9, nnunetv2, transformers, huggingface_hub); `idc` brings
# obstore, `preview` matplotlib for the serve-core preview.
#
# Weights policy differs from the other engines, deliberately. The VoxTell checkpoint and
# the precomputed text-embedding bank are baked at BUILD (small, and they cover the common
# prompts with no text backbone at all). But a prompt outside that bank is embedded on the
# fly by Qwen3-Embedding-4B - ~8 GB, which would bloat the image and slow every cold pull,
# and the cold pull IS the cold start here. So HF_HOME points at the PERSISTENT weights
# volume instead: the backbone is fetched once ever and every later cold container reads it
# locally - the same treatment nnU-Net's weights already get.
def _voxtell_image():
    voxtell_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_sync(extras=["voxtell", "preview"], frozen=False)
        # Bake the checkpoint into the image at a FIXED path (it is small) and address it by
        # VOXTELL_MODEL, so it stays findable after HF_HOME moves to the volume below.
        .run_commands(
            "python -c \""
            "import shutil;"
            "from voxtell.inference.predictor import download_voxtell_model as d;"
            "shutil.copytree(d(), '/opt/voxtell/model', dirs_exist_ok=True)\""
        )
        # The runtime caches - the small embedding bank, and the Qwen3 backbone that only a
        # prompt outside that bank needs - live on the PERSISTENT weights volume, so they are
        # fetched once ever and every later cold container reads them locally.
        .env({"VOXTELL_MODEL": "/opt/voxtell/model", "HF_HOME": f"{WEIGHTS_ROOT}/hf"})
        .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
        .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack")
    )
    return voxtell_image


@app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS, image=_voxtell_image(),
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
              CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class VoxTellWorker(_WorkerBase):
    """The VoxTell engine worker: free-text prompts instead of a fixed task.

    The only worker whose compute reads the job's ``options`` for what to
    segment - ``{"prompts": [...]}`` - which is also what makes two prompt lists
    two different cache entries."""

    engine = "voxtell"

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
        _pkg_dir()
        import torch  # noqa: F401
        import voxtell.inference.predictor  # noqa: F401 - the model import graph
        import haversack  # noqa: F401
        if GPU_SNAPSHOT:
            from haversack.engines import voxtell
            voxtell._get_predictor("cuda")

    def _compute(self, input_path, meta, on_progress, token):
        from haversack.engines import voxtell
        opts = dict(meta.get("options") or {})
        seg = voxtell.segment(input_path, opts.get("prompts"), device="cuda",
                              progress=on_progress, cancel=token)
        # The text backbone and embedding bank land in HF_HOME on the weights
        # volume; commit once per container so the next cold start reads them
        # instead of re-downloading (the whole point of caching them there).
        if not getattr(self, "_hf_committed", False):
            try:
                weights_vol.commit()
                self._hf_committed = True
            except Exception as e:                  # never fail a finished job on this
                print(f"[voxtell] weights volume commit failed: {e}", flush=True)
        return seg


#: What the composer in modal_app imports.
WORKER = VoxTellWorker

# The other half of the handshake. A Modal WORKER container imports the module the class
# LIVES IN - this one - not modal_app, so the composer over there ran while this module
# was still initializing and skipped it rather than reading a class that did not exist
# yet. Registering here means both import orders end with the same complete map, and
# nothing has to know which way round it was entered.
sys.modules["haversack.modal_app"].ENGINE_WORKERS.setdefault(ENGINE, WORKER)
