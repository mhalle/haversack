"""The Modal deployment for the FastSurfer engine: its image and its worker.

Beside the runtime it deploys, rather than a thousand lines away in
:mod:`haversack.modal_app`. That module imports this one exactly when this engine is
enabled, AFTER defining everything named below - Modal resolves `@app.cls` at import,
so the class must be created then, and importing conditionally is what keeps a
deployment from building an image for an engine it does not run.

What used to live in modal_app and does not any more: a module-level `FASTSURFER` flag
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
ENGINE = "fastsurfer"

# Each engine image is built by a FUNCTION, called from inside the `if <ENGINE>:`
# that defines its worker - so an engine this deployment does not enable costs no
# build. They used to be module-level expressions, and Modal built every one of them
# on every deploy: with HAVERSACK_FASTSURFER unset, two deploys still ran the
# FastSurfer image's checkpoint fetch and both died on a Zenodo 504 (2026-09-07).
# One engine's upstream having a bad day must not stop a deploy that does not use it.

# FastSurfer engine image. uv-NATIVE:
# `uv_sync` installs the project's deps for the `fastsurfer` + `idc` extras straight
# from pyproject (`--no-install-project`, so haversack stays mounted, not installed) -
# the fastsurfer-lean git source + rev live ONLY in [tool.uv.sources], not here.
# `fastsurfer` pulls fastsurfer-lean (CNN inference deps incl. matplotlib, no
# monai/meshpy/torchio); `idc` pulls obstore (source fetch); core deps (SimpleITK
# etc.) come with the sync. frozen=False: this repo gitignores uv.lock (pyproject is
# the source of truth), so resolve at build.
def _fs_image():
    _FS_CKPT = os.environ.get("HAVERSACK_FASTSURFER_CHECKPOINTS")
    fs_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")                       # uv needs git for the git source in pyproject
        .uv_sync(extras=["fastsurfer"], frozen=False)
        .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack", copy=True)
    )
    if _FS_CKPT:
        # Ship a local checkpoint directory into the image (a user's pre-fetched copy).
        fs_image = fs_image.add_local_dir(_FS_CKPT, remote_path="/opt/fastsurfer-checkpoints", copy=True)
    else:
        # Bake the ~66 MB checkpoints at BUILD via haversack's own Zenodo fetch (sha256-verified,
        # stdlib) - so cold containers never download them, and the build never touches
        # FastSurfer's b2share host, whose certificate chain fails in the container (2026-09-03).
        fs_image = fs_image.run_commands(
            "PYTHONPATH=/root/pkg python -c "
            "'from haversack.engines.fastsurfer import ensure_checkpoints;"
            "ensure_checkpoints(\"/opt/fastsurfer-checkpoints\")'")
    fs_image = (fs_image
                .env({"HAVERSACK_FASTSURFER_CHECKPOINTS": "/opt/fastsurfer-checkpoints"})
                .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ}))
    return fs_image


@app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS, image=_fs_image(),
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
              CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class FastSurferWorker(_WorkerBase):
    """The FastSurfer engine worker: the shared scheduler + serve-core from
    _WorkerBase, with FastSurfer's image and compute."""

    engine = "fastsurfer"

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """Heavy imports paid once per deploy, before the memory snapshot; later
        cold containers restore from it. Stays per-worker because its body IS this
        image's import set. Classic snapshot => imports only, no CUDA; with
        HAVERSACK_GPU_SNAPSHOT the model is built onto the GPU so a restored container
        starts model-ready."""
        _pkg_dir()
        import torch  # noqa: F401
        import FastSurferCNN.run_prediction  # noqa: F401 - the CNN import graph
        import haversack  # noqa: F401
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if GPU_SNAPSHOT:
            from haversack.engines import fastsurfer
            fastsurfer._get_runner("cuda", 8)

    def _compute(self, input_path, meta, on_progress, token):
        from haversack.engines import fastsurfer
        # input_path is a SimpleITK image when read-ahead pre-read it
        # (memory-in, decode-once) or a path otherwise; segment() takes both
        # and writes no temp files (model is cached across jobs on this worker).
        return fastsurfer.segment(input_path, device="cuda")


#: What the composer in modal_app imports.
WORKER = FastSurferWorker

# The other half of the handshake. A Modal WORKER container imports the module the class
# LIVES IN - this one - not modal_app, so the composer over there ran while this module
# was still initializing and skipped it rather than reading a class that did not exist
# yet. Registering here means both import orders end with the same complete map, and
# nothing has to know which way round it was entered.
sys.modules["haversack.modal_app"].ENGINE_WORKERS.setdefault(ENGINE, WORKER)
