"""The Modal deployment for the the MONAI bundles engine: its image and its worker.

Beside the runtime it deploys, rather than a thousand lines away in
:mod:`haversack.modal_app`. That module imports this one exactly when this engine is
enabled, AFTER defining everything named below - Modal resolves `@app.cls` at import,
so the class must be created then, and importing conditionally is what keeps a
deployment from building an image for an engine it does not run.

What used to live in modal_app and does not any more: a module-level `MONAI` flag
whose absence silently dropped the engine from every deploy, and an entry in a
name-string map that `globals()` was searched for, whose typo did the same. The
composer takes :data:`WORKER` directly, so neither failure mode exists to guard.
"""
import os

import modal

from haversack.modal_app import (CACHE_ROOT, GPU, INPUTS_ROOT, MAX_CONTAINERS,
                                 SCALEDOWN, SCRATCH_ROOT, SNAPSHOT, WEIGHTS_ROOT,
                                 _WorkerBase, _cls_extra, _pkg_dir, _RUNTIME_KNOBS,
                                 app, cache_vol, inputs_vol, scratch_vol, weights_vol)

#: The engine this adapter deploys. Must equal the module suffix and a registry key;
#: `test_engine_completeness` reconciles all three.
ENGINE = "monai"

# MONAI engine image (built only when enabled). The `monai` extra brings monai + torch;
# the curated bundles declare their own dependency set (itk, pytorch-ignite, einops, timm,
# ...) and the image carries the union - `tools/gen_monai_manifest.py` prints it, so the
# list is derived from the bundles rather than guessed. Weights are NOT baked: this is a
# catalog, so bundles install per task into the persistent weights volume (like nnU-Net and
# MOOSE), which is why this worker's _prepare/_ensure do real work.
def _monai_image():
    monai_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_sync(extras=["monai", "preview"], frozen=False)
        .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
        .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack")
    )
    return monai_image


@app.cls(gpu=GPU, timeout=3600, memory=40960, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS, image=_monai_image(),
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
              CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class MonaiWorker(_WorkerBase):
    """The MONAI engine worker: a CATALOG of bundles, so unlike the other engine
    workers its _prepare/_ensure do real work - bundles install per task into the
    weights volume, exactly as the nnU-Net worker installs its models."""

    engine = "monai"

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """Heavy imports before the memory snapshot (see FastSurferWorker.preload)."""
        _pkg_dir()
        import torch  # noqa: F401
        import monai  # noqa: F401 - eagerly loads transforms/networks/inferers...
        # ...but monai/__init__ EXCLUDES monai.bundle from that eager load
        # ("(^(monai.bundle))" in its exclude_pattern), and monai.bundle is the
        # only part this engine actually calls. Importing it here is what puts
        # it in the snapshot instead of on the first request after every restore.
        import monai.bundle  # noqa: F401
        import monai.transforms  # noqa: F401 - the chain every bundle composes
        import haversack  # noqa: F401

    def _bundle_of(self, task: str) -> str:
        return str(task).partition(":")[2] or str(task)

    def _prepare(self, task: str, progress=None) -> dict:
        from haversack.ecosystems import MonaiEcosystem
        bundle = self._bundle_of(task)
        MonaiEcosystem().ensure(bundle, WEIGHTS_ROOT, progress=progress)
        weights_vol.commit()
        self._ensured.add(task)
        return {"engine": self.engine, "task": task, "bundle": bundle}

    def _ensure(self, task: str) -> None:
        if task not in self._ensured:
            self._prepare(task)

    def _compute(self, input_path, meta, on_progress, token):
        from haversack.engines import monai_bundle
        return monai_bundle.segment(input_path, self._bundle_of(meta["task"]),
                                    root=WEIGHTS_ROOT, device="cuda",
                                    progress=on_progress, cancel=token)


#: What the composer in modal_app imports.
WORKER = MonaiWorker
