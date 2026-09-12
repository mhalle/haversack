"""The engine registry: which runtimes exist, and what each one is.

**Ecosystem vs Engine.** An *ecosystem* is what the user selects - a catalog of
tasks with the ``eco:task@version`` grammar (``ts``, ``moose``, ``mrsegmentator``,
``custom``, ``fastsurfer``, ``synthstrip``). An *engine* is the runtime that actually
runs a task: its container image, its compute, its weights identity. **Many
ecosystems map to one engine** - ``ts``, ``moose``, ``mrsegmentator`` and ``custom``
are four catalogs of nnU-Net models, all run by the ``nnunetv2`` engine.

This module is the single source of truth for that mapping and for every fact
that used to be spelled once per engine per call site: the enable flag, the
weights identity that keys the result cache, and the ecosystem -> engine route.
Adding an engine starts with one row here and does not end there - it took 18
edits in 7 files to reach a green suite when measured on 2026-09-08. Four of
those were in :mod:`haversack.modal_app`; since the workers moved out of it the
same day, the Modal side is one new file beside the engine instead,
``engines/modal_<engine>.py`` (its image and its ``@app.cls`` worker), which
modal_app composes by iterating this registry.

Deliberately a **static registry, not a plugin framework**. That is not only
YAGNI; three properties of this system make discovered plugins impossible to
do honestly, and they are worth stating so the question stops being reopened:

1. ``import haversack`` must pull no torch (``docs/dependency-discipline.md``).
   Anything discovered at import time drags its runtime in with it.
2. Modal resolves ``@app.cls`` decorators at import, so a worker and its image
   are declared statically - an optional engine's in an adapter module that is
   imported only when that engine is enabled. An engine that appeared at runtime
   could not have a Modal worker at all.
3. Engines live in mutually conflicting environments - synthstrip pins
   ``numpy<2`` where the torch extra resolves past 2 - so "the set of installed
   engines" is not a coherent question to ask one interpreter.

What DOES keep the cost down is this row owning every per-engine fact that code
outside the engine needs. Each field below was added because a consumer was
otherwise keeping its own copy, and the copies drift: ``dist`` exists because
``/v1/version`` hand-listed engine packages and silently omitted two engines,
and ``label_names`` because the ranked builder tested ``engine == "fastsurfer"``.
When adding an engine needs an edit somewhere new, the fix is a field here, not
a branch there. ``tests/test_engine_completeness.py`` walks this dict and fails
naming the step that was missed.

It is also deliberately **light**:
no torch, no SimpleITK, not even an import of the engine modules. ``importing
haversack`` must stay torch-free (see ``docs/dependency-discipline.md``), and
``info()`` on the lean API image reads these constants, so the version literals
live *here* and the engine modules re-export them. The one dependency it does
take is pydantic, for the parameter schemas below - 27 ms to import and 7.8 MB
on disk, against a core that already requires SimpleITK's 183 MB, and it buys
the published schema, the enforced validation and the OpenAPI document from one
declaration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from ..schemas import GRADED_RESTORE, NoParams, Params, VoxTellParams

# The nnU-Net runtime. Named for the upstream major version it runs: nnU-Net v1
# has a different checkpoint layout and would need its own loader, hence its own
# engine, rather than a flag on this one.
NNUNETV2 = "nnunetv2"


def _truthy(value: str | None) -> bool:
    return (value or "0") not in ("0", "false", "no", "")


@dataclass(frozen=True)
class Engine:
    """One runtime. ``weights_identity`` is the engine's contribution to the
    result-cache key.

    It is a **constant** (or ``None``), never a function of the task, for two
    reasons. Engines bake their weights into their image, so there is nothing
    per-task to look up. And ``info()`` is called once per task by ``/v1/tasks``
    and ``/v1/version`` - 70+ tasks on a default catalog - so a per-task lookup
    here would put a filesystem walk over a mounted weights Volume in the path of
    two hot endpoints on the lean API container. The nnU-Net engine's weights are
    per-task and install-time, so its identity stays ``None`` here and is computed
    by :meth:`haversack.segmenter.Segmenter.describe` from the spec's weights ids.
    """

    name: str
    enabled_env: str | None = None          # None = always on (the default engine)
    weights_identity: Callable[[], list[dict]] | None = None
    # The engine's compute entry point for in-process runs: called by
    # Segmenter._run_engine as compute(image, device=, batch_size=, progress=,
    # cancel=, probabilities=) and returns a Segmentation. Set for fastsurfer
    # (local, 2026-09-03); the Modal worker keeps its own _compute hook. None
    # means the engine runs only on Modal so far.
    compute: Callable | None = None
    #: The importable module that proves the engine's runtime is present in THIS
    #: environment - checked with find_spec, never imported. Installed runtime =
    #: enabled locally (see `enabled`).
    runtime_module: str | None = None
    #: The extra that installs that runtime, in its own environment (pyproject's
    #: [tool.uv] conflicts): UV_PROJECT_ENVIRONMENT=.venvs/<name> uv sync --extra <extra>.
    extra: str | None = None
    #: The DISTRIBUTION names this engine brings, for `/v1/version`'s package
    #: report - which pins the rev a deployment is actually running. Not
    #: derivable from `runtime_module`: the import name and the distribution
    #: differ for two engines already (FastSurferCNN <- fastsurfer-lean,
    #: synthstrip_torch <- synthstrip-torch), and one engine's answer is more
    #: than one package. Includes what the engine pulls in that we want pinned
    #: in the report (surfa, under synthstrip), not just the top-level name.
    #: Reported only where installed, so listing all of them here is right.
    dist: tuple[str, ...] = ()
    #: The store this engine keeps under the CACHE root, as ``(subdirectory, environment
    #: override)`` - what ``haversack cache usage`` reports and ``cache clean`` may sweep.
    #: Data, not a call into the engine, so cache admin can read it without importing an
    #: engine module (which is why it hand-copied FastSurfer's location until 2026-09-08,
    #: making every OTHER engine's cache invisible to both commands). None for an engine
    #: that caches nothing there - weights baked into the image, fetched to the weights
    #: volume, or left in the runtime's own hub cache.
    cache_store: tuple[str, str | None] | None = None
    #: ``task -> {label id: name}`` when the labels this engine emits are in the
    #: ENGINE's namespace rather than the catalog's - FastSurfer carries
    #: FreeSurfer's aparc+aseg ids, which no ecosystem knows. A thunk, called at
    #: use and never at import, so this module keeps taking no engine imports.
    #: None means the labels are the ecosystem's and the catalog names them.
    label_names: Callable[[str], dict] | None = None
    #: Whether a stored result may be SERVED for a repeat request (RFC 9111
    #: `no-cache` when False - results are still stored, so artifacts, the job's
    #: `key` and its `links` all keep working; `no-store` is a different thing we
    #: do not offer). False suits an engine whose request space is effectively
    #: unbounded, where a lookup almost never hits and the entries only push
    #: useful ones out of a shared LRU - VoxTell, whose free-text prompts hash
    #: into the key, is the case this exists for.
    serve_from_cache: bool = True
    #: This engine's own cache epoch: the per-engine half of ``serve.CACHE_EPOCH``. Bump it
    #: when THIS engine would compute different bytes from the same request - its inference
    #: path, its conform, its restore - and only its results are recomputed. None (the
    #: default) adds nothing to the key, so an engine that never bumps keeps exactly the
    #: keys it always had. Kept out of ``weights_identity`` on purpose: that also labels the
    #: probabilities an engine stores, and a processing change is not a new set of weights.
    #: Exists because the global epoch is global (2026-09-12): FastSurfer's 0.7 mm floor
    #: changes only FastSurfer's sub-0.7 mm results, and bumping CACHE_EPOCH for it would
    #: have thrown away every nnU-Net result as well.
    cache_epoch: str | None = None
    #: The engine's own parameters, as a pydantic model. Generates the JSON
    #: Schema ``describe()`` publishes AND validates the request at submit, from
    #: one declaration. Our processing knobs are a separate group and are not
    #: listed here - see :func:`haversack.schemas.parameter_groups`.
    parameters: type[Params] = NoParams
    #: Whether this engine can consume more than one input image. Introspection
    #: precedes capability on purpose: a multi-channel model may *declare* its
    #: roles (so a client can see what it would need) while the engine that runs
    #: it still refuses to be handed them.
    multi_input: bool = False
    #: Read-only facts about what this engine does to a result - published so a
    #: client can see behavior it cannot change. Empty when the answer is
    #: per-task rather than per-engine (MONAI, where each bundle's own
    #: postprocessing decides).
    behavior: dict = field(default_factory=dict)
    #: Whether haversack's processing knobs (``grid``, ``interp``, ...) apply. False
    #: for engines that run someone else's chain end to end, where offering them
    #: would be advertising a knob we do not turn.
    processing_knobs: bool = True
    description: str = ""


def _fastsurfer_identity() -> list[dict]:
    return [{"id": "fastsurfer", "version": "vinn-v2"}]


def _fastsurfer_compute(image, **kw):
    """Local FastSurfer run; the engine module (and FastSurfer) import only here."""
    from .fastsurfer import run_local
    return run_local(image, **kw)


def _fastsurfer_label_names(task: str) -> dict:
    """FreeSurfer aparc+aseg id -> name, read from FastSurfer's own colour LUT.

    Ignores ``task``: this engine has one label namespace whatever it is asked for.
    """
    from .fastsurfer import label_names
    return label_names()


def _synthstrip_identity() -> list[dict]:
    return [{"id": "synthstrip", "version": "v1"}]


def _synthstrip_compute(image, **kw):
    from .synthstrip import run_local
    return run_local(image, **kw)


def _voxtell_identity() -> list[dict]:
    return [{"id": "voxtell", "version": "v1.1"}]


# No _monai_identity: the MONAI engine runs a CATALOG of bundles, so the identity
# is per bundle+version and comes from MonaiEcosystem.weights_identity, not from a
# constant here. That is what Engine.weights_identity=None means.


ENGINES: dict[str, Engine] = {
    NNUNETV2: Engine(
        name=NNUNETV2,
        runtime_module="nnunetv2", extra="torch",
        dist=("nnunetv2", "torch"),
        # the only engine whose surrounding pipeline is ours, so the only one
        # where our processing knobs are real
        behavior=GRADED_RESTORE,
        description="nnU-Net v2 networks (TotalSegmentator, MOOSE, MRSegmentator, and stock models)",
    ),
    "fastsurfer": Engine(
        name="fastsurfer",
        enabled_env="HAVERSACK_FASTSURFER",
        weights_identity=_fastsurfer_identity,
        compute=_fastsurfer_compute,
        runtime_module="FastSurferCNN", extra="fastsurfer",
        dist=("fastsurfer-lean",),
        label_names=_fastsurfer_label_names,
        cache_store=("fastsurfer-checkpoints", "HAVERSACK_FASTSURFER_CHECKPOINTS"),
        behavior=GRADED_RESTORE,
        processing_knobs=False,
        # 1 (2026-09-12): inputs finer than 0.7 mm are processed at 0.7 (VOX_FLOOR_MM in
        # engines/fastsurfer.py), where they used to run on FastSurfer's unfloored "min"
        cache_epoch="1",
        description="FastSurferVINN 2.5D view-aggregation parcellation",
    ),
    "synthstrip": Engine(
        name="synthstrip",
        enabled_env="HAVERSACK_SYNTHSTRIP",
        runtime_module="synthstrip_torch", extra="synthstrip",
        # surfa is synthstrip-torch's conform/reorient dependency, pinned in
        # the report because a change in it moves the output grid.
        dist=("synthstrip-torch", "surfa"),
        weights_identity=_synthstrip_identity,
        compute=_synthstrip_compute,
        behavior=GRADED_RESTORE,
        processing_knobs=False,
        description="SynthStrip brain extraction (signed distance transform)",
    ),
    "voxtell": Engine(
        name="voxtell",
        enabled_env="HAVERSACK_VOXTELL",
        runtime_module="voxtell", extra="voxtell",
        dist=("voxtell",),
        weights_identity=_voxtell_identity,
        # free text means an unbounded key space and interactive, low-reuse
        # requests: memoizing them evicts results that do get re-read
        serve_from_cache=False,
        parameters=VoxTellParams,
        processing_knobs=False,
        behavior={"restore": {
            "mode": "none", "owner": "engine",
            "note": "masks come back on the reader's grid; haversack does not resample"}},
        description="VoxTell free-text promptable segmentation (prompts are input)",
    ),
    "monai": Engine(
        name="monai",
        enabled_env="HAVERSACK_MONAI",
        runtime_module="monai", extra="monai",
        dist=("monai",),
        weights_identity=None,          # per bundle - see MonaiEcosystem
        processing_knobs=False,
        # The first engine that can be handed more than one image. Which bundles
        # actually want that is per task and read from each bundle's declared
        # input channels; this only says the runtime knows what to do with them.
        multi_input=True,
        # no `behavior` constant: each bundle's own postprocessing decides
        # whether it inverts probabilities or a labelmap, so the fact is per task
        # and read from the installed bundle - see MonaiEcosystem.describe_task
        description="MONAI model zoo bundles (each carries its own transforms)",
    ),
}

# Which engine runs each ecosystem's tasks. Ecosystems not listed here run on the
# default engine, so the nnU-Net catalogs need no entry.
ECOSYSTEM_ENGINE: dict[str, str] = {
    "fastsurfer": "fastsurfer",
    "synthstrip": "synthstrip",
    "voxtell": "voxtell",
    "monai": "monai",
}


def engine_for(ecosystem: str) -> Engine:
    """The engine that runs ``ecosystem``'s tasks (the default engine if the
    ecosystem declares none)."""
    return ENGINES[ECOSYSTEM_ENGINE.get(ecosystem, NNUNETV2)]


def engine_for_task(task: str) -> Engine:
    """The engine for a canonical ``eco:task`` name. Routes on the *grammar*
    rather than on hardcoded task prefixes, and falls back to the default engine
    for a bare name (every wire form is canonicalized before it reaches here)."""
    return engine_for(str(task).partition(":")[0])


def available(name: str) -> bool:
    """Whether the engine's runtime is importable in this environment. Looks the
    module up (find_spec) and never imports it, so it is cheap and side-effect free."""
    import importlib.util
    eng = ENGINES[name]
    return eng.runtime_module is None or importlib.util.find_spec(eng.runtime_module) is not None


def enabled(name: str) -> bool:
    """Whether ``name`` can run here. The environment flag decides when it is set
    (``=1`` on, ``=0`` off - the Modal deploy sets it per image); when it is unset,
    an engine that can run in-process is enabled exactly when its runtime is
    installed, so a per-engine venv (``uv sync --extra fastsurfer``) needs no
    further switch. Read on every call (never cached) so a test can monkeypatch
    either signal; callers that must decide at import time - Modal resolves
    decorators then - snapshot the result themselves."""
    eng = ENGINES[name]
    if eng.enabled_env is None:
        return True
    flag = os.environ.get(eng.enabled_env)
    if flag is not None:
        return _truthy(flag)
    return eng.compute is not None and available(name)


def enabled_engines() -> list[str]:
    """Names of the engines this deployment can actually run."""
    return [n for n in ENGINES if enabled(n)]


def engine_env_vars() -> tuple[str, ...]:
    """Every engine enable flag, for forwarding into a container's environment.
    Derived, so a new engine cannot be forgotten here (a knob that exists at
    deploy time but not in the container is a bug this project has already hit)."""
    return tuple(e.enabled_env for e in ENGINES.values() if e.enabled_env)
