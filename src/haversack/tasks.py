"""Task catalog: a named task -> the model(s) to run and how their labels combine.

Reads the same registry JSON the MLX toolkit ships (``ts_tasks.json``), but with no dependency
on that package - it imports mlx, which does not exist off Apple silicon. Only the parts haversack
executes are modelled here: single-model tasks and label-union tasks. Cascades are recorded
but not runnable yet, and say so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AmbiguousModel, ModelNotFound, UnsupportedModel
from typing import Mapping

WeightsId = int | str

# Where each weights LAYOUT keeps its models: (env vars, default root). A layout is not an
# ecosystem (a catalog) and not an engine (a runtime) - it is just which install tree the
# weights live in. The names match the vocabulary used everywhere else: ``ts`` is
# TotalSegmentator's own tree, ``nnunetv2`` is a stock nnU-Net results tree. The layout
# *under* either root is nnU-Net's own:
# ``Dataset<id>_<name>/<trainer>__<plans>__<config>/``.
# The env vars and default path are TotalSegmentator's and nnU-Net's, so they keep their
# upstream spelling.
LAYOUTS = {
    "ts": (("TOTALSEG_WEIGHTS_PATH", "nnUNet_results"),
           Path("~/.totalsegmentator/nnunet/results")),
    "nnunetv2": (("nnUNet_results",), None),
}


@dataclass(frozen=True)
class UnionPart:
    """One model of a label-union task, and how its local classes map to global labels."""

    weights_id: WeightsId
    label_remap: Mapping[int, int]
    name: str = ""


@dataclass(frozen=True)
class CascadeStep:
    """One stage of a cascade. All but the last exist to crop the next: run the model, take the
    bounding box of ``crop_to_classes`` in its output, dilate by ``dilation_mm``, and restrict
    the following stage to that box. The last stage (``crop_to_classes`` empty) is the target
    whose labels become the result.

    A stage either runs a model (``weights_id``) or reuses another task's output as the crop
    source (``crop_from_task``, e.g. teeth cropping from craniofacial_structures). The LAST
    stage may instead be a label union (``union``): every part runs on the one box the stages
    before it found, and paints its classes into one output in order, a later part over an
    earlier one - TotalSegmentator's ``headneck_muscles`` (2026-09-22), which crops like
    ``headneck_bones_vessels`` and then runs Datasets 778 and 779 on that crop, combining them
    exactly as it combines ``total``'s five parts."""

    weights_id: WeightsId | None = None
    crop_to_classes: tuple[int, ...] = ()
    dilation_mm: float = 10.0
    crop_from_task: str | None = None
    union: tuple["UnionPart", ...] = ()


def dataset_labels(ds: dict, where: str = "dataset.json") -> dict[int, str]:
    """``{label value: name}`` from an nnU-Net ``dataset.json``, background and ``ignore``
    dropped.

    The one reading of a checkpoint's labels. :meth:`TaskSpec.from_model_folder` builds an
    installed model's label map with it, and ``haversack catalog mine`` reads a remote
    archive's with it (2026-09-12), so the index and an installed model cannot disagree about
    what the same file says. Region-based labels - a name mapping to several values - raise,
    as they always have here.

    Background and ``ignore`` are roles a value plays, not segments. nnU-Net never predicts
    the ignore label - its label manager skips the key by name and requires it to be the
    highest value, one past the rest - so listing it named a segment no result can contain:
    TotalVibe's ``vibe`` and ``vibe_sagittal`` reported 73 structures for their 72 (found
    2026-09-13). duckn's segmentation extension keeps such roles with the value, not as
    segments. The key is matched exactly, as nnU-Net matches it.
    """
    labels = ds.get("labels") or {}
    if any(isinstance(v, (list, tuple)) for v in labels.values()):
        raise UnsupportedModel(
            f"{where}: region-based labels (a label mapping to several values) are not "
            "supported yet - haversack takes the argmax of a softmax head")
    return {int(v): str(k) for k, v in labels.items() if int(v) != 0 and k != "ignore"}


def dataset_modality(ds: dict) -> str:
    """The first input channel's name in an nnU-Net ``dataset.json`` - what a model folder's
    spec reports as its modality."""
    chan = ds.get("channel_names") or ds.get("modality") or {"0": "unknown"}
    return str(next(iter(chan.values())))


@dataclass(frozen=True)
class TaskSpec:
    name: str
    #: Which preprocessing lineage the model was trained under - "ts"
    #: (TotalSegmentator: corner convention, no crop) or "nnunetv2" (stock
    #: nnU-Net: center convention, crop-to-nonzero). NOT the ecosystem that
    #: lists the task, and NOT the engine that runs it.
    lineage: str = "ts"
    modality: str = "CT"
    shape: str = "single"
    single: WeightsId | None = None
    union: tuple[UnionPart, ...] = ()
    cascade: tuple[CascadeStep, ...] = ()
    label_map: Mapping[int, str] = field(default_factory=dict)
    #: The orientation the model must see, when its *packaging* reorients rather
    #: than its declared nnU-Net reader: MRSegmentator wraps stock checkpoints in
    #: a reader that forces LPS, so the plans' ``SimpleITKIO`` (no reorientation)
    #: is not the truth. None (the default) follows the declared reader - RAS for
    #: the TotalSegmentator lineage, the stored order for a plain nnU-Net model.
    orientation: str | None = None
    #: Which model folder to run for a weights id, when its dataset may hold more than one:
    #: ``{dataset id: {"trainer": ..., "plans": ...}}``, either key optional. A dataset is
    #: ``<trainer>__<plans>__<configuration>/`` folders, and TotalSegmentator's v3 release
    #: ships two ``3d_fullres`` folders in each of 831-836 - ``nnUNetPlans`` (upstream's
    #: default) and ``nnUNetResEncUNetLPlans_8`` (its ``model_size="small"``). The registry
    #: says which one the task means, as upstream's own task config does; the resolver
    #: refuses a dataset it cannot narrow to one folder rather than pick (2026-09-21).
    models: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: The sliding window's tile step, when the task states one; None runs nnU-Net's 0.5.
    #: TotalSegmentator tiles ``total``, ``total_v3`` and ``total_mr`` at 0.8 (nnunet.py:
    #: faster, "dice 0.001 worse"), and matching it took ts.v3:total from 99.86 % to 99.98 %
    #: voxel agreement with upstream (2026-09-21). Stated by ts.v3's registry only: ts.v2
    #: keeps 0.5, which its cached results were computed with. A stated step enters the
    #: warm-model key and the result key (``serve.weights_versions_of``).
    step_size: float | None = None

    def model_choice(self, weights_id) -> dict:
        """``{"trainer"?, "plans"?}`` this task states for ``weights_id`` - the keyword
        arguments every resolve of that weights id must carry. Empty when it states none,
        which is every task whose datasets hold a single model folder."""
        return dict(self.models.get(_dataset_key(weights_id)) or {})

    @classmethod
    def from_model_folder(cls, folder, *, name: str | None = None) -> "TaskSpec":
        """A single-model task read straight from a stock nnU-Net result folder.

        Takes ``.../Dataset<id>_<name>/<trainer>__<plans>__<config>/`` (or the dataset folder,
        resolved by :func:`resolve_model_folder`) and builds the spec from its ``dataset.json``:
        labels become the label map, ``channel_names`` the modality. This is how a caller uses
        haversack with their own nnU-Net model, with no catalog entry anywhere.
        """
        f = resolve_model_folder(folder)
        if not (f / "dataset.json").is_file():
            raise ModelNotFound(f"{f} has no dataset.json - not a trained nnU-Net model folder "
                                "(expected .../Dataset<id>_<name>/<trainer>__<plans>__<config>/)")
        ds = json.loads((f / "dataset.json").read_text(encoding="utf-8"))
        return cls(name=name or f.parent.name, lineage="nnunetv2",
                   modality=dataset_modality(ds), shape="single", single=str(f),
                   label_map=dataset_labels(ds, where=f.name))

    @property
    def parts(self) -> list[tuple[WeightsId, Mapping[int, int] | None, str]]:
        """``(weights id, local->global remap or None, part name)`` in paint order (single / union)."""
        if self.single is not None:
            return [(self.single, None, self.name)]
        if self.union:
            return [(p.weights_id, dict(p.label_remap), p.name or str(p.weights_id)) for p in self.union]
        raise NotImplementedError(
            f"task {self.name!r} is a {self.shape!r} task; use .cascade for cascade tasks")

    @property
    def weights_ids(self) -> list[WeightsId]:
        """Every model the task needs, for provisioning - single, union parts, or cascade stages."""
        if self.single is not None:
            return [self.single]
        if self.union:
            return [p.weights_id for p in self.union]
        return [w for st in self.cascade
                for w in ([st.weights_id] if st.weights_id is not None
                          else [p.weights_id for p in st.union])]


#: What a registry entry's ``models`` may state per dataset: the two name components of a
#: ``<trainer>__<plans>__<configuration>`` folder that are not the configuration (that one
#: stays a per-job policy knob). Anything else is refused on load, so a misspelled key
#: cannot quietly state nothing and let the resolver refuse at run time instead.
MODEL_CHOICE_KEYS = ("trainer", "plans")


def _dataset_key(weights_id) -> str:
    """Unpadded decimal for a numeric dataset id (8 and 008 are one dataset), else as given -
    the same canonical form as ``weights_fetch.dataset_key``, restated so this module stays
    free of the fetcher."""
    t = str(weights_id).strip()
    return str(int(t)) if t.isdigit() else t


def _model_choices(raw, where: str) -> dict:
    out = {}
    for wid, choice in (raw or {}).items():
        extra = sorted(set(choice) - set(MODEL_CHOICE_KEYS))
        if extra:
            raise ValueError(f"{where}: models[{wid!r}] states {extra}; only "
                             f"{list(MODEL_CHOICE_KEYS)} name a model folder")
        out[_dataset_key(wid)] = {k: str(v) for k, v in choice.items() if v}
    return out


def _union_parts(raw) -> tuple:
    return tuple(UnionPart(weights_id=p["weights_id"],
                           label_remap={int(k): int(v) for k, v in p.get("label_remap", {}).items()},
                           name=p.get("name", ""))
                 for p in raw or ())


def _check_cascade(stages, where: str) -> None:
    """A stage does exactly one thing, and only the last may be a union: a union's output is
    a result, and nothing reads a crop box out of one (2026-09-22). Refused on load, so a
    registry typo is a message naming the task rather than a stage that runs nothing."""
    for i, st in enumerate(stages):
        does = [k for k, v in (("weights_id", st.weights_id is not None),
                               ("crop_from_task", st.crop_from_task is not None),
                               ("union", bool(st.union))) if v]
        if len(does) != 1:
            raise ValueError(f"{where}: cascade stage {i + 1} states {does or 'nothing'}; "
                             "a stage runs one model, reuses one task, or is a union")
        if st.union and i != len(stages) - 1:
            raise ValueError(f"{where}: cascade stage {i + 1} is a union but not the last "
                             "stage; only the final stage may combine models")


def _step_size(raw, where: str) -> float | None:
    if raw is None:
        return None
    step = float(raw)
    if not 0.0 < step <= 1.0:
        raise ValueError(f"{where}: step_size {raw!r} is not a tile step in (0, 1]")
    return step


class TaskCatalog:
    """The named tasks of an ecosystem, from its registry JSON."""

    def __init__(self, layout: str = "ts", path: str | Path | None = None):
        self.layout = layout
        self._specs: dict[str, TaskSpec] = {}
        self._load(Path(path) if path else self._builtin(layout))

    @staticmethod
    def _builtin(layout: str) -> Path:
        here = Path(__file__).parent / "data"
        name = {"ts": "ts_tasks.json"}.get(layout)
        if name is None:
            raise ValueError(f"no built-in task registry for layout {layout!r}")
        return here / name

    def _load(self, path: Path) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw
        items = list(items.values()) if isinstance(items, dict) else items
        for d in items:
            union = _union_parts(d.get("union"))
            cascade = tuple(CascadeStep(weights_id=st.get("weights_id"),
                                        crop_to_classes=tuple(st.get("crop_to_classes") or ()),
                                        dilation_mm=float(st.get("dilation_mm", 10.0)),
                                        crop_from_task=st.get("crop_from_task"),
                                        union=_union_parts(st.get("union")))
                            for st in d.get("cascade") or ())
            _check_cascade(cascade, f"{Path(path).name}: {d['name']}")
            self._specs[d["name"]] = TaskSpec(
                name=d["name"], lineage=d.get("lineage", "ts"), modality=d.get("modality", "CT"),
                shape=d.get("shape", "single"), single=d.get("single"), union=union,
                cascade=cascade, orientation=d.get("orientation"),
                models=_model_choices(d.get("models"), f"{Path(path).name}: {d['name']}"),
                step_size=_step_size(d.get("step_size"), f"{Path(path).name}: {d['name']}"),
                label_map={int(k): str(v) for k, v in (d.get("label_map") or {}).items()})

    def get(self, name) -> TaskSpec:
        """A task by name (or the lineage-qualified ``ts:total`` form); a TaskSpec passes through."""
        if isinstance(name, TaskSpec):
            return name
        if name in self._specs:
            return self._specs[name]
        if ":" in name:
            lineage, _, bare = name.partition(":")
            spec = self._specs.get(bare)
            if spec is not None and spec.lineage == lineage:
                return spec
        try:
            return self._specs[name]
        except KeyError:
            raise LookupError(f"unknown task {name!r}; {len(self._specs)} known, e.g. "
                              f"{sorted(self._specs)[:6]}") from None

    __getitem__ = get

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


def weights_root(layout: str = "ts", explicit=None) -> Path:
    """Explicit argument, then environment, then the layout's default location."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_vars, default = LAYOUTS.get(layout, ((), None))
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    if default is None:
        raise ModelNotFound(f"no weights root for layout {layout!r}; pass model_root or set {env_vars}")
    return default.expanduser()


# Preference order when a dataset ships several configurations and the caller named none.
# 3d_fullres is nnU-Net's default and the only one haversack runs today: the cascade needs a
# lowres prediction as an extra input channel, and 2d needs a slice-wise loop.
CONFIG_PREFERENCE = ("3d_fullres", "3d_lowres", "2d")
UNSUPPORTED_CONFIGS = {"3d_cascade_fullres": "needs the 3d_lowres prediction as an extra input channel"}


def _dataset_dirs(root: Path, weights_id) -> list[Path]:
    """``Dataset<id>_*`` directories, tolerating zero-padded ids (Dataset008 vs Dataset8)."""
    pats = [f"Dataset{weights_id}_*"]
    t = str(weights_id).strip()
    if t.isdigit():
        pats += [f"Dataset{int(t)}_*", f"Dataset{int(t):03d}_*"]
    seen: dict[str, Path] = {}
    for pat in pats:
        for d in sorted(root.glob(pat)):
            seen.setdefault(d.name, d)
    return list(seen.values()) or sorted(root.glob(str(weights_id)))


def _folder_parts(folder: Path) -> tuple[str, str, str]:
    """``(trainer, plans, configuration)`` of a ``<trainer>__<plans>__<configuration>`` folder."""
    trainer, plans, config = folder.name.split("__")
    return trainer, plans, config


def resolve_model_folder(weights_id: WeightsId, *, layout: str = "ts", model_root=None,
                         configuration: str | None = None, trainer: str | None = None,
                         plans: str | None = None) -> Path:
    """``Dataset<id>_*`` under the weights root -> its ``trainer__plans__config`` folder.

    A model folder path passes through unchanged, so a caller can point haversack straight at a
    stock nnU-Net result directory. When a dataset ships several configurations (a trained
    nnU-Net commonly has 2d / 3d_lowres / 3d_fullres / 3d_cascade_fullres), ``configuration``
    picks one; otherwise :data:`CONFIG_PREFERENCE` decides, rather than whichever sorts first.

    ``trainer`` and ``plans`` narrow the folders first (a task's ``models`` entry states them).
    If more than one folder still has the chosen configuration, this raises
    :class:`~haversack.errors.AmbiguousModel` instead of choosing: until 2026-09-21 the
    folders were keyed by configuration alone, so of TotalSegmentator v3's two ``3d_fullres``
    folders in each of Datasets 831-836 the one that sorted last - the small ResEnc model,
    ``nnUNetResEncUNetLPlans_8`` - ran in place of upstream's default, silently.
    """
    p = Path(str(weights_id)).expanduser()
    if p.is_dir() and p.name.count("__") == 2:
        t, pl, _ = _folder_parts(p)
        if (trainer and t != trainer) or (plans and pl != plans):
            raise ModelNotFound(f"{p.name} is not the {trainer or '*'}__{plans or '*'} model "
                                "this task states")
        return p
    root = Path(p) if p.is_dir() else weights_root(layout, model_root)
    matches = ([root] if p.is_dir() else _dataset_dirs(root, weights_id))
    if not matches:
        raise ModelNotFound(f"no Dataset{weights_id}_* under {root}")
    dataset = matches[0]
    configs = sorted(c for c in dataset.iterdir()
                     if c.is_dir() and not c.name.startswith(".") and c.name.count("__") == 2)
    if not configs:
        raise ModelNotFound(f"no trainer__plans__config folder in {dataset}")
    wanted = [c for c in configs
              if (trainer is None or _folder_parts(c)[0] == trainer)
              and (plans is None or _folder_parts(c)[1] == plans)]
    if not wanted:
        raise ModelNotFound(f"no {trainer or '*'}__{plans or '*'}__* model in {dataset.name}; "
                            f"have {[c.name for c in configs]}")
    by_config: dict[str, list[Path]] = {}
    for c in wanted:
        by_config.setdefault(_folder_parts(c)[2], []).append(c)

    def only(config: str) -> Path:
        found = by_config[config]
        if len(found) > 1:
            raise AmbiguousModel(
                f"{dataset.name} holds {len(found)} {config!r} models "
                f"({', '.join(f.name for f in found)}) and nothing states which to run. A "
                "catalog task states it in its registry entry as models: {id: {plans, "
                "trainer}}; to run one directly, pass its model folder path.")
        return found[0]

    if configuration is not None:
        if configuration not in by_config:
            raise ModelNotFound(f"configuration {configuration!r} not in {dataset.name}; "
                                f"have {sorted(by_config)}")
        return only(configuration)
    for name in CONFIG_PREFERENCE:
        if name in by_config:
            return only(name)
    if len(wanted) == 1:
        return wanted[0]
    why = "; ".join(f"{k} ({UNSUPPORTED_CONFIGS[k]})" for k in sorted(by_config) if k in UNSUPPORTED_CONFIGS)
    raise ModelNotFound(
        f"no runnable configuration in {dataset.name}; have {sorted(by_config)}"
        + (f" - unsupported: {why}" if why else "") + ". Pass configuration=... to choose.")


def _resolve_spec(task, catalog, progress=None) -> "TaskSpec":
    """A TaskSpec, a catalog name, or a path to a stock nnU-Net model folder. Lives here
    (torch-free) rather than in pipeline so `describe()` and the serve front-end can resolve
    a task without importing the inference stack (torch). ``progress`` goes to a first-use
    install, and is passed only when given: a plain TaskCatalog's ``get`` takes no such
    argument."""
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, Path) or (isinstance(task, str) and Path(task).expanduser().is_dir()):
        return TaskSpec.from_model_folder(task)
    return catalog.get(task) if progress is None else catalog.get(task, progress=progress)


def _uses_nnunet_preprocessing(spec) -> bool:
    return spec.lineage == "nnunetv2"
