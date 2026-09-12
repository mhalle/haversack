"""Model ecosystems: where models come from, as pluggable registries.

The mirror of :mod:`haversack.sources` - sources answer "where inputs come from",
ecosystems answer "where models come from". A :class:`ModelEcosystem` names its
tasks, installs their weights, and materializes each task's :class:`TaskSpec`.
The rule learned from the stale total_mr class map applies throughout:
**the checkpoint is the spec** - a catalog holds only what the checkpoint
cannot know (where to download it, and how to compose multiple models).

**An ecosystem is a catalog, not a runtime.** What runs a task is an *engine*
(:mod:`haversack.engines.registry`), named by ``ModelEcosystem.engine``; many
ecosystems map to one engine. The four nnU-Net catalogs below all run on
``nnunetv2``; FastSurfer and SynthStrip each bring a catalog *and* an engine.

The nnU-Net catalogs:

- ``ts`` - the TotalSegmentator catalog. Its tasks are *compositions* (unions,
  cascades, remaps) that exist only as application logic, so it carries a full
  task registry (guarded by the remap drift test).
- ``moose`` - MOOSE/moosez models. Bare, self-describing nnU-Net checkpoints
  on public release assets: the manifest holds name -> url + folder, and the
  spec is read from the installed checkpoint's own dataset.json.
- ``mrsegmentator`` - MRSegmentator's two whole-body MRI models, the same
  shape as ``moose`` (bare checkpoints, manifest of url + folder + version)
  with one fact the checkpoint cannot know: its packaging forces LPS.
- ``custom`` - local model folders the operator registers explicitly.

Engine catalogs (present only where their engine is enabled, so the catalog can
never list a task no worker can run): ``fastsurfer``, ``synthstrip``,
``voxtell`` (free-text prompts), ``monai`` (the model-zoo bundles - the first
catalog of many tasks on a NEW engine, and the first with multi-input models).

An :class:`EcosystemCatalog` federates a registry of ecosystems behind the
same interface :class:`haversack.tasks.TaskCatalog` exposes, so ``Segmenter`` and
the server run unchanged. Tasks whose weights are not yet installed are
listed but unmaterialized: ``info()`` says so without downloading, ``get()``
installs on demand (the same behavior TS weights ids always had).

Naming has three layers (user decision 2026-08-25): the **canonical name is
ecosystem-qualified** (``ts:total_fast``, ``moose:clin_ct_fast_organs``) and
is what listings, result-cache keys, and provenance carry; the **short name**
is a resolution convenience, accepted when exactly one ecosystem offers it
and refused with the qualified candidates when ambiguous - so two ecosystems
may legitimately ship the same short name; beneath both sits the **hash**,
the content-addressed result key. Nothing rejects collisions anymore; only
the ambiguous short form becomes unusable.
"""
import contextlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .engines import registry as _registry
from .errors import InputError, ModelNotFound, UnsupportedModel
from .tasks import TaskCatalog, TaskSpec

MOOSE_MANIFEST = Path(__file__).parent / "data" / "moose_weights.json"
MRSEGMENTATOR_MANIFEST = Path(__file__).parent / "data" / "mrsegmentator_weights.json"
DENTALSEGMENTATOR_MANIFEST = Path(__file__).parent / "data" / "dentalsegmentator_weights.json"
TOTALVIBE_MANIFEST = Path(__file__).parent / "data" / "totalvibe_weights.json"
CADS_MANIFEST = Path(__file__).parent / "data" / "cads_weights.json"


def manifest_entry(entries: dict, task: str, *, what: str = "", generator: str = "") -> dict:
    """One task's manifest entry, with the fields an install needs checked.

    A free function for the same reason :func:`catalog_folder` is one: the check
    has to reach every catalog, including the one that does not sit on
    :class:`ZipManifestEcosystem`. Checked at USE, not at construction - every
    catalog is built by :func:`default_ecosystems`, so raising over one bad entry
    took down `haversack tasks`, the server and the Modal worker for a fault that
    should cost exactly one task."""
    entry = entries[task]
    for key in ("url", "folder"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise ModelNotFound(
                f"{what or task}: no usable {key!r} in its manifest entry"
                + (f" - regenerate it with {generator}" if generator else ""))
    return entry


def catalog_folder(root, bucket: str, folder, *, what: str = "", generator: str = "") -> Path:
    """``<root>/<bucket>/<folder>``, with the manifest value checked rather than
    trusted.

    The value is joined onto the weights root and then handed to file operations -
    including a ``shutil.rmtree`` on the failure path - so an absolute path or a
    ``..`` segment escapes the catalog's bucket entirely (``Path("/x") / "/abs"``
    is ``/abs``). It comes from a manifest generated out of a remote archive's own
    directory names, so it is not ours to trust.

    A free function, not a method: :class:`MRSegmentatorEcosystem` deliberately
    does not sit on :class:`ZipManifestEcosystem`, and when this check lived on
    that base it simply did not apply there - which is the failure mode a shared
    check exists to prevent.
    """
    text = str(folder)
    base = Path(root).expanduser() / bucket
    out = base / text
    # Syntactic first, then containment: `.` has no path parts at all and would
    # otherwise resolve to the bucket itself, which an install would then unpack
    # over and a failure would rmtree.
    bad = (not text or Path(text).is_absolute() or ".." in Path(text).parts
           # a dot-leading segment is our own scratch namespace (.unzip-*,
           # .install-*, .lock-*), which the debris sweep deletes
           or any(p.startswith(".") for p in Path(text).parts)
           or out == base or base not in out.parents)
    if bad:
        raise ModelNotFound(
            f"manifest folder {text!r} for {what or 'a task'} is not a relative path inside "
            f"the catalog" + (f" - regenerate it with {generator}" if generator else ""))
    return out


class ModelEcosystem:
    """One model catalog: task names, weight installation, spec loading.

    An ecosystem is what the *user* selects. What actually runs its tasks is an
    :mod:`~haversack.engines.registry` engine, named by :attr:`engine` - many
    ecosystems map to one engine (``ts``, ``moose`` and ``custom`` are three
    catalogs of nnU-Net models, all run by ``nnunetv2``).

    An ecosystem whose engine has no :class:`~haversack.tasks.TaskSpec` - FastSurfer
    and SynthStrip run their own networks - sets :attr:`has_task_spec` to False
    and inherits the whole interface below unchanged; it does not need to
    override anything to refuse.

    Two axes, deliberately independent, because a catalog of engine models needs
    one of each and welding them together is what would make it fight this class:
    :attr:`has_task_spec` says whether the nnU-Net pipeline can run the task;
    :meth:`materialized` / :meth:`ensure` say where the weights come from. The
    per-task hooks :meth:`weights_identity` and :meth:`describe_task` exist for
    the same reason - a one-model engine answers both from constants, a catalog
    answers them from its manifest and its installed models.
    """

    name: str = ""
    description: str = ""
    #: Which registry engine runs this ecosystem's tasks.
    engine: str = _registry.NNUNETV2
    #: False when tasks are run by an engine's own network rather than an
    #: nnU-Net TaskSpec (drives ``spec()`` and the ``task_spec`` info flag).
    #: Independent of where the weights live - an ecosystem can have no TaskSpec
    #: and still install per task; see :meth:`materialized` / :meth:`ensure`.
    has_task_spec: bool = True
    #: Pre-install metadata for ecosystems that know it statically (a one-model
    #: engine knows its own modality and label set). A catalog leaves these None
    #: and answers per task instead - see :meth:`describe_task`.
    modality: str | None = None
    structures: list | None = None

    def tasks(self) -> list:
        raise NotImplementedError

    def weights_identity(self, task: str, root) -> list | None:
        """This task's contribution to the result-cache key, or None.

        Defaults to the engine's constant identity, which is what a one-model
        engine has (its weights ship with its image). A *catalog* of models
        overrides this to answer per task - it already holds a manifest with
        versions in memory, so this stays cheap. It must be cheap: ``info()``
        runs once per task on ``/v1/tasks`` and ``/v1/version``, so anything that
        walks the weights volume here lands on two hot endpoints.

        Returns None when the engine has no constant and the ecosystem declares
        nothing - the nnU-Net path, where the identity is computed per task from
        the spec's weights ids by :meth:`haversack.segmenter.Segmenter.describe`.
        """
        identity = _registry.ENGINES[self.engine].weights_identity
        return identity() if identity is not None else None

    def describe_task(self, task: str, root) -> dict:
        """Per-task metadata an ecosystem can read from the model itself once it
        is installed - ``modality`` and ``structures``. Empty by default; the
        nnU-Net ecosystems get theirs from the TaskSpec, one-model engines from
        their class attributes, and a catalog reads its model's own metadata
        (the "the checkpoint is the spec" rule)."""
        return {}

    def materialized(self, task: str, root) -> bool:
        """Whether spec() can answer without installing anything."""
        raise NotImplementedError

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        """Install the task's weights under ``root`` (idempotent). ``version``
        pins a release: an already-installed different version is an error,
        never silently served."""
        raise NotImplementedError

    def spec(self, task: str, root) -> TaskSpec:
        """The task's spec; requires materialized() unless the ecosystem
        carries composition data of its own.

        Refuses with :class:`UnsupportedModel` when the ecosystem has no
        TaskSpec at all (an engine's own network), and with
        ``NotImplementedError`` otherwise - a half-written ecosystem must keep
        looking like a bug, not like an unsupported model.
        """
        if not self.has_task_spec:
            eng = _registry.ENGINES[self.engine]
            raise UnsupportedModel(
                f"{self.name} is an engine, not an nnU-Net task: it runs on the "
                f"{self.engine} engine, which has no TaskSpec. Run it from the engine's "
                f"own environment (uv sync --extra {eng.extra}; Segmenter routes it there) "
                f"or deploy with {eng.enabled_env}=1 to serve it from an engine worker.")
        raise NotImplementedError

    def info(self, task: str, root) -> dict:
        """Cheap metadata that never downloads: ecosystem, engine, materialized,
        and whatever is knowable pre-install.

        Uniform for every ecosystem, engines included - one shape for clients.
        ``structures`` is always a list; ``weights_installed`` appears when the
        engine carries a constant weights identity (engines bake their weights
        into their image), and is otherwise computed per task from the spec's
        install sidecars by :meth:`haversack.segmenter.Segmenter.describe`.
        """
        out = {"name": task, "ecosystem": self.name, "engine": self.engine,
               "task_spec": self.has_task_spec,
               "materialized": self.materialized(task, root)}
        if self.modality is not None:
            out["modality"] = self.modality
        if self.structures is not None:
            out["structures"] = list(self.structures)
        identity = self.weights_identity(task, root)
        if identity is not None:
            out["weights_installed"] = identity
        if out["materialized"]:
            if self.has_task_spec:
                try:
                    spec = self.spec(task, root)
                except ModelNotFound as e:
                    # The weights are there but this build cannot choose among
                    # them (several configurations, none preferred). Report that,
                    # with the resolver's own remedy - propagating turns a real
                    # install into "unknown task" at every door, and `ensure`
                    # will not repair it because it thinks the task is installed.
                    out["unresolved"] = str(e)
                    return out
                out["modality"] = spec.modality
                # LABEL order, matching Segmenter.describe - not alphabetical. And
                # the labels themselves: a caller reading a result needs label ->
                # name, and cannot assume the labels are 1..N (feet_bones uses
                # 1-17 and 99-117) or that the names sort meaningfully (several
                # checkpoints name their structures with numbers).
                out["structures"] = [spec.label_map[k] for k in sorted(spec.label_map)]
                out["label_map"] = {str(k): spec.label_map[k] for k in sorted(spec.label_map)}
            else:
                # a catalog of engine models reads its own metadata per task
                out.update(self.describe_task(task, root))
        return out


class TSEcosystem(ModelEcosystem):
    """TotalSegmentator: composed tasks from the shipped registry JSON; weights
    by id from the release-asset manifest (license-gated models refuse with an
    actionable message). Always materialized - the composition data carries
    the label maps, and the remap drift test keeps them honest."""

    name = "ts"
    description = "TotalSegmentator task catalog"

    def __init__(self):
        self._catalog = TaskCatalog("ts")

    def tasks(self) -> list:
        return self._catalog.names()

    def materialized(self, task: str, root) -> bool:
        return True

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        from .weights_fetch import ensure_task_weights, installed_version
        paths = ensure_task_weights(task, root, catalog=self._catalog,
                                    progress=progress, tag=version)
        if version is not None:
            for pth in paths:
                rec = installed_version(pth) or {}
                # exact-or-error: an UNKNOWN installed tag (no sidecar - TS's
                # own downloader or a hand copy) must NOT silently satisfy a
                # pin. The whole point of @version is reproducibility.
                if rec.get("tag") != version:
                    have = rec.get("tag") or "unknown (no version sidecar)"
                    raise ModelNotFound(
                        f"{task}@{version}: {Path(pth).name} is installed at "
                        f"tag {have!r} - remove it to install the pinned version")

    def spec(self, task: str, root) -> TaskSpec:
        return self._catalog.get(task)


class ZipManifestEcosystem(ModelEcosystem):
    """Bare nnU-Net checkpoints published as zips, described by a JSON manifest.

    The shape MOOSE, DentalSegmentator and TotalVibeSegmentator all have, factored
    out once rather than copied per catalog: a manifest of ``name -> url + folder
    + tag`` (+ a digest when the host publishes one), an install under
    ``<root>/<bucket>/``, and a spec read from the installed checkpoint's own
    ``dataset.json`` - so there is no class map here to drift from the weights.
    The manifest holds ONLY what the checkpoint cannot say about itself: where to
    download it, which folder it unpacks to (needed *before* anything is read),
    and which release it is.

    Subclasses declare data - :attr:`name`, :attr:`bucket`, :attr:`MANIFEST`,
    :attr:`generator` - and override a method only for a genuine difference in
    the *packaging*:

    - :meth:`_unpack_into` - where the zip's top-level entries land. The default
      is ``<root>/<bucket>``, which is right when the zip carries its own
      ``Dataset<id>_<name>/`` parent (MOOSE, DentalSegmentator). A zip whose top
      level is the bare ``<trainer>__<plans>__<config>/`` folder overrides it to
      the model folder itself (TotalVibeSegmentator).
    - :meth:`spec` - to add a fact the checkpoint's own metadata carries in a
      place ``TaskSpec.from_model_folder`` does not look (an orientation).

    :class:`MRSegmentatorEcosystem` deliberately does NOT sit on this base: its
    zips are flat (``dataset.json`` at the archive root, no directory at all), so
    the unit that must land atomically is the configuration folder and the
    install is a staging directory plus one rename. That is a different install,
    not a parameter of this one.
    """

    #: Subdirectory of the weights root this catalog installs under. Keeps one
    #: catalog's ``Dataset*`` folders from colliding with another's.
    bucket: str = ""
    #: Path to the packaged manifest.
    MANIFEST: Path | None = None
    #: The tool that regenerates it, named when an install does not match it.
    generator: str = ""

    @staticmethod
    def _entries_of(raw, where) -> dict:
        """The ``{task: entry}`` map, or a clear error. A manifest is data like
        any other input here - hand-edited, or written by a generator against a
        changed upstream - so a wrong shape must say so rather than surface as a
        KeyError from an attribute lookup three calls later."""
        entries = raw.get("tasks") if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and "tasks" not in raw and all(
                isinstance(v, dict) for v in raw.values()):
            entries = raw                  # a bare {name: entry} map is a fair shape
        if not isinstance(entries, dict) or not entries:
            raise ModelNotFound(f"{where}: expected a non-empty 'tasks' object, "
                                f"found {type(entries).__name__}")
        # Shape only. Per-entry fields are checked when a task is USED, not here:
        # every catalog is constructed by default_ecosystems(), so raising on one
        # bad entry took down `haversack tasks`, the server and the Modal worker
        # over a manifest fault that should cost exactly one task.
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                raise ModelNotFound(f"{where}: task {name!r} is {type(entry).__name__}, "
                                    "expected an object")
        return entries



    def __init__(self, manifest=None):
        if not self.bucket:
            # joining "" is a no-op, so this would install Dataset* folders
            # straight into the weights root, colliding with TotalSegmentator's
            raise ValueError(f"{type(self).__name__} declares no bucket; a catalog "
                             "installs under <weights root>/<bucket>/")
        path = Path(manifest or self.MANIFEST)
        self._entries = self._entries_of(json.loads(path.read_text(encoding="utf-8")), path.name)

    def tasks(self) -> list:
        return sorted(self._entries)

    def _folder(self, task: str, root) -> Path:
        """The nnU-Net dataset folder this task installs to."""
        entry = manifest_entry(self._entries, task, what=f"{self.name} task {task!r}",
                               generator=self.generator)
        return catalog_folder(root, self.bucket, entry["folder"],
                              what=f"{self.name} task {task!r}", generator=self.generator)

    def _unpack_into(self, task: str, root) -> Path:
        """Where the zip's top-level entries land."""
        return Path(root).expanduser() / self.bucket

    def _expected_top(self, task: str, root) -> str | None:
        """The single name the archive may contain, or ``None`` when the unpack is
        already scoped to this task's own directory and so can hit nothing else."""
        if self._unpack_into(task, root) == self._folder(task, root):
            return None
        return Path(str(self._entries[task]["folder"])).parts[0]

    def _resolved(self, task: str, root):
        """This task's runnable nnU-Net model folder, or ``None``.

        Deliberately stricter than "some ``dataset.json`` exists underneath". A
        catalog whose zips unpack straight into the model folder puts the
        installer's own staging directory *inside* the directory this question is
        asked about, so a half-written unpack - or one whose archive had the wrong
        shape - would otherwise read as installed and never be retried.
        ``resolve_model_folder`` skips dot-prefixed directories, which is exactly
        what the staging directories are, and it answers the question ``spec()``
        actually needs answered.
        """
        folder = self._folder(task, root)
        if not folder.is_dir():
            return None
        # A configuration folder nnU-Net would recognize, holding its dataset.json.
        # Dot-prefixed directories are skipped, which is what excludes the
        # installer's own staging tree. Deliberately NOT resolve_model_folder:
        # that also applies CONFIG_PREFERENCE and raises when a folder ships
        # several configurations and none is preferred - a real install, which
        # must not be reported absent (and so re-downloaded, or worse).
        def usable(c) -> bool:
            """A configuration folder whose dataset.json actually PARSES. The
            file merely existing is not enough: a corrupt one installed
            "successfully", made materialized() true forever, and then raised
            JSONDecodeError out of every spec() and info() with no way back."""
            if not (c.is_dir() and not c.name.startswith(".") and c.name.count("__") == 2):
                return False
            try:
                json.loads((c / "dataset.json").read_text(encoding="utf-8"))
                return True
            except Exception:
                # deliberately broad: this is a predicate every caller treats as
                # one. A deeply nested dataset.json raises RecursionError, which
                # is not an OSError or a ValueError, and "not right now" is the
                # answer here rather than an exception out of a question.
                return False

        def is_config(c) -> bool:
            # deliberately the same predicate resolve_model_folder uses, with no
            # dataset.json requirement: a configuration folder missing one is
            # still the folder spec() will pick, and hiding it here made
            # materialized() true while every spec() raised, unrepairably
            return (c.is_dir() and not c.name.startswith(".")
                    and c.name.count("__") == 2)

        try:
            configs = [c for c in sorted(folder.iterdir()) if is_config(c)]
        except OSError:
            # the tree moved under us (a concurrent install's rename). This is a
            # predicate every caller treats as one; "not right now" is the answer,
            # not an exception out of a question.
            return None
        if not configs:
            return None
        # The one spec() will load, not the first alphabetically. resolve_model_folder
        # applies CONFIG_PREFERENCE, and `..._2d` sorts before `..._3d_fullres`: a
        # valid 2d beside a corrupt 3d_fullres made materialized() true while every
        # spec() and info() raised, which is precisely what validating the JSON was
        # supposed to prevent. When the preference cannot choose, the first is
        # returned and spec() reports the ambiguity itself.
        from .tasks import CONFIG_PREFERENCE
        by_config = {c.name.rsplit("__", 1)[1]: c for c in configs}
        chosen = next((by_config[name] for name in CONFIG_PREFERENCE if name in by_config),
                      configs[0])
        return chosen if usable(chosen) else None

    def materialized(self, task: str, root) -> bool:
        if task not in self._entries:
            return False
        return self._resolved(task, root) is not None

    @contextlib.contextmanager
    def _install_lock(self, task: str, root):
        """Serialize installs of ONE model folder, across threads and processes.

        There was no lock here, and two callers wanting the same task at once is
        the ordinary case - two `prepare` requests, a prepare racing an
        on-demand install, two workers on a shared weights volume. Without it,
        one install's failure cleanup deletes another's finished weights, and
        four at once leave nothing installed while one reports success.

        The lock is per model folder, so two different tasks in one bucket still
        install in parallel (they touch disjoint names, which is what
        ``expect_top`` guarantees). An advisory lock through
        :mod:`haversack.filelock` - ``flock``, or ``msvcrt`` on Windows, where
        this used to fall back to no lock at all. Only where the lock file itself
        cannot be opened does the install proceed unlocked, which is no worse
        than before.
        """
        from . import filelock
        bucket = Path(root).expanduser() / self.bucket
        bucket.mkdir(parents=True, exist_ok=True)
        entry = manifest_entry(self._entries, task, what=f"{self.name} task {task!r}",
                               generator=self.generator)
        stem = str(entry["folder"]).replace("/", "_")
        with filelock.held(bucket / f".lock-{stem}"):
            yield

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        with self._install_lock(task, root):
            self._ensure_locked(task, root, progress=progress, version=version)

    def _ensure_locked(self, task: str, root, progress=None, version=None) -> None:
        from .weights_fetch import _write_sidecar, installed_version
        entry = manifest_entry(self._entries, task, what=f"{self.name} task {task!r}",
                               generator=self.generator)
        if version is not None:
            # check the INSTALLED tag, always - matching the manifest tag is
            # not proof the bytes ON DISK are that release (a regenerated
            # manifest bumps the tag; the old folder keeps its old sidecar)
            rec = installed_version(self._folder(task, root)) or {}
            if rec.get("tag") == version:
                return                     # provably the pinned bytes
            if self.materialized(task, root):
                have = rec.get("tag") or "unknown (no version sidecar)"
                raise ModelNotFound(
                    f"{task}@{version}: installed at tag {have!r} - remove "
                    f"{self._folder(task, root).name} to install the pinned version")
            if version != entry.get("tag"):
                raise ModelNotFound(
                    f"{task}@{version}: this manifest offers tag "
                    f"{entry.get('tag')!r} only")
        if self.materialized(task, root):
            return
        dest_parent = self._unpack_into(task, root)
        folder = self._folder(task, root)
        # Decided before anything is created, and only trustworthy because the
        # install lock is held: unlocked, another process could finish its own
        # install here between this line and the cleanup below.
        pre_existing = folder.exists()
        dest_parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_and_extract_zip(entry["url"], dest_parent, progress=progress,
                                      sha256=entry.get("sha256"), md5=entry.get("md5"),
                                      expect_top=self._expected_top(task, root),
                                      work_dir=Path(root).expanduser() / self.bucket,
                                      hint=f"The manifest may be stale; regenerate it with "
                                           f"{self.generator}")
            # What spec() needs is a *resolvable* model folder, so that is what is
            # checked - not merely that the directory exists. For a catalog whose
            # zips have no Dataset parent, that directory is one we just created,
            # so its existence proves nothing about what landed inside it.
            if self._resolved(task, root) is None:
                raise ModelNotFound(
                    f"{self.name} asset for {task!r} did not unpack to an nnU-Net model "
                    f"folder at {entry['folder']!r} - the manifest may be stale; "
                    f"regenerate it with {self.generator}")
        except BaseException:
            # Remove only what THIS call created. A directory that was already
            # there is the user's - possibly weights installed by hand, or by an
            # older version in a layout this one does not resolve - and deleting
            # it because a download failed (an offline machine is enough) would
            # destroy data no retry can bring back. BaseException, not Exception,
            # so a Ctrl-C part-way through an unpack is cleaned up too.
            if not pre_existing:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        # whichever digest was actually verified - recording only sha256 made an
        # md5-checked install (every Zenodo asset) read as unverified in provenance
        _write_sidecar(folder, task, str(entry.get("tag", "unknown")),
                       {"url": entry["url"], "md5": entry.get("md5")},
                       entry.get("sha256"))

    def spec(self, task: str, root) -> TaskSpec:
        if not self.materialized(task, root):
            raise ModelNotFound(
                f"{self.name} task {task!r} is not installed under {root}; prepare it "
                "first (weights install on demand when the task runs)")
        return TaskSpec.from_model_folder(self._folder(task, root), name=task)

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        entry = self._entries.get(task) or {}
        out["tag"] = entry.get("tag")
        # The licence travels with the task. These are third-party weights and
        # some carry an attribution condition (DentalSegmentator's are CC BY 4.0);
        # a note in a docstring does not discharge that for whoever consumes the
        # output, so whatever the manifest recorded is published here.
        for key in ("license", "release", "description", "modality", "dataset_id"):
            if entry.get(key):
                out[key] = entry[key]
        return out


class MooseEcosystem(ZipManifestEcosystem):
    """MOOSE (moosez): bare nnU-Net checkpoints from public GitHub release
    assets, installed under ``<root>/moose/<Dataset folder>`` and read through
    ``TaskSpec.from_model_folder`` - labels come from each checkpoint's own
    dataset.json, so there is no class map here to drift. The manifest
    (``data/moose_weights.json``, regenerated by tools/gen_moose_manifest.py
    from the moosez registry) holds only name -> url + folder + release tag."""

    name = "moose"
    description = "MOOSE (moosez) model zoo"
    bucket = "moose"
    MANIFEST = MOOSE_MANIFEST
    generator = "tools/gen_moose_manifest.py"

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        m = re.match(r"(clin|preclin)_(ct|mr|pt|fdg_pt|pt_fdg)_", task)
        if "modality" not in out and m:
            out["modality"] = m.group(2).upper().replace("FDG_PT", "PT").replace("PT_FDG", "PT")
        return out


class MRSegmentatorEcosystem(ModelEcosystem):
    """MRSegmentator (Haentze et al., Charite; Apache-2.0): two stock nnU-Net v2
    checkpoints for abdominal / pelvic / thoracic MRI - ``base`` (40 structures,
    also usable on CT) and ``body_comp`` (10 body-composition classes).

    Same shape as :class:`MooseEcosystem` - bare checkpoints on public assets,
    labels read from each installed checkpoint's own dataset.json - with two
    differences that are properties of the *packaging*, not of the checkpoint:

    - **The zips are flat.** ``dataset.json``, ``plans.json``, ``version.json``
      and ``fold_0..4/`` sit at the top level, so the unit that must land
      atomically is the *configuration* folder, not a ``Dataset*`` parent. Each
      model is installed under ``<root>/mrsegmentator/<Dataset>/<trainer>__
      <plans>__<config>/`` (the names come from the checkpoint itself - see
      tools/gen_mrsegmentator_manifest.py) via a staging directory and one
      rename, and ``materialized()`` looks only at that folder.
    - **The reader forces LPS.** MRSegmentator replaces nnU-Net's reader with one
      that ``DICOMOrient``s every input to LPS before the network, while the plans
      still declare ``SimpleITKIO`` (no reorientation). Both models were trained
      without mirroring, so left/right is a real fact to the network, and following
      the declared reader would feed it acquisition order instead. The spec
      therefore carries ``orientation="LPS"`` - the one line the checkpoint cannot
      say about itself.

    The manifest tag is upstream's own ``weights_version`` (the number inside each
    zip's ``version.json``), so a pinned install is checked against the bytes on
    disk twice: the install sidecar, and the version file the zip itself carries.

    Folds: upstream's default is the five-fold ensemble with mirroring TTA; its
    ``--fast`` runs fold 0 alone with no mirroring. haversack's default (fold 0, no
    mirroring, step 0.5) is the latter; pass ``folds=[0, 1, 2, 3, 4]`` for the
    ensemble.
    """

    name = "mrsegmentator"
    description = "MRSegmentator whole-body MRI models"
    modality = "MR"
    _BUCKET = "mrsegmentator"

    def __init__(self, manifest=None):
        path = Path(manifest or MRSEGMENTATOR_MANIFEST)
        self._entries = ZipManifestEcosystem._entries_of(json.loads(path.read_text(encoding="utf-8")), path.name)

    def tasks(self) -> list:
        return sorted(self._entries)

    def _folder(self, task: str, root) -> Path:
        entry = manifest_entry(self._entries, task, what=f"mrsegmentator task {task!r}",
                               generator="tools/gen_mrsegmentator_manifest.py")
        return catalog_folder(root, self._BUCKET, entry["folder"],
                              what=f"mrsegmentator task {task!r}",
                              generator="tools/gen_mrsegmentator_manifest.py")

    def materialized(self, task: str, root) -> bool:
        if task not in self._entries:
            return False
        # the file must PARSE, not merely exist: a corrupt one installed
        # "successfully" and then raised out of every spec() with no way back
        ds = self._folder(task, root) / "dataset.json"
        try:
            json.loads(ds.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False

    @staticmethod
    def _installed_weights_version(folder: Path) -> str | None:
        """What the zip's own ``version.json`` says is installed here, if anything."""
        vf = Path(folder) / "version.json"
        if not vf.is_file():
            return None
        try:
            v = json.loads(vf.read_text(encoding="utf-8")).get("weights_version")
        except (json.JSONDecodeError, OSError):
            return None
        return None if v is None else str(v)

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        import os
        from .weights_fetch import _write_sidecar, installed_version
        entry = manifest_entry(self._entries, task, what=f"mrsegmentator task {task!r}",
                               generator="tools/gen_mrsegmentator_manifest.py")
        folder = self._folder(task, root)
        if version is not None:
            # the INSTALLED bytes decide, never the manifest (see MooseEcosystem)
            rec = installed_version(folder) or {}
            if rec.get("tag") == version:
                return
            if self.materialized(task, root):
                have = rec.get("tag") or self._installed_weights_version(folder) \
                    or "unknown (no version sidecar)"
                raise ModelNotFound(
                    f"{task}@{version}: installed at version {have!r} - remove "
                    f"{folder} to install the pinned version")
            if version != entry.get("tag"):
                raise ModelNotFound(
                    f"{task}@{version}: this manifest offers version "
                    f"{entry.get('tag')!r} only")
        if self.materialized(task, root):
            return
        # A flat zip: extract into a staging sibling, then ONE rename makes the
        # configuration folder appear whole - never a dataset.json without folds.
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=folder.parent, prefix=".install-"))
        try:
            _download_and_extract_zip(entry["url"], staging, progress=progress,
                                      sha256=entry.get("sha256"),
                                      # sweep and stage in the BUCKET: the
                                      # destination here is itself a `.install-*`
                                      # directory, so scratch left inside it is
                                      # exactly what nothing else can find
                                      work_dir=folder.parent)
            if not (staging / "dataset.json").is_file() or not any(staging.glob("fold_*")):
                raise ModelNotFound(
                    f"mrsegmentator asset for {task!r} did not unpack to a flat nnU-Net "
                    "configuration folder (dataset.json + fold_*) - the manifest may be "
                    "stale; regenerate it with tools/gen_mrsegmentator_manifest.py")
            have = self._installed_weights_version(staging)
            if have is not None and have != str(entry.get("tag")):
                raise ModelNotFound(
                    f"mrsegmentator asset for {task!r} carries weights_version {have!r} but "
                    f"the manifest says {entry.get('tag')!r} - regenerate the manifest")
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            os.replace(staging, folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        _write_sidecar(folder, task, str(entry.get("tag", "unknown")), {"url": entry["url"]},
                       entry.get("sha256"))

    def spec(self, task: str, root) -> TaskSpec:
        import dataclasses
        if not self.materialized(task, root):
            raise ModelNotFound(
                f"mrsegmentator task {task!r} is not installed under {root}; prepare it "
                "first (weights install on demand when the task runs)")
        spec = TaskSpec.from_model_folder(self._folder(task, root), name=task)
        return dataclasses.replace(spec, orientation="LPS")

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        entry = self._entries.get(task, {})
        out["tag"] = entry.get("tag")
        out["orientation"] = "LPS"
        return out


class DentalSegmentatorEcosystem(ZipManifestEcosystem):
    """DentalSegmentator (Dot et al., Journal of Dentistry 2024; weights CC BY 4.0):
    one stock nnU-Net v2 model for dento-maxillo-facial CT and CBCT - upper skull,
    mandible, upper teeth, lower teeth and the mandibular canal.

    **Its plans permute the axes** (``transpose_forward=(1, 0, 2)``), which
    haversack refuses by default, so running it needs ``allow_transpose=True``
    (the command line and the server each expose it as ``--allow-transpose``). The transposed path has
    since been checked against nnU-Net's own predictor on this model - 99.86 % of
    voxels, Dice 0.999 on the two large structures - but the gate is a
    project-wide policy and is left as it is.

    The reference client (the DentalSegmentator Slicer extension) also removes
    connected components under 60 mm³ from every structure except the mandibular
    canal; haversack does no post-processing, so its output carries speckle that
    the reference tool would have cleaned.

    Otherwise the plainest possible catalog on :class:`ZipManifestEcosystem`: the Zenodo
    asset carries its own ``Dataset112_DentalSegmentator_v100/`` parent, so the
    default install (unpack into ``<root>/dentalsegmentator/``) is already right
    and only the manifest differs. Two details are the packaging's, not the
    model's: the archive is zipped by macOS and carries an ``__MACOSX`` tree,
    which the installer drops, and Zenodo publishes an md5 rather than a sha256,
    which is therefore what is verified.
    """

    name = "dentalsegmentator"
    description = "DentalSegmentator CBCT / CT dento-maxillo-facial model"
    modality = "CT"                       # CBCT too; the checkpoint declares "CT"
    bucket = "dentalsegmentator"
    MANIFEST = DENTALSEGMENTATOR_MANIFEST
    generator = "tools/gen_dentalsegmentator_manifest.py"




class TotalVibeEcosystem(ZipManifestEcosystem):
    """TotalVibeSegmentator (Graf et al., Apache-2.0): stock nnU-Net v2 models for
    whole-torso VIBE MRI, plus the CT and single-organ models published beside
    them - ``vibe`` (72 structures, axial), ``vibe_sagittal``, ``ct_bones``,
    ``body_regions``, ``vertebrae``, ``feet_bones`` and ``pancreas``.

    Two things differ from the MOOSE shape, both properties of the packaging:

    - **The zip has no Dataset parent.** Its top level is the
      ``<trainer>__<plans>__<config>`` folder (five of the seven also carry an
      ``other_downloads.json`` beside it, naming the extra-fold assets upstream
      would fetch next), so ``_unpack_into`` puts the contents inside
      the ``Dataset<id>`` directory this catalog creates - which is what
      upstream's own downloader does, and what makes the result an ordinary
      nnU-Net results tree.
    - **The orientation is per model and the checkpoint states it.** Upstream
      reorients every input to a model's own axis codes before the network (RAS
      for the whole-body and CT models, LPS for the body-region, vertebra and
      pancreas ones) while the plans still declare ``SimpleITKIO``, which does not
      reorient. That is MRSegmentator's LPS problem exactly, except here upstream
      records the answer in each ``dataset.json`` - so :meth:`spec` reads it and
      nothing is hardcoded.

    The release publishes more models than this catalog offers, and the manifest
    accounts for **every** one of them: seven tasks, and twelve recorded
    exclusions with a reason each - multi-channel (the nnU-Net path takes one
    input image), unreadable (published without ``dataset.json``/``plans.json``),
    or single-channel but outside the curated set. ``tools/gen_totalvibe_manifest.py``
    verifies each reason against the asset and refuses to run if the release
    grows a model that is in neither list, so a drop cannot be silent. Upstream's
    extra ``_1``/``_2`` assets are further folds; haversack's default is fold 0,
    so the base asset is what installs.

    Three assets are published with no digest, and the manifest says which
    (``unverified``): their download is checked against nothing, because GitHub
    states nothing to check it against.

    ``vibe_sagittal`` and ``pancreas`` permute the axes in their plans, so they
    need ``allow_transpose=True`` like :class:`DentalSegmentatorEcosystem`; the
    other five run by default.

    ``feet_bones`` carries a ``labels_mapping`` that upstream applies after
    inference, renumbering 19 of its structures into the 99-117 range. haversack
    does not apply it, so its label INTEGERS differ from upstream's for that task
    while the names are unchanged and correctly paired - ``mask("100")`` finds the
    same structure either way, but a numerical diff against an upstream NIfTI
    will not line up.

    ``body_regions`` and ``feet_bones`` report numbered structures (``"1"``,
    ``"2"``, ...) because that is what their checkpoints' own ``dataset.json``
    calls them. Upstream names those regions in its README only, and a class map
    written here from prose is the stale-class-map failure this package refuses
    everywhere else - so the numbers stand until upstream names them.
    """

    name = "totalvibe"
    description = "TotalVibeSegmentator whole-body VIBE MRI and CT models"
    bucket = "totalvibe"
    MANIFEST = TOTALVIBE_MANIFEST
    generator = "tools/gen_totalvibe_manifest.py"

    def _unpack_into(self, task: str, root) -> Path:
        # the archive's top level is the configuration folder itself; the
        # Dataset<id> parent is this catalog's to create
        return self._folder(task, root)

    def _declared_orientation(self, task: str, root) -> str | None:
        """The axis codes the installed checkpoint's own dataset.json asks for."""
        from .tasks import resolve_model_folder
        try:
            folder = resolve_model_folder(self._folder(task, root))
            raw = json.loads((folder / "dataset.json").read_text(encoding="utf-8")).get("orientation")
        except (OSError, ValueError, ModelNotFound):
            return None
        code = "".join(str(c) for c in raw).upper() if isinstance(raw, (list, tuple)) \
            else str(raw or "").upper()
        # one letter per anatomical axis, each axis named once - anything else is
        # not an orientation and must not reach DICOMOrient
        pairs = ("RL", "AP", "SI")
        if len(code) == 3 and all(sum(c in pair for c in code) == 1 for pair in pairs):
            return code
        return None

    def spec(self, task: str, root) -> TaskSpec:
        import dataclasses
        spec = super().spec(task, root)
        # Both facts are applied HERE rather than in info(), so `describe()` -
        # which builds its answer from the spec - cannot disagree with `info()`,
        # and cannot report a modality that flips once the weights are installed.
        fields = {}
        orientation = self._declared_orientation(task, root)
        if orientation is not None:
            fields["orientation"] = orientation
        modality = (self._entries.get(task) or {}).get("modality")
        if modality:
            fields["modality"] = modality
        return dataclasses.replace(spec, **fields) if fields else spec

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        # modality/description/dataset_id come from the manifest through the base:
        # a channel named "any" or "ct" is a contrast claim, not a modality, and
        # upstream states the modality in prose.
        if out["materialized"]:
            orientation = self._declared_orientation(task, root)
            if orientation:
                out["orientation"] = orientation
        return out


class CADSEcosystem(ZipManifestEcosystem):
    """CADS (Xu et al., arXiv 2507.22953): nine nnU-Net v2 ResEnc-L checkpoints, one per model
    group (T551-T559), naming 167 CT structures between them - ``organs``, ``vertebrae``,
    ``cardiac``, ``muscles``, ``ribs``, ``oar``, ``head``, ``headneck`` and ``bodyregions``.
    The weights are upstream's ``open`` release (CC BY-SA 4.0). Its ``research`` weights are
    non-commercial and its ``reference`` weights add challenge datasets under their own terms,
    and haversack has no license gate to offer either behind.

    The packaging is MOOSE's: a ``Dataset<id>`` parent, and macOS zip litter the installer
    drops. One fact is missing from the checkpoint's account of itself, and :meth:`spec`
    states it: **CADS preprocesses the TotalSegmentator way** - reorient to RAS, resample with
    TotalSegmentator's ``change_spacing`` - while its plans say ``SimpleITKIO`` and its
    ``dataset.json`` declares no orientation. Read as a stock nnU-Net model, every
    left/right structure came out on the wrong side (mean Dice 0.034 against upstream). With
    lineage ``ts`` and ``interp="nearest"`` - upstream restores its labels nearest-neighbor -
    all nine matched upstream's own inference at 0.998-1.0 over the whole volume of a
    chest-abdomen-pelvis and a whole-body CT (2026-09-11), and ``organs`` through this
    catalog on MPS agreed on 99.9995 % of voxels (2026-09-12). haversack's default linear
    restore moves boundaries off upstream's: ``organs`` at mean Dice 0.970, its adrenals 0.91.

    Three differences from upstream remain, and are documented rather than hidden:

    - CADS's copy of ``change_spacing`` pads with scipy's ``constant`` 0 HU where
      TotalSegmentator's uses ``nearest``, and haversack resamples the TotalSegmentator way.
      Only short volumes notice: a 34-slice head CT matched at 0.977-0.998.
    - Upstream runs ``head`` and ``headneck`` only when ``cardiac`` finds a brain, crops them
      to a box around it, and removes small components from five groups. haversack runs each
      task as asked and post-processes nothing.
    - Each task is one model, and there is no combined task. The nine overlap by design -
      ``bodyregions`` holds the cavities and tissues the other groups subdivide - so one label
      map of all nine loses structures: upstream's own paints the thoracic cavity over every
      lung lobe.
    """

    name = "cads"
    description = "CADS whole-body CT models (open weights, CC BY-SA 4.0)"
    bucket = "cads"
    MANIFEST = CADS_MANIFEST
    generator = "tools/gen_cads_manifest.py"

    def spec(self, task: str, root) -> TaskSpec:
        import dataclasses
        # The TotalSegmentator lineage - RAS, corner-aligned resampling, no crop to nonzero -
        # is what CADS's own preprocessing does, and what its checkpoints do not say.
        return dataclasses.replace(super().spec(task, root), lineage="ts")


class CustomEcosystem(ModelEcosystem):
    """The operator's own model folders: always materialized, nothing to install.
    The folder is read through from_model_folder, so the checkpoint's
    dataset.json is the spec here too."""

    name = "custom"
    description = "operator-registered local nnU-Net model folders"

    def __init__(self, models: dict | None = None):
        self._models = {str(k): Path(v) for k, v in (models or {}).items()}

    def tasks(self) -> list:
        return sorted(self._models)

    def materialized(self, task: str, root) -> bool:
        return task in self._models and self._models[task].is_dir()

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        if not self.materialized(task, root):
            raise ModelNotFound(f"custom task {task!r}: folder "
                                f"{self._models.get(task)} does not exist")
        if version is not None:
            from .weights_fetch import installed_version
            rec = installed_version(self._models[task]) or {}
            if rec.get("tag") != version:
                raise ModelNotFound(
                    f"{task}@{version}: custom folder records "
                    f"{rec.get('tag') or 'no version metadata'}")

    def spec(self, task: str, root) -> TaskSpec:
        self.ensure(task, root)
        return TaskSpec.from_model_folder(self._models[task], name=task)


#: Archive members macOS's Finder adds to a zip: an ``__MACOSX/`` tree of
#: AppleDouble sidecars, and ``.DS_Store``. They are never part of a model, and
#: unpacking them would move a stray ``__MACOSX`` directory into the weights root
#: beside the Dataset folder (the DentalSegmentator asset is zipped this way).
_JUNK_DIRS = ("__MACOSX",)
_JUNK_NAMES = (".DS_Store",)


def _is_junk(name: str) -> bool:
    head = name.split("/", 1)[0]
    base = name.rstrip("/").rsplit("/", 1)[-1]
    return head in _JUNK_DIRS or base in _JUNK_NAMES or base.startswith("._")


def _verify_digest(path: Path, algo: str, expected: str) -> None:
    """Hash ``path`` with ``algo`` and refuse a mismatch. Which algorithm is not
    ours to choose: we verify what the host publishes - GitHub release assets
    carry a sha256 digest, Zenodo's file API carries an md5 - and verifying the
    published one beats recording a stronger one we would have to compute by
    downloading the asset ourselves at manifest time."""
    import hashlib
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        path.unlink(missing_ok=True)
        raise InputError(f"weights zip {algo} digest mismatch: expected {expected}, "
                         f"got {h.hexdigest()}")


#: Ceiling on one weights download. Generous - the largest asset any shipped
#: catalog names is about 1 GB - but finite, because most entries have no digest.
def _weights_cap() -> int:
    """The per-download ceiling, read at call time and never fatal.

    Parsed at import, a malformed HAVERSACK_MAX_WEIGHTS_GB took down `import
    haversack.ecosystems` - and with it the CLI, the server and the Modal
    worker - over a typo in an environment variable."""
    raw = os.environ.get("HAVERSACK_MAX_WEIGHTS_GB") or "24"
    try:
        gb = float(raw)
    except ValueError:
        gb = 24.0
    return int(max(gb, 1.0) * (1 << 30))

#: How long install scratch has to be untouched before a later install removes it.
#: Long enough that a slow download in another process is never swept.
_DEBRIS_AGE_S = 6 * 3600


def _sweep_install_debris(scratch: Path) -> None:
    """Remove staging left by a killed install: a `.unzip-*` tree or a part
    downloaded `tmp*.zip`. A SIGKILL cannot run the cleanup, and nothing else
    ever looks here - `cache clean` deliberately does not sweep the weights root -
    so without this the debris is permanent and a gigabyte at a time."""
    import time as _time
    cutoff = _time.time() - _DEBRIS_AGE_S
    for child in [*scratch.glob(".unzip-*"), *scratch.glob(".install-*")]:
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass
    for child in scratch.glob("tmp*.zip"):
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink(missing_ok=True)
        except OSError:
            pass


def _download_and_extract_zip(url: str, dest_parent: Path, *, progress=None,
                              sha256: str | None = None, md5: str | None = None,
                              expect_top: str | None = None, work_dir: Path | None = None,
                              hint: str = "") -> None:
    """Fetch a release zip and unpack it under ``dest_parent`` atomically,
    verifying the digest the manifest supplies (``sha256``, or ``md5`` where
    that is all the host publishes) and refusing any member whose path would
    escape (zip-slip).

    ``expect_top`` is the one top-level name the archive is allowed to contain.
    It matters because unpacking REPLACES a directory of that name: with a stale
    manifest, an archive that carries some other task's ``Dataset*`` folder would
    otherwise overwrite that task's weights and leave it silently running this
    one's model - a wrong-model-no-error failure, the worst kind here. Checked
    before anything is moved. ``None`` means the destination is already scoped to
    one task, so there is nothing of anyone else's to hit.

    ``work_dir`` is where the download and the unpack are staged; it defaults to
    ``dest_parent``, but a caller whose destination IS a model folder passes the
    directory above it, so an interrupted install never leaves scratch inside a
    model."""
    import shutil
    import tempfile
    from . import fetchlib
    from .progress import InstallProgress
    say = InstallProgress.of(progress)
    scratch = Path(work_dir) if work_dir is not None else dest_parent
    scratch.mkdir(parents=True, exist_ok=True)
    _sweep_install_debris(scratch)
    cap = _weights_cap()
    what = f"downloading {url.rsplit('/', 1)[-1]}"
    say(what)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=scratch, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with fetchlib.open(url, timeout=1800) as r:
                total, done = fetchlib.content_length(r), 0
                say.download(done, total, what)
                while chunk := r.read(1 << 20):
                    tmp.write(chunk)
                    done += len(chunk)
                    if done > cap:
                        # 22 of the shipped manifest entries carry no digest, and a
                        # release asset can be replaced under a published tag, so a
                        # download with nothing to check it against needs a ceiling.
                        raise InputError(
                            f"{url.rsplit('/', 1)[-1]}: over the {cap} byte "
                            "cap for a weights download (HAVERSACK_MAX_WEIGHTS_GB)")
                    say.download(done, total, what)
                say.download(done, done, what)
        except BaseException as e:
            # BaseException: a Ctrl-C here used to leak the part-downloaded file,
            # and these are 100 MB - 1 GB, so a few interrupted attempts strand
            # gigabytes that nothing sweeps.
            tmp.close()                    # Windows cannot unlink an open file
            tmp_path.unlink(missing_ok=True)
            from .errors import Cancelled
            if isinstance(e, (Cancelled, KeyboardInterrupt, SystemExit)):
                raise                      # a cancel is not a network failure; say so
            raise InputError(f"weights download failed: {e}") from e
    if sha256:
        _verify_digest(tmp_path, "sha256", sha256)
    if md5:
        _verify_digest(tmp_path, "md5", md5)
    say.unpack(f"unpacking {url.rsplit('/', 1)[-1]}")
    # extract into a sibling temp dir, then os.replace into place: an
    # interrupted unpack must never leave a folder that materialized() calls
    # complete (dataset.json present, checkpoints missing) - the exact trap
    # fetch_one's temp+rename avoids on the TS side
    staging = Path(tempfile.mkdtemp(dir=scratch, prefix=".unzip-"))
    try:
        _unpack(tmp_path, staging, dest_parent, url, expect_top, hint)
    finally:
        tmp_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _unpack(tmp_path: Path, staging: Path, dest_parent: Path, url: str,
            expect_top: str | None, hint: str) -> None:
    """Open, check and move the archive's contents.

    Wrapped by its caller because everything here can fail on bytes we did not
    write: a host answering 200 with an HTML error page, a truncated body, a
    member name the filesystem will not take. Those are `InputError` - the
    download was bad - not tracebacks. `NotImplementedError` matters especially:
    `UnsupportedModel` inherits it, so a zip using a compression method the
    stdlib lacks would otherwise be indistinguishable from this package's own
    "half-written ecosystem" marker.
    """
    import os
    import shutil
    import zipfile
    try:
        with zipfile.ZipFile(tmp_path) as z:
            base = staging.resolve()
            members = [m for m in z.infolist() if not _is_junk(m.filename)]
            for m in members:
                target = (staging / m.filename).resolve()
                if not target.is_relative_to(base):
                    raise InputError(f"zip member escapes destination: {m.filename!r}")
            z.extractall(staging, members=members)
        tops = sorted(c.name for c in staging.iterdir())
        if expect_top is not None and tops != [expect_top]:
            # ModelNotFound, not InputError: the archive is not what the catalog
            # says it is, which is a weights problem, not the caller's input.
            raise ModelNotFound(
                f"{url.rsplit('/', 1)[-1]} unpacks to {tops}, not {[expect_top]} - refusing "
                "to install it, because moving those names into place would replace whatever "
                "already owns them" + (f". {hint}" if hint else ""))
        for child in staging.iterdir():        # move the unpacked top-level
            dest = dest_parent / child.name     # folder(s) into place atomically
            if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
                dest.unlink(missing_ok=True)   # rmtree silently no-ops on these
            elif dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(child, dest)
    except (ModelNotFound, InputError):
        raise
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError) as e:
        raise InputError(
            f"{url.rsplit('/', 1)[-1]} could not be unpacked ({type(e).__name__}: {e}) - the "
            "download is not the archive this catalog expects"
            + (f". {hint}" if hint else "")) from e


class EngineEcosystem(ModelEcosystem):
    """An ecosystem whose tasks are run by an engine's own network rather than an
    nnU-Net TaskSpec.

    That is the ONLY thing this class asserts. Whether the weights ship with the
    engine's image is a separate question, answered by
    :class:`ImageBakedEcosystem` below - an engine ecosystem is free to install
    per task instead (a catalog of models does), and conflating the two is what
    would force a many-task engine catalog to fight this class.
    """

    has_task_spec = False


class ImageBakedEcosystem(EngineEcosystem):
    """An engine ecosystem whose weights ship inside the engine's own image:
    always materialized, nothing to install. The one-model engines
    (FastSurfer, SynthStrip, VoxTell) are all this shape.

    Subclasses declare data only - name, engine, task_names, modality, structures.
    """

    #: Task names this engine offers.
    task_names: tuple = ()

    def tasks(self) -> list:
        return list(self.task_names)

    def materialized(self, task: str, root) -> bool:
        return True                      # weights ship with the engine's image

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        return None


class FastSurferEcosystem(ImageBakedEcosystem):
    """FastSurfer whole-brain parcellation (2.5D view-aggregation, not nnU-Net).
    Its checkpoints are baked into the FastSurfer worker image."""

    name = "fastsurfer"
    engine = "fastsurfer"
    description = "FastSurfer whole-brain parcellation (engine)"
    task_names = ("brain",)
    modality = "MR (T1)"

    @property
    def structures(self) -> list:
        """The real DKTatlas label names, from the engine's own LUT - so a client
        can enumerate them like any other task's."""
        from .engines.fastsurfer import load_lut
        return sorted(v["name"] for v in load_lut().values())


def engine_of(ecosystem) -> str:
    """The engine name an ecosystem declares. Ecosystems are duck-typed here (an
    object with ``name``/``tasks()``/``spec()`` is enough), so one that predates
    the engine layer - or a test stand-in - falls back to the default engine."""
    return getattr(ecosystem, "engine", _registry.NNUNETV2)


def registry(ecosystems=None) -> dict:
    """Normalize a list of ecosystems into ``{name: ecosystem}``. Duplicate
    ecosystem names are rejected; duplicate *task* names across ecosystems are
    fine - the canonical ``eco:task`` form disambiguates, and only the short
    form goes ambiguous."""
    out = {}
    for e in (default_ecosystems() if ecosystems is None else list(ecosystems)):
        if not e.name or ":" in e.name or e.name in out:
            raise ValueError(f"bad or duplicate ecosystem name {e.name!r}")
        # An unknown engine would route silently to the default worker at spawn;
        # catching the typo here keeps "which engine runs this?" answerable.
        if engine_of(e) not in _registry.ENGINES:
            raise ValueError(f"ecosystem {e.name!r} declares unknown engine "
                             f"{engine_of(e)!r}; known: {sorted(_registry.ENGINES)}")
        out[e.name] = e
    return out


class SynthStripEcosystem(ImageBakedEcosystem):
    """SynthStrip brain extraction (skull-strip): a contrast-agnostic learned
    brain-mask UNet, not nnU-Net. Weights are baked into the worker image."""

    name = "synthstrip"
    engine = "synthstrip"
    description = "SynthStrip brain extraction / skull-strip (engine)"
    task_names = ("mask",)
    modality = "MR (any contrast)"
    structures = ["Brain"]


class VoxTellEcosystem(ImageBakedEcosystem):
    """VoxTell free-text promptable segmentation. Unlike every other catalog entry,
    ``voxtell:text`` has **no fixed label set** - the prompts are an input, passed
    as ``options={"prompts": [...]}``, and they hash into the result-cache key. So
    ``structures`` is deliberately absent from info(): what it segments is whatever
    the caller asks for."""

    name = "voxtell"
    engine = "voxtell"
    description = ("VoxTell free-text promptable segmentation (engine); "
                   'prompts are an input: options={"prompts": ["liver", ...]}')
    task_names = ("text",)
    modality = "CT / MR / PET"


MONAI_MANIFEST = Path(__file__).parent / "data" / "monai_bundles.json"


#: Whose job co-registration is - published on every multi-input task so a client
#: reads it up front instead of discovering it from a refusal. A multi-channel
#: network consumes ONE tensor, so its channels must already share a grid, and
#: producing that is a registration step belonging upstream where the caller can
#: see and check it. Slicer registers before it ever calls us; doing it silently
#: inside an inference call would be a geometry decision taken on someone else's
#: behalf, which is the shape of every geometry bug this project has paid for.
ASSUMED_PREREGISTERED = {
    "mode": "assumed-preregistered", "owner": "caller",
    "note": "channels are stacked in the model's declared order and must already "
            "be co-registered on a common grid; haversack does not register or "
            "resample them, and refuses inputs whose grids differ",
}


def _ordered_channel_def(channel_def) -> list:
    """A model's declared channel names, in channel order.

    Keys are strings in the JSON (``"0"``, ``"1"``, ...), so they are sorted
    numerically rather than lexically - a ten-channel model must not put
    ``"10"`` between ``"1"`` and ``"2"``. Returns ``[]`` when nothing usable is
    declared, which the caller reads as "this model did not name its inputs".
    """
    if not isinstance(channel_def, dict):
        return []
    try:
        items = sorted(channel_def.items(), key=lambda kv: int(kv[0]))
    except (TypeError, ValueError):
        return []
    return [str(v) for _, v in items]


class MonaiEcosystem(EngineEcosystem):
    """The MONAI model zoo: a CATALOG of bundles run by the ``monai`` engine.

    The first ecosystem in the "many tasks on a new engine" shape - MOOSE is many
    tasks on the *existing* nnU-Net engine, because its models are nnU-Net
    checkpoints, while a MONAI bundle brings its own network *and* its own
    transform chain. So this is an :class:`EngineEcosystem` (no nnU-Net TaskSpec)
    that nonetheless installs per task, which is exactly the pair of axes the
    base class keeps apart.

    **The bundle is the spec**: labels, modality and channel count are read from
    each installed bundle's own ``configs/metadata.json``, never from the
    manifest, which holds only download + listing facts (the rule that keeps this
    from repeating the stale total_mr class map). See medseg/docs/monai-bundles.md.
    """

    name = "monai"
    engine = "monai"
    description = "MONAI model zoo bundles (engine)"

    def __init__(self, manifest: Path | None = None):
        raw = json.loads(Path(manifest or MONAI_MANIFEST).read_text(encoding="utf-8"))
        self._bundles = raw.get("bundles", raw)

    def tasks(self) -> list:
        return sorted(self._bundles)

    def _entry(self, task: str) -> dict:
        try:
            return self._bundles[task]
        except KeyError:
            raise ModelNotFound(
                f"unknown monai bundle {task!r}; this build curates "
                f"{sorted(self._bundles)}") from None

    def _dir(self, task: str, root) -> Path:
        # version in the path: two versions of a bundle can coexist, and an
        # @version pin then resolves to its own directory rather than fighting.
        return Path(root) / "monai" / f"{task}_v{self._entry(task)['version']}"

    def materialized(self, task: str, root) -> bool:
        d = self._dir(task, root)
        return (d / "configs" / "metadata.json").is_file()

    def ensure(self, task: str, root, progress=None, version=None) -> None:
        """Install the bundle through MONAI's own downloader.

        Deliberately not a zip fetch of the manifest's ``url``: the zoo has moved
        hosting, and its newest entries point at a Hugging Face *repo page* rather
        than a downloadable archive (which is also why they publish no checksum).
        ``monai.bundle.download`` is the one thing that knows all of
        monaihosting / huggingface_hub / github / ngc, so the manifest supplies
        the name and version and MONAI resolves where that actually lives.
        """
        entry = self._entry(task)
        if version is not None and version != entry["version"]:
            raise ModelNotFound(
                f"{task}@{version}: this build curates {task} v{entry['version']}; "
                "regenerate the manifest to serve another version")
        if self.materialized(task, root):
            return
        from monai.bundle import download          # worker-side only; not on the api image

        dest = self._dir(task, root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"downloading MONAI bundle {task} v{entry['version']}")
        # MONAI unpacks to <bundle_dir>/<name>; we keep versioned directories so two
        # versions can coexist, so download into a temp parent and move into place.
        staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".bundle-"))
        try:
            download(name=task, version=entry["version"], bundle_dir=str(staging),
                     source=entry.get("source") or "monaihosting", progress=False)
            unpacked = staging / task
            if not (unpacked / "configs" / "metadata.json").is_file():
                raise ModelNotFound(
                    f"{task}: the downloaded bundle has no configs/metadata.json under "
                    f"{unpacked} - the layout is not what this ecosystem expects")
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            os.replace(unpacked, dest)             # atomic: never a half-installed bundle
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def bundle_metadata(self, task: str, root) -> dict:
        """The installed bundle's own metadata.json (the spec)."""
        return json.loads((self._dir(task, root) / "configs" / "metadata.json").read_text(encoding="utf-8"))

    def bundle_root(self, task: str, root) -> Path:
        """Where the installed bundle lives - what the engine runs."""
        return self._dir(task, root)

    def weights_identity(self, task: str, root) -> list:
        """Per bundle+version, not one constant for the whole engine: two bundles
        must not collide on one cached result. Read from the manifest already in
        memory, so ``/v1/tasks`` stays cheap."""
        entry = self._entry(task)
        out = {"id": task, "version": entry["version"]}
        if entry.get("checksum"):        # the zoo omits it on recent releases
            out["sha1"] = entry["checksum"]
        return [out]

    def describe_task(self, task: str, root) -> dict:
        """Modality, structures, input roles and restore behavior, all read from
        the installed bundle."""
        from .schemas import input_specs, single_input

        fmt = (self.bundle_metadata(task, root).get("network_data_format") or {})
        inp = (fmt.get("inputs") or {}).get("image") or {}
        channel_def = ((fmt.get("outputs") or {}).get("pred") or {}).get("channel_def") or {}
        names = [str(v) for k, v in sorted(channel_def.items(),
                                           key=lambda kv: int(kv[0]))
                 if str(v).lower() != "background"]
        # The same structural signal the result path uses (see
        # monai_bundle.resolve_label_names): a labelmap declares a `background`
        # entry because value 0 needs a name; a REGION head does not, because no
        # channel is background. Publishing a region head's channel names as
        # `structures` would have a client render "Tumor core" over a result that
        # reports label 1/2/4 - describe and the result must not disagree about
        # what those names mean.
        if any(str(v).lower() == "background" for v in channel_def.values()):
            out = {"structures": names}
        else:
            out = {"structures": [], "regions": names}
        modality = str(inp["modality"]) if inp.get("modality") else None
        if modality:
            out["modality"] = modality

        n_in = int(inp.get("num_channels") or 1)
        roles = _ordered_channel_def(inp.get("channel_def"))
        if n_in == 1:
            out["inputs"] = single_input(modality)
        elif len(roles) == n_in:
            out["inputs"] = input_specs(roles, modality=modality)
            out["channel_names"] = roles
        else:
            # The bundle claims N channels and names fewer - renalStructures_CECT
            # declares 3 and names one ("image"). There is no honest way to bind
            # the rest: position is exactly what cannot be trusted here, since
            # MONAI's BraTS bundle orders T1c first where nnU-Net's own BraTS
            # convention puts FLAIR there. So: refuse, and say why.
            out["inputs"] = None
            out["inputs_incomplete"] = {
                "channels": n_in, "named": roles,
                "reason": f"the bundle declares {n_in} input channels but names "
                          f"{len(roles)}; haversack will not bind inputs by position"}
            out["channel_names"] = roles or [f"channel_{i}" for i in range(n_in)]
        out["behavior"] = {"restore": self._restore_fact(task, root)}
        if out.get("regions"):
            out["behavior"]["labels"] = {
                "mode": "regions", "owner": "bundle",
                "note": "this bundle emits overlapping regions as output "
                        "channels, not mutually exclusive labels; haversack reports "
                        "the voxel values it wrote and does not name them"}
        if out.get("inputs") and len(out["inputs"]) > 1:
            out["behavior"]["alignment"] = dict(ASSUMED_PREREGISTERED)
        return out

    def _restore_fact(self, task: str, root) -> dict:
        """How this bundle brings its prediction back to the input grid.

        A *fact*, not a knob: the bundle's own postprocessing decides, and the
        two orders in the wild give materially different boundaries.
        ``spleen_ct_segmentation`` inverts its spacing transform BEFORE argmax,
        so probabilities are resampled and the boundary is graded;
        ``wholeBody_ct_segmentation`` argmaxes first and inverts a labelmap with
        ``nearest_interp``, which snaps every boundary to the model's grid. We do
        not override either - running the bundle's own config is the whole point
        of this engine - so the least we can do is let a client see which one it
        is getting before choosing a model.
        """
        try:
            from .engines.monai_bundle import inference_config
            cfg = inference_config(self.bundle_root(task, root))
            if cfg.suffix != ".json":
                # YAML would need a parser the lean API image does not carry
                raise ValueError(f"{cfg.suffix} config")
            post = json.loads(cfg.read_text(encoding="utf-8")).get("postprocessing") or {}
            transforms = post.get("transforms") if isinstance(post, dict) else None
            order = [str((t or {}).get("_target_", "")) for t in (transforms or [])
                     if isinstance(t, dict)]
            invert = next((i for i, t in enumerate(order) if t.endswith("Invertd")), None)
            argmax = next((i for i, t in enumerate(order) if t.endswith("AsDiscreted")), None)
        except Exception as e:                      # unreadable/absent/YAML: say so
            return {"mode": "unknown", "owner": "bundle", "note": f"not determined ({e})"}
        if invert is None:
            # nothing inverts, so the prediction stays on the model's grid and
            # haversack's own nearest-neighbor resample is what the caller gets
            return {"mode": "label-nearest", "owner": "haversack",
                    "note": "the bundle does not invert its spacing transform; "
                            "haversack resamples the labelmap to the input grid"}
        if argmax is not None and argmax < invert:
            return {"mode": "label-nearest", "owner": "bundle",
                    "note": "this bundle argmaxes before inverting its spacing "
                            "transform, so boundaries are snapped to the model's grid"}
        return {"mode": "graded", "owner": "bundle",
                "note": "this bundle inverts its spacing transform before argmax, "
                        "so class probabilities are resampled and argued after"}

    def info(self, task: str, root) -> dict:
        out = super().info(task, root)
        entry = self._entry(task)
        # listing facts, so an uninstalled task still says something useful
        out.setdefault("modality", entry.get("modality"))
        out["bundle_version"] = entry["version"]
        if entry.get("task"):
            out["summary"] = entry["task"]
        # the bundle's own credit, from its metadata.json via the manifest: a
        # bundle's authors are not the MONAI team's, and its references are
        # what it asks to be cited
        for key in ("description", "authors", "copyright", "references", "data_source"):
            if entry.get(key):
                out[key] = entry[key]
        if not out.get("materialized"):
            out["n_structures"] = max(int(entry.get("n_labels", 1)) - 1, 0)
            out.update(self._preinstall_inputs(entry, out.get("modality")))
        return out

    def _preinstall_inputs(self, entry: dict, modality) -> dict:
        """What can honestly be said about a bundle's inputs before it is
        installed.

        The manifest records the channel *count* (generator-derived from the
        bundle's own metadata), which is enough for the single-input case that
        covers most of the zoo. It does not always carry the channel *names*, and
        a multi-channel task without names cannot be described as one image
        without lying about it - so that case reports no inputs and says where
        the names come from.
        """
        from .schemas import input_specs, single_input

        n_in = int(entry.get("in_channels") or 1)
        roles = _ordered_channel_def(entry.get("channel_def"))
        if len(roles) == n_in:
            out = {"inputs": input_specs(roles, modality=modality)}
            if n_in > 1:
                # the caller has to know this BEFORE assembling a request, not
                # after being refused for it
                out["behavior"] = {"alignment": dict(ASSUMED_PREREGISTERED)}
            return out
        if n_in <= 1:
            return {"inputs": single_input(modality)}
        return {"inputs": None,
                "inputs_hint": f"this bundle takes {n_in} input channels; their "
                               "names are read from the bundle once installed"}


#: The ecosystems each engine contributes when its engine is enabled. The
#: nnU-Net catalogs are always present; engine catalogs appear only where their
#: engine does, so the catalog can never list a task no worker can run.
_ENGINE_ECOSYSTEMS = {"fastsurfer": FastSurferEcosystem,
                      "synthstrip": SynthStripEcosystem,
                      "voxtell": VoxTellEcosystem,
                      "monai": MonaiEcosystem}


def default_ecosystems() -> list:
    """The catalogs this deployment serves: the nnU-Net ones, plus one per
    enabled engine. Enablement is the registry's answer (read from the
    environment per call), so the catalog and the workers cannot disagree."""
    ecos = [TSEcosystem(), MooseEcosystem(), MRSegmentatorEcosystem(),
            DentalSegmentatorEcosystem(), TotalVibeEcosystem(), CADSEcosystem()]
    ecos += [cls() for engine, cls in _ENGINE_ECOSYSTEMS.items()
             if _registry.enabled(engine)]
    return ecos


class EcosystemCatalog:
    """A TaskCatalog-compatible federation over an ecosystem registry.

    Canonical names are ``eco:task``; short names resolve when unique. ``get()``
    materializes on demand (installing weights if needed - the same
    on-first-use behavior TS weights ids always had); ``info()`` never
    downloads. A TaskSpec or a model-folder path passes through ``get``
    untouched, as with TaskCatalog."""

    def __init__(self, ecosystems=None, *, root=None):
        self.registry = registry(ecosystems)
        self.root = root
        self._short: dict[str, list] = {}
        for ename, e in self.registry.items():
            for t in e.tasks():
                self._short.setdefault(t, []).append(ename)

    def names(self) -> list:
        return sorted(f"{ename}:{t}" for t, enames in self._short.items()
                      for ename in enames)

    def resolve(self, name: str) -> tuple:
        """``(ecosystem, short_task, canonical, version)`` for any name form.

        The grammar is ``[eco:]name[@version]`` - three layers plus the hash
        beneath: the short name resolves when exactly one ecosystem offers it,
        the canonical form is ecosystem-qualified, and ``@version`` pins a
        weights release at install time (the canonical name stays unversioned;
        actual versions live in the result key's weights component). Unknown
        names and ambiguous short names raise LookupError - the ambiguity
        error lists the qualified candidates to use instead."""
        name = str(name)
        version = None
        if "@" in name:
            name, _, version = name.rpartition("@")
            if not version or not name:
                raise LookupError(f"malformed task name {name!r}@{version!r}")
        if ":" in name:
            ename, _, short = name.partition(":")
            eco = self.registry.get(ename)
            if eco is None or ename not in self._short.get(short, ()):
                raise LookupError(f"unknown task {name!r}")
            return eco, short, f"{ename}:{short}", version
        enames = self._short.get(name)
        if not enames:
            raise LookupError(f"unknown task {name!r}; {len(self._short)} known, "
                              f"e.g. {self.names()[:6]}")
        if len(enames) > 1:
            raise LookupError(f"short name {name!r} is ambiguous; use one of "
                              + ", ".join(f"{e}:{name}" for e in sorted(enames)))
        return self.registry[enames[0]], name, f"{enames[0]}:{name}", version

    def __contains__(self, name) -> bool:
        try:
            self.resolve(name)
            return True
        except LookupError:
            return False

    def __len__(self) -> int:
        return sum(len(v) for v in self._short.values())

    def ecosystem_of(self, name: str):
        try:
            return self.resolve(name)[0]
        except LookupError:
            return None

    def engine_of(self, name: str) -> str | None:
        """Which engine runs ``name`` - the catalog-side answer to the question
        the Modal dispatcher answers from the task's grammar. ``None`` if the
        name does not resolve."""
        eco = self.ecosystem_of(name)
        return None if eco is None else engine_of(eco)

    def installed(self, name) -> bool:
        """Whether ``get(name)`` finds the task's weights in place, so it installs nothing."""
        eco, short, canonical, version = self.resolve(name)
        return eco.materialized(short, self.root)

    def get(self, name, progress=None) -> TaskSpec:
        """The task's spec, installing its weights first if they are not in place.
        ``progress`` reaches the install (a job's Reporter, a message callback, or None)."""
        if isinstance(name, TaskSpec):
            return name
        eco, short, canonical, version = self.resolve(name)
        if version is not None or not eco.materialized(short, self.root):
            eco.ensure(short, self.root, progress=progress, version=version)
        spec = eco.spec(short, self.root)
        if spec.name != canonical:
            import dataclasses
            spec = dataclasses.replace(spec, name=canonical)
        return spec

    __getitem__ = get

    def info(self, name: str) -> dict:
        eco, short, canonical, version = self.resolve(name)
        out = eco.info(short, self.root)
        out["name"] = canonical
        if version is not None:
            out["version_requested"] = version
        # Credit travels with every task record: the task's own facts from its
        # manifest, its ecosystem's license and papers, and its engine's - so a
        # client that found the task can find whom to thank without knowing
        # which catalog it came from.
        from . import attribution
        out["attribution"] = attribution.for_task(canonical, out)
        return out

    def prepare(self, name: str, progress=None) -> dict:
        """Install the task's weights now and return its full info."""
        eco, short, canonical, version = self.resolve(name)
        eco.ensure(short, self.root, progress=progress, version=version)
        out = eco.info(short, self.root)
        out["name"] = canonical
        return out
