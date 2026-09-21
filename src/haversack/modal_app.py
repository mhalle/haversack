"""The haversack REST protocol deployed on Modal - same contract, platform underneath.

    HAVERSACK_PROXY_AUTH=0 HAVERSACK_GPU=A10 haversack modal deploy     # or: modal deploy <this file>

One `modal deploy` of this file gives a scale-to-zero deployment of the exact
contract `haversack serve` speaks locally, so the client and the Slicer module cannot
tell the difference:

- an ASGI function on a cheap CPU container runs :func:`haversack.serve.create_app` over
  a :class:`ModalExecutor` - stateless by construction, since `.spawn()` detaches the
  GPU work and all state lives in a `modal.Dict` (job metadata + progress snapshots)
  and a jobs Volume (uploads, label outputs);
- a GPU `Worker` class holds a warm ``Segmenter(cache_models=5)`` across jobs
  (``scaledown_window`` keeps it alive between a session's runs); weights
  self-provision into the shared ``haversack-weights`` Volume on first use;
- where the LocalExecutor supplies a bounded FIFO, here **Modal is the queue**:
  spawn enqueues and the autoscaler drains up to ``HAVERSACK_MAX_CONTAINERS``
  (default 1: parallel requests queue and run serially on one warm worker - the
  economical posture, and every job after the first is warm; raise it at deploy
  time for cohort fan-out, up to the plan's GPU cap). ``accepting`` is always
  true - the backlog has no bound to enforce;
- progress writes are rate-limited (~4/s) because each Dict write is an RPC; the
  server's SSE endpoint reads them through its poll branch (``supports_push=False``);
- cancel is `FunctionCall.cancel()` - the container stops, billing stops;
- auth is Modal proxy auth (on by default; ``HAVERSACK_PROXY_AUTH=0`` to disable for a
  smoke test) - per-person tokens minted and revoked in the Modal dashboard, zero
  auth code here. The client sends them as Modal-Key / Modal-Secret headers.

Deploy-time configuration is by environment variable because Modal resolves
decorators at import: HAVERSACK_GPU (default L40S - wins `total` outright; A10 is the
economical fast-mode choice), HAVERSACK_APP_NAME, HAVERSACK_SCALEDOWN (seconds, default 120 -
conservative: a forgotten/left-up deploy idles at most ~2 min of GPU (~$0.07 on L40S)
before scaling down; raise it to keep a busy server warmer), HAVERSACK_PROXY_AUTH,
HAVERSACK_SNAPSHOT (memory snapshots, default ON - measured 2026-08-24: cold spawn->start
10-14 s -> 6.4-6.7 s, one 35 s snapshot-creation run per deploy), HAVERSACK_CACHE_VOLUME (the
result cache's volume, default ``<app name>-cache`` - see ``CACHE_VOLUME``).

The image mounts the *running* haversack package (works from an editable checkout or an
installed wheel alike). TODO(release): switch to ``uv_pip_install("haversack==<ver>")``
once published, so a deploy is pinned to a version instead of a working tree.
"""
import functools
import importlib
import os
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("HAVERSACK_APP_NAME", "haversack-serve")
#: The Modal volume that holds the RESULT CACHE, by default this app's own (2026-09-20).
#: Every other per-app store is named after the app, so a deployment under a new name began
#: with an empty result cache - though a result key holds no app name at all (identity x task
#: x options x weights versions x epoch, see ``serve.result_key``), which makes a cache
#: portable: what `haversack-radar-val` computed answers the same requests under any name.
#: Only the cache is nameable. Scratch, the inputs store and the jobs Dict hold one
#: deployment's job ids, uploads and flights, and stay ``{APP_NAME}-...``.
#:
#: Two cases, and they differ. Reusing a cache whose first deployment is GONE is completely
#: safe: it is the same volume read by the same code. Two LIVE deployments on one cache behave
#: like more containers of one app - publication by generation is what already lets several
#: workers write one volume - with one difference: single flight lives in the per-app jobs
#: Dict (the ``inflight:`` markers), so each app can compute the same key at once. That is
#: duplicate work, never corruption: both publish a generation, one pointer wins, and the
#: other generation is pruned once no reader holds it.
#:
#: One thing to set in BOTH cases: every publication evicts down to the publishing app's own
#: ``HAVERSACK_RESULTS_KEEP`` (default 500), so a deployment that adopts a 2,000-entry cache
#: with the default would evict 1,500 results at its first job. Deploy it with a bound at
#: least the size of the cache it adopts; with two live apps the smaller bound is the real one.
#: An unset or empty value means the default; nothing is renamed or migrated.
CACHE_VOLUME = os.environ.get("HAVERSACK_CACHE_VOLUME") or f"{APP_NAME}-cache"
GPU = os.environ.get("HAVERSACK_GPU", "L40S")
PROXY_AUTH = os.environ.get("HAVERSACK_PROXY_AUTH", "1") not in ("0", "false", "no")
SCALEDOWN = int(os.environ.get("HAVERSACK_SCALEDOWN", "120"))
GPU_SNAPSHOT = os.environ.get("HAVERSACK_GPU_SNAPSHOT", "0") not in ("0", "false", "no", "")
SNAPSHOT = (os.environ.get("HAVERSACK_SNAPSHOT", "1") not in ("0", "false", "no")) or GPU_SNAPSHOT
WARM_TASK = os.environ.get("HAVERSACK_WARM_TASK", "ts.v2:total_fast")   # qualified: bare names are refused
MAX_CONTAINERS = int(os.environ.get("HAVERSACK_MAX_CONTAINERS", "1"))
SHM_CACHE_GB = float(os.environ.get("HAVERSACK_SHM_CACHE_GB", "8"))
JOBS_TTL_H = float(os.environ.get("HAVERSACK_JOBS_TTL_H", "72"))
#: The deliverables this deployment renders: the default of a request that names none and
#: the ceiling of one that does (``jobpolicy.wanted_deliverables``), read in the api
#: container (the submit door, ``links``) and in every worker (what is rendered).
ARTIFACTS = set(filter(None, os.environ.get("HAVERSACK_ARTIFACTS",
                                            "preview,statistics").split(",")))
#: Why a cache hit here could not deliver what its list named (``_unrendered_on_hit``).
DELIVERABLE_NEEDS_A_COMPUTE = (
    "not rendered for this stored result, and this deployment renders deliverables only in "
    "the worker that computes a result; Cache-Control: no-cache recomputes it with its "
    "deliverables")
DELIVERABLE_NOT_VISIBLE = (
    "this server cannot see the result store's latest state yet (a volume reload was "
    "refused), so it cannot tell whether this was rendered; ask again shortly")
RESULTS_KEEP = int(os.environ.get("HAVERSACK_RESULTS_KEEP", "500"))
WEIGHTS_ROOT, SCRATCH_ROOT, CACHE_ROOT = "/weights", "/scratch", "/cache"
INPUTS_ROOT = "/inputs"
#: container-local copies of a running job's own files - its uploads and its labels -
#: so that nothing reads them from the scratch volume outside ``_vol_lock`` (see
#: ``_stage_uploads``)
JOB_LOCAL_ROOT = "/tmp/haversack-jobs"
# Inputs get a bigger floor than fetched series: a re-fetchable IDC series
# costs a download when evicted, an uploaded volume is simply gone.
INPUTS_GB = float(os.environ.get("HAVERSACK_INPUTS_GB", "50"))
PUBLIC = os.environ.get("HAVERSACK_PUBLIC", "0") not in ("0", "false", "no", "")


def _engine_registry():
    """The engine registry, imported the same way the rest of haversack is (mounted
    package, with the sys.path shim applied first)."""
    try:
        from haversack.engines import registry
    except ImportError:
        sys.path.insert(0, "/root/pkg")
        from haversack.engines import registry
    return registry


_engines = _engine_registry()
# Which engines this deployment runs. Snapshotted at import because Modal
# resolves the @app.cls decorators now; the registry reads the same env vars.


def _pkg_dir() -> Path:
    try:
        import haversack
    except ImportError:                      # inside the container: the mounted copy
        sys.path.insert(0, "/root/pkg")
        import haversack
    return Path(haversack.__file__).parent


# Knobs read at RUNTIME inside the container must be forwarded into the image
# env at deploy time - a deploy-shell variable does not otherwise exist in the
# container (found the hard way: a TTL override that never took effect).
_RUNTIME_KNOBS = ("HAVERSACK_SHM_CACHE_GB", "HAVERSACK_JOBS_TTL_H", "HAVERSACK_RESULTS_KEEP",
                  "HAVERSACK_WARM_TASK", "HAVERSACK_ARTIFACTS",
                  # HAVERSACK_PUBLIC gates a module-level `if PUBLIC:` around the
                  # twin function. The deploy-time import registers it, but
                  # the CONTAINER re-imports this module - without the knob
                  # forwarded, its PUBLIC is False, the attribute never
                  # exists, and every request 303s while the runner crash-
                  # loops on AttributeError (hit live 2026-08-25).
                  "HAVERSACK_PUBLIC",
                  # Read by the Worker at construction (allow_transpose). Without
                  # it forwarded the container's copy is unset, so the three tasks
                  # whose plans permute the axes are listed, described, accepted,
                  # and then refused inside the GPU worker - the exact
                  # unreachability the flag was added to end.
                  "HAVERSACK_ALLOW_TRANSPOSE",
                  # Which cloud the idc: source fetches from first; a deployment
                  # in Google Cloud sets gcp and reads IDC's mirror without egress.
                  "HAVERSACK_IDC_CLOUD",
                  # EVERY other knob this module reads at import, because the container
                  # re-imports it and a missing one silently takes its default there.
                  # The app name did (2026-09-12): a deploy with --app-name ran its
                  # containers as "haversack-serve", so a worker committed a volume that
                  # was not mounted and every job failed - and the job records went to
                  # the DEFAULT app's `haversack-serve-jobs` Dict, which is looked up by
                  # name and so did not fail at all. INPUTS_GB and GPU_SNAPSHOT are read
                  # at runtime too; the decorator-only ones are forwarded so a container
                  # never disagrees with its deploy. test_every_import_time_knob_reaches_
                  # the_container keeps this list and the reads below in step.
                  "HAVERSACK_APP_NAME", "HAVERSACK_GPU", "HAVERSACK_PROXY_AUTH",
                  "HAVERSACK_SCALEDOWN", "HAVERSACK_GPU_SNAPSHOT", "HAVERSACK_SNAPSHOT",
                  "HAVERSACK_MAX_CONTAINERS", "HAVERSACK_INPUTS_GB", "HAVERSACK_API_MIRROR_GB",
                  # Which volume is the result cache (2026-09-20). Unforwarded it would be
                  # the app-name defect again: every container re-derives `<app>-cache`, so
                  # the deploy mounts the named cache at /cache while the containers commit
                  # and reload the app's own volume, which is mounted nowhere.
                  "HAVERSACK_CACHE_VOLUME",
                  *_engines.engine_env_vars())

# Base image (the ASGI api container + the nnU-Net GPU Worker). uv-NATIVE: the nnU-Net
# worker's deps come from pyproject extras - `torch` (torch/nnunetv2/scipy/scikit-image),
# `serve` (fastapi/uvicorn/python-multipart/matplotlib), `cuda` (triton,
# the CUDA restore backend). The inference stack itself is core (torch, nnunetv2, scipy,
# scikit-image, all from PyPI). apt git: uv sync resolves the whole project's lock, which
# touches the engine git sources.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(extras=["torch", "serve", "cuda"], frozen=False)
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack")
)













# Front-end image for the ASGI api/public functions. The api never RUNS inference - only
# catalog/describe + orchestration + cache/publish - and `import haversack` plus the whole
# describe path stay torch-free at runtime, which is what `tests/test_layering.py` enforces.
#
# It does not follow that the image is small, and the comment here claimed it was long
# after it stopped being true. This said "carries NO torch / nnunetv2 / triton / CUDA" and
# justified the image by cold-start time. Then the inference stack moved from the `torch`
# extra into core `dependencies` (2026-09-03, so that `uvx ... haversack segment` works
# from a bare install), and `uv_sync` installs core - so torch, nnunetv2, scipy and
# scikit-image have been in here ever since, and the `torch` extra is now an empty alias.
# What this image still avoids is the CUDA tier (triton, the CUDA restore backend) and the
# engine extras. Whether it should get its leanness back - a `--no-deps` install of the
# five names the README's "Lean install" lists - is an open question that wants a
# cold-start measurement first, not a guess (2026-09-08).
api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                       # uv sync resolves the whole lock (engine git sources)
    .uv_sync(extras=["serve"], frozen=False)
    .env({k: os.environ[k] for k in _RUNTIME_KNOBS if k in os.environ})
    .add_local_dir(_pkg_dir(), remote_path="/root/pkg/haversack")
)

app = modal.App(APP_NAME, image=image)
weights_vol = modal.Volume.from_name("haversack-weights", create_if_missing=True)
scratch_vol = modal.Volume.from_name(f"{APP_NAME}-scratch", create_if_missing=True)
# Uploaded inputs, addressed by their own bytes. A volume of its own rather than
# a prefix under the scratch one because the lifetimes are opposites: job
# directories churn and are deleted, while a preloaded input exists precisely to
# survive the scale-to-zero that makes preloading worth doing. Separate volumes
# are also what makes "inputs outlive results" expressible - an evicted result
# can be recomputed from its recipe, an evicted upload is simply gone.
inputs_vol = modal.Volume.from_name(f"{APP_NAME}-inputs", create_if_missing=True)
jobs_dict = modal.Dict.from_name(f"{APP_NAME}-jobs", create_if_missing=True)
# The one store a deployment may name (HAVERSACK_CACHE_VOLUME, see CACHE_VOLUME above): its
# keys carry no app name. The three above stay per app, and a test reads these five lines
# to keep it so - scratch and the Dict hold THIS deployment's job ids and flights.
cache_vol = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)

# -- the api container's view of the cache volume ---------------------------------
#
# Two ways a result the workers had committed read as absent here, both found on
# 2026-09-19 (haversack-radar-val: 191 of 300 finished ts.v2:total jobs answered 410
# "purged", and 200 for the same URLs minutes later; reproduced on haversack-visible-smoke):
#
# 1. A reload IN ANOTHER THREAD hides the whole volume from this one. While a reload runs,
#    every path on the volume is ENOENT to the container's other threads - measured on
#    Modal: a thread listing a 260-entry directory while another thread reloaded got
#    ENOENT 734,714 times in 735,326, and never with no reload running; the api's own
#    /cache listing emptied and refilled every 0.5-3 s under load (6,201 disappearances in
#    9 minutes). The api reloaded on every request's thread, so a lookup racing any other
#    request's reload missed - and was believed. Hence ``_cache_view``: lookups hold it
#    shared, a reload holds it exclusive, and nothing is read from the volume outside it.
# 2. Modal REFUSES a reload while a file on the volume is open in the container, and every
#    streamed result held its file open for the transfer - so under concurrent downloads
#    reloads failed, silently, and the view went stale. Hence the local copies: the api
#    serves a container-local copy of the generation (``_mirror``), taken under the shared
#    lock, and holds no volume file open past the lookup.
#
# And a miss is an answer only when read from a view newer than the question
# (``_confirm_cache_absent``); when no reload takes, the answer is 503, not 410.


class _ViewLock:
    """Readers-writer lock over the cache volume's view within this container. Writer
    preferring: under a hundred concurrent requests a reload would otherwise never find
    a moment with no reader. Reads are short - a pointer, a lease, a copy of a few MB -
    so a reload waits milliseconds; ``acquire_exclusive`` is bounded all the same."""

    def __init__(self):
        self._cv = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting = 0

    def shared(self):
        import contextlib

        @contextlib.contextmanager
        def held():
            with self._cv:
                while self._writer or self._waiting:
                    self._cv.wait()
                self._readers += 1
            try:
                yield
            finally:
                with self._cv:
                    self._readers -= 1
                    if not self._readers:
                        self._cv.notify_all()
        return held()

    def acquire_exclusive(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            self._waiting += 1
            try:
                while self._writer or self._readers:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(left)
                self._writer = True
                return True
            finally:
                self._waiting -= 1
                self._cv.notify_all()          # readers held back by a waiter that left

    def release_exclusive(self) -> None:
        with self._cv:
            self._writer = False
            self._cv.notify_all()

    def exclusive(self, timeout: float = 60.0):
        import contextlib

        @contextlib.contextmanager
        def held():
            if not self.acquire_exclusive(timeout):
                raise TimeoutError("the cache volume stayed busy")
            try:
                yield
            finally:
                self.release_exclusive()
        return held()


_cache_view = _ViewLock()
#: how long a reload waits for in-flight lookups to finish before it counts as refused
CACHE_RELOAD_WAIT_S = 2.0
#: monotonic START of the newest reload that succeeded here: the view holds everything
#: committed before it
_cache_view_as_of = float("-inf")
_cache_view_stamp = threading.Lock()
#: one reload at a time; a thread that queued behind another's takes its outcome when that
#: reload STARTED after this thread asked - the question it would have asked itself
_cache_reload_mutex = threading.Lock()
#: monotonic START of the newest reload that FAILED here
_cache_view_failed_at = float("-inf")
#: a plain lookup reuses a view reloaded this recently: every request reloading was a queue
#: of writers that starved the readers (adversarial review, 2026-09-19: a lookup waited 6 s
#: behind four threads). A MISS is still confirmed against a view newer than the request.
CACHE_FRESH_S = 1.0
#: serializes the confirming reloads, so a thread queued behind another's success
#: re-checks instead of reloading again
_cache_confirm_lock = threading.Lock()
#: waits between the confirming reload attempts - bounded: a request is not held longer
#: than about this plus the reloads, and then answers 503 with Retry-After
CACHE_CONFIRM_DELAYS_S = (0.0, 0.25, 0.5, 1.0)
_reload_failure_logged = {}

#: container-local copies of result generations, served in place of the volume's files
MIRROR_ROOT = "/tmp/haversack-results"
MIRROR_MAX_BYTES = int(float(os.environ.get("HAVERSACK_API_MIRROR_GB", "2")) * 2**30)
#: a copy younger than this is never removed: a response may still be about to open it
MIRROR_MIN_AGE_S = 600.0
_mirror_placed = [0]
#: held while a copy is made or refreshed, and by the trim per copy it removes: a copy
#: handed out cannot be taken between the trim's age check and its delete
_mirror_lock = threading.Lock()


def _reload_logged(vol, name: str) -> bool:
    """``vol.reload()``, True when it took. A refusal is logged at most once a minute
    per volume - it was swallowed without a word before, which is how the 410s above
    went unexplained."""
    try:
        vol.reload()
        return True
    except Exception as e:                  # noqa: BLE001 - the view is simply not refreshed
        _log_refusal(name, f"{type(e).__name__}: {e}")
        return False


def _log_refusal(name: str, why: str) -> None:
    now = time.monotonic()
    if now - _reload_failure_logged.get(name, float("-inf")) >= 60.0:
        _reload_failure_logged[name] = now
        print(f"[volume] {name} reload refused: {why}", flush=True)


def _reload_cache_view(max_age: float = 0.0) -> bool:
    """Make the view at least as new as this call, less ``max_age`` seconds: True when
    it is. Reloads once no lookup is mid-read (see ``_cache_view``), one reload at a time,
    and a caller that waited behind another's reload takes that one's outcome when it
    started after the caller asked - so N concurrent askers cost one reload, not a queue
    of N writers. Never called with ``_cache_view`` held by this thread."""
    global _cache_view_as_of, _cache_view_failed_at
    asked = time.monotonic()
    with _cache_reload_mutex:
        if _cache_view_as_of >= asked - max_age:
            return True
        if _cache_view_failed_at >= asked:
            return False                       # one that started after we asked failed
        t = time.monotonic()
        if not _cache_view.acquire_exclusive(CACHE_RELOAD_WAIT_S):
            _log_refusal("cache", f"lookups kept it busy for {CACHE_RELOAD_WAIT_S} s")
            ok = False
        else:
            try:
                ok = _reload_logged(cache_vol, "cache")
            finally:
                _cache_view.release_exclusive()
        with _cache_view_stamp:
            if ok:
                _cache_view_as_of = max(_cache_view_as_of, t)
            else:
                _cache_view_failed_at = max(_cache_view_failed_at, t)
        return ok


def _mirror(src: Path, dst: Path) -> Path:
    """``dst``, a container-local copy of the directory ``src`` (a result generation):
    every file ``src`` holds now that ``dst`` lacks is copied in, atomically per file.
    A generation's files never change once placed - artifacts only arrive - so a copy
    is refreshed by adding, never by rewriting. Called under ``_cache_view`` shared."""
    import shutil
    import uuid
    with _mirror_lock:
        dst.mkdir(parents=True, exist_ok=True)
        for f in os.scandir(src):
            if f.name.startswith(".") or f.name.endswith(".tmp") or not f.is_file():
                continue
            out = dst / f.name
            if not out.exists():
                tmp = dst / f".{f.name}.{uuid.uuid4().hex[:8]}"
                shutil.copyfile(f.path, tmp)
                os.replace(tmp, out)
                _mirror_placed[0] += 1
        os.utime(dst)                          # in use: the trim's age is this
        due = _mirror_placed[0] >= 64
        if due:
            _mirror_placed[0] = 0
    if due:
        # outside the volume lock - it reads no volume - and outside _mirror_lock but
        # per copy: a trim of GBs must hold neither reloads nor lookups
        threading.Thread(target=_trim_mirror, name="haversack-mirror-trim",
                         daemon=True).start()
    return dst


def _copy_local(src: Path, out: Path) -> Path:
    """``src`` copied to ``out`` atomically (temp + rename, never rewritten in place: a
    response may be streaming the previous copy - adversarial review, 2026-09-19, a
    200 MB stream came out 2 MB), under ``_mirror_lock``, its directory touched."""
    import shutil
    import uuid
    with _mirror_lock:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.parent / f".{out.name}.{uuid.uuid4().hex[:8]}"
        shutil.copyfile(src, tmp)
        os.replace(tmp, out)
        os.utime(out.parent)
    return out


def _trim_mirror() -> None:
    """Keep the copies under ``MIRROR_MAX_BYTES``, oldest-used first, never one used in
    the last ``MIRROR_MIN_AGE_S`` (a response may be about to open it; one already open
    keeps its bytes through an unlink)."""
    import shutil
    root = Path(MIRROR_ROOT)
    gens = []
    for d in root.glob("*/*"):
        try:
            gens.append((d.stat().st_mtime, sum(f.stat().st_size for f in d.iterdir()), d))
        except OSError:
            continue
    total = sum(g[1] for g in gens)
    for _, size, d in sorted(gens, key=lambda g: g[0]):
        if total <= MIRROR_MAX_BYTES:
            break
        with _mirror_lock:                     # asked again, and removed, under the lock
            try:
                if d.stat().st_mtime > time.time() - MIRROR_MIN_AGE_S:
                    continue                   # handed out since the listing
            except OSError:
                continue
            gone = d.parent / f".trim-{d.name}"
            try:
                os.replace(d, gone)            # out of reach at once; deleted after
            except OSError:
                continue
        shutil.rmtree(gone, ignore_errors=True)
        total -= size


def _read_cache(key: str):
    """The entry under ``key`` from the view as it stands - ``(local labels path,
    result)`` or None - read under ``_cache_view`` shared, and handed out as a local
    copy: the volume's own files are never open outside the lock."""
    from haversack.serve import RESULT_NAME, ResultCache
    with _cache_view.shared():
        hit = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).get(key)   # leased, as before
        if hit is None:
            return None
        g = Path(hit[0]).parent
        return _mirror(g, Path(MIRROR_ROOT) / key / g.name) / RESULT_NAME, hit[1]


def _cache_view_since(since: float) -> None:
    """Return once this container's view of the cache volume is newer than ``since`` (a
    monotonic time). Reloads, a bounded number of times, only when no reload since
    ``since`` has taken; raises ``ResultsNotVisible`` when none does, so the caller answers
    "not visible here yet" rather than "gone" or "never computed". The one loop behind
    every absence this container vouches for: a lookup's miss (``_confirm_cache_absent``)
    and a listing, whose every row NOT in it is a miss (``_list_cache``)."""
    from haversack.serve import ResultsNotVisible
    for delay in CACHE_CONFIRM_DELAYS_S:
        if delay:
            time.sleep(delay)
        with _cache_confirm_lock:
            if _cache_view_as_of >= since or _reload_cache_view():
                return
    raise ResultsNotVisible("this server cannot see the result cache's latest state yet "
                            "(a volume reload was refused); retry shortly")


def _confirm_cache_absent(key: str, since: float):
    """The entry under ``key`` as seen from a view newer than ``since`` (a monotonic
    time), or None when that view has no such entry - a VERIFIED miss. Raises
    ``ResultsNotVisible`` when no such view can be had (see ``_cache_view_since``)."""
    _cache_view_since(since)
    return _read_cache(key)


#: What this container's listings remember of each publication's meta.json, for as long
#: as the container lives - see ``serve.ListingMemo`` (made on first use: serve is a
#: call-time import here).
_listing_memo = []


def _list_cache(*, keys=None, limit=None, after=None, accept=None, match=None) -> tuple:
    """``ResultCache.list`` over the cache VOLUME, from the api container or the twin.

    The 2026-09-19 rules, all three. (1) The listing is read from a view newer than the
    request, or not at all: a result missing from it reads as "not computed", and a client
    that lists to decide what to compute would compute it again - the listing's form of
    the false 410. A refused reload is a 503, never a shorter list. (2) Nothing is read
    while another thread may be reloading: ``ResultCache.list`` reads on a pool of threads,
    which are "other threads" to a reload exactly as another request's is, so every batch
    of its reads runs inside ``_cache_view`` held shared by the thread that started it, and
    that hold ends only when the batch has. Per batch, not per listing: a cold scan of
    2,000 entries is seconds, a reload waits ``CACHE_RELOAD_WAIT_S`` for readers and every
    lookup queues behind a waiting reload (writer preference), so one long hold would stall
    the container's reads; ``serve.LIST_CHUNK`` bounds a hold to well under that wait. (3)
    It hands out no path and takes no lease - rows are copied out as plain data - so no
    volume file is open, or relied on, once it returns. ``ResultCache()`` mkdirs its root,
    a touch of the volume, so it too is built under the lock; the hold is never nested
    (a second shared acquire behind a waiting reload would wait on itself)."""
    from haversack.serve import LIST_WORKERS, ListingMemo, ResultCache
    _cache_view_since(time.monotonic())
    if not _listing_memo:
        _listing_memo.append(ListingMemo())
    with _cache_view.shared():
        cache = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP)
    return cache.list(keys=keys, limit=limit, after=after, accept=accept, match=match,
                      memo=_listing_memo[0], hold=_cache_view.shared, workers=LIST_WORKERS)


# -- `result:` references: the two readers a ResultSource resolves through here --------
#
# A reference is resolved twice - in the api container at submit, to key the job on the
# referenced output's digest, and again in the worker that fetches it (a lease taken here
# does not reach a worker's pruning, so the worker asks again and compares) - and both
# read the CACHE volume, so both follow the rules of 2026-09-19 above and below: nothing
# is read from the volume while another thread of the container may be reloading it, what
# is handed out is a local copy, and a miss is believed only from a view newer than the
# question. ``fresh`` is how the source asks for that view; it asks only after a stale one
# answered "missing" or "other bytes", the two answers a stale view can get wrong.


def _api_result_entry(key: str, *, fresh: bool = False):
    """``ResultSource``'s reader in the api container: ``cache_get``'s own lookup (a view
    up to ``CACHE_FRESH_S`` old, read under ``_cache_view`` shared, handed out as the
    container-local mirror of the generation - meta.json included), and for ``fresh`` the
    confirming reload, which raises ``ResultsNotVisible`` (a 503 at submit) when no reload
    takes rather than let a stale miss refuse a job whose mask is there."""
    import contextlib

    @contextlib.contextmanager
    def held():
        if fresh:
            yield _confirm_cache_absent(key, time.monotonic())
        else:
            _reload_cache_view(max_age=CACHE_FRESH_S)
            yield _read_cache(key)
    return held()


def _worker_result_entry(vol_lock):
    """``ResultSource``'s reader in a worker, over that worker's ``_vol_lock``.

    The lock is held from the reload through the lookup AND the caller's block - the
    source copies the file inside it - because a worker's other threads commit and reload
    this volume (the previous job's artifact overlap, the sweep), and either hides every
    path on it for its duration. ``ResultCache()`` is built under it too: it mkdirs its
    root. Without ``fresh`` there is no reload: the view a worker already has usually
    holds the entry, and the pinned digest says whether it is the right one. With it, a
    refused reload is retried over ``CACHE_CONFIRM_DELAYS_S``; what is visible after the
    last one may still do (the digest decides), and an entry that is simply not in view
    is ``ResultsNotVisible`` - "cannot see it yet" - never "gone".
    """
    import contextlib

    def entry(key: str, *, fresh: bool = False):
        @contextlib.contextmanager
        def held():
            from haversack.serve import ResultCache, ResultsNotVisible
            attempts = CACHE_CONFIRM_DELAYS_S if fresh else (0.0,)
            for i, delay in enumerate(attempts):
                if delay:
                    time.sleep(delay)              # never with the lock held
                last = i == len(attempts) - 1
                with vol_lock:
                    took = _reload_logged(cache_vol, "cache") if fresh else True
                    hit = (ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).get(key)
                           if took or last else None)
                    if took or hit is not None:
                        yield hit
                        return
            raise ResultsNotVisible(
                f"this worker cannot see result {key[:12]}... in the result cache yet (a "
                "volume reload was refused); submit again shortly")
        return held()
    return entry


def _worker_sources(vol_lock) -> dict:
    """A worker's source registry: every hosted source, and ``result:`` over this worker's
    view of the cache volume. One function, so the tests drive the wiring ``setup`` runs."""
    from haversack.sources import ResultSource, default_sources, registry
    return registry(default_sources() + [ResultSource(_worker_result_entry(vol_lock))])


def _worker_series_cache(sources: dict, root, budget_bytes: int):
    """A worker's staging cache over ``sources``: every fetch through the one door that
    records what the bytes are and where they came from."""
    from haversack.serve import SeriesCache
    from haversack.sources import fetch_recording_origin

    def fetch_source(key, entry, credentials=None):
        prefix, ident = key.split(":", 1)
        return fetch_recording_origin(sources[prefix], ident, entry, credentials)

    return SeriesCache(Path(root), fetch_source, budget_bytes=budget_bytes)


def _check_volumes_attached() -> None:
    """Fail fast, with the fix, if a mounted volume is not actually attached.

    A memory snapshot (HAVERSACK_SNAPSHOT, on by default) captures the container with its
    volume handles; if one of the deployment's volumes (`{APP_NAME}-scratch`, `-inputs`, and
    the cache, `CACHE_VOLUME`) is deleted and recreated afterwards, a restored container
    carries the dead handle and the FIRST WRITE fails deep in a job with a cryptic "volume
    vo-... not attached" (2026-09-03). A tiny write here surfaces it at startup instead, and
    names the remedy. Runs post-restore (in `setup`, snap-less), never during snapshotting -
    a volume touch there is unsafe.
    """
    import os
    for root in (SCRATCH_ROOT, CACHE_ROOT, INPUTS_ROOT, WEIGHTS_ROOT):
        probe = Path(root) / f".attach-probe-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as e:
            raise RuntimeError(
                f"volume mounted at {root} is not writable ({e}). This usually means the "
                f"volume was deleted while a memory snapshot still referenced it. Redeploy "
                f"once with HAVERSACK_SNAPSHOT=0 to rebuild a clean snapshot, or delete and "
                f"redeploy the whole app (`modal app stop {APP_NAME}` then deploy)." ) from e


# -- marker operations -------------------------------------------------------
# The ownership rules for the two marker namespaces, at module level so unit
# tests can reach them (the closures they used to live in survived every
# mutation). The Dict has no CAS: each guard is get-then-op, which closes the
# always-loses cases; the residual one-RPC window is irreducible here.

def _install_inflight(key: str, jid: str) -> None:
    """Guarded install: never stomp a newer flight's marker. (A fresh submit
    installs directly - a submit is always the genuinely newest flight.)"""
    if jobs_dict.get(f"inflight:{key}") in (None, jid):
        jobs_dict[f"inflight:{key}"] = jid


def _release_inflight(key: str, jid: str) -> None:
    """Compare-and-delete: under duplicate flights the marker names the
    LATEST job, and this one may not be it - deleting unconditionally made
    probes 404 while the survivor still ran and left DELETE unable to
    cancel it."""
    if jobs_dict.get(f"inflight:{key}") == jid:
        try:
            del jobs_dict[f"inflight:{key}"]
        except Exception:
            pass


def _set_pending_marker(key: str, jid: str, names=None) -> None:
    """Refuse-if-present: a duplicate flight must not ACQUIRE ownership by
    stomping - it would then legally clear the marker while the sibling's
    overlap still renders, and probes would read a definitive 404 for
    artifacts that land seconds later.

    ``names`` is what this render will place - the job's own deliverables (2026-09-20) -
    so a GET of one it declined reads a definitive absence at once, and a cache hit that
    wants it knows this render will not bring it. A marker with no ``names`` is a
    previous deploy's, which rendered its whole set."""
    if jobs_dict.get(f"artifacts:{key}") is None:
        jobs_dict[f"artifacts:{key}"] = {"state": "pending", "t": time.time(), "job": jid,
                                         **({} if names is None else {"names": list(names)})}


def _clear_pending_marker(key: str, jid: str) -> None:
    """Owner-only clear. A legacy marker without a job field (pre-ownership
    deploys) is treated as unowned and clearable by anyone - the sweep's
    rule."""
    m = jobs_dict.get(f"artifacts:{key}")
    if isinstance(m, dict) and m.get("job") not in (None, jid):
        return
    try:
        del jobs_dict[f"artifacts:{key}"]
    except Exception:
        pass


def _clear_own_artifacts_marker(jid: str, meta: dict) -> None:
    """Failure-path cleanup: drop this job's artifacts-pending marker (the
    overlap worker owns it on success). Without this a put that raised
    after set_pending left probes answering 202 until the sweep."""
    key = meta.get("cache_key")
    if key:
        _clear_pending_marker(key, jid)


def _jobs_snapshot() -> list:
    """Every ``(key, value)`` in the jobs Dict, from ONE streamed RPC.

    The worker-side scans used to list the keys and then `get` each record: a
    cohort of N jobs cost O(N) RPCs per scan and O(N^2) over the cohort, all on
    the Dict the API writes to. Measured 2026-09-19 with six workers and 200
    queued jobs: the jobs drained during a 30 s submit burst went from 15 to
    199 once each scan was one stream. ``items()`` is a single DictContents
    stream. It is neither ordered nor atomic - a write landing during the stream
    may or may not be in it - so a scan may DECIDE from it, but anything it
    deletes or fails is re-read first, as the per-key reads always were."""
    return [(str(k), v) for k, v in jobs_dict.items()]


def _prefetch_candidate(current_jid: str, engine: str | None = None):
    """The oldest OTHER queued job whose input may be warmed, as
    ``(kind, series_key, jid)``, or None. Which jobs qualify is decided by
    :func:`haversack.jobpolicy.prefetchable`, shared with the local server's
    head-of-queue check: this scan kept its own list of exclusions and had
    fallen two behind it (multi-input jobs, and jobs that asked for fresh
    bytes - which it then pinned, so their own refresh was refused).

    ``engine`` restricts the scan to jobs that will run on THIS worker. Every
    worker class has its own container, series cache and read-ahead, and the
    jobs dict is shared by all of them - so without the filter each worker
    warmed whichever queued job was oldest, engine regardless. Seen live on
    2026-09-06 with five engines deployed: the SynthStrip container pre-read
    the nnU-Net worker's upload into a read-ahead nothing there would pop,
    the nnU-Net worker spent its one-ahead slot staging a FastSurfer job's
    series and never warmed its own next job, and every container downloaded
    the same MRI once.

    Reads the Dict as one :func:`_jobs_snapshot` - it runs every 2 s per busy
    worker, and a `get` per record made that O(records) RPCs each time."""
    cands = []
    for k, m in _jobs_snapshot():
        if ":" in k or k == current_jid:
            continue                       # namespaced markers (inflight:/
                                           # artifacts:/cancel:) are not job
                                           # records; cancel: values are bare
                                           # floats and crashed this scan
        m = m if isinstance(m, dict) else {}
        if not prefetchable(state=m.get("state"), kind=m.get("kind"),
                            refresh_input=m.get("refresh_input"),
                            sources=m.get("source")):
            continue
        if engine is not None and m.get("task") and \
                _engines.engine_for_task(m["task"]).name != engine:
            continue                       # another worker's job: not ours to warm
        src = (m.get("source") or [{"kind": "upload"}])[0]
        sk = source_cache_key(src)
        if sk is not None and sk.ident:
            cands.append((m.get("created", 0), sk.kind, sk.key, m["id"]))
        else:
            cands.append((m.get("created", 0), "upload", None, m["id"]))
    return min(cands)[1:] if cands else None


def _prefetch_next(current_jid: str, stop, cache, read_ahead, vol_lock,
                   engine: str | None = None) -> None:

    """Best-effort CPU downloader, parallel to this GPU job: watch the shared
    jobs Dict for the oldest OTHER queued idc job and stage its series into the
    /dev/shm series cache. Scans every 2 s for the length of the run - a single
    scan at job start misses jobs whose submit lands moments later (the
    warm-chain case). With max_containers=1 the same container serves the next
    input, so the staging is always warm-handed. One-ahead only; the cache owns
    claims (atomic mkdir), commits (``.done`` marker) and LRU eviction, and a
    failed staging leaves nothing behind."""
    import threading

    def work():
        import shutil
        try:
            while not stop.is_set():
                nxt = _prefetch_candidate(current_jid, engine)
                if nxt is None:
                    stop.wait(2.0)
                    continue
                kind, series, njid = nxt
                if kind != "upload":
                    if cache.staging(series) or read_ahead.has(series):
                        stop.wait(2.0)
                        continue
                    if not cache.has(series):
                        t_f = time.time()
                        if not cache.prefetch(series):
                            stop.wait(2.0)
                            continue
                        print(f"[prefetch] {series[:13]} staged in {time.time() - t_f:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                    t_r = time.time()
                    if fill_read_ahead(series, cache=cache,
                                       read_ahead=read_ahead):
                        print(f"[read-ahead] {series[:13]} read in {time.time() - t_r:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                    return                     # one-ahead only
                # upload: bytes already sit on the jobs volume - copy the file to
                # tmpfs under the lock (a reload racing save+commit could drop a
                # result; and reading a stable local copy sidesteps any question
                # of open handles across later reloads), then read it there.
                if read_ahead.has(njid):
                    stop.wait(2.0)
                    continue
                t_r = time.time()
                tmp = Path("/dev/shm") / f"preread_{njid}"
                try:
                    local = None
                    with vol_lock:
                        scratch_vol.reload()
                        srcs = list((Path(SCRATCH_ROOT) / njid).glob("input_*"))
                        if srcs:
                            tmp.mkdir(exist_ok=True)
                            local = tmp / srcs[0].name
                            shutil.copy2(srcs[0], local)
                    if local is None:
                        stop.wait(2.0)         # upload not visible yet; retry
                        continue
                    if fill_read_ahead(njid, read_ahead=read_ahead,
                                       path=local):
                        print(f"[read-ahead] upload {njid} read in {time.time() - t_r:.1f}s "
                              f"(parallel to {current_jid})", flush=True)
                        return                 # one-ahead only
                    stop.wait(2.0)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print(f"[prefetch] failed: {e}", flush=True)

    threading.Thread(target=work, name="haversack-prefetch", daemon=True).start()


def _purgeable(meta: dict, now: float, ttl_s: float) -> bool:
    """The shared retention rule - see :func:`haversack.jobpolicy.purgeable`.
    Kept as a name here because the module's callers and tests use it."""
    from haversack.jobpolicy import purgeable

    return purgeable(meta, now, ttl_s)


#: How long a record must have been active before its call is probed. A call
#: finishes and emits `done` in one go, and terminal-wins lets one terminal
#: state replace another, so a probe racing that could turn a fresh `done` into
#: `failed`; two minutes is far past that window and far short of a stop.
ORPHAN_MIN_AGE_S = 120.0


def _retire_container(jid: str) -> None:
    """Take no further input in this container once one of its jobs was cancelled.

    Modal cancels a running sync input (the API's DELETE calls FunctionCall.cancel) by
    a signal whose handler raises InputCancellation at whatever bytecode the job was
    on - an import included. On 2026-09-19 a job cancelled in its first inference left
    `torch._dynamo` half initialized, and the next job in the same warm container failed
    on `partially initialized module 'torch._dynamo' has no attribute 'utils'`. What
    else such an interruption leaves behind cannot be known, so the container is not
    reused: the next job starts in a fresh one, restored from the snapshot."""
    print(f"[cancel] {jid} cancelled while running; this container takes no more inputs",
          flush=True)
    try:
        import modal.experimental
        modal.experimental.stop_fetching_inputs()
    except Exception as e:                  # never mask the cancellation itself
        print(f"[cancel] could not retire the container: {e}", flush=True)
    _drain_background()


#: Daemon threads a finished job may leave running, which die with the container.
_BACKGROUND_THREADS = ("haversack-artifacts", "haversack-sweep")


def _drain_background(timeout_s: float = 120.0) -> None:
    """Let an earlier job's artifact overlap, and a sweep, finish before the container
    goes. They are daemons: a container that exits under them kills them mid-way, and
    the overlap's last act is clearing its `artifacts:` marker - seen on 2026-09-19 as
    ClientClosed in `_clear_pending_marker` on a stopped container, a marker that then
    answers 202 to every probe of the key until the 900 s sweep. Retiring a container
    makes that exit immediate, so the retire waits (bounded) first."""
    deadline = time.monotonic() + timeout_s
    for t in list(threading.enumerate()):
        if t is threading.current_thread() or t.name not in _BACKGROUND_THREADS:
            continue
        t.join(max(0.0, deadline - time.monotonic()))


def _check_cancel_handler(jid: str) -> None:
    """Log when Modal's SIGUSR1 handler - its cancel - is no longer the installed one.

    A diagnostic for the open question of 2026-09-19: Modal logged the cancel signal
    of a job in model loading, and no InputCancellation reached run_job. Locally the
    same signal propagates through loading and inference alike. If something in the
    worker replaces the handler, this names it the first time it happens."""
    import signal
    h = signal.getsignal(signal.SIGUSR1)
    name = getattr(h, "__name__", repr(h))
    if name != "_cancel_input_signal_handler":
        print(f"[cancel] {jid}: SIGUSR1 handler is {name!r}, not Modal's", flush=True)


def _own_call_id():
    """The id of the call this worker is serving, or None outside one."""
    try:
        return modal.current_function_call_id()
    except Exception:
        return None


#: The failures of a probe that say nothing about the CALL - the caller's own link
#: to Modal dropped, was throttled, or hit a service hiccup. Every other exception
#: from ``FunctionCall.get(timeout=0)`` is Modal's report of how the call ended.
_PROBE_TRANSIENT = ("ConnectionError", "ServiceError", "ResourceExhaustedError",
                    "InternalError", "ClientClosed", "AuthError")


def _call_state(call_id) -> str:
    """``live``, ``finished``, ``dead`` or ``unknown``, from Modal's own view of a
    spawned call.

    Measured 2026-09-06 against modal 1.5.5: a queued or running call raises the
    builtin TimeoutError from ``get(timeout=0)``; a call whose function returned
    hands back its result; a call cancelled by ``modal app stop`` raises
    RemoteError; an unknown id raises NotFoundError. Read from the 1.5.5 source
    since: a crashed container (OOM, segfault, lost host) is InternalFailure, and a
    call whose container failed before our code ran re-raises that remote exception
    as its own class (RuntimeError, OSError...) - all of them ended calls.

    Until 2026-09-19 every exception counted as dead; once the API's single-flight
    lookup probed on every read of a key, one dropped connection failed a running
    job (review). An allowlist of ENDED exceptions came next and was wrong the other
    way: a crashed worker's InternalFailure read as unknown, forever live, and its
    key's flight never cleared (review round 2). So the short list is the transient
    one, checked by class name within ``modal.exception`` and as the builtin
    ConnectionError family; every other failure is an ended call.
    """
    if not call_id:
        return "dead"
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return "live"
    except Exception as e:
        transient = tuple(c for c in (getattr(modal.exception, n, None)
                                      for n in _PROBE_TRANSIENT) if isinstance(c, type))
        if isinstance(e, transient + (ConnectionError,)):
            return "unknown"
        return "dead"
    return "finished"


def _fail_if_orphaned(jid: str, meta: dict, now: float) -> bool:
    """Fail an active record whose spawned call is gone; True when it did.

    The one rule for "this flight will never land", shared by the worker's reconcile
    and the API's single-flight lookup: only a record older than ``ORPHAN_MIN_AGE_S``
    is probed (a fresh one may not have its call_id yet, and a call finishing races
    its own terminal emit), and the record is re-read immediately before it is
    failed, so a job finishing meanwhile is left alone."""
    if now - float(meta.get("started") or meta.get("created") or now) < ORPHAN_MIN_AGE_S:
        return False
    if _call_state(meta.get("call_id")) in ("live", "unknown"):
        return False
    cur = jobs_dict.get(jid) or {}                 # re-read: it may just have finished
    if cur.get("state") not in ("queued", "running"):
        return False
    _emit(jid, {"state": "failed", "finished": now,
                "error": "orphaned: the deployment that spawned this job was "
                         "stopped or replaced before it finished - resubmit it"})
    return True


def _reconcile_orphans(current_jid: str | None = None, now: float | None = None,
                       snapshot: list | None = None) -> list:
    """Fail every active record whose spawned call is gone. Returns their ids.

    The local server reconciles its job store at startup; this deployment had
    no counterpart, so a record whose call was cancelled by ``modal app stop``
    stayed ``queued`` for good. Five of them, 76 hours old, were found on
    2026-09-06, and they did three kinds of damage: they were never purged
    (queued records do not age out, on purpose - an active record that is
    stale is a symptom to surface); their ``inflight:`` markers made every
    probe of their keys report a flight that would never land; and, being the
    oldest queued records, they were what the prefetcher warmed on every job,
    so no real queued job was ever staged ahead. A ``running`` record whose
    call has returned is the same thing from the other side: the worker died
    before its terminal emit, and no emit is coming.

    Runs at container start and in the retention sweep, which passes the
    ``snapshot`` it has already taken (see :func:`_jobs_snapshot`). Only records
    older than ``ORPHAN_MIN_AGE_S`` are probed, and the record is re-read
    immediately before it is failed, so a job finishing while this runs is left
    alone.
    """
    now = time.time() if now is None else now
    failed = []
    if snapshot is None:
        try:
            snapshot = _jobs_snapshot()
        except Exception:
            return failed
    for k, m in snapshot:
        if ":" in k or k == current_jid:
            continue
        m = m if isinstance(m, dict) else {}
        if m.get("state") not in ("queued", "running"):
            continue
        if _fail_if_orphaned(k, m, now):
            failed.append(k)
    if failed:
        print(f"[reconcile] {len(failed)} orphaned job(s) failed: {' '.join(failed)}", flush=True)
    return failed


#: The least time between two retention sweeps in one container. The sweep
#: reads the whole jobs Dict, and ran after EVERY job: at 1342 keys that was
#: ~130 s per job (2026-09-19). Nothing it does is urgent at this scale - the
#: TTL is hours, an orphan is only probed after ORPHAN_MIN_AGE_S, a marker only
#: aged out after 900 s, and a finished job releases its own inflight marker in
#: `finally`.
JOBS_SWEEP_EVERY_S = 60.0

_sweep_lock = threading.Lock()
_last_sweep = float("-inf")               # time.monotonic() of this container's last claim
_sweep_thread = None                      # the running sweep, if any


def _sweep_due() -> bool:
    """Claim this container's next sweep: True at most once per
    ``JOBS_SWEEP_EVERY_S`` (and on the first job, since container start runs
    only the reconcile). The claim is taken before the sweep runs, so a sweep
    that fails is not retried by every following job."""
    global _last_sweep
    with _sweep_lock:
        t = time.monotonic()
        if t - _last_sweep < JOBS_SWEEP_EVERY_S:
            return False
        _last_sweep = t
        return True


def _bound_jobs_store(current_jid: str, vol_lock=None):
    """The retention policy for the jobs store, after every job: delete the
    finished job's own input upload (the bulk of the bytes - nothing reads an
    input after the job is terminal) every time, and at most once per
    ``JOBS_SWEEP_EVERY_S`` start :func:`_sweep_jobs_store` in a background
    thread, which is returned (None when no sweep was due). Keeps the jobs Dict
    listable and the jobs Volume bounded by traffic x TTL at ~result-size per
    job instead of ~input-size.

    The sweep is off the job's path, and ``vol_lock`` is held only for file
    operations and commits, never across Dict round trips. Before 2026-09-19
    the sweep ran here, inline and under the lock, after every job: with 1342
    Dict keys (558 queued, each probed) that was ~130 s per job, `run_job` did
    not return until it finished, and the overlap thread's `place` waited on
    the lock - so it logged as "preview 127s" (the render takes ~1 s). Even
    one snapshot plus the orphan probes a deep queue needs is tens of seconds,
    which is why "once a minute" alone would still stall a job a minute."""
    import contextlib
    lock = vol_lock if vol_lock is not None else contextlib.nullcontext()
    try:
        with lock:
            jdir = Path(SCRATCH_ROOT) / current_jid
            for f in jdir.glob("input_*"):
                f.unlink(missing_ok=True)
            scratch_vol.commit()
    except Exception as e:
        print(f"[purge] failed: {e}", flush=True)
    global _sweep_thread
    if _sweep_thread is not None and _sweep_thread.is_alive():
        return None                        # a deep queue's sweep can outlast the interval
    if not _sweep_due():
        return None

    def run():
        try:
            _sweep_jobs_store(current_jid, vol_lock)
        except Exception as e:
            print(f"[purge] failed: {e}", flush=True)

    _sweep_thread = threading.Thread(target=run, name="haversack-sweep", daemon=True)
    _sweep_thread.start()
    return _sweep_thread


def _sweep_jobs_store(current_jid: str, vol_lock=None) -> None:
    """Fail orphans, purge terminal records + their directories past
    HAVERSACK_JOBS_TTL_H, and drop markers whose job is gone - all decided from
    ONE :func:`_jobs_snapshot`. Every marker delete re-reads the marker (and,
    for an inflight marker whose job the snapshot did not show, the job) first:
    the snapshot is not atomic, and a newer flight may have installed its own
    marker since."""
    import shutil
    now, ttl_s = time.time(), JOBS_TTL_H * 3600.0
    snap = _jobs_snapshot()
    failed = set(_reconcile_orphans(current_jid, now, snap))  # first, so their
                                                               # markers drop below
    records = {k: v for k, v in snap if ":" not in k}
    purged = [k for k, m in records.items()
              if k != current_jid and k not in failed and _purgeable(m, now, ttl_s)]
    if purged:
        # Directories first, records after: the sweep outlives run_job in a daemon
        # thread and may wait here on the next job's save, so a container scaled
        # down or preempted in between must leave records naming what is left, for
        # the next sweep to retry. The other order leaked directories no record
        # names, which nothing ever cleans (review, 2026-09-19).
        import contextlib
        with vol_lock if vol_lock is not None else contextlib.nullcontext():
            for k in purged:
                shutil.rmtree(Path(SCRATCH_ROOT) / k, ignore_errors=True)
            scratch_vol.commit()
    for k in purged:
        try:
            del jobs_dict[k]
        except Exception:
            pass
    gone = set(purged)
    for k, v in snap:
        if k.startswith("inflight:"):
            jid = v                            # markers hold the job id
            if not isinstance(jid, str) or jid in gone:
                tgt = None
            elif jid in failed:
                tgt = {"state": "failed"}
            else:
                tgt = records.get(jid)
            # A flight that has landed - or crashed - is no flight. Judged by
            # state, not by `purgeable(ttl=0)`: that is a strict "older than
            # zero seconds", which a record failed in this same pass (the
            # reconcile above stamps the same `now`) does not satisfy, so its
            # marker lived on to the next job.
            if tgt is not None and tgt.get("state") not in _TERMINAL:
                continue
            if jobs_dict.get(k) != jid:
                continue                       # a newer flight owns it now
            if tgt is None and isinstance(jid, str) and jid not in gone:
                fresh = jobs_dict.get(jid)     # absent from the snapshot only?
                if isinstance(fresh, dict) and fresh.get("state") not in _TERMINAL:
                    continue
            try:
                del jobs_dict[k]
            except Exception:
                pass
        elif k.startswith("artifacts:"):       # a killed overlap thread leaves
            if now - float((v if isinstance(v, dict) else {}).get("t") or 0) <= 900:
                continue                       # a stale pending marker behind
            m = jobs_dict.get(k)               # re-read: a new flight may own it
            if now - float((m if isinstance(m, dict) else {}).get("t") or 0) > 900:
                try:
                    del jobs_dict[k]
                except Exception:
                    pass
        elif k.startswith("cancel:"):
            if now - float(v or 0) <= 900:
                continue
            if now - float(jobs_dict.get(k) or 0) > 900:
                try:
                    del jobs_dict[k]
                except Exception:
                    pass
    if purged:
        print(f"[purge] {len(purged)} finished jobs past {JOBS_TTL_H:g}h TTL", flush=True)


from haversack.jobpolicy import (TERMINAL as _TERMINAL,  # noqa: E402
                                 fill_read_ahead, prefetchable, record_inputs,
                                 refresh_cached_input, source_cache_key,
                                 take_pre_read)


def _emit(jid: str, update: dict) -> None:
    """Merge an update into the job's Dict record, terminal-wins: once a
    record is terminal, only idempotent terminal re-writes land - a worker
    progress emit racing an API cancel can no longer resurrect the record to
    'running' (which wedged it forever: the purge never touches active
    states, so the inflight marker and 202 probes lived eternally). The
    read-modify-write is still not atomic - this closes the lost-CANCEL
    class, which is the one with an unbounded blast radius."""
    meta = jobs_dict.get(jid) or {}
    if meta.get("state") in _TERMINAL and update.get("state") not in _TERMINAL:
        return
    if meta.get("state") == "cancelled" and update.get("state") not in (None, "cancelled"):
        # A cancel is the caller's word and final. Another terminal state may replace
        # a terminal one (a `done` landing after the orphan rule's `failed` is the job
        # recovering), but a worker that ran on past a DELETE must not turn the record
        # back into `done` - it did on the smoke of 2026-09-19.
        return
    meta.update(update)
    jobs_dict[jid] = meta


def _cancel_requested(jid: str) -> bool:
    """Whether the API has asked this job to stop (its `cancel:` marker)."""
    key = f"cancel:{jid}"
    if hasattr(jobs_dict, "contains"):
        return bool(jobs_dict.contains(key))
    return jobs_dict.get(key) is not None


_cls_extra = {"experimental_options": {"enable_gpu_snapshot": True}} if GPU_SNAPSHOT else {}


#: Serializes write+commit against reload on the inputs volume - a reload
#: between a write and its commit would discard the write.
_INPUTS_LOCK = threading.Lock()


def _content_store():
    """The input store, on its own Volume so it outlives every container.

    Volumes are not a POSIX-coherent shared filesystem: a write is published by
    commit() and someone else's write is seen by reload(). The single-writer
    claim inside SeriesCache is therefore not a true mutex across containers -
    but content addressing makes that harmless, because two writers of the same
    digest write identical bytes. What must hold is that a reader never sees a
    half-written entry, and the .done marker plus commit-after-write gives that:
    a reload lands on a committed version or the previous one.
    """
    from haversack.content import ContentStore
    from haversack.serve import SeriesCache

    def _no_fetch(key, entry):             # nothing is ever FETCHED into this one
        raise FileNotFoundError(f"{key} is not held by this server")

    cache = SeriesCache(Path(INPUTS_ROOT) / "content", _no_fetch,
                        budget_bytes=int(INPUTS_GB * (1 << 30)))
    return ContentStore(cache, commit=inputs_vol.commit, refresh=inputs_vol.reload,
                        lock=_INPUTS_LOCK)


def _refresh_series(ctx, meta: dict, key: str, rep, already: set | None = None,
                    *, kind: str | None = None, ident: str | None = None) -> None:
    """Drop a cached input when the caller sent ``Cache-Control: no-cache``.

    The same rule the local server follows, from the same place - it lives in
    :func:`haversack.jobpolicy.refresh_cached_input` rather than being restated
    here, which is how the two copies used to drift. Only the recording differs:
    the local executor sets a field on its job record, this emits into the Modal
    jobs dict.
    """
    refresh_cached_input(key,
                         wanted=bool((meta or {}).get("refresh_input")),
                         cache=ctx.series_cache,
                         read_ahead=getattr(ctx, "read_ahead", None),
                         reporter=rep,
                         on_skipped=lambda: _emit(meta.get("id"),
                                                  {"input_refresh_skipped": True}),
                         source=getattr(ctx, "_sources", {}).get(kind) if kind else None,
                         identifier=ident,
                         already=already)


# -- one worker container, several threads, four volumes ---------------------------
#
# While a thread runs ``Volume.reload()``, every path on THAT volume is ENOENT to the
# container's other threads, and the reload is refused while any of them holds a file on
# it open. Measured on Modal 2026-09-19 (modal 1.5.5, a thread listing 260 entries while
# another looped): a reload of the same volume, 63,412 ENOENT in 72,989 listings; a
# reload or commit of ANOTHER volume, none; a commit of the same volume, none in 61
# commits - alone, with another container committing, and with files open for reading
# or writing, none refused either. Modal's source reloads after a commit whenever the
# server's reply asks it to (``skip_reload``), which it never did here; commits are
# locked like reloads all the same, because that is the server's choice, not ours.
#
# A worker runs up to four threads at once: the job, the prefetcher (reloads scratch
# for a queued upload), the previous job's artifact overlap (places into the cache,
# commits it) and the retention sweep (removes job directories, commits scratch). So
# every access to the scratch or cache volume in a worker holds ``ctx._vol_lock``, as
# does every reload or commit of either. Held only for file operations - a copy, a
# save, a put, a commit - never across a compute or a Dict scan (a sweep holding it once
# logged as a 127 s preview).
#
# Before 2026-09-19 the job itself broke this: it dropped the lock, then ran on its
# upload from the volume and read its saved labels back from it (digest, artifact pair,
# the cache put) - a prefetcher reload in between was a FileNotFoundError failing the
# job, and the job's open upload made the prefetcher's reload raise, ending the
# prefetcher. It now works on container-local copies taken under the lock. ``_put``
# held no lock either; it does now, for the commit (above) and because it touches the
# cache volume the artifact thread commits.
#
# The weights and inputs volumes are touched only by the job thread in a worker, and a
# reload or commit of one volume hides no other, so their commits (``_ensure``,
# VoxTell's HF cache, the content store) race nothing here.


def _stage_uploads(ctx, jdir: Path, local: Path) -> list:
    """Copy the job's uploaded inputs (``input_*``) off the scratch volume into the
    container-local ``local``, under ``ctx._vol_lock`` with the reload that makes them
    visible; returns the local copies. The job reads only these: a reload or commit in
    another thread hides the volume's files for its duration, and the compute that
    reads an upload lasts far longer than any lock may be held."""
    import shutil
    local.mkdir(parents=True, exist_ok=True)
    out = []
    with ctx._vol_lock:
        try:
            scratch_vol.reload()
        except Exception as e:             # noqa: BLE001 - refused; what is visible may do
            print(f"[stage] {jdir.name}: scratch reload refused: {e}", flush=True)
        for f in sorted(jdir.glob("input_*")):
            dst = local / f.name
            shutil.copyfile(f, dst)
            out.append(dst)
    if not out:
        raise FileNotFoundError(f"job {jdir.name}: no uploaded input is visible on the "
                                f"scratch volume")
    return out


def _execute_job(ctx, jid: str, source_tokens: dict | None = None) -> str | None:
    """The engine-agnostic job body shared by every worker: fetch/stage/
    read, then ctx._ensure + ctx._compute (the engine), then save +
    publish_completion + artifact overlap. Only _ensure/_compute/_prepare
    differ per engine; everything else - queue, cache, markers, prefetch,
    single-flight, cancel - is identical."""
    from dataclasses import asdict

    from haversack.errors import Cancelled
    from haversack.progress import CancelToken, Reporter
    meta = jobs_dict.get(jid)
    if meta is None:
        return None
    if meta.get("state") == "cancelled":
        # Cancelled before it started: none of the job ran, so there is nothing to
        # distrust, and retiring here would cost a batch cancel a cold start per job.
        # A signal landing in these few lines raises through run_job and retires there.
        return None
    jdir = Path(SCRATCH_ROOT) / jid
    local = Path(JOB_LOCAL_ROOT) / jid          # this job's files, off the volume
    last = {"t": 0.0}

    def on_progress(p):
        now = time.time()
        if now - last["t"] >= 0.25 or p.stage in ("restore", "finalize"):
            last["t"] = now
            if _cancel_requested(jid):
                token.cancel()         # cooperative: honored at the next check
            _emit(jid, {"progress": asdict(p)})

    token = CancelToken()
    started = time.time()
    pinned = []
    # the content store is a DIFFERENT SeriesCache from ctx.series_cache on this
    # deployment (an inputs volume against /dev/shm), so its pins release through
    # its own object and cannot share the list above
    content_pinned = []
    # The worker names its own call: the API's `call_id` emit after the spawn and
    # this one are both read-modify-writes of the record, and a warm worker that read
    # it first wrote it back without the id - which the orphan rule reads as dead once
    # the job is two minutes old, failing it mid-run (review, 2026-09-19).
    _emit(jid, {"state": "running", "started": started,
                **({"call_id": cid} if (cid := _own_call_id()) else {})})
    prefetch_stop = threading.Event()
    _prefetch_next(jid, prefetch_stop, ctx.series_cache, ctx.read_ahead,
                   ctx._vol_lock, engine=getattr(ctx, "engine", None))   # CPU downloader + pre-reader
    outcome = None
    try:
        if meta.get("kind") == "prepare":
            rep = Reporter.of(on_progress, cancel=token)
            rep.stage("weights", meta["task"])
            result = ctx._prepare(meta["task"], progress=rep)
            _emit(jid, {"state": "done", "finished": time.time(), "result": result})
            return
        from haversack.serve import run_name
        # per-container weights provisioning (engine's own), under the caller's pin if any
        ctx._ensure(run_name(meta["task"], meta.get("version")))
        entries = meta.get("source") or [{"kind": "upload"}]
        if len(entries) > 1:
            # A multi-input task. Everything needed is already on `source`: each
            # entry carries the CANONICAL role the wire bound it to, uploads were
            # written to the job dir under that role, and remote siblings fetch
            # exactly like a single-input job through the same pinned cache.
            rep = Reporter.of(on_progress, cancel=token)
            uploads = (_stage_uploads(ctx, jdir, local)
                       if any(e.get("kind", "upload") == "upload" for e in entries) else [])
            staged = {}
            refreshed: set = set()     # one refresh per identifier, not per role
            for entry in entries:
                role = entry.get("role") or "image"
                kind = entry.get("kind", "upload")
                if kind == "upload":
                    # exact, by the name serve saved it under - never a prefix match
                    from haversack.serve import upload_role
                    found = [u for u in uploads if upload_role(u.name) == role]
                    if len(found) != 1:
                        raise FileNotFoundError(
                            f"job {jid}: {len(found)} uploads for role {role!r} on the "
                            f"scratch volume, expected one")
                    staged[role] = found[0]
                    continue
                if kind == "input":
                    # pinned like any other input: the content store is LRU, and
                    # without this an entry could be evicted from under a running
                    # job. The local twin has always pinned here (`_from_store`);
                    # this side did not, and on a shared inputs volume the eviction
                    # that matters comes from ANOTHER container, whose pins this
                    # one cannot see either way (2026-09-08).
                    digest = str(entry.get("id") or entry.get("sha256") or "")
                    ctx.content.pin(digest)
                    content_pinned.append(digest)
                    staged[role] = ctx.content.resolve(digest)
                    continue
                sk = source_cache_key(entry)
                ident, key = sk.ident, sk.key
                _refresh_series(ctx, meta, key, rep, refreshed,   # BEFORE our own pin
                                kind=kind, ident=entry.get("id"))
                ctx.series_cache.pin(key)
                pinned.append(key)
                rep.stage("fetch", f"{role} {ident[:8]}")
                t_f = time.time()
                staged[role] = ctx.series_cache.get_or_fetch(
                    key, check=rep.check,
                    credentials=(source_tokens or {}).get(kind))
                print(f"[fetch] {role} {ident[:13]} {time.time() - t_f:.1f}s", flush=True)
                rep.check()
            input_path = staged
        elif (kind := entries[0].get("kind", "upload")) == "input":
            # content this server already holds: nothing to fetch or copy - but
            # pinned, for the reason above
            digest = str(entries[0].get("id") or "")
            ctx.content.pin(digest)
            content_pinned.append(digest)
            input_path = ctx.content.resolve(digest)
        elif kind != "upload":
            src = entries[0]
            rep = Reporter.of(on_progress, cancel=token)
            sk = source_cache_key(src)
            ident, key = sk.ident, sk.key
            _refresh_series(ctx, meta, key, rep,       # BEFORE our own pin
                            kind=kind, ident=src.get("id"))
            ctx.series_cache.pin(key)
            pinned.append(key)
            if ctx.series_cache.has(key):
                how = "cached"
            elif ctx.series_cache.staging(key):
                how = "prefetched"
            else:
                how = "inline"
            rep.stage("fetch", ident[:13] if how == "inline" else how)
            t_f = time.time()
            input_path = ctx.series_cache.get_or_fetch(
                key, check=rep.check,
                credentials=(source_tokens or {}).get(kind))
            print(f"[fetch] {ident[:13]} {how} {time.time() - t_f:.1f}s", flush=True)
            rep.check()
            preread = take_pre_read(ctx.read_ahead, key,
                                    fresh_bytes_wanted=bool(meta.get("refresh_input")))
            if preread is not None:
                rep.stage("read", "preread")
                print(f"[read] {ident[:13]} preread", flush=True)
                input_path = preread
        else:
            preread = take_pre_read(ctx.read_ahead, jid, fresh_bytes_wanted=False)
            if preread is not None:
                rep2 = Reporter.of(on_progress, cancel=token)
                rep2.stage("read", "preread")
                print(f"[read] upload {jid} preread", flush=True)
                input_path = preread
            else:
                input_path = _stage_uploads(ctx, jdir, local)[0]
        from haversack.serve import RESULT_NAME, ResultCache, reference_input
        s = ctx._compute(input_path, meta, on_progress, token)
        # Asked once more before anything is saved or published, as the local server
        # does. The cooperative token is honored only at a patch, and Modal's own
        # cancel - a signal raising InputCancellation - did not reach run_job on the
        # smoke of 2026-09-19: a job DELETEd during model loading finished, published,
        # and reported done.
        if token.cancelled or _cancel_requested(jid):
            raise Cancelled("cancelled before publication")
        record_inputs(s, entries, meta.get("input_identity") or [], ctx.series_cache)
        # Saved locally and copied to the volume under the lock: everything after this
        # reads the labels - their digest, the artifact pair, the cache put - and reads
        # the local file, which no reload in another thread can hide.
        import shutil
        local.mkdir(parents=True, exist_ok=True)
        labels = local / RESULT_NAME
        s.save(labels)
        with ctx._vol_lock:
            jdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(labels, jdir / RESULT_NAME)   # the api's fallback reads it
            scratch_vol.commit()
        from haversack.serve import result_payload
        result = result_payload(s, labels)
        # The publication order (re-key, pair load, pending marker,
        # cache put, done, overlap start) lives in one place -
        # haversack.serve.publish_completion; this side supplies the Dict
        # markers, the volume commit, and _emit. The marker landing
        # before the put also closes the last C8-class window here:
        # the entry becomes visible at commit, and the marker must
        # never trail it.
        from haversack.serve import publish_completion

        def _migrate(old_key: str, new_key: str) -> None:
            _release_inflight(old_key, jid)
            _install_inflight(new_key, jid)
            meta["cache_key"] = new_key
            _emit(jid, {"cache_key": new_key})

        # What THIS job renders beside its labels: the request's list, off the record
        # this job read once at its start (no second Dict read, and never a scan), held
        # to what this container renders - the api checked it against ITS setting, and a
        # worker still warm from the previous deploy may have another. A record with no
        # list is a previous deploy's, or a path-surface ask: the deployment's set.
        from haversack.jobpolicy import wanted_deliverables
        wanted = wanted_deliverables(meta.get("deliverables"), ARTIFACTS)

        def _set_pending(key: str) -> None:
            _set_pending_marker(key, jid, wanted)

        def _clear_pending(key: str) -> None:
            _clear_pending_marker(key, jid)

        def _put(key: str) -> str:
            # Under the lock (see the block above _stage_uploads): the previous job's
            # artifact thread places into and commits this volume, and a commit that
            # ends in a reload would hide the generation add_artifact checks for - which
            # then drops the artifact without a word - or fail this put mid-way.
            with ctx._vol_lock:
                gen = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).put(
                    key, labels, result,
                    {"identity": meta.get("input_identity"), "task": meta["task"],
                     "options": meta.get("options"), "job": jid,
                     "computed": started})
                cache_vol.commit()
            return gen

        def _mark_done() -> None:
            _emit(jid, {"state": "done", "finished": time.time(),
                        "result": result})

        def _start(pair, key: str, generation=None) -> None:
            threading.Thread(target=ctx._artifact_worker,
                             args=(pair, key, jid, meta["task"], generation, wanted),
                             name="haversack-artifacts", daemon=True).start()

        meta["cache_key"], _ = publish_completion(
            segmenter=ctx.seg, task=meta["task"],
            identity=tuple(meta.get("input_identity") or ()),
            options=meta.get("options") or {},
            cache_key=meta.get("cache_key"),
            labels_path=labels, input_image=reference_input(input_path),
            # the request's list, not ARTIFACTS: nothing it declined is rendered, and an
            # empty list starts no thread and loads no pair
            artifacts=wanted, cache_enabled=True,
            migrate_key=_migrate, set_pending=_set_pending,
            clear_pending=_clear_pending, put=_put,
            mark_done=_mark_done, start_worker=_start)
    except Cancelled:
        _emit(jid, {"state": "cancelled", "finished": time.time()})
        _clear_own_artifacts_marker(jid, meta)
        outcome = "cancelled"
    except Exception as e:               # noqa: BLE001 - reported to the client
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)                          # worker log for diagnosis
        _emit(jid, {"state": "failed", "finished": time.time(),
                    "error": f"{type(e).__name__}: {e}\n--- traceback ---\n{tb[-1600:]}"})
        _clear_own_artifacts_marker(jid, meta)   # a put that failed after
    finally:                                     # set_pending must not 202
        for key in pinned:                       # until the sweep
            ctx.series_cache.unpin(key)
        for digest in content_pinned:
            ctx.content.unpin(digest)
        prefetch_stop.set()            # end the scan loop with the run
        import shutil
        shutil.rmtree(local, ignore_errors=True)
        _bound_jobs_store(jid, ctx._vol_lock)
        if meta.get("cache_key"):
            _release_inflight(meta["cache_key"], jid)
    return outcome


class _WorkerBase:
    """Everything every worker does regardless of engine: the source registry +
    shared caches, the artifact overlap, and the job entry point.

    A plain (undecorated) base class - Modal collects ``@modal.enter`` /
    ``@modal.method`` across the MRO, so the three decorated workers below
    inherit these; the base itself must NOT be decorated or it becomes a
    ``modal.Cls`` instance that cannot be subclassed. What stays per-worker is
    exactly what differs: ``preload`` (its body IS the image's import set, and it
    runs pre-snapshot), the engine hooks, and the decorator's image/memory.
    """

    #: The registry engine this worker runs (a plain string: Modal harvests
    #: class attributes, so never hold the Engine object itself here).
    engine: str = _engines.NNUNETV2

    def _engine_setup(self) -> None:
        """Per-engine construction, after the shared setup. The nnU-Net worker
        builds a Segmenter; engine workers attach a describe-only shim."""
        self.seg = _EngineShim(self.engine)

    @modal.enter()
    def setup(self):
        """Post-restore, GPU attached: everything CUDA-adjacent lives here (unless
        the GPU snapshot already carries it)."""
        _pkg_dir()
        _check_volumes_attached()
        from haversack.serve import ReadAhead
        self._vol_lock = threading.Lock()    # scan-thread reload vs save+commit
        # after the lock: the `result:` source reads the cache volume under it
        self._sources = _worker_sources(self._vol_lock)
        self.series_cache = _worker_series_cache(self._sources, "/dev/shm/series_cache",
                                                 int(SHM_CACHE_GB * (1 << 30)))
        self.read_ahead = ReadAhead()
        self.content = _content_store()      # uploads referred to by digest
        # Never reset: under the GPU snapshot `preload` already ran and recorded
        # WARM_TASK here, and wiping it costs a redundant prepare + a multi-GB
        # weights_vol.commit() on the first job of every restored container.
        if not hasattr(self, "_ensured"):
            self._ensured = set()        # tasks whose weights this container verified
        if not hasattr(self, "seg"):
            self._engine_setup()
        try:
            # Records orphaned by the previous deployment are failed before the
            # first job's prefetcher scans for them; the per-job pass below
            # would only catch them after that first job had warmed one.
            _reconcile_orphans()
        except Exception as e:                    # never keep a container from starting
            print(f"[reconcile] skipped: {e}", flush=True)

    def _artifact_worker(self, pair, cache_key: str, jid: str, task: str,
                         generation=None, names=None) -> None:
        """Post-done artifacts via the shared overlap body; this side's place
        is vol-locked add_artifact + tmpfs cleanup, and finish commits once,
        logs, and always deletes the pending marker. ``names`` is what this job
        renders - its request's deliverables; None is the deployment's set. It changes
        WHAT is rendered and nothing about where the volume is touched: every place and
        the one commit still hold ``_vol_lock``, and a job that renders nothing never
        starts this thread."""
        from haversack.serve import ResultCache, artifact_overlap

        def _place(name: str, path) -> bool:
            with self._vol_lock:
                # constructed under the lock too: ResultCache() mkdirs its root, which a
                # scratch-or-cache reload in another thread hides (review, 2026-09-19)
                cache = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP)
                ok = cache.add_artifact(cache_key, name, path, generation=generation)
            Path(path).unlink(missing_ok=True)
            return ok

        def _finish(placed) -> None:
            try:
                if placed:
                    with self._vol_lock:
                        cache_vol.commit()
                print("[artifacts] overlap "
                      + " ".join(f"{n} {dt:.1f}s" for n, dt in placed)
                      + f" placed={[n for n, _ in placed]}", flush=True)
            except Exception as e:
                print(f"[artifacts] overlap failed: {e}", flush=True)
            finally:
                _clear_pending_marker(cache_key, jid)

        artifact_overlap(pair, task, ARTIFACTS if names is None else names,
                         preview_out=Path("/dev/shm") / f"preview_{jid}.png",
                         statistics_out=Path("/dev/shm") / f"stats_{jid}.json",
                         place=_place, finish=_finish)

    # -- engine hooks; _execute_job calls these. Engines that ship their weights
    # in their image have nothing to install, so these are the defaults.
    def _prepare(self, task: str, progress=None) -> dict:
        return {"engine": self.engine, "task": task,
                "note": "weights ship with the engine image"}

    def _ensure(self, task: str) -> None:
        """Nothing to install - but a PINNED name is checked against this container's own
        build through its catalog (ImageBakedEcosystem.ensure refuses a version it does not
        run). The API checked the pin against ITS registry, and a redeploy does not preempt
        a warm worker still running the previous build."""
        if "@" in str(task):
            from haversack.ecosystems import EcosystemCatalog
            EcosystemCatalog().prepare(task)

    def _compute(self, input_path, meta, on_progress, token):
        raise NotImplementedError

    @modal.method()
    def run_job(self, jid: str, source_tokens: dict | None = None) -> None:
        try:
            outcome = _execute_job(self, jid, source_tokens)
        except modal.exception.InputCancellation:
            _retire_container(jid)
            raise
        _check_cancel_handler(jid)
        if outcome == "cancelled":
            # The API cancels twice - the cooperative marker and FunctionCall.cancel -
            # and when the job ends by the first, the second's signal may already
            # have raised somewhere a library swallowed it: the smoke of 2026-09-19
            # saw the signal arrive and no InputCancellation reach run_job. Where it
            # landed is unknowable, so a cancelled job retires the container either way.
            _retire_container(jid)


@app.cls(gpu=GPU, timeout=3600, memory=32768, scaledown_window=SCALEDOWN,
         max_containers=MAX_CONTAINERS,
         volumes={WEIGHTS_ROOT: weights_vol, SCRATCH_ROOT: scratch_vol,
                  CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol},
         enable_memory_snapshot=SNAPSHOT, **_cls_extra)
class Worker(_WorkerBase):
    """The nnU-Net worker: runs every ecosystem whose engine is ``nnunetv2``
    (ts.v2, moose, custom) through the Segmenter."""

    engine = _engines.NNUNETV2

    def _engine_setup(self):
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from haversack import Segmenter
        self.seg = Segmenter(device="cuda", weights=WEIGHTS_ROOT, cache_models=5,
                             # the same deployment policy `haversack serve` takes as a
                             # flag: a transposed model runs only where the operator
                             # said so, and a request can never ask for it
                             allow_transpose=(os.environ.get("HAVERSACK_ALLOW_TRANSPOSE")
                                              or "").strip().lower()
                             not in ("0", "false", "no", "off", ""))

    _gpu_setup = _engine_setup          # legacy name used by the snapshot path

    @modal.enter(snap=SNAPSHOT)
    def preload(self):
        """The import bill, paid once per deploy: Modal snapshots memory after this
        and boots later cold containers from the snapshot. With classic snapshots
        CUDA must not be touched here - plain imports only. With the experimental
        GPU snapshot (HAVERSACK_GPU_SNAPSHOT=1) the CUDA state itself is captured, so
        the Segmenter is built and WARM_TASK's model loaded ONTO the GPU before the
        snapshot - a restored cold container then starts with a loaded model."""
        _pkg_dir()
        import nnunetv2  # noqa: F401
        import torch     # noqa: F401
        import haversack     # noqa: F401 - pulls the pipeline import chain
        if GPU_SNAPSHOT:
            from haversack.weights_fetch import ensure_task_weights
            self._engine_setup()
            ensure_task_weights(WARM_TASK, WEIGHTS_ROOT, progress=None)
            self.seg.warm(WARM_TASK)
            self._ensured = {WARM_TASK}

    # -- engine hooks (nnU-Net): unlike the image-baked engines, weights install
    # into the shared volume, so these do real work.
    def _prepare(self, task: str, progress=None) -> dict:
        r = self.seg.prepare(task, progress=progress)
        weights_vol.commit()
        self._ensured.add(task)
        return r

    def _ensure(self, task: str) -> None:
        if task not in self._ensured:
            # Volume.commit scans the whole multi-GB weights tree, so ensure+
            # commit once per container, not per job. seg.prepare is catalog-
            # aware: ts.v2, moose, custom all install through it.
            self.seg.prepare(task)
            weights_vol.commit()
            self._ensured.add(task)

    def _compute(self, input_path, meta, on_progress, token):
        from haversack.serve import run_name
        return self.seg.segment(input_path, run_name(meta["task"], meta.get("version")),
                                progress=on_progress,
                                cancel=token, **(meta.get("options") or {}))


class _EngineShim:
    """A Segmenter-shaped stand-in for engine workers, so ``publish_completion``'s
    re-key (weights_versions_of -> describe) reports the same weights identity the
    API-side describe does. Both read the registry, so they cannot drift - the
    divergence that once made every bare read 404 a cached result."""

    _catalog = None

    def __init__(self, engine: str):
        self._engine = engine

    def engine_for(self, task):
        """This worker's engine row, whatever the task: ``weights_versions_of`` reads its
        ``cache_epoch`` into the key. Missing until 2026-09-12, the first day an engine
        declared one - the API keyed FastSurfer with ``fastsurfer@epoch=1`` and this shim
        re-keyed without it, publishing every result into a slot nothing looks up (caught
        by test_the_engine_shim_reports_the_weights_identity_the_api_reports)."""
        return _engines.ENGINES[self._engine]

    def describe(self, task):
        identity = _engines.ENGINES[self._engine].weights_identity
        if identity is not None:
            return {"weights_installed": identity()}
        # An engine whose identity is PER TASK rather than constant - a CATALOG
        # like MONAI, where two bundles must not collide on one cached result.
        # `weights_identity=None` means "the ecosystem answers", so ask it, the
        # same way the API-side describe does. Returning [] here instead (which
        # this did until 2026-08-27) degrades weights_versions_of to "unknown",
        # and publish_completion then re-keys the finished result onto a key the
        # API never computes - so every job of that engine published into a slot
        # nothing would ever look up, and none of them ever hit the cache.
        cls = type(self)
        if cls._catalog is None:
            from haversack.ecosystems import EcosystemCatalog
            cls._catalog = EcosystemCatalog(root=WEIGHTS_ROOT)
        try:
            info = cls._catalog.info(task) or {}
        except Exception:
            return {"weights_installed": []}
        return {"weights_installed": info.get("weights_installed") or []}

    def resolve_task(self, t):
        return t


















# `modal deploy src/haversack/modal_app.py` loads this file BY PATH, so it lands in
# sys.modules under a synthetic name and `haversack.modal_app` is left unclaimed. The
# adapters below import from that canonical name, and without this line that import
# executes this module a SECOND time as a different object - re-entering the composer
# while each adapter is still half-initialized, and failing with the very error the
# composer raises for a direct adapter import. The deploy failed exactly there, which
# is the one thing no static check could have told us (2026-09-09).
#
# setdefault, not assignment: under a normal `import haversack.modal_app` the name is
# already this module and claiming it again would be a no-op at best.
sys.modules.setdefault("haversack.modal_app", sys.modules[__name__])

#: engine name -> the worker class this deployment can run.
#:
#: Each optional engine's image and `@app.cls` worker live in
#: ``haversack/engines/modal_<engine>.py``, beside the runtime they deploy, and are
#: imported HERE: at the bottom, after everything they import from this module exists,
#: and only when the engine is enabled - Modal resolves the decorators at import, so an
#: image must not be built for an engine this deployment does not run.
#:
#: This replaces a map from engine name to CLASS NAME STRING that was looked up in
#: ``globals()``, plus one module-level flag per engine. Both had a silent, deploy-fatal
#: failure mode - a missing flag or a mistyped class name dropped the engine from every
#: deploy while its variable was set to 1 - and both needed `_wiring_problems` to catch
#: what this shape makes impossible instead.
ENGINE_WORKERS: dict = {_engines.NNUNETV2: Worker}
for _name in _engines.ENGINES:
    if _name == _engines.NNUNETV2 or not _engines.enabled(_name):
        continue
    _mod_name = f"haversack.engines.modal_{_name}"
    _partial = sys.modules.get(_mod_name)
    if _partial is not None and getattr(_partial.__spec__, "_initializing", False):
        # This adapter is what imported US: it is mid-execution further up the stack and
        # will define its own class when it resumes. That is the order a Modal WORKER
        # container uses - Modal imports the module the class LIVES in, which is the
        # adapter, not this module - and treating it as a broken adapter is what made
        # every engine worker crash on start while the api container was fine, because
        # the api enters through this module instead (2026-09-09).
        continue
    _adapter = importlib.import_module(_mod_name)
    _worker = getattr(_adapter, "WORKER", None)
    if _worker is None:
        raise ImportError(
            f"{_adapter.__name__} defines no WORKER - every adapter must export the class "
            "the composer deploys.")
    if getattr(_adapter, "ENGINE", None) != _name:
        raise ImportError(
            f"{_adapter.__name__} says it deploys {getattr(_adapter, 'ENGINE', None)!r} "
            f"but its filename says {_name!r}")
    ENGINE_WORKERS[_name] = _worker


def _worker_classes() -> dict:
    """engine name -> worker class, for the engines this deployment can run.

    Filtered on each call rather than returned frozen, so a test that patches an enable
    flag sees the effect. An engine that was off when this module was imported has no
    adapter loaded at all and simply is not here - which is the point: its image was
    never built either."""
    return {n: c for n, c in ENGINE_WORKERS.items() if _engines.enabled(n)}


def _spawn_worker(task: str, jid: str, source_tokens=None):
    """Dispatch to the worker for this task's engine.

    Routes on the *grammar* - every wire form is canonicalized to ``eco:task``
    before it gets here - through the engine registry, so a new engine needs no
    branch: one registry row plus a worker class. An ecosystem with no engine
    entry (every nnU-Net catalog) falls through to the default engine."""
    engine = _engines.engine_for_task(task).name
    workers = _worker_classes()
    if engine not in workers:
        env = _engines.ENGINES[engine].enabled_env
        raise RuntimeError(f"the {engine} engine is not enabled on this "
                           f"deployment (set {env}=1 at deploy)")
    return workers[engine]().run_job.spawn(jid, source_tokens=source_tokens)


class ModalExecutor:
    """The :func:`haversack.serve.create_app` executor protocol over Modal primitives."""

    #: Supplied by :func:`api` after construction - the API container builds a
    #: catalog-only Segmenter (device is cosmetic there; jobs run on the Worker).
    #: Declared here because `submit` reads it through `weights_versions_of`, so
    #: it is part of this class's contract rather than a field `api` happens to
    #: attach. It does NOT fail loudly if left unset - `weights_versions_of`
    #: catches and answers ["unknown"], and create_app's `seg = executor.segmenter`
    #: then works with None - so declaring the slot is what makes the requirement
    #: visible at all.
    #: tests/test_executor_contract.py checks the declared surface against what
    #: create_app actually touches, and an undeclared slot reads as a gap.
    segmenter = None

    # One lock per API container: a scratch_vol.reload() here discards other
    # requests' uncommitted upload writes (the api function runs many inputs
    # concurrently), so every reload and every upload-write+commit serialize
    # through it. The worker has its own _vol_lock for the same reason.
    volume_guard = threading.Lock()

    @functools.cached_property
    def content(self):
        """Where a PUT /v1/inputs lands, and what a {"kind": "input"} source is
        checked against at submit. The same volume the workers read, so content
        stored here is resolvable there."""
        return _content_store()

    @property
    def sources(self):
        # every hosted source, and `result:` over THIS container's view of the cache
        # volume - the executor builds that one, as LocalExecutor does (it is in nobody's
        # default_sources()). A worker has its own, over its own lock: _worker_sources.
        from haversack.sources import ResultSource, default_sources, registry
        return registry(default_sources() + [ResultSource(_api_result_entry)])

    def submit_prepare(self, jid, jdir, task):
        meta = {"id": jid, "task": task, "options": {}, "kind": "prepare",
                "state": "queued", "created": time.time(), "source": []}
        jobs_dict[jid] = meta
        call = _spawn_worker(task, jid)
        # the orphan rule reads a record with no call as dead: a prepare that
        # ran past two minutes was failed by the reconcile (review, 2026-09-19)
        _emit(jid, {"call_id": call.object_id})
        return meta

    #: What this deployment renders: the default of a request that names no deliverables,
    #: and the most one may name (the submit door refuses the rest), as on the local server
    artifacts = ARTIFACTS

    def _pending_marker(self, key: str):
        """The live ``artifacts:`` marker for ``key`` - one Dict get - or None."""
        m = jobs_dict.get(f"artifacts:{key}")
        if not (isinstance(m, dict) and m.get("state") == "pending"):
            return None
        # the same 900 s rule the local executor applies on read: a marker a killed
        # overlap thread left behind must not answer 202 forever
        if time.time() - float(m.get("t") or 0) > 900:
            # and the marker goes, as it does locally: left in place, the writer's
            # refuse-if-present rule would decline to mark a genuinely new flight
            try:
                jobs_dict.pop(f"artifacts:{key}", None)
            except Exception:                  # noqa: BLE001 - a read path must not fail on it
                pass
            return None
        return m

    def artifact_state(self, key: str, name: str | None = None) -> str:
        """As the local executor's: "pending" while a worker's overlap is still placing
        this entry's artifacts - and, asked about ONE deliverable, only when that render
        was asked for it (the marker's ``names``; a marker without them is a previous
        deploy's, which rendered its whole set)."""
        from haversack.jobpolicy import pending_covers
        m = self._pending_marker(key)
        if m is None:
            return "absent"
        return ("pending" if name is None or pending_covers(m.get("names"), name)
                else "absent")

    def _unrendered_on_hit(self, key: str, wanted, hit) -> dict:
        """``{name: why}`` for what a cache hit's list names and its stored generation
        will not have (2026-09-20). Empty when the generation holds everything asked
        for, or a render still running will bring the rest.

        Here a hit cannot render what is missing, as the local server does, and says
        so: artifacts are rendered by the worker that computes a result, from the input
        it staged and under its ``_vol_lock``; a hit reaches no worker, this container
        cannot see a worker's staged inputs, and a volume read in 2 GB of api memory is
        not where a whole CT belongs. A render-only job is the follow-up, not a second
        mechanism slipped in here.

        Costs one Dict get (never a scan), and only when something is missing. A miss
        is believed only from a view newer than the marker's read: the worker places,
        COMMITS, and only then clears its marker, so once no render is pending a fresh
        view holds whatever one placed - the rule every miss here follows (2026-09-19).
        """
        from haversack.jobpolicy import RENDER_BUSY, missing_deliverables, pending_covers
        from haversack.serve import ResultsNotVisible
        missing = missing_deliverables(wanted, Path(hit[0]).parent)
        if not missing:
            return {}
        m = self._pending_marker(key)
        since = time.monotonic()               # AFTER the marker's read
        if m is not None:
            return {d: RENDER_BUSY for d in missing
                    if not pending_covers(m.get("names"), d)}
        try:
            again = _confirm_cache_absent(key, since)
        except ResultsNotVisible:
            return {d: DELIVERABLE_NOT_VISIBLE for d in missing}
        if again is not None:
            missing = missing_deliverables(missing, Path(again[0]).parent)
        return {d: DELIVERABLE_NEEDS_A_COMPUTE for d in missing}

    def cache_list(self, *, keys=None, limit=None, after=None, accept=None, match=None):
        return _list_cache(keys=keys, limit=limit, after=after, accept=accept, match=match)

    supports_push = False                    # SSE uses the server's poll branch
    accepting = True                         # Modal's backlog is the queue

    def new_job_dir(self):
        import uuid
        jid = uuid.uuid4().hex[:12]
        d = Path(SCRATCH_ROOT) / jid
        d.mkdir(parents=True, exist_ok=True)
        return jid, d

    _weights_reload_lock = threading.Lock()
    _weights_reloaded_at = 0.0
    _wv_cache: dict = {}                   # task -> (versions, stamped at)

    def _fresh_weights_versions(self, task):
        """weights_versions_of, but stale-proof: an API container's mounted
        weights volume is frozen at container start, so after the worker
        first-installs a task this side kept deriving weights=["unknown"] -
        every probe missed the re-keyed entry and every Prefer'd GET
        recomputed, for the container's remaining lifetime. On "unknown",
        reload the volume and re-derive - THROTTLED to once per 30 s per
        container: "unknown" is also the honest permanent answer for weights
        haversack did not install (TS-installed, hand-copied), and an unthrottled
        version reloaded a multi-GB volume on every HEAD probe forever."""
        from haversack.serve import weights_versions_of
        cls = type(self)
        cached = cls._wv_cache.get(task)
        if cached is not None and time.time() - cached[1] < 30.0:
            return cached[0]               # the listing derives per ENTRY -
                                           # without this that is a describe()
                                           # volume walk per row
        wv = weights_versions_of(self.segmenter, task)
        if any("unknown" in str(v) for v in wv):
            reloaded = self._reload_weights()
            if reloaded is None:           # throttled: answer as-is, uncached
                return wv
            if not reloaded:               # the reload failed
                cls._wv_cache[task] = (wv, time.time())
                return wv
            wv = weights_versions_of(self.segmenter, task)
        cls._wv_cache[task] = (wv, time.time())
        return wv

    def _reload_weights(self):
        """Reload the frozen weights volume, throttled to once per 30 s per container:
        True reloaded, False the reload failed, None throttled. One throttle for the key's
        freshness and the pin check, so together they reload no more often than before."""
        cls = type(self)
        with cls._weights_reload_lock:
            if time.time() - cls._weights_reloaded_at < 30.0:
                return None
            cls._weights_reloaded_at = time.time()
            try:
                weights_vol.reload()
            except Exception:
                return False
        return True

    def installed_versions(self, task):
        """serve.installed_versions, stale-proof the way _fresh_weights_versions is: this
        container's weights volume is frozen at start, so a task a worker has installed
        since reads as unknown - and unknown refuses every pinned read and keeps every
        pinned submit out of the cache until something reloads it (review 2026-09-12)."""
        from haversack.serve import installed_versions
        have = installed_versions(self.segmenter, task)
        if have is None and self._reload_weights():
            have = installed_versions(self.segmenter, task)
        return have

    def weights_versions(self, task) -> list:
        """What ``resource_key`` keys ``task`` on - the listing asks it once a task a
        request and derives its keys itself (``serve.result_key``), where asking
        ``resource_key`` key by key described the task again for every option set and
        every identity (2026-09-20). One door, so the two cannot disagree."""
        return self._fresh_weights_versions(task)

    def resource_key(self, identity, task: str, opts=None) -> str:
        """The key of one identity - what the path surface asks - or of several: the
        listing's key round trip hands over every role's identity of a multi-input
        entry, which is what such an entry was keyed on (2026-09-20)."""
        from haversack.serve import result_key
        ids = (identity,) if isinstance(identity, str) else tuple(identity)
        return result_key(ids, task, opts or {}, self.weights_versions(task))

    def submit(self, jid, jdir, input_path, task, options, *, source=None,
               identity=(), no_cache: bool = False, source_tokens=None,
               inputs: tuple = (), refresh_input: bool = False,
               version: str | None = None, deliverables=None):
        # `deliverables` is the request's list (None: it named none). It is written on
        # the job's record - which the worker reads ONCE, when the job starts, so the
        # list reaches it with no Dict read of its own - and it never reaches
        # `result_key`: what is rendered beside a result is not part of what it is.
        # `refresh_input` is recorded on the job meta and read by the worker's
        # fetch (`_refresh_series`), the way the local executor's dispatcher does.
        # `inputs` (the role -> local path binding) is accepted for signature
        # parity with LocalExecutor and deliberately not forwarded: this executor
        # is stateless by construction, and the worker rebuilds the binding from
        # `source` - each entry carries its canonical role, and uploads were
        # written into the job dir under that role. Sending server-local paths
        # through a Dict to another container would be sending it a lie.
        from haversack.jobpolicy import wanted_deliverables
        from haversack.serve import RESULT_NAME, result_key
        wanted = wanted_deliverables(deliverables, ARTIFACTS)
        with self.volume_guard:
            # Make any upload visible to the worker - and only then: a commit was
            # 0.67 s of every submit's 1.48 (2026-09-19), paid by idc:/input: jobs
            # that wrote nothing. Judged by what IS in the directory, not by the
            # source kinds, so no caller can write there and have it go unseen. An
            # empty one is removed, leaving nothing uncommitted behind; the worker
            # creates it when it saves (Segmentation.save makes its parents).
            try:
                jdir.rmdir()
            except OSError:                     # not empty (or already gone)
                scratch_vol.commit()
        key = None
        if identity:
            key = result_key(identity, task, options,
                             self._fresh_weights_versions(task))
            if not no_cache:
                hit = self.cache_get(key)
                if hit is not None:
                    meta = {"id": jid, "task": task, "options": options,
                            "input_identity": list(identity), "state": "done",
                            "cached": True, "created": time.time(),
                            "started": time.time(), "finished": time.time(),
                            # the VOLUME's path: hit[0] is this container's own copy,
                            # which no other container (nor this one restarted) has
                            "result": hit[1],
                            "cache_path": str(Path(CACHE_ROOT) / key
                                              / Path(hit[0]).parent.name / RESULT_NAME),
                            # the handle the job result route resolves - and leases -
                            # the entry by; cache_path names one generation, which a
                            # later publication of the key lets pruning reclaim
                            "cache_key": key,
                            "deliverables": list(wanted),
                            # a pinned ask answered from the cache still reports its pin,
                            # as the local executor does (seen missing on Modal, 2026-09-12)
                            **({"version": version} if version else {})}
                    unavailable = self._unrendered_on_hit(key, wanted, hit)
                    if unavailable:
                        meta["deliverables_unavailable"] = unavailable
                    jobs_dict[jid] = meta
                    return meta
        meta = {"id": jid, "task": task, "options": options,
                "source": list(source or [{"kind": "upload"}]),
                "input_identity": list(identity), "cache_key": key,
                "deliverables": list(wanted),
                "refresh_input": bool(refresh_input),
                # the caller's pin: the worker runs run_name(task, version), so its
                # catalog installs that version or refuses it (see serve.run_name)
                **({"version": version} if version else {}),
                "state": "queued", "created": time.time()}
        jobs_dict[jid] = meta
        # no marker for a pin the API could not verify: it is keyed on an UNKNOWN
        # installed version (see LocalExecutor.submit); the worker's re-key installs one
        if key and not (version and no_cache):
            jobs_dict[f"inflight:{key}"] = jid
        call = _spawn_worker(task, jid, source_tokens)
        _emit(jid, {"call_id": call.object_id})   # merge, never clobber worker emits
        return meta

    def cache_get(self, key):
        from haversack.serve import ResultCache
        _reload_cache_view(max_age=CACHE_FRESH_S)
        return _read_cache(key)

    def confirm_absent(self, key, since):
        """A miss ``cache_get`` returned, re-asked of a view newer than ``since``: the
        entry, None when it is verifiably absent, or ResultsNotVisible (a 503) when no
        reload takes. See :func:`_confirm_cache_absent`."""
        return _confirm_cache_absent(key, since)

    #: jid -> monotonic time its call was last seen live, so a key asked about
    #: repeatedly costs one call probe per FLIGHT_LIVE_TTL_S, not one per request
    _flight_seen_live: dict = {}
    FLIGHT_LIVE_TTL_S = 30.0

    def find_inflight(self, key):
        jid = jobs_dict.get(f"inflight:{key}")
        if not jid:
            return None
        meta = jobs_dict.get(jid) or {}
        if meta.get("state") not in ("queued", "running"):
            return None
        return jid if self._flight_alive(jid, meta) else None

    def _flight_alive(self, jid, meta) -> bool:
        """Whether an active record's spawned call can still finish it.

        A deployment stopped mid-job leaves its records `queued`/`running` and their
        `inflight:` markers in the Dict, and only a WORKER's reconcile failed them - so
        after a restart with no new jobs nothing did, and every plain GET of such a key
        joined a flight that would never land and waited out its 30 s (the 1,206-job
        run of 2026-09-19 worked around it with HEAD). The lookup now applies the
        reconcile's own rule, :func:`_fail_if_orphaned`, so the record is failed here
        and the next asker sees no flight. A live answer is remembered for
        ``FLIGHT_LIVE_TTL_S``: the probe is a ~60 ms round trip."""
        seen = self._flight_seen_live.get(jid)
        if seen is not None and time.monotonic() - seen < self.FLIGHT_LIVE_TTL_S:
            return True
        now = time.time()
        if _fail_if_orphaned(jid, meta, now):
            self._flight_seen_live.pop(jid, None)
            print(f"[reconcile] orphaned job {jid} failed on a read of its key", flush=True)
            return False
        if now - float(meta.get("started") or meta.get("created") or now) >= ORPHAN_MIN_AGE_S:
            seen = self._flight_seen_live
            if len(seen) > 4096:                   # an api container lives for days
                cut = time.monotonic() - self.FLIGHT_LIVE_TTL_S
                for k in [k for k, t in seen.items() if t < cut]:
                    seen.pop(k, None)
            seen[jid] = time.monotonic()           # probed, and live
        return True

    def cache_delete(self, key):
        from haversack.serve import ResultCache
        _reload_cache_view()
        # exclusive, like a reload: Modal reloads after a commit when its server asks
        # (it never hid a file in 61 measured commits, but nothing promises that)
        with _cache_view.exclusive():
            deleted = ResultCache(CACHE_ROOT, keep=RESULTS_KEEP).delete(key)
            if deleted:
                cache_vol.commit()
        return deleted

    def status_of(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None
        keys = ("id", "task", "state", "created", "started", "finished",
                "progress", "error", "input_identity", "cached",
                # the result handle and the options its URL form depends on -
                # serve's job route turns these into `key` + `links`
                "cache_key", "options",
                # the caller asked for fresh bytes and did not get them; the
                # local executor reports this, so this deployment must too
                "input_refresh_skipped",
                # the caller's pin beside the canonical task - the local executor reports
                # it, and a whitelist without it dropped it here (review 2026-09-12)
                "version",
                # what the job renders beside its labels, and what a cache hit could
                # not: serve's job route builds `links` from the two, and a whitelist
                # without them would advertise this deployment's whole set for a job
                # that declined it
                "deliverables", "deliverables_unavailable")
        d = {k: meta.get(k) for k in keys if meta.get(k) is not None}
        if meta.get("state") == "done" and meta.get("result") is not None:
            d["result"] = meta["result"]
        return d

    def statuses(self):
        out = []
        try:
            keys = list(jobs_dict.keys())
        except Exception:                    # keys() availability varies by client version
            return out
        for jid in keys:
            if ":" in str(jid):              # inflight:/artifacts:/cancel: markers
                continue
            try:
                s = self.status_of(jid)
            except Exception:                # one bad row must not truncate the listing
                continue
            if s:
                out.append({k: s.get(k) for k in
                            ("id", "task", "state", "created", "started", "finished")})
        return out

    def cancel(self, jid):
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, False
        state = meta.get("state")
        if state in ("queued", "running"):
            call_id = meta.get("call_id")
            if call_id:
                try:
                    modal.FunctionCall.from_id(call_id).cancel()
                except Exception:
                    pass
            jobs_dict[f"cancel:{jid}"] = time.time()
            _emit(jid, {"state": "cancelled", "finished": time.time()})
            return "cancelled", False
        import shutil
        try:
            del jobs_dict[jid]
        except Exception:
            pass
        shutil.rmtree(Path(SCRATCH_ROOT) / jid, ignore_errors=True)
        scratch_vol.commit()
        return state, True

    def result_file(self, jid):
        """(state, the job's own copy of its labels or None). serve's result route asks
        this only after the job's published cache entry, which is the copy this
        container can rely on: the scratch file is the worker's, and on 2026-09-19 the
        api container did not see it for 162 of 440 finished IDC jobs."""
        from haversack.serve import RESULT_NAME
        meta = jobs_dict.get(jid)
        if meta is None:
            return None, None
        if meta.get("cache_path"):
            p = Path(meta["cache_path"])
            _reload_cache_view()
            with _cache_view.shared():         # read, and copied out, clear of reloads
                if not p.exists():
                    return meta["state"], None
                g = _mirror(p.parent, Path(MIRROR_ROOT) / "_gen" / p.parent.name)
            return meta["state"], g / p.name
        # The worker's scratch file, read and copied out under the guard every scratch
        # reload takes: a reload in another thread hides the volume from this one (see
        # _cache_view), which is likely what the 162 missing files above were. A refused
        # reload (open files) leaves a view that may predate the worker's save: judge by
        # what is visible only once a reload has taken, and say "not visible yet" rather
        # than "gone" if none does.
        p = Path(SCRATCH_ROOT) / jid / RESULT_NAME
        local = Path(MIRROR_ROOT) / "_jobs" / jid / RESULT_NAME
        for delay in CACHE_CONFIRM_DELAYS_S:
            if delay:
                time.sleep(delay)
            with self.volume_guard:
                fresh = _reload_logged(scratch_vol, "scratch")
                if p.exists():
                    return meta["state"], _copy_local(p, local)
                if fresh:
                    return meta["state"], None
        from haversack.serve import ResultsNotVisible
        raise ResultsNotVisible("this server cannot see the job's own result yet (a "
                                "volume reload was refused); retry shortly")


@app.function(cpu=2.0, memory=2048, scaledown_window=300, image=api_image,
              volumes={SCRATCH_ROOT: scratch_vol, WEIGHTS_ROOT: weights_vol,
                       CACHE_ROOT: cache_vol, INPUTS_ROOT: inputs_vol})
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=PROXY_AUTH)
def api():
    _pkg_dir()
    os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
    from haversack import Segmenter
    from haversack.serve import create_app

    ex = ModalExecutor()
    # catalog/describe only - jobs run on the Worker; device string is cosmetic here
    ex.segmenter = Segmenter(device="cpu", weights=WEIGHTS_ROOT)
    return create_app(ex)


if PUBLIC:
    @app.function(cpu=1.0, memory=1024, scaledown_window=300, image=api_image,
                  volumes={CACHE_ROOT: cache_vol, WEIGHTS_ROOT: weights_vol})
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app(requires_proxy_auth=False)
    def public():
        """The anonymous read-only twin (HAVERSACK_PUBLIC=1): cache hits only, no
        compute path in the function at all - it cannot spend GPU by
        construction. Shares the cache volume with the authed api."""
        _pkg_dir()
        os.environ["TOTALSEG_WEIGHTS_PATH"] = WEIGHTS_ROOT
        from haversack import Segmenter
        from haversack.serve import (create_public_app, installed_versions,
                                 result_key, weights_versions_of)
        seg = Segmenter(device="cpu", weights=WEIGHTS_ROOT)

        def weights_fn(task):
            return weights_versions_of(seg, task)

        def key_fn(identity, task, opts=None):
            ids = (identity,) if isinstance(identity, str) else tuple(identity)
            return result_key(ids, task, opts or {}, weights_fn(task))

        def get(key):
            _reload_cache_view(max_age=CACHE_FRESH_S)
            return _read_cache(key)

        def inflight(key):
            jid = jobs_dict.get(f"inflight:{key}")
            if not jid:
                return None
            meta = jobs_dict.get(jid) or {}
            if meta.get("state") not in ("queued", "running"):
                return None
            return {"progress": meta.get("progress")}

        return create_public_app(key_fn, get, seg.tasks, inflight=inflight,
                                 list_fn=_list_cache, resolve_fn=seg.resolve_task,
                                 weights_fn=weights_fn,
                                 # a miss read from a view a refused reload left stale
                                 # is a 503, not a 404 (see _confirm_cache_absent)
                                 confirm_absent=_confirm_cache_absent,
                                 # so a pinned read can be answered, not refused outright
                                 versions_fn=lambda t: installed_versions(seg, t))
