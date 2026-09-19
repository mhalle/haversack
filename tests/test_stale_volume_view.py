"""A miss read from a view the api cannot trust is not an answer: 410 only for a verified purge.

Found on Modal 2026-09-19 (haversack-radar-val, 0.12.3, 300 ts.v2:total jobs on IDC, 16
concurrent clients): ``GET /v1/jobs/{id}/result?format=nii.gz`` answered 410 "purged" for
191 finished jobs whose entries the workers had committed; minutes later the same URLs, and
the same results by path, answered 200. Two causes, both in the api container:

- a reload in one thread hides the whole volume from the container's other threads, and
  every request reloaded - so lookups racing another request's reload missed (the api's
  /cache listing emptied and refilled every 0.5-3 s on haversack-visible-smoke);
- Modal refuses a reload while a file on the volume is open, which every streamed result
  held, so under downloads the view went stale - silently.

Two volume doubles here: one whose view lags (an entry committed "elsewhere" enters the
view only when a reload takes, after a set number of refusals), and one whose reload
hides the root from other threads while it runs.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack.serve import RESULT_NAME, create_app  # noqa: E402

from test_job_result_cache import _NrrdSeg, _Segmenter  # noqa: E402

U = "0be27d1c-9410-47ff-9c9f-a44b26a4bd55"
PATH = f"/v1/idc/{U}/total_fast/labels.seg.nrrd"
OPEN_FILES = "there are open files preventing the operation"


class _LaggingVolume:
    """The cache volume as the api container sees it: what the workers committed sits in
    ``hidden`` until a reload takes, and the first ``refusals`` reloads raise as Modal's
    does with files open."""

    def __init__(self, root: Path, refusals: int):
        self.root, self.hidden = Path(root), Path(str(root) + ".committed")
        self.refusals, self.reloads = refusals, 0

    def reload(self):
        self.reloads += 1
        if self.refusals > 0:
            self.refusals -= 1
            raise RuntimeError(OPEN_FILES)
        if self.hidden.exists():                   # the committed state comes into view
            if self.root.exists():
                import shutil
                shutil.rmtree(self.root)
            self.hidden.rename(self.root)

    def commit(self):
        pass


def _modal(monkeypatch, tmp_path, *, refusals: int):
    pytest.importorskip("modal")
    from haversack import modal_app as m
    jobs = {}
    monkeypatch.setattr(m, "jobs_dict", jobs)
    monkeypatch.setattr(m, "CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(m, "SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setattr(m, "MIRROR_ROOT", str(tmp_path / "mirror"))
    monkeypatch.setattr(m, "CACHE_CONFIRM_DELAYS_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(m, "_cache_view_as_of", float("-inf"))
    vol = _LaggingVolume(tmp_path / "cache", refusals)
    monkeypatch.setattr(m, "cache_vol", vol)
    monkeypatch.setattr(m, "scratch_vol",
                        types.SimpleNamespace(reload=lambda: None, commit=lambda: None))
    ex = m.ModalExecutor()
    ex.segmenter = _Segmenter(steps=1)
    monkeypatch.setattr(ex, "_fresh_weights_versions", lambda task: [])
    return m, jobs, vol, ex, TestClient(create_app(ex))


def _commit_elsewhere(m, vol, tmp_path, key, jid="87557074d052"):
    """A worker's publication: the entry exists in the committed state, not in this view."""
    from haversack.content import digest_file
    from haversack.serve import ResultCache
    src = tmp_path / "worker" / RESULT_NAME
    src.parent.mkdir(parents=True)
    _NrrdSeg().save(src)
    res = {"outputs": [{"name": "labels", "sha256": digest_file(src)}]}
    ResultCache(m.CACHE_ROOT).put(key, src, res, {"task": "total_fast", "job": jid})
    Path(m.CACHE_ROOT).rename(vol.hidden)
    return src, res


def _done(jobs, jid, key, res):
    jobs[jid] = {"id": jid, "task": "total_fast", "state": "done", "cache_key": key,
                 "result": res}


def _key(ex):
    return ex.resource_key(f"idc:{U}", "total_fast", {})


# -- the job result route ------------------------------------------------------------

@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_a_done_job_is_read_once_the_view_catches_up(monkeypatch, tmp_path, fmt):
    """cache_get's reload is refused; the confirming reload takes and finds the entry.
    On 0.12.3 this was the 410."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    src, res = _commit_elsewhere(m, vol, tmp_path, "k" * 64)
    _done(jobs, "87557074d052", "k" * 64, res)
    r = client.get("/v1/jobs/87557074d052/result" + (f"?format={fmt}" if fmt else ""))
    assert r.status_code == 200, r.text
    if fmt is None:
        assert r.content == src.read_bytes()
    assert vol.reloads == 2


@pytest.mark.parametrize("fmt", [None, "nii.gz"])
def test_a_done_job_that_cannot_be_seen_is_503_not_410(monkeypatch, tmp_path, fmt):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=10 ** 6)
    _, res = _commit_elsewhere(m, vol, tmp_path, "k" * 64)
    _done(jobs, "1fb1b9aa07d8", "k" * 64, res)
    r = client.get("/v1/jobs/1fb1b9aa07d8/result" + (f"?format={fmt}" if fmt else ""))
    assert r.status_code == 503, r.text
    assert r.headers["retry-after"] == "5"
    assert r.json()["detail"]["code"] == "not_visible_yet"
    # bounded: cache_get's reload, then one per confirming attempt - and no more
    assert vol.reloads == 1 + len(m.CACHE_CONFIRM_DELAYS_S)
    vol.refusals = 0                               # the open files close: it is there
    assert client.get("/v1/jobs/1fb1b9aa07d8/result").status_code == 200


def test_a_verified_purge_is_still_410(monkeypatch, tmp_path):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    _done(jobs, "dcc9dc034570", "k" * 64,
          {"outputs": [{"name": "labels", "sha256": "sha256:0"}]})
    assert client.get("/v1/jobs/dcc9dc034570/result").status_code == 410


def test_a_reload_since_the_question_is_not_repeated(monkeypatch, tmp_path):
    """A miss read right after a reload that took is already verified: no second
    reload per miss on the hot path."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    _done(jobs, "dcc9dc034570", "k" * 64,
          {"outputs": [{"name": "labels", "sha256": "sha256:0"}]})
    assert client.get("/v1/jobs/dcc9dc034570/result").status_code == 410
    assert vol.reloads == 1


def test_from_job_promotion_says_503_not_gone(monkeypatch, tmp_path):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=10 ** 6)
    _, res = _commit_elsewhere(m, vol, tmp_path, "k" * 64)
    _done(jobs, "87557074d052", "k" * 64, res)
    monkeypatch.setattr(type(ex), "content", types.SimpleNamespace())
    r = client.post("/v1/inputs", data={"from_job": "87557074d052"})
    assert r.status_code == 503, r.text


# -- the path surface ----------------------------------------------------------------

def test_path_get_reads_an_entry_the_stale_view_missed(monkeypatch, tmp_path):
    """On 0.12.3 a plain GET said 404 "not materialized" - and with Prefer: wait it
    would have computed the result again."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    src, _ = _commit_elsewhere(m, vol, tmp_path, _key(ex))
    r = client.get(PATH)
    assert r.status_code == 200, r.text
    assert r.content == src.read_bytes()


def test_path_get_that_cannot_see_is_503_and_computes_nothing(monkeypatch, tmp_path):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=10 ** 6)
    _commit_elsewhere(m, vol, tmp_path, _key(ex))
    monkeypatch.setattr(ex, "submit", lambda *a, **k: pytest.fail("a duplicate compute"))
    for headers in ({}, {"Prefer": "wait=0"}):
        r = client.get(PATH, headers=headers)
        assert r.status_code == 503, r.text
        assert r.headers["retry-after"]
    assert client.head(PATH).status_code == 503
    assert client.get(PATH.replace("labels.seg.nrrd", "meta.json")).status_code == 503


def test_path_head_and_meta_see_it_once_a_reload_takes(monkeypatch, tmp_path):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    _commit_elsewhere(m, vol, tmp_path, _key(ex))
    assert client.head(PATH).status_code == 200
    vol.refusals = 1
    r = client.get(PATH.replace("labels.seg.nrrd", "meta.json"))
    assert r.status_code == 200, r.text


def test_a_verified_miss_by_path_is_still_404(monkeypatch, tmp_path):
    """Refreshing the view must not turn every uncomputed resource into a 503."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=1)
    assert client.get(PATH).status_code == 404
    assert client.head(PATH).status_code == 404


def test_the_public_twin_confirms_its_misses_too(tmp_path):
    from haversack.serve import ResultCache, ResultsNotVisible, create_public_app, result_key
    key_fn = lambda identity, task, opts=None: result_key((identity,), task, opts or {}, [])
    cache = ResultCache(tmp_path / "c")
    src = tmp_path / RESULT_NAME
    _NrrdSeg().save(src)
    committed = ResultCache(tmp_path / "committed")
    committed.put(key_fn(f"idc:{U}", "total_fast"), src, {"outputs": []}, {})

    def stale(key, since):
        raise ResultsNotVisible("reload refused")

    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                                          confirm_absent=stale))
    assert client.get(PATH).status_code == 503
    client = TestClient(create_public_app(key_fn, cache.get, lambda: ["total_fast"],
                                          confirm_absent=lambda k, s: committed.get(k)))
    assert client.get(PATH).status_code == 200


# -- a reload in one thread hides the volume from the others ------------------------

class _HidingVolume:
    """Modal's reload as measured on 2026-09-19: while it runs, every path on the volume
    is ENOENT to the container's OTHER threads (734,714 of 735,326 listings, never with
    no reload running). Here the root is renamed away for the reload's duration."""

    def __init__(self, root: Path):
        import threading
        self.root, self.away = Path(root), Path(str(root) + ".reloading")
        self.reloads = 0
        self._one = threading.Lock()           # Modal serializes a process's reloads

    def reload(self):
        import time
        with self._one:
            self.reloads += 1
            self.root.rename(self.away)
            time.sleep(0.002)
            self.away.rename(self.root)

    def commit(self):
        self.reload()                          # the worst case: Modal may reload after one


def test_lookups_racing_reloads_never_miss_a_published_entry(monkeypatch, tmp_path):
    """Every api request reloaded in its own thread, so a lookup racing another request's
    reload missed - and a miss read after a reload is believed: 410 for a finished job.
    Pre-fix, some of these lookups return None."""
    import threading
    m, jobs, _, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    from haversack.serve import ResultCache
    src = tmp_path / "worker" / RESULT_NAME
    src.parent.mkdir(parents=True)
    _NrrdSeg().save(src)
    ResultCache(m.CACHE_ROOT).put("k" * 64, src, {"outputs": []}, {"task": "total_fast"})
    monkeypatch.setattr(m, "cache_vol", _HidingVolume(tmp_path / "cache"))
    misses, errors = [], []

    def ask():
        for _ in range(60):
            try:
                if ex.cache_get("k" * 64) is None:
                    misses.append(1)
            except Exception as e:             # noqa: BLE001 - a crash is a failure too
                errors.append(e)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]
    assert not misses, f"{len(misses)} of 480 lookups missed a published entry"


def test_the_served_copy_outlives_the_volume_view(monkeypatch, tmp_path):
    """A route opens the file after the lookup returns (FileResponse opens at send): what
    it opens is a container-local copy, not the volume's file, which a reload hides."""
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    src, res = _commit_elsewhere(m, vol, tmp_path, _key(ex))
    hit = ex.cache_get(_key(ex))
    assert hit is not None and not str(hit[0]).startswith(m.CACHE_ROOT)
    import shutil
    shutil.rmtree(m.CACHE_ROOT)                 # the volume's view, gone mid-request
    assert Path(hit[0]).read_bytes() == src.read_bytes()


def test_an_artifact_placed_after_the_copy_is_still_found(monkeypatch, tmp_path):
    m, jobs, vol, ex, client = _modal(monkeypatch, tmp_path, refusals=0)
    src, res = _commit_elsewhere(m, vol, tmp_path, _key(ex))
    assert ex.cache_get(_key(ex)) is not None          # copied before the artifact exists
    from haversack.serve import ResultCache
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert ResultCache(m.CACHE_ROOT).add_artifact(_key(ex), "preview.png", png)
    r = client.get(PATH.replace("labels.seg.nrrd", "preview.png"))
    assert r.status_code == 200, r.text
    assert r.content == png.read_bytes()
