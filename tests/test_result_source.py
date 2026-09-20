"""`result:<key>` - a result this server computed, as the input of another job - and the
label-map role that consumes one. docs/result-references.md is the design note.

End to end through the REST surface, with two doubles standing in for engines:

* a PRODUCER (`total_fast`) that returns a real :class:`haversack.result.Segmentation`, so
  what lands in the result cache is a real ``.seg.nrrd`` written by the real writer - names,
  label values and provenance in its header;
* a CONSUMER (`organ_means`) that declares roles ``image`` and ``mask``, reads the mask with
  :func:`haversack.labelmap.read_label_map` and reports each structure's mean intensity BY
  NAME. No shipped task takes (image, mask); the feature is exercised without adding one.

The guards here reconcile two independent sources each - a key minted by the real
``result_key`` against the grammar, a digest taken by the upload door against the one the
reference resolves to, the header the real writer writes against what the reader reads -
because a check that compares a thing to itself always passes (AGENTS.md, 2026-09-08).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
sitk = pytest.importorskip("SimpleITK")
from fastapi.testclient import TestClient  # noqa: E402

from haversack import content, jobpolicy, serve  # noqa: E402
from haversack.errors import InputError, RequestError, UnresolvedReference  # noqa: E402
from haversack.labelmap import LABELS_KIND, read_label_map  # noqa: E402
from haversack.serve import LocalExecutor, create_app  # noqa: E402
from haversack.sources import (OUTPUT_NAME_RE, RESULT_KEY_RE, RESULT_PIN_RE,  # noqa: E402
                               DataSource, ResultRef, ResultSource, check_identifier)

from test_serve import FakeSegmenter, wait_state  # noqa: E402

SHAPE = (4, 8, 8)                                   # Z, Y, X
NAMES = {1: "liver", 2: "spleen", 3: "kidney_right"}   # 3 is declared and absent


def _ct() -> np.ndarray:
    """A volume whose every voxel differs, so a mean over the wrong voxels is a wrong number."""
    return np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE)


def _labels(variant: int = 0) -> np.ndarray:
    a = np.zeros(SHAPE, np.uint8)
    a[:2] = 1                                       # liver
    a[2:, :, :4] = 2                                # spleen
    if variant:
        a[3, :, 4:] = 1                             # a recompute that came out differently
    return a


def _image(array: np.ndarray):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing((0.8, 0.8, 2.5))
    img.SetOrigin((-10.0, 5.0, 40.0))
    return img


def _ct_bytes(tmp: Path) -> bytes:
    f = tmp / "ct.nii.gz"
    sitk.WriteImage(_image(_ct()), str(f), True)
    return f.read_bytes()


class ToyCT(DataSource):
    """A repository of one CT, which says where it came from and under what terms - so
    the tests can watch the ORIGINAL data's license and citation survive a hop."""
    prefix = "toy"
    id_pattern = r"sp[0-9]{3}"
    description = "test specimens"
    LICENSE = {"name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"}
    CITE = [{"text": "Specimen Atlas (2026)", "doi": "10.0000/specimens", "for": "dataset"}]

    def __init__(self, tmp: Path):
        self._bytes = _ct_bytes(tmp)

    def fetch(self, identifier, dest_dir, *, credentials=None):
        d = Path(dest_dir) / "series"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ct.nii.gz").write_bytes(self._bytes)
        return d

    def describe_input(self, identifier, fetched=None, credentials=None):
        return {"origin": {"collection": "specimens", "uid": identifier,
                           "determined_by": "the toy repository"},
                "license": dict(self.LICENSE), "cite": [dict(c) for c in self.CITE]}


class Tasks(FakeSegmenter):
    """The producer and the two consumers, behind one catalog."""

    def __init__(self):
        super().__init__(steps=1)
        self.variant = 0
        self.seen = []                              # what each consumer was handed

    def tasks(self):
        return ["total_fast", "organ_means", "relabel"]

    def describe(self, task):
        from haversack.schemas import input_specs, label_input
        if task not in self.tasks():
            raise KeyError(task)
        d = {"name": task, "engine": "nnunetv2"}
        if task == "organ_means":
            d["inputs"] = input_specs(["image"]) + [label_input("mask")]
        elif task == "relabel":
            d["inputs"] = [label_input("mask")]
        return d

    def segment(self, image, task, *, progress=None, cancel=None, **options):
        super().segment(image, task, progress=progress, cancel=cancel, **options)
        return getattr(self, "_" + task)(image)

    @staticmethod
    def _seg(array, names, prov):
        from haversack.grid import Grid
        from haversack.result import Segmentation
        from haversack.values import LabelSchema
        return Segmentation(labels=_image(array), schema=LabelSchema(names=dict(names)),
                            grid=Grid(SHAPE, (2.5, 0.8, 0.8)), spec=None,
                            provenance=dict(prov))

    def _total_fast(self, image):
        return self._seg(_labels(self.variant), NAMES, {
            "task": "total_fast", "device": "fake",
            "models": [{"weights": "Dataset297", "version": "v2.0.4"}],
            "attribution": {"ecosystem": "ts.v2", "engine": "nnunetv2",
                            "license": {"weights": "Apache-2.0"},
                            "cite": [{"doi": "10.1148/ryai.230024"}]}})

    def _organ_means(self, image):
        from haversack import io as nio
        assert isinstance(image, dict) and set(image) == {"image", "mask"}, image
        ct = nio.read_image(image["image"])
        lm = read_label_map(image["mask"])
        voxels = sitk.GetArrayFromImage(ct)
        self.seen.append(lm)
        means = {name: float(voxels[lm.mask(name)].mean()) for name in sorted(lm.names.values())}
        # with a tolerance, as any consumer must: the CT came through NIfTI, whose header
        # is float32 (0.8 mm reads 0.800000012), and the mask through NRRD, which is double
        a, b = lm.geometry, nio.geometry_of(ct)
        same = a.shape_zyx == b.shape_zyx and all(
            np.allclose(getattr(a, f), getattr(b, f), atol=1e-5)
            for f in ("spacing_zyx", "origin_xyz", "direction_xyz"))
        return self._seg(lm.array, lm.names, {
            "task": "organ_means", "mean_intensity": means, "mask_task": lm.task,
            "same_grid": bool(same)})

    def _relabel(self, image):
        lm = read_label_map(image)                  # a single input arrives as a path
        self.seen.append(lm)
        keep = (lm.array == lm.label_of("liver")).astype(np.uint8)
        return self._seg(keep, {1: "liver"}, {"task": "relabel", "mask_task": lm.task})


@pytest.fixture
def server(tmp_path, monkeypatch):
    from dataclasses import replace

    from haversack.engines import registry
    monkeypatch.setitem(registry.ENGINES, "nnunetv2",
                        replace(registry.ENGINES["nnunetv2"], multi_input=True))
    seg = Tasks()
    ex = LocalExecutor(seg, workdir=tmp_path / "w", cache_dir=tmp_path / "rc",
                       sources=[ToyCT(tmp_path)], artifacts=())
    yield seg, ex, TestClient(create_app(ex))
    ex.close()


def _post(client, task, source, *, files=None, headers=None):
    return client.post("/v1/jobs", data={"task": task, "source": json.dumps(source)},
                       files=files, headers=headers or {})


def _done(client, r):
    assert r.status_code == 202, r.text
    s = wait_state(client, r.json()["id"])
    assert s["state"] == "done", s.get("error")
    return s


def _upstream(client, ident="sp042", headers=None):
    return _done(client, _post(client, "total_fast", [{"kind": "toy", "id": ident}],
                               headers=headers))


def _consume(client, key, *, image="sp042", headers=None):
    return _post(client, "organ_means",
                 [{"kind": "result", "id": key, "role": "mask"},      # listed mask-first:
                  {"kind": "toy", "id": image, "role": "image"}],     # binding is by name
                 headers=headers)


def _digest_of(status) -> str:
    return status["result"]["outputs"][0]["sha256"]


# -- the feature, end to end ------------------------------------------------------------

def test_one_jobs_labels_are_another_jobs_mask_read_by_structure_name(server):
    seg, ex, client = server
    up = _upstream(client)
    down = _done(client, _consume(client, up["key"]))

    ct, lab = _ct(), _labels()
    want = {"liver": float(ct[lab == 1].mean()), "spleen": float(ct[lab == 2].mean())}
    got = down["result"]["provenance"]
    assert got["mean_intensity"] == want            # absent kidney_right: not a segment
    assert got["mask_task"] == "total_fast" and got["same_grid"] is True

    lm = seg.seen[0]
    assert lm.names == {1: "liver", 2: "spleen"}
    assert lm.task == "total_fast" and lm.provenance["device"] == "fake"
    assert lm.geometry.spacing_zyx == (2.5, 0.8, 0.8)
    assert lm.geometry.origin_xyz == (-10.0, 5.0, 40.0)
    assert np.array_equal(lm.array, lab)


def test_the_identity_is_the_referenced_outputs_digest_not_the_key(server):
    _, _, client = server
    up = _upstream(client)
    down = _done(client, _consume(client, up["key"]))
    assert down["input_identity"] == sorted(["image=toy:sp042", f"mask={_digest_of(up)}"])
    assert up["key"] not in json.dumps(down["input_identity"])
    # ...and the bare form, the named form and the pinned form are one request
    for ident in (f"{up['key']}!labels", f"{up['key']}@{_digest_of(up)}",
                  f"{up['key']}!labels@{_digest_of(up)}"):
        again = _done(client, _consume(client, ident))
        assert again["key"] == down["key"] and again.get("cached") is True, ident


def test_referring_to_a_result_and_uploading_its_bytes_are_one_request(server, tmp_path):
    """The digest-as-identity rule, reconciled with a door that knows nothing of it: the
    upload route hashes the bytes as they stream in. Same bytes, same identity, same key -
    so the second ask is a hit on the first's entry, whichever door came first."""
    _, _, client = server
    up = _upstream(client)
    labels = client.get(f"/v1/jobs/{up['id']}/result").content
    assert f"sha256:{hashlib.sha256(labels).hexdigest()}" == _digest_of(up)
    sent = _done(client, _post(
        client, "organ_means",
        [{"kind": "toy", "id": "sp042", "role": "image"}, {"kind": "upload", "role": "mask"}],
        files={"mask": ("labels.seg.nrrd", labels)}))
    referred = _done(client, _consume(client, up["key"]))
    assert referred["input_identity"] == sent["input_identity"]
    assert referred["key"] == sent["key"] and referred.get("cached") is True


def test_an_uploaded_image_and_a_referenced_mask(server, tmp_path):
    """Upload a CT, refer to its segmentation: the commonest request there is, and the one
    shape the first tests never sent - an UPLOAD bound before the reference. The submit
    path's upload branch has a local named `declared`, and the role specs rode in under
    the same name: after the image's upload the mask's role read as an image, and every
    such job was refused `wrong_input_kind`. Found by the Modal smoke, 2026-09-20."""
    seg, _, client = server
    up = _upstream(client)
    for order in ([0, 1], [1, 0]):                  # the source list's order must not matter
        source = [{"kind": "upload", "role": "image"},
                  {"kind": "result", "id": up["key"], "role": "mask"}]
        r = _post(client, "organ_means", [source[i] for i in order],
                  files={"image": ("ct.nii.gz", _ct_bytes(tmp_path))})
        s = _done(client, r)
        ct, lab = _ct(), _labels()
        assert s["result"]["provenance"]["mean_intensity"]["spleen"] == float(ct[lab == 2].mean())
        assert [i for i in s["input_identity"] if i.startswith("mask=")] == [f"mask={_digest_of(up)}"]


def test_a_single_input_task_can_take_a_label_map(server):
    seg, _, client = server
    up = _upstream(client)
    s = _done(client, _post(client, "relabel", [{"kind": "result", "id": up["key"]}]))
    assert s["input_identity"] == [_digest_of(up)]
    assert s["result"]["provenance"]["mask_task"] == "total_fast"
    assert list(s["result"]["names"].values()) == ["liver"]


# -- refusals, at submit ----------------------------------------------------------------

def _jobs(client) -> int:
    return len(client.get("/v1/jobs").json()["jobs"])


def test_a_result_that_is_not_there_is_refused_at_submit_with_the_fix(server, tmp_path):
    seg, ex, client = server
    missing = "0" * 64
    before, dirs = _jobs(client), set((tmp_path / "w").iterdir())
    r = _consume(client, missing)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "result_missing" and detail["key"] == missing
    assert missing in detail["message"] and "run the job that produces it first" in detail["message"]
    assert "\n" not in detail["message"]
    # never a queued job that fails later in a worker - and nothing left behind
    assert _jobs(client) == before and set((tmp_path / "w").iterdir()) == dirs
    assert [c[1] for c in seg.calls] == []


def test_an_evicted_result_is_refused_the_same_way(server):
    _, ex, client = server
    up = _upstream(client)
    assert ex.cache_delete(up["key"])
    r = _consume(client, up["key"])
    assert r.status_code == 409 and r.json()["detail"]["code"] == "result_missing"


def test_an_unknown_output_name_is_refused_naming_the_ones_there_are(server):
    _, _, client = server
    up = _upstream(client)
    r = _consume(client, f"{up['key']}!vectors")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_output" and detail["outputs"] == ["labels"]
    assert f"result:{up['key']}!labels" in detail["message"] and "\n" not in detail["message"]


def test_a_pin_on_other_bytes_is_refused_not_followed(server):
    _, _, client = server
    up = _upstream(client)
    other = "sha256:" + "1" * 64
    r = _consume(client, f"{up['key']}@{other}")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "result_changed"
    assert detail["expected"] == other and detail["actual"] == _digest_of(up)


def test_a_label_map_bound_to_an_image_role_is_refused(server):
    """`total_fast` takes an image. Handed its own labels it would segment a label map as
    though it were a CT and cache the result."""
    seg, _, client = server
    up = _upstream(client)
    n = len(seg.calls)
    r = _post(client, "total_fast", [{"kind": "result", "id": up["key"]}])
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "wrong_input_kind"
    assert detail["output_kind"] == LABELS_KIND and detail["role_kind"] == "image"
    assert len(seg.calls) == n


@pytest.mark.parametrize("ident, says", [
    ("0123456789ab", "looks like a job id"),
    ("https://other.example/v1/x", "another server"),
    ("A" * 64, "not a valid result identifier"),
    ("0" * 64 + "!Labels", "not a valid result identifier"),
    ("0" * 64 + "@sha256-tree:" + "0" * 64, "not a valid result identifier"),
])
def test_a_malformed_reference_is_a_422_that_says_what_was_wrong(server, ident, says):
    _, _, client = server
    r = _consume(client, ident)
    assert r.status_code == 422, r.text
    assert says in str(r.json()["detail"])


def test_a_reference_takes_no_credentials(server):
    _, _, client = server
    up = _upstream(client)
    r = _consume(client, up["key"], headers={"Haversack-Source-Token": "result=secret"})
    assert r.status_code == 422 and "no credentials" in str(r.json()["detail"])


def test_a_server_without_a_result_cache_does_not_offer_the_kind(tmp_path):
    seg = Tasks()
    ex = LocalExecutor(seg, workdir=tmp_path / "w", sources=[ToyCT(tmp_path)], artifacts=())
    try:
        client = TestClient(create_app(ex))
        assert "result" not in {s["prefix"] for s in client.get("/v1/sources").json()["sources"]}
        r = _post(client, "relabel", [{"kind": "result", "id": "0" * 64}])
        assert r.status_code == 422 and "no result cache" in str(r.json()["detail"])
    finally:
        ex.close()


# -- no-cache, republication, and the window between submit and the worker --------------

def _watch(ex) -> list:
    """Every question the source asks of the result cache, as ``(key, fresh)``."""
    src, asked = ex.sources["result"], []
    real = src._reader

    def reader(key, *, fresh=False):
        asked.append((key, fresh))
        return real(key, fresh=fresh)
    src._reader = reader
    return asked


def test_a_republished_result_is_a_new_downstream_request(server):
    """Why the identity is not the key: `no-cache` republishes other bytes under the SAME
    key, and a downstream result keyed on it would outlive the mask it was computed from."""
    seg, _, client = server
    up = _upstream(client)
    first = _done(client, _consume(client, up["key"]))
    seg.variant = 1
    again = _upstream(client, headers={"Cache-Control": "no-cache"})
    assert again["key"] == up["key"] and _digest_of(again) != _digest_of(up)

    second = _done(client, _consume(client, up["key"]))
    assert second["key"] != first["key"] and not second.get("cached")
    ct, lab = _ct(), _labels(1)
    assert second["result"]["provenance"]["mean_intensity"]["liver"] == float(ct[lab == 1].mean())
    # the old pin is refused by name rather than served from the entry it once keyed
    r = _consume(client, f"{up['key']}@{_digest_of(up)}")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "result_changed"


def test_no_cache_downstream_resolves_again_and_never_recomputes_upstream(server):
    seg, ex, client = server
    up = _upstream(client)
    first = _done(client, _consume(client, up["key"]))
    asked = _watch(ex)
    before = [c[1] for c in seg.calls]

    forced = _done(client, _consume(client, up["key"], headers={"Cache-Control": "no-cache"}))
    ran = [c[1] for c in seg.calls][len(before):]
    assert ran == ["organ_means"], ran              # the consumer again; the producer never
    assert forced["key"] == first["key"] and not forced.get("cached")
    assert not forced.get("input_refresh_skipped")
    # asked at submit, and again by the fetch that replaced the discarded copy
    assert [k for k, _ in asked] == [up["key"], up["key"]], asked


def _queue_behind_a_running_job(seg, client, key):
    """A consumer job submitted and still QUEUED: the dispatcher is held on another job."""
    seg.gate = threading.Event()
    blocker = _post(client, "total_fast", [{"kind": "toy", "id": "sp777"}])
    assert blocker.status_code == 202
    r = _consume(client, key)
    assert r.status_code == 202, r.text
    assert wait_state(client, r.json()["id"], ("queued",))["state"] == "queued"
    return r.json()["id"]


def _republish(ex, key, tmp_path, array):
    path = tmp_path / "republished.seg.nrrd"
    Tasks._seg(array, NAMES, {"task": "total_fast"}).save(path)
    result = {"outputs": [{"name": "labels", "sha256": content.digest_file(path),
                           "bytes": path.stat().st_size}]}
    ex.cache.put(key, path, result, {"task": "total_fast"})
    return result["outputs"][0]["sha256"]


def test_a_result_that_changes_before_the_worker_reads_it_fails_the_job_by_name(
        server, tmp_path):
    """On Modal a lease taken at submit does not reach a worker's pruning, so the worker
    asks AGAIN and compares with the digest the job was keyed on."""
    seg, ex, client = server
    up = _upstream(client)
    jid = _queue_behind_a_running_job(seg, client, up["key"])
    now = _republish(ex, up["key"], tmp_path, _labels(1))
    seg.gate.set()
    s = wait_state(client, jid)
    assert s["state"] == "failed", s
    assert "the referenced result changed" in s["error"] and now in s["error"]
    assert "organ_means" not in [c[1] for c in seg.calls]     # never computed from other bytes


def test_a_result_evicted_before_the_worker_reads_it_fails_the_job_by_name(server):
    seg, ex, client = server
    up = _upstream(client)
    jid = _queue_behind_a_running_job(seg, client, up["key"])
    assert ex.cache_delete(up["key"])
    seg.gate.set()
    s = wait_state(client, jid)
    assert s["state"] == "failed" and f"no result {up['key']}" in s["error"], s
    assert "organ_means" not in [c[1] for c in seg.calls]


def test_resolving_a_reference_takes_a_lease_on_the_entry(server):
    """Through the LEASED lookup, as the path surface and the job result route read an
    entry: reclamation must know a reader holds the generation a fetch is about to copy."""
    _, ex, client = server
    up = _upstream(client)
    lease = Path(ex.cache_get(up["key"])[0]).parent / ex.cache.LEASE_NAME
    lease.unlink()
    ex.sources["result"].pin(up["key"])
    assert lease.exists()


# -- the source itself: one resolution, a pinned identifier, verified bytes --------------

def _entry(tmp_path, body=b"labels", *, stated=None, name="labels"):
    """A cache entry as a reader hands it over: ``(path, result.json)``."""
    path = tmp_path / "g" / serve.RESULT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    (path.parent / "meta.json").write_text(json.dumps({"task": "ts.v2:total"}), encoding="utf-8")
    return path, {"outputs": [{"name": name, "bytes": len(body),
                               "sha256": stated or content.digest_file(path)}]}


def _source(hit, asked=None):
    import contextlib

    def reader(key, *, fresh=False):
        if asked is not None:
            asked.append(fresh)
        return contextlib.nullcontext(hit(fresh) if callable(hit) else hit)
    return ResultSource(reader)


KEY = "c" * 64


def test_pin_writes_the_whole_answer_into_the_identifier(tmp_path):
    path, result = _entry(tmp_path)
    src = _source((path, result))
    pinned = src.pin(KEY)
    assert pinned == f"{KEY}!labels@{content.digest_file(path)}"
    assert src.pin(pinned) == pinned and src.pin(f"{KEY}!labels") == pinned
    assert src.identity(pinned) == content.digest_file(path)
    check_identifier(src, pinned)                   # one grammar: the pinned form is valid


def test_identity_does_no_lookup_and_refuses_to_guess(tmp_path):
    asked = []
    src = _source(_entry(tmp_path), asked)
    with pytest.raises(InputError, match="not pinned"):
        src.identity(KEY)
    assert src.identity(f"{KEY}@sha256:{'9' * 64}") == f"sha256:{'9' * 64}"
    assert asked == []
    with pytest.raises(InputError, match="not pinned"):
        src.fetch(KEY, tmp_path)
    assert asked == []


def test_a_stale_view_is_asked_again_before_it_is_believed(tmp_path):
    """A view that is behind can answer "missing" or "other bytes" wrongly, never "these
    bytes" wrongly - the digest decides - so only those two answers cost a fresh look."""
    hit = _entry(tmp_path)
    asked = []
    assert _source(hit, asked).pin(KEY).startswith(KEY) and asked == [False]

    asked.clear()
    late = _source(lambda fresh: hit if fresh else None, asked)
    assert late.pin(KEY).startswith(KEY) and asked == [False, True]

    asked.clear()
    old = _entry(tmp_path / "old", b"the previous generation")
    moved_on = _source(lambda fresh: hit if fresh else old, asked)
    pinned = f"{KEY}!labels@{hit[1]['outputs'][0]['sha256']}"
    assert moved_on.pin(pinned) == pinned and asked == [False, True]

    asked.clear()
    with pytest.raises(UnresolvedReference) as e:
        _source(None, asked).pin(KEY)
    assert e.value.code == "result_missing" and e.value.status == 409 and asked == [False, True]


def test_fetch_copies_verifies_and_records_what_the_result_said(tmp_path):
    path, result = _entry(tmp_path, b"the labels")
    result["provenance"] = {"task": "ignored: meta.json is what the server keyed on",
                            "models": [{"weights": "Dataset291", "version": "v2.0.1"}]}
    src = _source((path, result))
    pinned = src.pin(KEY)
    (tmp_path / "entry").mkdir()
    got = src.fetch(pinned, tmp_path / "entry")
    assert [p.name for p in got.iterdir()] == [serve.RESULT_NAME]
    assert (got / serve.RESULT_NAME).read_bytes() == b"the labels"
    said = src.describe_input(pinned, got)
    assert said["origin"]["result"] == KEY and said["origin"]["output"] == "labels"
    assert said["origin"]["task"] == "ts.v2:total"
    assert said["origin"]["weights"] == ["Dataset291=v2.0.1"]
    assert said["license"] is None and said["derived_from"] == {"results": [], "inputs": []}


def test_a_copy_that_is_not_the_pinned_bytes_is_never_handed_over(tmp_path):
    """A generation replaced mid-copy, or a view torn by a reload: result.json still states
    the digest the job was keyed on, and the bytes that arrived are other bytes."""
    path, result = _entry(tmp_path, b"what is really there", stated="sha256:" + "5" * 64)
    src = _source((path, result))
    (tmp_path / "entry").mkdir()
    with pytest.raises(InputError, match="the bytes read are sha256:"):
        src.fetch(f"{KEY}!labels@sha256:{'5' * 64}", tmp_path / "entry")
    assert list((tmp_path / "entry" / "series").iterdir()) == []


def test_a_copy_cut_short_leaves_nothing_in_the_entry(tmp_path, monkeypatch):
    """The volume vanished mid-copy (a reload in another thread, were the lock not held).
    The entry directory outlives a failed fetch - the next one claims inside it - so a
    partial file left there would sit beside the good copy for the entry's whole life."""
    import shutil as _shutil
    path, result = _entry(tmp_path)
    src = _source((path, result))
    pinned = src.pin(KEY)
    (tmp_path / "entry").mkdir()

    def cut_short(a, b, **k):
        Path(b).write_bytes(b"half")
        raise FileNotFoundError(a)
    monkeypatch.setattr(_shutil, "copyfile", cut_short)
    with pytest.raises(FileNotFoundError):
        src.fetch(pinned, tmp_path / "entry")
    assert list((tmp_path / "entry" / "series").iterdir()) == []
    monkeypatch.undo()
    got = src.fetch(pinned, tmp_path / "entry")    # and the retry is whole
    assert [p.name for p in got.iterdir()] == [serve.RESULT_NAME]


def test_the_command_line_says_where_a_reference_belongs(tmp_path):
    from haversack.sources import materialize
    with pytest.raises(InputError, match="names a result on a haversack server.*remote submit"):
        materialize("result:" + "0" * 64, cache_dir=tmp_path)


def test_an_entry_with_no_outputs_is_refused_with_the_fix(tmp_path):
    path, _ = _entry(tmp_path)
    with pytest.raises(UnresolvedReference) as e:
        _source((path, {})).pin(KEY)
    assert e.value.code == "result_unreadable" and "no-cache" in str(e.value)


def test_a_second_output_is_named_and_refused_until_the_cache_can_hand_one_over(tmp_path):
    """The grammar selects any listed output; the cache hands back the primary one's file
    and no other (non-label outputs are step 2 of the design note)."""
    path, result = _entry(tmp_path)
    result["outputs"].append({"name": "vectors", "sha256": "sha256:" + "7" * 64, "bytes": 1})
    src = _source((path, result))
    assert src.pin(KEY).startswith(f"{KEY}!labels@")
    with pytest.raises(RequestError) as e:
        src.pin(f"{KEY}!vectors")
    assert e.value.code == "unsupported_output" and e.value.status == 422


def test_forget_drops_nothing_because_nothing_is_remembered(tmp_path):
    asked = []
    src = _source(_entry(tmp_path), asked)
    src.pin(KEY)
    src.forget(KEY)
    src.pin(KEY)
    assert asked == [False, False] and not [k for k in vars(src) if k != "_reader"]


# -- provenance: the hop by reference, the original data by value -----------------------

def _mask_record(status):
    (rec,) = [r for r in status["result"]["provenance"]["inputs"] if r.get("kind") == "result"]
    return rec


def test_the_original_datas_license_and_citation_survive_the_hop(server):
    _, _, client = server
    up = _upstream(client)
    down = _done(client, _consume(client, up["key"]))
    rec = _mask_record(down)
    assert rec["role"] == "mask" and rec["identity"] == _digest_of(up)
    assert rec["content"]["digest"] == _digest_of(up) and rec["content"]["files"] == 1
    origin = rec["origin"]
    assert origin["result"] == up["key"] and origin["output"] == "labels"
    assert origin["task"] == "total_fast" and origin["weights"] == ["Dataset297=v2.0.4"]
    assert rec["attribution"]["cite"] == [{"doi": "10.1148/ryai.230024"}]
    assert rec["license"] is None                   # not one license: see describe_input
    (leaf,) = rec["derived_from"]["inputs"]
    assert leaf["identity"] == "toy:sp042" and leaf["license"] == ToyCT.LICENSE
    assert leaf["cite"] == ToyCT.CITE and leaf["origin"]["collection"] == "specimens"
    assert rec["derived_from"]["results"] == []


def test_a_chain_carries_hops_by_reference_and_each_original_input_once(server):
    """image and mask both descend from one CT, then a third job consumes the second's
    output: the record stays flat - no hop nests the records of the hop above it."""
    _, ex, client = server
    a = _upstream(client)
    b = _done(client, _consume(client, a["key"]))
    c = _done(client, _post(client, "relabel", [{"kind": "result", "id": b["key"]}]))
    rec = _mask_record(c)
    assert rec["origin"]["result"] == b["key"] and rec["origin"]["task"] == "organ_means"
    (hop,) = rec["derived_from"]["results"]
    assert hop == {"result": a["key"], "output": "labels", "digest": _digest_of(a),
                   "task": "total_fast", "weights": ["Dataset297=v2.0.4"]}
    (leaf,) = rec["derived_from"]["inputs"]         # once, though two paths lead to it
    assert leaf["identity"] == "toy:sp042" and leaf["license"] == ToyCT.LICENSE
    assert "role" not in leaf
    assert "derived_from" not in json.dumps(rec["derived_from"])
    # by value where it matters: the terms outlive both upstream entries
    assert ex.cache_delete(a["key"]) and ex.cache_delete(b["key"])
    kept = client.get(f"/v1/jobs/{c['id']}").json()
    assert _mask_record(kept)["derived_from"]["inputs"][0]["cite"] == ToyCT.CITE


# -- not path-addressable ---------------------------------------------------------------

def test_a_result_reference_gets_no_path_surface_and_no_path_links(server):
    _, _, client = server
    up = _upstream(client)
    s = _done(client, _post(client, "relabel", [{"kind": "result", "id": up["key"]}]))
    assert set(s["links"]) == {"self", "events", "result"}, s["links"]
    assert not [r.path for r in client.app.routes if r.path.startswith("/v1/result")]
    assert [r.path for r in client.app.routes if r.path.startswith("/v1/toy/")]
    listed = {x["prefix"]: x for x in client.get("/v1/sources").json()["sources"]}
    assert listed["result"]["path_addressable"] is False and listed["result"]["enabled"]
    assert listed["toy"]["path_addressable"] is True
    for e in client.get("/v1/segmentations").json()["segmentations"]:
        if e["task"] == "relabel":
            assert "links" not in e


def test_no_content_digest_is_path_addressable():
    """The rule lives in resource_links and is asked of content.is_digest: the `sha256:`
    spelling it used to test let an uploaded DICOM series' tree digest through."""
    for ident in ("sha256:" + "a" * 64, "sha256-tree:" + "a" * 64):
        assert content.is_digest(ident)
        assert serve.resource_links("total_fast", [ident], {}) == {}
    assert serve.resource_links("total_fast", ["toy:sp042"], {})["labels"] == \
        "/v1/toy/sp042/total_fast/labels.seg.nrrd"


# -- one fact, one home: the grammar against the things it names ------------------------

def test_the_grammar_is_the_alphabet_of_what_the_server_really_mints(server):
    """The key's alphabet is serve.result_key's, the pin's is content.digest_file's and the
    output name is serve.result_payload's. None of the three is restated here: a real job's
    status is matched against the fragments the identifier is built from."""
    _, _, client = server
    up = _upstream(client)
    assert re.fullmatch(RESULT_KEY_RE, up["key"])
    assert up["key"] == serve.result_key(up["input_identity"], "total_fast", {},
                                        serve.weights_versions_of(Tasks(), "total_fast"))
    (out,) = up["result"]["outputs"]
    assert re.fullmatch(OUTPUT_NAME_RE, out["name"]) and re.fullmatch(RESULT_PIN_RE, out["sha256"])
    assert out["sha256"].startswith(content.BLOB)
    ref = ResultRef(up["key"], out["name"], out["sha256"])
    assert ResultSource.parse(str(ref)) == ref
    assert re.fullmatch(ResultSource.id_pattern, str(ref))


def test_the_primary_output_is_the_one_the_etag_names(server):
    """`result:<key>` means the first output. serve.etag_of decides a result's validator
    the same way, independently: if one moved, a reference would pin bytes the result
    route does not serve under that ETag."""
    _, ex, client = server
    up = _upstream(client)
    pinned = ex.sources["result"].pin(up["key"])
    etag = client.get(f"/v1/jobs/{up['id']}/result").headers["etag"]
    assert etag == f'"{ResultSource.parse(pinned).digest}"'
    assert etag == serve.etag_of(up["key"], up["result"])


def test_no_identifier_can_name_a_host():
    """The SSRF boundary, by construction: nothing a host, a path or a URL needs can be
    spelled in ANY of the three parts, so a reference to another server stays impossible.

    Each character is SUBSTITUTED, at constant length. The first version inserted it, so
    every case was refused for being 65 characters long and a key alphabet widened to
    `[0-9a-f./]` passed the whole file (mutation run, 2026-09-20): refused for the wrong
    reason is not refused."""
    parts = {"key": "0" * 64, "name": "labels", "pin": "sha256:" + "0" * 64}
    whole = "{key}!{name}@{pin}"
    assert re.fullmatch(ResultSource.id_pattern, whole.format(**parts))
    for needs in "/.?#% \\:&=":
        for part, text in parts.items():
            for at in (0, len(text) // 2, len(text) - 1):
                bent = dict(parts, **{part: text[:at] + needs + text[at + 1:]})
                assert len(bent[part]) == len(text)
                ident = whole.format(**bent)
                assert not re.fullmatch(ResultSource.id_pattern, ident), (part, at, ident)
    assert not re.fullmatch(ResultSource.id_pattern, "0" * 64 + "@evil.example:" + "0" * 64)
    assert ResultSource.prefix not in {s.prefix for s in __import__(
        "haversack.sources", fromlist=["default_sources"]).default_sources()}


def test_the_prefetcher_leaves_a_reference_alone_by_the_sources_own_name():
    assert ResultSource.prefix in jobpolicy.NOT_STAGED_AHEAD
    ask = dict(state="queued", kind=None, refresh_input=False)
    assert not jobpolicy.prefetchable(sources=[{"kind": ResultSource.prefix, "id": KEY}], **ask)
    assert jobpolicy.prefetchable(sources=[{"kind": "toy", "id": "sp042"}], **ask)


# -- the label-map role and its reader --------------------------------------------------

def test_a_task_declares_a_label_map_role_by_kind(server):
    _, _, client = server
    inputs = client.get("/v1/tasks/organ_means").json()["inputs"]
    assert [(i["name"], i["kind"]) for i in inputs] == [("image", "image"), ("mask", "labels")]
    assert "channel" not in inputs[1] and inputs[1]["required"] is True
    from haversack import schemas
    assert schemas.LABELS_KIND is LABELS_KIND and schemas.INPUT_KINDS == ("image", "labels")
    assert schemas.input_kind({}) == "image" and schemas.input_kind(inputs[1]) == "labels"


def test_the_reader_reads_what_the_real_writer_writes(tmp_path):
    """Writer and reader are separate code that share only the header's key names."""
    prov = {"task": "ts.v2:total", "device": "mps"}
    Tasks._seg(_labels(), NAMES, prov).save(tmp_path / "labels.seg.nrrd")
    header = sitk.ReadImage(str(tmp_path / "labels.seg.nrrd"))
    assert header.GetMetaData("Segment1_Name") == "spleen"      # the keys are Slicer's
    lm = read_label_map(tmp_path / "labels.seg.nrrd")
    assert lm.names == {1: "liver", 2: "spleen"} and lm.task == "ts.v2:total"
    assert lm.provenance == prov and lm.path.endswith("labels.seg.nrrd")
    assert lm.label_of("spleen") == 2 and int(lm.mask("spleen").sum()) == int((_labels() == 2).sum())
    assert lm.geometry.shape_zyx == SHAPE
    with pytest.raises(KeyError, match="no structure named 'pancreas'.*liver, spleen"):
        lm.mask("pancreas")


def test_a_staged_input_is_a_directory_of_one_file(tmp_path):
    d = tmp_path / "entry" / "series"
    d.mkdir(parents=True)
    Tasks._seg(_labels(), NAMES, {}).save(d / "labels.seg.nrrd")
    (d / ".hidden").write_text("not content", encoding="utf-8")
    assert read_label_map(d).names == {1: "liver", 2: "spleen"}
    (d / "second.seg.nrrd").write_bytes(b"x")
    with pytest.raises(InputError, match="holds 2 files"):
        read_label_map(d)
    with pytest.raises(InputError, match="label map not found"):
        read_label_map(tmp_path / "nope.seg.nrrd")


def test_the_server_record_names_the_task_when_the_header_does_not(tmp_path):
    """FastSurfer, SynthStrip, VoxTell and MONAI bundles write no `task` into their
    provenance; the result cache's meta.json has it for every engine, and a fetched
    reference leaves it in the record beside the bytes."""
    d = tmp_path / "entry" / "series"
    d.mkdir(parents=True)
    Tasks._seg(_labels(), NAMES, {"engine": "fastsurfer"}).save(d / "labels.seg.nrrd")
    assert read_label_map(d).task is None
    about = {"digest": content.digest_file(d / "labels.seg.nrrd")}
    (tmp_path / "entry" / ".input.json").write_text(json.dumps(
        {"kind": "result", "content": about, "origin": {"task": "fastsurfer:asegdkt"}}),
        encoding="utf-8")
    assert read_label_map(d).task == "fastsurfer:asegdkt"
    assert read_label_map(d / "labels.seg.nrrd").task == "fastsurfer:asegdkt"
    (tmp_path / "entry" / ".input.json").write_text(json.dumps(
        {"kind": "idc", "content": about, "origin": {"task": "not a result's record"}}),
        encoding="utf-8")
    assert read_label_map(d).task is None


def test_somebody_elses_record_two_levels_up_names_nothing(tmp_path):
    """A user's own label map in `mine/sub/`, and an unrelated `.input.json` in `mine/`: the
    reader believed it and named a task that never made this file - the false provenance
    claim `sources._dicom_facts` was rewritten to stop making (review, 2026-09-20). A record
    counts only beside a `series/` directory AND when it is about these very bytes."""
    header = {"task": "my_own_task"}
    for folder in ("sub", "series"):
        d = tmp_path / "mine" / folder
        d.mkdir(parents=True)
        Tasks._seg(_labels(), NAMES, header).save(d / "mine.seg.nrrd")
    (tmp_path / "mine" / ".input.json").write_text(json.dumps(
        {"kind": "result", "content": {"digest": "sha256:" + "0" * 64},
         "origin": {"task": "ts.v2:total"}}), encoding="utf-8")
    assert read_label_map(tmp_path / "mine" / "sub" / "mine.seg.nrrd").task == "my_own_task"
    assert read_label_map(tmp_path / "mine" / "series").task == "my_own_task"   # other bytes


def test_a_users_own_label_map_is_never_hashed_to_look_for_a_record(tmp_path, monkeypatch):
    """The digest is what decides whose record it is; the `series/` test before it is what
    keeps a user's own file - any size, anywhere - from being read twice to find that out."""
    d = tmp_path / "mine" / "sub"
    d.mkdir(parents=True)
    Tasks._seg(_labels(), NAMES, {"task": "my_own_task"}).save(d / "mine.seg.nrrd")
    (tmp_path / "mine" / ".input.json").write_text(json.dumps(
        {"kind": "result", "origin": {"task": "ts.v2:total"}}), encoding="utf-8")
    monkeypatch.setattr(content, "digest_file", lambda p: pytest.fail(f"hashed {p}"))
    assert read_label_map(d / "mine.seg.nrrd").task == "my_own_task"


def test_the_name_the_server_keyed_on_outranks_what_the_header_claims(tmp_path):
    """They agree whenever both exist - which is why a mutant that swapped them survived
    the first mutation run. The rule is still a rule: a header is what a FILE says about
    itself, the record is what this server keyed the result on."""
    d = tmp_path / "entry" / "series"
    d.mkdir(parents=True)
    Tasks._seg(_labels(), NAMES, {"task": "what the header says"}).save(d / "labels.seg.nrrd")
    (tmp_path / "entry" / ".input.json").write_text(json.dumps(
        {"kind": "result", "content": {"digest": content.digest_file(d / "labels.seg.nrrd")},
         "origin": {"task": "ts.v2:total"}}), encoding="utf-8")
    assert read_label_map(d).task == "ts.v2:total"


def test_a_label_map_without_names_is_refused_unless_asked_for(tmp_path):
    f = tmp_path / "labels.nii.gz"
    sitk.WriteImage(_image(_labels()), str(f), True)
    with pytest.raises(InputError, match="names no segments.*seg.nrrd"):
        read_label_map(f)
    lm = read_label_map(f, require_names=False)
    assert lm.names == {} and lm.task is None and np.array_equal(lm.array, _labels())


def test_a_layered_segmentation_is_refused_rather_than_read_as_layer_zero(tmp_path):
    img = _image(_labels())
    for k, v in {"Segment0_Name": "a", "Segment0_LabelValue": "1", "Segment0_Layer": "0",
                 "Segment1_Name": "b", "Segment1_LabelValue": "1", "Segment1_Layer": "1"}.items():
        img.SetMetaData(k, v)
    sitk.WriteImage(img, str(tmp_path / "layered.seg.nrrd"), True)
    with pytest.raises(InputError, match="layer 1"):
        read_label_map(tmp_path / "layered.seg.nrrd")
    vec = sitk.Compose([_image(_labels()), _image(_labels())])
    sitk.WriteImage(vec, str(tmp_path / "vector.seg.nrrd"), True)
    with pytest.raises(InputError, match="single-layer 3D label map"):
        read_label_map(tmp_path / "vector.seg.nrrd")


def test_reading_a_label_map_needs_none_of_the_heavy_stack():
    import subprocess
    import sys
    code = ("import sys, haversack, haversack.labelmap, haversack.sources\n"
            "bad = [m for m in ('torch', 'pydantic', 'duckn', 'zarr', 'rankfield') "
            "if m in sys.modules]\nassert not bad, bad\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


# -- adversarial review, 2026-09-20: each finding pinned, each surviving mutant given a test --

def test_a_label_map_role_is_required_because_the_binder_says_so():
    """`label_input(required=False)` declared an optional role that `bind_sources` then
    refused: one fact in two places, and only one of them was read."""
    import inspect

    from haversack.schemas import bind_sources, input_specs, label_input
    assert "required" not in inspect.signature(label_input).parameters
    inputs = input_specs(["image"]) + [label_input("mask")]
    assert all(i["required"] is True for i in inputs)
    with pytest.raises(RequestError) as e:
        bind_sources([{"kind": "toy", "id": "sp042", "role": "image"}], inputs,
                     multi_input=True, task="t")
    assert e.value.code == "missing_role"


def test_a_hit_on_an_answer_first_computed_from_an_upload_says_so_and_no_cache_mends_it(server):
    """The documented cost of a reference and an upload being one request (SERVER.md,
    "Results as inputs"): provenance describes the computation that produced the answer."""
    _, _, client = server
    up = _upstream(client)
    labels = client.get(f"/v1/jobs/{up['id']}/result").content
    _done(client, _post(client, "relabel", [{"kind": "upload"}],
                        files={"file": ("labels.seg.nrrd", labels)}))
    hit = _done(client, _post(client, "relabel", [{"kind": "result", "id": up["key"]}]))
    assert hit.get("cached") is True
    assert [r["kind"] for r in hit["result"]["provenance"]["inputs"]] == ["upload"]
    again = _done(client, _post(client, "relabel", [{"kind": "result", "id": up["key"]}],
                                headers={"Cache-Control": "no-cache"}))
    (rec,) = again["result"]["provenance"]["inputs"]
    assert rec["kind"] == "result" and rec["derived_from"]["inputs"][0]["license"] == ToyCT.LICENSE


@pytest.mark.parametrize("stated, name", [
    ("sha256-tree:" + "a" * 64, "labels"), ("sha256:" + "A" * 64, "labels"),
    ("", "labels"), ("sha256:" + "a" * 64, ""), ("sha256:" + "a" * 64, "Not A Name"),
])
def test_pin_never_mints_an_identifier_its_own_grammar_refuses(tmp_path, stated, name):
    """`_judge` checked the stated digest with content.is_digest, which takes a tree digest
    and uppercase hex; the grammar takes neither, so pin() returned a string identity()
    then refused. Both halves - the digest and the name - are held to the fragments."""
    path, result = _entry(tmp_path)
    result["outputs"][0].update(sha256=stated, name=name)
    with pytest.raises(UnresolvedReference) as e:
        _source((path, result)).pin(KEY)
    assert e.value.code == "result_unreadable"


def test_a_malformed_upstream_provenance_thins_the_record_and_never_fails_the_fetch(tmp_path):
    from haversack.sources import _carried_forward
    odd = [{"kind": "result", "identity": ["unhashable"], "origin": "a string",
            "derived_from": ["a list"]}, "not a record", {"kind": "upload", "identity": None,
            "license": {"name": "L1"}}, {"kind": "upload", "identity": None,
                                         "license": {"name": "L2"}}]
    got = _carried_forward(odd)
    # two records that state NO identity are two inputs: folding them lost a license
    assert [r["license"]["name"] for r in got["inputs"]] == ["L1", "L2"]
    path, result = _entry(tmp_path, b"the labels")
    result["provenance"] = {"inputs": odd}
    src = _source((path, result))
    (tmp_path / "entry").mkdir()
    fetched = src.fetch(src.pin(KEY), tmp_path / "entry")
    assert (fetched / serve.RESULT_NAME).read_bytes() == b"the labels"


def test_only_an_engines_own_statement_reads_as_a_weights_version():
    from haversack.sources import _stated_weights
    assert _stated_weights({"fastsurfer_version": "2.5.4", "device": "cuda"}) == ["fastsurfer=2.5.4"]
    assert _stated_weights({"bundle_version": "0.6.1", "engine": "monai"}) == ["bundle=0.6.1"]
    assert _stated_weights({"x_version": {"a": 1}, "_version": "1", "haversack_version": "0.12",
                            "y_version": ""}) == []


def test_terms_travel_a_chain_whose_middle_never_touches_the_original(server):
    """relabel of relabel of total_fast: the CT reaches the last record ONLY through
    `derived_from.inputs`, and the first hop only through `derived_from.results`. The chain
    test above binds the CT again in the middle, so it could not see either being dropped."""
    _, _, client = server
    a = _upstream(client)
    b = _done(client, _post(client, "relabel", [{"kind": "result", "id": a["key"]}]))
    c = _done(client, _post(client, "relabel", [{"kind": "result", "id": b["key"]}]))
    rec = _mask_record(c)
    assert [(h["result"], h["task"]) for h in rec["derived_from"]["results"]] == [(a["key"], "total_fast")]
    assert [(r["identity"], r["license"]) for r in rec["derived_from"]["inputs"]] == \
        [("toy:sp042", ToyCT.LICENSE)]
    assert isinstance(rec["origin"]["computed"], float)


def test_a_hop_reached_by_two_paths_is_listed_once():
    from haversack.sources import _carried_forward
    hop = {"result": "k1", "output": "labels", "digest": "sha256:1", "task": "t", "weights": []}
    via = {"kind": "result", "identity": "sha256:2", "origin": {"result": "k2"},
           "derived_from": {"results": [hop], "inputs": []}}
    got = _carried_forward([via, dict(via, identity="sha256:3", origin={"result": "k3"})])
    assert [h["result"] for h in got["results"]] == ["k2", "k1", "k3"]


def test_an_outputs_own_kind_is_what_the_role_is_checked_against(tmp_path):
    """Generic on purpose (step 2 adds kinds): the entry says what its output is, and an
    entry that says nothing is a label map, as every one published so far is."""
    path, result = _entry(tmp_path)
    result["outputs"][0]["kind"] = "field"
    src = _source((path, result))
    assert src.pin(KEY, "field").startswith(KEY)
    with pytest.raises(RequestError) as e:
        src.pin(KEY, LABELS_KIND)
    assert e.value.code == "wrong_input_kind" and e.value.detail["output_kind"] == "field"


def test_the_source_refuses_by_itself_what_the_wire_refuses(tmp_path):
    """Every door calls check(): the hints must not depend on check_identifier running first."""
    src = _source(_entry(tmp_path))
    with pytest.raises(InputError, match="looks like a job id"):
        src.check("0123456789ab")
    said = src.describe_input(f"{KEY}!labels")           # nothing fetched: the reference alone
    assert said["origin"] == {"result": KEY, "output": "labels",
                              "determined_by": "the reference itself; the entry was not read"}


def test_the_lookup_never_runs_on_the_event_loop(server):
    """On Modal `pin` waits out a volume reload; on the loop that froze the api container
    (2026-09-19). A thread started by to_thread has no running loop; the loop's own does."""
    import asyncio
    _, ex, client = server
    up = _upstream(client)
    src, on_loop = ex.sources["result"], []
    real = src.pin

    def pin(identifier, kind=None):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(identifier, kind)
    src.pin = pin
    _done(client, _post(client, "relabel", [{"kind": "result", "id": up["key"]}]))
    assert on_loop == [False]


def test_a_lookup_that_refuses_as_an_input_error_is_a_422_not_a_500(server, tmp_path):
    _, ex, client = server
    up = _upstream(client)
    dirs = set((tmp_path / "w").iterdir())

    def pin(identifier, kind=None):
        raise InputError("this lookup cannot be made")
    ex.sources["result"].pin = pin
    r = _post(client, "relabel", [{"kind": "result", "id": up["key"]}])
    assert r.status_code == 422 and "this lookup cannot be made" in str(r.json()["detail"])
    assert set((tmp_path / "w").iterdir()) == dirs          # and no job directory is left


def _seg_file(path, fields, array=None, pixel=None):
    img = _image(_labels() if array is None else array)
    if pixel is not None:
        img = sitk.Cast(img, pixel)
    for k, v in fields.items():
        img.SetMetaData(k, v)
    sitk.WriteImage(img, str(path), True)
    return path


def test_the_reader_refuses_to_choose_between_colliding_segments(tmp_path):
    two_on_one = {"Segment0_Name": "liver", "Segment0_LabelValue": "1",
                  "Segment1_Name": "spleen", "Segment1_LabelValue": "1"}
    with pytest.raises(InputError, match="'liver' and 'spleen' share label value 1"):
        read_label_map(_seg_file(tmp_path / "a.seg.nrrd", two_on_one))
    one_on_two = {"Segment0_Name": "kidney", "Segment0_LabelValue": "1",
                  "Segment1_Name": "kidney", "Segment1_LabelValue": "2"}
    with pytest.raises(InputError, match="two segments are named 'kidney'"):
        read_label_map(_seg_file(tmp_path / "b.seg.nrrd", one_on_two))


def test_names_without_label_values_and_an_image_are_each_called_what_they_are(tmp_path):
    unvalued = {"Segment0_Name": "liver", "Segment1_Name": "spleen", "Segment1_LabelValue": "x"}
    with pytest.raises(InputError, match="names 2 segment.s. and gives none a LabelValue"):
        read_label_map(_seg_file(tmp_path / "a.seg.nrrd", unvalued))
    ok = {"Segment0_Name": "liver", "Segment0_LabelValue": "1", "Segment1_Name": "loose"}
    assert read_label_map(_seg_file(tmp_path / "b.seg.nrrd", ok)).names == {1: "liver"}
    with pytest.raises(InputError, match="a label map is integers"):
        read_label_map(_seg_file(tmp_path / "c.seg.nrrd", ok, pixel=sitk.sitkFloat32))


def test_a_copy_that_never_matches_says_what_to_do_about_it(tmp_path):
    path, result = _entry(tmp_path, b"what is really there", stated="sha256:" + "5" * 64)
    (tmp_path / "entry").mkdir()
    with pytest.raises(InputError, match="if it repeats.*no-cache"):
        _source((path, result)).fetch(f"{KEY}!labels@sha256:{'5' * 64}", tmp_path / "entry")
