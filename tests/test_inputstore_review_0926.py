"""The input store's review of 2026-09-26: each test reproduces one defect a reviewer found in
step 6 (docs/cache-consolidation.md) and fails on the tree it was found in (``4b5ecb2``).

1. `cache clean inputs` removed the CURRENT reader version's ref in place of the listed one.
2. Concurrent callers each built a view of one key; all but the last were orphaned.
3. A view handed to one caller was deleted by another's unpin or by the loose reap; and the
   input-status route answered an eviction between its two looks with a 500.
4. A blob write killed mid-way left its temporary file forever.
5. `cache usage` / `clean` counted hard-linked bytes once per name.
6. `Input.array` read header bytes as voxels for a negative start, and errors varied by form.
7. The legacy-upload shim ran on the event loop.
8. A second process placing an export deleted the one the first had placed and returned.
9. `cache clean inputs <item>` could not name an ingested input, and said nothing.
10. Unrelated keys on one economy-lock stripe waited for each other's fetch.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from haversack import cache_admin, inputstore, sources
from haversack.errors import InputError
from haversack.inputstore import CommandInputs, InputStore, ServerInputs, key_for, ref_name

ROOT = Path(__file__).resolve().parent.parent


def _nifti(path, value=7, shape=(24, 5, 4)):
    arr = (np.arange(np.prod(shape), dtype=np.int16).reshape(shape) + value)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.8, 0.9, 2.0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return Path(path)


@pytest.fixture
def flag(monkeypatch, tmp_path):
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    monkeypatch.setenv("HAVERSACK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(inputstore, "_COMMAND", {})
    return tmp_path


def _export(ci, key):
    return ci.exports / Path(ref_name(key)).stem       # spelled out: the name predates the fix


def _command_store():
    return inputstore.command_inputs(sources.default_input_store())


# -- 1: clean by the listed ref's own key ------------------------------------------------

def test_clean_removes_an_older_reader_versions_ref_and_keeps_the_current_one(flag, monkeypatch):
    from haversack import input_copy
    ci = _command_store()
    ct = _nifti(flag / "local" / "ct.nii.gz")
    monkeypatch.setattr(input_copy, "READER_VERSION", 2)
    d, _ = ci.ingest_with_identity(ct)
    old_key = key_for(d)
    monkeypatch.setattr(input_copy, "READER_VERSION", 3)
    ci.ingest_with_identity(ct)
    new_key = key_for(d)
    assert old_key != new_key and {k for _, k, _ in ci.store._refs()} == {old_key, new_key}
    old_ref = ci.store.store.root / ref_name(old_key)
    os.utime(old_ref, (time.time() - 5 * 86400,) * 2)
    assert _export(ci, old_key).is_dir()

    cache_admin.clean("inputs", older_than_days=1)

    assert {k for _, k, _ in ci.store._refs()} == {new_key}      # the r2 ref went, r3 stayed
    assert not _export(ci, old_key).exists()
    assert _export(ci, new_key).is_dir()
    assert ci.store.has(d)


def test_clean_of_an_identity_takes_every_reader_versions_ref_of_it(flag, monkeypatch):
    from haversack import input_copy
    ci = _command_store()
    ct = _nifti(flag / "local" / "ct.nii.gz")
    monkeypatch.setattr(input_copy, "READER_VERSION", 2)
    d, _ = ci.ingest_with_identity(ct)
    monkeypatch.setattr(input_copy, "READER_VERSION", 3)
    ci.ingest_with_identity(ct)
    other, _ = ci.ingest_with_identity(_nifti(flag / "local" / "other.nii.gz", value=99))
    cache_admin.clean("inputs", item=d, dry_run=False)
    assert [doc["identity"] for _, _, doc in ci.store._refs()] == [other]


# -- 9: an ingested input can be named ---------------------------------------------------

def test_clean_names_an_ingested_input_by_its_path_or_digest(flag):
    ci = _command_store()
    a = _nifti(flag / "local" / "a.nii.gz", value=1)
    b = _nifti(flag / "local" / "b.nii.gz", value=2)
    da, _ = ci.ingest_with_identity(a)
    db, _ = ci.ingest_with_identity(b)
    r = cache_admin.clean("inputs", item=str(a))
    assert r["removed"] == [da]
    assert [doc["identity"] for _, _, doc in ci.store._refs()] == [db]
    cache_admin.clean("inputs", item=db)
    assert ci.store._refs() == []


def test_clean_names_an_ingested_folder_by_its_path(flag):
    from test_several_series import THREE, write_series
    ci = _command_store()
    folder = flag / "local" / "series"
    folder = write_series(folder, 3, THREE, value=6)
    d, _ = ci.ingest_with_identity(folder)
    assert d.startswith("sha256-tree:")
    cache_admin.clean("inputs", item=str(folder))
    assert ci.store._refs() == []
    assert inputstore.local_identity(folder) == d


def test_an_item_that_names_nothing_is_refused_not_ignored(flag, monkeypatch):
    _command_store().ingest_with_identity(_nifti(flag / "local" / "a.nii.gz"))
    with pytest.raises(InputError, match="names no input"):
        cache_admin.clean("inputs", item=str(flag / "no-such-file.nii.gz"))
    monkeypatch.delenv("HAVERSACK_INPUT_STORE")
    (flag / "cache" / "inputs" / "fake").mkdir(parents=True)
    with pytest.raises(InputError, match="not a source spec"):
        cache_admin.clean("inputs", item="sha256:" + "0" * 64)


# -- 5: each file's bytes once -----------------------------------------------------------

def test_usage_counts_a_hard_linked_file_once(flag):
    ci = _command_store()
    ci.ingest_with_identity(_nifti(flag / "local" / "a.nii.gz"))
    root = sources.default_input_store()
    files = [f for f in root.rglob("*") if f.is_file()]
    by_name = sum(f.stat().st_size for f in files)
    by_inode = sum({(f.stat().st_dev, f.stat().st_ino): f.stat().st_size for f in files}.values())
    assert by_name > by_inode                          # the export IS a hard link to the blob
    row = {r["name"]: r for r in cache_admin.usage()}["inputs"]
    assert row["bytes"] == by_inode


# -- 2 and 3: views ----------------------------------------------------------------------

def _server_store(tmp_path, **kw):
    si = ServerInputs(tmp_path / "input_store", None, **kw)
    d = si.put_file(_nifti(tmp_path / "up" / "a.nii.gz", value=1))
    d1 = si.put_file(_nifti(tmp_path / "up" / "b.nii.gz", value=2))
    return si, d, d1


def test_concurrent_callers_share_one_view_and_none_is_orphaned(tmp_path):
    si, d, _ = _server_store(tmp_path)
    real = si.store.materialize

    def slow(identity, dest):
        time.sleep(0.2)                                # every caller arrives while it builds
        return real(identity, dest)
    si.store.materialize = slow
    barrier, got = threading.Barrier(8), []

    def go():
        barrier.wait()
        si.pin(d)
        got.append(si.path(d))
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len({str(p) for p in got}) == 1
    assert len(list(si._views_root.iterdir())) == 1
    for _ in range(8):
        si.unpin(d)
    assert list(si._views_root.iterdir()) == []       # the last unpin leaves nothing behind


def test_a_failed_build_lets_the_waiters_build_their_own(tmp_path):
    si, d, _ = _server_store(tmp_path)
    real, calls = si.store.materialize, []

    def flaky(identity, dest):
        calls.append(1)
        time.sleep(0.1)
        if len(calls) == 1:
            raise OSError("the first build fails")
        return real(identity, dest)
    si.store.materialize = flaky
    out, errs = [], []

    def go():
        try:
            out.append(si.resolve(d))
        except OSError as e:
            errs.append(e)
    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(errs) == 1 and len(out) == 3 and len({str(p) for p in out}) == 1
    assert len(list(si._views_root.iterdir())) == 1


def test_a_reused_loose_view_is_kept_for_its_new_holder(tmp_path):
    si, d, d1 = _server_store(tmp_path)
    si.LOOSE_VIEW_S = 0.5
    first = si.resolve(d)
    time.sleep(0.6)
    second = si.resolve(d)                             # handed the same view, later
    si.resolve(d1)                                     # builds a view: the loose reap runs
    assert second == first and second.exists()


def test_an_unpin_keeps_a_view_a_loose_holder_was_just_handed(tmp_path):
    si, d, _ = _server_store(tmp_path)
    loose = si.resolve(d)                              # nobody holds a pin: a lease
    si.pin(d)
    assert si.path(d) == loose
    si.unpin(d)                                        # a job on the same input ends
    assert loose.exists()


def _app(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from haversack.serve import LocalExecutor, create_app
    from test_serve import FakeSegmenter
    monkeypatch.setenv("HAVERSACK_INPUT_STORE", "blobs")
    ex = LocalExecutor(FakeSegmenter(), workdir=tmp_path)
    return ex, TestClient(create_app(ex), raise_server_exceptions=False)


def _upload(client, fill):
    from test_serve import volume_bytes
    raw = volume_bytes(fill)
    d = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert client.put(f"/v1/inputs/{d}", content=raw).status_code in (200, 201)
    return d


def test_the_input_status_route_holds_what_it_reads(tmp_path, monkeypatch):
    """A job's last unpin while the route reads the view it was handed: a 500 before."""
    from haversack import input_copy
    ex, client = _app(tmp_path, monkeypatch)
    d = _upload(client, 31)
    ex.content.pin(d)                                  # a job on this input...
    ex.content.fast_path(d)
    real = input_copy.info

    def job_ends_meanwhile(path):
        ex.content.unpin(d)                            # ...ends while the route looks
        return real(path)
    monkeypatch.setattr(input_copy, "info", job_ends_meanwhile)
    r = client.get(f"/v1/inputs/{d}")
    assert r.status_code == 200, r.text
    assert r.json()["stored_form"] == "input_copy"
    ex.close()


def test_an_eviction_between_the_routes_two_looks_is_input_gone(tmp_path, monkeypatch):
    ex, client = _app(tmp_path, monkeypatch)
    d = _upload(client, 32)
    real, n = ex.content.has, []

    def evicted_after_answering(digest):
        got = real(digest)
        if not n:
            n.append(1)
            ex.content.store.forget(digest)             # evicted between `has` and `resolve`
        return got
    monkeypatch.setattr(ex.content, "has", evicted_after_answering)
    r = client.get(f"/v1/inputs/{d}")
    assert r.status_code == 404, r.text
    assert r.json()["detail"]["code"] == "input_gone"
    ex.close()


# -- 7: the shim off the loop ------------------------------------------------------------

def test_a_reference_to_held_content_is_looked_up_off_the_event_loop(tmp_path, monkeypatch):
    import asyncio
    ex, client = _app(tmp_path, monkeypatch)
    d = _upload(client, 33)
    real, on_loop = ex.content.has, []

    def has(digest):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(digest)
    monkeypatch.setattr(ex.content, "has", has)
    r = client.post("/v1/jobs", data={"task": "total_fast",
                                      "source": json.dumps([{"kind": "input", "sha256": d}])})
    assert r.status_code == 202, r.text
    assert on_loop and on_loop[0] is False
    ex.close()


# -- 4: a killed write's temporary file ---------------------------------------------------

_WRITER = """
import sys, time
import provender.disk as disk
def stalled(tmp, file):
    with open(tmp, "wb") as f:
        f.write(b"x" * 65536)
    print("WRITING", flush=True)
    time.sleep(120)
disk._write = stalled
from haversack.inputstore import InputStore
InputStore(sys.argv[1], None).put_file(sys.argv[2])
"""


def _stalled_writer(tmp_path):
    ct = _nifti(tmp_path / "up" / "a.nii.gz")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in (str(Path(inputstore.__file__).resolve().parents[1]),
                    os.environ.get("PYTHONPATH", "")) if p)}
    proc = subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path / "s"), str(ct)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    line = proc.stdout.readline()
    assert line.strip() == "WRITING", proc.stderr.read()
    return proc


def _temps(tmp_path):
    return list((tmp_path / "s" / "store").rglob("*.provender-tmp"))


def test_a_live_writers_temporary_file_is_never_taken(tmp_path):
    proc = _stalled_writer(tmp_path)
    try:
        assert len(_temps(tmp_path)) == 1
        InputStore(tmp_path / "s", None, grace_s=0).evict()   # open + evict, no grace at all
        assert len(_temps(tmp_path)) == 1
    finally:
        proc.kill()
        proc.wait()


def test_a_killed_writers_temporary_file_is_reaped_once_its_grace_is_out(tmp_path):
    proc = _stalled_writer(tmp_path)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    assert len(_temps(tmp_path)) == 1
    InputStore(tmp_path / "s", None)                  # an hour's grace: a fresh one stays
    assert len(_temps(tmp_path)) == 1
    InputStore(tmp_path / "s", None, grace_s=0)       # opening the store reaps it
    assert _temps(tmp_path) == []


def test_cache_clean_takes_a_killed_writers_temporary_file(flag):
    ci = _command_store()
    ci.ingest_with_identity(_nifti(flag / "local" / "a.nii.gz"))
    blobs = ci.store.store.root / "blobs" / "sha256"
    litter = blobs / ".0123abcd.0123456789ab.provender-tmp"
    litter.write_bytes(b"x" * 4096)
    os.utime(litter, (time.time() - 7200,) * 2)
    inputstore._COMMAND.clear()
    cache_admin.clean("inputs")
    assert not litter.exists()


# -- 6: one slicing rule -------------------------------------------------------------------

SLICES = [(0, 24), (5, 5), (18, 30), (-2, 24), (-3, -1), (10, 5), (25, 30), (0, 25), (-40, 3)]


@pytest.mark.parametrize("compression", ["zstd", "uncompressed", "none-a-plain-file"])
def test_array_slices_as_python_slices_on_every_form(flag, monkeypatch, compression):
    from haversack import inputs
    ct = _nifti(flag / "local" / f"ct-{compression}.nii.gz")
    if compression == "none-a-plain-file":
        x = inputs.Input(None, ct, None)
        assert not x.is_copy
    else:
        monkeypatch.setenv("HAVERSACK_INPUT_COPY_COMPRESSION", compression)
        x = inputs.open(ct)
        assert x.is_copy
    full = x.array()
    assert full.shape == (24, 5, 4)
    for lo, hi in SLICES:
        got = x.array((lo, hi))
        want = full[lo:hi]
        assert got.shape == want.shape and np.array_equal(got, want), (lo, hi)


@pytest.mark.parametrize("compression", ["zstd", "uncompressed", "none-a-plain-file"])
def test_array_refuses_what_is_not_two_integers_the_same_way_on_every_form(
        flag, monkeypatch, compression):
    from haversack import inputs
    ct = _nifti(flag / "local" / f"ct-{compression}.nii.gz")
    if compression == "none-a-plain-file":
        x = inputs.Input(None, ct, None)
    else:
        monkeypatch.setenv("HAVERSACK_INPUT_COPY_COMPRESSION", compression)
        x = inputs.open(ct)
    for bad in [(1.5, 3), (1,), "ab", (None, 3), (True, 3)]:
        with pytest.raises(ValueError, match="two integers"):
            x.array(bad)


# -- 8: an export placed by another process stays ----------------------------------------

def test_an_export_another_process_placed_is_kept_not_replaced(flag):
    ci = CommandInputs(flag / "cmd")
    d = ci.store.put_file(_nifti(flag / "local" / "a.nii.gz"))
    dest = ci.export_dir(d)
    assert not dest.exists()
    real, theirs = ci.store.materialize, {}

    def racing(identity, where):
        out = real(identity, where)
        if not theirs:                                 # meanwhile, elsewhere, the same export
            other = dest.parent / ".theirs"
            real(identity, other)
            os.rename(other, dest)
            theirs["ino"] = dest.stat().st_ino
        return out
    ci.store.materialize = racing
    path = ci.get_or_fetch(d, fetch=None)
    assert path.exists() and dest.stat().st_ino == theirs["ino"]
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]   # nothing left aside


def test_a_broken_export_is_replaced_whole(flag):
    ci = CommandInputs(flag / "cmd")
    d = ci.store.put_file(_nifti(flag / "local" / "a.nii.gz"))
    path = ci.get_or_fetch(d, fetch=None)
    path.unlink()                                      # a copy a crash cut short (never a write
    path.write_bytes(b"truncated")                     # through the link: that is the blob)
    again = ci.get_or_fetch(d, fetch=None)
    assert again == path and again.stat().st_size == next(
        iter(ci.store.ref(d)["files"].values()))["size"]


_GETTER = """
import sys
from haversack.inputstore import CommandInputs
ci = CommandInputs(sys.argv[1])
p = ci.get_or_fetch(sys.argv[2], fetch=None)
print(p, p.exists(), p.stat().st_ino)
"""


def test_processes_asking_for_one_export_at_once_all_get_it(flag):
    ci = CommandInputs(flag / "cmd")
    d = ci.store.put_file(_nifti(flag / "local" / "a.nii.gz"))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in (str(Path(inputstore.__file__).resolve().parents[1]),
                    os.environ.get("PYTHONPATH", "")) if p)}
    procs = [subprocess.Popen([sys.executable, "-c", _GETTER, str(flag / "cmd"), d],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
             for _ in range(6)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), [e for _, e in outs]
    lines = {o.strip() for o, _ in outs}
    assert len(lines) == 1 and " True " in lines.pop()


# -- 10: unrelated keys do not wait on one stripe ----------------------------------------

def _stripe(key, n):
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") % n


def test_unrelated_fetches_that_shared_a_stripe_no_longer_wait(tmp_path, monkeypatch):
    keys = {}
    for i in range(20000):
        k = f"idc:k{i}"
        keys.setdefault(_stripe(key_for(k), 256), []).append(k)
    a, b = next((a, b) for ks in keys.values() for a in ks for b in ks
                if _stripe(key_for(a), 4096) != _stripe(key_for(b), 4096))
    monkeypatch.setattr(inputstore, "ECONOMY_WAIT_S", 3.0)
    store = InputStore(tmp_path / "s", None, grace_s=0)
    held, release = threading.Event(), threading.Event()

    def slow(identity, entry):
        held.set()
        release.wait(10)
        (Path(entry) / "series").mkdir()
        _nifti(Path(entry) / "series" / "ct.nii.gz")
        return Path(entry) / "series"

    def quick(identity, entry):
        (Path(entry) / "series").mkdir()
        _nifti(Path(entry) / "series" / "ct.nii.gz", value=5)
        return Path(entry) / "series"
    t = threading.Thread(target=lambda: store.ensure(a, fetch=slow))
    t.start()
    try:
        assert held.wait(10)
        t0 = time.monotonic()
        store.ensure(b, fetch=quick)
        assert time.monotonic() - t0 < 2.0
    finally:
        release.set()
        t.join()
