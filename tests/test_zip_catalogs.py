"""The zip-manifest catalogs: one install, three packagings.

MOOSE, DentalSegmentator and TotalVibeSegmentator all publish bare nnU-Net
checkpoints as zips and all read their labels from the installed checkpoint, so
they share :class:`ZipManifestEcosystem`. What differs is only how the archive is
laid out, and these tests pin each difference: a zip with its own ``Dataset*``
parent (MOOSE, DentalSegmentator), a zip whose top level is the bare
configuration folder (TotalVibeSegmentator), macOS zip litter, and the digest the
host happens to publish. No network - every asset here is built in memory.
"""
import hashlib
import io
import json
import re
import time
import zipfile
from unittest import mock

import pytest

import urllib.request

from haversack import fetchlib
from haversack.ecosystems import (CADSEcosystem, DentalSegmentatorEcosystem, EcosystemCatalog,
                              TotalVibeEcosystem, ZipManifestEcosystem, registry)
from haversack.errors import InputError, ModelNotFound

CONFIG = "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"


def _model_files(labels=None, orientation=None, checkpoint="checkpoint_final.pth") -> dict:
    dataset = {"channel_names": {"0": "CT"},
               "labels": {"background": 0, **(labels or {"mandible": 1, "upper_teeth": 2})},
               "numTraining": 1, "file_ending": ".nii.gz"}
    if orientation is not None:
        dataset["orientation"] = orientation
    return {"dataset.json": json.dumps(dataset),
            "plans.json": json.dumps({"image_reader_writer": "SimpleITKIO",
                                      "configurations": {"3d_fullres": {}}}),
            f"fold_0/{checkpoint}": "weights"}


def _zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)
    return buf.getvalue()


def _nested_zip(dataset="Dataset112_DentalSegmentator_v100", litter=True, **kw) -> bytes:
    """DentalSegmentator's shape: a Dataset parent, and macOS zip litter."""
    members = {f"{dataset}/nnUNetTrainer__nnUNetPlans__3d_fullres/{n}": b
               for n, b in _model_files(**kw).items()}
    if litter:
        members[f"{dataset}/.DS_Store"] = "junk"
        members[f"__MACOSX/{dataset}/._dataset.json"] = "junk"
    return _zip(members)


def _bare_config_zip(**kw) -> bytes:
    """TotalVibeSegmentator's shape: the configuration folder IS the top level."""
    return _zip({f"{CONFIG}/{n}": b for n, b in _model_files(**kw).items()})


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _serve(payload: bytes):
    return mock.patch.object(fetchlib, "urlopen", lambda *a, **k: _Resp(payload))


def _expecting(eco, payload: bytes, task: str):
    """Point the entry's digest at the fake asset - the shipped manifests carry
    the real ones, so a fake zip is correctly refused until the entry expects it."""
    entry = {**eco._entries[task]}
    entry.pop("sha256", None)
    entry.pop("md5", None)
    algo = "md5" if eco.name == "dentalsegmentator" else "sha256"
    entry[algo] = hashlib.new(algo, payload).hexdigest()
    eco._entries[task] = entry
    return eco


# --- what the manifests may and may not carry -------------------------------

def test_manifests_hold_only_what_the_checkpoint_cannot_know():
    for eco in (DentalSegmentatorEcosystem(), TotalVibeEcosystem(), CADSEcosystem()):
        assert eco.tasks(), eco.name
        for task, e in eco._entries.items():
            where = f"{eco.name}:{task}"
            assert e["url"].startswith("https://"), where
            assert e["tag"], where
            assert e["folder"].startswith("Dataset"), where
            # labels are read from the checkpoint, never listed here
            assert "labels" not in e and "label_map" not in e, where
            digest = e.get("sha256") or e.get("md5")
            if digest:
                assert re.fullmatch(r"[0-9a-f]{32,64}", digest), where


def test_both_catalogs_are_registered_and_run_on_the_nnunet_engine(tmp_path):
    reg = registry(None)
    assert {"dentalsegmentator", "totalvibe"} <= set(reg)
    cat = EcosystemCatalog(root=tmp_path)
    assert "dentalsegmentator:base" in cat.names()
    assert cat.engine_of("dentalsegmentator:base") == "nnunetv2"
    assert cat.engine_of("totalvibe:vibe") == "nnunetv2"
    info = cat.info("totalvibe:vibe")
    assert info["modality"] == "MR" and info["materialized"] is False
    assert info["dataset_id"] == "100"          # upstream identifies it by number only
    assert cat.info("dentalsegmentator:base")["modality"] == "CT"


def test_every_published_totalvibe_asset_is_accounted_for():
    """The release carries more models than the catalog offers. A drop with no
    reason recorded is indistinguishable from an oversight, so the offered tasks
    and the recorded exclusions must PARTITION the published assets exactly -
    checkable offline because the generator records what the release held."""
    raw = json.loads(TotalVibeEcosystem.MANIFEST.read_text(encoding="utf-8"))
    published = set(raw["release_assets"])
    offered = {e["dataset_id"] for e in raw["tasks"].values()}
    excluded = set(raw["excluded"])
    assert offered and excluded
    assert offered.isdisjoint(excluded)
    assert offered | excluded == published, (
        f"unaccounted for: {sorted(published - offered - excluded)}")
    for idx, reason in raw["excluded"].items():
        assert len(reason) > 20, f"{idx} has no real reason: {reason!r}"
    # the two the docstrings single out, and the one that shares their shape
    for idx in ("282", "085", "086"):
        assert "channel" in raw["excluded"][idx], idx


def test_the_manifest_says_which_downloads_are_unverified():
    """Three assets are published with no digest. Silence would read as "checked",
    so the manifest names them and the reason."""
    raw = json.loads(TotalVibeEcosystem.MANIFEST.read_text(encoding="utf-8"))
    unverified = set(raw["unverified"]["tasks"])
    assert unverified == {n for n, e in raw["tasks"].items() if "sha256" not in e}
    assert unverified and raw["unverified"]["why"]
    for name, e in raw["tasks"].items():
        assert ("sha256" in e) == (name not in unverified)


# --- the install shapes -----------------------------------------------------

def test_nested_zip_installs_its_dataset_folder_and_drops_macos_litter(tmp_path):
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload):
        eco.ensure("base", tmp_path)
    root = tmp_path / "dentalsegmentator"
    assert (root / "Dataset112_DentalSegmentator_v100"
            / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json").is_file()
    assert not (root / "__MACOSX").exists()            # never lands beside the model
    assert not list(root.rglob(".DS_Store"))
    assert eco.materialized("base", tmp_path)
    spec = eco.spec("base", tmp_path)
    assert spec.lineage == "nnunetv2" and set(spec.label_map.values()) == {"mandible", "upper_teeth"}
    with _serve(b"not a zip"):                          # idempotent
        eco.ensure("base", tmp_path)


def test_bare_config_zip_gets_the_dataset_parent_the_archive_lacks(tmp_path):
    """TotalVibeSegmentator ships the configuration folder at the archive root;
    the Dataset<id> parent is the catalog's to create, which is what makes the
    result an ordinary nnU-Net results tree."""
    payload = _bare_config_zip(orientation=["L", "P", "S"], checkpoint="checkpoint_best.pth")
    eco = _expecting(TotalVibeEcosystem(), payload, "body_regions")
    with _serve(payload):
        eco.ensure("body_regions", tmp_path)
    assert (tmp_path / "totalvibe" / "Dataset278" / CONFIG / "dataset.json").is_file()
    assert eco.materialized("body_regions", tmp_path)
    assert eco.spec("body_regions", tmp_path).name == "body_regions"


def test_cads_offers_every_release_asset_and_each_is_verified(tmp_path):
    """The open release is exactly the nine models, T551-T559; each download is checked
    against the sha256 GitHub publishes, and between them they name the 167 structures of the
    paper's Supplementary Table 2."""
    raw = json.loads(CADSEcosystem.MANIFEST.read_text(encoding="utf-8"))
    tasks = raw["tasks"]
    assert {e["folder"] + ".zip" for e in tasks.values()} == set(raw["release_assets"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) for e in tasks.values())
    assert sum(e["structures"] for e in tasks.values()) == 167
    assert {e["license"] for e in tasks.values()} == {"CC-BY-SA-4.0"}   # the open weights only
    cat = EcosystemCatalog(root=tmp_path)
    assert cat.engine_of("cads:organs") == "nnunetv2"
    info = cat.info("cads:organs")
    assert info["modality"] == "CT" and info["dataset_id"] == "551" and info["materialized"] is False


def test_cads_runs_in_the_totalsegmentator_lineage_so_its_sides_are_right(tmp_path):
    """CADS reorients to RAS before its network, but its plans say SimpleITKIO and its
    dataset.json names no orientation. Read as a stock nnU-Net model the same folder keeps
    the acquisition's axis order, and every left/right structure lands on the wrong side
    (mean Dice 0.034 against upstream, 2026-09-11). The catalog's spec must say RAS."""
    from pathlib import Path

    from haversack import io as nio
    from haversack.pipeline import canonical_orientation_for
    from haversack.tasks import TaskSpec

    class _Store:
        def resolve(self, weights_id, *, configuration=None):
            return Path(weights_id)

    payload = _nested_zip(dataset="Dataset551_Totalseg251",
                          labels={"kidney_right": 2, "kidney_left": 3})
    eco = _expecting(CADSEcosystem(), payload, "organs")
    with _serve(payload):
        eco.ensure("organs", tmp_path)
    spec = eco.spec("organs", tmp_path)
    assert spec.lineage == "ts"
    assert set(spec.label_map.values()) == {"kidney_right", "kidney_left"}
    assert canonical_orientation_for(spec, _Store()) == nio.CANONICAL
    stock = TaskSpec.from_model_folder(spec.single)
    assert canonical_orientation_for(stock, _Store()) is None          # the mirror it prevents


def test_cads_makes_vertebrae_an_ambiguous_short_name(tmp_path):
    """`cads:vertebrae` collides with `totalvibe:vertebrae`, which the bare name used to mean.
    The CHANGELOG says so; the error names both, and the qualified names still resolve."""
    cat = EcosystemCatalog(root=tmp_path)
    with pytest.raises(LookupError, match="ambiguous.*cads:vertebrae.*totalvibe:vertebrae"):
        cat.resolve("vertebrae")
    assert cat.resolve("cads:vertebrae")[2] == "cads:vertebrae"
    assert cat.resolve("totalvibe:vertebrae")[2] == "totalvibe:vertebrae"


def test_totalvibe_orientation_is_read_from_the_checkpoint_not_hardcoded(tmp_path):
    """Upstream reorients per model while the plans declare a reader that does
    not reorient - MRSegmentator's LPS problem, except stated per model in the
    checkpoint's own dataset.json, so it is read."""
    payload = _bare_config_zip(orientation=["L", "P", "S"])
    eco = _expecting(TotalVibeEcosystem(), payload, "body_regions")
    with _serve(payload):
        eco.ensure("body_regions", tmp_path)
    assert eco.spec("body_regions", tmp_path).orientation == "LPS"
    assert eco.info("body_regions", tmp_path)["orientation"] == "LPS"

    other = _bare_config_zip(orientation=["R", "A", "S"])
    eco2 = _expecting(TotalVibeEcosystem(), other, "vibe")
    with _serve(other):
        eco2.ensure("vibe", tmp_path)
    assert eco2.spec("vibe", tmp_path).orientation == "RAS"


@pytest.mark.parametrize("declared", [None, ["X", "Y", "Z"], ["R", "L", "S"], "nonsense", []])
def test_an_orientation_that_is_not_one_is_ignored_rather_than_passed_on(tmp_path, declared):
    """Anything that is not three letters naming each anatomical axis once must
    not reach DICOMOrient; the spec then follows the model's declared reader."""
    payload = _bare_config_zip(orientation=declared)
    eco = _expecting(TotalVibeEcosystem(), payload, "vibe")
    with _serve(payload):
        eco.ensure("vibe", tmp_path)
    assert eco.spec("vibe", tmp_path).orientation is None


# --- shared base behavior ---------------------------------------------------

def test_the_published_digest_is_verified_whichever_one_it_is(tmp_path):
    """Zenodo publishes md5 and GitHub sha256; the installer verifies what the
    manifest carries, and a mismatch leaves nothing installed."""
    payload = _nested_zip()
    eco = DentalSegmentatorEcosystem()                  # the REAL md5
    assert "md5" in eco._entries["base"]
    with _serve(payload), pytest.raises(InputError, match="md5 digest mismatch"):
        eco.ensure("base", tmp_path)
    assert not eco.materialized("base", tmp_path)
    assert not (tmp_path / "dentalsegmentator" / "Dataset112_DentalSegmentator_v100").exists()

    vibe = TotalVibeEcosystem()
    assert re.fullmatch(r"[0-9a-f]{64}", vibe._entries["vibe"]["sha256"])
    with _serve(_bare_config_zip()), pytest.raises(InputError, match="sha256 digest mismatch"):
        vibe.ensure("vibe", tmp_path)


def test_a_pin_is_checked_against_the_installed_bytes(tmp_path):
    """The shared rule: matching the manifest tag is not proof the bytes on disk
    are that release."""
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    tag = eco._entries["base"]["tag"]
    with pytest.raises(ModelNotFound, match="offers tag"):
        eco.ensure("base", tmp_path, version="not-a-release")
    with _serve(payload):
        eco.ensure("base", tmp_path)
        eco.ensure("base", tmp_path, version=tag)       # sidecar agrees: no raise
    with pytest.raises(ModelNotFound, match="remove"):
        eco.ensure("base", tmp_path, version="v999")


def test_an_asset_that_unpacks_to_the_wrong_folder_names_the_generator(tmp_path):
    payload = _zip({"Dataset999_Other/x__y__z/dataset.json": "{}"})
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload), pytest.raises(ModelNotFound, match="gen_dentalsegmentator_manifest"):
        eco.ensure("base", tmp_path)


def test_the_base_declares_where_a_catalog_unpacks(tmp_path):
    """The one hook that separates the two packagings, asserted directly so a
    future catalog knows which it is choosing."""
    dental, vibe = DentalSegmentatorEcosystem(), TotalVibeEcosystem()
    assert dental._unpack_into("base", tmp_path) == tmp_path / "dentalsegmentator"
    assert vibe._unpack_into("vibe", tmp_path) == tmp_path / "totalvibe" / "Dataset100"
    assert issubclass(TotalVibeEcosystem, ZipManifestEcosystem)


def test_a_bare_config_catalog_still_checks_what_landed_inside_the_parent(tmp_path):
    """The Dataset parent is one this catalog creates, so its existence proves
    nothing - the check is that a resolvable nnU-Net model folder is inside it."""
    payload = _zip({"dataset.json": "{}", "fold_0/checkpoint_final.pth": "w"})  # a flat zip
    eco = _expecting(TotalVibeEcosystem(), payload, "vibe")
    with _serve(payload), pytest.raises(ModelNotFound, match="did not unpack.*gen_totalvibe"):
        eco.ensure("vibe", tmp_path)


def test_a_failed_install_leaves_nothing_and_keeps_saying_why(tmp_path):
    """The catalog whose zips unpack straight into the model folder is the one
    that can wedge: a wrong-shaped archive lands *inside* the directory that
    decides whether the task is installed. The actionable message has to survive
    a retry, not be produced once and then short-circuited away."""
    payload = _zip({"Dataset100/cfg__a__3d_fullres/dataset.json": "{}"})   # wrong shape
    eco = _expecting(TotalVibeEcosystem(), payload, "vibe")
    for _ in range(2):
        with _serve(payload), pytest.raises(ModelNotFound, match="did not unpack.*gen_totalvibe"):
            eco.ensure("vibe", tmp_path)
        assert not eco.materialized("vibe", tmp_path)
    assert not (tmp_path / "totalvibe" / "Dataset100").exists()


def test_an_interrupted_unpack_is_not_an_install(tmp_path):
    """The installer stages under a dot-prefixed directory. Staging now lives in
    the bucket rather than the model folder, but a tree left inside the model
    folder by an older build - or by anything else - must still not read as an
    install: `materialized()` looks for a configuration folder, and skips
    dot-prefixed names."""
    eco = TotalVibeEcosystem()
    staged = tmp_path / "totalvibe" / "Dataset100" / ".unzip-abc123" / CONFIG
    staged.mkdir(parents=True)
    (staged / "dataset.json").write_text("{}")
    (staged / "plans.json").write_text("{}")
    assert not eco.materialized("vibe", tmp_path)
    # and the finished article, for contrast
    done = tmp_path / "totalvibe" / "Dataset100" / CONFIG
    done.mkdir(parents=True)
    (done / "dataset.json").write_text("{}")
    assert eco.materialized("vibe", tmp_path)


def test_modality_does_not_change_when_the_weights_arrive(tmp_path):
    """A channel named "any" is a contrast claim, not a modality, so the curated
    one is applied on the SPEC - which is what describe() reads - and not only in
    info(), where it would flip from MR to "any" the moment a task installed."""
    payload = _bare_config_zip(orientation=["R", "A", "S"])
    eco = _expecting(TotalVibeEcosystem(), payload, "vibe")
    before = eco.info("vibe", tmp_path)["modality"]
    with _serve(payload):
        eco.ensure("vibe", tmp_path)
    assert before == "MR"
    assert eco.info("vibe", tmp_path)["modality"] == "MR"
    assert eco.spec("vibe", tmp_path).modality == "MR"     # what describe() reports


def test_a_failed_install_never_deletes_weights_it_did_not_create(tmp_path):
    """Cleanup removes only what the call created. A directory that was already
    there is the user's - weights installed by hand, or by an older version in a
    layout this one does not resolve - and a transient failure (an offline
    machine is enough) must not destroy data no retry can bring back."""
    eco = DentalSegmentatorEcosystem()
    folder = eco._folder("base", tmp_path)
    config = folder / "nnUNetTrainerV2__nnUNetPlansv2.1"        # a layout we cannot resolve
    config.mkdir(parents=True)
    (config / "checkpoint_final.pth").write_bytes(b"the user's own weights")
    (folder / "dataset.json").write_text("{}")
    assert not eco.materialized("base", tmp_path)               # and yet: not ours to delete

    def offline(*a, **k):
        raise OSError("network is unreachable")

    with mock.patch.object(fetchlib, "urlopen", offline), pytest.raises(Exception):
        eco.ensure("base", tmp_path)
    assert (config / "checkpoint_final.pth").read_bytes() == b"the user's own weights"


def test_an_archive_that_would_replace_another_task_is_refused(tmp_path):
    """Unpacking REPLACES a directory of the archive's own top-level name. With a
    stale manifest that is another task's weights, and the victim would then run
    this task's model with no error at all - so the names are checked before
    anything is moved."""
    victim = tmp_path / "dentalsegmentator" / "Dataset001_other" / "cfg__p__3d_fullres"
    victim.mkdir(parents=True)
    (victim / "checkpoint.pth").write_bytes(b"the other task's weights")
    payload = _zip({"Dataset001_other/cfg__p__3d_fullres/dataset.json": "{}",
                    "Dataset001_other/cfg__p__3d_fullres/checkpoint.pth": "wrong model"})
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload), pytest.raises(ModelNotFound, match="unpacks to.*gen_dentalsegmentator"):
        eco.ensure("base", tmp_path)
    assert (victim / "checkpoint.pth").read_bytes() == b"the other task's weights"


def test_a_real_install_in_an_unpreferred_layout_still_counts_as_installed(tmp_path):
    """`installed` must not mean `resolve_model_folder succeeds`: that also
    applies the configuration preference and rejects a folder shipping several
    configurations, none preferred. That is a real install, and reporting it
    absent would re-download it."""
    eco = DentalSegmentatorEcosystem()
    folder = eco._folder("base", tmp_path)
    for name in ("t__p__3d_fullres_bs8", "t__p__3d_cascade_fullres"):
        d = folder / name
        d.mkdir(parents=True)
        (d / "dataset.json").write_text("{}")
    assert eco.materialized("base", tmp_path)


def test_two_installs_of_one_task_serialize_and_the_second_downloads_nothing(tmp_path):
    """Two callers wanting one task at once is ordinary here - two prepares, a
    prepare racing an on-demand install, two workers on a shared volume.
    Unlocked, they interleave a destroy-then-move and the loser's failure cleanup
    deletes the winner's finished weights. Under the lock the second caller waits,
    finds the work done, and fetches nothing."""
    import threading
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    downloads, overlapping = [], []
    active = threading.Semaphore(1)

    def urlopen(*a, **k):
        if not active.acquire(blocking=False):
            overlapping.append(1)             # two downloads at once = no mutual exclusion
        else:
            time.sleep(0.05)
            active.release()
        downloads.append(1)
        return _Resp(payload)

    with mock.patch.object(fetchlib, "urlopen", urlopen):
        threads = [threading.Thread(target=lambda: results.append(_capture(eco.ensure, "base", tmp_path)))
                   for _ in range(2)]
        results = []
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
    assert [r for r in results if r] == []            # neither caller failed
    assert overlapping == []                          # the downloads did not overlap
    assert len(downloads) == 1                        # the second found the work done
    assert eco.materialized("base", tmp_path)


def _capture(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except BaseException as e:                        # noqa: BLE001 - the point is to catch all
        return type(e).__name__
    return None


def test_four_concurrent_installs_of_one_task_leave_exactly_one(tmp_path):
    """Unlocked, four at once left NOTHING installed while one of them reported
    success - the destroy-then-move in the unpacker interleaves."""
    import threading
    from haversack.weights_fetch import installed_version
    payload = _bare_config_zip()                       # the unpack-into-the-model-folder layout
    eco = _expecting(TotalVibeEcosystem(), payload, "vibe")
    errors = []
    with mock.patch.object(fetchlib, "urlopen", lambda *a, **k: _Resp(payload)):
        threads = [threading.Thread(target=lambda: errors.append(_capture(eco.ensure, "vibe", tmp_path)))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    assert [e for e in errors if e] == []
    assert eco.materialized("vibe", tmp_path)
    assert installed_version(eco._folder("vibe", tmp_path))
    assert not list((tmp_path / "totalvibe").glob(".unzip-*"))


def test_a_download_that_is_not_an_archive_is_an_error_not_a_traceback(tmp_path):
    """A host answering 200 with an HTML error page, a proxy interstitial or a
    truncated body is an ordinary failure. The download half was wrapped and the
    unpack half was not, so these surfaced as tracebacks - and one of the escaping
    types is NotImplementedError, which UnsupportedModel inherits, making a bad
    download indistinguishable from this package's own half-written marker."""
    from haversack.errors import HaversackError
    eco = DentalSegmentatorEcosystem()
    for payload in (b"<html>404 Not Found</html>", b"PK\x03\x04truncated", b""):
        eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
        with mock.patch.object(fetchlib, "urlopen", lambda *a, **k: _Resp(payload)):
            with pytest.raises(HaversackError, match="could not be unpacked"):
                eco.ensure("base", tmp_path)
        assert not eco.materialized("base", tmp_path)


def test_a_weights_download_says_who_is_asking(tmp_path):
    """The installer sent Python's default User-Agent, and the Cloudflare rule in
    front of MOOSE's one non-GitHub asset (``clin_ct_dental``) answers that with
    403 and anything named with 200 - so a live asset failed mid-install as if
    it were dead. The request must carry the package's name and version."""
    from haversack import __version__
    from haversack import ecosystems as eco_mod
    payload = _nested_zip(litter=False)
    seen = []

    def urlopen(req, *a, **k):
        seen.append(req)
        return _Resp(payload)

    with mock.patch.object(fetchlib, "urlopen", urlopen):
        eco_mod._download_and_extract_zip("http://h/w.zip", tmp_path)
    assert len(seen) == 1
    req = seen[0]
    assert isinstance(req, urllib.request.Request), "a bare URL is sent as Python-urllib"
    assert req.get_header("User-agent") == f"haversack/{__version__}"
    assert (tmp_path / "Dataset112_DentalSegmentator_v100").is_dir()


def test_a_checkpoint_whose_dataset_json_is_corrupt_is_not_installed(tmp_path):
    """`installed` meant the file existed, not that it parsed - so a corrupt
    archive installed "successfully", made materialized() true forever, and then
    raised JSONDecodeError out of every spec() and info() with no way back."""
    payload = _zip({"Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/"
                    "dataset.json": "this is not json",
                    "Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres/"
                    "fold_0/checkpoint_final.pth": "w"})
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload), pytest.raises(ModelNotFound, match="did not unpack"):
        eco.ensure("base", tmp_path)
    assert not eco.materialized("base", tmp_path)


@pytest.mark.parametrize("raw", [{}, {"tasks": None}, {"tasks": {}}, {"tasks": {"t": None}}])
def test_a_manifest_of_the_wrong_shape_says_so_at_construction(tmp_path, raw):
    """A manifest is data like any other input - hand-edited, or written by a
    generator against a changed upstream. A wrong shape surfaced as a KeyError or
    an AttributeError from three calls away."""
    m = tmp_path / "bad.json"
    m.write_text(json.dumps(raw))
    with pytest.raises(ModelNotFound):
        DentalSegmentatorEcosystem(manifest=m)


@pytest.mark.parametrize("entry", [{"url": None, "folder": "Dataset1"},
                                   {"url": "https://h/a.zip"}])
def test_one_bad_manifest_entry_costs_one_task_not_the_process(tmp_path, entry):
    """Per-entry fields are checked when the task is USED. Every catalog is
    constructed by default_ecosystems(), so raising at construction took down
    `haversack tasks`, the server and the Modal worker over a fault in one
    entry - and the check has to reach MRSegmentator too, which does not sit on
    the shared base."""
    m = tmp_path / "one-bad.json"
    m.write_text(json.dumps({"tasks": {"base": entry}}))
    eco = DentalSegmentatorEcosystem(manifest=m)          # construction survives
    assert eco.tasks() == ["base"]
    with pytest.raises(ModelNotFound, match="no usable"):
        eco.ensure("base", tmp_path)


def test_a_manifest_folder_that_escapes_the_bucket_is_refused_for_every_catalog(tmp_path):
    """The value is joined onto the weights root and then handed to file
    operations, including an rmtree on the failure path, and it comes from a
    generator reading a remote archive's own directory names. MRSegmentator does
    not sit on the shared base, and when this check lived there it did not apply
    to it at all."""
    from haversack.ecosystems import MRSegmentatorEcosystem
    for cls, task in ((DentalSegmentatorEcosystem, "base"), (TotalVibeEcosystem, "vibe"),
                      (MRSegmentatorEcosystem, "base")):
        eco = cls()
        for bad in ("/tmp/ABSOLUTE", "../../ESCAPED", ".", "", "a/../..", ".install-x"):
            eco._entries[task] = {**eco._entries[task], "folder": bad}
            with pytest.raises(ModelNotFound, match="not a relative path|no usable"):
                eco._folder(task, tmp_path)


def test_a_download_larger_than_the_ceiling_is_refused(tmp_path, monkeypatch):
    """Most manifest entries carry no digest, and a release asset can be replaced
    under a published tag, so a download with nothing to check it against needs a
    ceiling. The ceiling is read at call time: parsed at import, a typo in the
    environment variable took down the CLI, the server and the Modal worker."""
    from haversack import ecosystems as eco_mod
    from haversack.errors import InputError

    class _Big(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setenv("HAVERSACK_MAX_WEIGHTS_GB", "0.000001")     # ~1 KB, after the 1 GB floor
    assert eco_mod._weights_cap() >= 1 << 30                       # a floor, so this cannot bite
    monkeypatch.setenv("HAVERSACK_MAX_WEIGHTS_GB", "not-a-number")
    assert eco_mod._weights_cap() == 24 * (1 << 30)                # a typo is not fatal
    monkeypatch.delenv("HAVERSACK_MAX_WEIGHTS_GB")

    monkeypatch.setattr(eco_mod, "_weights_cap", lambda: 1024)
    with mock.patch.object(fetchlib, "urlopen", lambda *a, **k: _Big(b"x" * 8192)):
        with pytest.raises(InputError, match="over the .* byte cap"):
            eco_mod._download_and_extract_zip("http://h/w.zip", tmp_path)
    assert not list(tmp_path.glob("*.zip"))                        # the part-file is gone


def test_an_md5_verified_install_records_the_digest_it_verified(tmp_path):
    """Zenodo publishes md5 and nothing else. Recording it under "sha256" made
    provenance claim a hash it is not; recording nothing made a verified install
    read as unverified."""
    from haversack.weights_fetch import installed_version
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload):
        eco.ensure("base", tmp_path)
    rec = installed_version(eco._folder("base", tmp_path))
    assert rec["md5"] == eco._entries["base"]["md5"]
    assert rec["sha256"] is None


def test_a_non_directory_in_the_way_is_replaced_rather_than_erroring(tmp_path):
    """rmtree silently no-ops on a file or a symlink, so os.replace then raised a
    NotADirectoryError out of the installer. Symlinking a model folder at a
    shared read-only weights volume is a plausible operations layout."""
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    bucket = tmp_path / "dentalsegmentator"
    bucket.mkdir(parents=True)
    (bucket / "Dataset112_DentalSegmentator_v100").write_text("a file, not a directory")
    with _serve(payload):
        eco.ensure("base", tmp_path)
    assert eco.materialized("base", tmp_path)


def test_describe_degrades_and_carries_the_label_map(tmp_path):
    """Two things a caller needs from describe(): the label -> name mapping, since
    labels are neither contiguous nor alphabetical; and, when the weights are
    installed in a shape this build cannot choose among, the resolver's own
    remedy rather than an exception that reads as "unknown task"."""
    from haversack import Segmenter
    payload = _nested_zip()
    eco = _expecting(DentalSegmentatorEcosystem(), payload, "base")
    with _serve(payload):
        eco.ensure("base", tmp_path)
    seg = Segmenter(weights=tmp_path, device="cpu")
    d = seg.describe("dentalsegmentator:base")
    assert d["label_map"] == {"1": "mandible", "2": "upper_teeth"}
    assert seg.structures("dentalsegmentator:base") == ["mandible", "upper_teeth"]

    # now make it unresolvable: several configurations, none preferred
    folder = eco._folder("base", tmp_path)
    for c in list(folder.iterdir()):
        if c.is_dir() and c.name.count("__") == 2:
            c.rename(folder / "t__p__3d_fullres_bs8")
    (folder / "t__p__3d_cascade_fullres").mkdir()
    (folder / "t__p__3d_cascade_fullres" / "dataset.json").write_text(
        (folder / "t__p__3d_fullres_bs8" / "dataset.json").read_text(encoding="utf-8"))
    seg2 = Segmenter(weights=tmp_path, device="cpu")
    d2 = seg2.describe("dentalsegmentator:base")
    assert "no runnable configuration" in d2["unresolved"]
    assert seg2.structures("dentalsegmentator:base") == []      # not a KeyError


def test_installed_considers_the_folder_the_loader_will_pick(tmp_path):
    """`_resolved` must consider exactly the folders `resolve_model_folder`
    considers. Requiring a dataset.json to be a candidate hid a configuration
    that the loader then chose anyway: materialized() said yes, every spec()
    raised, and ensure() returned early so nothing ever repaired it."""
    good = json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "a": 1},
                       "numTraining": 1, "file_ending": ".nii.gz"})
    eco = DentalSegmentatorEcosystem()
    folder = eco._folder("base", tmp_path)
    (folder / "nnUNetTrainer__nnUNetPlans__2d").mkdir(parents=True)
    (folder / "nnUNetTrainer__nnUNetPlans__2d" / "dataset.json").write_text(good)
    (folder / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir(parents=True)   # no dataset.json
    assert not eco.materialized("base", tmp_path), \
        "the preferred configuration is unusable, so this is not an install"

    # and the preferred one being CORRUPT is the same answer
    (folder / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json").write_text("{not json")
    assert not eco.materialized("base", tmp_path)
    # while a usable preferred one is
    (folder / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json").write_text(good)
    assert eco.materialized("base", tmp_path)


def test_mrsegmentator_also_refuses_a_checkpoint_that_does_not_parse(tmp_path):
    """The corrupt-JSON trap was closed on the shared base, which MRSegmentator
    does not sit on - the same shape as the folder check that had to become a
    free function."""
    from haversack.ecosystems import MRSegmentatorEcosystem
    eco = MRSegmentatorEcosystem()
    folder = eco._folder("base", tmp_path)
    folder.mkdir(parents=True)
    (folder / "dataset.json").write_text("{not json")
    assert not eco.materialized("base", tmp_path)
    (folder / "dataset.json").write_text('{"channel_names": {"0": "MR"}, "labels": {}}')
    assert eco.materialized("base", tmp_path)


def test_mrsegmentator_stages_where_the_debris_sweep_can_find_it(tmp_path):
    """Its staging directory IS a `.install-*`, so with no work_dir the sweep ran
    inside the debris rather than beside it, and a SIGKILL stranded 1.2 GB that
    nothing would ever collect."""
    import inspect
    from haversack.ecosystems import MRSegmentatorEcosystem
    src = inspect.getsource(MRSegmentatorEcosystem.ensure)
    assert "work_dir=folder.parent" in src


def test_a_manifest_folder_may_not_name_our_own_scratch(tmp_path):
    """`.unzip-*`, `.install-*` and `.lock-*` are the installer's namespace; a
    model installed into one would be deleted by the debris sweep hours later."""
    from haversack.ecosystems import catalog_folder
    for bad in (".install-x", ".unzip-y", "Dataset1/.lock-z"):
        with pytest.raises(ModelNotFound, match="not a relative path"):
            catalog_folder(tmp_path, "moose", bad)
