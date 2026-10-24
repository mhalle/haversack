"""The `lnq` catalog (2026-10-22): Steve Pieper's lymph-node models, installed from release zips
that hold whole training checkpoints, by reading each fold's network weights in place.

Offline: a fake release - a flat meta.zip and one STORED zip per fold holding a checkpoint
shaped like LNQ's (weights, the optimizer's momentum, a numpy training log) - served through
`fetchlib.urlopen` with HTTP Range honored. The manifest is written by the generator's own
`entry_for`, so the generator and the install are held to each other.

The oracle for what installs is nnU-Net's own ensemble over the ORIGINAL checkpoints: the
installed weights-only folder must predict what upstream's folder predicts.
"""
import collections
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")
pytest.importorskip("dynamic_network_architectures")

from haversack import ecosystems as E
from haversack.errors import InputError, ModelNotFound

import test_multifold_surgery as MF

REPO = Path(__file__).resolve().parents[1]
BASE = "https://example.invalid/releases/download/tiny-v1-1.0.0/"
FOLDS = (0, 1, 2)


def _generator():
    spec = importlib.util.spec_from_file_location("gen_lnq_manifest",
                                                  REPO / "tools" / "gen_lnq_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zip(members: dict, method=zipfile.ZIP_STORED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _full_checkpoint(state: dict, *, configuration="3d_fullres") -> bytes:
    """What nnU-Net writes at the end of training: the weights and everything else."""
    momentum = {i: torch.randn_like(t) for i, t in enumerate(v for k, v in state.items()
                                                            if not k.startswith("decoder.encoder."))}
    buf = io.BytesIO()
    torch.save({"network_weights": state,
                "optimizer_state": {"state": {i: {"momentum_buffer": m} for i, m in momentum.items()},
                                    "param_groups": [{"lr": 1e-5, "momentum": 0.99}]},
                "grad_scaler_state": {"scale": 65536.0},
                "logging": {"mean_fg_dice": [__import__("numpy").float32(0.7)]},
                "_best_ema": 0.7, "current_epoch": 1000, "trainer_name": "nnUNetTrainer",
                "init_args": {"configuration": configuration, "fold": 0,
                              "device": torch.device("cpu")},
                "inference_allowed_mirroring_axes": (0, 1, 2)}, buf)
    return buf.getvalue()


class Server:
    """URL -> bytes, as fetchlib.urlopen sees them: whole, or a Range (206)."""

    def __init__(self, files: dict):
        self.files, self.served = dict(files), collections.Counter()

    def urlopen(self, req, timeout=None):
        url = req.full_url
        if url not in self.files:
            raise AssertionError(f"unexpected URL {url}")
        data, status = self.files[url], 200
        rng = req.get_header("Range")
        if rng:
            lo, hi = (int(x) for x in re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups())
            data, status = data[lo:hi + 1], 206
        self.served[url] += len(data)
        r = io.BytesIO(data)
        r.status, r.headers = status, {"Content-Length": str(len(data))}
        return r


def build_release(tmp: Path):
    """A tiny ResEnc 3-fold model as LNQ publishes one, the files a server holds, and the
    original folder (the oracle)."""
    original = MF.make_model_folder(tmp / "orig")
    files = {}
    meta = {"dataset.json": json.dumps(MF.DATASET), "plans.json": json.dumps(MF.PLANS),
            "dataset_fingerprint.json": "{}", "card.json": json.dumps({"license": "CC BY-NC 4.0"})}
    files[BASE + "tiny-v1-1.0.0-meta.zip"] = _zip(meta)
    for f in FOLDS:
        state = torch.load(original / f"fold_{f}" / "checkpoint_final.pth",
                           weights_only=True)["network_weights"]
        files[BASE + f"tiny-v1-1.0.0-fold{f}.zip"] = _zip(
            {f"tiny-v1/fold_{f}/checkpoint_final.pth": _full_checkpoint(state)})
    return original, files


def registry_model(files: dict) -> dict:
    import hashlib
    assets = [{"role": "meta" if "meta" in u else f"fold_{u[-5]}", "filename": u.rsplit("/", 1)[1],
               "sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}
              for u, b in files.items()]
    return {"name": "tiny-v1", "version": "1.0.0", "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/", "modality": "CT",
            "trainer_class": "nnUNetTrainer", "plans_identifier": "nnUNetPlans",
            "configuration": "3d_fullres", "folds": list(FOLDS),
            "recommended_use": "a test", "segment_name": "Tiny", "snomed_code": "1",
            "snomed_term": "t", "color": [1, 2, 3], "attribution": "tiny",
            "weights_url_template": BASE + "{filename}", "weights_assets": assets}


class _Release(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.original, cls.files = build_release(cls.tmp)
        cls.gen = _generator()
        server = Server(cls.files)
        with mock.patch("haversack.fetchlib.urlopen", server.urlopen):
            work = cls.tmp / "gen"
            work.mkdir()
            cls.entry = cls.gen.entry_for(registry_model(cls.files), {}, work)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.manifest = self.root / "lnq_weights.json"
        self.manifest.write_text(json.dumps({"tasks": {"tiny-v1": self.entry}}))
        self.eco = E.LNQEcosystem(manifest=self.manifest)
        # read sizes scaled to a 190 KB fixture, as 64 KB / 8 MB are to an 815 MB fold
        for name, size in (("_INDEX_BLOCK", 1024), ("_STORAGE_BLOCK", 4096)):
            # on the base, so every catalog of the packaging reads at the scaled sizes (review)
            patch = mock.patch.object(E.ZipManifestEcosystem, name, size)
            patch.start()
            self.addCleanup(patch.stop)
        self.server = Server(self.files)
        patcher = mock.patch("haversack.fetchlib.urlopen", self.server.urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)

    def install(self, **kw):
        self.eco.ensure("tiny-v1", self.root, **kw)
        return self.eco._folder("tiny-v1", self.root)


class TheGeneratorDescribesTheRelease(_Release):
    def test_the_entry_names_the_model_and_every_fold(self):
        e = self.entry
        self.assertEqual(e["folder"], "Dataset990_Tiny/nnUNetTrainer__nnUNetPlans__3d_fullres")
        self.assertEqual(sorted(e["folds"]), ["0", "1", "2"])
        self.assertEqual(e["license"], "CC-BY-4.0")
        self.assertEqual(e["dataset_id"], "990")
        for f in e["folds"].values():
            self.assertEqual(f["member"], f"tiny-v1/fold_{f['url'][-5]}/checkpoint_final.pth")
            self.assertRegex(f["weights_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len({f["weights_sha256"] for f in e["folds"].values()}), 3)

    def test_a_zip_that_is_not_the_registrys_stops_it(self):
        model = registry_model(self.files)
        model["weights_assets"][1]["sha256"] = "0" * 64
        work = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, work, True)
        with self.assertRaises(SystemExit):
            self.gen.entry_for(model, {}, work)

    def test_a_registry_naming_another_model_stops_it(self):
        model = registry_model(self.files)
        model["plans_identifier"] = "nnUNetResEncUNetMPlans"
        work = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, work, True)
        with self.assertRaisesRegex(SystemExit, "plans_identifier"):
            self.gen.entry_for(model, {}, work)


class TheInstall(_Release):
    def test_it_is_a_stock_model_folder_with_every_fold(self):
        folder = self.install()
        self.assertTrue(self.eco.materialized("tiny-v1", self.root))
        self.assertEqual(folder, self.root / "lnq" / self.entry["folder"])
        for name in ("dataset.json", "plans.json", "card.json"):    # upstream's, as published
            self.assertTrue((folder / name).is_file(), name)
        self.assertEqual(sorted(p.name for p in folder.iterdir() if p.name.startswith("fold_")),
                         ["fold_0", "fold_1", "fold_2"])
        self.assertEqual([p.name for p in folder.parent.iterdir() if p.name.startswith(".")], [])
        spec = self.eco.spec("tiny-v1", self.root)
        self.assertEqual(spec.folds, FOLDS)
        self.assertEqual(spec.lineage, "nnunetv2")
        self.assertEqual(dict(spec.label_map), {1: "a", 2: "b"})
        from haversack.weights_fetch import installed_version
        self.assertEqual(installed_version(folder)["tag"], "1.0.0")

    def test_each_fold_is_its_weights_alone_loadable_without_a_full_unpickle(self):
        folder = self.install()
        for f in FOLDS:
            ck = torch.load(folder / f"fold_{f}" / "checkpoint_final.pth", weights_only=True)
            self.assertNotIn("optimizer_state", ck)
            self.assertEqual(ck["init_args"], {"configuration": "3d_fullres"})
            orig = torch.load(self.original / f"fold_{f}" / "checkpoint_final.pth",
                              weights_only=True)["network_weights"]
            self.assertEqual(list(ck["network_weights"]), list(orig))
            for k in orig:
                self.assertTrue(torch.equal(ck["network_weights"][k], orig[k]), k)
            w = ck["network_weights"]
            aliases = [k for k in w if k.startswith("decoder.encoder.")]
            self.assertTrue(aliases)
            for k in aliases:                 # the encoder's two names stay one storage
                self.assertEqual(w[k].untyped_storage().data_ptr(),
                                 w[k[len("decoder."):]].untyped_storage().data_ptr())

    def test_the_optimizer_state_is_not_fetched(self):
        self.install()
        for f in FOLDS:
            url = BASE + f"tiny-v1-1.0.0-fold{f}.zip"
            self.assertLess(self.server.served[url], 0.75 * len(self.files[url]),
                            f"fold {f}: fetched {self.server.served[url]} of {len(self.files[url])}")

    def test_it_predicts_what_upstreams_folder_predicts(self):
        folder = self.install()
        x = MF.preprocessed()
        want = MF.stock_logits(self.original, FOLDS, x)
        _, got = MF.haversack_logits(folder, FOLDS, x, surgery=False)
        self.assertTrue(torch.allclose(got, want, atol=1e-4), float((got - want).abs().max()))

    def test_a_pinned_version_is_held_to(self):
        self.install(version="1.0.0")
        self.eco.ensure("tiny-v1", self.root, version="1.0.0")
        with self.assertRaises(ModelNotFound):
            self.eco.ensure("tiny-v1", self.root, version="2.0.0")


class TheInstallRefuses(_Release):
    def _assert_nothing_installed(self):
        self.assertFalse(self.eco.materialized("tiny-v1", self.root))
        lnq = self.root / "lnq"
        leftovers = [p for p in lnq.rglob("*") if p.name.startswith(".install-")] if lnq.exists() else []
        self.assertEqual(leftovers, [])
        self.assertFalse((lnq / self.entry["folder"]).exists())

    def test_weights_that_are_not_the_manifests(self):
        self.entry["folds"]["1"]["weights_sha256"], old = "0" * 64, self.entry["folds"]["1"]["weights_sha256"]
        try:
            self.manifest.write_text(json.dumps({"tasks": {"tiny-v1": self.entry}}))
            eco = E.LNQEcosystem(manifest=self.manifest)
            with self.assertRaisesRegex(InputError, "fold 1: network weights digest"):
                eco.ensure("tiny-v1", self.root)
        finally:
            self.entry["folds"]["1"]["weights_sha256"] = old
        self._assert_nothing_installed()

    def test_a_checkpoint_of_another_configuration(self):
        url = BASE + "tiny-v1-1.0.0-fold2.zip"
        with zipfile.ZipFile(io.BytesIO(self.files[url])) as zf:
            ck = torch.load(io.BytesIO(zf.read("tiny-v1/fold_2/checkpoint_final.pth")),
                            weights_only=False)
        self.server.files[url] = _zip({"tiny-v1/fold_2/checkpoint_final.pth":
                                       _full_checkpoint(ck["network_weights"],
                                                        configuration="3d_lowres")})
        entry = json.loads(json.dumps(self.entry))
        entry["folds"]["2"]["size"] = len(self.server.files[url])
        self.manifest.write_text(json.dumps({"tasks": {"tiny-v1": entry}}))
        with self.assertRaisesRegex(ModelNotFound, "3d_lowres"):
            E.LNQEcosystem(manifest=self.manifest).ensure("tiny-v1", self.root)
        self._assert_nothing_installed()

    def test_a_meta_zip_that_is_not_the_manifests(self):
        url = self.entry["url"]
        data = self.files[url]
        self.server.files[url] = data[:40] + bytes([data[40] ^ 1]) + data[41:]   # one bit
        with self.assertRaises(InputError):
            self.eco.ensure("tiny-v1", self.root)
        self._assert_nothing_installed()


class TheEnsembleIsTheDefault(_Release):
    """A task whose catalog states its folds runs all of them unless the caller says otherwise,
    and says so in its result key; nothing else's key moves."""

    def _segmenter(self, **kw):
        from haversack.segmenter import Segmenter
        catalog = E.EcosystemCatalog([self.eco], root=self.root)
        return Segmenter(weights=self.root, catalog=catalog, device="cpu", dtype="fp32", **kw)

    def _image(self):
        import numpy as np
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(np.random.default_rng(3).normal(40, 30, (13, 37, 29))
                                     .astype("float32"))
        img.SetSpacing((1.0, 1.0, 2.0))
        return img

    def test_describe_and_the_key_name_the_ensemble_before_and_after_install(self):
        from haversack.serve import weights_versions_of
        seg = self._segmenter()
        before = seg.describe("lnq:tiny-v1")
        self.assertEqual(before["folds_default"], [0, 1, 2])
        self.assertIn("folds=0,1,2", weights_versions_of(seg, "lnq:tiny-v1"))
        self.install()
        after = seg.describe("lnq:tiny-v1")
        self.assertEqual(after["folds_default"], [0, 1, 2])
        self.assertEqual(after["folds"], [0, 1, 2])
        self.assertIn("folds=0,1,2", weights_versions_of(seg, "lnq:tiny-v1"))

    def test_a_run_uses_every_fold_and_a_callers_choice_wins(self):
        self.install()
        seg = self._segmenter()
        out = seg.segment(self._image(), "lnq:tiny-v1")
        self.assertEqual(out.provenance["models"][0]["folds"], [0, 1, 2])
        out = seg.segment(self._image(), "lnq:tiny-v1", folds=(1,))
        self.assertEqual(out.provenance["models"][0]["folds"], [1])
        seg1 = self._segmenter(folds=(2,))
        self.assertEqual(seg1.describe("lnq:tiny-v1")["folds_default"], [2])
        out = seg1.segment(self._image(), "lnq:tiny-v1")
        self.assertEqual(out.provenance["models"][0]["folds"], [2])

    def test_a_task_that_states_no_folds_still_runs_fold_0_and_keys_as_before(self):
        from haversack.segmenter import Segmenter
        from haversack.serve import weights_versions_of
        from haversack.tasks import TaskSpec
        folder = self.original
        seg = Segmenter(weights=self.root, device="cpu", dtype="fp32",
                        catalog=E.EcosystemCatalog([self.eco], root=self.root))
        out = seg.segment(self._image(), TaskSpec.from_model_folder(folder))
        self.assertEqual(out.provenance["models"][0]["folds"], [0])
        self.assertEqual(seg.describe(TaskSpec.from_model_folder(folder))["folds_default"], [0])
        self.assertFalse(any(v.startswith("folds=") for v in
                             weights_versions_of(seg, TaskSpec.from_model_folder(folder))))


class WhatTheMutantsFoundInTheInstall(_Release):
    """Each written to kill a mutant the first tests let through (2026-10-22)."""

    # L3: a folder missing one fold's checkpoint is not materialized
    def test_a_folder_missing_a_fold_is_not_installed(self):
        folder = self.install()
        (folder / "fold_1" / "checkpoint_final.pth").unlink()
        self.assertFalse(self.eco.materialized("tiny-v1", self.root))

    # L8: a pin to a tag the manifest does not offer is refused before anything is fetched
    def test_a_pin_to_another_tag_is_refused_before_install(self):
        with self.assertRaisesRegex(ModelNotFound, "offers tag"):
            self.eco.ensure("tiny-v1", self.root, version="2.0.0")
        self.assertFalse(self.eco.materialized("tiny-v1", self.root))
        self.assertEqual(sum(self.server.served.values()), 0)

    # L10: the installed checkpoint keeps the mirroring axes nnU-Net's predictor reads
    def test_the_installed_checkpoint_keeps_its_mirroring_axes(self):
        folder = self.install()
        for f in FOLDS:
            ck = torch.load(folder / f"fold_{f}" / "checkpoint_final.pth", weights_only=True)
            self.assertEqual(tuple(ck["inference_allowed_mirroring_axes"]), (0, 1, 2))

    # P6: warm loads the ensemble a run will use, so the run reuses it
    def test_warm_loads_the_ensemble_the_run_uses(self):
        from haversack.segmenter import Segmenter
        from haversack import ecosystems as E
        self.install()
        catalog = E.EcosystemCatalog([self.eco], root=self.root)
        seg = Segmenter(weights=self.root, catalog=catalog, device="cpu", dtype="fp32",
                        cache_models=4)
        self.assertEqual(seg.warm("lnq:tiny-v1"), 1)
        img = TheEnsembleIsTheDefault._image(self)
        seg.segment(img, "lnq:tiny-v1")
        self.assertEqual(len(seg.models), 1, "the run loaded a model warm() had not")


class WhatTheMutantsFoundInTheGenerator(_Release):
    # G3: a registry listing one model twice stops the generator
    def test_a_registry_listing_a_model_twice_stops_it(self):
        model = registry_model(self.files)
        registry = {"schema": "lnq-segmenter-registry/1", "project": {},
                    "models": [model, dict(model, version="2.0.0")]}
        out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out, True)
        with mock.patch.object(self.gen.fetchlib, "get_json", return_value=registry), \
             mock.patch.object(sys, "argv", ["gen", "--out", str(out / "m.json")]):
            with self.assertRaisesRegex(SystemExit, "more than once"):
                self.gen.main()
        self.assertFalse((out / "m.json").exists())

    # G4: a fold zip with two checkpoints stops the generator rather than picking one
    def test_a_fold_zip_with_two_checkpoints_stops_it(self):
        url = BASE + "tiny-v1-1.0.0-fold0.zip"
        with zipfile.ZipFile(io.BytesIO(self.files[url])) as zf:
            pth = zf.read("tiny-v1/fold_0/checkpoint_final.pth")
        work = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, work, True)
        p = work / "two.zip"
        p.write_bytes(_zip({"a/fold_0/checkpoint_final.pth": pth,
                              "b/fold_0/checkpoint_final.pth": pth}))
        with self.assertRaisesRegex(SystemExit, "2 checkpoint_final.pth"):
            self.gen.describe_fold(p)


class AnyZipCatalogGetsTheFoldCheckpointInstall(_Release):
    """The packaging belongs to ZipManifestEcosystem, not to LNQ (2026-10-23): a catalog that
    declares nothing but data - name, bucket, manifest - installs the same release the same way."""

    def test_a_catalog_of_data_only_installs_and_states_its_folds(self):
        class Other(E.ZipManifestEcosystem):
            name, bucket, generator = "other", "other", "tools/none.py"
        eco = Other(manifest=self.manifest)
        eco.ensure("tiny-v1", self.root)
        folder = eco._folder("tiny-v1", self.root)
        self.assertEqual(folder, self.root / "other" / self.entry["folder"])
        self.assertTrue(eco.materialized("tiny-v1", self.root))
        self.assertEqual(eco.spec("tiny-v1", self.root).folds, FOLDS)
        self.assertEqual(eco.info("tiny-v1", self.root)["folds"], list(FOLDS))
        for f in FOLDS:
            ck = torch.load(folder / f"fold_{f}" / "checkpoint_final.pth", weights_only=True)
            self.assertNotIn("optimizer_state", ck)

    def test_an_entry_without_folds_installs_as_it_always_did(self):
        self.assertFalse(any("folds" in e for eco in (E.MooseEcosystem(), E.CADSEcosystem())
                             for e in eco._entries.values()))
        self.assertEqual(E.MooseEcosystem()._folds(next(iter(E.MooseEcosystem().tasks()))), ())


class Other(E.ZipManifestEcosystem):
    """A catalog of the fold-checkpoint packaging that is not LNQ: data only."""
    name, bucket, generator, modality = "other", "other", "tools/none.py", "CT"


class TheFoldCheckpointEntryIsCheckedAndGeneral(_Release):
    """From the 2026-10-23 review of the generalization: shapes LNQ never had, malformed entries,
    and the rules the whole-zip packaging already kept."""

    def _eco(self, **tasks):
        path = self.root / "other.json"
        path.write_text(json.dumps({"tasks": tasks}))
        return Other(manifest=path)

    def _leftovers(self):
        bucket = self.root / "other"
        return sorted(str(p.relative_to(bucket)) for p in bucket.rglob("*")
                      if not p.name.startswith(".lock-")) if bucket.exists() else []

    def test_fold_all_installs_runs_and_keys_as_all(self):
        from haversack.segmenter import Segmenter
        from haversack.serve import weights_versions_of
        entry = json.loads(json.dumps(self.entry))
        entry["folds"] = {"all": entry["folds"]["0"]}
        entry.pop("weights_sha256")                  # derived by the install when absent
        eco = self._eco(**{"tiny-v1": entry})
        eco.ensure("tiny-v1", self.root)
        folder = eco._folder("tiny-v1", self.root)
        self.assertTrue((folder / "fold_all" / "checkpoint_final.pth").is_file())
        self.assertEqual(eco.spec("tiny-v1", self.root).folds, ("all",))
        from haversack.weights_fetch import installed_version
        self.assertEqual(installed_version(folder)["sha256"],
                         E.fold_checkpoints_digest(entry["folds"]))
        seg = Segmenter(weights=self.root, catalog=E.EcosystemCatalog([eco], root=self.root),
                        device="cpu", dtype="fp32")
        out = seg.segment(TheEnsembleIsTheDefault._image(self), "other:tiny-v1")
        self.assertEqual(out.provenance["models"][0]["folds"], ["all"])
        self.assertIn("folds=all", weights_versions_of(seg, "other:tiny-v1"))

    def _malformed(self):
        # each its own folder: sharing the good entry's would (rightly) refuse both
        e = lambda **kw: dict(json.loads(json.dumps(self.entry)), **{
            "folder": "Dataset992_Bad/nnUNetTrainer__nnUNetPlans__3d_fullres", **kw})
        folds = self.entry["folds"]
        no_url = {k: {n: v for n, v in f.items() if n != "url"} for k, f in folds.items()}
        return {
            "a-list": (e(folds=[1, 2]), "non-empty object"),
            "empty": (e(folds={}), "non-empty object"),
            "all-and-numbers": (e(folds={"all": folds["0"], "1": folds["1"]}), "alone"),
            "no-url": (e(folds=no_url), "states no url"),
            "one-part-folder": (e(folder="nnUNetTrainer__nnUNetPlans__3d_fullres"), "'folder' must be"),
        }

    def test_a_malformed_entry_is_refused_by_name_and_costs_only_itself(self):
        for name, (entry, why) in self._malformed().items():
            with self.subTest(entry=name):
                eco = self._eco(**{"good": self.entry, name: entry})
                self.assertIn(why, eco.info(name, self.root).get("unresolved", ""))
                self.assertEqual(eco.info("good", self.root)["folds"], list(FOLDS))
                self.assertFalse(eco.materialized(name, self.root))
                with self.assertRaisesRegex(ModelNotFound, why.replace("'", ".")):
                    eco.ensure(name, self.root)
                self.assertEqual(self._leftovers(), [])

    def test_two_tasks_of_one_folder_are_refused(self):
        eco = self._eco(a=self.entry, b=self.entry)
        with self.assertRaisesRegex(ModelNotFound, "same folder as \\['b'\\]"):
            eco.ensure("a", self.root)

    def test_a_folder_it_did_not_make_is_kept_not_deleted(self):
        eco = self._eco(**{"tiny-v1": self.entry})
        folder = eco._folder("tiny-v1", self.root)
        folder.mkdir(parents=True)
        (folder / "dataset.json").write_text("{}")
        (folder / "MY_NOTES.txt").write_text("mine")
        with self.assertRaisesRegex(ModelNotFound, "exists but is not a complete install"):
            eco.ensure("tiny-v1", self.root)
        self.assertEqual((folder / "MY_NOTES.txt").read_text(), "mine")

    def test_a_nested_json_zip_is_refused_by_the_flat_rule_and_leaves_nothing(self):
        import hashlib
        nested = _zip({"x/dataset.json": json.dumps(MF.DATASET), "x/plans.json": json.dumps(MF.PLANS)})
        self.server.files[self.entry["url"]] = nested
        entry = dict(self.entry, sha256=hashlib.sha256(nested).hexdigest())
        eco = self._eco(**{"tiny-v1": entry})
        with self.assertRaisesRegex(ModelNotFound, "flat"):
            eco.ensure("tiny-v1", self.root)
        self.assertEqual(self._leftovers(), [])

    def test_a_pin_refusal_names_the_whole_folder(self):
        eco = self._eco(**{"tiny-v1": self.entry})
        eco.ensure("tiny-v1", self.root)
        with self.assertRaises(ModelNotFound) as cm:
            eco.ensure("tiny-v1", self.root, version="9.9")
        self.assertIn(str(eco._folder("tiny-v1", self.root)), str(cm.exception))

    def test_one_dataset_is_one_lock_whatever_its_tasks_folders(self):
        other = dict(self.entry, folder=self.entry["folder"].replace("3d_fullres", "3d_lowres"))
        eco = self._eco(a=self.entry, b=other, c=dict(self.entry, folder="Dataset991_Else/x__y__z"))
        held = []
        import contextlib
        with mock.patch("haversack.filelock.held", lambda p: (held.append(p.name), contextlib.nullcontext())[1]):
            for t in ("a", "b", "c"):
                with eco._install_lock(t, self.root):
                    pass
        self.assertEqual(held[0], held[1])
        self.assertNotEqual(held[0], held[2])

    def test_a_listing_states_the_catalogs_modality_where_the_entry_states_none(self):
        entry = {k: v for k, v in self.entry.items() if k != "modality"}
        eco = self._eco(**{"tiny-v1": entry})
        # a channel named for its normalization, not a modality: the listing must say what
        # info() says - the catalog's - not the channel name
        meta = _zip({"dataset.json": json.dumps(dict(MF.DATASET, channel_names={"0": "noNorm"})),
                     "plans.json": json.dumps(MF.PLANS)})
        listing = eco.label_listing("tiny-v1", None, _ZipReader({entry["url"]: meta}))
        self.assertEqual(listing["modality"], "CT")
        self.assertEqual(eco.info("tiny-v1", None)["modality"], "CT")

    def test_an_empty_folds_entry_is_never_read_as_the_other_packaging(self):
        """``folds: {}`` belongs to this packaging and is refused - it must not fall through to
        the whole-zip check, which would take a stray config folder under it as an install."""
        entry = dict(self.entry, folds={})
        eco = self._eco(**{"tiny-v1": entry})
        stray = eco._folder("tiny-v1", self.root) / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        stray.mkdir(parents=True)
        (stray / "dataset.json").write_text(json.dumps(MF.DATASET))
        self.assertFalse(eco.materialized("tiny-v1", self.root))


class _ZipReader:
    def __init__(self, files):
        self.files = files

    def zip(self, url):
        return zipfile.ZipFile(io.BytesIO(self.files[url]))

