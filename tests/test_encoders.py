"""The encoders (2026-09-23): the registry, pinned weights, the RADAR family end to end on random
weights, the nnU-Net family's tiling, and the two commands.

The real-weights checks are not here: on the sample CT both families' tokens were compared with
feldglas's tools, which made the RADAR study's fields, and were identical (RADAR on the real
checkpoint; ts.v2:total_fast and ts.v2:total on MPS in fp16, every lattice). What IS here is what
could silently move a field without anyone comparing: where the model grid sits in the world, the
slabbed encoder against the whole one, and the weights refusing bytes that are not the pinned ones.
The nnU-Net family runs end to end only where TotalSegmentator's weights are installed (``-m slow``).
"""
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest

from haversack.encoders import ALIASES, ENCODERS, WeightsFile, resolve
from haversack.encoders import registry
from haversack.encoders import weights as W
from haversack.errors import InputError


class _WeightsRoot(unittest.TestCase):
    """Every test here gets its own weights root: nothing may touch ~/.haversack/encoders."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {W.ROOT_ENV: str(self.tmp / "weights")})
        self._env.start()
        self.assertEqual(W.root(), self.tmp / "weights")

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()


class TestRegistry(unittest.TestCase):
    def test_names_follow_the_task_grammar(self):
        for name, spec in ENCODERS.items():
            self.assertEqual(name, spec.name)
            family, _, short = name.partition(":")
            self.assertTrue(family and short, name)

    def test_old_names_resolve_to_their_encoders(self):
        for old, new in ALIASES.items():
            self.assertIs(resolve(old), ENCODERS[new])

    def test_a_pin_must_be_the_revision_this_haversack_runs(self):
        spec = ENCODERS["radar:pretrain"]
        self.assertIs(resolve(f"radar:pretrain@{spec.revision[:10]}"), spec)
        self.assertIs(resolve(f"radar@{spec.revision}"), spec)
        with self.assertRaisesRegex(InputError, "not deadbeef"):
            resolve("radar:pretrain@deadbeef")

    def test_an_unknown_name_lists_its_family(self):
        with self.assertRaisesRegex(InputError, "radar:pretrain"):
            resolve("radar:nope")
        with self.assertRaisesRegex(InputError, "no encoder"):
            resolve("nothing")

    def test_downloaded_weights_are_pinned_to_a_revision_not_a_branch(self):
        for spec in ENCODERS.values():
            for wf in spec.weights:
                self.assertIn(f"/resolve/{spec.revision}/", wf.url)
                self.assertEqual(len(wf.sha256), 64)
                self.assertGreater(wf.size, 0)

    def test_an_nnunet_encoder_is_its_task_and_tiles_on_whole_tokens(self):
        from haversack.ecosystems import EcosystemCatalog
        for spec in ENCODERS.values():
            if spec.family != "nnunet":
                continue
            self.assertFalse(spec.weights, "an nnU-Net encoder uses its task's weights")
            eco, short, _, _ = EcosystemCatalog().resolve(spec.uses_task)     # torch-free
            self.assertTrue(short)
            for lat in spec.lattices:
                self.assertTrue(all(spec.options["align"] % k == 0 for k in lat.kernel), (spec.name, lat))

    def test_unmeasured_facts_are_absent_not_guessed(self):
        for spec in ENCODERS.values():
            if spec.family == "nnunet":
                self.assertTrue(all(l.receptive_mm is None and l.support_offset_mm is None for l in spec.lattices))


class TestWeights(_WeightsRoot):
    def _spec(self, payload: bytes, size=None):
        wf = WeightsFile(url="https://example.org/w.pth", sha256=hashlib.sha256(payload).hexdigest(),
                         size=len(payload) if size is None else size, name="w.pth")
        return dataclasses.replace(ENCODERS["radar:pretrain"], weights=(wf,))

    @contextlib.contextmanager
    def _serving(self, body: bytes):
        with mock.patch("haversack.fetchlib.open", side_effect=lambda *a, **k: contextlib.nullcontext(io.BytesIO(body))) as m:
            yield m

    def test_fetch_installs_the_pinned_bytes_once(self):
        body = b"weights" * 1000
        spec = self._spec(body)
        self.assertFalse(W.installed(spec))
        with self._serving(body) as m:
            (p,) = W.fetch(spec)
            W.fetch(spec)
        self.assertEqual(m.call_count, 1, "a verified install is not fetched again")
        self.assertEqual(p.read_bytes(), body)
        self.assertTrue(W.installed(spec))
        self.assertEqual(p.parent, W.root() / "radar" / spec.revision)

    def test_other_bytes_are_refused_and_leave_nothing(self):
        spec = self._spec(b"the real ones")
        with self._serving(b"other bytes!!"), self.assertRaisesRegex(InputError, "refused"):
            W.fetch(spec)
        self.assertEqual([p.name for p in W.directory(spec).iterdir()], [])

    def test_a_server_sending_more_than_the_pin_is_cut_off(self):
        spec = self._spec(b"small")
        with self._serving(b"x" * 100), self.assertRaisesRegex(InputError, "more than the pinned"):
            W.fetch(spec)
        self.assertEqual(list(W.directory(spec).iterdir()), [])

    def test_a_file_changed_after_install_is_not_installed(self):
        body = b"abc" * 100
        spec = self._spec(body)
        with self._serving(body):
            (p,) = W.fetch(spec)
        p.write_bytes(b"abd" * 100)                         # same size, other bytes: the sidecar no longer describes it
        self.assertFalse(W.installed(spec))

    def test_adopt_verifies_like_a_download(self):
        body = b"mine" * 50
        spec = self._spec(body)
        src = self.tmp / "copy.pth"
        src.write_bytes(b"nope" * 50)
        with self.assertRaisesRegex(InputError, "refused"):
            W.adopt(spec, src)
        self.assertFalse(W.installed(spec))
        src.write_bytes(body)
        W.adopt(spec, src)
        self.assertTrue(W.installed(spec))
        self.assertEqual(src.read_bytes(), body, "the user's copy is left where it was")
        gone = W.remove(spec)
        self.assertEqual(len(gone), 2)                      # the file and its sidecar
        self.assertFalse(W.installed(spec))


def _sphere_ct(size_xyz=(80, 72, 24), spacing=(1.5, 1.5, 5.0), direction=None, center_lps=(10.0, -20.0, 30.0),
               radius=25.0):
    """A body-like sphere of soft tissue in air, whose center in the world is known - and OFF the
    image's middle, so a grid mirrored about the middle misplaces it."""
    import SimpleITK as sitk
    img = sitk.Image([int(v) for v in size_xyz], sitk.sitkFloat32)
    img.SetSpacing(spacing)
    if direction is not None:
        img.SetDirection(direction)
    # put the sphere's center at the image's middle voxel, then move the origin so it lands on center_lps
    mid = [(n - 1) / 2 for n in size_xyz]
    img.SetOrigin((0.0, 0.0, 0.0))
    off = np.asarray((12.0, -9.0, 20.0))                            # the sphere's center from the image's middle
    o = np.asarray(center_lps) - off - np.asarray(img.TransformContinuousIndexToPhysicalPoint(mid))
    img.SetOrigin(tuple(float(v) for v in o))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in size_xyz], indexing="ij"), -1).reshape(-1, 3)
    D = np.asarray(img.GetDirection()).reshape(3, 3) * np.asarray(spacing)
    world = idx @ D.T + np.asarray(img.GetOrigin())
    inside = np.linalg.norm(world - np.asarray(center_lps), axis=1) <= radius
    vol = np.where(inside, 40.0, -1000.0).reshape(size_xyz).transpose(2, 1, 0)
    out = sitk.GetImageFromArray(vol.astype(np.float32))
    out.CopyInformation(img)
    return out


def _world_centroid(prepared, threshold):
    """The world centroid of the model voxels above ``threshold``, through the prepared grid."""
    t = prepared.tensor.detach().cpu().numpy().reshape(prepared.tensor.shape[-3:])
    idx = np.argwhere(t > threshold).mean(0)                        # (Z, Y, X)
    rows = np.asarray(prepared.grid["directions"])
    return np.asarray(prepared.grid["origin"]) + idx @ rows


class _RandomRadar(_WeightsRoot):
    """RADAR's architecture with random weights, installed under the root like the real checkpoint,
    as a test encoder with a small minimum shape (the real one pads to 96 x 256 x 384)."""

    def setUp(self):
        super().setUp()
        torch = pytest.importorskip("torch")
        pytest.importorskip("dynamic_network_architectures")
        from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
        base = ENCODERS["radar:pretrain"]
        a = base.options["arch"]
        torch.manual_seed(0)
        enc = PlainConvEncoder(1, a["n_stages"], a["features_per_stage"], torch.nn.Conv3d, a["kernel_sizes"],
                               a["strides"], a["n_conv_per_stage"], conv_bias=True, norm_op=torch.nn.BatchNorm3d,
                               norm_op_kwargs={}, nonlin=torch.nn.ReLU, return_skips=True)
        sd = {f"visual_encoder.UNet.encoder.{k}": v for k, v in enc.state_dict().items()}
        for key, ch, _ in base.options["projections"]:
            p = torch.nn.Conv3d(ch, 256, 1)
            sd[f"visual_encoder.{key}.weight"], sd[f"visual_encoder.{key}.bias"] = p.weight, p.bias
        ck = self.tmp / "ck.pth"
        torch.save({"model": sd}, ck)
        body = ck.read_bytes()
        wf = WeightsFile(url="https://example.org/ck.pth", sha256=hashlib.sha256(body).hexdigest(), size=len(body),
                         name="ck.pth")
        self.spec = dataclasses.replace(base, name="radar:test", weights=(wf,),
                                        options={**base.options, "min_shape": (32, 64, 64)})
        self._reg = mock.patch.dict(registry.ENCODERS, {"radar:test": self.spec})
        self._reg.start()
        W.adopt(self.spec, ck)

    def tearDown(self):
        self._reg.stop()
        super().tearDown()


class TestRadarGeometry(_RandomRadar):
    def _check_center(self, direction):
        from haversack.encoders import radar
        center = (10.0, -20.0, 30.0)
        prepared = radar.prepare(self.spec, _sphere_ct(direction=direction, center_lps=center))
        got = _world_centroid(prepared, 0.5)
        # the model grid is 1 x 1 x 5 mm: the centroid is placed well inside one voxel
        self.assertLess(np.abs(got - center).max(), 1.0, (direction, got))

    def test_the_model_grid_sits_where_the_image_is_axial_lps(self):
        self._check_center(None)

    def test_the_model_grid_sits_where_the_image_is_flipped(self):
        self._check_center((-1, 0, 0, 0, 1, 0, 0, 0, -1))           # RAS-stored rows, feet-first slices

    def test_the_input_grid_is_the_delivered_image(self):
        from haversack.encoders import radar
        img = _sphere_ct()
        g = radar.prepare(self.spec, img).extra["input_grid"]
        self.assertEqual(g["shape"], list(img.GetSize()))
        self.assertEqual(g["origin"], list(np.round(img.GetOrigin(), 9)))
        self.assertEqual(g["space"], "left-posterior-superior")

    def test_a_slabbed_embedding_is_the_whole_embedding(self):
        import torch
        from haversack.encoders import radar
        model = radar.load(self.spec, W.directory(self.spec), torch.device("cpu"), torch.float32)
        prepared = radar.prepare(self.spec, _sphere_ct())
        whole = radar.run(self.spec, model, prepared, "cpu", torch.float32, slab=0)
        slabbed = radar.run(self.spec, model, prepared, "cpu", torch.float32, slab=3)
        for a, b in zip(whole, slabbed):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_a_checkpoint_missing_a_key_is_refused(self):
        import torch
        from haversack.encoders import radar
        ck = W.path(self.spec, self.spec.weights[0])
        sd = torch.load(ck, weights_only=True)["model"]
        sd.pop("visual_encoder.proj2.bias")
        d = self.tmp / "bad"
        d.mkdir()
        torch.save({"model": sd}, d / self.spec.weights[0].name)
        with self.assertRaises(KeyError):
            radar.load(self.spec, d, torch.device("cpu"), torch.float32)


class TestEmbedEndToEnd(_RandomRadar):
    def test_embed_writes_a_field_feldglas_reads(self):
        pytest.importorskip("feldglas.store")
        import SimpleITK as sitk
        from feldglas.store import read_field
        from haversack.encoders.pipeline import embed
        ct = self.tmp / "ct.nii.gz"
        sitk.WriteImage(_sphere_ct(), str(ct))
        out = self.tmp / "f.zarr.zip"
        r = embed("radar:test", str(ct), out, device="cpu")
        f = read_field(out)
        self.assertEqual([len(t) for t in f.tokens], r["tokens"])
        self.assertEqual([tuple(k) for k in f.kernels], [l.kernel for l in self.spec.lattices])
        self.assertEqual(tuple(f.grid.shape), tuple(r["model_grid"]))
        p = f.provenance
        self.assertEqual(p.encoder, "radar:test")
        self.assertEqual(p.weights, f"sha256:{self.spec.weights[0].sha256}")
        self.assertEqual(p.license, "CC-BY-NC-SA-4.0")
        self.assertEqual(p.input["identity"]["digest"][:7], "sha256:")
        self.assertEqual(p.input["grid"]["shape"], [80, 72, 24])
        self.assertEqual(f.embedding.layers, ("deep", "mid", "fine"))
        self.assertEqual([c["doi"] for c in p.extra["attribution"]["cite"]], ["10.1126/science.aec6129"])

    def test_the_embed_command_says_what_it_wrote(self):
        """The command's closing line read ``r['field']`` after the result key became
        ``embedding`` (the 2026-09-24 rename): every ``haversack embed`` without --json wrote
        its field and then died with a KeyError, exit 1 (found 2026-09-25)."""
        import contextlib
        import io

        import SimpleITK as sitk
        from haversack import cli
        ct = self.tmp / "ct.nii.gz"
        sitk.WriteImage(_sphere_ct(), str(ct))
        out = self.tmp / "f.zarr.zip"
        so, se = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            code = cli.main(["embed", str(ct), "-e", "radar:test", "-o", str(out), "--device", "cpu", "-q"])
        self.assertEqual(code, 0, se.getvalue())
        self.assertIn(str(out), so.getvalue())
        self.assertIn("radar:test on cpu", so.getvalue())

    def test_embed_refuses_before_any_work(self):
        from haversack.encoders.pipeline import embed
        with self.assertRaisesRegex(InputError, "zarr.zip"):
            embed("radar:test", "nowhere.nii.gz", self.tmp / "f.npz")
        W.remove(self.spec)
        with self.assertRaisesRegex(InputError, "weights fetch radar:test"):
            embed("radar:test", "nowhere.nii.gz", self.tmp / "f.zarr.zip")

    def test_fp16_on_the_cpu_is_refused(self):
        import SimpleITK as sitk
        from haversack.encoders.pipeline import embed
        ct = self.tmp / "ct.nii.gz"
        sitk.WriteImage(_sphere_ct(), str(ct))
        with self.assertRaisesRegex(InputError, "fp16 on the CPU"):
            embed("radar:test", str(ct), self.tmp / "f.zarr.zip", device="cpu", dtype="fp16")


class TestNnunetTiling(unittest.TestCase):
    def test_tiles_cover_every_voxel_on_whole_tokens(self):
        from haversack.encoders.nnunet import padded_extent, tile_slices
        patch, kernels, align = (48, 64, 80), ((4, 4, 4), (8, 8, 8), (16, 16, 16)), 16
        padded = tuple(padded_extent(n, p, align) for n, p in zip((101, 130, 70), patch))
        self.assertEqual(padded, (112, 144, 80))
        cover = np.zeros(padded, bool)
        for vox, toks in tile_slices(padded, patch, kernels, align):
            cover[vox] = True
            for sl, k in zip(toks, kernels):
                self.assertEqual([s.start * kk for s, kk in zip(sl, k)], [v.start for v in vox])
                self.assertEqual([(s.stop - s.start) * kk for s, kk in zip(sl, k)], list(patch))
        self.assertTrue(cover.all())

    def test_token_weights_average_the_gaussian_per_token(self):
        from haversack.encoders.nnunet import token_weights
        g = np.random.default_rng(0).random((8, 8, 8))
        w = token_weights(g, (4, 4, 4))
        self.assertEqual(w.shape, (2, 2, 2))
        self.assertAlmostEqual(w[1, 0, 1], g[4:, :4, 4:].mean())


@pytest.mark.slow
class TestNnunetEmbed(unittest.TestCase):
    """End to end on TotalSegmentator's real weights, where they are installed."""

    def test_total_fast_embeds_the_sphere(self):
        pytest.importorskip("feldglas.store")
        from haversack.errors import InputError as E
        from haversack.encoders.pipeline import embed
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as d:
            ct = pathlib.Path(d) / "ct.nii.gz"
            sitk.WriteImage(_sphere_ct(size_xyz=(160, 160, 40), spacing=(1.0, 1.0, 3.0)), str(ct))
            try:
                r = embed("ts.v2:total_fast", str(ct), pathlib.Path(d) / "f.zarr.zip", device="cpu")
            except E as e:
                if "not installed" in str(e):
                    pytest.skip(str(e))
                raise
            self.assertEqual(len(r["tokens"]), 3)


class TestCommands(_WeightsRoot):
    def _run(self, argv):
        from haversack import cli
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_encoders_lists_every_encoder(self):
        code, out, _ = self._run(["encoders", "--json"])
        self.assertEqual(code, 0)
        rows = {r["name"]: r for r in json.loads(out)["encoders"]}
        self.assertEqual(set(rows), set(ENCODERS))
        self.assertIs(rows["radar:pretrain"]["installed"], False)
        self.assertIsInstance(rows["ts.v2:total_fast"]["installed"], bool)   # its task's weights, here or not
        self.assertIn("radar", rows["radar:pretrain"]["aliases"])
        self.assertEqual(rows["radar:pretrain"]["attribution"]["title"], "RADAR")
        dois = [c["doi"] for c in rows["ts.v2:total"]["attribution"]["cite"]]
        self.assertIn("10.1148/ryai.230024", dois, "an nnU-Net encoder cites its task's makers")
        self.assertNotIn("10.1148/radiol.241613", dois, "the MRI paper is for MR models")

    def test_embed_with_an_unknown_encoder_is_one_line(self):
        code, _, err = self._run(["embed", "x.nii.gz", "-e", "nope:nope", "-o", str(self.tmp / "f.zarr.zip")])
        self.assertEqual(code, 2)
        self.assertIn("no encoder", err)
        self.assertNotIn("Traceback", err)

    def test_an_unknown_encoder_names_every_encoder(self):
        """``radar:nope`` listed the radar family alone, as if haversack embedded with
        nothing else (seen on a smoke deployment, 2026-09-25)."""
        with self.assertRaises(InputError) as e:
            resolve("radar:nope")
        for name in ENCODERS:
            self.assertIn(name, str(e.exception))

    def test_weights_fetch_from_a_wrong_file_is_refused(self):
        bad = self.tmp / "bad.pth"
        bad.write_bytes(b"not radar")
        code, _, err = self._run(["weights", "fetch", "radar:pretrain", "--from", str(bad)])
        self.assertEqual(code, 2)
        self.assertIn("refused", err)
        self.assertFalse(W.installed(ENCODERS["radar:pretrain"]))


if __name__ == "__main__":
    unittest.main()
