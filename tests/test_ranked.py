"""haversack's side of the ranked encoding: the emit hook and the re-export of rankfield."""
import inspect
import unittest

import numpy as np
import pytest
import torch

rf = pytest.importorskip("rankfield")

from haversack import ranked   # noqa: E402


def _logits(K=8, shape=(6, 10, 12), seed=0):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
    out = []
    for k in range(K):
        c = torch.rand(3, generator=g) * torch.tensor(shape, dtype=torch.float32)
        d = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).sqrt()
        out.append(6.0 - 0.9 * d)
    return torch.stack(out)


class TestReexport(unittest.TestCase):
    def test_the_encoding_is_the_librarys(self):
        self.assertIs(ranked.encode, rf.encode)
        self.assertIs(ranked.margin, rf.margin)
        self.assertIs(ranked.decode_groups, rf.decode_groups)
        self.assertIs(ranked.RankedCode, rf.RankField)
        self.assertEqual(ranked.RANKED_VERSION, rf.FORMAT_VERSION)


class TestEmit(unittest.TestCase):
    def test_emit_encodes_stamps_and_sinks(self):
        got = []
        spec = ranked.RankedSpec(sink=lambda part, code: got.append((part, code)), depth=4, clip=6.0)
        code = ranked.emit(spec, "organs", _logits(K=8), engine="nnunetv2", task="ts:total")
        self.assertEqual(len(got), 1)
        part, sunk = got[0]
        self.assertEqual(part, "organs")
        self.assertIs(sunk, code)
        self.assertEqual(code.meta["engine"], "nnunetv2")
        self.assertEqual(code.meta["depth"], 4)
        self.assertEqual(code.meta["clip"], 6.0)
        self.assertEqual(code.meta["version"], rf.FORMAT_VERSION)
        self.assertEqual(code.meta["keep"], "shell")

    def test_emit_is_a_noop_without_a_spec(self):
        self.assertIsNone(ranked.emit(None, "organs", _logits(K=4)))

    def test_emit_does_not_lose_the_encoders_own_meta(self):
        spec = ranked.RankedSpec(sink=lambda part, code: None, depth=3)
        code = ranked.emit(spec, "p", _logits(K=5), part="p")
        for key in ("mode", "classes", "shape", "support_max", "gap_curve", "gap_range"):
            self.assertIn(key, code.meta)

    def test_sink_key_is_a_string_even_when_the_caller_passes_an_index(self):
        got = []
        ranked.emit(ranked.RankedSpec(sink=lambda part, code: got.append(part)), 3, _logits(K=4))
        self.assertEqual(got, ["3"])

    def test_the_first_three_parameters_stay_positional_only(self):
        kinds = [p.kind for p in inspect.signature(ranked.emit).parameters.values()]
        self.assertEqual(kinds[:3], [inspect.Parameter.POSITIONAL_ONLY] * 3)
        self.assertEqual(kinds[3], inspect.Parameter.VAR_KEYWORD)


class TestCaches(unittest.TestCase):
    def test_the_distance_field_reads_levels_through_the_table(self):
        """The caches decode support bytes like every other reader: through the block's
        level table, so a 0.3 store's log byte is not read as 0.2's uniform one."""
        lg = _logits(K=4, shape=(8, 8, 8))
        code = ranked.encode(lg, depth=4)
        lut = rf.levels(code.meta)
        d3 = ranked.distance_field(code.ranks, code.support, clip=code.clip, spacing_zyx=(1, 1, 1),
                                   truncation=4.0, levels=lut, device="cpu")
        uni = ranked.distance_field(code.ranks, code.support, clip=code.clip, spacing_zyx=(1, 1, 1),
                                    truncation=4.0, levels=(1 - np.arange(256) / 255) * code.clip, device="cpu")
        self.assertEqual(d3.shape, (8, 8, 8))
        self.assertTrue((d3 != uni).any())


if __name__ == "__main__":
    unittest.main()
