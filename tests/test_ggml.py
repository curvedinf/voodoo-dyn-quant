"""Unit tests for the ggml quant bridge (no GPU required).

Skips quantize/dequantize when libggml-base is unavailable so the suite runs
on a bare checkout; registry/ladder logic is always tested.
"""

import pytest
import torch

from voodoo_quant import ggml


def test_registry_and_ladder():
    types = ggml.list_quant_types()
    assert "IQ4_XS" in types and "Q8_0" in types
    # Ladder is a subset of the registry, ordered coarse -> fine.
    ladder = ggml.QUANT_LADDER
    assert set(ladder) <= set(types)
    bpw = [ggml.bytes_per_weight(qt) for qt in ladder]
    assert bpw == sorted(bpw), "ladder must be monotonically finer"


def test_default_candidates_exclude_ternary():
    cands = ggml.default_candidate_types()
    assert "TQ1_0" not in cands and "TQ2_0" not in cands
    assert cands[0] == "IQ1_S" and cands[-1] == "Q8_0"


def test_bytes_per_weight_matches_registry():
    assert ggml.bytes_per_weight("Q8_0") == pytest.approx(34 / 32)
    assert ggml.bytes_per_weight("IQ4_XS") == pytest.approx(136 / 256)
    with pytest.raises(ValueError):
        ggml.get_quant_info("NOT_A_QUANT")


@pytest.mark.skipif(ggml.find_libggml() is None, reason="libggml-base not built")
class TestQuantizeRoundtrip:
    def test_roundtrip_f32(self):
        torch.manual_seed(0)
        w = torch.randn(8, 512)
        # Per-type relative-error ceilings scale with bits/weight (random
        # normal data; measured values: Q8_0 0.006, Q4_K 0.075, IQ4_XS 0.080,
        # IQ2_XXS 0.352).
        ceilings = {"Q8_0": 0.02, "Q4_K": 0.12, "IQ4_XS": 0.12, "IQ2_XXS": 0.45}
        for qt, ceil in ceilings.items():
            qb = ggml.quantize_tensor(w, qt)
            w2 = ggml.dequantize_tensor(qb, qt, 8, 512)
            err = ((w2 - w).abs().mean() / w.abs().mean()).item()
            assert err < ceil, f"{qt} roundtrip error {err:.3f} >= {ceil}"

    def test_output_shape(self):
        w = torch.randn(4, 256)
        qb = ggml.quantize_tensor(w, "Q4_K")
        nblocks = 4 * (256 // 256)
        assert qb.shape == (nblocks, 144)

    def test_shape_validation(self):
        w = torch.randn(4, 100)  # not block-aligned
        with pytest.raises(ValueError):
            ggml.quantize_tensor(w, "Q4_K")
