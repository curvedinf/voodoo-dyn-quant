"""Core correctness invariants (CPU-runnable).

1. TP slicing: block-aligned shard slices of quantized candidates are
   bit-identical to quantizing the shard directly — the invariant that lets
   ranks share one full-tensor candidate cache.
2. Mixed-quant gates: forward, gradient flow, temperature concentration,
   per-role candidate menus.

Skips when libggml-base is unavailable.
"""

import pytest
import torch

from voodoo_quant import ggml

requires_ggml = pytest.mark.skipif(ggml.find_libggml() is None, reason="libggml-base not built")


@requires_ggml
class TestTPSliceInvariants:
    def test_row_slice_bit_identical(self):
        from voodoo_quant.parallel import ShardSpec

        torch.manual_seed(0)
        w = torch.randn(8, 1024)
        q = ggml.quantize_tensor(w, "Q4_K")
        spec = ShardSpec(full_out=8, full_in=1024, row_offset=2, row_len=3)
        sliced = spec.slice_qbytes(q, "Q4_K", 1024)
        direct = ggml.quantize_tensor(w[2:5].contiguous(), "Q4_K")
        assert torch.equal(sliced.reshape(-1), direct.reshape(-1))

    def test_col_slice_bit_identical(self):
        from voodoo_quant.parallel import ShardSpec

        torch.manual_seed(1)
        w = torch.randn(8, 1024)
        q = ggml.quantize_tensor(w, "IQ4_XS")
        spec = ShardSpec(full_out=8, full_in=1024, col_offset=256, col_len=256)
        sliced = spec.slice_qbytes(q, "IQ4_XS", 1024)
        direct = ggml.quantize_tensor(w[:, 256:512].contiguous(), "IQ4_XS")
        assert torch.equal(sliced.reshape(-1), direct.reshape(-1))

    def test_misaligned_col_slice_rejected(self):
        from voodoo_quant.parallel import ShardSpec

        w = torch.randn(8, 1024)
        q = ggml.quantize_tensor(w, "Q4_K")
        with pytest.raises(ValueError):
            ShardSpec(full_out=8, full_in=1024, col_offset=128, col_len=256).slice_qbytes(
                q, "Q4_K", 1024
            )


@requires_ggml
class TestMixedQuantGates:
    def test_forward_and_gradients(self, tmp_path, monkeypatch):
        # isolate the candidate cache so the test never sees stale entries
        from voodoo_quant import layers

        monkeypatch.setattr(layers, "CANDIDATE_CACHE_DIR", tmp_path)
        torch.manual_seed(0)
        lin = layers.MixedQuantLinear(
            in_features=512, out_features=64,
            candidate_types=["IQ2_XXS", "Q4_K", "Q8_0"],
            source_weight=torch.randn(64, 512),
            lazy=False, tensor_name="test.proj",
        )
        y = lin(torch.randn(4, 512))
        assert y.shape == (4, 64)
        assert torch.allclose(lin.get_probs().sum(), torch.tensor(1.0))

        lin.gates.requires_grad_(True)
        lin(torch.randn(4, 512)).sum().backward()
        assert lin.gates.grad is not None and lin.gates.grad.abs().sum() > 0

    def test_temperature_concentration(self):
        from voodoo_quant.layers import MixedQuantLinear, resolve_candidate_types

        lin = MixedQuantLinear(
            512, 64, ["IQ2_XXS", "Q4_K"], torch.randn(64, 512) * 0.01 + 0.5,
            lazy=False, tensor_name="t2.proj",
        )
        lin.set_temperature(0.01)
        # gates still ~uniform -> assignment is argmax; prob mass at low tau is ~1
        assert lin.get_assignment() in ("IQ2_XXS", "Q4_K")

    def test_role_menus(self):
        from voodoo_quant.layers import resolve_candidate_types

        base = ["Q8_0", "Q5_K", "IQ3_S", "IQ2_S"]
        attn = ["Q5_K", "IQ3_S", "IQ2_S"]
        got_attn = resolve_candidate_types("layers.0.self_attn.qkv", base, attn, None)
        got_mlp = resolve_candidate_types("layers.0.mlp.up_proj", base, attn, None)
        assert "Q8_0" not in got_attn
        assert got_mlp == base  # non-attention keeps the global menu
