"""MTP sidecar quantization spec — pure logic, no GGUF/libggml needed."""

from voodoo_quant.tools.sidecar import (
    DEFAULT_SIDECAR_KV_QUANT,
    DEFAULT_SIDECAR_QUANT,
    describe,
    is_sidecar_name,
    sidecar_budget,
    sidecar_quant_for_tensor,
)


def test_defaults():
    assert DEFAULT_SIDECAR_QUANT == "Q6_K"
    assert DEFAULT_SIDECAR_KV_QUANT == "Q8_0"


def test_sidecar_detection():
    assert is_sidecar_name("blk.24.nextn.eh_proj.weight")
    assert not is_sidecar_name("blk.0.attn_q.weight")


def test_big_weights_quantize_to_level():
    # 0.8B-MTP's real eh_proj: (2048, 1024)
    assert sidecar_quant_for_tensor("blk.24.nextn.eh_proj.weight", (2048, 1024)) == "Q6_K"
    assert sidecar_quant_for_tensor("blk.24.nextn.eh_proj.weight", (2048, 1024), level="Q5_K") == "Q5_K"


def test_27b_families():
    # Qwen3.8-27B's nextn block carries full projections (treatise: 15 tensors,
    # Q6_K with k/v at Q8_0)
    assert sidecar_quant_for_tensor("blk.64.nextn.attn_q.weight", (4096, 4096)) == "Q6_K"
    assert sidecar_quant_for_tensor("blk.64.nextn.attn_k.weight", (1024, 4096)) == "Q8_0"
    assert sidecar_quant_for_tensor("blk.64.nextn.attn_v.weight", (1024, 4096)) == "Q8_0"
    assert sidecar_quant_for_tensor("blk.64.nextn.ffn_gate.weight", (18944, 4096)) == "Q6_K"
    assert sidecar_quant_for_tensor("blk.64.nextn.attn_output.weight", (4096, 4096)) == "Q6_K"


def test_norms_and_state_stay_f32():
    for name in (
        "blk.24.nextn.enorm.weight",
        "blk.24.nextn.hnorm.weight",
        "blk.24.nextn.shared_head_norm.weight",
        "blk.64.nextn.attn_norm.weight",
        "blk.64.nextn.ssm_a",
        "blk.64.nextn.ssm_dt.bias",
        "blk.64.nextn.ssm_conv1d.weight",
        "blk.64.nextn.some.bias",
    ):
        assert sidecar_quant_for_tensor(name, (1024,)) is None, name
        assert sidecar_quant_for_tensor(name, (1024, 1024)) is None, name


def test_tiny_or_misaligned_stay_f32():
    # 1-D
    assert sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (1024,)) is None
    # a single block row is legal for ggml but pointless as a "big weight";
    # the spec only asks for >= 2 rows x 1 block
    assert sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (2, 256)) == "Q6_K"
    assert sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (1, 256)) is None
    # cols not 256-aligned falls back to Q8_0 when 32-aligned
    assert sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (64, 32)) == "Q8_0"
    assert sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (64, 33)) is None


def test_budget_exact_bytes():
    tensors = {
        "blk.24.nextn.eh_proj.weight": (2048, 1024),        # Q6_K
        "blk.24.nextn.enorm.weight": (1024,),               # F32
        "blk.64.nextn.attn_k.weight": (1024, 4096),         # Q8_0
    }
    b = sidecar_budget(tensors)
    assert b["blk.24.nextn.eh_proj.weight"] == 2048 * 1024 * (210 / 256)
    assert b["blk.24.nextn.enorm.weight"] == 1024 * 4
    assert b["blk.64.nextn.attn_k.weight"] == 1024 * 4096 * (34 / 32)
    # the whole point: Q6_K sidecar beats the old BF16 copy-through
    bf16 = 2048 * 1024 * 2
    assert b["blk.24.nextn.eh_proj.weight"] < bf16 * 0.5


def test_unknown_level_raises():
    import pytest

    with pytest.raises(ValueError):
        sidecar_quant_for_tensor("blk.0.nextn.eh_proj.weight", (2048, 1024), level="NOT_A_QUANT")


def test_describe():
    text = describe("Q6_K")
    assert "Q6_K" in text and "Q8_0" in text and "F32" in text
