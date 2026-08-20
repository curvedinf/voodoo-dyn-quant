"""Architecture adapter registry + LFM2 TP/export correctness.

Runs on CPU (model built from a shrunken config, no checkpoint download).
Network access only for the config fetch (cached by transformers after the
first run); skip if offline.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _tiny_lfm2_cfg():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-2.6B", trust_remote_code=True)
    cfg.num_hidden_layers = 6
    cfg.hidden_size = 512
    cfg.intermediate_size = 1024
    cfg.head_dim = 64
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 2
    cfg.vocab_size = 2000
    cfg.pad_token_id = None
    cfg.layer_types = ["conv", "conv", "full_attention", "conv", "full_attention", "conv"]
    return cfg


@pytest.fixture(scope="module")
def lfm2_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(_tiny_lfm2_cfg(), trust_remote_code=True).float()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model


def test_registry_resolves_adapters():
    from voodoo_quant.arch import adapter_for_model, adapters

    names = {a.__name__ for a in adapters()}
    assert {"QwenHybridAdapter", "LFM2Adapter"} <= names


def test_lfm2_adapter_resolves(lfm2_model):
    from voodoo_quant.arch import adapter_for_model

    assert adapter_for_model(lfm2_model).__name__ == "LFM2Adapter"


def test_lfm2_gguf_mapping(lfm2_model):
    """Every selectable/norm key maps to the verified lfm2 GGUF names."""
    from voodoo_quant.arch import adapter_for_model

    a = adapter_for_model(lfm2_model)
    cases = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.embedding_norm.weight": "token_embd_norm.weight",
        "model.layers.0.operator_norm.weight": "blk.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight": "blk.0.ffn_norm.weight",
        "model.layers.0.conv.in_proj.weight": "blk.0.shortconv.in_proj.weight",
        "model.layers.0.conv.out_proj.weight": "blk.0.shortconv.out_proj.weight",
        "model.layers.0.conv.conv.weight": "blk.0.shortconv.conv.weight",
        "model.layers.0.feed_forward.w1.weight": "blk.0.ffn_gate.weight",
        "model.layers.0.feed_forward.w3.weight": "blk.0.ffn_up.weight",
        "model.layers.0.feed_forward.w2.weight": "blk.0.ffn_down.weight",
        "model.layers.2.self_attn.q_proj.weight": "blk.2.attn_q.weight",
        "model.layers.2.self_attn.out_proj.weight": "blk.2.attn_output.weight",
        "model.layers.2.self_attn.q_layernorm.weight": "blk.2.attn_q_norm.weight",
    }
    for hf, want in cases.items():
        got = a.gguf_name(hf)
        assert got == want, (hf, got, want)
    assert a.transpose_embedding() is True


def test_lfm2_tp_reconstruction(lfm2_model):
    """tp=2 shard of each layer type reconstructs the unsharded math exactly."""
    import copy

    import torch.nn.functional as F

    from voodoo_quant import parallel as par
    from voodoo_quant.arch import adapter_for_model

    adapter = adapter_for_model(lfm2_model)

    def shard(rank, idx):
        m = copy.deepcopy(lfm2_model)
        par.TP.world_size, par.TP.rank = 2, rank
        layer = m.model.layers[idx]
        adapter.shard_layer(layer)
        return layer

    torch.manual_seed(7)
    h = torch.randn(2, 32, 512)

    # ---- attention layer (2): column q/k/v, row out_proj ----
    la0, la1 = shard(0, 2), shard(1, 2)
    fa = lfm2_model.model.layers[2].self_attn
    with torch.no_grad():
        for nm in ("q_proj", "k_proj", "v_proj"):
            w0 = getattr(la0.self_attn, nm).weight.data
            w1 = getattr(la1.self_attn, nm).weight.data
            wf = getattr(fa, nm).weight.data
            assert torch.allclose(torch.cat([w0, w1], 0), wf, atol=1e-6), nm
        assert torch.allclose(
            torch.cat([la0.self_attn.out_proj.weight.data, la1.self_attn.out_proj.weight.data], 1),
            fa.out_proj.weight.data, atol=1e-6,
        )

    # ---- conv layer (0): grouped-row in_proj, dw channel slice, row out_proj ----
    lc0, lc1 = shard(0, 0), shard(1, 0)
    fc = lfm2_model.model.layers[0].conv
    hid, lh = 512, 256
    with torch.no_grad():
        ipf = F.linear(h, fc.in_proj.weight.data)
        Bf, Cf, xf = ipf.chunk(3, dim=-1)
        out_full = F.linear(Cf * (Bf * xf), fc.out_proj.weight.data)

        def rank_out(lc):
            ip = F.linear(h, lc.conv.in_proj.weight.data)
            B, C, x = ip.chunk(3, dim=-1)
            return F.linear(C * (B * x), lc.conv.out_proj.weight.data)

        out_tp = rank_out(lc0) + rank_out(lc1)
        assert torch.allclose(out_tp, out_full, atol=1e-4)
        # rank-local shapes are tp-sized
        assert lc0.conv.in_proj.weight.shape == (3 * lh, hid)
        assert lc0.conv.conv.weight.shape[0] == lh
        assert lc0.conv.out_proj.weight.shape == (hid, lh)

    # ---- MLP (feed_forward w1/w3 column, w2 row) ----
    lf0, lf1 = shard(0, 2), shard(1, 2)
    ff = lfm2_model.model.layers[2].feed_forward
    with torch.no_grad():
        g_full = F.silu(F.linear(h, ff.w1.weight.data)) * F.linear(h, ff.w3.weight.data)
        out_full = F.linear(g_full, ff.w2.weight.data)
        g0 = F.silu(F.linear(h, lf0.feed_forward.w1.weight.data)) * F.linear(h, lf0.feed_forward.w3.weight.data)
        g1 = F.silu(F.linear(h, lf1.feed_forward.w1.weight.data)) * F.linear(h, lf1.feed_forward.w3.weight.data)
        out_tp = F.linear(g0, lf0.feed_forward.w2.weight.data) + F.linear(g1, lf1.feed_forward.w2.weight.data)
        assert torch.allclose(out_tp, out_full, atol=1e-4)


def test_role_menus_use_adapter_segments():
    from voodoo_quant.layers import resolve_candidate_types

    base = ["Q8_0", "Q5_K", "IQ3_S", "IQ2_S"]
    attn = ["Q5_K", "IQ3_S"]
    # conv.* is attention-side only for adapters that say so; the layer-level
    # default (qwen segments) treats mlp.* as non-attention either way.
    got = resolve_candidate_types("layers.0.mlp.w1", base, attn, None)
    assert got == base
