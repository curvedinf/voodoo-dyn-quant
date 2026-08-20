"""LFM2 / LFM2.5 (LiquidAI) architecture adapter.

Verified against ``LiquidAI/LFM2.5-2.6B`` (config + meta-model) and its GGUF
(``LiquidAI/LFM2.5-2.6B-GGUF``, arch ``lfm2``): 30 layers, 22
``Lfm2ShortConv`` (fused ``[B|C|x]`` ``in_proj``, depthwise conv,
``out_proj``) + 8 ``Lfm2Attention`` (GQA 32q/8kv, head_dim 64), tied
embeddings, SwiGLU MLP named ``feed_forward.w1/w3/w2`` (gate/up/down).

GGUF tensor mapping (llama.cpp names):

=========================  ============================
HF                         GGUF
=========================  ============================
model.embed_tokens.weight  token_embd.weight   (TRANSPOSED: [hidden, vocab])
model.embedding_norm       token_embd_norm
layers.N.operator_norm     blk.N.attn_norm
layers.N.ffn_norm          blk.N.ffn_norm
self_attn.q_layernorm      blk.N.attn_q_norm
self_attn.k_layernorm      blk.N.attn_k_norm
self_attn.{q,k,v}_proj     blk.N.attn_{q,k,v}
self_attn.out_proj         blk.N.attn_output
conv.in_proj               blk.N.shortconv.in_proj
conv.out_proj              blk.N.shortconv.out_proj
conv.conv (depthwise)      blk.N.shortconv.conv
feed_forward.w1/w3/w2      blk.N.ffn_{gate,up,down}
=========================  ============================

TP sharding: attention is standard GQA column/row; the shortconv layer shards
by hidden channels — ``in_proj`` uses the grouped-row pattern (this rank's
channel slice in each of the three thirds; same mechanism as the Qwen GDN
``in_proj_qkv``), the depthwise conv slices its channel dim directly, and
``out_proj`` is row-parallel. The ``chunk(3)`` in the HF forward happens after
the projection, so each third is channel-major and the gather is exact.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn

from voodoo_quant.arch import register
from voodoo_quant.arch.base import ArchAdapter

_SHORTCONV_RE = re.compile(r"^model\.layers\.(\d+)\.conv\.(in_proj|out_proj)\.weight$")
_ATTN_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$")
_FFN_RE = re.compile(r"^model\.layers\.(\d+)\.feed_forward\.(w1|w2|w3)\.weight$")

_PROJ_TO_GGUF = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "out_proj": "attn_output"}
_FFN_TO_GGUF = {"w1": "ffn_gate", "w3": "ffn_up", "w2": "ffn_down"}


@register
class LFM2Adapter(ArchAdapter):
    hf_model_types = ("lfm2",)
    ggml_archs = ("lfm2",)
    # shortconv layers are the attention-side family for role menus
    attention_segments = frozenset({"self_attn", "conv"})

    # -- tensor-parallel sharding -------------------------------------------

    @classmethod
    def shard_layer(cls, layer) -> list[str]:
        from voodoo_quant.parallel import _shard_child

        tp, r = cls._tp_state()
        if tp < 2:
            return []
        handled: list[str] = []

        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, nn.Module) and getattr(attn, "q_proj", None) is not None:
            for nm in ("q_proj", "k_proj", "v_proj"):
                m = getattr(attn, nm)
                assert m.out_features % tp == 0, f"{nm} out {m.out_features} % tp={tp}"
                lq = m.out_features // tp
                _shard_child(attn, nm, row_offset=r * lq, row_len=lq)
            o = attn.out_proj
            assert o.in_features % tp == 0, f"out_proj in {o.in_features} % tp={tp}"
            lo = o.in_features // tp
            _shard_child(attn, "out_proj", col_offset=r * lo, col_len=lo)
            handled.append("self_attn")

        conv = getattr(layer, "conv", None)
        if isinstance(conv, nn.Module) and getattr(conv, "in_proj", None) is not None:
            hidden = conv.in_proj.in_features
            assert hidden % tp == 0, f"shortconv hidden {hidden} % tp={tp}"
            lh = hidden // tp
            rows = torch.cat([
                torch.arange(r * lh, (r + 1) * lh),
                torch.arange(hidden + r * lh, hidden + (r + 1) * lh),
                torch.arange(2 * hidden + r * lh, 2 * hidden + (r + 1) * lh),
            ])
            _shard_child(conv, "in_proj", row_index=rows)
            # Depthwise conv sees only B*x ([hidden] channels); slice the
            # channel dim (a buffer, not a selectable Linear).
            dw = conv.conv
            if isinstance(dw, nn.Conv1d) and dw.weight.shape[0] == hidden:
                with torch.no_grad():
                    dw.weight.data = dw.weight.data[r * lh : (r + 1) * lh].clone()
                    if dw.bias is not None:
                        dw.bias.data = dw.bias.data[r * lh : (r + 1) * lh].clone()
            # out_proj consumes this rank's channel slice: row-parallel,
            # all-reduce completes the sum.
            m = conv.out_proj
            assert m.in_features % tp == 0, f"shortconv out_proj in {m.in_features} % tp={tp}"
            lo = m.in_features // tp
            _shard_child(conv, "out_proj", col_offset=r * lo, col_len=lo)
            handled.append("conv")

        ffn = getattr(layer, "feed_forward", None)
        if isinstance(ffn, nn.Module) and getattr(ffn, "w1", None) is not None:
            g = ffn.w1
            assert g.out_features % tp == 0, f"ffn intermediate {g.out_features} % tp={tp}"
            li = g.out_features // tp
            _shard_child(ffn, "w1", row_offset=r * li, row_len=li)
            _shard_child(ffn, "w3", row_offset=r * li, row_len=li)
            _shard_child(ffn, "w2", col_offset=r * li, col_len=li)
            handled.append("feed_forward")
        return handled

    @staticmethod
    def _tp_state() -> tuple[int, int]:
        from voodoo_quant.parallel import TP

        return TP.world_size, TP.rank

    # -- GGUF export mapping -------------------------------------------------

    @classmethod
    def gguf_name(cls, hf_name: str) -> Optional[str]:
        m = _SHORTCONV_RE.match(hf_name)
        if m:
            return f"blk.{m.group(1)}.shortconv.{m.group(2)}.weight"
        if hf_name.endswith(".conv.conv.weight"):
            n = hf_name.split(".")[2]
            return f"blk.{n}.shortconv.conv.weight"
        m = _ATTN_RE.match(hf_name)
        if m:
            return f"blk.{m.group(1)}.{_PROJ_TO_GGUF[m.group(2)]}.weight"
        m = _FFN_RE.match(hf_name)
        if m:
            return f"blk.{m.group(1)}.{_FFN_TO_GGUF[m.group(2)]}.weight"
        if hf_name == "model.embed_tokens.weight":
            return "token_embd.weight"
        if hf_name == "model.embedding_norm.weight":
            return "token_embd_norm.weight"
        if hf_name.endswith(".operator_norm.weight"):
            n = hf_name.split(".")[2]
            return f"blk.{n}.attn_norm.weight"
        if hf_name.endswith(".ffn_norm.weight"):
            n = hf_name.split(".")[2]
            return f"blk.{n}.ffn_norm.weight"
        if hf_name.endswith(".self_attn.q_layernorm.weight"):
            n = hf_name.split(".")[2]
            return f"blk.{n}.attn_q_norm.weight"
        if hf_name.endswith(".self_attn.k_layernorm.weight"):
            n = hf_name.split(".")[2]
            return f"blk.{n}.attn_k_norm.weight"
        return None  # lm_head (tied) and unknowns fall through

    @classmethod
    def transpose_embedding(cls) -> bool:
        return True
