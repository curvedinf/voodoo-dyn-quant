"""Qwen3.5-family hybrid SSM/attention architecture adapter.

Delegates to the sharding primitives ported with the original training stack
(:mod:`voodoo_quant.parallel`'s ``tp_ssm`` / ``tp_attention`` / ``tp_mlp``).
GGUF name mapping stays in the exporter's built-in table (the adapter returns
None so the exporter falls through), keeping the battle-tested mapping
untouched.
"""

from __future__ import annotations

from typing import Optional

from voodoo_quant.arch import register
from voodoo_quant.arch.base import ArchAdapter


@register
class QwenHybridAdapter(ArchAdapter):
    hf_model_types = ("qwen3_5", "qwen3_6", "qwen3_8")
    ggml_archs = ("qwen35", "qwen36", "qwen38")
    attention_segments = frozenset({"self_attn", "linear_attn"})

    @classmethod
    def shard_layer(cls, layer) -> list[str]:
        from voodoo_quant.parallel import tp_attention, tp_mlp, tp_ssm

        handled: list[str] = []
        gdn = getattr(layer, "linear_attn", None)
        if isinstance(gdn, __import__("torch").nn.Module):
            tp_ssm(gdn)
            handled.append("linear_attn")
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, __import__("torch").nn.Module):
            tp_attention(attn)
            handled.append("self_attn")
        if getattr(layer, "mlp", None) is not None:
            tp_mlp(layer.mlp)
            handled.append("mlp")
        return handled

    @classmethod
    def gguf_name(cls, hf_name: str) -> Optional[str]:
        return None  # exporter's built-in qwen table is authoritative
