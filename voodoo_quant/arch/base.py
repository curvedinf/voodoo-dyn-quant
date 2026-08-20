"""Architecture adapter base class.

Voodoo's core (``ggml``, ``layers``, ``training``) is architecture-agnostic:
it walks ``nn.Linear`` / ``nn.Embedding`` modules and quantizes whatever it
finds. Everything architecture-specific lives in adapters that subclass
:class:`ArchAdapter` and register themselves in :mod:`voodoo_quant.arch`.

An adapter owns three kinds of policy:

1. **Tensor-parallel sharding** (:meth:`shard_layer`) — which submodules of a
   decoder layer to shard and how, using the arch-neutral sharding primitives
   in :mod:`voodoo_quant.parallel` (mechanism lives there, policy here).
2. **GGUF export mapping** (:meth:`gguf_name`, :meth:`transpose_embedding`) —
   HF tensor names/layouts -> llama.cpp GGUF names/layouts.
3. **Role grouping** (:attr:`attention_segments`) — which path segments mark
   "attention-side" tensors, used by per-role candidate menus
   (``--attention_candidates`` / ``--non_attention_candidates``).
"""

from __future__ import annotations

from typing import Optional


# Path segments that mark attention-side tensors for per-role candidate menus.
# Segment-EXACT match (``conv1d`` does NOT match ``conv``).
DEFAULT_ATTENTION_SEGMENTS: frozenset[str] = frozenset(
    {"self_attn", "linear_attn", "attn", "conv", "shortconv"}
)


class ArchAdapter:
    """Base class; adapters override what their architecture needs."""

    #: transformers ``config.model_type`` values this adapter handles
    hf_model_types: tuple[str, ...] = ()

    #: llama.cpp ``general.architecture`` value(s) for the export mapping
    ggml_archs: tuple[str, ...] = ()

    #: path segments marking attention-side tensors (per-role menus)
    attention_segments: frozenset[str] = DEFAULT_ATTENTION_SEGMENTS

    @classmethod
    def matches_config(cls, config) -> bool:
        return getattr(config, "model_type", None) in cls.hf_model_types

    @classmethod
    def matches_gguf_arch(cls, arch: str) -> bool:
        return arch in cls.ggml_archs

    # -- tensor-parallel sharding ------------------------------------------

    @classmethod
    def shard_layer(cls, layer) -> list[str]:
        """Shard one decoder layer in place for the current TP rank.

        Returns the submodule names handled. Implementations must shard EVERY
        parameter of the layer consistently (mixed full/sharded submodules
        produce wrong all-reduce sums). Use the primitives from
        :mod:`voodoo_quant.parallel` (:func:`_shard_child`, :func:`tp_mlp`, ...).
        """
        return []

    # -- GGUF export mapping -------------------------------------------------

    @classmethod
    def gguf_name(cls, hf_name: str) -> Optional[str]:
        """Map an HF state-dict key to its GGUF tensor name, or None to fall
        back to the exporter's built-in mapping tables."""
        return None

    @classmethod
    def transpose_embedding(cls) -> bool:
        """True when llama.cpp stores the token embedding transposed
        ([embedding_dim, vocab] instead of [vocab, embedding_dim])."""
        return False

    # -- role grouping -------------------------------------------------------

    @classmethod
    def is_attention_tensor(cls, hf_name: str) -> bool:
        return any(seg in cls.attention_segments for seg in hf_name.split("."))
