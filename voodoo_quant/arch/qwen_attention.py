"""Tensor-parallel (TP) adapter for the Qwen3.5/3.8 full-attention layers.

The counterpart of :mod:`voodoo_quant.arch.qwen_ssm` for ``Qwen3_5Attention``
(the 16 full-attention layers of Qwen3.8-27B; the other 48 are
GatedDeltaNet/SSM):

* ``q_proj``: column-parallel on the query heads (24 -> 6/rank at tp=4).  Its
  output is per-head ``[q | gate]`` pairs (``out = Hq * head_dim * 2``), so a
  contiguous per-rank head range keeps each head's query and its sigmoid gate
  together and the stock ``view`` + ``chunk(2, -1)`` works on the local slice.
* ``k_proj`` / ``v_proj``: column-parallel on the kv heads (4 -> 1/rank at
  tp=4).  GQA groups are contiguous (q head h uses kv head ``h // (Hq/Hkv)``),
  so rank r's q heads exactly map onto rank r's kv heads — no cross-rank
  attention traffic.
* ``o_proj``: row-parallel (input columns ``[r*Hq_local*D, ...)``); each rank
  computes a partial ``[B, S, hidden]`` and the partials are SUM-all-reduced
  (``_RowParallelSum``, the single-process equivalent being the sum of copies).
* ``q_norm`` / ``k_norm`` are per-head RMSNorms (weight ``[head_dim]``), so
  they stay unsharded and simply run on the rank's head slice.
* The gated-attention sigmoid multiplies per-rank, then the rank-local
  ``o_proj`` partial is reduced — same ordering as the stock forward.

Like ``qwen_ssm``, everything is patched **in place** on the existing module
objects AFTER ``replace_linear_with_mixed_quant``: MixedQuant modules are
row/column-sharded through the additive ``shard_rows`` / ``shard_cols`` API, so
the candidate-cache keys under ``.cache/mixed_quant_candidates/`` keep the
full-tensor names and the baked checkpoint is the full model.

The attention core itself is NOT reimplemented: the patched forward calls the
configured attention interface (``rocm_triton`` / sdpa / eager) with the
rank-local head counts.  ``flash_attn`` handles 6:1 GQA (and 1 kv-head) plus
head_dim 256 natively on gfx908 — see
:func:`voodoo_quant.hardware.rocm_flash.register_rocm_triton_tp`, the TP-aware
re-registration of the ``rocm_triton`` backend that also fixes two bugs of the
original wrapper (return-layout and launch device).  This module never imports
that backend: CUDA-only / CPU users must not hit the ``flash_attn`` import.

Trainer usage: ``--tp_attention N`` (torchrun, one rank per GPU) or simulation
without a process group (correctness path).  ``--tensor_parallel N`` shards
attention (and everything else) through ``voodoo_quant.parallel.tp_patch_model``
BEFORE MixedQuant replacement instead — applying this adapter on top is
rejected.
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared TP machinery from the sibling adapter (same workstream; kept as the
# single source of truth for the autograd collectives and module sharding).
from voodoo_quant.arch.qwen_ssm import (
    TPContext,
    _ReplicatedInput,
    _RowParallelSum,
    _register_gate_sync,
    _shard_cols_of_module,
    _shard_rows_of_module,
    _proj_weight,
    resolve_tp_context,
)

__all__ = [
    "AttentionShardSpec",
    "apply_tp_attention",
]

_MODELING = None


def _modeling():
    """Cached import of transformers' qwen3_5 modeling module."""
    global _MODELING
    if _MODELING is None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        _MODELING = m
    return _MODELING


def _attn_class() -> type:
    return _modeling().Qwen3_5Attention


# ---------------------------------------------------------------------------
# Shard geometry
# ---------------------------------------------------------------------------

class AttentionShardSpec:
    """Per-rank head/row/column ranges of a ``Qwen3_5Attention`` layer."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int, world_size: int):
        if num_q_heads % world_size or num_kv_heads % world_size:
            raise ValueError(
                f"attention heads not divisible by TP world size: "
                f"Hq={num_q_heads}, Hkv={num_kv_heads}, ws={world_size}"
            )
        if (num_q_heads // num_kv_heads) * num_kv_heads != num_q_heads:
            raise ValueError(f"Hq={num_q_heads} is not a multiple of Hkv={num_kv_heads}")
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.q_per = num_q_heads // world_size
        self.kv_per = num_kv_heads // world_size

    def q_rows(self, r: int) -> list[tuple[int, int]]:
        """q_proj output rows of rank r (per-head q|gate pairs, hence 2*D)."""
        return [(r * self.q_per * 2 * self.head_dim, (r + 1) * self.q_per * 2 * self.head_dim)]

    def kv_rows(self, r: int) -> list[tuple[int, int]]:
        """k/v_proj output rows of rank r (one block per local kv head)."""
        return [(r * self.kv_per * self.head_dim, (r + 1) * self.kv_per * self.head_dim)]

    def o_cols(self, r: int) -> tuple[int, int]:
        """o_proj input columns of rank r (its q heads' attention outputs)."""
        return (r * self.q_per * self.head_dim, (r + 1) * self.q_per * self.head_dim)


# ---------------------------------------------------------------------------
# TP forward
# ---------------------------------------------------------------------------

def _tp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs,
):
    """TP replacement for ``Qwen3_5Attention.forward`` (training-time paths).

    Cached/generation paths are unsupported: with sharded heads the KV cache
    tensors would be per-rank, which the stock Cache object cannot represent.
    """
    spec: AttentionShardSpec = self._tp_spec
    ctx: TPContext = self._tp_ctx
    if past_key_values is not None:
        raise NotImplementedError(
            "voodoo_quant/arch/qwen_attention.py: KV caches are not supported by the attention "
            "tensor-parallel adapter; it is a training-time adapter (use_cache=False)."
        )
    if position_embeddings is None:
        raise ValueError("tp_attention: position_embeddings (cos, sin) are required")

    m = _modeling()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    interface: Any = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, m.eager_attention_forward
    )
    dropout = 0.0 if not self.training else self.attention_dropout
    cos, sin = position_embeddings
    B, S = hidden_states.shape[0], hidden_states.shape[1]
    D = self.head_dim

    if ctx.simulate:
        # Single-process correctness path: run the FULL projections once, then
        # slice each rank's heads out of them and run the interface per rank
        # with rank-local head counts (the exact numerics real TP performs).
        # Per-head ops (norms, rotary, gate) are head-local, so slicing the
        # full result equals computing on the shard.
        qg = self.q_proj(hidden_states).view(B, S, -1, 2 * D)
        q_full, gate_full = torch.chunk(qg, 2, dim=-1)
        q_full = self.q_norm(q_full).transpose(1, 2)  # [B, Hq, S, D]
        k_full = self.k_norm(self.k_proj(hidden_states).view(B, S, -1, D)).transpose(1, 2)
        v_full = self.v_proj(hidden_states).view(B, S, -1, D).transpose(1, 2)
        q_full, k_full = m.apply_rotary_pos_emb(q_full, k_full, cos, sin)

        out = None
        for r in range(ctx.world_size):
            qs, qe = r * spec.q_per, (r + 1) * spec.q_per
            ks, ke = r * spec.kv_per, (r + 1) * spec.kv_per
            o_r, _ = interface(
                self, q_full[:, qs:qe], k_full[:, ks:ke], v_full[:, ks:ke],
                attention_mask, dropout=dropout, scaling=self.scaling, **kwargs,
            )
            o_r = o_r.reshape(B, S, -1) * torch.sigmoid(gate_full[:, :, qs:qe, :].reshape(B, S, -1))
            c0, c1 = spec.o_cols(r)
            part = F.linear(o_r, _proj_weight(self.o_proj)[:, c0:c1])
            out = part if out is None else out + part
        return out, None

    # Real TP: this rank's projections were physically sharded, so q/k/v are
    # natively rank-local.  The input is wrapped so each rank's partial
    # d(hidden_states) is SUM-all-reduced in backward (replicated activations
    # must carry the full gradient into the previous layer); the o_proj partial
    # is reduced in forward.
    hs = _ReplicatedInput.apply(hidden_states, ctx.group)
    nq = spec.q_per
    q, gate = torch.chunk(self.q_proj(hs).view(B, S, nq, 2 * D), 2, dim=-1)
    gate = gate.reshape(B, S, -1)
    q = self.q_norm(q).transpose(1, 2)  # [B, Hq_local, S, D]
    k = self.k_norm(self.k_proj(hs).view(B, S, -1, D)).transpose(1, 2)
    v = self.v_proj(hs).view(B, S, -1, D).transpose(1, 2)
    q, k = m.apply_rotary_pos_emb(q, k, cos, sin)

    attn_output, _ = interface(
        self, q, k, v, attention_mask,
        dropout=dropout, scaling=self.scaling, **kwargs,
    )
    attn_output = attn_output.reshape(B, S, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    part = self.o_proj(attn_output)  # row-parallel partial [B, S, hidden]
    # A parallel.py-style MixedQuant built with a col shard_spec all-reduces in
    # its own forward; our post-replacement shard_cols()/Linear rebuilds never
    # do.  Reduce here unless the module already did (belt-and-braces: the
    # tp_patch_model path is rejected at patch time).
    o_spec = getattr(self.o_proj, "shard_spec", None)
    if not (o_spec is not None and o_spec.col_offset is not None):
        part = _RowParallelSum.apply(part, ctx.group)
    return part, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _shard_attention_module(mod: nn.Module, ctx: TPContext, sync_gates: bool = True) -> AttentionShardSpec:
    if getattr(mod, "_tp_spec", None) is not None:
        raise RuntimeError("tp_attention: module is already attention-sharded (double TP application)")
    # Guard against the parallel tp_patch_model workstream (--tensor_parallel):
    # its shards replace the projections with *Parallel* wrappers BEFORE
    # MixedQuant replacement (which then carry a shard_spec).  Sharding again
    # on top silently corrupts.
    for child_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        child = getattr(mod, child_name, None)
        if getattr(child, "tp_shard_spec", None) is not None or \
                getattr(child, "shard_spec", None) is not None or (
            child is not None and "Parallel" in type(child).__name__
        ):
            raise RuntimeError(
                f"tp_attention: {child_name} is already sharded by voodoo_quant.parallel "
                f"({type(child).__name__}); use either --tensor_parallel (tp_patch_model) "
                "or --tp_attention, never both."
            )

    spec = AttentionShardSpec(
        mod.config.num_attention_heads, mod.config.num_key_value_heads,
        mod.head_dim, ctx.world_size,
    )
    mod._tp_spec = spec
    mod._tp_ctx = ctx

    if ctx.real:
        r = ctx.rank
        mod.q_proj = _shard_rows_of_module(mod.q_proj, spec.q_rows(r))
        mod.k_proj = _shard_rows_of_module(mod.k_proj, spec.kv_rows(r))
        mod.v_proj = _shard_rows_of_module(mod.v_proj, spec.kv_rows(r))
        mod.o_proj = _shard_cols_of_module(mod.o_proj, spec.o_cols(r))
        # The stock eager/sdpa fallbacks repeat kv heads by this factor; with
        # rank-local heads it is (Hq/ws)/(Hkv/ws) == Hq/Hkv — unchanged — but
        # set it from the local geometry so it can never disagree with the
        # sharded projections.
        mod.num_key_value_groups = spec.q_per // spec.kv_per
        if sync_gates:
            for sub in (mod.q_proj, mod.k_proj, mod.v_proj, mod.o_proj):
                _register_gate_sync(sub, ctx.group)

    mod.forward = types.MethodType(_tp_attention_forward, mod)
    return spec


def apply_tp_attention(model: nn.Module, tp_size: int | None = None, group: Any = None,
                       sync_gates: bool = True, verbose: bool = True) -> list[nn.Module]:
    """Shard every Qwen3_5Attention (full-attention) layer in ``model``.

    Must run AFTER ``replace_linear_with_mixed_quant`` so the MixedQuant
    candidates are loaded from the full-tensor cache first; the adapter then
    row/column-shards those modules in place (tensor names stay untouched, so
    candidate-cache keys and the final bake are unaffected).

    Modes (mirroring ``voodoo_quant.arch.qwen_ssm.apply_ssm_tp``):
      * real TP — a >1-rank process group is initialized (torchrun) or ``group``
        is given: each rank keeps only its head group; o_proj partials and the
        replicated-input gradients are all-reduced; MixedQuant gate grads are
        SUM-all-reduced so the replicated gates receive the exact unsharded
        gradient.
      * simulation — no process group and ``tp_size > 1``: all shards are
        computed locally and summed (correctness path only, no memory split).

    Returns the list of patched modules (empty when there is nothing to do).
    """
    ctx = resolve_tp_context(tp_size, group)
    cls = _attn_class()
    mods = [m for m in model.modules() if isinstance(m, cls)]
    if ctx is None or not mods:
        if verbose:
            mode = "no TP context (world_size=1)" if ctx is None else "no full-attention layers"
            print(f"  [tp_attention] {mode}; nothing to do", flush=True)
        return []

    for m in mods:
        _shard_attention_module(m, ctx, sync_gates=sync_gates)

    if verbose:
        s = mods[0]._tp_spec
        if ctx.real:
            print(
                f"  [tp_attention] {len(mods)} attention layers sharded: "
                f"rank {ctx.rank}/{ctx.world_size} "
                f"({s.q_per} q-heads, {s.kv_per} kv-head per rank)"
                + (", gate-grad sync on" if sync_gates else ""),
                flush=True,
            )
        else:
            print(
                f"  [tp_attention] {len(mods)} attention layers in SIMULATION mode "
                f"(tp_size={ctx.world_size}, single process — correctness only)",
                flush=True,
            )
    return mods
