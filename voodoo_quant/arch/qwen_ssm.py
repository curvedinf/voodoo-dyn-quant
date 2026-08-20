"""Tensor-parallel (TP) adapter for the Qwen3.5/3.8 GatedDeltaNet (linear attention / SSM) layers.

The class in ``transformers.models.qwen3_5.modeling_qwen3_5`` is named
``Qwen3_5GatedDeltaNet`` (a.k.a. "LinearAttention" layer); it is the SSM half of
Qwen3.8-27B's hybrid stack (48 of 64 layers).  This module shards one such layer
across TP ranks by *value-head groups*:

* ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``: row-sharded on
  the v-head boundary (48 v-heads -> 12/rank at tp=4; 16 k-heads -> 4/rank).
  K-heads repeat into v-head groups (``repeat_interleave(nv/nk)``), so sharding
  on the v-head boundary keeps every (k-head, v-head) pair inside one rank.
* ``conv1d``: per-head depthwise channels -> sharded identically to the qkv rows.
* ``A_log`` / ``dt_bias``: sharded by local head count.
* ``g`` / ``beta``: computed per-rank from the sharded ``a``/``b`` projections.
* ``chunk_gated_delta_rule``: runs per-rank on its local heads using the
  T-chunked torch fallback (the FLA kernels have a broken backward on gfx908 —
  see :func:`voodoo_quant.hardware.rocm_flash.register_flash_attn_triton`).
* ``out_proj``: row-parallel (Megatron sense: weight partitioned along the
  *input*/value dim, each rank holds ``[hidden, value_dim/tp]``, computes a
  partial ``[B, S, hidden]`` and the results are summed with an all-reduce).

Everything is patched **in place** on the existing module objects, so state-dict
names — and therefore the MixedQuant candidate-cache keys under
``.cache/mixed_quant_candidates/`` — are unchanged.  ``MixedQuantLinear``
modules are row/column-sharded through their additive ``shard_rows`` /
``shard_cols`` API *after* the full-tensor candidates were loaded from the
cache (llama.cpp block quantization is row-independent, so slicing candidate
rows/columns on 256-block boundaries is bit-identical to quantizing the slice).

Unit test idea (see ``tests/test_tp_ssm.py``): 2 ranks, 8 v-heads -> 4 v-heads
and 2 k-heads per rank; the full-sequence TP output must equal the unsharded
module's output (forward and ``hidden_states`` gradients), both in real
2-process mode (gloo/nccl) and in single-process simulation mode::

    python tests/test_tp_ssm.py              # simulate + kernel checks
    python tests/test_tp_ssm.py --mode dist  # spawns 2 gloo ranks on CPU

Trainer usage (per rank, under torchrun)::

    torchrun --nproc_per_node=4 <trainer> ... --tp_ssm 4

Without a process group, ``--tp_ssm N`` runs the same math in single-process
*simulation* mode (all N shards computed locally and summed): a correctness
path only — it does not split memory.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "T_CHUNK",
    "TPContext",
    "SSMShardSpec",
    "apply_ssm_tp",
    "install_t_chunked_fallback",
    "t_chunked_gated_delta_rule",
    "resolve_tp_context",
]

# Time-chunk size for the torch gated-delta fallback (matches
# voodoo_quant.hardware.rocm_flash.register_flash_attn_triton; T=2048 drops the
# forward-only peak of the fallback from ~4 GiB to ~1.6 GiB at T=8192).
T_CHUNK = 2048

_MODELING = None


def _modeling():
    """Cached import of transformers' qwen3_5 modeling module."""
    global _MODELING
    if _MODELING is None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        _MODELING = m
    return _MODELING


def _gdn_class() -> type:
    """The GatedDeltaNet / linear-attention class (name varies by transformers version)."""
    m = _modeling()
    cls = getattr(m, "Qwen3_5GatedDeltaNet", None) or getattr(m, "Qwen3_5LinearAttention")
    return cls


# ---------------------------------------------------------------------------
# T-chunked torch fallback for chunk_gated_delta_rule.
#
# FLA's chunk_gated_delta_rule BACKWARD faults at Qwen3.8 SSM shapes (T=8192,
# 16K/48V heads, 128-dim) on gfx908 — reproduced standalone 2026-08-15 —
# corrupting memory inside the final model backward.  The transformers module
# falls back to torch_chunk_gated_delta_rule when the imported FLA symbols are
# None; we wrap that fallback to chunk over T with the recurrent state threaded
# via initial_state (mirror of the wrapper in
# voodoo_quant.hardware.rocm_flash).
# ---------------------------------------------------------------------------

_RAW_TORCH_CHUNK = None


def _resolve_raw_torch_chunk():
    """The un-chunked transformers fallback, unwrapping any T-chunked wrapper.

    Both this module and ``voodoo_quant.hardware.rocm_flash.register_flash_attn_triton``
    tag their wrappers with ``_voodoo_raw_torch_chunk`` so whichever installs
    first, the other unwraps to the same raw function instead of double-chunking.
    """
    global _RAW_TORCH_CHUNK
    if _RAW_TORCH_CHUNK is None:
        fn = _modeling().torch_chunk_gated_delta_rule
        _RAW_TORCH_CHUNK = getattr(fn, "_voodoo_raw_torch_chunk", None) or fn
        t_chunked_gated_delta_rule._voodoo_raw_torch_chunk = _RAW_TORCH_CHUNK
    return _RAW_TORCH_CHUNK


def t_chunked_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kw,
):
    """torch_chunk_gated_delta_rule chunked over the time dim in T_CHUNK steps.

    The recurrent state is threaded across chunks via ``initial_state``.  Output
    is identical in structure to the raw fallback: ``(out, final_state_or_None)``.
    """
    raw = _resolve_raw_torch_chunk()
    T = q.shape[1]
    outs, state = [], initial_state
    for t0 in range(0, T, T_CHUNK):
        t1 = min(T, t0 + T_CHUNK)
        o, state = raw(
            q[:, t0:t1],
            k[:, t0:t1],
            v[:, t0:t1],
            g[:, t0:t1],
            beta[:, t0:t1],
            chunk_size=chunk_size,
            initial_state=state,
            output_final_state=True,  # thread state across chunks
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        outs.append(o)
    out = torch.cat(outs, dim=1)
    return out, (state if output_final_state else None)


def install_t_chunked_fallback():
    """Null out the FLA kernels and route the torch fallback through the T-chunked
    wrapper.  Idempotent; mirrors (and interops with)
    ``voodoo_quant.hardware.rocm_flash.register_flash_attn_triton``.

    Also fixes existing instances: the GatedDeltaNet ``__init__`` captures the
    kernel symbol at construction time, so modules built before this call keep
    holding whatever they captured.
    """
    m = _modeling()
    raw = _resolve_raw_torch_chunk()
    t_chunked_gated_delta_rule._voodoo_raw_torch_chunk = raw
    m.chunk_gated_delta_rule = None
    m.fused_recurrent_gated_delta_rule = None
    current = m.torch_chunk_gated_delta_rule
    if getattr(current, "_voodoo_raw_torch_chunk", None) is None:
        # Raw (un-chunked) fallback is still installed at module level: replace it.
        m.torch_chunk_gated_delta_rule = t_chunked_gated_delta_rule
    return m.torch_chunk_gated_delta_rule


# ---------------------------------------------------------------------------
# TP context
# ---------------------------------------------------------------------------


@dataclass
class TPContext:
    """Resolved tensor-parallel context used by the SSM adapter."""

    world_size: int
    rank: int
    group: Any = None  # dist process group for the all-reduce (None -> WORLD)
    simulate: bool = False  # single process, loop over all shards locally

    @property
    def real(self) -> bool:
        return self.world_size > 1 and not self.simulate


def resolve_tp_context(tp_size: int | None = None, group: Any = None) -> TPContext | None:
    """Real TP when a >1-rank process group is available, simulation when only
    ``tp_size > 1`` is given, ``None`` when there is nothing to shard."""
    if group is not None:
        return TPContext(dist.get_world_size(group), dist.get_rank(group), group, simulate=False)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        ws = dist.get_world_size()
        if tp_size not in (None, 1) and tp_size != ws:
            raise ValueError(
                f"--tp_ssm {tp_size} does not match the initialized process group world_size={ws}; "
                "pass a matching degree (or a subgroup via apply_ssm_tp(group=...))."
            )
        return TPContext(ws, dist.get_rank(), None, simulate=False)
    if tp_size is not None and tp_size > 1:
        return TPContext(tp_size, 0, None, simulate=True)
    return None


class _RowParallelSum(torch.autograd.Function):
    """y = sum_r x_r across the TP group; each rank contributes one partial.

    Every rank holds the identical summed output afterwards, so the backward is
    the identity: each rank's partial receives the same upstream gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        grp = group if group is not None else dist.group.WORLD
        if x.dtype == torch.bfloat16:
            try:
                backend = dist.get_backend(grp)
            except Exception:
                backend = ""
            if "gloo" in str(backend).lower():
                # gloo has no bf16 all-reduce; round-trip through fp32.
                y = x.float()
                dist.all_reduce(y, op=dist.ReduceOp.SUM, group=grp)
                return y.to(x.dtype)
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=grp)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _ReplicatedInput(torch.autograd.Function):
    """Identity forward; SUM-all-reduce the (partial) input gradient in backward.

    Real TP replicates the layer input on every rank, but each rank's backward
    through its head group produces only that rank's partial d(hidden_states);
    the replicated activation needs the summed gradient before it flows into
    the previous layer (the input-side counterpart of ``_RowParallelSum``,
    mirroring Megatron's ``f`` operator on a column-parallel linear).
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grp = ctx.group if ctx.group is not None else dist.group.WORLD
        if grad.dtype == torch.bfloat16:
            try:
                backend = dist.get_backend(grp)
            except Exception:
                backend = ""
            if "gloo" in str(backend).lower():
                g = grad.float()
                dist.all_reduce(g, op=dist.ReduceOp.SUM, group=grp)
                return g.to(grad.dtype), None
        g = grad.clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=grp)
        return g, None


# ---------------------------------------------------------------------------
# Shard geometry
# ---------------------------------------------------------------------------


class SSMShardSpec:
    """Row/column ranges of a GatedDeltaNet's per-head tensors for each rank.

    ``num_v_heads`` v-heads split into ``world_size`` contiguous groups; the
    k-heads (``num_k_heads``) repeat into v-head groups via
    ``repeat_interleave(nv // nk)``, so the k-heads needed by rank ``r`` are the
    contiguous block ``[r * k_per, (r+1) * k_per)`` — sharding on the v-head
    boundary keeps every (k, v) pair together.
    """

    def __init__(self, num_v_heads: int, num_k_heads: int, head_k_dim: int, head_v_dim: int,
                 world_size: int):
        if num_v_heads % world_size != 0 or num_k_heads % world_size != 0:
            raise ValueError(
                f"tp_ssm: heads must divide evenly by world_size={world_size} "
                f"(got {num_v_heads} v-heads / {num_k_heads} k-heads)."
            )
        v_per, k_per = num_v_heads // world_size, num_k_heads // world_size
        if v_per % k_per != 0:
            raise ValueError(
                f"tp_ssm: local v-heads ({v_per}) must be a multiple of local k-heads "
                f"({k_per}) so repeat_interleave stays within a rank."
            )
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.world_size = world_size
        self.v_per, self.k_per = v_per, k_per
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim

    # -- per-rank ranges -----------------------------------------------------
    def v_range(self, r: int) -> tuple[int, int]:
        return r * self.v_per, (r + 1) * self.v_per

    def k_range(self, r: int) -> tuple[int, int]:
        return r * self.k_per, (r + 1) * self.k_per

    def local_k_dim(self, r: int) -> int:
        return self.k_per * self.head_k_dim

    def local_v_dim(self, r: int) -> int:
        return self.v_per * self.head_v_dim

    def qkv_rows(self, r: int) -> list[tuple[int, int]]:
        """Output-row ranges of in_proj_qkv for rank r: [q | k | v] segments."""
        k0, k1 = self.k_range(r)
        v0, v1 = self.v_range(r)
        kd = self.key_dim
        return [
            (k0 * self.head_k_dim, k1 * self.head_k_dim),
            (kd + k0 * self.head_k_dim, kd + k1 * self.head_k_dim),
            (2 * kd + v0 * self.head_v_dim, 2 * kd + v1 * self.head_v_dim),
        ]

    def z_rows(self, r: int) -> list[tuple[int, int]]:
        v0, v1 = self.v_range(r)
        return [(v0 * self.head_v_dim, v1 * self.head_v_dim)]

    def head_rows(self, r: int) -> list[tuple[int, int]]:
        """Rows of in_proj_b / in_proj_a / A_log / dt_bias for rank r."""
        return [self.v_range(r)]

    def out_cols(self, r: int) -> tuple[int, int]:
        """Input-column range of out_proj (row-parallel) for rank r."""
        v0, v1 = self.v_range(r)
        return v0 * self.head_v_dim, v1 * self.head_v_dim


# ---------------------------------------------------------------------------
# Generic projection sharding (MixedQuant / nn.Linear / Q8Linear)
# ---------------------------------------------------------------------------


def _shard_rows_of_module(module: nn.Module, row_ranges: list[tuple[int, int]]) -> nn.Module:
    """Row-shard a projection module in place (output-feature ranges)."""
    total = sum(e - s for s, e in row_ranges)
    shard_rows = getattr(module, "shard_rows", None)  # MixedQuant additive API
    if callable(shard_rows):
        shard_rows(row_ranges)
        return module
    if isinstance(module, nn.Linear):
        w = module.weight.data
        new = nn.Linear(
            module.in_features, total, bias=module.bias is not None,
            device=w.device, dtype=w.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(torch.cat([w[s:e] for s, e in row_ranges], dim=0))
            if module.bias is not None:
                new.bias.copy_(torch.cat([module.bias.data[s:e] for s, e in row_ranges], dim=0))
        return new
    qb = getattr(module, "qweight", None)  # Q8Linear-style flat qbytes
    if isinstance(qb, torch.Tensor) and hasattr(module, "out_features"):
        row_bytes = qb.numel() // module.out_features
        if row_bytes * module.out_features != qb.numel():
            raise TypeError(f"tp_ssm: cannot row-shard {type(module).__name__} (ragged qbytes)")
        q2d = qb.view(module.out_features, row_bytes)
        new_qb = torch.cat([q2d[s:e] for s, e in row_ranges], dim=0).reshape(-1).contiguous()
        module.out_features = total
        module.qweight = new_qb.to(qb.device)  # name is a registered buffer -> stays a buffer
        module._w = None  # invalidate the dequant cache
        return module
    raise TypeError(f"tp_ssm: cannot row-shard projection of type {type(module).__name__}")


def _shard_cols_of_module(module: nn.Module, col_range: tuple[int, int]) -> nn.Module:
    """Column-shard a projection module in place (input-feature range).

    Used for the row-parallel ``out_proj``: each rank keeps every output row but
    only its local value-dim input columns.
    """
    s, e = col_range
    shard_cols = getattr(module, "shard_cols", None)  # MixedQuant additive API
    if callable(shard_cols):
        shard_cols(col_range)
        return module
    if isinstance(module, nn.Linear):
        w = module.weight.data
        new = nn.Linear(
            e - s, module.out_features, bias=module.bias is not None,
            device=w.device, dtype=w.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(w[:, s:e].contiguous())
            if module.bias is not None:
                new.bias.copy_(module.bias.data)
        return new
    qb = getattr(module, "qweight", None)
    if isinstance(qb, torch.Tensor) and hasattr(module, "in_features"):
        from voodoo_quant.ggml import get_quant_info

        info = get_quant_info("Q8_0")  # Q8Linear stores Q8_0 bytes
        nbpr = qb.numel() // module.out_features // info.block_bytes
        if nbpr * module.out_features * info.block_bytes != qb.numel():
            raise TypeError(f"tp_ssm: cannot column-shard {type(module).__name__} (ragged qbytes)")
        b0, b1 = s // info.block_size, e // info.block_size
        if s % info.block_size or e % info.block_size:
            raise ValueError(
                f"tp_ssm: column range [{s}, {e}) is not block-aligned "
                f"({info.block_size}); cannot slice quantized bytes."
            )
        q3 = qb.view(module.out_features, nbpr, info.block_bytes)[:, b0:b1, :]
        module.in_features = e - s
        module.qweight = q3.reshape(-1).contiguous().to(qb.device)
        module._w = None
        return module
    raise TypeError(f"tp_ssm: cannot column-shard projection of type {type(module).__name__}")


def _shard_conv1d(conv: nn.Conv1d, row_ranges: list[tuple[int, int]]) -> nn.Conv1d:
    """Rebuild the per-head depthwise conv on the rank's channel ranges."""
    w = conv.weight.data  # [conv_dim, 1, kernel]
    total = sum(e - s for s, e in row_ranges)
    kernel = conv.kernel_size[0]
    new = nn.Conv1d(
        total, total, kernel, groups=total,
        padding=conv.padding[0], bias=conv.bias is not None,
        device=w.device, dtype=w.dtype,
    )
    with torch.no_grad():
        new.weight.copy_(
            torch.cat([w[s:e] for s, e in row_ranges], dim=0).reshape(total, 1, kernel)
        )
        if conv.bias is not None:
            new.bias.copy_(torch.cat([conv.bias.data[s:e] for s, e in row_ranges], dim=0))
    return new


def _register_gate_sync(module: nn.Module, group: Any) -> None:
    """All-reduce (SUM) gate gradients of a sharded MixedQuant module.

    The gates are replicated across ranks, but each rank's backward only sees
    its own heads' contribution to the loss; the SUM-reduce reproduces exactly
    the unsharded gradient.  Disable with VOODOO_TP_GATE_SYNC=0.
    """
    if os.environ.get("VOODOO_TP_GATE_SYNC", "1") != "1":
        return
    g = getattr(module, "gates", None)
    if not isinstance(g, nn.Parameter) or getattr(g, "_voodoo_tp_sync", False):
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    grp = group if group is not None else dist.group.WORLD

    def _hook(p, _grp=grp):
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=_grp)

    try:
        g.register_post_accumulate_grad_hook(_hook)
        g._voodoo_tp_sync = True
    except Exception as exc:  # pragma: no cover - very old torch only
        print(f"  [tp_ssm] WARNING: gate-grad sync hook failed ({exc!r}); "
              "ranks may drift apart — call sync manually before step().", flush=True)


# ---------------------------------------------------------------------------
# TP forward
# ---------------------------------------------------------------------------


def _causal_conv_apply(mod: nn.Module, x_chw: torch.Tensor, seq_idx) -> torch.Tensor:
    """Causal depthwise conv + silu, exactly as the upstream forward does it."""
    if mod.causal_conv1d_fn is not None:
        return mod.causal_conv1d_fn(
            x=x_chw,
            weight=mod.conv1d.weight.squeeze(1),
            bias=mod.conv1d.bias,
            activation=mod.activation,
            seq_idx=seq_idx,
        )
    return F.silu(mod.conv1d(x_chw)[:, :, : x_chw.shape[-1]])


def _proj_weight(module: nn.Module) -> torch.Tensor:
    """Differentiable [out, in] weight of a projection (simulation path)."""
    gm = getattr(module, "get_mixed_weight", None)
    if callable(gm):
        return gm()
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor):
        return w
    qw = getattr(module, "_weight", None)  # Q8Linear
    if callable(qw):
        return qw()
    raise TypeError(f"tp_ssm: cannot extract weight from {type(module).__name__}")


def _run_shard(mod: nn.Module, spec: SSMShardSpec, r: int, conv_chs: torch.Tensor,
               z: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
               A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Shared per-shard math: post-conv channels -> normed core (pre out_proj).

    ``conv_chs`` is ``[B, S, C_local]`` (already conv+silu, channel order q|k|v
    of this shard); ``z``/``b``/``a``/``A_log``/``dt_bias`` are this shard's
    slices.  Returns the gated-normed core ``[B, S, v_dim_local]``.
    """
    B, S, _ = conv_chs.shape
    dk, dv = spec.head_k_dim, spec.head_v_dim
    kd_l, vd_l = spec.local_k_dim(r), spec.local_v_dim(r)
    q, k, v = torch.split(conv_chs, [kd_l, kd_l, vd_l], dim=-1)
    q = q.reshape(B, S, -1, dk)
    k = k.reshape(B, S, -1, dk)
    v = v.reshape(B, S, -1, dv)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf.
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    kh, vh = q.shape[2], v.shape[2]
    if vh // kh > 1:
        q = q.repeat_interleave(vh // kh, dim=2)
        k = k.repeat_interleave(vh // kh, dim=2)

    core_attn_out, _ = t_chunked_gated_delta_rule(
        q, k, v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out = core_attn_out.reshape(-1, dv)
    z = z.reshape(-1, dv)
    core_attn_out = mod.norm(core_attn_out, z)
    return core_attn_out.reshape(B, S, -1)


def _tp_gdn_forward(self, hidden_states: torch.Tensor, cache_params=None,
                    attention_mask: torch.Tensor | None = None, **kwargs):
    """TP replacement for Qwen3_5GatedDeltaNet.forward (training-time paths).

    Cached/generation paths are not supported: with sharded heads the cache
    tensors are per-rank, which the stock Cache object cannot represent.
    """
    spec: SSMShardSpec = self._tp_spec
    ctx: TPContext = self._tp_ctx

    if cache_params is not None:
        has_state = getattr(cache_params, "has_previous_state", None)
        if callable(has_state) and cache_params.has_previous_state(self.layer_idx):
            raise NotImplementedError(
                "voodoo_quant/arch/qwen_ssm.py: cached (generation) paths are not supported by "
                "the SSM tensor-parallel adapter; it is a training-time adapter."
            )

    m = _modeling()
    hs = m.apply_mask_to_padding_states(hidden_states, attention_mask)
    B, S, _ = hs.shape
    seq_idx = kwargs.get("seq_idx")

    if ctx.simulate:
        # Single-process correctness path: compute the FULL projections/conv
        # once, then slice each rank's channels/heads out of them.  Depthwise
        # conv and every per-head op are channel/head-local, so slicing the
        # full result equals computing on the shard.
        full_conv = _causal_conv_apply(self, self.in_proj_qkv(hs).transpose(1, 2), seq_idx)
        full_z = self.in_proj_z(hs)
        full_b = self.in_proj_b(hs)
        full_a = self.in_proj_a(hs)
        out = None
        for r in range(ctx.world_size):
            chs = torch.cat(
                [full_conv[:, s:e, :] for s, e in spec.qkv_rows(r)], dim=1
            ).transpose(1, 2)
            (z0, z1), = spec.z_rows(r)
            (h0, h1), = spec.head_rows(r)
            core = _run_shard(
                self, spec, r, chs,
                full_z[..., z0:z1], full_b[..., h0:h1], full_a[..., h0:h1],
                self.A_log[h0:h1], self.dt_bias[h0:h1],
            )
            c0, c1 = spec.out_cols(r)
            part = F.linear(core, _proj_weight(self.out_proj)[:, c0:c1])
            out = part if out is None else out + part
        return out

    # Real TP: this rank's modules were physically row/column-sharded, so the
    # projections natively produce only the local heads' channels.  The input is
    # wrapped so each rank's partial d(hidden_states) is SUM-all-reduced in
    # backward (the replicated activation must carry the full gradient).
    hs_red = _ReplicatedInput.apply(hs, ctx.group)
    chs = _causal_conv_apply(self, self.in_proj_qkv(hs_red).transpose(1, 2), seq_idx).transpose(1, 2)
    core = _run_shard(
        self, spec, ctx.rank, chs,
        self.in_proj_z(hs_red), self.in_proj_b(hs_red), self.in_proj_a(hs_red),
        self.A_log, self.dt_bias,
    )
    part = self.out_proj(core)  # row-parallel partial over the full hidden dim
    return _RowParallelSum.apply(part, ctx.group)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _shard_ssm_module(mod: nn.Module, ctx: TPContext, sync_gates: bool = True) -> SSMShardSpec:
    if getattr(mod, "_tp_spec", None) is not None:
        raise RuntimeError("tp_ssm: module is already SSM-sharded (double TP application)")
    # Guard against the parallel voodoo_quant.parallel workstream (tp_patch_model /
    # parallel.tp_ssm): its shards replace children with *Parallel* wrapper
    # classes carrying a tp_shard_spec.  Both paths shard the same heads;
    # applying both silently corrupts.  Pick ONE per run.
    for child_name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        child = getattr(mod, child_name, None)
        if getattr(child, "tp_shard_spec", None) is not None or (
            child is not None and "Parallel" in type(child).__name__
        ):
            raise RuntimeError(
                "tp_ssm: GatedDeltaNet projections already sharded by voodoo_quant.parallel "
                f"({child_name} is {type(child).__name__}); apply either tp_patch_model "
                "OR arch.qwen_ssm.apply_ssm_tp, never both."
            )
    spec = SSMShardSpec(
        mod.num_v_heads, mod.num_k_heads, mod.head_k_dim, mod.head_v_dim, ctx.world_size
    )
    mod._tp_spec = spec
    mod._tp_ctx = ctx
    # Instances capture the chunk kernel at __init__; make sure they use the
    # T-chunked fallback even if they were built before any patch ran.  (The
    # TP forward calls t_chunked_gated_delta_rule directly, so this is mostly
    # belt-and-braces for anyone poking at the module by hand.)
    mod.chunk_gated_delta_rule = t_chunked_gated_delta_rule

    if ctx.real:
        r = ctx.rank
        mod.in_proj_qkv = _shard_rows_of_module(mod.in_proj_qkv, spec.qkv_rows(r))
        mod.in_proj_z = _shard_rows_of_module(mod.in_proj_z, spec.z_rows(r))
        mod.in_proj_b = _shard_rows_of_module(mod.in_proj_b, spec.head_rows(r))
        mod.in_proj_a = _shard_rows_of_module(mod.in_proj_a, spec.head_rows(r))
        mod.out_proj = _shard_cols_of_module(mod.out_proj, spec.out_cols(r))
        mod.conv1d = _shard_conv1d(mod.conv1d, spec.qkv_rows(r))
        (h0, h1), = spec.head_rows(r)
        mod.A_log = nn.Parameter(
            mod.A_log.detach()[h0:h1].clone(), requires_grad=mod.A_log.requires_grad
        )
        mod.dt_bias = nn.Parameter(
            mod.dt_bias.detach()[h0:h1].clone(), requires_grad=mod.dt_bias.requires_grad
        )
        if sync_gates:
            for sub in (mod.in_proj_qkv, mod.in_proj_z, mod.in_proj_b, mod.in_proj_a,
                        mod.out_proj):
                _register_gate_sync(sub, ctx.group)

    mod.forward = types.MethodType(_tp_gdn_forward, mod)
    return spec


def apply_ssm_tp(model: nn.Module, tp_size: int | None = None, group: Any = None,
                 sync_gates: bool = True, verbose: bool = True) -> list[nn.Module]:
    """Shard every Qwen3.5 GatedDeltaNet (linear-attention) layer in ``model``.

    Must run AFTER ``replace_linear_with_mixed_quant`` so the MixedQuant
    candidates are loaded from the full-tensor cache first; the adapter then
    row/column-shards those modules in place (tensor names stay untouched).

    Modes:
      * real TP — a >1-rank process group is initialized (torchrun) or ``group``
        is given: each rank keeps only its head group and ``out_proj`` partials
        are all-reduced; MixedQuant gate grads are SUM-all-reduced so the
        replicated gates receive the exact unsharded gradient.
      * simulation — no process group and ``tp_size > 1``: all shards are
        computed locally and summed (correctness path only).

    Returns the list of patched modules (empty when there is nothing to do).
    """
    ctx = resolve_tp_context(tp_size, group)
    cls = _gdn_class()
    mods = [m for m in model.modules() if isinstance(m, cls)]
    if ctx is None or not mods:
        if verbose:
            mode = "no TP context (world_size=1)" if ctx is None else "no GatedDeltaNet layers"
            print(f"  [tp_ssm] {mode}; nothing to do", flush=True)
        return []

    install_t_chunked_fallback()
    for m in mods:
        _shard_ssm_module(m, ctx, sync_gates=sync_gates)

    if verbose:
        if ctx.real:
            print(
                f"  [tp_ssm] {len(mods)} GatedDeltaNet layers sharded: "
                f"rank {ctx.rank}/{ctx.world_size} "
                f"({mods[0]._tp_spec.v_per} v-heads, {mods[0]._tp_spec.k_per} k-heads, "
                f"{mods[0]._tp_spec.local_v_dim(ctx.rank)} value dims per rank)"
                + (", gate-grad sync on" if sync_gates else ""),
                flush=True,
            )
        else:
            print(
                f"  [tp_ssm] {len(mods)} GatedDeltaNet layers in SIMULATION mode "
                f"(tp_size={ctx.world_size}, single process — correctness only)",
                flush=True,
            )
    return mods
