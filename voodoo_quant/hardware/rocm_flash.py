"""Vendor-specific flash-attention backend registration (ROCm / Triton-AMD).

Everything here is optional, AMD-GPU-only machinery: it registers the
``rocm_triton`` attention interface with transformers and downgrades
flash-linear-attention to a T-chunked torch fallback whose backward does not
fault on gfx908.  The generic sharding code (:mod:`voodoo_quant.parallel`,
:mod:`voodoo_quant.arch`) never imports this module; trainers and launch
scripts import it lazily so CUDA-only and CPU users never hit the
``flash_attn`` import.
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "register_flash_attn_triton",
    "register_rocm_triton",  # alias for the base registration
    "register_rocm_triton_tp",
]


def register_flash_attn_triton():
    """Register the curvedinf/flash-attention Triton-AMD backend with transformers.

    The backend (gfx908-capable, GQA + head_dim 256 verified) is selected via
    `--attn_implementation rocm_triton`.  Requires triton 3.2.x
    (the installed 3.6 crashes with illegal memory access) and the
    FLASH_ATTENTION_TRITON_AMD_ENABLE env var, set here before import.
    """
    name = "rocm_triton"
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if name in ALL_ATTENTION_FUNCTIONS:
        return
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    from flash_attn import flash_attn_func

    def _fwd(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        # transformers hands over [B, H, S, D]; flash_attn wants [B, S, H, D].
        # GQA (HQ != HKV) and causal masking are handled natively by the kernel.
        out = flash_attn_func(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            causal=bool(getattr(module, "is_causal", True)),
            dropout_p=dropout if module.training else 0.0,
        )
        return out.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register(name, _fwd)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, flash_attention_mask)

    # Downgrade flash-linear-attention to the torch fallback: FLA's
    # chunk_gated_delta_rule BACKWARD faults at Qwen3.8 SSM shapes (T=8192,
    # 16K/48V heads, 128-dim) on gfx908 — reproduced standalone 2026-08-15 —
    # corrupting memory inside the final model backward.  The transformers
    # Qwen3_5 module falls back to torch_chunk_gated_delta_rule when the
    # imported symbols are None.  Wrap the fallback to chunk over T (recurrent
    # state threaded via initial_state): forward-only peak drops from ~4 GiB
    # (T=8192) to ~1.6 GiB (T=2048), which fits a loaded shard.
    import transformers.models.qwen3_5.modeling_qwen3_5 as _q35m
    _raw_torch_chunk = _q35m.torch_chunk_gated_delta_rule

    def _t_chunked_gated_delta_rule(q, k, v, g, beta, chunk_size=64, initial_state=None,
                                    output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        T = q.shape[1]
        step = 2048
        outs, state = [], initial_state
        for t0 in range(0, T, step):
            t1 = min(T, t0 + step)
            o, state = _raw_torch_chunk(
                q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g[:, t0:t1], beta[:, t0:t1],
                chunk_size=chunk_size, initial_state=state,
                output_final_state=True,  # thread state across chunks
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outs.append(o)
        out = torch.cat(outs, dim=1)
        return out, (state if output_final_state else None)

    _q35m.chunk_gated_delta_rule = None
    _q35m.fused_recurrent_gated_delta_rule = None
    # Tag the raw fallback so voodoo_quant/arch/qwen_ssm.py can unwrap this
    # wrapper instead of double-chunking when it installs its own copy.
    _t_chunked_gated_delta_rule._voodoo_raw_torch_chunk = _raw_torch_chunk
    # patch instances' bound method lazily: modules capture the symbol at
    # __init__ (self.chunk_gated_delta_rule = chunk_gated_delta_rule or fallback);
    # since we None the module attr BEFORE model build, instances get the
    # fallback directly.  Patch the fallback itself so the T-chunking applies.
    _q35m.torch_chunk_gated_delta_rule = _t_chunked_gated_delta_rule


# Public alias: the trainer's historical private name stays importable, the
# public name is what new code should use.
register_flash_attn_triton  # referenced by register_rocm_triton_tp below


def register_rocm_triton_tp(name: str = "rocm_triton"):
    """(Re-)register the ``rocm_triton`` attention backend, TP-aware.

    Replaces the entry :func:`register_flash_attn_triton` installs (registering
    the same name overwrites it) with a wrapper that is safe under TP and fixes
    two bugs of the original:

    * return layout — the transformers attention-interface contract is
      ``[B, S, H, D]`` (see ``sdpa_attention_forward`` / ``eager_attention_
      forward``, which both ``transpose(1, 2)`` before returning; the stock
      ``Qwen3_5Attention.forward`` then does ``reshape(*input_shape, -1)``).
      The original wrapper returned flash_attn's native ``[B, S, H, D]``
      transposed to ``[B, H, S, D]``, which the reshape scrambles — verified
      on gfx908 against sdpa: rel error ~1.17 for the old convention vs
      ~5e-3 (bf16 noise) for the correct one.
    * launch device — the Triton-AMD fork launches its kernels on the CURRENT
      CUDA device.  With tensors on cuda:1..3 while cuda:0 is current the
      launch faults with ``hipErrorIllegalAddress`` (reproduced standalone on
      this host, forward AND backward); the wrapper pins the device to the
      query's.  Under torchrun TP each rank has already set its LOCAL_RANK
      device, so this is cheap insurance; under single-process pipeline
      sharding (attention layers on non-zero shards) it is required.

    Under TP the query/key/value arrive already rank-local (6 q-heads /
    1 kv-head per rank at ws=4 for the 27B config) — flash_attn handles that
    GQA ratio and head_dim 256 natively on gfx908, so NO sdpa/math fallback
    is needed; the wrapper is identical for the sharded and unsharded cases.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    from flash_attn import flash_attn_func

    def _fwd(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        # transformers hands over [B, H, S, D]; flash_attn wants [B, S, H, D].
        # GQA (HQ != HKV, including rank-local 6:1 / 1 kv-head) and causal
        # masking are handled natively by the kernel.
        with torch.cuda.device(query.device):
            out = flash_attn_func(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                causal=bool(getattr(module, "is_causal", True)),
                dropout_p=dropout if module.training else 0.0,
            )
        # flash_attn returns [B, S, H, D] — exactly the interface contract;
        # return it unchanged (the caller reshapes to [B, S, H*D]).
        return out.contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register(name, _fwd)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, flash_attention_mask)


# Alias used by the trainer (and launchers): the base, non-TP registration.
register_rocm_triton = register_flash_attn_triton
