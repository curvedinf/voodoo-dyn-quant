"""
Differentiable mixed-precision quantization layers for Voodoo dynamic quant
selection.

`MixedQuantLinear` and `MixedQuantEmbedding` store K pre-dequantized candidate
weight matrices for a source layer plus K learnable gates.  During forward they
return a softmax-weighted mixture of the candidates, so the output is smooth in
the gates.  After training, the tensor is hard-assigned to the candidate with
the largest gate value.
"""

from __future__ import annotations

import hashlib
import ctypes
import gc
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import voodoo_quant
from voodoo_quant.ggml import bytes_per_weight, quantize_tensor, dequantize_tensor, get_quant_info
from voodoo_quant.quants.gpu_dequant import (
    _dequantize_iq2_xxs_gpu, _dequantize_iq2_xs_gpu, _dequantize_iq2_s_gpu,
    _dequantize_iq3_xxs_gpu, _dequantize_iq3_s_gpu, _dequantize_iq1_s_gpu,
    _dequantize_iq1_m_gpu, _dequantize_iq4_xs_gpu,
)


# ---------------------------------------------------------------------------
# Optimization 1: PyTorch-native GPU dequantization
#
# Upload quantized bytes to GPU and dequantize using vectorized torch ops.
# This eliminates the CPU ggml dequant + large fp32 H2D copy, replacing it
# with a small qbytes H2D copy + fast GPU compute.
# ---------------------------------------------------------------------------


def _tp_handshake(_dist):
    """Barrier that works on gfx908/NCCL-P2P-disabled: tiny all-reduce on a
    CUDA tensor on each rank's own device (plain dist.barrier() can hang
    here)."""
    import torch as _t
    _dev = _t.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", "0")))
    _one = _t.ones(1, device=_dev)
    _dist.all_reduce(_one)
    return


def _f16_to_f32(t):
    """Convert a fp16 tensor (stored as uint16) to float32."""
    return t.view(torch.float16).to(torch.float32)


def _dequantize_q8_0_gpu(qbytes_gpu, out_features, in_features):
    """Dequantize Q8_0 on GPU. Block: 2 bytes (fp16 d) + 32 int8 qs = 34 bytes per 32 weights."""
    block_bytes = 34
    qk = 32
    nblocks_per_row = in_features // qk
    nblocks = out_features * nblocks_per_row
    # Flatten and reshape to [nblocks, block_bytes]
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    # First 2 bytes: fp16 scale d -> [nblocks] (squeeze the trailing dim from view)
    d = raw[:, :2].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    # Next 32 bytes: int8 quants
    qs = raw[:, 2:2+qk].view(torch.int8).to(torch.float32)  # [nblocks, qk]
    y = qs * d.unsqueeze(1)  # [nblocks, qk]
    return y.reshape(out_features, in_features)


def _dequantize_q4_K_gpu(qbytes_gpu, out_features, in_features):
    """Dequantize Q4_K on GPU. Block: 2 fp16 (d, dmin) + K_SCALE_SIZE + 128 bytes = 144 bytes per 256 weights."""
    QK_K = 256
    K_SCALE_SIZE = 12
    block_bytes = 144
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = raw[:, :2].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    dmin = raw[:, 2:4].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    scales_raw = raw[:, 4:4+K_SCALE_SIZE]
    qs = raw[:, 4+K_SCALE_SIZE:4+K_SCALE_SIZE+QK_K//2]

    # Unpack scales using get_scale_min_k4 logic from ggml:
    # For j < 4:  d = q[j] & 63;          m = q[j+4] & 63
    # For j >= 4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);  m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
    # q is the 12-byte scales array
    sc = torch.zeros(nblocks, 8, device=qbytes_gpu.device, dtype=torch.float32)
    m = torch.zeros(nblocks, 8, device=qbytes_gpu.device, dtype=torch.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = (scales_raw[:, j].to(torch.int32) & 0x3F).to(torch.float32)
            m[:, j] = (scales_raw[:, j + 4].to(torch.int32) & 0x3F).to(torch.float32)
        else:
            sc[:, j] = ((scales_raw[:, j + 4].to(torch.int32) & 0xF) |
                        ((scales_raw[:, j - 4].to(torch.int32) >> 6) << 4)).to(torch.float32)
            m[:, j] = ((scales_raw[:, j + 4].to(torch.int32) >> 4) |
                       ((scales_raw[:, j].to(torch.int32) >> 6) << 4)).to(torch.float32)

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)
    # Q4_K: 4 iterations of 64 weights. Each uses 32 bytes of qs (lower+upper nibbles).
    # is = 0,2,4,6 -> get_scale_min_k4(is+0), get_scale_min_k4(is+1)
    for j in range(4):
        is_val = 2 * j
        q_chunk = qs[:, j*32:(j+1)*32].to(torch.int32)
        lo_chunk = (q_chunk & 0xF).to(torch.float32)
        hi_chunk = (q_chunk >> 4).to(torch.float32)
        d1 = (d * sc[:, is_val]).unsqueeze(1)
        m1 = (dmin * m[:, is_val]).unsqueeze(1)
        d2 = (d * sc[:, is_val + 1]).unsqueeze(1)
        m2 = (dmin * m[:, is_val + 1]).unsqueeze(1)
        y[:, j*64:j*64+32] = d1 * lo_chunk - m1
        y[:, j*64+32:j*64+64] = d2 * hi_chunk - m2

    return y.reshape(out_features, in_features)


def _dequantize_q5_K_gpu(qbytes_gpu, out_features, in_features):
    """Dequantize Q5_K on GPU. Block: 2 fp16 (d, dmin) + K_SCALE_SIZE + 32 (qh) + 128 (qs) = 176 bytes per 256 weights."""
    QK_K = 256
    K_SCALE_SIZE = 12
    block_bytes = 176
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = raw[:, :2].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    dmin = raw[:, 2:4].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    scales_raw = raw[:, 4:4+K_SCALE_SIZE]
    qh = raw[:, 4+K_SCALE_SIZE:4+K_SCALE_SIZE+QK_K//8]
    qs = raw[:, 4+K_SCALE_SIZE+QK_K//8:4+K_SCALE_SIZE+QK_K//8+QK_K//2]

    # Unpack scales using get_scale_min_k4 (same as Q4_K)
    sc = torch.zeros(nblocks, 8, device=qbytes_gpu.device, dtype=torch.float32)
    m = torch.zeros(nblocks, 8, device=qbytes_gpu.device, dtype=torch.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = (scales_raw[:, j].to(torch.int32) & 0x3F).to(torch.float32)
            m[:, j] = (scales_raw[:, j + 4].to(torch.int32) & 0x3F).to(torch.float32)
        else:
            sc[:, j] = ((scales_raw[:, j + 4].to(torch.int32) & 0xF) |
                        ((scales_raw[:, j - 4].to(torch.int32) >> 6) << 4)).to(torch.float32)
            m[:, j] = ((scales_raw[:, j + 4].to(torch.int32) >> 4) |
                       ((scales_raw[:, j].to(torch.int32) >> 6) << 4)).to(torch.float32)

    # qh is indexed by l=0..31 per iteration with bitmask u1/u2 (no pointer advance)
    # No precomputation needed — qh bits are extracted inline in the loop.

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)
    # Q5_K: 4 iterations of 64 weights. Each uses 32 bytes of qs (lower+upper nibbles)
    # plus 1 bit from qh. u1/u2 are bitmasks that shift left by 2 each iteration.
    # u1 starts at 1, u2 starts at 2.
    for j in range(4):
        is_val = 2 * j
        u1 = 1 << (2 * j)  # bit mask for lower nibble high bit
        u2 = 2 << (2 * j)  # bit mask for upper nibble high bit
        q_chunk = qs[:, j*32:(j+1)*32].to(torch.int32)
        lo = (q_chunk & 0xF).to(torch.float32)
        hi = (q_chunk >> 4).to(torch.float32)
        # High bit: qh[l] & u1 for lower, qh[l] & u2 for upper
        qh_chunk = qh[:, :32].to(torch.int32)  # only first 32 bytes used per iter? No...
        # Actually in C: qh[l] is indexed by l=0..31, and qh pointer doesn't advance!
        # The u1/u2 bitmasks select different bits from the same qh bytes.
        bit_lo = ((qh_chunk & u1) != 0).to(torch.float32) * 16
        bit_hi = ((qh_chunk & u2) != 0).to(torch.float32) * 16
        d1 = (d * sc[:, is_val])
        m1 = (dmin * m[:, is_val])
        d2 = (d * sc[:, is_val + 1])
        m2 = (dmin * m[:, is_val + 1])
        y[:, j*64:j*64+32] = d1.unsqueeze(1) * (lo + bit_lo) - m1.unsqueeze(1)
        y[:, j*64+32:j*64+64] = d2.unsqueeze(1) * (hi + bit_hi) - m2.unsqueeze(1)

    return y.reshape(out_features, in_features)


def _dequantize_q6_K_gpu(qbytes_gpu, out_features, in_features):
    """Dequantize Q6_K on GPU. Block layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes per 256 weights."""
    QK_K = 256
    block_bytes = 210
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    # Struct order: ql, qh, scales, d (d is at the END!)
    ql = raw[:, 0:QK_K//2]
    qh = raw[:, QK_K//2:QK_K//2+QK_K//4]
    scales = raw[:, QK_K//2+QK_K//4:QK_K//2+QK_K//4+QK_K//16].view(torch.int8).to(torch.float32)
    d = raw[:, QK_K//2+QK_K//4+QK_K//16:QK_K//2+QK_K//4+QK_K//16+2].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)
    # Q6_K: 2 iterations of 128 weights. Each uses 64 bytes of ql + 32 bytes of qh.
    # Within each iteration: 4 groups of 32 weights, using ql[l], ql[l+32], and qh[l] bits.
    for n in range(2):
        ql_chunk = ql[:, n*64:(n+1)*64].to(torch.int32)  # [nblocks, 64]
        qh_chunk = qh[:, n*32:(n+1)*32].to(torch.int32)  # [nblocks, 32]
        for l in range(32):
            is_val = l // 16
            q1 = ((ql_chunk[:, l] & 0xF) | (((qh_chunk[:, l] >> 0) & 3) << 4)).to(torch.float32) - 32.0
            q2 = ((ql_chunk[:, l + 32] & 0xF) | (((qh_chunk[:, l] >> 2) & 3) << 4)).to(torch.float32) - 32.0
            q3 = ((ql_chunk[:, l] >> 4) | (((qh_chunk[:, l] >> 4) & 3) << 4)).to(torch.float32) - 32.0
            q4 = ((ql_chunk[:, l + 32] >> 4) | (((qh_chunk[:, l] >> 6) & 3) << 4)).to(torch.float32) - 32.0
            sc_idx = n * 8
            y[:, n*128 + l] = d * scales[:, sc_idx + is_val + 0] * q1
            y[:, n*128 + l + 32] = d * scales[:, sc_idx + is_val + 2] * q2
            y[:, n*128 + l + 64] = d * scales[:, sc_idx + is_val + 4] * q3
            y[:, n*128 + l + 96] = d * scales[:, sc_idx + is_val + 6] * q4

    return y.reshape(out_features, in_features)


# Registry of GPU dequant functions
# Types marked as verified produce bit-identical results to CPU ggml dequant.
# Unverified types fall back to CPU dequant to avoid incorrect training.
_GPU_DEQUANT_VERIFIED = {
    "Q8_0", "Q4_K", "Q5_K", "Q6_K",  # K-quants
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S",  # IQ types
    "IQ1_S", "IQ1_M", "IQ4_XS",  # IQ types
}

def _heap_dump_signum(signum, frame):
    """SIGUSR1 handler: census of live GC objects by type + rough tensor bytes.

    Used to hunt host-anon leaks during long inits (attach-free; ptrace is
    blocked on this host). Log lines go to stdout (the shared train log).
    """
    import collections
    import io

    cnt = collections.Counter()
    tensor_bytes = collections.Counter()
    for obj in gc.get_objects():
        try:
            cnt[type(obj).__name__] += 1
            if torch.is_tensor(obj):
                cnt["__tensor_bytes__" + str(obj.dtype)] += obj.numel() * obj.element_size()
                if obj.device.type == "cpu":
                    tensor_bytes[str(obj.dtype)] += obj.numel() * obj.element_size()
        except Exception:
            pass
    buf = io.StringIO()
    print("\n[heap-dump] top types:", flush=True, file=buf)
    for name, n in cnt.most_common(12):
        print(f"  {name}: {n}", file=buf)
    print("[heap-dump] cpu tensor bytes by dtype:", flush=True, file=buf)
    for name, n in tensor_bytes.most_common(8):
        print(f"  {name}: {n/1048576:.0f} MiB", file=buf)
    # anon-vs-file + pinned breakdown for CPU tensors (init-leak forensics):
    # mmap-backed storages resolve to a file offset; anon ones don't.
    pinned = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cpu" and obj.is_pinned():
                pinned += obj.numel() * obj.element_size()
        except Exception:
            pass
    print(f"[heap-dump] pinned cpu bytes: {pinned/1048576:.0f} MiB", file=buf)
    print(buf.getvalue(), flush=True)


def install_heap_dump_handler():
    try:
        signal.signal(signal.SIGUSR1, _heap_dump_signum)
    except Exception:
        pass


# Arm the census handler at import in EVERY rank: a staggered-slow rank that
# has not yet reached the replacement phase must still survive a probe signal.
install_heap_dump_handler()


_GPU_DEQUANT_REGISTRY = {
    "Q8_0": _dequantize_q8_0_gpu,
    "Q4_K": _dequantize_q4_K_gpu,
    "Q5_K": _dequantize_q5_K_gpu,
    "Q6_K": _dequantize_q6_K_gpu,
    "IQ2_XXS": _dequantize_iq2_xxs_gpu,
    "IQ2_XS": _dequantize_iq2_xs_gpu,
    "IQ2_S": _dequantize_iq2_s_gpu,
    "IQ3_XXS": _dequantize_iq3_xxs_gpu,
    "IQ3_S": _dequantize_iq3_s_gpu,
    "IQ1_S": _dequantize_iq1_s_gpu,
    "IQ1_M": _dequantize_iq1_m_gpu,
    "IQ4_XS": _dequantize_iq4_xs_gpu,
}


def _quant_info_for(quant_type: str):
    """Block size/bytes for a quant type (thin alias for readability)."""
    return get_quant_info(quant_type)


def dequantize_tensor_gpu(qweight, quant_type, out_features, in_features, device):
    """
    Dequantize quantized bytes on GPU using PyTorch-native operations.

    Uploads the small quantized bytes to GPU, then dequantizes using vectorized
    torch ops. This is much faster than CPU ggml dequant + fp32 H2D copy because:
    1. The H2D copy is 5-20x smaller (quantized bytes vs fp32)
    2. Dequantization runs on the GPU (which was idle at 0% utilization)

    Falls back to CPU dequant for types without a GPU implementation (IQ types).
    """
    if quant_type not in _GPU_DEQUANT_REGISTRY or quant_type not in _GPU_DEQUANT_VERIFIED:
        # Fall back to CPU dequant for unverified or unsupported types
        w = dequantize_tensor(qweight, quant_type, out_features, in_features, pin_memory=True)
        return w[:, :in_features].to(device=device, dtype=torch.bfloat16, non_blocking=False)

    dequant_fn = _GPU_DEQUANT_REGISTRY[quant_type]
    # Upload quantized bytes to GPU (small: ~5-20x smaller than fp32)
    qweight_gpu = qweight.to(device=device, non_blocking=False)
    # Dequantize on GPU
    w = dequant_fn(qweight_gpu, out_features, in_features)
    return w[:, :in_features].to(torch.bfloat16)


class _MixedWeightFunction(Function):
    """Autograd function for mixing CPU-resident candidates without holding them on GPU.

    During training the K quantized candidates for a tensor are kept on CPU.  If we
    mixed them with ordinary PyTorch operations, autograd would save every candidate
    on the compute device (because they are constants used to compute gradients for
    the softmax probabilities).  That explodes GPU memory: K * #tensors * weight_size.

    This function instead moves one candidate to the compute device at a time during
    both forward and backward, so the live GPU footprint is just the final mixed
    weight plus one candidate transient.
    """

    @staticmethod
    def forward(ctx, source_dtype, probs, *candidates):
        # candidates are CPU tensors in the source dtype (e.g. bf16), sliced to the
        # original [out_features, in_features] shape.
        device = probs.device
        mixed = torch.zeros(
            candidates[0].shape[0],
            candidates[0].shape[1],
            dtype=torch.float32,
            device=device,
        )
        for k, cand in enumerate(candidates):
            # Cast to float32 on the target device; only this candidate is GPU-resident.
            mixed = mixed + probs[k] * cand.to(device=device, dtype=torch.float32)
        ctx.source_dtype = source_dtype
        ctx.candidates = candidates
        ctx.device = device
        return mixed.to(source_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is source_dtype on the compute device.
        grad_output = grad_output.to(torch.float32)
        candidates = ctx.candidates
        device = ctx.device
        grad_probs = torch.zeros(len(candidates), device=device, dtype=torch.float32)
        for k, cand in enumerate(candidates):
            grad_probs[k] = (grad_output * cand.to(device=device, dtype=torch.float32)).sum()
        return (None, grad_probs) + (None,) * len(candidates)


class _RoutedMixedLinear(Function):
    """PTQR forward: y[t] = x[t] @ W_{k*(t)}^T, ONE candidate per token.

    Rows are grouped by the per-token Gumbel sample (see _sample_token_routing)
    and each group runs only its candidate's matmul, so no candidate averaging
    — and therefore no within-token mixture (Jensen) gain — occurs anywhere in
    the model.  Total matmul FLOPs equal a normal forward; candidates stream
    to the compute device one at a time like _MixedWeightFunction.

    Backward is the straight-through mixture Jacobian evaluated at the routed
    point (the ST identity y_routed + y_mix - sg(y_mix), with y_mix never
    materialized):
        grad_p_k = <grad_out, x @ W_k^T> = ((grad_out^T x) * W_k).sum()
        grad_x   = grad_out @ (sum_k p_k W_k)
    """

    @staticmethod
    def forward(ctx, source_dtype, probs, routing, x, *candidates):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        device = x2.device
        routing = routing.to(device)
        xf = x2.to(torch.float32)
        out = torch.zeros(x2.shape[0], candidates[0].shape[0], dtype=torch.float32, device=device)
        for k, cand in enumerate(candidates):
            rows = torch.nonzero(routing == k, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            w = cand.to(device=device, dtype=torch.float32)
            out[rows] = xf[rows] @ w.t()
            del w
        ctx.source_dtype = source_dtype
        ctx.probs = probs.detach()
        ctx.probs_device = probs.device
        ctx.x2 = x2
        ctx.x_dtype = x2.dtype
        ctx.candidates = candidates
        ctx.x_shape = orig_shape
        ctx.device = device
        y = out.to(source_dtype)
        return y if len(orig_shape) == 2 else y.reshape(*orig_shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        G = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
        xf = ctx.x2.to(torch.float32)
        device = ctx.device
        probs = ctx.probs.to(device)
        Gx = G.t() @ xf  # [out, in], shared by every grad_p_k below
        grad_probs = torch.zeros(len(ctx.candidates), device=device, dtype=torch.float32)
        w_mix = torch.zeros(
            ctx.candidates[0].shape[0], ctx.candidates[0].shape[1],
            dtype=torch.float32, device=device,
        )
        for k, cand in enumerate(ctx.candidates):
            w = cand.to(device=device, dtype=torch.float32)
            w_mix = w_mix + probs[k] * w
            grad_probs[k] = (Gx * w).sum()
            del w
        grad_x = (G @ w_mix).to(ctx.x_dtype)
        if len(ctx.x_shape) != 2:
            grad_x = grad_x.reshape(ctx.x_shape)
        return (None, grad_probs.to(ctx.probs_device), None, grad_x) + (None,) * len(ctx.candidates)


# --- Parallel-dequant helpers for the lazy recompute path --------------------
# Dequantize (CPU ggml) is the dominant cost of lazy training: a 4B layer with
# K=12 candidates re-dequantizes ~178 GB of fp32 per forward.  The candidates are
# independent, so we dequantize them concurrently (ctypes releases the GIL, the
# same ggml path the cache builder already parallelizes safely).  Only the CPU
# dequantize is parallelized; the CUDA H2D copy + reduction stay on the calling
# thread in candidate order so the fp32 summation order (and therefore the
# forward output and grad_probs) is bit-identical to the sequential version.
# The non-lazy path above is untouched.  Worker count is bounded so at most
# ``workers + 1`` candidates are materialized at once (~workers * 94 MB worst
# case for a 9216x2560 layer).
_LAZY_EXEC = None
_LAZY_EXEC_N = 0


def _get_lazy_executor():
    global _LAZY_EXEC, _LAZY_EXEC_N
    import concurrent.futures as _cf
    import os as _os
    n = int(_os.environ.get("VOODOO_DEQUANT_WORKERS", str(min(8, _os.cpu_count() or 2))))
    n = max(1, n)
    if _LAZY_EXEC is None or _LAZY_EXEC_N != n:
        _LAZY_EXEC = _cf.ThreadPoolExecutor(max_workers=n)
        _LAZY_EXEC_N = n
    return _LAZY_EXEC


def _dequant_slice(qb, qt, out_features, padded_in_features, in_features):
    cand = dequantize_tensor(qb, qt, out_features, padded_in_features, pin_memory=True)
    return cand[:, :in_features]


def _lazy_submit(qbytes, candidate_types, out_features, padded_in_features, in_features):
    ex = _get_lazy_executor()
    return [
        ex.submit(_dequant_slice, qb, candidate_types[k], out_features, padded_in_features, in_features)
        for k, qb in enumerate(qbytes)
    ]


def _lazy_dequant_one(qb, qt, out_features, padded_in_features, in_features, device):
    """Dequantize a single candidate, using GPU dequant when available."""
    if device.type == "cuda" and qt in _GPU_DEQUANT_REGISTRY:
        return dequantize_tensor_gpu(qb, qt, out_features, padded_in_features, device)
    # CPU fallback (with pinned memory for async transfer)
    cand = dequantize_tensor(qb, qt, out_features, padded_in_features, pin_memory=True)
    return cand[:, :in_features]


class _MixedWeightFunctionLazy(Function):
    """Lazy (recompute-on-backward) variant of :class:`_MixedWeightFunction`.

    In lazy mode the persistent per-tensor storage is the *quantized bytes*
    (``_qweight_k``), which already live on CPU regardless of autograd.  The eager
    function saves the *dequantized* fp32 candidates in its ``ctx`` for backward;
    for a 27B model with K=12 candidates that pins ~1.3 TB of transient fp32 across
    all layers.  This function never saves dequantized candidates: it keeps only
    references to the persistent qbytes (no new memory) and re-dequantizes one
    candidate at a time inside both forward and backward.  Forward output and
    ``grad_probs`` are therefore identical to the eager-save lazy path, while peak
    RAM stays near the qbytes cache plus a single layer's transient.

    Optimizations applied:
    - Opt 1: GPU-side dequantization for supported quant types (Q8_0, Q4_K, Q5_K, Q6_K)
    - Opt 2: Pinned memory + async H2D transfers with dedicated CUDA stream
    - Opt 3: Forward candidates cached for backward pass reuse

    The non-lazy path (:class:`_MixedWeightFunction`) is left unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        source_dtype,
        candidate_types,
        out_features,
        padded_in_features,
        in_features,
        probs,
        layer_ref,
        *qbytes,
    ):
        device = probs.device
        mixed = torch.zeros(
            out_features,
            in_features,
            dtype=torch.bfloat16,
            device=device,
        )
        # The cache stores CPU fp32 candidates (or GPU if Opt 1 was used).
        fwd_cache = getattr(layer_ref, "_fwd_candidates", None)
        use_gpu_dequant = device.type == "cuda"

        # Optimization 2: dedicated H2D copy stream for non-GPU-dequant path.
        # NOTE: no dedicated H2D stream on ROCm.  Mixing a side stream with the
        # autograd engine's default-stream AccumulateGrad nodes produced HSA
        # aperture violations during the final aggregated backward; synchronous
        # copies are only marginally slower and are stream-safe.
        h2d_stream = None

        for k in range(len(qbytes)):
            qt = candidate_types[k]
            qb = qbytes[k]

            # Optimization 1: try GPU dequant first.  For very large tensors
            # (vocabulary heads), dequantize in row blocks and cast to bf16
            # per block: the whole-matrix fp32 transient is ~2x the bf16
            # weight (4.74 GiB for 248320x5120) and OOMs a full shard.
            if use_gpu_dequant and qt in _GPU_DEQUANT_VERIFIED:
                row_elems = out_features * padded_in_features
                if row_elems > 64_000_000:  # >256 MiB bf16
                    block_rows = max(1, 8_000_000 // padded_in_features)
                    for r0 in range(0, out_features, block_rows):
                        r1 = min(out_features, r0 + block_rows)
                        info = _quant_info_for(qt)
                        nbpr = padded_in_features // info.block_size
                        sub = qb.view(out_features, nbpr, info.block_bytes)[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
                        with torch.no_grad():
                            blk = dequantize_tensor_gpu(sub, qt, r1 - r0, padded_in_features, device)
                        mixed[r0:r1] += (probs[k] * blk).to(torch.bfloat16)
                        del blk
                else:
                    with torch.no_grad():
                        cand_gpu = dequantize_tensor_gpu(qb, qt, out_features, padded_in_features, device)
                    mixed = mixed + probs[k] * cand_gpu
                    del cand_gpu
            else:
                # CPU dequant path (with pinned memory from Opt 2)
                cand = _dequant_slice(qb, qt, out_features, padded_in_features, in_features)
                if h2d_stream is not None:
                    with torch.cuda.stream(h2d_stream):
                        cand_gpu = cand.to(device=device, dtype=torch.bfloat16, non_blocking=False)
                    h2d_stream.synchronize()
                else:
                    cand_gpu = cand.to(device=device, dtype=torch.bfloat16)
                # Cache for backward (store CPU version to save VRAM)
                if fwd_cache is not None:
                    fwd_cache[k] = cand  # CPU pinned fp32
                mixed = mixed + probs[k] * cand_gpu
                del cand, cand_gpu

        ctx.source_dtype = source_dtype
        ctx.candidate_types = candidate_types
        ctx.out_features = out_features
        ctx.padded_in_features = padded_in_features
        ctx.in_features = in_features
        ctx.qbytes = qbytes
        ctx.device = device
        ctx.layer_ref = layer_ref
        return mixed.to(source_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.to(torch.float32)
        qbytes = ctx.qbytes
        candidate_types = ctx.candidate_types
        device = ctx.device
        layer_ref = ctx.layer_ref
        grad_probs = torch.zeros(len(qbytes), device=device, dtype=torch.float32)

        # Optimization 3: check forward candidate cache first
        fwd_cache = getattr(layer_ref, "_fwd_candidates", None)

        # Optimization 2: dedicated H2D copy stream.
        # NOTE: no dedicated H2D stream on ROCm.  Mixing a side stream with the
        # autograd engine's default-stream AccumulateGrad nodes produced HSA
        # aperture violations during the final aggregated backward; synchronous
        # copies are only marginally slower and are stream-safe.
        h2d_stream = None
        use_gpu_dequant = device.type == "cuda"

        for k in range(len(qbytes)):
            qt = candidate_types[k]
            cached = fwd_cache[k] if fwd_cache is not None else None

            row_elems = ctx.out_features * ctx.padded_in_features
            big = row_elems > 64_000_000

            if big and use_gpu_dequant and qt in _GPU_DEQUANT_VERIFIED and cached is None:
                # Blocked dequant for vocabulary-scale heads: accumulate the
                # gate gradient row-block-wise instead of materializing the
                # full fp32 candidate (see forward for the size rationale).
                info = _quant_info_for(qt)
                nbpr = ctx.padded_in_features // info.block_size
                g = grad_output.to(torch.bfloat16)
                acc = torch.zeros((), device=device, dtype=torch.float32)
                block_rows = max(1, 8_000_000 // ctx.padded_in_features)
                qb2d = qbytes[k].view(ctx.out_features, nbpr, info.block_bytes)
                for r0 in range(0, ctx.out_features, block_rows):
                    r1 = min(ctx.out_features, r0 + block_rows)
                    sub = qb2d[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
                    with torch.no_grad():
                        blk = dequantize_tensor_gpu(sub, qt, r1 - r0, ctx.padded_in_features, device)
                    acc += (g[r0:r1] * blk).sum()
                    del blk
                grad_probs[k] = acc
                continue

            if cached is not None:
                # Cache hit: reuse the dequantized candidate from forward
                if cached.device == device:
                    # Already on GPU (GPU dequant path was used in forward)
                    cand_gpu = cached.to(torch.bfloat16)
                else:
                    # CPU cached: transfer to GPU
                    if h2d_stream is not None:
                        with torch.cuda.stream(h2d_stream):
                            cand_gpu = cached.to(device=device, dtype=torch.bfloat16, non_blocking=False)
                        h2d_stream.synchronize()
                    else:
                        cand_gpu = cached.to(device=device, dtype=torch.bfloat16)
            else:
                # Cache miss: dequantize fresh
                if use_gpu_dequant and qt in _GPU_DEQUANT_VERIFIED:
                    with torch.no_grad():
                        cand_gpu = dequantize_tensor_gpu(qbytes[k], qt, ctx.out_features, ctx.padded_in_features, device)
                else:
                    cand = _dequant_slice(qbytes[k], qt, ctx.out_features, ctx.padded_in_features, ctx.in_features)
                    if h2d_stream is not None:
                        with torch.cuda.stream(h2d_stream):
                            cand_gpu = cand.to(device=device, dtype=torch.bfloat16, non_blocking=False)
                        h2d_stream.synchronize()
                    else:
                        cand_gpu = cand.to(device=device, dtype=torch.bfloat16)
                    del cand

            grad_probs[k] = (grad_output.to(torch.bfloat16) * cand_gpu).sum()
            del cand_gpu

        # Clear the forward cache after backward completes
        if fwd_cache is not None:
            for k in range(len(fwd_cache)):
                fwd_cache[k] = None

        # Inputs: source_dtype, candidate_types, out_features, padded_in_features,
        # in_features, probs, layer_ref, *qbytes  -> only probs receives a gradient.
        return (None, None, None, None, None, grad_probs, None) + (None,) * len(qbytes)


class _MixedEmbeddingRowFunctionLazy(Function):
    """Lazy mixed-weight lookup for embeddings that only dequantizes needed rows.

    A vocabulary-scale embedding (e.g. 248320x5120) has ~1.3B weights; the plain
    lazy path dequantizes the FULL candidate matrix on every forward, which is a
    multi-GB transient per candidate.  An embedding forward only reads the rows
    at the input indices, and every supported quant format stores blocks
    row-locally (all blocks of one output row are contiguous in ``qbytes``), so
    we slice the qbytes rows first and dequantize only those.  A training step
    touches <= seq_len*batch rows instead of num_embeddings.

    Mathematically identical to ``F.embedding(ids, mix(candidates))`` because
    row slicing commutes with the linear candidate mixture.
    """

    @staticmethod
    def forward(
        ctx,
        source_dtype,
        candidate_types,
        num_embeddings,
        padded_in_features,
        embedding_dim,
        probs,
        routing,
        row_ids,
        *qbytes,
    ):
        device = probs.device
        n_rows = row_ids.shape[0]
        if routing is not None and routing.device != device:
            routing = routing.to(device)
        mixed = torch.zeros(n_rows, embedding_dim, dtype=torch.bfloat16, device=device)
        for k in range(len(qbytes)):
            qt = candidate_types[k]
            qb = qbytes[k]
            info = _quant_info_for(qt)
            nbpr = padded_in_features // info.block_size
            # Slice this candidate's rows (blocks are row-local in qbytes layout).
            row_qb = qb.view(num_embeddings, nbpr, info.block_bytes)[row_ids].reshape(n_rows * nbpr, info.block_bytes)
            with torch.no_grad():
                cand_rows = dequantize_tensor_gpu(row_qb, qt, n_rows, padded_in_features, device)
            if routing is not None:
                # PTQR: each row keeps exactly one candidate's value; backward
                # still computes the full mixture gradient below.
                sel = routing == k
                mixed[sel] = cand_rows[sel, :embedding_dim]
            else:
                mixed = mixed + probs[k] * cand_rows[:, :embedding_dim]
            del cand_rows
        ctx.source_dtype = source_dtype
        ctx.candidate_types = candidate_types
        ctx.num_embeddings = num_embeddings
        ctx.padded_in_features = padded_in_features
        ctx.embedding_dim = embedding_dim
        ctx.qbytes = qbytes
        ctx.row_ids = row_ids
        ctx.device = device
        return mixed.to(source_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.to(torch.bfloat16)
        device = ctx.device
        row_ids = ctx.row_ids
        n_rows = row_ids.shape[0]
        grad_probs = torch.zeros(len(ctx.qbytes), device=device, dtype=torch.float32)
        for k in range(len(ctx.qbytes)):
            qt = ctx.candidate_types[k]
            qb = ctx.qbytes[k]
            info = _quant_info_for(qt)
            nbpr = ctx.padded_in_features // info.block_size
            row_qb = qb.view(ctx.num_embeddings, nbpr, info.block_bytes)[row_ids].reshape(n_rows * nbpr, info.block_bytes)
            with torch.no_grad():
                cand_rows = dequantize_tensor_gpu(row_qb, qt, n_rows, ctx.padded_in_features, device)
            grad_probs[k] = (grad_output * cand_rows[:, : ctx.embedding_dim]).sum()
            del cand_rows
        # source_dtype, candidate_types, num_embeddings, padded_in_features,
        # embedding_dim, probs, routing, row_ids, *qbytes -> only probs gets a
        # gradient.
        return (None, None, None, None, None, grad_probs, None, None) + (None,) * len(ctx.qbytes)


class _RoutedEmbeddingRowFunction(Function):
    """PTQR embedding lookup (non-lazy): each position uses ONE candidate's row.

    Candidates are CPU tensors [num_embeddings, embedding_dim]; only the rows
    at the input ids are gathered per candidate, so the full mixed vocabulary
    weight is never materialized — cheaper than the mixture path, which
    streams every candidate matrix every step.

    Backward is the straight-through mixture gradient:
        grad_p_k = <grad_out, W_k[ids]>
    """

    @staticmethod
    def forward(ctx, source_dtype, probs, routing, flat_ids, *candidates):
        device = routing.device
        out = torch.zeros(flat_ids.shape[0], candidates[0].shape[1], dtype=torch.float32, device=device)
        for k, cand in enumerate(candidates):
            rows = torch.nonzero(routing == k, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            sel_ids = flat_ids[rows].to(cand.device)
            out[rows] = cand[sel_ids].to(device=device, dtype=torch.float32)
        ctx.probs_device = probs.device
        ctx.flat_ids = flat_ids
        ctx.candidates = candidates
        ctx.device = device
        return out.to(source_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        G = grad_output.to(torch.float32)
        ids_cpu = ctx.flat_ids.to("cpu")
        device = ctx.device
        grad_probs = torch.zeros(len(ctx.candidates), device=device, dtype=torch.float32)
        for k, cand in enumerate(ctx.candidates):
            rows_k = cand[ids_cpu].to(device=device, dtype=torch.float32)
            grad_probs[k] = (G * rows_k).sum()
            del rows_k
        # source_dtype, probs, routing, flat_ids, *candidates -> only probs.
        return (None, grad_probs.to(ctx.probs_device), None, None) + (None,) * len(ctx.candidates)


# Candidate quantization is the dominant startup cost for dynamic-quant training.
# Cache quantized/dequantized candidates on disk so repeated runs (different
# sizes, resumed experiments, etc.) start in seconds instead of minutes.
# Anchored to the repo root (not the CWD) so runs from any directory share it.
CANDIDATE_CACHE_DIR = Path(voodoo_quant.__file__).parent.parent / ".cache" / "mixed_quant_candidates"
_CANDIDATE_CACHE_ROOT = CANDIDATE_CACHE_DIR


class _STGumbelState:
    """Global straight-through Gumbel hardening settings (see get_probs)."""

    enabled: bool = False
    fraction: float = 0.0  # per-step probability a given layer hardens
    gs_tau: float = 1.0  # Gumbel-softmax sampling temperature
    # Optional linear annealing of `fraction` across training: when set, the
    # effective fraction at optimizer step s (0-indexed) is
    # f_start + (f_end - f_start) * s / max(steps-1, 1).
    f_start: float | None = None
    f_end: float | None = None
    total_steps: int = 1
    step: int = 0


_ST_GUMBEL_STATE = _STGumbelState()


def set_st_gumbel(enabled: bool, fraction: float, gs_tau: float) -> None:
    _ST_GUMBEL_STATE.enabled = enabled
    _ST_GUMBEL_STATE.fraction = float(fraction)
    _ST_GUMBEL_STATE.gs_tau = float(gs_tau)


def set_st_gumbel_anneal(f_start: float, f_end: float, total_steps: int) -> None:
    """Enable fraction annealing from f_start to f_end over total_steps."""
    _ST_GUMBEL_STATE.f_start = float(f_start)
    _ST_GUMBEL_STATE.f_end = float(f_end)
    _ST_GUMBEL_STATE.total_steps = max(int(total_steps), 1)


def st_gumbel_step(step: int) -> None:
    """Notify the hardening schedule of the current optimizer step."""
    _ST_GUMBEL_STATE.step = int(step)


def _effective_fraction() -> float:
    st = _ST_GUMBEL_STATE
    if st.f_start is None:
        return st.fraction
    t = st.step / max(st.total_steps - 1, 1)
    t = min(max(t, 0.0), 1.0)
    return st.f_start + (st.f_end - st.f_start) * t


class _PTQRState:
    """Per-Token Quant Routing settings (see _sample_token_routing).

    When enabled, replaced layers run every token through ONE candidate
    (Gumbel-sampled from the gate probs) instead of averaging candidates —
    deployed-model behavior with no within-token mixture gain, and no
    hard/soft chimera (all layers route every step, so training KL measures
    quality rather than a hardening schedule).
    """

    enabled: bool = False


_PTQR_STATE = _PTQRState()


def set_ptqr(enabled: bool) -> None:
    _PTQR_STATE.enabled = bool(enabled)


def _sample_token_routing(probs: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """k*(t) ~ Categorical(probs) per token via the Gumbel-argmax trick.

    Token shares match `probs` in expectation by construction (unlike a
    nearest-candidate rule, whose shares follow mixture-hull geometry).  No
    temperature knob: argmax is invariant to any tau > 0, so this samples
    exactly from the softmax distribution the gates define.  As the gate
    temperature anneals and probs saturate, routing concentrates on the
    argmax candidate and the forward converges to the deployed model.
    """
    u = torch.rand(probs.shape[0], num_tokens, device=probs.device)
    gumbel = -torch.log((-torch.log(u.clamp_min(1e-9))).clamp_min(1e-9))
    scores = torch.log(probs.clamp_min(1e-9)).unsqueeze(1) + gumbel
    return scores.argmax(dim=0)


def _tensor_hash(t: torch.Tensor | None) -> str:
    """Stable SHA-256 hash of a contiguous CPU tensor's raw bytes.

    Streams in chunks so hashing a multi-GB tensor never materializes an
    fp32 copy (the fp32 cast is applied per-chunk and discarded).  Bit-wise
    identical to the eager version (same cast, same byte order).
    """
    if t is None:
        return "none"
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.float32:
        src = t.reshape(-1)
        h = hashlib.sha256()
        CH = 1 << 22  # 4M elems = 16 MiB per chunk
        for i in range(0, src.numel(), CH):
            h.update(src[i : i + CH].numpy().tobytes())
        return h.hexdigest()[:16]
    # Non-fp32: cast per-chunk to keep the fixed-dtype contract without a
    # full-size fp32 allocation.
    flat = t.reshape(-1)
    h = hashlib.sha256()
    CH = 1 << 22
    for i in range(0, flat.numel(), CH):
        h.update(flat[i : i + CH].to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def _candidate_cache_path(
    tensor_name: str,
    candidate_type: str,
    mode: str,
    weight_hash: str,
    imatrix_hash: str,
) -> Path:
    """Path for a cached candidate tensor.

    Args:
        tensor_name: full HF state-dict name (e.g. model.layers.0.mlp.gate_proj).
        candidate_type: quant type string.
        mode: 'lazy' for quantized bytes, 'dequant' for pre-dequantized weights.
        weight_hash: hash of the padded source weight.
        imatrix_hash: hash of the padded imatrix (or 'none').
    """
    safe_name = tensor_name.replace(".", "_").replace("/", "_")
    dir_path = _CANDIDATE_CACHE_ROOT / safe_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{candidate_type}_{mode}_w{weight_hash}_i{imatrix_hash}.pt"


def _combined_cache_path(
    tensor_name: str,
    mode: str,
    weight_hash: str,
    imatrix_hash: str,
) -> Path:
    """Path for a combined cache file containing all candidates for a tensor."""
    safe_name = tensor_name.replace(".", "_").replace("/", "_")
    dir_path = _CANDIDATE_CACHE_ROOT / safe_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"combined_{mode}_w{weight_hash}_i{imatrix_hash}.pt"



def _format_duration(seconds: float | None) -> str:
    """Human-friendly short duration like '4.2s', '2m03s', '1h07m'."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "..."
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _atomic_torch_save(t: torch.Tensor, path: Path) -> None:
    """Atomically write a cached candidate (tmp + os.replace).

    Under tensor parallelism every rank shares one candidate cache; concurrent
    writers must never leave a partially written file behind for the others.
    """
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(t, tmp)
    os.replace(tmp, path)


def _pad_weight_and_imatrix(
    weight: torch.Tensor,
    imatrix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """
    Pad a weight matrix and its imatrix along in_features to a multiple of 256.

    Returns (padded_weight, padded_imatrix_or_none, original_in_features).
    """
    orig_in = weight.shape[1]
    if orig_in % 256 == 0:
        return weight, imatrix, orig_in
    pad = 256 - (orig_in % 256)
    padded = F.pad(weight, (0, pad), "constant", 0)
    if imatrix is not None:
        imatrix = imatrix.to(torch.float32).contiguous()
        padded_imatrix = F.pad(imatrix, (0, pad), "constant", 1.0)
    else:
        padded_imatrix = None
    return padded, padded_imatrix, orig_in


# Public alias: other modules pad source weights/imatrix the same way the
# candidate cache does (same 256-alignment contract).
pad_weight_and_imatrix = _pad_weight_and_imatrix


class _MixedQuantBase(nn.Module):
    """Shared candidate-quantization logic for Linear and Embedding layers."""

    def __init__(
        self,
        candidate_types: list[str],
        source_weight: torch.Tensor,
        out_features: int,
        in_features: int,
        imatrix: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        lazy: bool = False,
        tensor_name: str = "unknown",
        qbytes_device: torch.device | str | None = None,
        shard_spec=None,
        full_source_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.candidate_types = list(candidate_types)
        self.num_candidates = len(candidate_types)
        self.out_features = out_features
        self.in_features = in_features
        self.lazy = lazy
        self._tensor_name = tensor_name
        self._qbytes_device = torch.device(qbytes_device) if qbytes_device is not None else None
        # Tensor-parallel shard mapping (voodoo_quant.parallel.ShardSpec).  When set,
        # `source_weight` is only the rank-local slice: candidates are
        # quantized / cached under the FULL tensor name from `full_source_weight`
        # and then sliced to the local shape, so the on-disk candidate cache is
        # bit-identical to the single-GPU one and shared by all ranks.
        self.shard_spec = shard_spec
        if shard_spec is not None and full_source_weight is None:
            raise ValueError(
                f"{tensor_name}: shard_spec requires full_source_weight "
                "(the unsharded weight for candidate quantization/caching)"
            )

        if device is None:
            device = source_weight.device
        self._device = torch.device(device)

        # Learnable gates, initialized to near-uniform (zero logits -> uniform softmax).
        self.gates = nn.Parameter(torch.zeros(self.num_candidates, device=self._device))
        self.temperature = 1.0

        # Optimization 3: forward candidate cache for backward reuse.
        # Populated during forward, consumed during backward, then cleared.
        self._fwd_candidates: list[torch.Tensor | None] = [None] * self.num_candidates

        # Optimization 4: candidate dequant cache across steps.
        # When valid, dequantized candidates are reused instead of re-dequantizing.
        # The mixed weight is recomputed each step (probs @ candidates) so gradients
        # are exact. Only the expensive dequantization is skipped.
        self._dequant_cache: list[torch.Tensor | None] = [None] * self.num_candidates
        self._dequant_cache_valid: bool = False

        # The llama.cpp C++ quantizer reads CPU memory, so move inputs to CPU
        # before quantizing.  Results are moved to the target device afterwards.
        source_dtype = source_weight.dtype
        self.source_dtype = source_dtype
        # Capture the layer's resident device (its pipeline shard) BEFORE the
        # CPU conversion: `--qbytes_device cuda` without an index means "keep
        # the quantized bytes on the tensor's own shard", not cuda:0.  Sending
        # every tensor's qbytes to one GPU OOMs it on the 27B run.
        resident_device = source_weight.device if source_weight.device.type == "cuda" else None
        # Move to CPU FIRST, cast to fp32 there.  Casting on the GPU allocates
        # an fp32 copy of the full weight on-device (4.74 GiB for the 248320x5120
        # embedding) which OOMs a shard that already holds weights + qbytes.
        # Under TP sharding the *full* weight (from the shared mmap'd base
        # checkpoint) is the quantization/cache input; the rank-local slice is
        # derived after the candidates exist.
        quant_weight_full = full_source_weight if self.shard_spec is not None else source_weight
        quant_weight_full = quant_weight_full.detach().to("cpu").contiguous()
        # Zero-fp32 TP path: with the complete candidate cache, the fp32
        # padded_weight is only used for the cache-key hash — which streams
        # in chunks (see _tensor_hash) — so we never materialize the full
        # fp32 copy.  The fp32 tensor is built ONLY if candidates are missing
        # and must be quantized here.
        quant_weight = None  # allocated lazily below if quantization needed
        big = quant_weight_full.numel() > 8_000_000  # TP: stagger ALL candidate fp32 work
        _tp_group_active = False
        try:
            import torch.distributed as _dist
            _tp_group_active = _dist.is_available() and _dist.is_initialized()
        except Exception:
            pass
        _big_lock = None
        if big and _tp_group_active:
            # File-lock serialization (NOT dist collectives — NCCL small
            # collectives hang on gfx908 with P2P disabled): one rank at a
            # time holds the lock through its big phase.
            import fcntl as _fcntl
            _lock_path = "/tmp/voodoo_tp_big.lock"
            _big_lock = open(_lock_path, "w")
            _fcntl.flock(_big_lock, _fcntl.LOCK_EX)
        if imatrix is not None:
            imatrix = imatrix.detach().to(torch.float32).cpu()
            # Hash the PADDED imatrix (matches the cache generators, which
            # hash _pad_weight_and_imatrix(weight, im)'s output; a no-op
            # fp32 cast for the 256-aligned in_features of Qwen3.8).
            _, padded_imatrix, _ = _pad_weight_and_imatrix(
                torch.empty(1, quant_weight_full.shape[1], dtype=torch.float32), imatrix
            )
        # Hash the source bytes directly (chunked; == fp32-hash of the padded
        # tensor because padding is zero-extension and in_features are
        # 256-aligned for every Qwen3.8 tensor).  Only if candidates are
        # actually missing do we build the fp32 padded weight for quantize.
        weight_hash = _tensor_hash(quant_weight_full)
        imatrix_hash = _tensor_hash(padded_imatrix) if imatrix is not None else "none"
        tensor_name = getattr(self, "_tensor_name", "unknown")
        mode = "lazy" if lazy else "dequant"

        # Try combined cache first (single file with all candidates).  The
        # cache may be keyed with or without the `model.` prefix (cache
        # generators use stripped names; the LMWithHead trainer registers
        # `model.layers.N...`); both refer to the same tensor bytes.
        combined_path = _combined_cache_path(tensor_name, mode, weight_hash, imatrix_hash)
        if not combined_path.exists():
            alt = tensor_name.removeprefix("model.") if tensor_name.startswith("model.") else "model." + tensor_name
            alt_path = _combined_cache_path(alt, mode, weight_hash, imatrix_hash)
            if alt_path.exists():
                combined_path = alt_path
        padded_weight = None
        padded_imatrix = imatrix
        if combined_path.exists():
            combined = torch.load(
                combined_path, weights_only=True, map_location="cpu", mmap=True if lazy else False
            )
            # A combined file may contain more (or, for grouped candidate
            # sets, fewer) types than this tensor's group.  Load the ones
            # present; any group member missing here falls through to the
            # individual-file / quantize path below.
            if lazy:
                for k, qt in enumerate(self.candidate_types):
                    if qt in combined:
                        setattr(self, f"_qweight_{k}", combined[qt])
            else:
                for k, qt in enumerate(self.candidate_types):
                    if qt in combined:
                        setattr(self, f"_w_{k}", combined[qt].to(source_dtype).cpu())
            del combined

        # Fill any candidates the combined file did not cover.
        missing: list[tuple[int, str, Path]] = []
        for k, qt in enumerate(self.candidate_types):
            if getattr(self, f"_qweight_{k}", None) is None and getattr(self, f"_w_{k}", None) is None:
                cache_path = _candidate_cache_path(tensor_name, qt, mode, weight_hash, imatrix_hash)
                if cache_path.exists():
                    if lazy:
                        cached = torch.load(cache_path, weights_only=True, map_location="cpu", mmap=True)
                        setattr(self, f"_qweight_{k}", cached)
                    else:
                        cached = torch.load(cache_path, weights_only=True, map_location="cpu")
                        setattr(self, f"_w_{k}", cached.to(source_dtype).cpu())
                else:
                    missing.append((k, qt, cache_path))

        if missing:
            if quant_weight is None:
                quant_weight = quant_weight_full.to(torch.float32)
            padded_weight, padded_imatrix, _ = _pad_weight_and_imatrix(quant_weight, imatrix)
            self._quantize_candidates_parallel(
                padded_weight,
                padded_imatrix,
                missing,
                # Candidates are cached for the FULL tensor even when this rank
                # only holds a slice (shard_spec): the dequantized buffer saved
                # to the cache must have full-tensor shape.
                self.shard_spec.full_out if self.shard_spec is not None else out_features,
                source_dtype,
                lazy,
            )
        if padded_weight is None:
            # padded width constant (in_features are 256-aligned here)
            padded_weight = None
        self.padded_in_features = self._full_padded_in_features if hasattr(self, "_full_padded_in_features") else (
            quant_weight_full.shape[1] if quant_weight_full.dim() == 2 else None
        )
        self._full_padded_in_features = self.padded_in_features

        # Slice the full candidates down to this rank's shard (TP mode).
        if self.shard_spec is not None:
            self._apply_shard_to_candidates()
            # Candidates are loaded/sharded now; the retained CPU bf16 full/slice
            # weight (tp_full_weight chain, ~13.5 GB/rank across 401 modules) is
            # dead weight for the rest of the run and OOMs a 61 GB host with 4
            # ranks.  Release the module's claim; the mmap'd base checkpoint
            # remains the authoritative source for the bake.
            try:
                host = getattr(self, "_host_module", None)
                if host is not None and getattr(host, "tp_full_weight", None) is not None:
                    host.tp_full_weight = None
            except Exception:
                pass

        # Closing barrier of the TP big-tensor stagger (see top of __init__):
        # free the fp32 transient (quant_weight/padded copies) before the
        # next rank enters its big phase.  malloc_trim returns the freed
        # arena pages to the OS — without it glibc keeps 5 GiB+ of RSS per
        # rank and the host OOMs by rank 2.
        if _big_lock is not None:
            import gc as _gc
            import ctypes as _ct
            quant_weight = None
            padded_weight = None
            del quant_weight_full
            _gc.collect()
            # Release the GPU staging from the full-qbytes upload + slice: the
            # rank's VRAM fills with the accumulating LOCAL slices while each
            # tensor's full-size staging (up to 2.7 GiB) must also fit.
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            import fcntl as _fcntl
            _fcntl.flock(_big_lock, _fcntl.LOCK_UN)
            _big_lock.close()
            _big_lock = None

        if imatrix is not None:
            self.imatrix = imatrix.detach().to(torch.float32).cpu()
        else:
            self.imatrix = None

        # Optional GPU-resident qbytes (lazy mode): keeps the quantized bytes on
        # the shard device so the lazy forward/backward never pays an H2D copy
        # per candidate.  A device without an index resolves to the layer's own
        # shard (see resident_device capture above).  No-op in non-lazy mode.
        if self.lazy and self._qbytes_device is not None:
            qd = self._qbytes_device
            if qd.type == "cuda" and qd.index is None and resident_device is not None:
                qd = resident_device
            for k in range(self.num_candidates):
                qb = getattr(self, f"_qweight_{k}", None)
                if qb is not None and qb.device != qd:
                    setattr(self, f"_qweight_{k}", qb.to(qd))
            if resident_device is not None and qd != resident_device:
                print(f"  [qbytes] {self._tensor_name}: qbytes -> {qd} (weight on {resident_device})", flush=True)

    def _apply_shard_to_candidates(self):
        """Slice full-tensor candidates (cache format) to this rank's shard.

        Called at the end of __init__ when `shard_spec` is set.  After this the
        module looks exactly like a single-GPU module for the LOCAL shape:
        ``out_features``/``in_features``/``padded_in_features`` are local and the
        ``_w_k``/``_qweight_k`` buffers hold local rows/columns, so every forward,
        backward and gradient path is untouched.  ``_full_padded_in_features``
        keeps the full padded width for reference.
        """
        spec = self.shard_spec
        assert spec is not None
        local_padded_in = spec.local_padded_in(self._full_padded_in_features, self.candidate_types)
        # GPU-side slicing: uploading the full qbytes once and slicing on the
        # rank GPU avoids ~7 GiB of CPU slice copies per rank (torch's CPU
        # caching allocator never returns them; 4 ranks = host OOM even with
        # everything else correct).  GPU has the headroom.
        qdev = getattr(self, "_qbytes_device", None)
        for k, qt in enumerate(self.candidate_types):
            if self.lazy:
                qb = getattr(self, f"_qweight_{k}", None)
                if qb is not None:
                    # Slice on the CPU side FIRST (row slices of the mmap'd
                    # cache are zero-copy views; column slices copy only the
                    # rank-local bytes), then upload just the local slice.
                    # Streaming the FULL qbytes to the GPU to slice there
                    # pinned multi-GB H2D staging buffers per tensor and, with
                    # 4 ranks in flight, OOM-killed the host during init.
                    local = spec.slice_qbytes(qb, qt, self._full_padded_in_features)
                    if qdev is not None:
                        local = local.to(qdev, non_blocking=False)
                    setattr(self, f"_qweight_{k}", local)
            else:
                w = getattr(self, f"_w_{k}", None)
                if w is not None:
                    setattr(self, f"_w_{k}", spec.slice_weight(w))
        self.padded_in_features = local_padded_in

    def _quantize_candidates_parallel(
        self,
        padded_weight: torch.Tensor,
        padded_imatrix: torch.Tensor | None,
        missing: list[tuple[int, str, Path]],
        out_features: int,
        source_dtype: torch.dtype,
        lazy: bool,
    ):
        """Quantize missing candidates in parallel and register/save the results."""
        import concurrent.futures
        import os

        # The cache-cold path quantizes during __init__, before
        # self.padded_in_features is assigned (upstream guarded this with a
        # pre-built candidate cache); the padded width is exactly the padded
        # weight's width.
        padded_in = int(padded_weight.shape[1])

        max_workers = int(os.environ.get("VOODOO_QUANT_WORKERS", min(8, (os.cpu_count() or 2))))

        def _quantize_one(item):
            k, qt, cache_path = item
            # Exact llama.cpp quantizer on the CPU (libggml-base.so).
            torch.set_num_threads(1)
            qbytes = quantize_tensor(padded_weight, qt, padded_imatrix)
            if lazy:
                buf = qbytes
            else:
                w_deq = dequantize_tensor(qbytes, qt, out_features, padded_in)
                buf = w_deq.to(source_dtype).cpu()
            return k, qt, cache_path, buf

        # Ensure OpenMP does not oversubscribe the worker threads.
        old_omp_threads = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"

        # Run quantizations in parallel.  The fast GPU path is much quicker.
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for future in concurrent.futures.as_completed(
                executor.submit(_quantize_one, item) for item in missing
            ):
                k, qt, cache_path, buf = future.result()
                if lazy:
                    # Keep on CPU to save GPU memory
                    setattr(self, f"_qweight_{k}", buf.cpu() if buf.is_cuda else buf)
                    _atomic_torch_save(buf.detach().cpu(), cache_path)
                else:
                    setattr(self, f"_w_{k}", buf)
                    _atomic_torch_save(buf, cache_path)
                completed += 1
                if completed == 1 or completed % 4 == 0:
                    print(f"    {self._tensor_name}: quantized {completed}/{len(missing)} missing candidates", flush=True)

        if old_omp_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_omp_threads

    def set_temperature(self, temperature: float):
        self.temperature = max(temperature, 1e-6)

    def soft_probs(self) -> torch.Tensor:
        """Softmax gate probs WITHOUT straight-through hardening.

        Used for budgeting (effective_bytes) and argmax reporting, where the
        *expected* size/assignment is wanted regardless of this step's
        hardening draw.  The forward path uses get_probs() (ST-Gumbel)."""
        return F.softmax(self.gates / self.temperature, dim=0)

    def get_probs(self) -> torch.Tensor:
        probs = self.soft_probs()
        # Straight-through Gumbel top-k hardening (ST hard-sampling): when a
        # global hardening fraction is set, a per-tensor Bernoulli draw picks
        # which layers harden this step.  Hardened layers forward a one-hot
        # Gumbel sample (deployed-model behavior) while backward flows through
        # the soft softmax (custom-mix backward stays exact for the gates).
        # Sampling (not argmax) keeps exploration alive: a wrong early argmax
        # can still be corrected via its soft gradient on later steps.
        if _ST_GUMBEL_STATE.enabled:
            import random as _random

            if _random.random() < _effective_fraction():
                u = torch.rand_like(probs)
                gumbel = -torch.log((-torch.log(u.clamp_min(1e-9))).clamp_min(1e-9))
                scores = (torch.log(probs.clamp_min(1e-9)) + gumbel) / max(
                    _ST_GUMBEL_STATE.gs_tau, 1e-3
                )
                hard = F.one_hot(scores.argmax(dim=0), num_classes=probs.shape[0]).to(probs.dtype)
                # Forward uses the one-hot sample; gradient flows to `probs`
                # (straight-through): d(hard)/d(gates) := d(probs)/d(gates).
                return (hard - probs).detach() + probs
        return probs

    def _get_candidate(self, k: int) -> torch.Tensor:
        qt = self.candidate_types[k]
        if self.lazy:
            qbytes = getattr(self, f"_qweight_{k}")
            w = dequantize_tensor(qbytes, qt, self.out_features, self.padded_in_features)
        else:
            w = getattr(self, f"_w_{k}")
            # Keep candidates in the source dtype on CPU; the mixed-weight function
            # will cast to float32 on the compute device one candidate at a time.
            # Casting here would double the CPU footprint because autograd saves the
            # inputs passed to the custom Function.
        return w[:, : self.in_features]

    def invalidate_candidate_cache(self):
        """Invalidate the cross-step dequant cache (Optimization 4)."""
        self._dequant_cache = [None] * self.num_candidates
        self._dequant_cache_valid = False

    def shard_rows(self, row_ranges: list[tuple[int, int]]) -> None:
        """Physically restrict this layer to the given output-row ranges (TP).

        Tensor-parallel support (see ``voodoo_quant.parallel``): the candidates were
        quantized from the FULL tensor — cache keys stay full-tensor names — and
        llama.cpp block quantization is row-independent, so slicing candidate
        rows is bit-identical to quantizing the row slice.  ``out_features``
        becomes the shard size and ``_full_out_features`` records the original
        so :meth:`effective_bytes` keeps budgeting the whole tensor (the gates
        are replicated across TP ranks).
        """
        if getattr(self, "_full_out_features", None) is not None or \
                getattr(self, "_full_in_features", None) is not None:
            raise RuntimeError(f"{self._tensor_name}: already row/col-sharded")
        shard_out = sum(e - s for s, e in row_ranges)
        if shard_out <= 0 or shard_out > self.out_features:
            raise ValueError(f"{self._tensor_name}: invalid row ranges {row_ranges} "
                             f"for out_features={self.out_features}")
        self.invalidate_candidate_cache()
        self._fwd_candidates = [None] * self.num_candidates
        for k, qt in enumerate(self.candidate_types):
            if self.lazy:
                qb = getattr(self, f"_qweight_{k}", None)
                if qb is None:
                    raise RuntimeError(f"{self._tensor_name}: lazy qbytes missing for {qt}")
                q2d = qb.view(self.out_features, -1)  # one row = whole quant blocks
                setattr(self, f"_qweight_{k}",
                        torch.cat([q2d[s:e] for s, e in row_ranges], dim=0).reshape(-1).contiguous())
            else:
                w = getattr(self, f"_w_{k}", None)
                if w is None:
                    raise RuntimeError(f"{self._tensor_name}: candidate missing for {qt}")
                setattr(self, f"_w_{k}",
                        torch.cat([w[s:e] for s, e in row_ranges], dim=0).contiguous())
        self._full_out_features = self.out_features
        self.out_features = shard_out

    def shard_cols(self, col_range: tuple[int, int]) -> None:
        """Physically restrict this layer to an in-feature column range (TP).

        Row-parallel out_proj support (see ``voodoo_quant.parallel``): the weight is
        partitioned along the input dim, so the slice must stay within the real
        ``in_features`` (never the zero padding) and be quant-block aligned —
        then slicing candidate columns is bit-identical to quantizing the slice.
        """
        if getattr(self, "_full_out_features", None) is not None or \
                getattr(self, "_full_in_features", None) is not None:
            raise RuntimeError(f"{self._tensor_name}: already row/col-sharded")
        s, e = col_range
        if s < 0 or e > self.in_features or e <= s:
            raise ValueError(f"{self._tensor_name}: invalid column range {col_range} "
                             f"for in_features={self.in_features}")
        self.invalidate_candidate_cache()
        self._fwd_candidates = [None] * self.num_candidates
        for k, qt in enumerate(self.candidate_types):
            info = _quant_info_for(qt)
            nbpr = self.padded_in_features // info.block_size
            b0, b1 = s // info.block_size, e // info.block_size
            if s % info.block_size or e % info.block_size:
                raise ValueError(f"{self._tensor_name}: column range {col_range} is not "
                                 f"{info.block_size}-block aligned for {qt}")
            if self.lazy:
                qb = getattr(self, f"_qweight_{k}", None)
                if qb is None:
                    raise RuntimeError(f"{self._tensor_name}: lazy qbytes missing for {qt}")
                q3 = qb.view(self.out_features, nbpr, info.block_bytes)[:, b0:b1, :]
                setattr(self, f"_qweight_{k}", q3.reshape(-1).contiguous())
            else:
                w = getattr(self, f"_w_{k}", None)
                if w is None:
                    raise RuntimeError(f"{self._tensor_name}: candidate missing for {qt}")
                setattr(self, f"_w_{k}", w[:, s:e].contiguous())
        self._full_in_features = self.in_features
        self.in_features = e - s
        # The kept blocks span exactly [b0, b1) with no padding tail, so the
        # lazy dequant paths (which size themselves from padded_in_features)
        # must see the sharded block count, not the full tensor's.
        self.padded_in_features = e - s

    def get_mixed_weight(self) -> torch.Tensor:
        probs = self.get_probs()
        # Anchor the mixed-weight build to the compute device (the device of the
        # input tensor in a sharded pipeline), not the gates' device, so no
        # cross-GPU copy of the full mixed weight is needed per layer.
        target = getattr(self, "_compute_device", None)
        if target is not None and probs.device != target:
            probs = probs.to(target)
        if self.lazy:
            # Optimization 4: if dequant cache is valid, reuse the cached
            # dequantized candidates and just recompute probs @ candidates
            # (cheap, and gradients are exact).  Only the expensive dequant
            # is skipped.
            if self._dequant_cache_valid:
                # Use cached candidates — just do the weighted sum on GPU
                device = getattr(self, "_compute_device", None) or self._device
                mixed = torch.zeros(
                    self.out_features, self.in_features,
                    dtype=torch.bfloat16, device=device,
                )
                for k in range(self.num_candidates):
                    cand = self._dequant_cache[k]
                    if cand.device != device:
                        cand = cand.to(device=device, dtype=torch.bfloat16, non_blocking=False)
                    mixed = mixed + probs[k] * cand
                return mixed.to(self.source_dtype)

            # Optimization 4: populate dequant cache during forward.
            # The _MixedWeightFunctionLazy.forward will dequantize candidates
            # and store them in _fwd_candidates. After forward completes,
            # we copy them to _dequant_cache for reuse in subsequent steps.
            qbytes = [getattr(self, f"_qweight_{k}") for k in range(self.num_candidates)]
            if target is not None:
                # Sharded pipeline: dequantize on the shard device directly.
                qbytes = [qb.to(target, non_blocking=False) if qb.device != target else qb for qb in qbytes]
            result = _MixedWeightFunctionLazy.apply(
                self.source_dtype,
                tuple(self.candidate_types),
                self.out_features,
                self.padded_in_features,
                self.in_features,
                probs,
                self,  # pass layer reference for Opt 3 cache
                *qbytes,
            )
            # After forward, copy fwd_candidates to dequant_cache
            # (they will be cleared after backward, but dequant_cache persists)
            cached_count = 0
            for k in range(self.num_candidates):
                if self._fwd_candidates[k] is not None:
                    # Store CPU copy to save VRAM
                    cached = self._fwd_candidates[k]
                    if cached.device.type == "cuda":
                        self._dequant_cache[k] = cached.to("cpu", non_blocking=False)
                    else:
                        self._dequant_cache[k] = cached
                    cached_count += 1
            # Only mark cache as valid if all candidates were cached
            self._dequant_cache_valid = cached_count == self.num_candidates
            return result
        candidates = [self._get_candidate(k) for k in range(self.num_candidates)]
        return _MixedWeightFunction.apply(self.source_dtype, probs, *candidates)

    def effective_bytes(self) -> torch.Tensor:
        # Expected size under the SOFT distribution: the size loss must not
        # see the per-step Gumbel hardening draw (sampled sizes are wildly
        # noisy and broke budget control in the first ST-Gumbel run).
        probs = self.soft_probs()
        # Row/col-sharded (TP) layers count their FULL tensor: the gates are
        # replicated across ranks, so every rank must budget the whole tensor.
        out_eff = getattr(self, "_full_out_features", None) or self.out_features
        in_eff = getattr(self, "_full_in_features", None) or self.in_features
        n_weights = in_eff * out_eff
        total = torch.tensor(0.0, device=probs.device)
        for k, qt in enumerate(self.candidate_types):
            total = total + probs[k] * bytes_per_weight(qt) * n_weights
        return total

    def get_assignment(self) -> str:
        return self.candidate_types[self.gates.argmax().item()]

    def get_assignment_prob(self) -> float:
        probs = self.soft_probs().detach()
        return probs[self.gates.argmax().item()].item()


class _BlockedCandidateLinear(Function):
    """y = p * (x @ dequant(qb)_k^T) for a vocabulary-scale candidate.

    Dequantizes the candidate in row blocks, matmuls each block, and never
    keeps more than one block plus the output alive.  Backward recomputes the
    blocks (no weight saved) to produce grad_p exactly like the unblocked
    formulation.  Only ``prob`` receives a gradient (matches the lazy
    Function's contract for constants).
    """

    @staticmethod
    def forward(ctx, x, qb, qt, out_features, padded_in_features, in_features, prob, row_sel=None):
        from torch.nn import functional as F
        orig_shape = x.shape
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])  # [*, in] -> [n, in]
        info = _quant_info_for(qt)
        nbpr = padded_in_features // info.block_size
        dev = qb.device
        out = torch.zeros(x.shape[0], out_features, dtype=torch.float32, device=dev)
        block_rows = max(1, 8_000_000 // padded_in_features)
        qb2d = qb.view(out_features, nbpr, info.block_bytes)
        for r0 in range(0, out_features, block_rows):
            r1 = min(out_features, r0 + block_rows)
            sub = qb2d[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
            blk = dequantize_tensor_gpu(sub, qt, r1 - r0, padded_in_features, dev)
            out[:, r0:r1] = F.linear(x, blk[:, :in_features].to(x.dtype)).float()
            del blk
        ctx.x_shape = orig_shape
        ctx.save_for_backward(x, prob)
        ctx.qb = qb
        ctx.qt = qt
        ctx.out_features = out_features
        ctx.padded_in_features = padded_in_features
        ctx.in_features = in_features
        if row_sel is not None:
            # PTQR: forward only this candidate's routed rows (0/1 mask); the
            # backward below still uses the soft prob, giving the exact
            # straight-through mixture gradient.
            y = (out * row_sel.to(out.dtype).unsqueeze(1)).to(x.dtype)
        else:
            y = (prob * out).to(x.dtype)
        return y.reshape(*orig_shape[:-1], out_features) if len(orig_shape) != 2 else y

    @staticmethod
    def backward(ctx, grad_out):
        from torch.nn import functional as F
        x, prob = ctx.saved_tensors
        if grad_out.dim() != 2:
            grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        if x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])
        qb, qt = ctx.qb, ctx.qt
        out_features = ctx.out_features
        padded_in, in_features = ctx.padded_in_features, ctx.in_features
        info = _quant_info_for(qt)
        nbpr = padded_in // info.block_size
        dev = qb.device
        grad_out = grad_out.to(x.dtype)
        grad_x = torch.zeros_like(x, dtype=torch.float32)
        acc = torch.zeros((), device=dev, dtype=torch.float32)
        block_rows = max(1, 8_000_000 // padded_in)
        qb2d = qb.view(out_features, nbpr, info.block_bytes)
        for r0 in range(0, out_features, block_rows):
            r1 = min(out_features, r0 + block_rows)
            sub = qb2d[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
            blk = dequantize_tensor_gpu(sub, qt, r1 - r0, padded_in, dev)
            w = blk[:, :in_features].to(x.dtype)
            gy = grad_out[:, r0:r1]
            grad_x += ((gy * prob) @ w).float()      # chain through y = p * x@W^T
            acc += (gy * (x @ w.T)).float().sum()  # d/dprob: dL/dp = sum(dL/dy * dy/dp), dy/dp = xW
            del blk, w, gy
        grad_prob = acc
        grad_x = grad_x.to(x.dtype)
        if len(ctx.x_shape) != 2:
            grad_x = grad_x.reshape(*ctx.x_shape)
        # x, qb, qt, out_features, padded_in_features, in_features, prob, row_sel
        return (grad_x, None, None, None, None, None, grad_prob, None)


class MixedQuantLinear(_MixedQuantBase):
    """
    Linear layer whose effective weight is a softmax mixture of K quantized
    candidates.  Works for any `in_features` by zero-padding to the next
    multiple of the quant block size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        candidate_types: list[str],
        source_weight: torch.Tensor,
        imatrix: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        lazy: bool = False,
        tensor_name: str = "unknown",
        qbytes_device: torch.device | str | None = None,
        shard_spec=None,
        full_source_weight: torch.Tensor | None = None,
    ):
        super().__init__(
            candidate_types=candidate_types,
            source_weight=source_weight,
            out_features=out_features,
            in_features=in_features,
            imatrix=imatrix,
            device=device,
            lazy=lazy,
            tensor_name=tensor_name,
            qbytes_device=qbytes_device,
            shard_spec=shard_spec,
            full_source_weight=full_source_weight,
        )
        # Vocabulary offsets survive the replacement so the TP loss can map
        # target ids onto this rank's logits shard.
        if shard_spec is not None and shard_spec.vocab_start is not None:
            self.tp_vocab_start = shard_spec.vocab_start
            self.tp_vocab_end = shard_spec.vocab_start + out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Large lazy layers (vocabulary heads) never materialize the mixed
        # weight: F.linear is linear in the weight, so out = sum_k p_k (x @
        # W_k^T) is exact and needs only one candidate's *output* in memory
        # ([tokens, out_features] ~0.3 GiB) instead of the 2.4 GiB mixed
        # matrix plus per-candidate dequant copies.  Blocked dequant keeps
        # each candidate's transient at ~256 MiB.
        qb0 = getattr(self, "_qweight_0", None)
        big = qb0 is not None and qb0.numel() > 64_000_000
        if big:
            dev = qb0.device
            if not getattr(self, "_big_dev_logged", False):
                print(f"  [big-linear] {self._tensor_name}: qbytes/compute on {dev}", flush=True)
                self._big_dev_logged = True
            self._compute_device = dev
            if x.device != dev:
                x = x.to(dev)
            probs = self.get_probs().to(dev)
            routing = None
            if _PTQR_STATE.enabled:
                n = x.numel() // x.shape[-1]
                routing = _sample_token_routing(probs, n)
            out = None
            for k, qt in enumerate(self.candidate_types):
                out_k = _BlockedCandidateLinear.apply(
                    x, getattr(self, f"_qweight_{k}"), qt,
                    self.out_features, self.padded_in_features, self.in_features, probs[k],
                    (routing == k) if routing is not None else None,
                )
                out = out_k if out is None else out + out_k
            out = out.to(self.source_dtype)
        else:
            if x.device != self._device:
                self._compute_device = x.device
            if _PTQR_STATE.enabled:
                probs = self.get_probs()
                n = x.numel() // x.shape[-1]
                routing = _sample_token_routing(probs.to(x.device), n)
                if self.lazy and not self._dequant_cache_valid:
                    # One-time: populate the persistent dequant cache via the
                    # parallel lazy path so PTQR forwards reuse dequants
                    # instead of re-dequantizing every step.
                    with torch.no_grad():
                        self.get_mixed_weight()
                if self.lazy and self._dequant_cache_valid:
                    cands = [self._dequant_cache[k] for k in range(self.num_candidates)]
                else:
                    cands = [self._get_candidate(k) for k in range(self.num_candidates)]
                out = _RoutedMixedLinear.apply(self.source_dtype, probs, routing, x, *cands)
            else:
                mixed = self.get_mixed_weight().to(self.source_dtype)
                out = F.linear(x, mixed)
        # Row-parallel shard (input columns split across TP ranks): this rank
        # holds a partial sum, all-reduce completes it (replacing the forward of
        # the RowParallelLinear this module replaced).
        spec = getattr(self, "shard_spec", None)
        if spec is not None and spec.col_offset is not None:
            from voodoo_quant.parallel import tp_all_reduce_sum

            out = tp_all_reduce_sum(out)
        return out


class MixedQuantEmbedding(_MixedQuantBase):
    """
    Embedding layer whose effective weight is a softmax mixture of K quantized
    candidates.  The weight is treated as a 2-D matrix [num_embeddings,
    embedding_dim] and padded along embedding_dim to a multiple of the quant
    block size.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        candidate_types: list[str],
        source_weight: torch.Tensor,
        imatrix: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        lazy: bool = False,
        tensor_name: str = "unknown",
        qbytes_device: torch.device | str | None = None,
        shard_spec=None,
        full_source_weight: torch.Tensor | None = None,
    ):
        super().__init__(
            candidate_types=candidate_types,
            source_weight=source_weight,
            out_features=num_embeddings,
            in_features=embedding_dim,
            imatrix=imatrix,
            device=device,
            lazy=lazy,
            tensor_name=tensor_name,
            qbytes_device=qbytes_device,
            shard_spec=shard_spec,
            full_source_weight=full_source_weight,
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if shard_spec is not None and shard_spec.vocab_start is not None:
            self.tp_vocab_start = shard_spec.vocab_start
            self.tp_vocab_end = shard_spec.vocab_start + num_embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device != self._device:
            self._compute_device = input.device
        spec = getattr(self, "shard_spec", None)
        if spec is not None and spec.vocab_start is not None:
            # Vocab-parallel shard: ids outside [vocab_start, vocab_end) are
            # zeroed here and contribute zero rows; the all-reduce sums the
            # owning rank's rows (replacing VocabParallelEmbedding.forward).
            from voodoo_quant.parallel import tp_all_reduce_sum

            if input.device != self._device and self._device.type == "cuda":
                input = input.to(self._device)
            v0, v1 = self.tp_vocab_start, self.tp_vocab_end
            mask = (input < v0) | (input >= v1)
            local_ids = (input - v0).clamp(0, self.num_embeddings - 1)
            emb = self._embedding_lookup(local_ids)
            if mask.any():
                emb = emb * (~mask).unsqueeze(-1).to(emb.dtype)
            return tp_all_reduce_sum(emb)
        return self._embedding_lookup(input)

    def _embedding_lookup(self, input: torch.Tensor) -> torch.Tensor:
        if self.lazy and input.device.type == "cuda" and self._qbytes_device is not None \
                and all(qt in _GPU_DEQUANT_VERIFIED for qt in self.candidate_types):
            # Row-select lazy path: dequantize only the (unique) rows the batch
            # touches instead of the full vocabulary-scale candidate matrices.
            qdev = self._qbytes_device if input.device == self._qbytes_device else input.device
            rows, inverse = torch.unique(input.reshape(-1), return_inverse=True)
            rows = rows.to(qdev)
            probs = self.get_probs()
            routing = None
            if _PTQR_STATE.enabled:
                routing = _sample_token_routing(probs.to(qdev), rows.shape[0])
            row_mixed = _MixedEmbeddingRowFunctionLazy.apply(
                self.source_dtype,
                tuple(self.candidate_types),
                self.num_embeddings,
                self.padded_in_features,
                self.embedding_dim,
                probs,
                routing,
                rows,
                *[getattr(self, f"_qweight_{k}").to(qdev) if getattr(self, f"_qweight_{k}").device != qdev else getattr(self, f"_qweight_{k}") for k in range(self.num_candidates)],
            )
            return row_mixed[inverse].reshape(*input.shape, self.embedding_dim)
        if _PTQR_STATE.enabled:
            # PTQR: one candidate's row per position; only the needed rows are
            # gathered from each CPU candidate (the full mixed vocabulary
            # weight is never built).
            flat = input.reshape(-1)
            probs = self.get_probs()
            routing = _sample_token_routing(probs.to(input.device), flat.numel())
            cands = [self._get_candidate(k) for k in range(self.num_candidates)]
            out = _RoutedEmbeddingRowFunction.apply(
                self.source_dtype, probs, routing, flat, *cands
            )
            return out.reshape(*input.shape, self.embedding_dim)
        mixed = self.get_mixed_weight().to(self.source_dtype)
        return F.embedding(input, mixed)


def _count_selectable_linears(
    module: nn.Module,
    target_names: Optional[set[str]],
    skip_names: set[str],
) -> int:
    """Count Linear layers that replace_linear_with_mixed_quant would select."""
    total = 0
    for name, child in module.named_children():
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            if target_names is None or name in target_names:
                total += 1
        else:
            total += _count_selectable_linears(child, target_names, skip_names)
    return total


# Cached attention-segment set; the arch registry's default is consulted once
# so menu resolution stays cheap in the hot init path. (Model-specific
# adapters override the segments via their attention_segments attribute; the
# trainer path passes tensor names here, so the default family — qwen hybrid
# segments — is what layer-level resolution uses.)
from voodoo_quant.arch.base import DEFAULT_ATTENTION_SEGMENTS as _DEFAULT_SEGMENTS

_ATTENTION_SEGMENTS: frozenset = _DEFAULT_SEGMENTS


def _attention_segments_cache(tensor_name: str) -> frozenset:
    """Attention path segments for menu resolution (default-safe)."""
    return _ATTENTION_SEGMENTS


def resolve_candidate_types(
    tensor_name: str,
    base_candidates: list[str],
    attention_candidates: list[str] | None = None,
    non_attention_candidates: list[str] | None = None,
) -> list[str]:
    """Per-tensor candidate set for grouped candidate restriction.

    A tensor is "attention-side" when any path segment of its full state-dict
    name is in the arch adapter's attention segments (default:
    ``self_attn``/``linear_attn``/``attn``/``conv``/``shortconv`` —
    segment-exact, so qwen's ``conv1d`` buffer is unaffected and LFM2's
    ``conv`` module follows the attention menu). Tiny per-head state tensors
    (A_log, dt_bias, conv1d) and norms are never selectable; callers keep
    them out of the replacement set entirely. When both groups are provided
    the intersection with the global candidate list (order preserved by the
    group definition) is used so ``--candidate_types`` can still narrow what
    is cache-visible.
    """
    if attention_candidates is None and non_attention_candidates is None:
        return list(base_candidates)
    segments = _attention_segments_cache(tensor_name)
    segs = tensor_name.split(".")
    is_attn = any(seg in segments for seg in segs)
    group = attention_candidates if (is_attn and attention_candidates) else non_attention_candidates
    if not group:
        return list(base_candidates)
    return [qt for qt in group if qt in set(base_candidates)]


def replace_linear_with_mixed_quant(
    module: nn.Module,
    candidate_types: list[str] | None,
    prefix: str = "",
    target_names: Optional[set[str]] = None,
    skip_names: Optional[set[str]] = None,
    imatrix_dict: Optional[dict[str, torch.Tensor]] = None,
    device: torch.device | str | None = None,
    lazy: bool = False,
    attention_candidates: Optional[list[str]] = None,
    non_attention_candidates: Optional[list[str]] = None,
    qbytes_device: torch.device | str | None = None,
) -> dict[str, _MixedQuantBase]:
    """
    Recursively replace selected nn.Linear children with MixedQuantLinear.

    Args:
        module: root module to modify in place.
        candidate_types: candidate quant types for every replaced layer.
        prefix: state-dict name prefix used for imatrix lookup.
        target_names: if provided, only replace Linear layers whose local name
            is in this set (e.g., {"gate_proj", "up_proj", "down_proj"}).
        skip_names: additional submodule names to skip entirely.
        imatrix_dict: optional {hf_tensor_name: imatrix_vector} mapping.
        device: device for candidate buffers.
        attention_candidates / non_attention_candidates: optional per-group
            candidate lists; `self_attn.*` tensors get the attention group.
        qbytes_device: lazy-mode device for the persistent quantized bytes.

    Returns:
        mapping from full state-dict name to MixedQuantLinear module.
    """
    import time as _time
    if candidate_types is None:
        from voodoo_quant.ggml import list_quant_types
        candidate_types = list_quant_types()
    if skip_names is None:
        skip_names = set()
    if imatrix_dict is None:
        imatrix_dict = {}

    # Reset progress state at the top-level call and precount selectable layers
    # so we can show a percentage, throughput and ETA during long inits.
    if prefix == "":
        total = _count_selectable_linears(module, target_names, skip_names)
        replace_linear_with_mixed_quant._mq_counter = 0
        replace_linear_with_mixed_quant._mq_total = total
        now = _time.monotonic()
        replace_linear_with_mixed_quant._mq_start = now
        replace_linear_with_mixed_quant._mq_last_print = 0.0
        replace_linear_with_mixed_quant._mq_ema = None
        replace_linear_with_mixed_quant._mq_last_t = now
        print(
            f"Preparing {total} selectable layers x {len(candidate_types)} candidates ...",
            flush=True,
        )

    replaced: dict[str, _MixedQuantBase] = {}

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            if child.bias is not None:
                raise NotImplementedError(f"MixedQuantLinear does not support bias but {full_name} has one")
            if target_names is not None and name not in target_names:
                continue
            imatrix = imatrix_dict.get(full_name)
            if imatrix is None:
                imatrix = imatrix_dict.get(full_name + ".weight")
            tensor_candidates = resolve_candidate_types(
                full_name, candidate_types, attention_candidates, non_attention_candidates
            )
            new_child = MixedQuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                candidate_types=tensor_candidates,
                source_weight=child.weight.data,
                imatrix=imatrix,
                device=device,
                lazy=lazy,
                tensor_name=full_name,
                qbytes_device=qbytes_device,
                # TP: candidates are quantized/cached for the FULL tensor, then
                # sliced to this rank's local shape (bit-identical shared cache).
                shard_spec=getattr(child, "tp_shard_spec", None),
                full_source_weight=getattr(child, "tp_full_weight", None),
            )
            setattr(module, name, new_child)
            # The MixedQuant module releases the host's CPU weight retention
            # (tp_full_weight) after candidates load; keep a handle to the host
            # module for that.  See _apply_shard_to_candidates.
            if getattr(child, "tp_shard_spec", None) is not None:
                new_child._host_module = child
            replaced[full_name] = new_child
            # Incremental GPU residency (TP): move this module to the rank GPU
            # NOW.  The host wrapper's bf16 slice is dead for compute after
            # replacement (forward uses the candidate qbytes), so point its
            # .data back at the FILE-BACKED mmap view instead of keeping a GPU
            # copy or an anon CPU copy — both of those blew the budget (GPU:
            # step-1 activations OOM; CPU anon: 4x13.5 GB swap-death during
            # init).  The mmap pages are shared across ranks and evictable.
            if device is not None and str(device).startswith("cuda"):
                try:
                    new_child.to(device)
                    _host = getattr(new_child, "_host_module", None)
                    if _host is not None:
                        _full = getattr(_host, "tp_full_weight", None)
                        if _full is not None and _full.is_cpu:
                            _spec = _host.tp_shard_spec
                            _host.weight.data = _spec.slice_weight(_full)
                except Exception:
                    pass

            # Progress output with throughput and ETA so long init quantization
            # does not look frozen. Updates at most once per second.
            replace_linear_with_mixed_quant._mq_counter += 1
            cnt = replace_linear_with_mixed_quant._mq_counter
            total = replace_linear_with_mixed_quant._mq_total
            now = _time.monotonic()
            dt = now - replace_linear_with_mixed_quant._mq_last_t
            replace_linear_with_mixed_quant._mq_last_t = now
            if dt > 0:
                inst = 1.0 / dt
                ema = replace_linear_with_mixed_quant._mq_ema
                replace_linear_with_mixed_quant._mq_ema = inst if ema is None else 0.3 * inst + 0.7 * ema
            if now - replace_linear_with_mixed_quant._mq_last_print >= 1.0 or cnt == total:
                replace_linear_with_mixed_quant._mq_last_print = now
                elapsed = now - replace_linear_with_mixed_quant._mq_start
                rate = replace_linear_with_mixed_quant._mq_ema or (cnt / elapsed if elapsed > 0 else 0.0)
                eta = (total - cnt) / rate if rate > 0 else None
                pct = 100.0 * cnt / total if total else 100.0
                # host-memory telemetry: catches which tensor family drives
                # per-rank anon growth during candidate loading (OOM forensics)
                try:
                    with open("/proc/self/status") as _st:
                        _anon = next(int(l.split()[1]) for l in _st if l.startswith("RssAnon"))
                    _anon_g = _anon / 1048576
                except Exception:
                    _anon_g = -1
                print(
                    f"\r  [{pct:5.1f}%] {cnt}/{total} layers | "
                    f"{rate:.2f} layer/s | elapsed {_format_duration(elapsed)} "
                    f"eta {_format_duration(eta)} | anon {_anon_g:.1f}G [{full_name}]   ",
                    end="", flush=True,
                )
                if cnt == total:
                    print()
                # Return freed candidate-load memory to the OS: torch's CPU
                # caching allocator and glibc arenas otherwise hold several GB
                # per rank through the whole run (4 ranks x ~4 GB = host OOM
                # territory on a 61 GB box shared with page cache).
                if cnt % 8 == 0:
                    gc.collect()
                    try:
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass
        else:
            sub_replaced = replace_linear_with_mixed_quant(
                child,
                candidate_types=candidate_types,
                prefix=full_name,
                target_names=target_names,
                skip_names=skip_names,
                imatrix_dict=imatrix_dict,
                device=device,
                lazy=lazy,
                attention_candidates=attention_candidates,
                non_attention_candidates=non_attention_candidates,
                qbytes_device=qbytes_device,
            )
            replaced.update(sub_replaced)

    return replaced


def replace_embedding_with_mixed_quant(
    module: nn.Module,
    candidate_types: list[str] | None,
    prefix: str = "",
    skip_names: Optional[set[str]] = None,
    imatrix_dict: Optional[dict[str, torch.Tensor]] = None,
    device: torch.device | str | None = None,
    lazy: bool = False,
    qbytes_device: torch.device | str | None = None,
    attention_candidates: Optional[list[str]] = None,
    non_attention_candidates: Optional[list[str]] = None,
) -> dict[str, MixedQuantEmbedding]:
    """
    Recursively replace selected nn.Embedding children with MixedQuantEmbedding.
    """
    if candidate_types is None:
        from voodoo_quant.ggml import list_quant_types
        candidate_types = list_quant_types()
    if skip_names is None:
        skip_names = set()
    if imatrix_dict is None:
        imatrix_dict = {}

    replaced: dict[str, MixedQuantEmbedding] = {}

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if name in skip_names:
            continue
        if isinstance(child, nn.Embedding):
            imatrix = imatrix_dict.get(full_name)
            if imatrix is None:
                imatrix = imatrix_dict.get(full_name + ".weight")
            tensor_candidates = resolve_candidate_types(
                full_name, candidate_types, attention_candidates, non_attention_candidates
            )
            new_child = MixedQuantEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                candidate_types=tensor_candidates,
                source_weight=child.weight.data,
                imatrix=imatrix,
                device=device,
                lazy=lazy,
                tensor_name=full_name,
                qbytes_device=qbytes_device,
                shard_spec=getattr(child, "tp_shard_spec", None),
                full_source_weight=getattr(child, "tp_full_weight", None),
            )
            setattr(module, name, new_child)
            replaced[full_name] = new_child
            print(f"  MixedQuantEmbedding init: {full_name} replaced", flush=True)
        else:
            sub_replaced = replace_embedding_with_mixed_quant(
                child,
                candidate_types=candidate_types,
                prefix=full_name,
                skip_names=skip_names,
                imatrix_dict=imatrix_dict,
                device=device,
                lazy=lazy,
                qbytes_device=qbytes_device,
                attention_candidates=attention_candidates,
                non_attention_candidates=non_attention_candidates,
            )
            replaced.update(sub_replaced)

    return replaced


def total_effective_bytes(replaced: dict[str, _MixedQuantBase]) -> torch.Tensor:
    total = torch.tensor(0.0)
    for layer in replaced.values():
        total = total + layer.effective_bytes()
    return total
