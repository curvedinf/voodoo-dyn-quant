"""Quantization specification for the MTP (nextn) sidecar.

The sidecar — the ``blk.N.nextn.*`` (and similar) tensors that exist in the
reference GGUF but not in the trained checkpoint's state dict — is copied
through during export. Historically it was rewritten as BF16/F32, which for a
27B-class model left ~810 MiB of BF16 weight where the reference's own
dynamic layout carries ~335 MiB (Q6_K, k/v Q8_0) — pure dead weight, since
the sidecar only matters under speculative decoding, not for perplexity.

This module prices the copy-through with real rules:

- big ``.weight`` matrices (eh_proj and, on models whose nextn layer has its
  own projections, attn/mlp/ssm weights) quantize to the requested level —
  default ``Q6_K``;
- ``attn_k``/``attn_v`` use ``kv_level`` (``Q8_0``) matching the measured
  reference layout;
- norms, per-head SSM state (``ssm_a``, ``ssm_dt``, conv kernels) and any
  tensor too small or misaligned for the target's block geometry stay F32;
- everything is quantized by the exact ggml quantizer, so the sidecar bytes
  are llama.cpp-identical like everything else in an export.

The spec is pure logic (no I/O) so it is unit-testable; the exporter applies
it to reference-only tensors.
"""

from __future__ import annotations

import re

from voodoo_quant.ggml import QUANT_LADDER, bytes_per_weight, get_quant_info

DEFAULT_SIDECAR_QUANT = "Q6_K"
DEFAULT_SIDECAR_KV_QUANT = "Q8_0"

# Tensor-name suffixes that must remain F32 regardless of the requested
# level: norms (RMS/LayerNorm gains), per-head SSM state, conv kernels, and
# any scalar/bias parameter.
_F32_SUFFIXES = (
    "norm.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head_norm.weight",
    ".ssm_a",
    ".ssm_dt.bias",
    ".ssm_dt.bias.weight",
    ".ssm_norm.weight",
    ".conv1d.weight",
    "conv1d.weight",
    ".conv.weight",
    ".shortconv.conv.weight",
    ".bias",
)

# The k/v projections of the sidecar's own attention (present on models
# whose nextn layer carries full projections, e.g. Qwen3.8-27B's blk.64).
_KV_SUFFIXES = (
    ".attn_k.weight",
    ".attn_v.weight",
)


def is_sidecar_name(gguf_name: str) -> bool:
    """True for reference-only tensor names that belong to an MTP sidecar."""
    return "nextn" in gguf_name


def _min_quantizable_numel(level: str) -> int:
    """Smallest tensor that can carry `level`: two block rows."""
    info = get_quant_info(level)
    return info.block_size * 2


def sidecar_quant_for_tensor(
    gguf_name: str,
    shape: tuple[int, ...],
    level: str = DEFAULT_SIDECAR_QUANT,
    kv_level: str = DEFAULT_SIDECAR_KV_QUANT,
) -> str | None:
    """Quant type for one sidecar tensor, or None to keep it F32.

    Args:
        gguf_name: GGUF tensor name (e.g. ``blk.24.nextn.eh_proj.weight``).
        shape: the tensor's [rows, cols] (or [n],) shape.
        level: requested level for large weights (default ``Q6_K``).
        kv_level: level for the sidecar's attention k/v (default ``Q8_0``,
            matching the measured reference layout).
    """
    for qt in (level, kv_level):
        get_quant_info(qt)  # raises on unknown types at spec time

    name = gguf_name.lower()
    if name.endswith(_F32_SUFFIXES):
        return None

    # 1-D tensors (biases, per-head state) and single-row tensors stay F32.
    if len(shape) < 2:
        return None
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 2 or cols < 1:
        return None

    chosen = kv_level if name.endswith(_KV_SUFFIXES) else level
    info = get_quant_info(chosen)
    if cols % info.block_size != 0:
        # block geometry does not fit; try the 32-wide Q8_0 as a fallback,
        # else keep F32
        if cols % 32 == 0 and rows * cols >= _min_quantizable_numel("Q8_0"):
            return "Q8_0"
        return None
    if rows * cols < _min_quantizable_numel(chosen):
        return None
    return chosen


def sidecar_budget(
    tensors: dict[str, tuple[int, ...]],
    level: str = DEFAULT_SIDECAR_QUANT,
    kv_level: str = DEFAULT_SIDECAR_KV_QUANT,
) -> dict[str, int]:
    """Exact stored bytes per sidecar tensor under the spec (name -> bytes)."""
    out: dict[str, int] = {}
    for name, shape in tensors.items():
        qt = sidecar_quant_for_tensor(name, shape, level, kv_level)
        numel = int(shape[0]) * (int(shape[1]) if len(shape) > 1 else 1)
        out[name] = int(numel * bytes_per_weight(qt)) if qt else numel * 4
    return out


def describe(level: str, kv_level: str = DEFAULT_SIDECAR_KV_QUANT) -> str:
    return (
        f"MTP sidecar: large weights -> {level}, attention k/v -> {kv_level}, "
        f"norms + per-head state -> F32 (ladder-rank "
        f"{QUANT_LADDER.index(level) if level in QUANT_LADDER else '?'}/"
        f"{len(QUANT_LADDER) - 1})"
    )
