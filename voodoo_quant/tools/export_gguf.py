#!/usr/bin/env python3
"""
Export a Voodoo (learned mixed-precision) PyTorch checkpoint to a GGUF.

The checkpoint is expected to contain a normal HF state dict plus a sidecar
quant_assignments.json that maps HF tensor names to GGUF quant types. Every
assigned linear weight is re-quantized with llama.cpp's C++ quantizer and
written with the assigned GGML type. Unassigned tensors (embeddings, norms,
biases, SSM params) are stored as BF16 or F32.

Usage:
    voodoo export \
        --checkpoint checkpoints/Qwen3.5-0.8B/Voodoo45/Qwen3.5-0.8B-Voodoo45.pt \
        --quant-assignments checkpoints/Qwen3.5-0.8B/Voodoo45/Qwen3.5-0.8B-Voodoo45.quant_assignments.json \
        --ref-gguf models/unsloth/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
        --output checkpoints/Qwen3.5-0.8B/Voodoo45/Qwen3.5-0.8B-Voodoo45.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType
from gguf import dequantize as gguf_dequantize
from gguf import quants as gguf_quants

from voodoo_quant.ggml import quantize_tensor
from voodoo_quant.layers import CANDIDATE_CACHE_DIR, _tensor_hash
from voodoo_quant.naming import canonicalize_gguf_path
from voodoo_quant.stats import log_stage


PROVENANCE_KEYS = {
    "general.repo_url",
    "general.quantized_by",
    "quantize.imatrix.chunks_count",
    "quantize.imatrix.dataset",
    "quantize.imatrix.entries_count",
    "quantize.imatrix.file",
}


QUANT_MAP = {
    "IQ1_S": GGMLQuantizationType.IQ1_S,
    "IQ1_M": GGMLQuantizationType.IQ1_M,
    "IQ2_XXS": GGMLQuantizationType.IQ2_XXS,
    "IQ2_XS": GGMLQuantizationType.IQ2_XS,
    "IQ2_S": GGMLQuantizationType.IQ2_S,
    "IQ3_XXS": GGMLQuantizationType.IQ3_XXS,
    "IQ3_S": GGMLQuantizationType.IQ3_S,
    "IQ4_XS": GGMLQuantizationType.IQ4_XS,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "TQ1_0": GGMLQuantizationType.TQ1_0,
    "TQ2_0": GGMLQuantizationType.TQ2_0,
}


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    """Copy model architecture metadata from the reference GGUF."""
    for key, field in reader.fields.items():
        if key == "general.architecture" or key.startswith("GGUF."):
            continue
        if key in PROVENANCE_KEYS:
            continue
        vtype = field.types[0]
        val = field.contents()
        if vtype == GGUFValueType.ARRAY:
            subtype = field.types[-1]
            if len(val) == 0:
                writer.add_key_value(key, [], vtype, sub_type=subtype)
            else:
                writer.add_key_value(key, val, vtype, sub_type=subtype)
        else:
            writer.add_key_value(key, val, vtype)


def tensor_to_gguf(
    tensor: torch.Tensor, ggml_type: GGMLQuantizationType
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Convert a PyTorch tensor to a GGUF-compatible numpy array."""
    if ggml_type == GGMLQuantizationType.BF16:
        f32 = tensor.detach().cpu().to(torch.float32).numpy()
        bf16_bytes = gguf_quants.BF16.quantize(f32)
        return bf16_bytes, bf16_bytes.shape
    if ggml_type == GGMLQuantizationType.F16:
        f16 = tensor.detach().cpu().to(torch.float16).numpy()
        return f16, f16.shape
    if ggml_type == GGMLQuantizationType.F32:
        f32 = tensor.detach().cpu().to(torch.float32).numpy()
        return f32, f32.shape
    raise ValueError(f"Unsupported GGML type: {ggml_type}")


# Set at runtime in run() from the reference GGUF. When the source model ties its
# embeddings, the reference GGUF has no separate ``output.weight`` tensor and
# ``lm_head.weight`` must be dropped (llama.cpp reuses token_embd.weight). When the
# model is untied (e.g. Qwen3.6-27B, tie_word_embeddings=False) the reference GGUF
# carries ``output.weight`` and we must write ``lm_head.weight`` as a (quantized)
# output.weight rather than dropping it.
_TIED_EMBEDDINGS = True
_ACTIVE_ADAPTER = None  # set by run() from the reference GGUF's arch string

# Qwen GDN head layout: K k-heads x V-per-K v-heads (head_dim 128). HF
# stores per-head tensors grouped [k][v]; llama.cpp's converter/runtime use
# tiled [v][k]. The 27B reference has 16 K-heads x 3 V-heads/K (48 V-heads)
# — verified bit-for-bit 2026-08-18 — but other sizes differ (0.8B: 8x2), so
# the counts are DERIVED from the reference GGUF's attn_qkv/attn_gate row
# counts at export time (see _derive_gdn_layout).
_GDN_K_HEADS = 16
_GDN_V_PER_K = 3
_GDN_HEAD_DIM = 128
_GDN_REORDER = True  # disabled when the reference has equal K/V head counts


def _derive_gdn_layout(reader: "GGUFReader", arch: str | None = None) -> None:
    """Derive the GDN head split from the reference GGUF's own hparams.

    Falls back to the defaults above when the reference carries no GDN
    metadata (non-hybrid models).
    """
    global _GDN_K_HEADS, _GDN_V_PER_K, _GDN_HEAD_DIM, _GDN_REORDER
    # Head counts come from the GGUF's own hparams (the converter wrote them
    # from the HF config): group count = K heads, time_step_rank = V heads.
    prefix = arch or "qwen35"

    def _field_u32(suffix: str) -> int | None:
        f = reader.fields.get(prefix + suffix)
        if f is None:
            return None
        try:
            return int(f.parts[f.data[0]][0])
        except Exception:
            return None

    k_heads = _field_u32(".ssm.group_count")
    v_heads = _field_u32(".ssm.time_step_rank")
    if not k_heads or not v_heads:
        return
    _GDN_K_HEADS = k_heads
    _GDN_V_PER_K = v_heads // k_heads if v_heads % k_heads == 0 else 1
    head_dim = _field_u32(".ssm.state_size") or _GDN_HEAD_DIM
    _GDN_HEAD_DIM = head_dim
    # Upstream converter rule (conversion/qwen.py modify_tensors): the V-head
    # grouped->tiled reorder is applied ONLY when num_k_heads != num_v_heads.
    # Equal counts (e.g. Qwen3.5-0.8B: 16 == 16) need NO reordering.
    _GDN_REORDER = k_heads != v_heads
    print(f"  GDN layout: {k_heads} K-heads, {v_heads} V-heads "
          f"({v_heads // k_heads if v_heads % k_heads == 0 else '?'}/K, head_dim {head_dim})"
          f"{'' if _GDN_REORDER else ' — heads equal, no V reorder needed'}")


def _reorder_v_heads(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Regroup per-head axis from HF [k][v] to llama.cpp tiled [v][k]."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    n = shape[dim]
    assert n == _GDN_K_HEADS * _GDN_V_PER_K * _GDN_HEAD_DIM or n % (_GDN_V_PER_K * _GDN_HEAD_DIM) == 0 or n == _GDN_K_HEADS * _GDN_V_PER_K, (
        f"bad head axis len {n} for layout {_GDN_K_HEADS}x{_GDN_V_PER_K}x{_GDN_HEAD_DIM}"
    )
    if n == _GDN_K_HEADS * _GDN_V_PER_K:
        hd = 1
    else:
        hd = _GDN_HEAD_DIM
    new_shape = shape[:dim] + [_GDN_K_HEADS, _GDN_V_PER_K, hd] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def _apply_gdn_reorder(gguf_name: str, hf_name: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply the tiled V-head reorder to GDN tensors on BOTH export paths.

    Skipped entirely when the reference model has equal K/V head counts
    (upstream converter rule — see _derive_gdn_layout).

    Row-wise (dim 0): ssm_a / ssm_dt.bias / ssm_alpha / ssm_beta (per V-head),
    attn_qkv V rows (after q+k rows), attn_gate rows, ssm_conv1d channels.
    Column-wise (dim 1): ssm_out in_features.
    """
    if not _GDN_REORDER:
        return weight
    if ".ssm_a" in gguf_name or ".ssm_dt.bias" in gguf_name or ".ssm_alpha" in gguf_name or ".ssm_beta" in gguf_name:
        return _reorder_v_heads(weight.float(), 0)
    if ".ssm_conv1d.weight" in gguf_name:
        # weight here is [conv_dim, 1, 4]; squeeze then reorder channel rows
        w = weight.squeeze(1).float()
        _qk = _GDN_HEAD_DIM * _GDN_K_HEADS * 2  # q+k channels pass through
        return torch.cat([w[:_qk], _reorder_v_heads(w[_qk:], 0)], dim=0)
    if ".attn_gate.weight" in gguf_name:
        return _reorder_v_heads(weight.float(), 0)
    if ".attn_qkv.weight" in gguf_name:
        w = weight.float()
        _qk = _GDN_HEAD_DIM * _GDN_K_HEADS * 2  # q(2048) + k(2048) rows unchanged
        return torch.cat([w[:_qk], _reorder_v_heads(w[_qk:], 0)], dim=0)
    if ".ssm_out.weight" in gguf_name:
        return _reorder_v_heads(weight.float(), 1)
    return weight


def map_hf_to_gguf(hf_name: str, arr: np.ndarray) -> tuple[str, np.ndarray, GGMLQuantizationType | None] | None:
    """
    Map an HF state-dict key to a GGUF tensor name and apply layout transforms.
    Returns (gguf_name, gguf_array, default_ggml_type) or None to skip.
    default_ggml_type is the type to use when the tensor is NOT in quant_assignments.

    An architecture adapter registered for the reference GGUF's arch string
    gets first claim on the mapping (see voodoo_quant.arch); the tables below
    are the qwen-family fallback and stay authoritative when no adapter
    matches or the adapter declines (returns None).
    """

    def rms(w: np.ndarray) -> np.ndarray:
        return w.astype(np.float32) + 1.0

    if _ACTIVE_ADAPTER is not None:
        gguf_name = _ACTIVE_ADAPTER.gguf_name(hf_name)
        if gguf_name is not None:
            if hf_name == "model.embed_tokens.weight" and _ACTIVE_ADAPTER.transpose_embedding():
                # llama.cpp stores this arch's embedding as [hidden, vocab].
                arr = arr.T.copy()
            is_norm = gguf_name.endswith("_norm.weight") or "norm" in gguf_name.rsplit(".", 1)[0]
            if hf_name == "model.embed_tokens.weight":
                return gguf_name, arr, None  # learned quant target; assignments decide
            if gguf_name.endswith(".shortconv.conv.weight"):
                # HF depthwise conv weight [hidden, 1, k] -> GGUF [hidden, k]
                return gguf_name, arr.squeeze(1).copy(), GGMLQuantizationType.F32
            if is_norm:
                return gguf_name, rms(arr), GGMLQuantizationType.F32
            return gguf_name, arr.copy(), None

    if hf_name == "model.embed_tokens.weight":
        # The embedding is a learned quant target. Let quant_assignments decide the
        # type; fall back to F32 only if no assignment is present.
        return "token_embd.weight", arr.copy(), None
    if hf_name == "model.norm.weight":
        return "output_norm.weight", rms(arr), GGMLQuantizationType.F32
    if hf_name == "lm_head.weight":
        if _TIED_EMBEDDINGS:
            # Tied embeddings: do not write a separate output.weight. llama.cpp will
            # reuse token_embd.weight for the language-modeling head.
            return None
        # Untied embeddings (e.g. Qwen3.6-27B): write a distinct output.weight.
        # quant_assignments decides the type; fall back to BF16 only if missing.
        return "output.weight", arr.copy(), GGMLQuantizationType.BF16

    m = re.match(r"model\.layers\.(\d+)\.input_layernorm\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_norm.weight", rms(arr), GGMLQuantizationType.F32

    m = re.match(r"model\.layers\.(\d+)\.post_attention_layernorm\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.post_attention_norm.weight", rms(arr), GGMLQuantizationType.F32

    m = re.match(r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_q.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_k.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_v.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_output.weight", arr.copy(), None

    m = re.match(r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_q_norm.weight", rms(arr), GGMLQuantizationType.F32
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_k_norm.weight", rms(arr), GGMLQuantizationType.F32

    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_qkv.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.attn_gate.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ssm_alpha.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ssm_beta.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.out_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ssm_out.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.conv1d\.weight", hf_name)
    if m:
        # HF: [conv_dim, 1, 4] -> GGUF: [conv_dim, 4]
        return f"blk.{m.group(1)}.ssm_conv1d.weight", arr.squeeze(1).copy(), GGMLQuantizationType.F32
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.A_log", hf_name)
    if m:
        # HF A_log = log(-GGUF ssm_a)  =>  GGUF ssm_a = -exp(A_log)
        a = -np.exp(arr.copy().astype(np.float32))
        return f"blk.{m.group(1)}.ssm_a", a, GGMLQuantizationType.F32
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.dt_bias", hf_name)
    if m:
        return f"blk.{m.group(1)}.ssm_dt.bias", arr.copy(), GGMLQuantizationType.F32
    m = re.match(r"model\.layers\.(\d+)\.linear_attn\.norm\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ssm_norm.weight", arr.copy(), GGMLQuantizationType.F32

    m = re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ffn_gate.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.mlp\.up_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ffn_up.weight", arr.copy(), None
    m = re.match(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight", hf_name)
    if m:
        return f"blk.{m.group(1)}.ffn_down.weight", arr.copy(), None

    return None


def add_quantized_tensor(
    writer: GGUFWriter,
    name: str,
    weight: torch.Tensor,
    quant_type: str,
) -> None:
    """Quantize a weight and add it to the GGUF writer."""
    qweight = quantize_tensor(weight.detach().cpu().to(torch.float32), quant_type)
    out_features = weight.shape[0]
    block_bytes = qweight.shape[1]
    nblocks_per_row = qweight.shape[0] // out_features
    # GGUF raw shape is [out_features, nblocks_per_row * block_bytes]
    raw = qweight.reshape(out_features, nblocks_per_row * block_bytes).contiguous().numpy()
    ggml_type = QUANT_MAP[quant_type]
    writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=ggml_type)


def _quantize_only(
    name: str,
    weight: torch.Tensor,
    quant_type: str,
    hf_name: str = "",
) -> tuple[str, bytes, tuple, str]:
    """Quantize a single weight and return the raw bytes + metadata.

    Thread-safe: quantize_tensor releases the GIL during the C call.
    Returns (gguf_name, raw_bytes, raw_shape, ggml_type_str).

    Memory-safe: `weight` is an mmap-backed bf16 view; the fp32 cast happens
    here (per-tensor, transient) unless the exact quantized bytes are already
    in the candidate cache, in which case they are copied straight from mmap.
    """
    import os
    torch.set_num_threads(1)
    out_features = weight.shape[0]
    if hf_name:
        cached = _load_cached_qbytes(hf_name, quant_type, weight)
        if cached is not None:
            raw = cached.contiguous().numpy()
            return name, raw, raw.shape, quant_type
    qweight = quantize_tensor(weight.detach().cpu().to(torch.float32), quant_type)
    block_bytes = qweight.shape[1]
    nblocks_per_row = qweight.shape[0] // out_features
    raw = qweight.reshape(out_features, nblocks_per_row * block_bytes).contiguous().numpy()
    return name, raw, raw.shape, quant_type


def _load_cached_qbytes(hf_name: str, quant_type: str, weight: torch.Tensor) -> torch.Tensor | None:
    """Return cached quantized bytes for (hf tensor, quant_type), or None.

    The candidate cache holds llama.cpp-quantizer output keyed by the weight
    hash — bit-identical to quantize_tensor(weight, quant_type) — so an export
    can reuse it instead of re-running the C quantizer and materializing the
    fp32 weight. Falls back to None whenever the exact generation/type is not
    cached (caller then quantizes from the weight).
    """
    try:
        from voodoo_quant.ggml import get_quant_info

        stripped = hf_name.removeprefix("model.")
        weight_hash = _tensor_hash(weight)
        p = CANDIDATE_CACHE_DIR / stripped.replace(".", "_") / f"combined_lazy_w{weight_hash}_inone.pt"
        if not p.exists():
            return None
        combined = torch.load(p, weights_only=True, map_location="cpu", mmap=True)
        qb = combined.get(quant_type)
        if qb is None:
            return None
        out_features = weight.shape[0]
        nblocks_per_row = weight.shape[1] // get_quant_info(quant_type).block_size
        return qb.reshape(out_features, -1)
    except Exception:
        return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export a Voodoo mixed-precision checkpoint to a llama.cpp-exact GGUF"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--quant-assignments", required=True)
    parser.add_argument("--ref-gguf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--quant-label",
        default=None,
        help="Override the UD-equivalent quant label used in the canonical "
             "GGUF filename (e.g. IQ2_M, Q6_K). Defaults to the curated "
             "per-size label in voodoo_quant.naming.",
    )
    return parser


def run(args):
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state_dict = ckpt["model_state_dict"]

    print(f"Loading quant assignments from {args.quant_assignments} ...")
    quant_assignments = json.loads(Path(args.quant_assignments).read_text())
    print(f"  {len(quant_assignments)} assignments")

    print(f"Loading reference GGUF from {args.ref_gguf} ...")
    reader = GGUFReader(args.ref_gguf)
    arch = reader.fields["general.architecture"].contents()

    global _TIED_EMBEDDINGS, _ACTIVE_ADAPTER
    _TIED_EMBEDDINGS = not any(t.name == "output.weight" for t in reader.tensors)
    print(f"  reference GGUF has separate output.weight: {not _TIED_EMBEDDINGS} "
          f"(tied_embeddings={_TIED_EMBEDDINGS})")

    from voodoo_quant.arch import adapter_for_gguf_arch

    _ACTIVE_ADAPTER = adapter_for_gguf_arch(str(arch))
    _derive_gdn_layout(reader, arch=str(arch))
    if _ACTIVE_ADAPTER is not None:
        print(f"  architecture adapter: {_ACTIVE_ADAPTER.__name__} (arch={arch})")
    else:
        print(f"  no architecture adapter for arch={arch}; using built-in mapping tables")

    writer = GGUFWriter(args.output, arch)
    copy_metadata(reader, writer)

    # ---- Phase 1: collect all tensors to quantize ----
    quant_jobs: list[tuple[str, str, torch.Tensor, str]] = []  # (gguf_name, hf_name, weight, quant_type)
    passthrough_jobs: list[tuple[str, np.ndarray, GGMLQuantizationType | None, str]] = []  # (gguf_name, arr, default_type, mode)
    written_gguf_names: set[str] = set()
    skipped = 0

    for hf_name, weight in state_dict.items():
        if not isinstance(weight, torch.Tensor):
            continue
        # Streaming: `map_hf_to_gguf` needs only the NAME + shape here.  Pass a
        # zero-size numpy shell (supports .copy/.astype/.squeeze like the real
        # array would) so no fp32/bf16 bytes are materialized; real data is
        # touched only in the worker (fp32 cast per-tensor, or cache bytes).
        shell = np.empty(weight.shape, dtype=np.float32)
        mapped = map_hf_to_gguf(hf_name, shell)
        # Language-model-only checkpoints store STRIPPED keys (layers.N.*,
        # embed_tokens.*, norm.*) while the mappers and quant assignments use
        # the full model.* spelling — try the prefixed name as a fallback.
        if mapped is None:
            prefixed = "model." + hf_name
            mapped = map_hf_to_gguf(prefixed, shell)
            if mapped is not None:
                hf_name = prefixed
        if mapped is None:
            skipped += 1
            print(f"Skipping unmapped HF tensor: {hf_name}", flush=True)
            continue
        gguf_name, _arr, default_type = mapped
        del _arr  # shell bytes are garbage; only the mapping metadata is used

        qa_key = hf_name
        if qa_key not in quant_assignments and qa_key.endswith(".weight"):
            qa_key = qa_key[:-7]
        if qa_key in quant_assignments:
            quant_type = quant_assignments[qa_key]
            if quant_type in QUANT_MAP:
                quant_jobs.append((gguf_name, hf_name, weight, quant_type))
                written_gguf_names.add(gguf_name)
                continue
            else:
                print(f"Unknown quant type '{quant_type}' for {hf_name}; falling back to default storage")

        # Default storage for unassigned tensors
        if isinstance(default_type, str):
            quant_jobs.append((gguf_name, hf_name, weight, default_type))
        else:
            passthrough_jobs.append((gguf_name, hf_name, weight, default_type, "passthrough"))
        written_gguf_names.add(gguf_name)

    # ---- Phase 2: parallelize all quantization ----
    import concurrent.futures
    finalize_workers = int(os.environ.get("VOODOO_FINALIZE_WORKERS", min(8, os.cpu_count() or 2)))
    old_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Quantizing {len(quant_jobs)} tensors with {finalize_workers} workers ...", flush=True)
    quantized_results: dict[str, tuple] = {}  # gguf_name -> (raw, raw_shape, quant_type)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=finalize_workers) as pool:
        futures = {}
        for gguf_name, hf_name, weight, qt in quant_jobs:
            # Apply the GDN tiled V-head reorder BEFORE quantizing so the
            # quantized bytes land in the layout llama.cpp expects (the
            # reorder is exact/lossless; quantization runs on reordered rows).
            weight = _apply_gdn_reorder(gguf_name, hf_name, weight)
            fut = pool.submit(_quantize_only, gguf_name, weight, qt, hf_name)
            futures[fut] = gguf_name
        for fut in concurrent.futures.as_completed(futures):
            name, raw, raw_shape, qt = fut.result()
            quantized_results[name] = (raw, raw_shape, qt)
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == len(quant_jobs):
                print(f"  quantized {completed}/{len(quant_jobs)} tensors ...", flush=True)

    if old_omp is not None:
        os.environ["OMP_NUM_THREADS"] = old_omp

    # ---- Phase 3: write all tensors to GGUF (sequential, fast) ----
    quantized = 0
    written = 0
    for gguf_name, hf_name, weight, qt in quant_jobs:
        raw, raw_shape, quant_type = quantized_results[gguf_name]
        ggml_type = QUANT_MAP[quant_type]
        writer.add_tensor(gguf_name, raw, raw_shape=raw_shape, raw_dtype=ggml_type)
        quantized += 1

    for gguf_name, hf_name, weight, default_type, mode in passthrough_jobs:
        ggml_type = default_type or GGMLQuantizationType.F32
        # Apply the mapper's layout transform to the REAL tensor (the mapping
        # pass ran on a zero-size shell, so name-only transforms like the
        # conv1d squeeze must be redone here on actual data).
        arr = weight.detach().cpu()
        if gguf_name.endswith(".ssm_conv1d.weight"):
            # HF: [conv_dim, 1, 4] -> GGUF: [conv_dim, 4]
            arr = arr.squeeze(1)
        elif gguf_name.endswith(".ssm_a"):
            # HF stores A_log; GGUF ssm_a = -exp(A_log).
            arr = -torch.exp(arr.to(torch.float32))
        # GDN tiled V-head reorder (see _apply_gdn_reorder): applies to the
        # per-head F32 tensors here and, via the quantized path, to the big
        # matrices. The reference GGUF stores tiled order — verified 2026-08-18.
        arr = _apply_gdn_reorder(gguf_name, hf_name, arr)
        if gguf_name.endswith((".ssm_alpha.weight", ".ssm_beta.weight", ".ssm_dt.bias", ".ssm_out.weight", ".attn_gate.weight")):
            arr = arr.to(torch.float32)
        if gguf_name.endswith(
            ("_norm.weight", ".attn_norm.weight", ".post_attention_norm.weight")
        ) and not gguf_name.endswith((".ssm_norm.weight", ".ssm_dt.bias")):
            # RMS norm weights are stored pre-shifted (+1.0) in GGUF for this
            # arch (matches the UD reference files); llama.cpp does not add 1.
            arr = arr.to(torch.float32) + 1.0
        data, shape = tensor_to_gguf(arr, ggml_type)
        writer.add_tensor(gguf_name, data, raw_shape=shape, raw_dtype=ggml_type)
        written += 1

    # Copy any tensors present in the reference GGUF but not in the checkpoint
    # state dict. For MTP variants this preserves the nextn_predict sidecar
    # (e.g. blk.N.nextn.* tensors) so the exported GGUF remains a valid MTP model.
    nextn_copied = 0
    for ref_tensor in reader.tensors:
        if ref_tensor.name in written_gguf_names:
            continue
        arr = gguf_dequantize(ref_tensor.data, ref_tensor.tensor_type).astype(np.float32)
        if ref_tensor.tensor_type == GGMLQuantizationType.F32:
            data, shape = arr, arr.shape
            ggml_type = GGMLQuantizationType.F32
        else:
            ggml_type = GGMLQuantizationType.BF16
            data, shape = tensor_to_gguf(torch.from_numpy(arr), ggml_type)
        writer.add_tensor(ref_tensor.name, data, raw_shape=shape, raw_dtype=ggml_type)
        nextn_copied += 1

    print(f"Wrote {quantized} quantized tensors, {written} default tensors; skipped {skipped}; copied {nextn_copied} reference-only tensors (MTP sidecar)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_path = Path(args.output)
    print(f"Wrote {args.output} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Rename to the canonical Voodoo GGUF name
    # (<slug>.Voodoo{NN}_{QUANT}.gguf) so Hugging Face shows it on the GGUF card.
    canonical = canonicalize_gguf_path(
        out_path, label_override=getattr(args, "quant_label", None)
    )
    if canonical != out_path:
        out_path.rename(canonical)
        print(f"Renamed to canonical Voodoo name: {canonical}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    model_dir = Path(args.output).parent
    with log_stage(
        stage="export",
        model_dir=model_dir,
        script=Path(__file__).name,
        args=vars(args),
    ):
        run(args)


if __name__ == "__main__":
    main()
