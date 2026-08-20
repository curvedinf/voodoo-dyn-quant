#!/usr/bin/env python3
"""
Train per-tensor dynamic quantization assignment.

Loads a source model, replaces selected Linear layers with a softmax mixture of
K candidate quant types, and learns the assignment via KL distillation against
the frozen source model plus a size-budget loss.  The output is a `.pt` file
containing the original source state dict plus a `quant_assignments` mapping
that the export/eval tools consume (see `voodoo_quant.tools`).

Usage:
    voodoo train \
        --model Qwen/Qwen3.5-0.8B-Base \
        --compression_ratio 0.45 \
        --candidate_types IQ2_XXS IQ3_XXS Q4_K \
        --max_steps 50 \
        --seq_len 512 --batch_size 1 --lr 0.5
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import re
import sys
from pathlib import Path

# Must be the FIRST voodoo_quant import: it points the Triton/Inductor caches
# at the project tree before any torch-heavy import compiles kernels.
import voodoo_quant.cache  # noqa: E402, F401

import os

# Dramatically reduce first-time FLA/Triton compile latency by skipping the
# autotuning benchmark sweep.  The first kernel configuration is compiled and
# used directly.  Set VOODOO_DISABLE_TRITON_AUTOTUNE=1 when autotune stalls.
if os.environ.get("VOODOO_DISABLE_TRITON_AUTOTUNE", "0") == "1":
    import triton
    from triton.runtime.autotuner import Autotuner

    def _fast_run(self, *args, **kwargs):
        config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            self.nargs = dict(zip(self.arg_names, args))
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    def _fast_warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        config = self.configs[0]
        ret = [self.fn.warmup(*args, **kwargs, **config.all_kwargs())]
        self.nargs = None
        return ret

    Autotuner.run = _fast_run
    Autotuner.warmup = _fast_warmup

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Keep PyTorch and any OpenMP-backed quantizer single-threaded on CPU to avoid
# oversubscription during candidate quantization init.
torch.set_num_threads(1)

from voodoo_quant.stats import log_stage
from voodoo_quant.layers import (
    MixedQuantEmbedding,
    MixedQuantLinear,
    _pad_weight_and_imatrix,
    install_heap_dump_handler,
    replace_embedding_with_mixed_quant,
    replace_linear_with_mixed_quant,
    total_effective_bytes,
)
from voodoo_quant.ggml import bytes_per_weight, dequantize_tensor, quantize_tensor


class TokenizedTensorDataset:
    """Sliding-window dataset over a pre-tokenized 1-D tensor."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.tokens = tokens

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1].long(), chunk[1:].long()



def _tp_handshake(_dist):
    """Barrier that works on gfx908/NCCL-P2P-disabled: tiny all-reduce on a
    CUDA tensor on each rank's own device (plain dist.barrier() can hang
    here)."""
    import torch as _t
    _dev = _t.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", "0")))
    _one = _t.ones(1, device=_dev)
    _dist.all_reduce(_one)
    return


def build_dataloader(tokenizer, seq_len: int, batch_size: int, data_dir: str, data_name: str):
    data_path = Path(data_dir) / data_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training tokens not found at {data_path}. "
            f"Run data preparation (see the data skill / `voodoo data`) or use "
            f"--data_dir data/qwen35-0.8b --data_name train_tokens.pt"
        )
    tokens = torch.load(data_path, weights_only=True)
    dataset = TokenizedTensorDataset(tokens, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


def bits_per_element(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 32.0
    if dtype in (torch.float16, torch.bfloat16):
        return 16.0
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        return 8.0
    if dtype == torch.int8:
        return 8.0
    # FP4 and other small types must be handled by the caller; default to 16.
    return 16.0


def tensor_bytes(t: torch.Tensor) -> float:
    return t.numel() * bits_per_element(t.dtype) / 8.0


def _storage_ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def compute_total_8bit_bytes(state_dict_meta: dict[str, tuple[int, int]]) -> float:
    """Total bytes if every parameter were stored at 8-bit precision.

    Voodoo sizes are defined as a percentage of the original model size at
    8-bit precision, so the byte budget uses this as its denominator.

    Tied weights (e.g. `lm_head.weight` sharing `embed_tokens.weight`) share the
    same underlying storage and must only be counted once, matching how GGUF
    stores a single tensor for them.

    Args:
        state_dict_meta: mapping from key to (storage_data_ptr, numel).
    """
    seen: set[int] = set()
    total = 0.0
    for ptr, numel in state_dict_meta.values():
        if ptr not in seen:
            seen.add(ptr)
            total += numel * 1.0
    return total


def compute_non_targeted_8bit_bytes(
    state_dict_meta: dict[str, tuple[int, int]],
    targeted_keys: set[str],
) -> float:
    """8-bit-normalized bytes of parameters that are not being dynamically quantized.

    Shared storages are counted once. If any key sharing a storage is targeted,
    the whole storage is treated as targeted.
    """
    ptr_bytes: dict[int, float] = {}
    ptr_targeted: dict[int, bool] = {}
    for k, (ptr, numel) in state_dict_meta.items():
        ptr_bytes[ptr] = numel * 1.0
        if ptr not in ptr_targeted:
            ptr_targeted[ptr] = False
        if k in targeted_keys:
            ptr_targeted[ptr] = True
    return sum(bytes_ for ptr, bytes_ in ptr_bytes.items() if not ptr_targeted[ptr])


def anneal_temperature(step: int, max_steps: int, start: float, end: float) -> float:
    if max_steps <= 1:
        return end
    frac = step / (max_steps - 1)
    return start * (end / start) ** frac


# ---------------------------------------------------------------------------
# Resilient finalization + journaled partial checkpoints
# ---------------------------------------------------------------------------
# The permanent-quantization ("bake") step used to read each original weight
# from `source_model.state_dict()[name + ".weight"]`.  After replacement the
# nn.Linear / nn.Embedding is a MixedQuant* module whose original weight was
# consumed to build candidates and is NOT registered as `.weight`, so that key
# is gone and the bake crashed with KeyError (losing the whole run).  The
# helpers below source the original weight from an authoritative mapping that is
# always complete (a memory-mapped base checkpoint, or a pre-replacement CPU
# snapshot), skip+warn instead of raising, and persist the learned assignments
# plus a rolling partial checkpoint before doing any risky work so a crash can
# be recovered with `--finalize_from_partial`.


def _snapshot_selectable_weights(
    model: torch.nn.Module,
    skip_names: set[str],
) -> dict[str, torch.Tensor]:
    """Capture CPU copies of every Linear/Embedding weight BEFORE replacement.

    Used only when there is no `--base_checkpoint` to mmap.  After replacement
    the MixedQuant* modules drop their `.weight`, so this snapshot is the only
    remaining source of the original weights for the bake.  Restricted to
    nn.Linear / nn.Embedding modules (the replaceable set) so it does not clone
    the whole model.  Returns a mapping of `name + ".weight"` -> CPU tensor.
    """
    snap: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name in skip_names:
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            if hasattr(module, "weight") and module.weight is not None:
                snap[name + ".weight"] = module.weight.detach().clone().cpu()
    return snap


def _build_original_sd(args, source_model, pre_replacement_snapshot):
    """Return (original_sd, cleanup) — an authoritative COMPLETE weight mapping.

    With `--base_checkpoint` this is the memory-mapped checkpoint state dict
    (lazy, ~0 extra RSS).  Without it, we merge the post-replacement source
    state dict (norms/biases/unreplaced) with the pre-replacement CPU snapshot
    (the replaced weights the MixedQuant* modules dropped).  Either way the
    result contains every key, so the bake and carry-over cannot KeyError.
    """
    if getattr(args, "base_checkpoint", None) is not None:
        return _open_base_checkpoint_mmap(args)
    merged = {**source_model.state_dict(), **(pre_replacement_snapshot or {})}
    return merged, None


def _open_base_checkpoint_mmap(args):
    """Open the base checkpoint mmap for lazy original-weight access.

    Returns (state_dict_like, cleanup) where cleanup releases the handle.  Returns
    (None, None) when no base checkpoint is configured.
    """
    if getattr(args, "base_checkpoint", None) is None:
        return None, None
    handle = torch.load(args.base_checkpoint, weights_only=True, map_location="cpu", mmap=True)
    sd = handle["model_state_dict"]

    def _cleanup():
        try:
            del sd
            del handle
        except Exception:
            pass

    return sd, _cleanup


def _resolve_weight(original_sd, weight_key: str):
    """Look up an original weight without raising.

    Returns (tensor, label) on success or (None, reason) when the key cannot be
    resolved.  `original_sd` is any mapping supporting `in` / indexing (a plain
    dict or a memory-mapped checkpoint state dict).
    """
    try:
        if weight_key in original_sd:
            return original_sd[weight_key], "original_sd"
        # TP replacement registers modules as `model.layers.N...` while the
        # language-model-only base checkpoint stores stripped `layers.N...`
        # keys — try both spellings.
        alt = weight_key.removeprefix("model.") if weight_key.startswith("model.") else "model." + weight_key
        if alt in original_sd:
            return original_sd[alt], "original_sd(alt-prefix)"
    except Exception as exc:  # mapping may be lazy/mmap; never crash the bake
        return None, f"{type(exc).__name__}: {exc}"
    return None, "key not present in original weights source"


def _partial_paths(args) -> tuple[Path, Path]:
    out_dir = Path(args.output_dir)
    return out_dir / "partial.pt", out_dir / "partial.meta.json"


def _atomic_torch_save(payload, path: Path) -> None:
    """Write `payload` to `path` atomically (tmp + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_text_save(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_partial(args, replaced, completed_opt_steps: int, tau: float, reason: str = "interval") -> bool:
    """Save a single rolling journaled partial checkpoint (gates + assignments).

    Overwrites `partial.pt` every call so only one partial exists.  Never raises:
    a failure to write the partial must not crash training.  Returns True on
    success.
    """
    partial_path, meta_path = _partial_paths(args)
    try:
        gates = {name: layer.gates.detach().cpu().clone() for name, layer in replaced.items()}
        assignments = {name: layer.get_assignment() for name, layer in replaced.items()}
        candidate_types = next(iter(replaced.values())).candidate_types if replaced else []
        meta = {
            "completed_optimizer_steps": int(completed_opt_steps),
            "tau": float(tau),
            "reason": reason,
            "time": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": args.model,
            "compression_ratio": getattr(args, "compression_ratio", None),
            "candidate_types": list(candidate_types),
            "output_name": args.output_name,
        }
        payload = {
            "format": "voodoo.partial.v1",
            "gates": gates,
            "assignments": assignments,
            "meta": meta,
        }
        _atomic_torch_save(payload, partial_path)
        _atomic_text_save(json.dumps(meta, indent=2, sort_keys=True), meta_path)
        print(
            f"  [partial] saved {partial_path} (opt_steps={completed_opt_steps}, "
            f"tensors={len(gates)}, reason={reason})",
            flush=True,
        )
        return True
    except Exception as exc:  # partial IO must never bring down a run
        print(f"  [partial] WARNING: failed to save partial checkpoint: {exc}", flush=True)
        return False


def load_partial(path: str) -> dict:
    """Load a partial checkpoint payload (raises on failure)."""
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(payload, dict) or "gates" not in payload:
        raise ValueError(f"Unrecognized partial checkpoint format at {path}")
    return payload


class _ReplacedShim:
    """Read-only stand-in for a MixedQuant layer on a NON-owning TP rank.

    Built by ``_tp_gather_replaced`` from the owning rank's gates so the
    rank-0 partial saves / finalization see every tensor under its full name.
    """

    def __init__(self, gates, assignment, prob, candidate_types, imatrix):
        self.gates = gates
        self._assignment = assignment
        self._prob = prob
        self.candidate_types = candidate_types
        self.imatrix = imatrix

    def get_assignment(self) -> str:
        return self._assignment

    def get_assignment_prob(self) -> float:
        return self._prob


def _tp_gather_replaced(replaced) -> dict:
    """Merge every rank's local {name: layer} mapping into one rank-0 view.

    Gate VALUES are identical across ranks for shared tensors (same zero init,
    all-reduced gradients, deterministic AdamW), so the merge is a union; the
    first rank claiming a name wins.  Uses ``all_gather_object`` (small CPU
    payloads: a few hundred names + tiny gate tensors).
    """
    from voodoo_quant.parallel import TP as tp

    local = {
        name: (
            layer.gates.detach().cpu().clone(),
            layer.get_assignment(),
            layer.get_assignment_prob(),
            list(layer.candidate_types),
            None if layer.imatrix is None else layer.imatrix.cpu().clone(),
        )
        for name, layer in replaced.items()
    }
    gathered: list = [None] * tp.world_size
    torch.distributed.all_gather_object(gathered, local)
    merged = {}
    for chunk in gathered:
        for name, (gates, assignment, prob, ctypes, imat) in chunk.items():
            if name not in merged:
                merged[name] = _ReplacedShim(gates, assignment, prob, ctypes, imat)
    return merged


def _tp_is_rank0() -> bool:
    """True on the logging/checkpointing rank (rank 0, or always without TP)."""
    from voodoo_quant.parallel import TP as tp

    return tp.rank == 0


def _tp_all_reduce_gate_grads(trainable) -> None:
    """Sum gate gradients across TP ranks in place (each rank owns a slice)."""
    from voodoo_quant.parallel import TP as tp

    if not tp.enabled:
        return
    for g in trainable:
        if g.grad is None:
            continue
        torch.distributed.all_reduce(g.grad, op=torch.distributed.ReduceOp.SUM)



class LMWithHead(nn.Module):
    """Qwen3_5TextModel + separate lm_head, matching stripped checkpoint keys.

    The Qwen3.8-27B language-model base checkpoint (and its Q8_0 teacher) store
    keys as ``layers.N.*`` / ``embed_tokens`` / ``lm_head`` — exactly the state
    dict of ``Qwen3_5TextModel`` plus a head.  Building ``AutoModelForCausalLM``
    instead would expect a ``model.`` prefix, miss every key, and rename the
    candidate-cache tensor paths away from the prebuilt cache.  This wrapper
    keeps module paths identical to the cache keys.
    """

    def __init__(self, text_model: nn.Module, vocab_size: int, logits_chunk: int = 0):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        hidden = text_model.config.hidden_size
        dtype = next(text_model.parameters()).dtype
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False).to(dtype)
        self.logits_chunk = logits_chunk

    def forward(self, input_ids=None, **kw):
        out = self.model(input_ids=input_ids, **kw)
        h = out.last_hidden_state
        if self.logits_chunk > 0 and torch.is_grad_enabled():
            flat = h.reshape(-1, h.shape[-1])
            parts = [self.lm_head(flat[i : i + self.logits_chunk]) for i in range(0, flat.shape[0], self.logits_chunk)]
            logits = torch.cat(parts, dim=0).reshape(*h.shape[:-1], -1)
        else:
            logits = self.lm_head(h)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)


def _register_flash_attn_triton():
    """Register the curvedinf/flash-attention Triton-AMD backend with transformers.

    Thin alias: the registration lives in ``voodoo_quant.hardware.rocm_flash``
    (vendor-specific code stays out of the trainer).  Lazy-imported at the call
    site so non-ROCm hosts never import it.
    """
    from voodoo_quant.hardware.rocm_flash import register_rocm_triton

    return register_rocm_triton()


def _tiny_recipe_override(cfg):
    """Shrink a Qwen3.5-family config to a tiny model for CLI integration tests.

    Enabled by VOODOO_TINY_RECIPE=1: keeps the architecture (hybrid
    linear/full attention, gated q, untied head) while making a training run
    take seconds on small GPUs.
    """
    import os
    if os.environ.get("VOODOO_TINY_RECIPE", "0") != "1":
        return cfg
    tests_dir = Path(__file__).parent.parent.parent / "tests"
    sys.path.insert(0, str(tests_dir))
    try:
        from tiny_recipe import TINY_CFG
    except ImportError:
        print(f"  [tiny-recipe] tests/tiny_recipe.py not found under {tests_dir}; skipping", flush=True)
        return cfg

    for k, v in TINY_CFG.items():
        setattr(cfg, k, v)
    return cfg


def _build_lm_model(args, dtype, attn_impl):
    """Build the student model, honoring the text-model wrapper for stripped
    language-model-only checkpoints (Qwen3.8-27B style).

    The 27B model is constructed on the META device (zero RAM) and weights are
    assigned from the mmap'd base checkpoint with ``assign=True`` (lazy page
    mapping, ~zero RSS); ``_shard_model`` then streams them to the GPUs.  A
    non-meta construction OOM-kills this 61 GB host (a 27B fp32 init alone
    exceeds RAM).
    """
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.text_model_class:
        from transformers.models.qwen3_5 import Qwen3_5TextModel

        # Qwen3_5's attention dispatches through ALL_ATTENTION_FUNCTIONS, so
        # flex_attention works once the class capability flag is enabled.
        Qwen3_5TextModel._supports_flex_attn = True
        text_cfg = getattr(cfg, "text_config", cfg)
        text_cfg = _tiny_recipe_override(text_cfg)
        if attn_impl is not None:
            text_cfg._attn_implementation = attn_impl
        with torch.device("meta"):
            text_model = Qwen3_5TextModel(text_cfg)
            vocab = getattr(text_cfg, "vocab_size", None)
            model = LMWithHead(text_model, vocab, logits_chunk=args.logits_chunk)
        if args.base_checkpoint is not None:
            ckpt = torch.load(args.base_checkpoint, weights_only=True, map_location="cpu", mmap=True)
            # LMWithHead nests the text model as .model, so its state-dict keys
            # carry a `model.` prefix, while the language-model-only base
            # checkpoint uses the stripped `layers.N...` form.  Strip both to a
            # common form for the assign-load (cheap dict view over mmap pages).
            raw_sd = ckpt["model_state_dict"]
            if "layers.0.input_layernorm.weight" in raw_sd:
                load_sd = {"model." + k: v for k, v in raw_sd.items() if k != "lm_head.weight"}
                load_sd["lm_head.weight"] = raw_sd["lm_head.weight"]
            else:
                load_sd = raw_sd
            missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)
            loaded = len(load_sd) - len(unexpected)
            print(f"  text-model wrapper: {loaded} tensors loaded, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected", flush=True)
            # Any parameter/buffer the checkpoint did not cover is still a meta
            # tensor; materialize it (empty) so failures happen loudly at use,
            # not at load.  (Complete checkpoints hit none of these.)
            fixed = 0
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            with torch.no_grad():
                for name, module in model.named_modules():
                    for pname, p in list(module.named_parameters(recurse=False)):
                        if p.is_meta:
                            new = torch.empty_like(p, device="cpu").normal_(0, 0.02)
                            module._parameters[pname] = new
                            fixed += 1
                    for bname, b in list(module.named_buffers(recurse=False)):
                        if b is not None and b.is_meta:
                            new = torch.zeros_like(b, device="cpu")
                            module._buffers[bname] = new
                            fixed += 1
            if fixed:
                print(f"  WARNING: {fixed} tensors missing from checkpoint; initialized empty", flush=True)
            del ckpt
            gc.collect()
        return model
    if args.base_checkpoint is not None:
        source_model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, attn_implementation=attn_impl
        ).to(dtype)
        ckpt = torch.load(args.base_checkpoint, weights_only=True, map_location="cpu", mmap=True)
        source_model.load_state_dict(ckpt["model_state_dict"], strict=False, assign=True)
        del ckpt
        gc.collect()
        return source_model
    return AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, attn_implementation=attn_impl
    )


def _layer_device_hop_hook(layer_device: torch.device):
    """Forward-pre-hook moving a decoder layer's inputs onto its shard device.

    Module names stay untouched (unlike a wrapper class), so MixedQuant tensor
    paths keep matching the candidate-cache keys.  Plain tensors and tuples of
    them are moved directly; BlockMask (flex attention) cannot be moved, so it
    is rebuilt for this device once and cached on the mask object (the block
    structure is device-independent, only the index tensors live on a device).
    """

    def hook(module, args, kwargs):
        dev = layer_device
        src = None
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                src = a.device
                break

        def _mv(v):
            if isinstance(v, torch.Tensor):
                return v.to(dev) if v.device != dev else v
            if isinstance(v, tuple):
                return tuple(_mv(t) for t in v)
            tn = type(v).__name__
            if tn == "BlockMask" and getattr(v, "device", None) != dev:
                cache = getattr(v, "_voodoo_dev_copies", None)
                if cache is None:
                    cache = {}
                    v._voodoo_dev_copies = cache
                if dev not in cache:
                    cache[dev] = v.to(dev)
                return cache[dev]
            if v is not None and hasattr(v, "to") and hasattr(v, "device"):
                try:
                    return v.to(dev)
                except Exception:
                    return v
            return v

        args = tuple(_mv(a) for a in args)
        for k in list(kwargs):
            kwargs[k] = _mv(kwargs[k])

        # Cross-device boundary: the autograd engine will later run this
        # layer's backward on `dev` and insert a copy back toward `src`.
        # ROCm corrupts when such copies race the producer on another device;
        # a full-fwd hook cannot fix backward, so we register a per-tensor
        # backward hook that SYNCHRONIZES the producing device before the
        # reciprocal copy executes (the copy node consumes this layer's
        # output; syncing `dev` before returning bounds the hazard).
        if src is not None and src != dev:
            def _bwd_sync(grad):
                torch.cuda.synchronize(dev)
                torch.cuda.synchronize(src)
                return grad
            try:
                out_handle = module.register_full_backward_hook(
                    lambda m, gi, go: tuple(_bwd_sync(g) if isinstance(g, torch.Tensor) else g for g in go)
                )
            except Exception:
                pass
        return args, kwargs

    return hook


def _materialize_meta(module: nn.Module):
    """Replace remaining meta tensors (non-persistent buffers like rotary
    inv_freq that state_dict loads never cover) with real CPU tensors."""
    for name, child in module.named_modules():
        for bname, b in list(child.named_buffers(recurse=False)):
            if b is not None and b.is_meta:
                new = torch.zeros(b.shape, dtype=b.dtype, device="cpu")
                if "inv_freq" in bname:
                    # rotary tables recompute on first forward anyway; but give
                    # them a sane arange-based default so .to() never sees meta
                    base = 10000.0
                    dim = max(1, b.shape[-1])
                    freqs = 1.0 / (base ** (torch.arange(0, dim, dtype=torch.float32) / dim))
                    new.copy_(freqs.to(b.dtype).expand(b.shape).reshape(b.shape))
                child._buffers[bname] = new


def _shard_model(model: nn.Module, layer_devices: dict[int, torch.device], device: torch.device):
    """Layer-wise pipeline sharding: decoder layers round-robin across devices.

    The text stack (``model.model``) holds ``layers``; each layer moves to its
    mapped device and gets a forward-pre-hook that hops activations onto that
    device (autograd inserts the reverse copy in backward).  Everything else
    (embeddings, final norm inside the text model; lm_head outside it) stays on
    the main device.
    """
    _materialize_meta(model)
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else getattr(model, "layers", None)
    if layers is None:
        print("  WARNING: no .layers container found; falling back to single-device move", flush=True)
        model.to(device)
        return
    for i, layer in enumerate(layers):
        target = layer_devices.get(i, device)
        layer.to(target)
        # Hop on every layer (no-op when tensors are already in place): a layer
        # on the main device still receives activations from the previous
        # shard's device.  The hook registers a backward sync at cross-device
        # boundaries (see _layer_device_hop_hook) to bound ROCm copy hazards.
        layer.register_forward_pre_hook(_layer_device_hop_hook(target), with_kwargs=True)
    holder = base if base is not None else model
    for name, child in holder.named_children():
        if name != "layers":
            child.to(device)
    # The lm_head consumes the final decoder layer's output; placing it on the
    # LAST layer's device avoids hauling the full hidden state back to the
    # main device and balances the main device's static load (which already
    # hosts the embedding + both heads' qbytes).  Only meaningful when the
    # model has an lm_head child (LMWithHead wrapper).
    head = getattr(model, "lm_head", None) if base is not None else None
    if isinstance(head, nn.Module):
        last_dev = layer_devices.get(len(layers) - 1, device)
        head.to(last_dev)
    if base is not None:
        head = getattr(model, "lm_head", None)
        last_dev = layer_devices.get(len(layers) - 1, device)
        embed = base.embed_tokens if base is not None else getattr(model, "embed_tokens", None)
        # Spread the two big static tensors: embeds live on the main device
        # (input tokens enter there), the head on the last layer's device.
        # With uniform 16/16/16/16 layers that balances every GPU's static
        # load while device 0 (activations + loss) stays the lightest.
        for name, child in model.named_children():
            if name == "model":
                continue
            if isinstance(child, type(head)) and child is head:
                child.to(last_dev)  # keep the head on the last layer's device
            else:
                child.to(device)
        if embed is not None and getattr(embed, "weight", None) is not None:
            if embed.weight.device != device:
                embed.to(device)
    # Final norm + head both live on the last layer's device; the norm needs
    # an explicit hop because the last decoder layer leaves hidden states on
    # that same device (no-op) but the norm's weight must sit with the head.
    final_norm = getattr(holder, "norm", None) if base is not None else getattr(model, "norm", None)
    if isinstance(final_norm, nn.Module):
        final_norm.to(last_dev if base is not None else device)
        final_norm.register_forward_pre_hook(
            _layer_device_hop_hook(last_dev if base is not None else device), with_kwargs=True
        )


def _module_device(module) -> torch.device:
    """Device of a module's primary tensor (Parameter, buffer, or MixedQuant)."""
    p = next(module.parameters(), None)
    if p is not None and not p.is_meta:
        return p.device
    b = next((b for b in module.buffers() if hasattr(b, "device")), None)
    if b is not None and not b.is_meta:
        return b.device
    return torch.device("cpu")


def _chunked_forward_loss(
    source_model, teacher, x, y, temperature: float,
    chunk: int, loss_device: str, teacher_device: str,
):
    """Token-chunked student/teacher forward + CE/KL loss.

    Runs each text stack once to get final hidden states, then computes lm_head
    logits and both losses per token chunk.  Peak memory is one chunk's
    [chunk, V] logits/softmax instead of the full [B*S, V] set.  Both losses
    are sums over tokens, so chunked sums divided by the token count are exact.
    """
    student_base, student_head = source_model.model, source_model.lm_head
    teacher_base, teacher_head = teacher.model, teacher.lm_head

    hidden = student_base(input_ids=x).last_hidden_state  # grad-tracking, last-shard device
    with torch.no_grad():
        # Route the teacher input to wherever its embedding actually lives
        # (main device in the sharded pipeline, teacher_device otherwise).
        emb_dev = getattr(getattr(teacher_base, "embed_tokens", None), "weight", None)
        t_dev = emb_dev.device if emb_dev is not None else torch.device(teacher_device)
        t_hidden = teacher_base(input_ids=x.to(t_dev) if t_dev != x.device else x).last_hidden_state

    H = hidden.shape[-1]
    s_flat = hidden.reshape(-1, H)
    t_flat = t_hidden.reshape(-1, H)
    y_flat = y.reshape(-1)
    n_tok = s_flat.shape[0]

    loss_dev = torch.device(loss_device)
    lm_total = 0.0
    kl_total = 0.0
    T = temperature
    # Per-chunk loss+backward with a DETACHED head input: autograd would
    # otherwise retain every chunk's [chunk, V] log-softmax until a single
    # final backward (~8 GiB at 8192 tokens) — OOM on any single GPU.  Each
    # chunk's backward is confined to the head subgraph (freed immediately);
    # its input-grad is accumulated into a hidden-grad tensor, and ONE final
    # model backward propagates that through the stack.  Exact: CE/KL are
    # token sums and gradients are linear.
    head_dev = _module_device(student_head)
    head_dev = head_dev if head_dev.type == "cuda" else loss_dev
    h_grad = torch.zeros_like(s_flat)
    n_chunks = (n_tok + chunk - 1) // chunk
    for ci, i in enumerate(range(0, n_tok, chunk)):
        if ci % 8 == 0:
            print(f"  [chunked-loss] chunk {ci}/{n_chunks}", flush=True)
        h_c = s_flat[i : i + chunk].detach().to(head_dev).requires_grad_(True)
        logits_c = student_head(h_c).to(loss_dev)
        y_c = y_flat[i : i + chunk].to(loss_dev)
        lm_c = F.cross_entropy(logits_c, y_c, reduction="sum")
        student_log_probs = F.log_softmax(logits_c / T, dim=-1)
        del logits_c
        with torch.no_grad():
            th_c = t_flat[i : i + chunk].to(_module_device(teacher_head))
            t_logits_c = teacher_head(th_c).to(loss_dev)
            teacher_probs = F.softmax(t_logits_c / T, dim=-1)
            del t_logits_c
        kl_c = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum() * (T * T)
        del student_log_probs, teacher_probs
        ((lm_c + kl_c) / n_tok).backward()
        with torch.no_grad():
            g = h_c.grad
            if g is None:
                g = torch.zeros_like(h_c)
            h_grad[i : i + chunk] = g.to(s_flat.device)
            torch.cuda.synchronize()  # fence chunk backward before the next
        lm_total += lm_c.item() / n_tok
        kl_total += kl_c.item() / n_tok
        del lm_c, kl_c, h_c
    # One aggregated backward through the model stack (reshape keeps hidden's
    # original [B, S, H] shape for the grad).  Single-threaded autograd +
    # stream-mismatch suppression: ROCm's multithreaded engine crossing the
    # per-device default streams produced HSA aperture violations here.
    torch.autograd.set_multithreading_enabled(False)
    print("  [chunked-loss] final model backward start", flush=True)
    hidden.backward(h_grad.reshape(hidden.shape))
    print("  [chunked-loss] final model backward done", flush=True)
    torch.autograd.set_multithreading_enabled(True)

    return lm_total, kl_total


def _tp_teacher_to_cpu(teacher) -> None:
    """Drop the Q8 teacher's GPU qbytes after its hidden states are extracted.

    With the GDN family selectable, the per-rank VRAM ledger at seq_len 8192
    is ~1-2 GB over the 32 GB card limit; the teacher stack (~3.6 GB/rank of
    qbytes) is the largest movable block and is inert between forwards.  The
    buffers are NOT copied to CPU (4 ranks x anon copies = host OOM); reload
    re-slices the file-backed mmap source (page cache, no anon cost).  The
    lm_head is NOT offloaded — the loss chunk loop uses it on-GPU afterwards.
    Cached full-dequant ``_w`` tensors are dropped too.
    """
    from voodoo_quant.parallel import _slice_q8_bytes

    # Spare teacher.lm_head: the loss chunk loop uses it on-GPU AFTER the
    # teacher body's hidden states are extracted.
    _head = getattr(teacher, "lm_head", None)
    # Restore non-Q8 state (norms/buffers) moved by _tp_teacher_to_device.
    _b = getattr(teacher, "model", None) or teacher
    for mod in _b.modules():
        for pname, par in list(getattr(mod, "_voodoo_param_home", {}).items()):
            setattr(mod, pname, par)
        mod._voodoo_param_home = {}
        for bname, buf in list(getattr(mod, "_voodoo_buf_home", {}).items()):
            setattr(mod, bname, buf)
        mod._voodoo_buf_home = {}
    seen = set()
    if _head is not None:
        seen.add(id(getattr(_head, "inner", _head)))
    for m in teacher.modules():
        inner = getattr(m, "inner", m)
        if id(inner) in seen:
            continue
        if hasattr(inner, "qweight") and isinstance(getattr(inner, "qweight", None), torch.Tensor):
            if inner.qweight.device.type == "cuda":
                inner.qweight = torch.empty(0, dtype=torch.uint8)
            inner._w = None
            seen.add(id(inner))
    torch.cuda.empty_cache()


def _tp_teacher_to_device(teacher, device) -> None:
    """Reload the Q8 teacher (layer stack AND embeddings) from the mmap source."""
    from voodoo_quant.parallel import _slice_q8_bytes
    _rel = [0]
    _names = []

    body = getattr(teacher, "model", None) or teacher
    seen = set()
    for m in teacher.modules():
        inner = getattr(m, "inner", m)
        shard = getattr(inner, "_voodoo_qb_shard", None)
        if shard is None or id(inner) in seen:
            continue
        seen.add(id(inner))
        src, full_out, full_in, kwargs = shard
        if kwargs.get("row_range") is None and kwargs.get("col_range") is None and kwargs.get("row_index") is None:
            qb_local = src.reshape(-1, 34) if src.dim() == 1 else src
        else:
            qb_local = _slice_q8_bytes(src, full_out, full_in, **kwargs)
        inner.qweight = qb_local.contiguous().to(device)
        _rel[0] += 1
        _names.append(type(m).__name__)
    # Non-Q8 teacher state (norms, A_log, dt_bias, conv1d buffers) lives on CPU
    # since the blanket .to(device) was removed — move it with the qbytes each
    # step, and return it to CPU on offload so VRAM stays free between steps.
    _b = getattr(teacher, "model", None) or teacher
    for mod in _b.modules():
        for pname, par in list(mod.named_parameters(recurse=False)):
            if par.device.type != device.type:
                mod._voodoo_param_home = getattr(mod, "_voodoo_param_home", {})
                mod._voodoo_param_home[pname] = par
                setattr(mod, pname, torch.nn.Parameter(par.detach().to(device), requires_grad=False))
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type != device.type and "qweight" not in bname:
                mod._voodoo_buf_home = getattr(mod, "_voodoo_buf_home", {})
                mod._voodoo_buf_home[bname] = buf
                setattr(mod, bname, buf.to(device))
    print(f"  [teacher-reload] reloaded {_rel[0]} Q8 modules; hosts: {sorted(set(_names))}", flush=True)


def _tp_vocab_parallel_losses(
    source_model, teacher, x, y, temperature: float, chunk: int, distill_weight: float = 1.0,
):
    """TP loss on per-rank vocab-sharded logits (exact, cross-rank logsumexp).

    Each rank's lm_head / teacher head produces logits only for its vocab shard.
    CE needs the full-vocab logsumexp; it is computed exactly from per-rank
    maxima and a DIFFERENTIABLE all-reduce SUM of the shifted exponentials
    (``tp_utils._AllReduceSum``: all-reduce forward, identity backward — the
    Megatron pattern).  Each rank differentiates only its own loss, so identity
    backward is exact; the trainer then all-reduces the gate gradients.

    KL(D_T ‖ D_S) = Σ_v p_T(v) (log p_T(v) − log p_S(v)) splits over vocab
    shards: teacher and student are sharded identically, so each rank sums its
    own shard's contribution and the cross-rank total is the full KL.  Both
    normalizers (lse) are the exact global ones.

    Returns per-mean losses like ``_chunked_forward_loss``: each term is
    already divided by the token count.
    """
    from voodoo_quant import parallel as tp_utils
    from voodoo_quant.parallel import TP as tp

    hidden = source_model.model(input_ids=x).last_hidden_state  # grad-tracking
    with torch.no_grad():
        # The teacher's weights are offloaded to CPU between steps (see
        # _tp_teacher_to_cpu); bring this rank's shard back for the forward.
        _tp_teacher_to_device(teacher, x.device)
        t_emb = getattr(getattr(teacher, "model", None), "embed_tokens", None)
        t_inner = getattr(t_emb, "inner", None)
        t_dev = (
            t_inner.qweight.device
            if getattr(t_inner, "qweight", None) is not None
            else torch.device(tp.device or "cpu")
        )
        t_hidden = teacher.model(input_ids=x.to(t_dev) if t_dev != x.device else x).last_hidden_state
        t_hidden = t_hidden.to(x.device, non_blocking=False)
        # Free the teacher shard immediately — only t_hidden (small) is needed
        # from here on, and the student backward needs the VRAM.
        _tp_teacher_to_cpu(teacher)

    H = hidden.shape[-1]
    s_flat = hidden.reshape(-1, H)
    t_flat = t_hidden.reshape(-1, H)
    y_flat = y.reshape(-1)
    n_tok = s_flat.shape[0]
    T = temperature

    s_head = source_model.lm_head
    print(f"  [loss-debug] t_head={type(teacher.lm_head).__name__} s_head={type(s_head).__name__}", flush=True)
    t_head = teacher.lm_head
    s_v0 = int(getattr(s_head, "tp_vocab_start", 0))
    t_v0 = int(getattr(t_head, "tp_vocab_start", 0))
    if tp.enabled and (s_v0 != t_v0 or s_head.out_features != t_head.out_features):
        raise RuntimeError(
            f"student/teacher vocab shards differ: student [{s_v0},+{s_head.out_features}) "
            f"teacher [{t_v0},+{t_head.out_features})"
        )

    def _global_lse(logits: torch.Tensor, differentiable: bool):
        """Exact full-vocab logsumexp from per-rank shards.

        Returns (lse, is_max_owner_max) — the max shift is a constant (zero
        gradient a.e., like any LSE stabilizer); the exponential sum is
        differentiable when requested via the all-reduce autograd function.
        """
        m_local = logits.detach().amax(dim=-1)
        if tp.enabled:
            m_all = m_local.clone().contiguous()
            torch.distributed.all_reduce(m_all, op=torch.distributed.ReduceOp.MAX)
            m_global = m_all
        else:
            m_global = m_local
        e_local = (logits - m_global.unsqueeze(-1)).exp().sum(-1)
        if differentiable and tp.enabled:
            e_sum = tp_utils.tp_all_reduce_sum(e_local)
        elif tp.enabled:
            e_sum = e_local.clone().contiguous()
            torch.distributed.all_reduce(e_sum, op=torch.distributed.ReduceOp.SUM)
        else:
            e_sum = e_local
        return m_global + e_sum.log()

    h_grad = torch.zeros_like(s_flat)
    lm_total = 0.0
    kl_total = 0.0
    # Value partials weight the shared lse term by 1/N so the cross-rank SUM of
    # partials equals the exact global CE (see the all-reduce at the end); the
    # BACKWARD keeps the full lse on every rank because each rank only
    # differentiates its own shard (softmax_v - [v==y in shard]) — exact.
    inv_world = 1.0 / (tp.world_size if tp.enabled else 1)
    n_chunks = (n_tok + chunk - 1) // chunk
    for ci, i in enumerate(range(0, n_tok, chunk)):
        if tp.rank == 0 and ci % 8 == 0:
            print(f"  [tp-loss] chunk {ci}/{n_chunks}", flush=True)
        h_c = s_flat[i : i + chunk].detach().requires_grad_(True)
        logits_c = s_head(h_c).float()  # [chunk, local vocab]
        lse = _global_lse(logits_c, differentiable=True)

        # CE (sum over tokens): only the shard owning the target id contributes
        # the -logit term; every rank contributes the denominator term via lse.
        y_local = y_flat[i : i + chunk] - s_v0
        in_shard = (y_local >= 0) & (y_local < logits_c.shape[-1])
        y_safe = y_local.clamp(0, logits_c.shape[-1] - 1)
        picked = logits_c.gather(1, y_safe.unsqueeze(1)).squeeze(1)
        ce_sum_c = (lse - torch.where(in_shard, picked, torch.zeros_like(picked))).sum()

        with torch.no_grad():
            tq = getattr(t_head, "qweight", None)
            if isinstance(tq, torch.Tensor):
                th_dev = tq.device
            else:
                th_dev = next(t_head.parameters()).device
            th_c = t_flat[i : i + chunk].to(th_dev)
            t_logits = t_head(th_c).float()
            t_lse = _global_lse(t_logits, differentiable=False)
            t_logp = t_logits - t_lse.unsqueeze(-1)
            t_prob = t_logp.exp()

        # Student log-probs stay differentiable (only the teacher side is
        # frozen): log p_S(v) = logits_v - lse for v in this rank's shard.
        s_logp = logits_c - lse.unsqueeze(-1)
        # Per-rank partial KL over THIS shard's vocab range (teacher and
        # student ranges are identical — asserted above).
        kl_c = (t_prob * (t_logp - s_logp)).sum() * (T * T)

        ((ce_sum_c + distill_weight * kl_c) / n_tok).backward()
        with torch.no_grad():
            g = h_c.grad if h_c.grad is not None else torch.zeros_like(h_c)
            h_grad[i : i + chunk] = g
        # value partial: exact after the cross-rank sum (lse counted 1/N per rank)
        ce_val_c = (lse * inv_world - torch.where(in_shard, picked, torch.zeros_like(picked))).sum()
        lm_total += ce_val_c.item() / n_tok
        kl_total += kl_c.item() / n_tok
        del logits_c, h_c

    # One aggregated backward through the model stack (same single-threaded
    # autograd + stream-mismatch suppression rationale as the chunked path).
    torch.autograd.set_multithreading_enabled(False)
    print("  [tp-loss] final model backward start", flush=True)
    hidden.backward(h_grad.reshape(hidden.shape))
    print("  [tp-loss] final model backward done", flush=True)
    torch.autograd.set_multithreading_enabled(True)
    # The per-rank CE/KL values above are PARTIALS (each rank owns only its
    # vocab shard's `picked`/KL terms); the backward through each rank's own
    # lse/picked/KL is exact, but for logging/return we need the exact GLOBAL
    # means: sum the partials across ranks once (values only, no gradients).
    if tp.enabled:
        part = torch.tensor([lm_total, kl_total], device=hidden.device)
        torch.distributed.all_reduce(part, op=torch.distributed.ReduceOp.SUM)
        lm_total, kl_total = part[0].item(), part[1].item()
    return lm_total, kl_total


def apply_partial_gates(replaced, payload: dict) -> int:
    """Load gates from a partial payload into the current replaced layers.

    Self-correcting: layers whose name or candidate-count does not match are
    skipped with a warning rather than crashing, so a partial from a slightly
    different config still recovers whatever it can.  Returns the number of
    layers restored.
    """
    gates = payload.get("gates", {})
    restored = 0
    skipped: list[tuple[str, str]] = []
    for name, layer in replaced.items():
        if name not in gates:
            skipped.append((name, "missing in partial"))
            continue
        g = gates[name]
        if g.numel() != layer.gates.numel():
            skipped.append((name, f"candidate mismatch {g.numel()} vs {layer.gates.numel()}"))
            continue
        with torch.no_grad():
            layer.gates.copy_(g.to(layer.gates.device, dtype=layer.gates.dtype))
        restored += 1
    for name, reason in skipped[:10]:
        print(f"  [partial] skip {name}: {reason}", flush=True)
    if len(skipped) > 10:
        print(f"  [partial] ... and {len(skipped) - 10} more skipped", flush=True)
    print(f"  [partial] restored gates for {restored}/{len(replaced)} layers", flush=True)
    return restored


def _free_replaced_residue(replaced) -> None:
    """Drop candidate buffers held by MixedQuant modules after their gates and
    assignments have been harvested.

    The finalizer only needs ``gates``, ``candidate_types`` and ``imatrix`` —
    all captured before this runs — while the leftover ``_qweight_``/``_w_``
    candidate buffers, forward caches and glibc arenas can hold tens of GB
    that the bake's accumulating state dict needs on the host.  Safe on
    ``_ReplacedShim`` objects (they simply lack the attributes).
    """
    for layer in replaced.values():
        for attr in list(vars(layer)):
            if attr.startswith(("_qweight_", "_w_")):
                setattr(layer, attr, None)
        for attr in ("_fwd_candidates", "_dequant_cache"):
            buf = getattr(layer, attr, None)
            if buf is not None:
                setattr(layer, attr, [None] * len(buf))
        if getattr(layer, "_dequant_cache_valid", False):
            layer._dequant_cache_valid = False
    gc.collect()
    try:
        import ctypes as _ct

        _ct.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _finalize_and_save(
    args,
    replaced,
    source_model,
    original_sd,
    original_sd_cleanup,
    non_targeted_bytes: float,
    target_bytes: float,
    compression_ratio: float,
    dtype: torch.dtype,
    tie_word_embeddings: bool,
    targeted_keys: set[str],
    device: torch.device,
) -> Path:
    """Build hard assignments, quantize permanently, and save the checkpoint.

    Self-correcting: each original weight is resolved via `_resolve_weight` so a
    missing key is skipped (and recorded) instead of crashing the run.  The
    learned assignments and a final partial are persisted BEFORE the risky
    quantization loop, and the whole body is wrapped so that even an unexpected
    error leaves the assignments + partial on disk for `--finalize_from_partial`.
    """
    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path = output_path.with_suffix(".quant_assignments.json")

    # 1) Hard assignments from the learned gates (cheap, cannot KeyError).
    quant_assignments: dict[str, str] = {}
    assignment_probs: dict[str, float] = {}
    imatrices: dict[str, torch.Tensor] = {}
    for name, layer in replaced.items():
        quant_assignments[name] = layer.get_assignment()
        assignment_probs[name] = layer.get_assignment_prob()
        if layer.imatrix is not None:
            imatrices[name] = layer.imatrix.cpu()

    print("\nFinal assignments:")
    counts: dict[str, int] = {}
    for qt in quant_assignments.values():
        counts[qt] = counts.get(qt, 0) + 1
    for qt, count in sorted(counts.items()):
        print(f"  {qt}: {count} tensors")

    # 1b) Apply post-hoc tensor upgrades for critical layers.
    #     The optimizer may under-quantize attention/SSM layers because short
    #     sequences don't produce strong gradients for these tensors.  Upgrade
    #     definitions are a list of {pattern, levels} dicts.  Each matched
    #     tensor gets its quant type bumped up (or down, for negative levels)
    #     by `levels` steps in the quant ladder, preserving relative ordering
    #     within the matched group.
    if getattr(args, "tensor_upgrades", None):
        upgrades = json.loads(args.tensor_upgrades)
        if upgrades:
            quant_ladder = [
                "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
                "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
            ]
            qt_index = {qt: i for i, qt in enumerate(quant_ladder)}
            upgraded = 0
            for name, qt in list(quant_assignments.items()):
                for spec in upgrades:
                    if re.search(spec["pattern"], name):
                        levels = int(spec.get("levels", 1))
                        old_idx = qt_index.get(qt, 0)
                        new_idx = max(0, min(old_idx + levels, len(quant_ladder) - 1))
                        new_qt = quant_ladder[new_idx]
                        if new_qt != qt:
                            quant_assignments[name] = new_qt
                            upgraded += 1
                        break
            if upgraded:
                print(f"\nTensor upgrades applied ({upgraded} tensors):")
                counts2: dict[str, int] = {}
                for qt in quant_assignments.values():
                    counts2[qt] = counts2.get(qt, 0) + 1
                for qt, count in sorted(counts2.items()):
                    print(f"  {qt}: {count} tensors")

    # 1c) Apply hard force_quant overrides (e.g. pin the last down_proj to Q8_0).
    #      A forced type must be among that tensor's trained candidates, otherwise
    #      its bytes were never quantized/cached.
    if getattr(args, "force_quant", None):
        forced = json.loads(args.force_quant)
        n_forced = 0
        for name, qt in forced.items():
            layer = replaced.get(name)
            if layer is None:
                print(f"  force_quant WARNING: {name} is not a replaced tensor; skipping", flush=True)
                continue
            if qt not in layer.candidate_types:
                # The bake re-quantizes from the ORIGINAL weights, so a forced
                # type does not need trained candidates (only the training-time
                # forward mix did).  Validate it is a real quant type, then pin.
                from voodoo_quant.ggml import get_quant_info

                try:
                    get_quant_info(qt)
                except Exception:
                    print(f"  force_quant WARNING: unknown quant type {qt} for {name}; skipping", flush=True)
                    continue
                print(f"  force_quant note: {qt} not among trained candidates of {name} "
                      f"({layer.candidate_types}); pinning anyway (bake quantizes from originals)",
                      flush=True)
            if quant_assignments.get(name) != qt:
                quant_assignments[name] = qt
                n_forced += 1
        if n_forced:
            print(f"force_quant: overrode {n_forced} tensor assignment(s)")

    # 1d) Budget-aware knapsack polish: one-rung moves per tensor ranked by
    #     measured benefit-per-byte (sensitivity table), greedily applied
    #     inside the size dead zone. The same injection point as
    #     --tensor_upgrades, but per-tensor, measured, and budget-aware —
    #     it spends slack on ranked upgrades and swaps worthless high-precision
    #     rungs down. Table reuse: warm-start's table when present; otherwise
    #     built here from the candidate cache (weight-space sensitivity —
    #     activations are gone by finalize time in the recovery path).
    if getattr(args, "polish", False):
        from voodoo_quant.training.sensitivity import knapsack_polish

        table = getattr(args, "_sens_table", None)
        if table is None:
            # Use the MEASURED table persisted by --warm_start (real
            # activations); polish only trusts measured prices.
            import pickle

            _tbl_path = Path(args.output_dir) / "sensitivity.pkl"
            if _tbl_path.exists():
                with open(_tbl_path, "rb") as f:
                    table = pickle.load(f)
                print(f"\nPolish: loaded measured sensitivity table from {_tbl_path}", flush=True)
        if table is None:
            print("\nPolish: SKIPPED — no measured sensitivity table found. Re-run training "
                  "with --warm_start; the measured table persists to sensitivity.pkl.",
                  flush=True)
        else:
            polished, moves = knapsack_polish(
                quant_assignments, table,
                budget_bytes=float(target_bytes),
                non_targeted_bytes=non_targeted_bytes,
                tolerance=float(getattr(args, "size_tolerance", 0.02)),
            )
            n_changed = sum(1 for t in polished if polished[t] != quant_assignments.get(t))
            quant_assignments = polished
            print(f"Knapsack polish: {len(moves)} moves applied, {n_changed} tensors changed", flush=True)
            ups = sum(1 for m in moves if m["dir"] == "up")
            downs = len(moves) - ups
            print(f"  ({ups} upgrades / {downs} downgrades)", flush=True)
            counts_p: dict[str, int] = {}
            for qt in quant_assignments.values():
                counts_p[qt] = counts_p.get(qt, 0) + 1
            for qt, count in sorted(counts_p.items()):
                print(f"  {qt}: {count} tensors")

    # 2) Persist assignments + a final partial FIRST so they survive any later
    #    failure.  This is the journal a recovery run can finalize from.
    try:
        assignments_path.write_text(json.dumps(quant_assignments, indent=2, sort_keys=True))
        print(f"Saved quant assignments to {assignments_path}", flush=True)
    except Exception as exc:
        print(f"  WARNING: could not write assignments sidecar: {exc}", flush=True)
    save_partial(args, replaced, completed_opt_steps=int(getattr(args, "max_steps", 0)), tau=0.0, reason="pre_finalize")

    # Free the MixedQuant candidate residue before the bake: the bake's state
    # dict grows to tens of GB of host anon, and the leftover candidate
    # buffers plus glibc arenas on top of it OOM-kill the host at save time.
    # Nothing below reads `replaced` again.
    _free_replaced_residue(replaced)

    # Free as much GPU memory as possible before the CPU-only final quantization.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    skipped_keys: list[tuple[str, str]] = []
    try:
        # 3) Apply the learned assignments permanently using the authoritative
        #    original-weight source (never source_model.state_dict()).
        print("\nApplying learned quant assignments permanently ...", flush=True)
        final_state_dict: dict[str, torch.Tensor] = {}

        # Parallel finalization: quantize+dequantize each tensor in a thread pool.
        # Each tensor is independent, and ggml's quantize_tensor/dequantize_tensor
        # release the GIL during the C calls, so threads give real speedup.
        import concurrent.futures
        # TP mode: rank 0 bakes while 3 ranks may still hold teardown residue;
        # 8 workers x fp32 tensor copies = ~22 GB anon (OOM-killed once).
        # 2 workers keeps the bake under the host ceiling.
        _tp_default = 2 if (getattr(args, "tensor_parallel", None) or 1) > 1 else 8
        finalize_workers = int(os.environ.get("VOODOO_FINALIZE_WORKERS",
                                               min(_tp_default, os.cpu_count() or 2)))
        old_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"

        def _finalize_one(item):
            idx, name, qt = item
            weight_key = name + ".weight"
            w, where = _resolve_weight(original_sd, weight_key)
            if w is None:
                return idx, weight_key, None, where
            torch.set_num_threads(1)
            w = w.detach().clone().cpu().to(torch.float32)
            imatrix = imatrices.get(name)
            padded_weight, padded_imatrix, orig_in = _pad_weight_and_imatrix(w, imatrix)
            qbytes = quantize_tensor(padded_weight, qt, padded_imatrix)
            w_deq = dequantize_tensor(qbytes, qt, w.shape[0], padded_weight.shape[1])
            w_deq = w_deq[:, :orig_in].to(dtype)
            return idx, weight_key, w_deq.cpu(), None

        # Bake biggest-first: the largest tensor (the embedding, ~5 GB as an
        # fp32 transient) is then quantized while the accumulating state dict
        # is still small instead of on top of its final size.
        def _item_numel(entry):
            _, (_n, _q) = entry
            _w, _ = _resolve_weight(original_sd, _n + ".weight")
            return 0 if _w is None else int(_w.numel())

        items = sorted(enumerate(quant_assignments.items()), key=_item_numel, reverse=True)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=finalize_workers) as pool:
            futures = {pool.submit(_finalize_one, (i, n, q)): i for i, (n, q) in items}
            for fut in concurrent.futures.as_completed(futures):
                idx, weight_key, result, err = fut.result()
                if result is None:
                    skipped_keys.append((weight_key, err))
                    print(f"  WARNING: skipping {weight_key}: {err}", flush=True)
                else:
                    final_state_dict[weight_key] = result
                completed += 1
                if completed == 1 or completed % 20 == 0 or completed == len(items):
                    print(f"  quantized {completed}/{len(items)} tensors ...", flush=True)

        if old_omp is not None:
            os.environ["OMP_NUM_THREADS"] = old_omp

        # With tied embeddings the output head shares the embedding weight, so keep
        # the saved lm_head copy identical to the quantized embedding.
        if tie_word_embeddings and "model.embed_tokens.weight" in final_state_dict:
            final_state_dict["lm_head.weight"] = final_state_dict["model.embed_tokens.weight"]

        # Carry over every remaining key from the authoritative original source so
        # the saved checkpoint is a complete state dict even when some targeted
        # keys were skipped.  `original_sd` is complete (base mmap or merged
        # snapshot), so this cannot KeyError.
        for k in original_sd.keys() if hasattr(original_sd, "keys") else list(original_sd):
            if k not in final_state_dict:
                # Keep the source views (often mmap-backed) instead of cloning:
                # cloning the non-targeted tensors (norms + SSM at F32, ~22 GB
                # for 27B) on top of the targeted dict would OOM the host;
                # torch.save serializes views straight from the page cache.
                final_state_dict[k] = original_sd[k]

        if skipped_keys:
            print(
                f"\nWARNING: {len(skipped_keys)} targeted tensor(s) could not be resolved "
                f"and were carried over unquantized: {[k for k, _ in skipped_keys][:10]}",
                flush=True,
            )

        final_selectable_bytes = sum(
            final_state_dict[name + ".weight"].numel() * bytes_per_weight(qt)
            for name, qt in quant_assignments.items()
            if (name + ".weight") in final_state_dict
        )
        final_total_bytes = final_selectable_bytes + non_targeted_bytes
        print(f"Final selectable weight bytes: {final_selectable_bytes / 1e6:.2f} MB")
        print(f"Final total model bytes: {final_total_bytes / 1e6:.2f} MB")
        print(f"Target total bytes: {target_bytes / 1e6:.2f} MB")

        print(f"Saving checkpoint to {output_path} ...", flush=True)
        target_bytes_scalar = target_bytes.item() if isinstance(target_bytes, torch.Tensor) else target_bytes
        torch.save({
            "model_state_dict": final_state_dict,
            # Keep top-level fields for backward compatibility with AGENTS.md docs.
            "quant_assignments": quant_assignments,
            "assignment_probs": assignment_probs,
            "effective_total_bytes": final_total_bytes,
            "effective_selectable_bytes": final_selectable_bytes,
            "target_bytes": target_bytes_scalar,
            "compression_ratio": compression_ratio,
            "target_bits": compression_ratio * 8.0,
            "source_dtype_info": {"dtype": args.dtype, "bits_per_element": bits_per_element(dtype)},
            "skipped_keys": skipped_keys,
            # Also store inside `extra` so downstream scripts that follow the AuxDecQ
            # checkpoint convention (e.g. eval_auxdecq.py) can find them without a sidecar.
            "extra": {
                "quant_assignments": quant_assignments,
                "assignment_probs": assignment_probs,
                "effective_total_bytes": final_total_bytes,
                "effective_selectable_bytes": final_selectable_bytes,
                "target_bytes": target_bytes_scalar,
                "compression_ratio": compression_ratio,
                "target_bits": compression_ratio * 8.0,
                "source_dtype_info": {"dtype": args.dtype, "bits_per_element": bits_per_element(dtype)},
                "skipped_keys": skipped_keys,
            },
        }, output_path)
        print(f"Saved dynamic quant base checkpoint to {output_path}", flush=True)
        print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB", flush=True)
        print(f"  Effective selectable bytes: {final_selectable_bytes / 1e6:.2f} MB", flush=True)
        print(f"  Effective total bytes: {final_total_bytes / 1e6:.2f} MB (target {target_bytes / 1e6:.2f} MB)", flush=True)
        return output_path
    except Exception as exc:
        # The assignments + final partial were already persisted above, so a
        # recovery run can finalize without retraining.  Surface the error but
        # leave the journal intact.
        print(
            f"\nERROR during finalization: {type(exc).__name__}: {exc}\n"
            f"  Assignments and partial checkpoint are preserved at {output_path.parent}.\n"
            f"  Recover with: --finalize_from_partial {output_path.parent / 'partial.pt'}",
            flush=True,
        )
        raise
    finally:
        if original_sd_cleanup is not None:
            original_sd_cleanup()


def build_parser():
    parser = argparse.ArgumentParser(description="Dynamic per-tensor quant selection")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--size_weight", type=float, default=10.0)
    parser.add_argument("--size_tolerance", type=float, default=0.02,
                        help="Dead-zone tolerance as a fraction of the target size (e.g., 0.02 = 2%%).")
    parser.add_argument("--compression_ratio", type=float, default=None,
                        help="Target size as a fraction of the original 8-bit model size.")
    parser.add_argument("--target_bits", type=float, default=None,
                        help="Virtual per-weight bit budget relative to an 8-bit original (e.g., 3.6 bits/weight = 45 percent of 8-bit).")
    parser.add_argument("--candidate_types", nargs="+", default=None,
                        help="Candidate quant types. Defaults to all supported types.")
    parser.add_argument("--target_layers", nargs="+", default=None,
                        help="Local names of Linear layers to make selectable. Default: all divisible Linear layers.")
    parser.add_argument("--skip_layers", nargs="+", default=None,
                        help="Full submodule names to skip entirely.")
    parser.add_argument("--imatrix", default=None,
                        help="Path to a llama.cpp imatrix GGUF for per-tensor importance weights.")
    parser.add_argument("--temp_start", type=float, default=1.0)
    parser.add_argument("--temp_end", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_device", default=None,
                        help="Device for the frozen teacher (default: same as --device). Use 'cpu' to keep VRAM under the cap for large models; teacher is a no-grad reference so immaterial CPU/GPU bf16 differences do not affect gate learning.")
    parser.add_argument("--loss_device", default=None,
                        help="Device for the CE/KL loss (default: same as --device). Use 'cpu' to move the large full-vocab softmax tensors off the GPU so large models stay under the VRAM cap; non-detaching CPU copies backprop to the GPU graph.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output_dir", default="./checkpoints/Qwen3.5-0.8B/Voodoo45")
    parser.add_argument("--output_name", default="voodoo_base.pt")
    parser.add_argument("--data_dir", default="data/calib")
    parser.add_argument("--data_name", default="train_tokens.pt")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--log_file", default="./logs/voodoo_train.jsonl")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--lazy", action="store_true",
                        help="Dequantize candidates on-the-fly instead of pre-dequantizing.  Saves memory.")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps before optimizer step.")
    parser.add_argument("--base_checkpoint", default=None,
                        help="Optional HF-compatible .pt state dict to use as the source/teacher instead of downloading --model weights. "
                             "Useful for MTP variants or Unsloth-converted bases where the --model identifier only provides the config.")
    parser.add_argument("--partial_save_interval", type=int, default=10,
                        help="Save a single rolling journaled partial checkpoint (gates + current argmax assignments) every N optimizer steps. "
                             "Overwrites partial.pt each time so only one exists. Set 0 to disable.")
    parser.add_argument("--resume_from_partial", default=None,
                        help="Path to a partial.pt whose gates should be loaded before training continues (optimizer/scheduler restart fresh).")
    parser.add_argument("--finalize_from_partial", default=None,
                        help="Skip training entirely: load gates from a partial.pt and run the permanent-quantization bake + save. "
                             "Recovery entrypoint for runs that crashed during finalization.")
    parser.add_argument("--candidate_cache_interval", type=int, default=5,
                        help="Re-dequantize candidates every N optimizer steps. Between re-dequantizations, "
                             "cached candidates are reused (gradients are still exact — only dequant is skipped). "
                             "Set to 1 to disable caching (re-dequant every step). Default: 5.")
    parser.add_argument("--tensor_upgrades", default=None,
                        help="JSON list of upgrade definitions applied post-training, pre-finalization. "
                             "Each definition: {pattern: regex, levels: N}. Matched tensors get "
                             "their quant type moved N steps up (N>0) or down (N<0) the quant ladder.")
    parser.add_argument("--st_gumbel_fraction", type=float, default=0.0,
                        help="Straight-through Gumbel hardening: per-step probability each layer "
                             "forwards a one-hot sampled candidate instead of the soft mixture "
                             "(backward stays soft/exact). 0 disables. Typical 0.5.")
    parser.add_argument("--st_gumbel_tau", type=float, default=1.0,
                        help="Gumbel sampling temperature for --st_gumbel_fraction (lower = more "
                             "argmax-like sampling). Ignored when fraction is 0.")
    parser.add_argument("--st_gumbel_anneal", default=None,
                        help="Anneal hardening fraction across training: 'START:END' (e.g. 0.2:0.8). "
                             "Overrides the fixed --st_gumbel_fraction schedule; fraction still enables ST.")
    parser.add_argument("--ptqr", action="store_true",
                        help="Per-Token Quant Routing: every replaced layer runs each token through "
                             "ONE candidate (Gumbel-sampled from the gate probs per token) instead of "
                             "averaging candidates. Removes the within-token mixture (Jensen) gain and "
                             "the hard/soft chimera; backward stays the exact soft mixture gradient. "
                             "Incompatible with --st_gumbel_fraction.")
    parser.add_argument("--budget_reduction", type=float, default=0.0,
                        help="Reduce target size budget by this fraction (0.05 = 5% smaller target). "
                             "Gives headroom for tensor upgrades without exceeding target size.")
    parser.add_argument("--warm_start", action="store_true",
                        help="Seed the gates from a MEASURED layout instead of zeros: one hooked forward pass "
                             "captures per-tensor input activations, s(t,q)=||(W_q-W)X||^2/||WX||^2 is computed "
                             "per candidate (the candidate cache already holds W_q), a greedy knapsack over "
                             "quality-per-byte produces the initial assignment, and gate logits start "
                             "concentrated on it. Fixes the exploration failure where logit-KL mis-attributes "
                             "deep-stack MLP precision (see WHITEPAPER.md §6).")
    parser.add_argument("--warm_start_batches", type=int, default=4,
                        help="Calibration batches used for the warm-start activation capture.")
    parser.add_argument("--warm_start_tokens", type=int, default=8192,
                        help="Max total tokens captured per tensor for the warm-start sensitivity.")
    parser.add_argument("--polish", action="store_true",
                        help="After the argmax freezes (post --tensor_upgrades), run a budget-aware one-rung "
                             "knapsack polish: moves ranked by measured benefit-per-byte from the sensitivity "
                             "table, greedily applied inside the size dead zone. Requires the measured table "
                             "from --warm_start (persists to sensitivity.pkl).")
    parser.add_argument("--attn_implementation", default=None,
                        choices=["eager", "sdpa", "flex_attention", "rocm_triton"],
                        help="Attention implementation for the student/teacher. rocm_triton uses the "
                        "curvedinf/flash-attention Triton-AMD backend (requires triton 3.2.x, FLASH_ATTENTION_TRITON_AMD_ENABLE). "
                        "flex_attention is the torch-native fallback. Note: this torch build has no flash/efficient "
                        "SDPA kernel for gfx908, so sdpa silently falls back to the O(L^2) math path.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on the student (required for long context).")
    parser.add_argument("--device_map", default=None,
                        help="Comma-separated CUDA devices for layer-wise sharding, e.g. 'cuda:0,cuda:1,cuda:2,cuda:3'. "
                        "Layers are distributed round-robin; the teacher uses the same mapping.")
    parser.add_argument("--tp_attention", type=int, default=0,
                        help="Tensor-parallel sharding of the full-attention layers via common/tp_attention.py "
                             "(24q/4kv heads -> 6q/1kv per rank at tp=4). Under torchrun (world_size>1) each rank keeps "
                             "its head group, o_proj partials/input grads are all-reduced and MixedQuant gate grads are "
                             "SUM-synced; without a process group, N>1 runs single-process simulation mode (correctness "
                             "only, no memory split). Incompatible with --tensor_parallel (which shards attention "
                             "pre-replacement through tp_utils). 0 = off.")
    parser.add_argument("--teacher_quant", default=None, choices=[None, "Q8_0"],
                        help="Load the teacher from a Q8_0-quantized checkpoint (--teacher_checkpoint) instead of BF16.")
    parser.add_argument("--teacher_checkpoint", default=None,
                        help="Teacher checkpoint path for --teacher_quant Q8_0 (model_state_dict + quant_meta).")
    parser.add_argument("--logits_chunk", type=int, default=0,
                        help="Compute the lm_head logits in token chunks of this size to bound peak VRAM "
                        "(0 = single pass, recommended 512-2048 for very large vocabularies).")
    parser.add_argument("--attention_candidates", nargs="+", default=None,
                        help="Candidate quant types for self_attn.* tensors (overrides the global set for them).")
    parser.add_argument("--non_attention_candidates", nargs="+", default=None,
                        help="Candidate quant types for all non-attention tensors (overrides the global set for them).")
    parser.add_argument("--force_quant", default=None,
                        help="JSON dict of {tensor_name: quant_type} overriding the learned assignment at bake time, "
                        "e.g. '{\"layers.63.mlp.down_proj\": \"Q8_0\"}'. The type must be among that tensor's candidates.")
    parser.add_argument("--qbytes_device", default=None, choices=[None, "cuda"],
                        help="Lazy mode: keep quantized candidate bytes resident on the (shard) GPU instead of CPU, "
                        "so forward/backward dequant has no H2D copy per candidate.")
    parser.add_argument("--text_model_class", default=None,
                        help="Build the language model as Qwen3_5TextModel + separate lm_head instead of "
                        "AutoModelForCausalLM. Use for language-model-only base checkpoints with stripped "
                        "`layers.N.*` keys (e.g. Qwen3.8-27B); module paths then match the candidate-cache keys.")
    parser.add_argument("--tp_ssm", type=int, default=0,
                        help="Tensor-parallel sharding of the SSM (GatedDeltaNet/linear-attention) layers via "
                        "common/tp_ssm.py. Under torchrun (world_size>1) each rank keeps its value-head group and "
                        "out_proj partials are all-reduced (pass the torchrun degree). Without a process group, "
                        "N>1 runs single-process simulation mode (correctness only, no memory split). 0 = off.")
    parser.add_argument("--tensor_parallel", type=int, default=None,
                        help="Full tensor parallelism (torchrun entry, one rank per GPU): tp_ssm + tp_attention "
                        "adapters, column/row-sharded MLP linears, VocabParallelEmbedding for embed_tokens/lm_head, "
                        "sharded Q8 teacher, per-rank vocab-shard logits combined exactly via cross-rank log-sum-exp. "
                        "TP patching runs BEFORE MixedQuant replacement so each rank sees rank-local shapes while "
                        "candidate-cache keys stay FULL-tensor names. Launch: torchrun --nproc_per_node N "
                        "train_dynamic_quant.py --tensor_parallel N ... (requires --base_checkpoint).")
    return parser


def run(args):
    """Execute training given parsed arguments."""
    if args.compression_ratio is None and args.target_bits is None:
        raise ValueError("Either --compression_ratio or --target_bits must be specified.")
    if args.compression_ratio is not None and args.target_bits is not None:
        raise ValueError("Specify only one of --compression_ratio or --target_bits.")

    # Straight-through Gumbel hardening (see MixedQuantLinear.get_probs).
    if getattr(args, "st_gumbel_fraction", 0.0) > 0.0 and getattr(args, "ptqr", False):
        raise ValueError("--ptqr and --st_gumbel_fraction are mutually exclusive.")
    if getattr(args, "ptqr", False):
        from voodoo_quant import layers as _layers

        _layers.set_ptqr(True)
        print("  [ptqr] enabled: per-token Gumbel routing from gate probs", flush=True)
    if getattr(args, "st_gumbel_fraction", 0.0) > 0.0:
        from voodoo_quant import layers as _layers

        _layers.set_st_gumbel(True, args.st_gumbel_fraction, getattr(args, "st_gumbel_tau", 1.0))
        if getattr(args, "st_gumbel_anneal", None):
            _fs, _fe = args.st_gumbel_anneal.split(":")
            _layers.set_st_gumbel_anneal(float(_fs), float(_fe), args.max_steps)
            print(f"  [st-gumbel] anneal: {args.st_gumbel_anneal} over {args.max_steps} steps", flush=True)
        print(
            f"  [st-gumbel] enabled: fraction={args.st_gumbel_fraction} "
            f"tau={getattr(args, 'st_gumbel_tau', 1.0)}",
            flush=True,
        )

    # Full tensor parallelism (--tensor_parallel under torchrun).  Initialized
    # first so every later `cuda` device reference resolves to this rank's
    # LOCAL_RANK GPU.
    tp_world = int(getattr(args, "tensor_parallel", 0) or 0)
    tp_enabled = False
    if tp_world:
        from voodoo_quant import parallel as tp_utils

        if int(os.environ.get("WORLD_SIZE", "1")) != tp_world or "RANK" not in os.environ:
            raise RuntimeError(
                f"--tensor_parallel {tp_world} requires a torchrun launch with the same degree "
                f"(got WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}). Use: "
                f"torchrun --nproc_per_node {tp_world} train_dynamic_quant.py --tensor_parallel {tp_world} ..."
            )
        if args.device_map:
            raise ValueError("--tensor_parallel is incompatible with --device_map (layer-wise sharding).")
        if getattr(args, "tp_ssm", 0):
            raise ValueError("--tensor_parallel already shards the SSM; drop --tp_ssm.")
        if getattr(args, "tp_attention", 0):
            raise ValueError("--tensor_parallel already shards attention; drop --tp_attention.")
        if args.base_checkpoint is None:
            raise ValueError(
                "--tensor_parallel requires --base_checkpoint: candidate quantization and the "
                "final bake need the FULL (mmap'd) weights; rank-local slices alone are not enough."
            )
        tp_enabled = tp_utils.tp_init()
        if not tp_enabled:
            raise RuntimeError("tp_init() failed: torchrun env (RANK/WORLD_SIZE/LOCAL_RANK) not found.")
        # Identical RNG on every rank: the DataLoader shuffle (and any random
        # init fallback) must produce the same sequence, otherwise ranks train
        # on different batches and the collectives deadlock on shape.
        torch.manual_seed(0)

    if args.candidate_types is None:
        from voodoo_quant.ggml import list_quant_types
        # Ternary quants (TQ1_0/TQ2_0) are excluded from the default candidate set
        # because they underperform relative to IQ1/IQ2 in current Voodoo runs.
        args.candidate_types = [qt for qt in list_quant_types() if not qt.startswith("TQ")]
        print(f"Using candidate types: {args.candidate_types}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher_device_str = args.teacher_device if args.teacher_device is not None else str(device)
    loss_device_str = args.loss_device if args.loss_device is not None else str(device)
    if tp_enabled:
        from voodoo_quant import parallel as tp_utils

        # All collectives (row-parallel sums, gate-grad all-reduce, vocab LSE)
        # are NCCL: teacher, loss and student must live on this rank's GPU.
        device = tp_utils.TP.device
        if teacher_device_str not in (str(device), "cuda"):
            print(f"  [tp] forcing teacher device to {device} (was {teacher_device_str})", flush=True)
        if loss_device_str not in (str(device), "cuda"):
            print(f"  [tp] forcing loss device to {device} (was {loss_device_str})", flush=True)
        teacher_device_str = str(device)
        loss_device_str = str(device)
    print(f"Teacher device: {teacher_device_str} (main device: {device})", flush=True)
    print(f"Loss device: {loss_device_str}", flush=True)

    # Layer-wise sharding across GPUs (--device_map).  The student's decoder
    # layers are distributed round-robin across the listed devices; the teacher
    # and the student share the mapping so teacher/student traffic stays local.
    shard_devices: list[torch.device] = []
    layer_devices: dict[int, torch.device] = {}
    if args.device_map:
        shard_devices = [torch.device(d.strip()) for d in args.device_map.split(",") if d.strip()]
        if device not in shard_devices:
            shard_devices.insert(0, device)
        print(f"Layer-wise sharding across {len(shard_devices)} devices: {shard_devices}", flush=True)
    num_layers = None  # resolved after config load

    print(f"Loading tokenizer for {args.model} ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as _tok_err:  # network/tokenizer unavailable; data is pre-tokenized
        print(f"  tokenizer load failed ({_tok_err!r}); continuing with pre-tokenized data only", flush=True)
        tokenizer = None

    print(f"Loading source model {args.model} ...", flush=True)
    t0 = time.time()
    attn_impl = args.attn_implementation
    if attn_impl is not None:
        print(f"  attn_implementation={attn_impl}", flush=True)
        if attn_impl == "rocm_triton":
            _register_flash_attn_triton()
            # Override the wrapper with the TP-aware variant from
            # common/tp_attention.py (same name => replaces): fixes the
            # [B,S,H,D] return-layout bug and pins the launch device to the
            # query's (the Triton-AMD fork faults on non-current devices).
            from voodoo_quant.hardware.rocm_flash import register_rocm_triton_tp

            register_rocm_triton_tp()
            print("  registered curvedinf/flash-attention Triton-AMD backend (TP-aware)", flush=True)
    source_model = _build_lm_model(args, dtype, attn_impl)
    source_model.config.use_cache = False
    source_model.eval()

    # Resolve the decoder-layer count and build the layer->device mapping.
    cfg_for_layers = getattr(source_model, "config", None)
    num_layers = getattr(getattr(cfg_for_layers, "text_config", cfg_for_layers), "num_hidden_layers", None) \
        or getattr(cfg_for_layers, "num_hidden_layers", 0)
    if shard_devices:
        n = len(shard_devices)
        if n > 1 and num_layers >= n:
            per = num_layers // n
            assignment = []
            for d in range(n):
                assignment += [d] * per
            while len(assignment) < num_layers:
                assignment.append(n - 1)
            # The LAST device hosts the lm_head + loss + final norm; SSM layers
            # there need a ~7 GiB torch-fallback recompute transient (FLA
            # backward is unusable on gfx908), which its VRAM cannot hold.
            # Distribute: last device keeps only its full-attention layers;
            # device 0 (which also hosts the embeddings) gets 16; the middle
            # devices take 18 each.  Static fits + one-at-a-time SSM
            # recompute headroom on every shard.
            last = n - 1
            # start from uniform, then rebuild with the target plan
            fa_layers = [i for i in range(num_layers) if (i + 1) % 4 == 0]
            ssm_layers = [i for i in range(num_layers) if (i + 1) % 4 != 0]
            assignment = [None] * num_layers
            # full-attention layers: last 4 stay on dev3 (adjacent to head);
            # the other 12 round-robin over dev0/dev1/dev2 only (4 each).
            fa_on_last = fa_layers[-(len(fa_layers) // n):]
            fa_rest = [i for i in fa_layers if i not in fa_on_last]
            for idx, i in enumerate(fa_rest):
                assignment[i] = idx % (n - 1)  # remaining FA round-robin over 0..n-2
            # SSM layers: device 0 also hosts the embeddings, so it takes a
            # lighter share; every device keeps headroom for the one-at-a-time
            # torch-fallback SSM recompute transient in backward.
            n_ssm = len(ssm_layers)
            if n < 3:
                # Small test configs: uniform split over the devices.
                layer_devices = {i: shard_devices[i % n] for i in range(num_layers)}
                print(f"  layer split: uniform over {n} devices (small config)", flush=True)
            else:
                # Activation-aware budget: every layer costs ~84 MiB/layer of
                # checkpoint boundary + up to ~0.9 GiB MLP / ~1.6 GiB SSM
                # (T-chunked) transient on its device.  Devices 0 and last
                # also host embeddings / heads, so they take fewer layers.
                # 64 layers -> 12 SSM + 4 FA on dev0, 16 SSM + 4 FA on dev1/2,
                # 4 FA + 8 SSM + head on dev3 (head output feeds loss there).
                # cuda:2 was the OOM point (20 layers, final-backward MLP
                # recompute short ~2-3 GiB at seq 8192); shift 2 of its SSM
                # layers to cuda:3, which peaks at ~18 GiB and has 14 GiB
                # spare.
                ssm_target = {0: 12, 1: 14, 2: 14, 3: 8}
                scale = n_ssm / 48.0 if num_layers != 64 else 1.0
                if scale != 1.0:
                    base = {0: 12, 1: 16, 2: 16, 3: 4}
                    ssm_target = {d: max(0, round(v * scale)) for d, v in base.items()}
                    # fix rounding drift
                    drift = n_ssm - sum(ssm_target.values())
                    ssm_target[1] += drift
                it = iter(ssm_layers)
                for d, t in ssm_target.items():
                    for _ in range(t):
                        assignment[next(it)] = d
                for i in fa_on_last:
                    assignment[i] = last
                # FA round-robin earlier may have put some FA on `last`; also
                # move any excess FA off last is unnecessary — fa_on_last is
                # the last len//n FA layers.  Dev3 FA count stays small.
                layer_devices = {i: shard_devices[assignment[i]] for i in range(num_layers)}
                counts = {str(shard_devices[a]): assignment.count(a) for a in set(assignment)}
                print(f"  layer split: {counts} (activation-balanced)", flush=True)
        else:
            layer_devices = {i: shard_devices[i % n] for i in range(num_layers)}
    print(f"  source model loaded in {time.time()-t0:.1f}s ({num_layers} decoder layers)", flush=True)

    # Keep the original state dict for the output checkpoint and capture metadata
    # for the byte budget before we clone, so tied weights are counted once.
    # Capture state dict on CPU before moving to GPU to avoid GPU memory pressure.
    print("Capturing source state-dict metadata for byte budget ...", flush=True)
    source_state_dict = source_model.state_dict()
    state_dict_meta = {
        k: (v.untyped_storage().data_ptr(), v.numel())
        for k, v in source_state_dict.items()
    }
    del source_state_dict
    gc.collect()
    # We do NOT clone the state dict here.  The original weights are kept in
    # source_model (which stays on CPU until we move it to GPU below).  At
    # save time we use source_model.state_dict() directly, so we avoid the
    # ~50 GB duplicate copy that used to live in original_state_dict.
    print("  metadata captured, skipping clone to save RAM", flush=True)

    # Now move source model to its training device(s).
    if tp_enabled:
        # Full TP mode: shard every parallelizable tensor in place BEFORE the
        # MixedQuant replacement (so each rank's replacement sees rank-local
        # Linear shapes); state_dict_meta above already captured the FULL model
        # (the byte budget is global).  `tp_full_weight` plain-attribute
        # references keep the full mmap'd weights reachable on CPU for
        # candidate quantization under FULL tensor names.
        from voodoo_quant.parallel import tp_patch_model

        _materialize_meta(source_model)  # rotary inv_freq etc. may still be meta
        tp_patch_model(source_model)
        source_model.to(device)
    elif shard_devices:
        _shard_model(source_model, layer_devices, device)
    else:
        source_model.to(device)

    if args.gradient_checkpointing:
        target = source_model.model if isinstance(source_model, LMWithHead) else source_model
        target.gradient_checkpointing_enable()
        # CPU-offload checkpoint boundaries: at seq 8192 each layer's saved
        # boundary is ~84 MiB; 20 layers/device keeps ~1.7-2.7 GiB of them on
        # the shard during backward and OOMs the MLP recompute.  Wrapping the
        # checkpointing func to store hidden_states on CPU (rehydrated inside
        # the closure at recompute time) frees that with no math change.
        # In TP mode boundaries are 1/N (per-rank slices), so they stay on the
        # GPU: the CPU offload would serialize every layer on H2D copies that
        # all ranks must replay in lockstep.
        if not tp_enabled:
            import torch.utils.checkpoint as _ckpt_mod

            def _cpu_boundary_checkpoint(function, *args, **kwargs):
                dev = args[0].device if args and isinstance(args[0], torch.Tensor) else None
                cpu_args = tuple(
                    a.to("cpu") if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                    for a in args
                )

                def _reh(fn, cargs, kwargs, device):
                    def inner(*unused):
                        rg = tuple(
                            a.to(device) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                            for a in cargs
                        )
                        return fn(*rg, **kwargs)
                    return inner

                return _ckpt_mod.checkpoint(_reh(function, cpu_args, kwargs, dev), *cpu_args, use_reentrant=False)

            for m in target.modules():
                if hasattr(m, "_gradient_checkpointing_func"):
                    m._gradient_checkpointing_func = _cpu_boundary_checkpoint
            print("Gradient checkpointing enabled on student (CPU-offloaded boundaries)", flush=True)
        else:
            print("Gradient checkpointing enabled on student (GPU-resident boundaries, TP mode)", flush=True)

    # Frozen teacher.  It is not needed for the finalize-only bake path.
    teacher = None
    if args.finalize_from_partial:
        print("finalize_from_partial: skipping teacher load (bake-only path)", flush=True)
    elif args.teacher_quant == "Q8_0":
        # Q8_0 teacher: build from config, wrap Linears/Embeddings with one-shot
        # Q8_0->bf16 dequantized weights, sharing the student's layer sharding.
        from voodoo_quant.training.q8_teacher import wrap_q8_teacher

        if args.teacher_checkpoint is None:
            raise ValueError("--teacher_quant Q8_0 requires --teacher_checkpoint")
        # TP host-RAM stagger: building the teacher materializes large mmap
        # pages per rank; 4 ranks concurrently OOM-kill the 61 GB host.  One
        # rank builds at a time (the TP shard step after this is cheap).
        _tp_ws = 1
        _tp_rk = 0
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _tp_ws = _dist.get_world_size()
                _tp_rk = _dist.get_rank()
        except Exception:
            pass
        # Serialize the whole teacher build across ranks with a file lock
        # (NCCL small collectives hang on gfx908; no dist calls here).
        _tch_lock = None
        if _tp_ws > 1:
            import fcntl as _fcntl
            _tch_lock = open("/tmp/voodoo_tp_teacher.lock", "w")
            _fcntl.flock(_tch_lock, _fcntl.LOCK_EX)
        print(f"Loading Q8_0 teacher from {args.teacher_checkpoint} ...", flush=True)
        t0 = time.time()
        tch_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        if args.text_model_class:
            from transformers.models.qwen3_5 import Qwen3_5TextModel

            Qwen3_5TextModel._supports_flex_attn = True
            text_cfg = getattr(tch_cfg, "text_config", tch_cfg)
            text_cfg = _tiny_recipe_override(text_cfg)
            if attn_impl is not None:
                text_cfg._attn_implementation = attn_impl
            with torch.device("meta"):
                tch_text = Qwen3_5TextModel(text_cfg)
                teacher = LMWithHead(tch_text, getattr(text_cfg, "vocab_size", None))
        else:
            teacher = AutoModelForCausalLM.from_config(
                tch_cfg, trust_remote_code=True, attn_implementation=attn_impl
            )
        tch_ckpt = torch.load(args.teacher_checkpoint, weights_only=True, map_location="cpu", mmap=True)
        wrap_q8_teacher(
            teacher,
            tch_ckpt["model_state_dict"],
            tch_ckpt["quant_meta"],
            layer_devices=layer_devices or None,
        )
        # Keep the mmap'd teacher state dict alive: the per-step offload/reload
        # cycle re-slices these file-backed tensors (see _tp_teacher_to_device).
        global _TCH_SD
        _TCH_SD = tch_ckpt["model_state_dict"]
        gc.collect()
        if tp_enabled:
            # Shard the teacher identically to the student (same adapters), so
            # student/teacher vocab shards stay aligned for the TP loss.
            # tp_patch slices ~6.5 GiB of CPU copies per rank before the GPU
            # upload — keep it INSIDE the per-rank lock so the four ranks'
            # transients never coexist (4x = host OOM), and trim before
            # releasing the next rank.
            from voodoo_quant.parallel import tp_patch_model

            _materialize_meta(teacher)
            if _tch_lock is None and _tp_ws > 1:
                import fcntl as _fcntl
                _tch_lock = open("/tmp/voodoo_tp_teacher.lock", "w")
                _fcntl.flock(_tch_lock, _fcntl.LOCK_EX)
            tp_patch_model(teacher)
            # tp_patch already uploaded every layer's shard; a blanket .to()
            # here would re-touch all buffers at once and overflow the GPU on
            # TP-2 (larger per-rank shards).  The offload below drops them to
            # the mmap source anyway; per-step reload handles placement.
            # With the GDN family selectable the per-rank VRAM ledger (model
            # shards 13.5 GB + candidate qbytes ~7.5 GB + activations) only
            # fits if the teacher is NOT GPU-resident while the student's
            # candidates load.  Drop the teacher to its mmap source now; the
            # loss fn reloads it per step (see _tp_teacher_to_device).
            if tp_enabled:
                _tp_teacher_to_cpu(teacher)
            gc.collect()
        # Release the teacher-build lock: trim arenas back to the OS first
        # so the next rank starts from a clean host-RAM floor.
        if _tch_lock is not None:
            try:
                import ctypes as _ct
                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            import fcntl as _fcntl
            _fcntl.flock(_tch_lock, _fcntl.LOCK_UN)
            _tch_lock.close()
            _tch_lock = None
        if shard_devices:
            _shard_model(teacher, layer_devices, device)
        teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Q8_0 teacher loaded in {time.time()-t0:.1f}s", flush=True)
    else:
        # Load teacher directly to GPU to avoid CPU memory pressure.
        print(f"Loading teacher model ...", flush=True)
        t0 = time.time()
        if args.base_checkpoint is not None:
            # Build from config (no weight download) then overlay the converted base
            # weights from source_model, and place on the teacher device explicitly.
            tch_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            teacher = AutoModelForCausalLM.from_config(
                tch_cfg, trust_remote_code=True, attn_implementation=attn_impl
            ).to(dtype)
            if tp_enabled:
                # Patch BEFORE assigning the student's (already rank-sharded)
                # state dict so shapes line up.
                from voodoo_quant.parallel import tp_patch_model

                tp_patch_model(teacher)
            # Load state dict from source_model (already has checkpoint weights), then
            # place the teacher on teacher_device_str (assign=True can otherwise leave
            # params on the source device when source_model is on GPU).
            teacher.load_state_dict(source_model.state_dict(), strict=False, assign=True)
            teacher.to(teacher_device_str)
        else:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map=teacher_device_str,
                attn_implementation=attn_impl,
            )
            if tp_enabled:
                from voodoo_quant.parallel import tp_patch_model

                tp_patch_model(teacher)
                # (shards uploaded per-layer by tp_patch; no blanket .to())
                _tp_teacher_to_cpu(teacher)  # per-step mmap reload (see loss fn)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  teacher model loaded in {time.time()-t0:.1f}s", flush=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("  GPU cache cleared after loading models", flush=True)

    # Optionally load per-tensor imatrices.
    imatrix_dict = {}
    if args.imatrix is not None:
        from voodoo_quant.imatrix import load_llamacpp_imatrix
        imatrix_dict = load_llamacpp_imatrix(args.imatrix)

    # Finalize-only + TP: the bake is a rank-0-only operation.  Non-zero
    # ranks must NOT rebuild 258 MixedQuant modules (each holds candidate
    # buffers; 3 ranks rebuilding concurrently OOMs the host while rank 0
    # bakes).  They exit here — the merged gates on rank 0 come from the
    # shared partial.pt, not from these ranks' modules.
    _finalize_tp_bypass = (
        args.finalize_from_partial is not None
        and (getattr(args, "tensor_parallel", None) or 1) > 1
        and not _tp_is_rank0()
    )
    if _finalize_tp_bypass:
        print("[finalize-tp] non-rank0 exiting before MixedQuant rebuild", flush=True)
        import ctypes as _ct
        import gc as _gc
        _gc.collect()
        try:
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        import os as _os
        _os._exit(0)

    print("Replacing target Linear layers with MixedQuantLinear ...", flush=True)
    install_heap_dump_handler()  # kill -USR1 <pid> dumps a heap census to the log
    if tp_enabled:
        # Staggered rank start: the replacement phase holds ~12.4 GB/rank of
        # live host anon, and 4 concurrent phases (plus the candidate-cache
        # page stream) exceed the 61 GB host.  Rank r waits for rank r-1's
        # marker before replacing; the file-lock chain avoids collectives
        # (NCCL small collectives hang on gfx908 with P2P disabled).  Training
        # itself runs lockstep as before — only init is serialized.
        import fcntl as _fcntl

        _chain_dir = Path(args.output_dir)
        _chain_dir.mkdir(parents=True, exist_ok=True)
        from voodoo_quant.parallel import TP as _tp
        _tp_rank = _tp.rank
        for _prev in range(_tp_rank):
            _lk = open(_chain_dir / f".rank{_prev}_replaced.done", "a+")
            while True:
                _fcntl.flock(_lk, _fcntl.LOCK_SH)
                _lk.seek(0)
                _done = _lk.read().strip() == "done"
                _fcntl.flock(_lk, _fcntl.LOCK_UN)
                if _done:
                    break
                time.sleep(5.0)
            _lk.close()
            print(f"  [rank-chain] rank {_tp_rank}: rank {_prev} replacement complete", flush=True)
        _own = open(_chain_dir / f".rank{_tp_rank}_replaced.done", "a+")
    skip_names = set(args.skip_layers) if args.skip_layers else set()
    target_names = set(args.target_layers) if args.target_layers is not None else None

    # GDN tiny state / per-head gates stay F32 (recipe): norms, A_log, dt_bias,
    # conv1d are non-Linear so they are never selected, but in_proj_a/in_proj_b
    # (alpha/beta) ARE Linear and must be excluded by name.
    skip_names.update({"in_proj_a", "in_proj_b"})

    # With tied word embeddings the input embedding and the output lm_head share
    # the same storage.  We quantize the embedding once and skip the lm_head
    # Linear replacement so the byte budget is not double-counted.
    tie_word_embeddings = getattr(source_model.config, "tie_word_embeddings", False)
    if tie_word_embeddings:
        skip_names.add("lm_head")
        print("  tie_word_embeddings=True: skipping lm_head Linear replacement", flush=True)

    # For the no-base-checkpoint path, capture the original weights of every
    # layer we are about to replace.  After replacement the MixedQuant* modules
    # drop their `.weight`, so this snapshot is the bake's only source of the
    # originals.  With --base_checkpoint we mmap that instead and skip the clone.
    pre_replacement_snapshot = None
    if args.base_checkpoint is None:
        print("Snapshotting original weights of selectable layers (no base checkpoint) ...", flush=True)
        pre_replacement_snapshot = _snapshot_selectable_weights(source_model, skip_names)
        print(f"  snapshotted {len(pre_replacement_snapshot)} weights", flush=True)

    t0 = time.time()
    qbytes_device = args.qbytes_device if args.qbytes_device else None
    if (getattr(args, "tensor_parallel", None) or 1) > 1:
        # TP mode: candidate qbytes MUST live on the rank's GPU — leaving
        # them on CPU keeps ~7 GiB of host anon per rank (4 ranks = host OOM)
        # and makes every forward pay H2D.  Resolve to the local rank device.
        qbytes_device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    replaced_linear = replace_linear_with_mixed_quant(
        source_model,
        candidate_types=args.candidate_types,
        prefix="",
        target_names=target_names,
        skip_names=skip_names,
        imatrix_dict=imatrix_dict,
        device=device,
        lazy=args.lazy,
        attention_candidates=args.attention_candidates,
        non_attention_candidates=args.non_attention_candidates,
        qbytes_device=qbytes_device,
    )
    print(f"Replaced {len(replaced_linear)} Linear layers in {time.time()-t0:.1f}s: {sorted(replaced_linear.keys())[:5]}...", flush=True)

    print("Replacing Embeddings with MixedQuantEmbedding ...", flush=True)
    # The vocab-tensor candidate quantization materializes multi-GB fp32 scratch;
    # serialize this phase across ranks (the rank-chain gates whole replacements
    # but a completed rank's steady state + a churning rank's embedding spike
    # still OOMs the host without this).
    _emb_lock = None
    if tp_enabled:
        import fcntl as _fcntl
        _emb_lock = open("/tmp/voodoo_tp_embed.lock", "w")
        _fcntl.flock(_emb_lock, _fcntl.LOCK_EX)
    t0 = time.time()
    replaced_embed = replace_embedding_with_mixed_quant(
        source_model,
        candidate_types=args.candidate_types,
        prefix="",
        skip_names=skip_names,
        imatrix_dict=imatrix_dict,
        device=device,
        lazy=args.lazy,
        qbytes_device=qbytes_device,
    )
    print(f"Replaced {len(replaced_embed)} Embedding layers in {time.time()-t0:.1f}s: {sorted(replaced_embed.keys())[:5]}...", flush=True)
    if _emb_lock is not None:
        import fcntl as _fcntl
        import ctypes as _ct
        gc.collect()
        try:
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        _fcntl.flock(_emb_lock, _fcntl.LOCK_UN)
        _emb_lock.close()
        _emb_lock = None
    if tp_enabled:
        # Minimum-footprint handoff: purge every MixedQuant module's cross-step
        # dequant cache and return freed arenas to the OS BEFORE releasing the
        # chain marker, so the next rank churns against this rank's floor
        # (not its post-init peak).
        import ctypes as _ct
        for _layer in list(replaced_linear.values()) + list(replaced_embed.values()):
            _inv = getattr(_layer, "invalidate_candidate_cache", None)
            if _inv is not None:
                _inv()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        # Signal the next rank that this rank's replacement phase (and its
        # ~12.4 GB transient) is complete — see the rank-chain above.
        import fcntl as _fcntl
        try:
            _own.seek(0); _own.truncate(); _own.write("done"); _own.flush()
        except NameError:
            pass

    replaced: dict[str, MixedQuantLinear | MixedQuantEmbedding] = {**replaced_linear, **replaced_embed}

    # Optional tensor-parallel SSM (GatedDeltaNet) sharding — common/tp_ssm.py.
    # Runs AFTER MixedQuant replacement: candidate-cache keys keep the
    # full-tensor names and the adapter row/column-shards those modules in
    # place.  Real TP under torchrun (each rank keeps its head group); without
    # a process group it is a single-process correctness simulation.
    if getattr(args, "tp_ssm", 0):
        from voodoo_quant.arch.qwen_ssm import apply_ssm_tp

        tp_ssm_modules = apply_ssm_tp(source_model, tp_size=args.tp_ssm or None)
        print(f"TP SSM adapter applied to {len(tp_ssm_modules)} GatedDeltaNet layers", flush=True)
        del tp_ssm_modules

    # Optional tensor-parallel attention (full-attention layers) sharding —
    # common/tp_attention.py.  Same contract as --tp_ssm: post-replacement
    # in-place sharding (cache keys keep full-tensor names), real TP under
    # torchrun, single-process simulation otherwise.
    if getattr(args, "tp_attention", 0):
        from voodoo_quant.arch.qwen_attention import apply_tp_attention

        tp_attn_modules = apply_tp_attention(source_model, tp_size=args.tp_attention)
        print(f"TP attention adapter applied to {len(tp_attn_modules)} attention layers", flush=True)
        del tp_attn_modules

    # Build the set of state-dict keys that are being dynamically quantized.
    targeted_keys = {name + ".weight" for name in replaced}
    if tie_word_embeddings:
        targeted_keys.add("lm_head.weight")

    if device.type == "cuda":
        print("GPU memory after replacement:", flush=True)
        print(torch.cuda.memory_summary(device=device, abbreviated=True), flush=True)
        # Spot-check that candidate buffers stayed on CPU.
        sample = list(replaced.values())[0]
        for k in range(min(3, sample.num_candidates)):
            cand = getattr(sample, f"_w_{k}", None)
            if cand is not None:
                print(f"  sample candidate {k} device: {cand.device}", flush=True)

    # Compute 8-bit-normalized original bytes and target bytes from the whole model.
    # Voodoo sizes are percentages of the original 8-bit model size, not the
    # source dtype size, so the denominator is numel * 1 byte regardless of dtype.
    original_bytes = compute_total_8bit_bytes(state_dict_meta)
    non_targeted_bytes = compute_non_targeted_8bit_bytes(state_dict_meta, targeted_keys)
    del state_dict_meta
    gc.collect()
    source_bits_per_elem = bits_per_element(dtype)
    if args.compression_ratio is not None:
        compression_ratio = args.compression_ratio
    else:
        # target_bits is interpreted as bits per weight relative to an 8-bit original.
        compression_ratio = args.target_bits / 8.0
    target_bytes = compression_ratio * original_bytes
    # Apply optional budget reduction (e.g., 5% smaller to leave headroom for tensor upgrades).
    if getattr(args, "budget_reduction", 0.0) > 0.0:
        target_bytes = target_bytes * (1.0 - args.budget_reduction)
        print(f"Budget reduction: {args.budget_reduction*100:.1f}% (new target: {target_bytes / 1e6:.2f} MB)")
    print(f"Original total bytes (8-bit reference): {original_bytes / 1e6:.2f} MB")
    print(f"Non-targeted bytes (8-bit reference): {non_targeted_bytes / 1e6:.2f} MB")
    print(f"Target bytes: {target_bytes / 1e6:.2f} MB (compression_ratio={compression_ratio:.4f}, target_bits={compression_ratio * 8.0:.2f})")

    # Finalize-only recovery path: load gates from a partial checkpoint and bake
    # without training.  Used to recover runs that crashed during finalization.
    # In TP mode only rank 0 bakes (it gathers every rank's assignments; the
    # bake is CPU-only from the full mmap'd base checkpoint, no TP needed).
    if args.finalize_from_partial is not None:
        # Finalize-only + TP runs WITHOUT collectives: non-rank0 ranks exited
        # early in _finalize_tp_bypass (before the MixedQuant rebuild), so a
        # barrier/gather here would wait forever on ranks that already left.
        # Rank 0 rebuilds every selectable tensor itself, so its local view is
        # already complete and the shared partial.pt carries the merged gates.
        merged = replaced
        print(f"\nFinalize-only: loading gates from {args.finalize_from_partial}", flush=True)
        apply_partial_gates(merged, load_partial(args.finalize_from_partial))
        original_sd, original_sd_cleanup = _build_original_sd(args, source_model, pre_replacement_snapshot)
        out = _finalize_and_save(
            args, merged, source_model, original_sd, original_sd_cleanup,
            non_targeted_bytes, target_bytes, compression_ratio, dtype,
            tie_word_embeddings, targeted_keys, device,
        )
        if tp_enabled:
            # No trailing barrier either — the peers are gone.  Destroy the
            # process group solo so shutdown never enters a collective.
            import torch.distributed as _dist

            try:
                _dist.destroy_process_group()
            except Exception:
                pass
        return out

    # Optional resume: seed the gates from a partial checkpoint before training.
    # Also record the journal's step count so the tau schedule and the partial
    # journal continue from where the dead run left off instead of restarting.
    args._resume_opt_steps = 0
    if args.resume_from_partial is not None:
        print(f"\nResuming gates from {args.resume_from_partial}", flush=True)
        _resume_payload = load_partial(args.resume_from_partial)
        apply_partial_gates(replaced, _resume_payload)
        _prev = int(_resume_payload.get("meta", {}).get("completed_optimizer_steps", 0) or 0)
        if _prev > 0:
            args._resume_opt_steps = _prev
            print(f"  [partial] journal at optimizer step {_prev}; tau/journal continue from there", flush=True)

    # Sensitivity warm-start: seed the gates from a MEASURED layout instead
    # of zeros. One hooked forward pass captures X_t per tensor; s(t,q) over
    # the cached candidates prices each rung; a greedy knapsack over
    # quality-per-byte gives the initial assignment; gates start concentrated
    # on it. Skipped when resuming (a journal already carries trained gates).
    if getattr(args, "warm_start", False) and args.resume_from_partial is None:
        print("\nWarm-start: measuring per-tensor sensitivity on calibration activations ...", flush=True)
        from voodoo_quant.training.sensitivity import (
            build_sensitivity_table,
            capture_layer_inputs,
            greedy_knapsack_layout,
            warm_start_gates,
        )

        print(f"  building calibration dataloader for activation capture ...", flush=True)
        warm_loader = build_dataloader(tokenizer, args.seq_len, args.batch_size, args.data_dir, args.data_name)
        warm_inputs = capture_layer_inputs(
            source_model, replaced, warm_loader, device,
            max_batches=args.warm_start_batches, max_tokens=args.warm_start_tokens,
        )
        del warm_loader
        sens_table = build_sensitivity_table(replaced, warm_inputs)
        del warm_inputs
        gc.collect()
        # persist the table next to the journal: the polish pass (this run or
        # a later --finalize_from_partial) reuses the MEASURED sensitivities.
        try:
            import pickle

            _tbl_path = Path(args.output_dir) / "sensitivity.pkl"
            with open(_tbl_path, "wb") as f:
                pickle.dump(sens_table, f)
            print(f"  sensitivity table saved to {_tbl_path}", flush=True)
        except Exception as exc:
            print(f"  WARNING: could not persist sensitivity table: {exc}", flush=True)
        warm_layout = greedy_knapsack_layout(sens_table, float(target_bytes), non_targeted_bytes)
        n_seeded = warm_start_gates(replaced, warm_layout)
        warm_total = sum(sens_table.bytes_[t][q] for t, q in warm_layout.items()) + non_targeted_bytes
        print(f"  warm start: seeded {n_seeded}/{len(replaced)} tensors; "
              f"initial layout {warm_total / 1e6:.1f} MB (target {float(target_bytes) / 1e6:.1f} MB)", flush=True)
        # keep the table for the polish pass if both are enabled
        if getattr(args, "polish", False):
            args._sens_table = sens_table
        else:
            del sens_table

    # Freeze everything except the gates.
    for p in source_model.parameters():
        p.requires_grad = False
    for layer in replaced.values():
        layer.gates.requires_grad = True

    trainable = [layer.gates for layer in replaced.values()]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    # TP host-RAM teardown: the mmap'd base checkpoint is opened MAP_PRIVATE,
    # and any tensor op that writes through a view (e.g. torch cache-path
    # resaves) copy-on-write dirties pages PER RANK — measured 8.1 GiB of
    # private-dirty per rank (4 ranks = 32 GiB of duplicated anon; the
    # recurring OOM at ~12.8 GB/rank).  Candidates are already loaded/sliced,
    # so ranks 1-3 drop every reference to the mapping (tp_full_weight attrs
    # hold the views alive).  Rank 0 keeps its handle for the final bake,
    # which reopens the checkpoint anyway via _build_original_sd.
    if tp_enabled:
        _dropped = 0
        for m in source_model.modules():
            if hasattr(m, "tp_full_weight"):
                m.tp_full_weight = None
                _dropped += 1
        gc.collect()
        try:
            import ctypes as _ct
            _ct.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        print(f"  [tp-mmap] rank dropped {_dropped} tp_full_weight refs", flush=True)

    print(f"Building dataloader from {args.data_dir}/{args.data_name} ...")
    dataloader = build_dataloader(tokenizer, args.seq_len, args.batch_size, args.data_dir, args.data_name)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-3)

    source_model.train()
    teacher.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU cache cleared before training loop", flush=True)
        try:
            import resource as _res
            _rss_gb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1048576
            _rk = os.environ.get("LOCAL_RANK", "?")
            print(f"  [mem] rank {_rk}: peak RSS {_rss_gb:.1f} GB at loop start", flush=True)
        except Exception:
            pass

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    if not tp_enabled or _tp_is_rank0():
        Path(args.log_file).write_text("")

    total_optimizer_steps = 0
    completed_opt_steps = 0
    if getattr(args, "_resume_opt_steps", 0) > 0:
        completed_opt_steps = args._resume_opt_steps
        total_optimizer_steps = completed_opt_steps * args.grad_accum_steps
    # Full-init start gate: a rank that finished its replacement early must NOT
    # begin training prep (dataloader/optimizer/activation working set) while
    # later ranks are still churning — that overlap is what the OOM killer has
    # been taking rank 0 for, eight times running.  Wait until every rank's
    # marker exists before the training loop starts.
    if tp_enabled:
        from voodoo_quant.parallel import TP as _gtp
        # While parked at the gate this rank's pages are cold; tell the OOM
        # killer to prefer swap-reclaim over killing us so a later rank's
        # churn doesn't take us down (restored once the gate opens).
        try:
            with open("/proc/self/oom_score_adj", "w") as _osa:
                _osa.write("-800")
        except Exception:
            pass
        for _r in range(_gtp.world_size):
            _m = Path(args.output_dir) / f".rank{_r}_replaced.done"
            while not (_m.exists() and _m.read_text().strip() == "done"):
                time.sleep(5.0)
        try:
            with open("/proc/self/oom_score_adj", "w") as _osa:
                _osa.write("0")
        except Exception:
            pass
        print("  [start-gate] all ranks replaced; training begins", flush=True)
    # Effective steps after accounting for grad accumulation. On resume, only
    # the steps NOT already journaled are consumed so the run ends at exactly
    # max_steps optimizer steps total (tau stays on the original schedule).
    effective_max_steps = args.max_steps * args.grad_accum_steps
    remaining_steps = max(0, effective_max_steps - total_optimizer_steps)
    if remaining_steps < effective_max_steps:
        print(f"  [partial] resume: {remaining_steps} of {effective_max_steps} steps remain", flush=True)
    pbar = tqdm(
        itertools.islice(dataloader, remaining_steps),
        total=effective_max_steps,
        desc="DQ",
        initial=total_optimizer_steps,
    )

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        # TP: stagger only the KERNEL COMPILATION, never the training step.
        # The decoder layers end in all-reduce collectives, so ranks must run
        # the forward in lockstep; serializing whole first steps (the old
        # chain-lock) deadlocked rank 0's first all-reduce against ranks
        # still waiting on their chain locks.  Instead: before iteration 0,
        # each rank warms the flash-attn/SSM kernels on a TINY dummy batch
        # under the chain lock (compile is shape-specialized but the warm
        # pass primes COMGR/Triton caches and — critically — the code-object
        # machinery), then all ranks enter the real loop together.  Host-RAM
        # safety comes from the tiny batch's small transients plus the
        # post-replacement mmap teardown.
        if (getattr(args, "tensor_parallel", None) or 1) > 1 and total_optimizer_steps == 0:
            import fcntl as _fcntl
            import os as _os
            _rk = int(_os.environ.get("LOCAL_RANK", "0"))
            _mine = open(f"/tmp/voodoo_tp_first_{_rk}.lock", "w")
            _fcntl.flock(_mine, _fcntl.LOCK_EX)
            if _rk > 0:
                _prev = open(f"/tmp/voodoo_tp_first_{_rk - 1}.lock", "w")
                _fcntl.flock(_prev, _fcntl.LOCK_EX)
                _fcntl.flock(_prev, _fcntl.LOCK_UN)
                _prev.close()
            # Collective-free warmup: the model forward all-reduces per layer,
            # so warming IT under the chain lock would deadlock (rank 0 in
            # all-reduce, rank 1 on flock).  Warm only the cross-rank-free
            # kernels that dominate compile time: flash-attn (the COMGR hog)
            # and the T-chunked SSM fallback, on dummy tensors.
            try:
                if attn_impl == "rocm_triton":
                    from flash_attn import flash_attn_func as _faf
                    for _wl in (128, 2048, args.seq_len):
                        _q = torch.randn(1, _wl, 6, 256, device=device, dtype=torch.bfloat16)
                        _k = torch.randn(1, _wl, 1, 256, device=device, dtype=torch.bfloat16)
                        _v = torch.randn(1, _wl, 1, 256, device=device, dtype=torch.bfloat16)
                        _ = _faf(_q, _k, _v, causal=True)
                    torch.cuda.synchronize()
                print(f"  [warm] rank {_rk} kernels warmed", flush=True)
            except Exception as _warm_err:  # warm failure must not kill the run
                print(f"  [warm] rank {_rk} warmup failed (continuing): {_warm_err}", flush=True)
            import ctypes as _ct
            import gc as _gc
            _gc.collect()
            try:
                _ct.CDLL("libc.so.6").malloc_trim(0)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _fcntl.flock(_mine, _fcntl.LOCK_UN)
            _mine.close()

        # Anneal temperature per effective step (after grad accum).
        effective_step = total_optimizer_steps // args.grad_accum_steps
        tau = anneal_temperature(effective_step, args.max_steps, args.temp_start, args.temp_end)

        # Optimization 4: invalidate candidate dequant cache every N optimizer steps.
        # Between invalidations, dequantized candidates are reused (gradients stay exact).
        if args.candidate_cache_interval > 1 and effective_step % args.candidate_cache_interval == 0:
            for layer in replaced.values():
                layer.invalidate_candidate_cache()

        for layer in replaced.values():
            layer.set_temperature(tau)

        T = args.distill_temperature
        if tp_enabled:
            # Vocab-parallel TP loss: per-rank logits over this rank's vocab
            # shard, combined exactly with a cross-rank log-sum-exp.  Chunking
            # over tokens bounds the [chunk, V/N] softmax exactly like the
            # single-GPU --logits_chunk path.
            lm_loss, distill_loss = _tp_vocab_parallel_losses(
                source_model, teacher, x, y, T,
                chunk=args.logits_chunk if args.logits_chunk > 0 else 128,
                distill_weight=args.distill_weight,
            )
            logits = None
            V = 0
        elif args.logits_chunk > 0:
            # Chunked forward+loss: run the text stack once for hidden states,
            # then compute lm_head logits, CE, and distillation KL per token
            # chunk so only chunk-sized [chunk, V] softmax tensors are live at
            # once (the full [B*S, V] materialization OOMs on 248k vocab).
            # Gradients are exact: both losses decompose as sums over tokens.
            lm_loss, distill_loss = _chunked_forward_loss(
                source_model, teacher, x, y, T,
                chunk=args.logits_chunk, loss_device=loss_device_str,
                teacher_device=teacher_device_str,
            )
            logits = None
            V = 0
        else:
            outputs = source_model(input_ids=x)
            logits = outputs.logits
            V = logits.size(-1)

            with torch.no_grad():
                if teacher_device_str != str(device):
                    teacher_out = teacher(input_ids=x.to(teacher_device_str))
                    teacher_logits = teacher_out.logits.to(device)
                else:
                    teacher_out = teacher(input_ids=x)
                    teacher_logits = teacher_out.logits

            if loss_device_str == "cpu":
                # Move the loss (and its large full-vocab softmax tensors) to CPU to
                # keep VRAM under the cap for large models. Non-detaching CPU copies
                # backprop to the GPU graph through CopyBackwards.
                lg = logits.to("cpu")
                yc = y.to("cpu")
                tlc = teacher_logits.to("cpu")
                lm_loss = F.cross_entropy(lg.view(-1, V), yc.view(-1))
                student_log_probs = F.log_softmax(lg / T, dim=-1)
                teacher_probs = F.softmax(tlc / T, dim=-1)
                per_token_kl = F.kl_div(
                    student_log_probs.view(-1, V),
                    teacher_probs.view(-1, V),
                    reduction="none",
                ).sum(dim=-1) * (T * T)
                distill_loss = per_token_kl.mean()
            else:
                lm_loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
                student_log_probs = F.log_softmax(logits / T, dim=-1)
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                per_token_kl = F.kl_div(
                    student_log_probs.view(-1, V),
                    teacher_probs.view(-1, V),
                    reduction="none",
                ).sum(dim=-1) * (T * T)
                distill_loss = per_token_kl.mean()

        if tp_enabled:
            # `replaced` only holds this rank's slices, so the size budget (a
            # global quantity) is the cross-rank SUM of per-rank effective
            # bytes.  Uses the differentiable all-reduce (identity backward):
            # each rank's gates appear only in its local sum, so rank-local
            # gradients are exact.
            from voodoo_quant.parallel import tp_all_reduce_sum

            current_bytes = tp_all_reduce_sum(total_effective_bytes(replaced).to(device)) + non_targeted_bytes
        else:
            current_bytes = total_effective_bytes(replaced).to(device) + non_targeted_bytes
        relative_size_error = torch.abs(current_bytes / target_bytes - 1.0)
        size_error = torch.relu(relative_size_error - args.size_tolerance)
        size_loss = size_error ** 2

        if tp_enabled or args.logits_chunk > 0:
            # The chunked path already ran per-chunk head backwards and one
            # aggregated model backward (hidden.backward).  Only the size
            # term remains; backward it into the gates directly.
            lm_loss = torch.tensor(lm_loss)
            distill_loss = torch.tensor(distill_loss)
            (args.size_weight * size_loss).backward()
        else:
            if loss_device_str == "cpu":
                loss = lm_loss + args.distill_weight * distill_loss + args.size_weight * size_loss.to("cpu")
            else:
                loss = lm_loss + args.distill_weight * distill_loss + args.size_weight * size_loss
            optimizer.zero_grad()
            loss.backward()

        if (total_optimizer_steps + 1) % args.grad_accum_steps == 0:
            if tp_enabled:
                # Each rank only produces gradients for its own gate slices;
                # the all-reduce makes every rank's optimizer state identical.
                _tp_all_reduce_gate_grads(trainable)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed_opt_steps += 1
            # Notify the ST-Gumbel fraction-anneal schedule of training progress.
            if getattr(args, "st_gumbel_anneal", None):
                from voodoo_quant import layers as _layers

                _layers.st_gumbel_step(completed_opt_steps)
            if args.partial_save_interval > 0 and (completed_opt_steps % args.partial_save_interval == 0):
                if tp_enabled:
                    # ALL ranks enter the gather (it is a collective); rank 0
                    # alone writes the journal with every rank's gates (its
                    # local `replaced` view only covers its slices).
                    merged = _tp_gather_replaced(replaced)
                    if _tp_is_rank0():
                        save_partial(args, merged, completed_opt_steps, tau, reason="interval")
                else:
                    save_partial(args, replaced, completed_opt_steps, tau, reason="interval")
        total_optimizer_steps += 1

        pbar.set_postfix({
            "step": f"{total_optimizer_steps}/{args.max_steps}",
            "tau": f"{tau:.4f}",
            "lm": f"{lm_loss.item():.4f}",
            "kl": f"{distill_loss.item():.4f}",
            "size_mb": f"{current_bytes.item() / 1e6:.2f}",
            "size_loss": f"{size_loss.item():.4f}",
        })

        if total_optimizer_steps % (args.log_interval * args.grad_accum_steps) == 0:
            log_line = (
                f"step {total_optimizer_steps // args.grad_accum_steps} | tau {tau:.4f} | "
                f"lm_loss {lm_loss.item():.4f} | distill_loss {distill_loss.item():.4f} | "
                f"size_mb {current_bytes.item() / 1e6:.2f} | size_loss {size_loss.item():.4f} | "
                f"total {float(lm_loss) + args.distill_weight * float(distill_loss) + args.size_weight * float(size_loss):.4f}"
            )
            print(log_line)
        if not tp_enabled or _tp_is_rank0():
            with open(args.log_file, "a") as f:
                json.dump({
                    "step": total_optimizer_steps // args.grad_accum_steps,
                    "tau": tau,
                    "lm_loss": lm_loss.item(),
                    "distill_loss": distill_loss.item(),
                    "size_mb": current_bytes.item() / 1e6,
                    "size_loss": size_loss.item(),
                    "total_loss": float(lm_loss) + args.distill_weight * float(distill_loss) + args.size_weight * float(size_loss),
                    "lr": optimizer.param_groups[0]["lr"],
                }, f)
                f.write("\n")

        if total_optimizer_steps // args.grad_accum_steps >= args.max_steps:
            break

    # Persist the learned assignments permanently via the resilient finalizer.
    # It sources each original weight from the authoritative mapping (base-
    # checkpoint mmap, or the pre-replacement snapshot) so it cannot KeyError on
    # replaced layers, persists the assignments + a final partial BEFORE the
    # risky quantization, and self-corrects by skipping any unresolvable tensor.
    # TP: rank 0 alone runs the (CPU-only) bake, from every rank's assignments;
    # the gates are identical on shared tensors, and rank-local tensors are
    # quantized from the FULL mmap'd base weights, so the output checkpoint is
    # identical to a single-GPU run's.
    final_replaced = replaced
    if tp_enabled:
        from voodoo_quant.parallel import barrier as _tp_barrier

        _tp_barrier()
        # The gather is a collective: every rank must enter it.  Only rank 0
        # runs the (CPU-only) bake from the merged view; the gates are
        # identical on shared tensors, and rank-local tensors are quantized
        # from the FULL mmap'd base weights, so the output checkpoint matches a
        # single-GPU run's.
        final_replaced = _tp_gather_replaced(replaced)
        if not _tp_is_rank0():
            _tp_barrier()
            from voodoo_quant.parallel import tp_finalize

            tp_finalize()
            return None
        # Rank 0: the real MixedQuant modules are no longer needed (the
        # gathered shims carry the gates/assignments); free their candidate
        # buffers before the bake claims the host memory.
        _free_replaced_residue(replaced)
    original_sd, original_sd_cleanup = _build_original_sd(args, source_model, pre_replacement_snapshot)
    _finalize_and_save(
        args, final_replaced, source_model, original_sd, original_sd_cleanup,
        non_targeted_bytes, target_bytes, compression_ratio, dtype,
        tie_word_embeddings, targeted_keys, device,
    )
    if tp_enabled:
        from voodoo_quant.parallel import barrier as _tp_barrier

        _tp_barrier()
        from voodoo_quant.parallel import tp_finalize

        tp_finalize()


def main():
    parser = build_parser()
    args = parser.parse_args()
    with log_stage(
        stage="train",
        model_dir=args.output_dir,
        script=Path(__file__).name,
        args={k: v for k, v in vars(args).items() if k != "base_checkpoint"},
    ):
        run(args)


if __name__ == "__main__":
    main()
