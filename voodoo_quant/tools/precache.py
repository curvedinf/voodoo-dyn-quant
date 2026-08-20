#!/usr/bin/env python3
"""Pre-build the quant candidate cache with llama.cpp's exact quantizer.

Loads the model architecture from its config (no weight download) and
quantizes each parameter from the converted checkpoint with
``libggml-base.so`` (the real llama.cpp quantizer, so the cache is
bit-identical to the exported GGUF).  Quantization runs on the CPU,
parallelized across cores (``VOODOO_QUANT_WORKERS``); it streams one tensor
at a time and stores all candidates for a tensor in a single combined file
under ``voodoo_quant.layers.CANDIDATE_CACHE_DIR`` to minimize I/O overhead
during training init.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from voodoo_quant.ggml import default_candidate_types, dequantize_tensor, quantize_tensor
from voodoo_quant.layers import (
    CANDIDATE_CACHE_DIR,
    _combined_cache_path,
    _format_duration,
    _pad_weight_and_imatrix,
    _tensor_hash,
    resolve_candidate_types,
)


class _Progress:
    """Lightweight, dependency-free progress + ETA for the cache build."""

    def __init__(self, total_tensors: int, candidates_per_tensor: int):
        self.total = total_tensors
        self.cpt = candidates_per_tensor
        self.start = time.monotonic()
        self.last_print = 0.0
        self.tensors_done = 0
        self.tensors_built = 0
        self.entries_built = 0
        self.entries_cached = 0
        self.errors = 0
        self._ema_build = None  # seconds per built tensor (EMA)
        self._last_build_t = self.start

    def update(self, built_entries: int, cached_entries: int, errors: int):
        self.tensors_done += 1
        self.entries_built += built_entries
        self.entries_cached += cached_entries
        self.errors += errors
        now = time.monotonic()
        if built_entries > 0:
            self.tensors_built += 1
            dt = now - self._last_build_t
            self._last_build_t = now
            if dt > 0:
                self._ema_build = (
                    dt if self._ema_build is None else 0.3 * dt + 0.7 * self._ema_build
                )
        if now - self.last_print >= 1.0 or self.tensors_done == self.total:
            self.last_print = now
            self._print(now)

    def _print(self, now: float):
        elapsed = now - self.start
        pct = 100.0 * self.tensors_done / self.total if self.total else 100.0
        # ETA assumes every remaining tensor needs building (conservative on resume).
        if self._ema_build:
            eta = (self.total - self.tensors_done) * self._ema_build
        elif elapsed > 0 and self.tensors_done > 0:
            eta = (self.total - self.tensors_done) * (elapsed / self.tensors_done)
        else:
            eta = None
        rate = (self.tensors_built / elapsed) if elapsed > 0 else 0.0
        print(
            f"\r  [{pct:5.1f}%] {self.tensors_done}/{self.total} tensors | "
            f"built {self.entries_built} cached {self.entries_cached} err {self.errors} | "
            f"{rate:.2f} build-tensor/s | "
            f"elapsed {_format_duration(elapsed)} eta {_format_duration(eta)}   ",
            end="", flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pre-build quant candidate cache (exact, CPU)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="Base HF model id whose architecture matches the checkpoint (config-only "
             "load, no weights are downloaded). Falls back to $VOODOO_PRECACHE_MODEL.",
    )
    parser.add_argument("--candidate_types", nargs="+", default=default_candidate_types())
    parser.add_argument("--attention_candidates", nargs="+", default=None)
    parser.add_argument("--non_attention_candidates", nargs="+", default=None)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--device", default="cpu", help="unused; quantization is CPU-only (kept for CLI compat)")
    args = parser.parse_args(argv)

    # Build the architecture from the (cached) config only, then overlay the converted
    # checkpoint weights.  from_pretrained(local_files_only=True) needs the base
    # model's weight files on disk, which GGUF-only families (e.g. Qwen3.5-2B/4B-MTP)
    # do not ship.  from_config needs no download.  The base id must match the
    # checkpoint architecture; give it via --model or VOODOO_PRECACHE_MODEL.
    _base_model = args.model or os.environ.get("VOODOO_PRECACHE_MODEL")
    if not _base_model:
        parser.error(
            "--model is required (or set VOODOO_PRECACHE_MODEL): the base HF model id "
            "whose architecture matches the checkpoint."
        )

    print(f"Lazy mode: {args.lazy}")
    print(f"Cache dir: {CANDIDATE_CACHE_DIR}")

    # Warm up ggml's lazy static init (IQ grid/neighbor tables) on the main
    # thread so the parallel per-tensor workers never race on it.
    print("Warming up ggml quantizer for all candidate types ...")
    _warm = torch.zeros(1, 256, dtype=torch.float32)
    for _qt in args.candidate_types:
        quantize_tensor(_warm, _qt, None)
    print("ggml warmup done")

    print(f"Loading model architecture from {_base_model} ...")
    from transformers import AutoConfig, AutoModelForCausalLM
    _cfg = AutoConfig.from_pretrained(_base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(_cfg, trust_remote_code=True).to(torch.bfloat16)
    # Load checkpoint with mmap and assign
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False, assign=True)
    print("Model loaded")

    # Select 2D tensors (use module name without .weight to match training)
    # Per-group candidate restriction mirrors training: linear_attn/self_attn
    # tensors use the attention set, everything else the non-attention set.
    # alpha/beta (in_proj_a/in_proj_b) stay F32 in the recipe — never cached.
    attention_candidates = None
    non_attention_candidates = None
    if args.attention_candidates or args.non_attention_candidates:
        attention_candidates = args.attention_candidates
        non_attention_candidates = args.non_attention_candidates

    selectable = []
    types_per_tensor = {}
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        if param.shape[1] % 256 != 0 and param.shape[1] % 32 != 0:
            continue
        module_name = name.removesuffix(".weight") if name.endswith(".weight") else name
        if module_name.endswith((".in_proj_a", ".in_proj_b")):
            continue  # recipe: alpha/beta stay F32
        tensor_types = resolve_candidate_types(
            module_name, args.candidate_types, attention_candidates, non_attention_candidates
        )
        if not tensor_types:
            continue
        selectable.append(module_name)
        types_per_tensor[module_name] = tensor_types

    total_entries = sum(len(v) for v in types_per_tensor.values())
    print(f"Found {len(selectable)} selectable tensors")
    print(f"Total cache entries: {total_entries}")

    progress = _Progress(len(selectable), 4)

    for idx, name in enumerate(selectable):
        param_name = name + ".weight"
        param = model.get_parameter(param_name)
        weight = param.detach().to(torch.float32).cpu().contiguous()
        out_features, in_features = weight.shape

        padded_weight, padded_imatrix, _ = _pad_weight_and_imatrix(weight, None)
        weight_hash = _tensor_hash(padded_weight)
        imatrix_hash = "none"
        mode = "lazy" if args.lazy else "dequant"
        padded_in = padded_weight.shape[1]

        # Check if combined cache exists
        combined_path = _combined_cache_path(name, mode, weight_hash, imatrix_hash)
        if combined_path.exists():
            progress.update(built_entries=0, cached_entries=len(types_per_tensor.get(name, [])), errors=0)
            del weight, padded_weight
            continue

        # Quantize all candidates EXACTLY with llama.cpp's ggml quantizer on the
        # CPU, in parallel (ctypes releases the GIL -> near-linear speedup, zero
        # approximation). One tensor at a time -> bounded memory; freed right after.
        import concurrent.futures

        def _build(qt: str):
            qbytes = quantize_tensor(padded_weight, qt, None)
            if args.lazy:
                return qt, qbytes
            w_deq = dequantize_tensor(qbytes, qt, out_features, padded_in)
            return qt, w_deq.to(torch.bfloat16).cpu()

        max_workers = int(os.environ.get("VOODOO_QUANT_WORKERS", min(16, (os.cpu_count() or 2))))
        errs_this = 0
        combined: dict[str, torch.Tensor] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_build, qt): qt for qt in types_per_tensor[name]}
            for fut in concurrent.futures.as_completed(futs):
                qt = futs[fut]
                try:
                    rqt, buf = fut.result()
                    combined[rqt] = buf
                except Exception as e:  # noqa: BLE001
                    errs_this += 1
                    print(f"\n  Error for {name} {qt}: {e}")

        # Save combined cache
        if combined:
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(combined, combined_path)

        progress.update(built_entries=len(combined), cached_entries=0, errors=errs_this)

        del weight, padded_weight

    elapsed = time.monotonic() - progress.start
    print(f"\nDone in {_format_duration(elapsed)}!")
    print(f"  Newly quantized: {progress.entries_built}")
    print(f"  Already cached:  {progress.entries_cached}")
    print(f"  Errors:          {progress.errors}")
    print(f"  Total entries:   {progress.entries_built + progress.entries_cached}")


if __name__ == "__main__":
    main()
