"""Create a language-model base checkpoint and a Q8_0 teacher from HF weights.

Produces the two artifacts large-model training needs:

- ``<slug>_base.pt``   — BF16 full weights as ``{"model_state_dict": ...}``
  (what ``--base_checkpoint`` mmaps for candidate quantization and the bake)
- ``<slug>_teacher_q8_0.pt`` — every Linear/Embedding weight stored as
  llama.cpp Q8_0 bytes + ``quant_meta`` (what ``--teacher_quant Q8_0
  --teacher_checkpoint`` consumes)

Quantization is llama.cpp-exact (libggml-base via ctypes) and parallel across
CPU cores — the GIL-releasing ctypes calls scale near-linearly.

Typical use (any HF checkpoint directory with safetensors):

    voodoo make-teacher --model_dir ./Qwen3.8-27B \
        --strip_prefix model.language_model. \
        --output_dir checkpoints/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path

import torch
from tqdm import tqdm

from voodoo_quant.ggml import quantize_tensor

_SKIP_FRAGMENTS = ("norm", "bias", "A_log", "dt_bias", "conv1d")


def load_state_dict(model_dir: Path, strip_prefix: str) -> dict[str, torch.Tensor]:
    """Stream safetensors shards, optionally stripping a wrapper prefix.

    Streams one file at a time and keeps every key (dropping keys would break
    completeness; the caller filters at quantize time).
    """
    from safetensors.torch import load_file

    sd: dict[str, torch.Tensor] = {}
    for path in sorted(list(model_dir.glob("*.safetensors")) + list(model_dir.glob("model-*.safetensors"))):
        if path.name.startswith(".") or path.name.endswith(".index.json"):
            continue
        shard = load_file(path)
        for k, v in shard.items():
            if strip_prefix and k.startswith(strip_prefix):
                sd[k[len(strip_prefix):]] = v
            else:
                sd[k] = v
        del shard
    if not sd:
        raise FileNotFoundError(f"no safetensors shards found under {model_dir}")
    return sd


def is_quantizable_key(key: str) -> bool:
    """True for Linear/Embedding weights that should be Q8_0-quantized."""
    if not key.endswith(".weight") or key.count(".") == 0:
        return False
    return not any(frag in key for frag in _SKIP_FRAGMENTS)


def quantize_state_dict_q8_0(
    sd: dict[str, torch.Tensor],
    workers: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    """Quantize all quantizable weights to Q8_0 (exact ggml, parallel)."""
    out_sd: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}

    keys = [k for k in sd if is_quantizable_key(k)]

    def _quantize_one(key: str):
        tensor = sd[key]
        w = tensor.detach().to(torch.float32).cpu().contiguous()
        qb = quantize_tensor(w, "Q8_0")
        return key, qb, {
            "quant_type": "Q8_0",
            "out_features": int(tensor.shape[0]),
            "in_features": int(tensor.shape[1]),
            "dtype": "uint8",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_quantize_one, k) for k in keys]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Q8_0 teacher"):
            key, qb, m = fut.result()
            out_sd[key] = qb
            meta[key] = m

    for key, tensor in sd.items():
        if key not in out_sd:
            out_sd[key] = tensor.cpu()

    return out_sd, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a BF16 base .pt and Q8_0 teacher from an HF checkpoint dir")
    parser.add_argument("--model_dir", required=True, help="HF checkpoint directory with safetensors shards")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_name", default=None, help="base checkpoint filename (default: <dir-name>_base.pt)")
    parser.add_argument("--teacher_name", default=None, help="teacher filename (default: <dir-name>_teacher_q8_0.pt)")
    parser.add_argument("--strip_prefix", default=None,
                        help="drop this prefix from tensor keys (e.g. 'model.language_model.' for multimodal checkpoints)")
    parser.add_argument("--workers", default=int(os.environ.get("VOODOO_QUANT_WORKERS", min(12, os.cpu_count() or 2))), type=int)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_dir.name

    base_path = output_dir / (args.base_name or f"{slug}_base.pt")
    teacher_path = output_dir / (args.teacher_name or f"{slug}_teacher_q8_0.pt")

    if base_path.exists():
        print(f"Loading existing base checkpoint {base_path} (mmap) ...")
        sd = torch.load(base_path, weights_only=True, map_location="cpu", mmap=True)["model_state_dict"]
        print(f"  {len(sd)} tensors")
    else:
        print(f"Loading weights from {model_dir} ...")
        sd = load_state_dict(model_dir, args.strip_prefix)
        print(f"  {len(sd)} tensors")
        print(f"Saving BF16 base checkpoint to {base_path} ...")
        torch.save({"model_state_dict": sd}, base_path)

    print(f"Quantizing teacher to Q8_0 with {args.workers} workers ...")
    q8_sd, meta = quantize_state_dict_q8_0(sd, args.workers)

    print(f"Saving Q8_0 teacher checkpoint to {teacher_path} ...")
    torch.save(
        {
            "model_state_dict": q8_sd,
            "quant_meta": meta,
            "extra": {"source": str(model_dir), "teacher_quant": "Q8_0",
                      "strip_prefix": args.strip_prefix},
        },
        teacher_path,
    )

    print("Done.")
    print(f"  Base:    {base_path} ({base_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  Teacher: {teacher_path} ({teacher_path.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
