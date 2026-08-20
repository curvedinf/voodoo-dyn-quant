#!/usr/bin/env python3
"""
Evaluate a learned mixed-precision Voodoo assignment against the BF16 teacher
and an optional Unsloth Dynamic baseline (torch PPL + KL).

Usage:
    voodoo eval \
        --checkpoint checkpoints/Qwen3.5-0.8B/Voodoo45/Qwen3.5-0.8B-Voodoo45.pt \
        --output_dir checkpoints/Qwen3.5-0.8B/Voodoo45/evals \
        --max_steps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Configure persistent Triton/Inductor caches before any torch/triton usage.
import voodoo_quant.cache  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from voodoo_quant.ggml import bytes_per_weight, dequantize_tensor, quantize_tensor
from voodoo_quant.layers import pad_weight_and_imatrix
from voodoo_quant.stats import log_stage
from voodoo_quant.tools.data import TokenizedTensorDataset


class QuantizedLinear(nn.Module):
    """nn.Linear stand-in frozen at its llama.cpp-quantized weight.

    Quantizes the source weight once with llama.cpp's own quantizer (via
    ``voodoo_quant.ggml``) and caches the dequantized baseline, so the eval
    forward is exactly the exported GGUF's weight. ``in_features`` that are
    not a multiple of the quant block size are zero-padded internally, the
    same way the candidate cache and the export do.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: str,
        weight: torch.Tensor,
        imatrix: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        with torch.no_grad():
            w = weight.detach().to(torch.float32).cpu()
            w_pad, imatrix_pad, _ = pad_weight_and_imatrix(w, imatrix)
            qweight = quantize_tensor(w_pad, quant_type, imatrix_pad)
            baseline = dequantize_tensor(
                qweight, quant_type, out_features, w_pad.shape[1]
            )[:, :in_features]
        self.register_buffer(
            "weight_baseline",
            baseline.to(device=device, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_baseline.to(x.dtype))


def _replace_linear_with_dynamic_quant(
    module: nn.Module,
    quant_assignments: dict[str, str],
    prefix: str = "",
    quantize_embeddings: bool = False,
    skip_attention: bool = False,
    imatrix_dict: dict[str, torch.Tensor] | None = None,
):
    """
    Recursively replace nn.Linear children with QuantizedLinear modules whose
    quant type is taken from `quant_assignments`.

    Tensors not present in `quant_assignments` are left unchanged (i.e., kept
    in the source dtype).  This allows a mixed-precision model where some
    layers remain at higher precision while others are quantized.
    """
    if imatrix_dict is None:
        imatrix_dict = {}

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if skip_attention and name == "self_attn":
            continue
        if isinstance(child, nn.Linear):
            if not quantize_embeddings and name in {"embed_tokens", "lm_head"}:
                continue
            if child.bias is not None:
                raise NotImplementedError(
                    f"QuantizedLinear does not support bias but {name} has one"
                )
            quant_type = quant_assignments.get(full_name)
            if quant_type is None:
                continue
            imatrix = imatrix_dict.get(full_name)
            if imatrix is None:
                imatrix = imatrix_dict.get(full_name + ".weight")
            new_child = QuantizedLinear(
                child.in_features,
                child.out_features,
                quant_type,
                child.weight.data,
                imatrix=imatrix.cpu() if imatrix is not None else None,
                device=child.weight.device,
            )
            setattr(module, name, new_child)
        else:
            _replace_linear_with_dynamic_quant(
                child,
                quant_assignments,
                prefix=full_name,
                quantize_embeddings=quantize_embeddings,
                skip_attention=skip_attention,
                imatrix_dict=imatrix_dict,
            )


def build_val_dataloader(tokenizer, seq_len: int, batch_size: int, data_dir: str = "data/fineweb_qwen", data_name: str = "val_tokens.pt"):
    val_path = Path(data_dir) / data_name
    if not val_path.exists():
        raise FileNotFoundError(f"Validation tokens not found at {val_path}")
    tokens = torch.load(val_path, weights_only=True)
    dataset = TokenizedTensorDataset(tokens, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_steps: int | None = None):
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Eval")
    for step, (x, y) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device)
        y = y.to(device)

        outputs = model(input_ids=x)
        nll = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += y.numel()

        ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
        pbar.set_postfix({"ppl": f"{ppl:.2f}"})

    return {"ppl": ppl, "nll": total_nll / total_tokens}


@torch.no_grad()
def evaluate_with_kl(model, teacher, dataloader, device, max_steps: int | None = None, temperature: float = 1.0):
    model.eval()
    teacher.eval()
    total_nll = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_kl_tokens = 0

    pbar = tqdm(dataloader, desc="Eval KL")
    for step, (x, y) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device)
        y = y.to(device)

        outputs = model(input_ids=x)
        teacher_logits = teacher(input_ids=x).logits

        nll = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += y.numel()

        T = temperature
        student_log_probs = F.log_softmax(outputs.logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(
            student_log_probs.cpu().float().view(-1, student_log_probs.size(-1)),
            teacher_probs.cpu().float().view(-1, teacher_probs.size(-1)),
            reduction="sum",
        ) * (T * T)
        total_kl += kl.item()
        total_kl_tokens += y.numel()

        ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
        pbar.set_postfix({"ppl": f"{ppl:.2f}", "kl": f"{total_kl / total_kl_tokens:.4f}"})

    return {
        "ppl": torch.exp(torch.tensor(total_nll / total_tokens)).item(),
        "nll": total_nll / total_tokens,
        "kl": total_kl / total_kl_tokens,
    }


def apply_quant_assignments(model, quant_assignments, imatrices=None):
    """In-place replace selected Linear layers with QuantizedLinear per assignment."""
    imatrix_dict = imatrices or {}
    _replace_linear_with_dynamic_quant(
        model,
        quant_assignments,
        prefix="",
        quantize_embeddings=False,
        skip_attention=False,
        imatrix_dict=imatrix_dict,
    )
    return model


def compute_total_bytes(state_dict, quant_assignments, source_dtype=torch.bfloat16):
    bits_per_elem = torch.finfo(source_dtype).bits
    selectable_keys = {name + ".weight" for name in quant_assignments}
    total = 0.0
    for key, t in state_dict.items():
        if key in selectable_keys:
            qt = quant_assignments[key[:-7]]  # strip '.weight'
            total += t.numel() * bytes_per_weight(qt)
        else:
            total += t.numel() * bits_per_elem / 8
    return total


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a Voodoo mixed-precision checkpoint (torch PPL + KL vs the BF16 teacher)")
    parser.add_argument("--checkpoint", required=True, help="Voodoo base checkpoint (.pt)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--ud_base_checkpoint", default=None, help="Optional Unsloth Dynamic GGUF-converted base for comparison.")
    parser.add_argument("--base_checkpoint", default=None, help="Optional HF-compatible .pt state dict to load into the model architecture from --model. Useful for MTP-converted bases.")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data_dir", default="data/fineweb_qwen")
    parser.add_argument("--data_name", default="val_tokens.pt")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--quant_assignments", default=None, help="Optional JSON path for assignments when the checkpoint does not contain them.")
    parser.add_argument("--output_dir", default=None, help="Directory to write the eval JSON. Defaults to the checkpoint's directory.")
    return parser


def run(args):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading Voodoo checkpoint from {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    quant_assignments = checkpoint.get("quant_assignments")
    if quant_assignments is None and args.quant_assignments is not None:
        print(f"Loading quant assignments from {args.quant_assignments} ...")
        quant_assignments = json.loads(Path(args.quant_assignments).read_text())
        print(f"  loaded {len(quant_assignments)} assignments")
    imatrices = checkpoint.get("imatrices", {})
    source_dtype_info = checkpoint.get("source_dtype_info", {})

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Building validation dataloader ...")
    dataloader = build_val_dataloader(tokenizer, args.seq_len, args.batch_size, args.data_dir, args.data_name)

    print("Loading BF16 teacher ...")
    if args.base_checkpoint is not None:
        print(f"  loading teacher config from {args.model} and weights from {args.base_checkpoint}", flush=True)
        teacher_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        teacher = AutoModelForCausalLM.from_config(teacher_config, trust_remote_code=True).to(dtype)
        base_ckpt = torch.load(args.base_checkpoint, weights_only=True, map_location="cpu")
        teacher.load_state_dict(base_ckpt["model_state_dict"], strict=False)
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=dtype,
        )
    teacher.to(device)
    teacher.config.use_cache = False
    teacher.eval()

    results = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "dynamic_quant": None,
        "ud_base": None,
    }

    print("Loading Voodoo mixed-precision model ...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(dtype)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    if quant_assignments is not None:
        model = apply_quant_assignments(model, quant_assignments, imatrices)
        model.to(device)
    model.config.use_cache = False

    if not args.no_compile:
        print("Compiling model ...")
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = torch.compile(layer, mode="default", dynamic=False)

    print("Evaluating learned mixed-precision model ...")
    dq_metrics = evaluate_with_kl(
        model, teacher, dataloader, device, max_steps=args.max_steps, temperature=args.temperature
    )
    if quant_assignments is not None:
        dq_bytes = compute_total_bytes(checkpoint["model_state_dict"], quant_assignments, dtype)
        dq_metrics["total_bytes_mb"] = dq_bytes / 1e6
    dq_metrics["compression_ratio"] = checkpoint.get("compression_ratio")
    dq_metrics["target_bits"] = checkpoint.get("target_bits")
    byte_str = f" total_MB={dq_metrics['total_bytes_mb']:.2f}" if "total_bytes_mb" in dq_metrics else ""
    print(
        f"Dynamic quant: PPL={dq_metrics['ppl']:.3f} NLL={dq_metrics['nll']:.4f} "
        f"KL={dq_metrics['kl']:.4f}{byte_str}"
    )
    results["dynamic_quant"] = dq_metrics

    if args.ud_base_checkpoint is not None:
        print(f"Loading Unsloth Dynamic base from {args.ud_base_checkpoint} ...")
        ud_ckpt = torch.load(args.ud_base_checkpoint, weights_only=True, map_location="cpu")
        ud_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(dtype)
        ud_model.load_state_dict(ud_ckpt["model_state_dict"], strict=False)
        ud_model.to(device)
        ud_model.config.use_cache = False

        if not args.no_compile:
            for i, layer in enumerate(ud_model.model.layers):
                ud_model.model.layers[i] = torch.compile(layer, mode="default", dynamic=False)

        print("Evaluating Unsloth Dynamic base ...")
        ud_metrics = evaluate_with_kl(
            ud_model, teacher, dataloader, device, max_steps=args.max_steps, temperature=args.temperature
        )
        print(
            f"Unsloth Dynamic base: PPL={ud_metrics['ppl']:.3f} NLL={ud_metrics['nll']:.4f} "
            f"KL={ud_metrics['kl']:.4f}"
        )
        results["ud_base"] = ud_metrics

    checkpoint_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{checkpoint_path.stem}.eval.json"
    else:
        output_path = checkpoint_path.with_suffix(".eval.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {output_path}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir:
        model_dir = Path(args.output_dir)
    else:
        model_dir = Path(args.checkpoint).parent
    with log_stage(
        stage="eval_pytorch",
        model_dir=model_dir,
        script=Path(__file__).name,
        args=vars(args),
    ):
        run(args)


if __name__ == "__main__":
    main()
