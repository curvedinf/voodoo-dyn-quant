#!/usr/bin/env python3
"""
Prepare calibration/eval token data for a target model.

Downloads a small streaming sample of a dataset (default: Fineweb) and
tokenizes it with the model tokenizer. Produces train/val token files that
can be used by both the calibration/training scripts and the evaluation
tool. The split is deterministic and a held-out val set is kept separate
from train.

Usage:
    voodoo data \
        --model Qwen/Qwen3.5-0.8B-Base \
        --n_samples 10000 \
        --val_ratio 0.05 \
        --seq_len 2048 \
        --output_dir ./data/fineweb_qwen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Configure persistent Triton/Inductor caches before any torch/triton usage.
import voodoo_quant.cache  # noqa: F401

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class TokenizedTensorDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over a pre-tokenized 1-D tensor."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.tokens = tokens

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1].long(), chunk[1:].long()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare calibration/eval token data for a model")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", default="./data/fineweb_qwen")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading {args.dataset} subset={args.subset}, split={args.split} (streaming) ...")
    ds = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=True,
    )

    print(f"Tokenizing first {args.n_samples} documents ...")
    doc_tokens: list[list[int]] = []
    for i, doc in enumerate(tqdm(ds, total=args.n_samples)):
        if i >= args.n_samples:
            break
        text = doc.get(args.text_key, "")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Insert EOS between documents to avoid cross-document n-grams.
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        doc_tokens.append(ids)

    total_tokens = sum(len(d) for d in doc_tokens)
    print(f"Total documents: {len(doc_tokens):,}, total tokens: {total_tokens:,}")

    # Deterministic shuffle at the document level, then split.
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(doc_tokens), generator=rng)
    doc_tokens = [doc_tokens[i] for i in perm.tolist()]

    n_val_docs = max(1, int(round(len(doc_tokens) * args.val_ratio)))
    val_doc_tokens = doc_tokens[:n_val_docs]
    train_doc_tokens = doc_tokens[n_val_docs:]

    def concat_and_chunk(docs: list[list[int]]) -> torch.Tensor:
        flat = [tid for doc in docs for tid in doc]
        n = (len(flat) // args.seq_len) * args.seq_len
        return torch.tensor(flat[:n], dtype=torch.long)

    val_tokens = concat_and_chunk(val_doc_tokens)
    train_tokens = concat_and_chunk(train_doc_tokens)

    print(f"Train tokens: {len(train_tokens):,} ({len(train_tokens) // args.seq_len} sequences)")
    print(f"Val tokens:   {len(val_tokens):,} ({len(val_tokens) // args.seq_len} sequences)")

    train_path = out_dir / "train_tokens.pt"
    val_path = out_dir / "val_tokens.pt"
    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "subset": args.subset,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
    }
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved train tokens to {train_path}")
    print(f"Saved val tokens to {val_path}")


if __name__ == "__main__":
    main()
