#!/usr/bin/env bash
# Reference launcher: Qwen3.5-0.8B Voodoo45 on a single GPU.
# The minimal happy path — every flag here is a good default for 0.5B–8B.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen3.5-0.8B-Base
SIZE=45
OUT=checkpoints/Qwen3.5-0.8B/Voodoo${SIZE}
mkdir -p "$OUT" logs

.venv/bin/voodoo train \
    --model "$MODEL" \
    --compression_ratio 0.${SIZE} \
    --max_steps 100 --lr 0.5 \
    --size_weight 100.0 --size_tolerance 0.02 \
    --seq_len 512 --batch_size 1 \
    --data_dir data/qwen35-0.8b --data_name train_tokens.pt \
    --output_dir "$OUT" \
    --output_name Qwen3.5-0.8B-Voodoo${SIZE}.pt \
    --log_file logs/qwen35_voodoo${SIZE}_train.jsonl \
    --device cuda --dtype bfloat16 --no_compile
