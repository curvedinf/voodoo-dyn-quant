#!/usr/bin/env bash
# Reference launcher: Qwen3.5-0.8B Voodoo45 on a single GPU.
# The minimal happy path — every flag here is a good default for 0.5B–8B.
#
# Recipe follows the 0.8B PTQR iteration campaign best practices
# (see skills/train/SKILL.md): --ptqr routing (beat the soft-mixture
# baseline at matched budget; ST-Gumbel hard-fraction is rejected),
# lr 0.25 (the stable end of the lr ladder; avoid 0.5), distill_weight 2.0,
# grad_accum 1, 50 steps (100 gave no benefit).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen3.5-0.8B-Base
SIZE=45
OUT=checkpoints/Qwen3.5-0.8B/Voodoo${SIZE}
mkdir -p "$OUT" logs

.venv/bin/voodoo train \
    --model "$MODEL" \
    --compression_ratio 0.${SIZE} \
    --ptqr \
    --max_steps 50 --lr 0.25 --distill_weight 2.0 --grad_accum_steps 1 \
    --size_weight 100.0 --size_tolerance 0.02 \
    --seq_len 512 --batch_size 1 \
    --data_dir data/qwen35-0.8b --data_name train_tokens.pt \
    --output_dir "$OUT" \
    --output_name Qwen3.5-0.8B-Voodoo${SIZE}.pt \
    --log_file logs/qwen35_voodoo${SIZE}_train.jsonl \
    --device cuda --dtype bfloat16 --no_compile
