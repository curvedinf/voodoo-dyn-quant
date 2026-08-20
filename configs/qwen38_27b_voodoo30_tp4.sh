#!/usr/bin/env bash
# Reference launcher: Qwen3.8-27B Voodoo30 on 4x GPUs (tensor parallel).
#
# This is the recipe distilled from the Voodoo50-vs-UD-IQ4_XS divergence
# attribution (see WHITEPAPER.md §6): per-role candidate menus with fine rungs
# both directions, a reversed post-hoc guardrail (MLP one rung UP, attention
# one rung DOWN), no force_quant pins, own imatrix, seq_len 8192.
# V30 final: 50 gate-learning steps (per SDGraft, 2026-08-20 — 100 steps gave
# no benefit at this scale either).
#
# Prerequisites (one-time):
#   1. make bootstrap                     # venv + libggml
#   2. voodoo data --model <model-id> ... # data/qwen38-longctx/train_tokens.pt (seq 8192+)
#   3. Full BF16 weights as an HF state dict + Q8 teacher (one command):
#      voodoo make-teacher --model_dir ./Qwen3.8-27B \
#          --output_dir checkpoints/Qwen3.8-27B
#      (produces qwen38_27b_base.pt + qwen38_27b_teacher_q8_0.pt —
#       adjust BASE_CKPT/TEACHER_Q8 below to the filenames it writes)
#   4. Optional but recommended: imatrix GGUF.
#   5. Pre-build the candidate cache (optional; init also builds it inline):
#      voodoo precache --checkpoint <base.pt> --attention_candidates ... (see skills/train)
#
# Adaptation: this script is Qwen3.8-shaped (SSM+attention hybrid). For a
# different architecture keep everything, drop --text_model_class if your base
# checkpoint is a full AutoModelForCausalLM state dict, and adjust the
# tensor_upgrades patterns to your tensor names (mlp gate/up/down + attention).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=checkpoints/Qwen3.8-27B/Voodoo30
MODEL_DIR=./Qwen3.8-27B            # local config dir or HF id
BASE_CKPT=checkpoints/Qwen3.8-27B/Qwen3.8-27B_base.pt
TEACHER_Q8=checkpoints/Qwen3.8-27B/Qwen3.8-27B_teacher_q8_0.pt
IMATRIX=checkpoints/Qwen3.8-27B/imatrix_v50.gguf
DATA=data/qwen38-longctx
mkdir -p "$OUT" logs

# Hardware env defaults (CUDA or ROCm; voodoo tp applies these automatically
# when you use `voodoo tp`, shown here explicitly for the raw torchrun path).
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_PARALLEL_COMPILE=1
export VOODOO_FINALIZE_WORKERS=8
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True   # ROCm
export NCCL_P2P_DISABLE=1                                # ROCm (unset on NVLink)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # CUDA (harmless on ROCm)

# Auto-resume from this run's own journal if present.
RESUME_ARGS=""
if [ -s "$OUT/partial.pt" ] && [ -s "$OUT/partial.meta.json" ]; then
    PREV=$($PY -c "import json;print(json.load(open('$OUT/partial.meta.json'))['completed_optimizer_steps'])" 2>/dev/null || echo 0)
    if [ "$PREV" -gt 0 ] && [ "$PREV" -lt 50 ]; then
        RESUME_ARGS="--resume_from_partial $OUT/partial.pt"
        echo "launcher: resuming from journal at optimizer step $PREV" >> "$OUT/train_tp.log"
    fi
fi

# Post-hoc guardrail (applied after gate training, before the bake):
#   - MLP tensors pinned to the menu floor move one rung UP
#   - attention tensors pinned to the menu ceiling move one rung DOWN
#     (measured worthless at high precision; frees ~1 GB for the MLP band)
UPGRADES='[
  {"pattern":"mlp\\.(gate|up|down)_proj$","levels":1},
  {"pattern":"linear_attn|self_attn","levels":-1}
]'

exec "$PY" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29600 \
    -m voodoo_quant.cli train \
    --model "$MODEL_DIR" \
    --base_checkpoint "$BASE_CKPT" \
    --text_model_class qwen3_5_text \
    --teacher_quant Q8_0 \
    --teacher_checkpoint "$TEACHER_Q8" \
    --compression_ratio 0.30 \
    --budget_reduction 0.03 \
    --tensor_upgrades "$UPGRADES" \
    --seq_len 8192 \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --max_steps 50 \
    --lr 0.5 \
    --size_weight 100.0 \
    --size_tolerance 0.02 \
    --lazy \
    --imatrix "$IMATRIX" \
    --candidate_types Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --tensor_parallel 4 \
    --gradient_checkpointing \
    --data_dir "$DATA" \
    --data_name train_tokens.pt \
    --output_dir "$OUT" \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --log_file logs/qwen38_voodoo30_train.jsonl \
    --device cuda:0 \
    --dtype bfloat16 \
    --partial_save_interval 1 \
    $RESUME_ARGS \
    --no_compile \
    >> "$OUT/train_tp.log" 2>&1
