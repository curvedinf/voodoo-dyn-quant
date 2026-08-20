# Skill: train

Train a Voodoo mixed-precision assignment for a model. Output: a baked
checkpoint (`<slug>-Voodoo{NN}.pt`), a `quant_assignments.json` sidecar, and a
`partial.pt` journal.

## 0. Preconditions

- `voodoo doctor` healthy (GPU visible, libggml found).
- Calibration data exists (`data/<model>/train_tokens.pt`) — else run the
  [data skill](../data/SKILL.md) first.
- Check GPU headroom before launching (`voodoo doctor`, `rocm-smi`/`nvidia-smi`).
- Pick the size: `Voodoo{NN}` = NN% of the 8-bit original
  (`--compression_ratio 0.NN`). Standard sizes: 25–80 step 5.

## 1. Small/medium model, single GPU

```bash
.venv/bin/voodoo train \
    --model Qwen/Qwen3.5-0.8B-Base \
    --compression_ratio 0.45 \
    --max_steps 100 --lr 0.5 --size_weight 100.0 --size_tolerance 0.02 \
    --seq_len 512 --batch_size 1 \
    --data_dir data/qwen35-0.8b --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.5-0.8B/Voodoo45 \
    --output_name Qwen3.5-0.8B-Voodoo45.pt \
    --device cuda --dtype bfloat16 --no_compile
```

Defaults that just work for 0.5B–8B: non-lazy candidates, batch 1, 100 steps.
First run quantizes the candidate cache (slow init is normal — watch the
progress line); later sizes start in seconds.

## 2. Large model, tensor parallel

Prerequisites for large models: a full-weights base `.pt` (and optionally a
Q8 teacher), created once from any HF checkpoint directory:

```bash
voodoo make-teacher --model_dir ./<model-dir> \
    --strip_prefix model.language_model. \   # only for multimodal wrappers
    --output_dir checkpoints/<family>
```

Then launch with `voodoo tp` (applies hardware env defaults, then torchrun):

```bash
.venv/bin/voodoo tp --nproc 4 -- \
    --model <model-or-local-config-dir> \
    --base_checkpoint <full-weights.pt> \
    --compression_ratio 0.30 --budget_reduction 0.03 \
    --tensor_upgrades '[{"pattern":"mlp\\.(gate|up|down)_proj$","levels":1},{"pattern":"linear_attn|self_attn","levels":-1}]' \
    --seq_len 8192 --batch_size 1 --grad_accum_steps 1 \
    --max_steps 100 --lr 0.5 --size_weight 100.0 --size_tolerance 0.02 \
    --lazy --imatrix <imatrix.gguf> \
    --candidate_types Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --tensor_parallel 4 \
    --gradient_checkpointing \
    --data_dir data/qwen38-longctx --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.8-27B/Voodoo30 \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --device cuda --dtype bfloat16 \
    --partial_save_interval 1 --no_compile
```

### Evidence-based defaults (from the divergence treatise)

- **Per-role menus with fine rungs both ways.** Attention and MLP get separate
  candidate lists. A missing mid rung makes "buy up" a +1.3 bpw cliff and the
  attention family hoards the MLP's budget — the single biggest measured
  failure mode.
- **Post-hoc guardrails**: `--tensor_upgrades` with `levels: -1` (down) is
  supported; `--budget_reduction 0.03` funds the MLP +1 rung without busting
  the target. No `--force_quant` pins (measured dead weight).
- **Sensitivity warm-start** (`--warm_start`): seed gates from a measured
  layout (real-activation per-tensor sensitivity → greedy knapsack) instead
  of zeros; the table persists to `sensitivity.pkl`. Recommended for large
  models where the cold-start mis-allocation failure is the dominant risk.
- **Knapsack polish** (`--polish`): after argmax, budget-aware one-rung swaps
  ranked by the measured table. Requires `--warm_start`'s table (persists to
  `sensitivity.pkl`); skips itself otherwise.
- **ST-Gumbel hardening** (`--st_gumbel_fraction`, e.g. 0.5): each layer
  forwards a one-hot Gumbel sample of its gates with that per-step probability
  (deployed-model behavior) while gradients stay soft/exact. Off by default;
  `--st_gumbel_tau` tunes sample sharpness.
- **Long context**: train at the seq_len you serve (8192); the quality gap of
  a bad allocation widens with context (5.3% → 9.2% from 512 → 8k).
- **Keep your own imatrix** (`--imatrix`); calibration volume is worth only
  ~0.2% PPL — assignment dominates ~15×.

## 3. Launch patterns

- **Layer-wise sharding** (no torchrun, heterogeneous GPUs):
  add `--device_map cuda:0,cuda:1,...` instead of TP.
- **Q8 teacher for huge models**: `--teacher_quant Q8_0
  --teacher_checkpoint <q8.pt>` halves teacher VRAM.
- **Auto-resume**: TP launchers resume from `output_dir/partial.pt` when
  present (see configs/qwen38_27b_voodoo30_tp4.sh for the reference script).

## 4. Monitoring / recovery

- JSONL training log: `--log_file` (loss, tau, size_mb per step).
- Journal: `partial.pt` every `--partial_save_interval` optimizer steps.
- Crash during bake → rerun same command with
  `--finalize_from_partial <outdir>/partial.pt`.
- Interrupted training → `--resume_from_partial <outdir>/partial.pt`.
- Do not interrupt the init/candidate-quantization phase (it is quantizing,
  not hung). Kill `-USR1 <pid>` dumps a heap census to the log.

## 5. Verify before declaring done

- Final log line shows total bytes within `size_tolerance` of target.
- `quant_assignments.json` exists and covers every targeted tensor.
- For release: export (next skill) and sanity-check the GGUF size within a
  few % of target.

## Supported architectures (adapters)

Adapters live in `voodoo_quant/arch/` — qwen3.5/3.6/3.8 hybrids and
LFM2/LFM2.5 (LiquidAI shortconv+GQA hybrids). Anything `transformers` loads
trains single-GPU/layer-wise without an adapter; TP requires one (it refuses
rather than shard blindly). Adding one is a single file: subclass
`ArchAdapter`, `@register` it, define `shard_layer` + `gguf_name` +
`attention_segments`, and add mapping/TP-reconstruction tests to
`tests/test_arch.py` (see `test_lfm2_tp_reconstruction` for the pattern —
shard two ranks, prove the math reconstructs the unsharded forward).
