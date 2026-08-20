# Voodoo Dynamic Quantization

**Learned per-tensor mixed-precision quantization for any LLM —
llama.cpp-exact by construction.**

Voodoo replaces hand-tuned quantization recipes with gradient descent: every
weight tensor in the model votes, through its own gates, on how many bits it
deserves, under a global size budget and full-model distillation loss. The
result is a per-tensor quant assignment exported as a normal GGUF that runs
in stock llama.cpp. Voodoo's [methodology reduces perplexity by up to 89.5%](https://voodooquant.com).

```
   teacher (frozen, BF16/Q8)          student = same model, weights replaced by
   ─────────────────────────          Σ pₖ · Wₖ   (softmax over K quant candidates)
        │ logits                              │ logits
        └──────────────┬──────────────────────┘
                       ▼
     L = CE + λ·KL(teacher ‖ student) + size-budget dead-zone penalty
                       ▼
     argmax gate per tensor → quant assignment → GGUF (llama.cpp-exact)
```

- **Architecture-agnostic.** If `transformers` can load it, Voodoo can
  quantize it — attention, SSM/linear-attention hybrids, gated-shortconv
  hybrids (LFM2/LFM2.5), tied embeddings, MoE-free any-shape Linears
  (zero-padded internally). Arch specifics (TP sharding policy, GGUF tensor
  mapping) live in small, detached adapters under `voodoo_quant/arch/`.
- **Hardware-agnostic.** CUDA or ROCm, single GPU, layer-wise sharding, or
  full tensor parallelism. Vendor specifics live in one tidy module.
- **Exact.** Every quantized byte is produced by llama.cpp's own C++
  quantizer via `libggml-base` — no train/deploy quantization gap, ever.

How it works: **[WHITEPAPER.md](WHITEPAPER.md)** · What to do: the skills
below · Agent conventions: [AGENTS.md](AGENTS.md)

### Is your organization looking for help?

I'm available for hire. Reach out at [https://voodooquant.com](https://voodooquant.com).

---

## Quickstart

```bash
make bootstrap        # venv + package + llama.cpp libggml (nice defaults)
make doctor           # verify GPU + torch + libggml resolve
voodoo data --model Qwen/Qwen3.5-0.8B-Base --output_dir data/qwen35-0.8b
configs/qwen35_08b_voodoo45.sh    # single-GPU train → Voodoo45
```

`make bootstrap` clones llama.cpp into `third_party/` and builds
`libggml-base` (CPU-only build — no GPU toolchain needed for quantization).
Already have a llama.cpp checkout? Set
`VOODOO_GGML_LIB=/path/to/libggml-base.so` and skip that step.

---

## Using this repo with an agent

The workflow is encoded as **skills** in `skills/` — install, data, train,
export, eval. If you use an agentic coding tool (Claude Code, Kimi, Cursor,
etc.), point it at this repo and it will drive the pipeline through the
skills; base instructions live in [AGENTS.md](AGENTS.md).

| You want | Skill | Command essence |
|---|---|---|
| Set up from clean checkout | `skills/install/SKILL.md` | `make bootstrap` |
| Prepare calibration data | `skills/data/SKILL.md` | `voodoo data --model <id>` |
| Create base `.pt` + Q8 teacher | `skills/train/SKILL.md` | `voodoo make-teacher --model_dir <hf-dir>` |
| Train a Voodoo size | `skills/train/SKILL.md` | `voodoo train --compression_ratio 0.45 ...` |
| Export GGUF | `skills/export/SKILL.md` | `voodoo export --checkpoint ... --ref-gguf ...` |
| Evaluate PPL/KL | `skills/eval/SKILL.md` | `voodoo eval ...` + `llama-perplexity` |

## CLI overview

```
voodoo train        gate training + bake (all trainer flags)
voodoo tp           torchrun wrapper with hardware env defaults applied
voodoo export       bake assignment → GGUF (canonical Voodoo naming)
voodoo eval         torch PPL + KL vs frozen teacher
voodoo data         tokenize calibration/eval corpus
voodoo make-teacher create a BF16 base .pt + Q8_0 teacher from an HF checkpoint dir
voodoo precache     pre-build the candidate cache for a large model
voodoo doctor       hardware/libggml/environment report
```

The pipeline, end to end:

```
data → train → export → eval
        └ journal: partial.pt (crash recovery / resume)
```

## The four ideas that make it work

1. **Gates, not heuristics.** Each selectable tensor holds K candidates
   (llama.cpp-exact dequantized copies); a temperature-annealed softmax mixes
   them during forward. Gradients through the mixture are per-tensor
   sensitivity estimates obtained for all tensors at once.
2. **A dead-zone size budget.** The size penalty is zero inside tolerance —
   tensors trade bytes freely; the quadratic only pulls back when the
   assignment misses the target. Sizes are percentages of the 8-bit original
   (`Voodoo45` = 45% of 8-bit).
3. **Exactness as a contract.** Candidates, bake, and export all call
   llama.cpp's quantizer. What the gates optimize is what the runtime serves.
4. **Menus with fine rungs.** The single biggest measured failure mode of
   mixed-precision optimizers is a coarse candidate ladder: attention with no
   mid rungs "buys up" a +1.3 bpw cliff and hoards the budget the MLP needs.
   Voodoo's defaults ship per-role menus and post-hoc guardrails
   (`--tensor_upgrades`, negative levels allowed) informed by direct
   attribution experiments (WHITEPAPER §6).

## Recipe: Qwen3.8-27B Voodoo30 on 4 GPUs (TP-4)

A complete, evidence-tuned reference for a 27B hybrid SSM/attention model on
four GPUs. The launcher is [`configs/qwen38_27b_voodoo30_tp4.sh`](configs/qwen38_27b_voodoo30_tp4.sh);
this section explains each choice so you can adapt it to your model.

```bash
# One-time: data (8k context — train at the length you serve)
voodoo data --model <model-id> --seq_len 8192 --output_dir data/qwen38-longctx

# One-time: base .pt + Q8 teacher from the HF checkpoint dir
# (multimodal wrappers: add --strip_prefix model.language_model.)
voodoo make-teacher --model_dir ./Qwen3.8-27B --output_dir checkpoints/Qwen3.8-27B

# Launch (TP-4; voodoo tp applies allocator/NCCL defaults for your vendor)
voodoo tp --nproc 4 -- \
    --model ./Qwen3.8-27B \
    --base_checkpoint checkpoints/Qwen3.8-27B/qwen38_27b_lm_base.pt \
    --text_model_class qwen3_5_text \
    --teacher_quant Q8_0 --teacher_checkpoint .../qwen38_27b_lm_teacher_q8_0.pt \
    --compression_ratio 0.30 --budget_reduction 0.03 \
    --tensor_upgrades '[{"pattern":"mlp\\.(gate|up|down)_proj$","levels":1},
                        {"pattern":"linear_attn|self_attn","levels":-1}]' \
    --seq_len 8192 --batch_size 1 --max_steps 100 \
    --lr 0.5 --size_weight 100.0 --size_tolerance 0.02 \
    --lazy --imatrix .../imatrix_v50.gguf \
    --candidate_types      Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --tensor_parallel 4 --gradient_checkpointing \
    --data_dir data/qwen38-longctx --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.8-27B/Voodoo30 \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --device cuda:0 --dtype bfloat16 --partial_save_interval 1 --no_compile
```

| Flag | Why |
|---|---|
| `--tensor_parallel 4` (via `voodoo tp`) | Megatron-style sharding of every parallelizable Linear + vocab-parallel head; exact cross-rank log-sum-exp CE/KL. Requires `--base_checkpoint` (full mmap'd weights). |
| `--teacher_quant Q8_0` | Q8 teacher halves teacher VRAM; sharded identically to the student and reloaded per step from mmap. |
| per-role candidate menus | Attention and MLP see different ladders — the fix for the measured budget-hoarding failure (WHITEPAPER §6.1). |
| `--tensor_upgrades` (+1 MLP / −1 attention) | Post-hoc guardrail: no big MLP left at the menu floor, no attention hoarding the ceiling; `--budget_reduction 0.03` funds it. Negative levels (down) are supported. |
| `--seq_len 8192` | SSM/hybrid gradients starve below 512; long context punishes MLP under-provision — train at serving length. |
| `--lazy` | Dequantize candidates on the fly — the memory regime that fits 27B on 4 GPUs. Smaller models: leave it off (faster). |
| `--partial_save_interval 1` | Journal every step; the launcher auto-resumes from it. |
| `--no_compile` | Skip torch.compile warm-up (recommended until the config is stable). |

**Adapting to your model:** keep everything structural; change `--model`,
checkpoints, data paths, and the `tensor_upgrades` regexes to your tensor
names. Drop `--text_model_class` if your base checkpoint is a full
`AutoModelForCausalLM` state dict. The per-role menus are worth keeping for
any architecture with a big MLP family — the failure mode they fix is not
Qwen-specific.

Memory regimes at 27B/TP-4: `--lazy` + gradient checkpointing + Q8 teacher is
the working set that fits 4×32 GB. The first run builds the candidate cache
(hours of pure-CPU ggml quantization, parallel across cores — this is the
exactness guarantee); pre-build it with `voodoo precache` if you will run
multiple sizes.

## Configuration notes

- **Sizes**: `--compression_ratio 0.NN` (or `--target_bits`). Standard sizes
  25–80 in steps of 5. Exported GGUFs carry a curated quant badge
  (`Qwen3.5-0.8B.Voodoo45_IQ2_M.gguf`) so HF model cards index them.
- **Parallelism**: `voodoo tp --nproc N` for tensor parallelism (torchrun);
  `--device_map cuda:0,cuda:1,...` for layer-wise sharding without collectives.
- **Env**: `voodoo tp` applies vendor defaults (allocator, NCCL, worker
  caps). Manual torchrun: source them from `voodoo doctor` output.
- **Env vars**: `VOODOO_GGML_LIB`, `VOODOO_QUANT_WORKERS`,
  `VOODOO_FINALIZE_WORKERS`, `VOODOO_TINY_RECIPE=1` (fast tests).

## Project layout

```
voodoo_quant/        the package (core → hardware → arch adapters)
  ggml.py            llama.cpp-exact quantization bridge
  layers.py          MixedQuant gates + candidate cache
  parallel.py        TP sharding primitives (hardware-agnostic)
  arch/              DETACHED architecture adapters (base + registry; qwen, lfm2)
  quants/            GPU dequant kernels
  hardware/          vendor detection + env defaults + ROCm flash backend
  training/          trainer, Q8 teacher
  tools/             data / precache / export / evaluate / make-teacher
skills/              agentic workflow skills (the intended UX)
configs/             reference launchers (0.8B single-GPU, 27B TP-4)
tests/               CPU-runnable unit tests
```

### Adding an architecture

Arch specifics never touch the core. Drop an adapter in `voodoo_quant/arch/`
subclassing `ArchAdapter` and `@register` it — you provide three optional
policies: `shard_layer` (TP sharding of one decoder layer, using the
primitives from `parallel.py`), `gguf_name`/`transpose_embedding` (HF →
llama.cpp tensor mapping), and `attention_segments` (which tensor families
follow the attention-side candidate menu). Without a registered adapter,
single-GPU and layer-wise training still work; TP refuses to run rather than
shard blindly. Shipped adapters: qwen3.5/3.6/3.8 hybrids and LFM2/LFM2.5
(shortconv+GQA hybrids, incl. the transposed `token_embd` GGUF layout and the
fused `[B|C|x]` shortconv `in_proj` grouped-row TP sharding).

## License

MIT (see [LICENSE](LICENSE)).
