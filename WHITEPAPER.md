# Voodoo Dynamic Quantization — Whitepaper

How learned per-tensor mixed-precision quantization works, why it is
llama.cpp-exact by construction, and what the divergence-attribution
experiments taught us about where quantization quality actually comes from.

Companion documents: `README.md` (usage), `AGENTS.md` (agent instructions).

---

## 1. The problem

A quantized LLM is not one decision but thousands: every weight tensor in the
model independently chooses a storage format — IQ1_S at ~1.6 bits/weight up to
Q8_0 at 8.5 — and the choices interact through a single global budget.
Static recipes (Unsloth Dynamic 2.0, GPTQ-style uniform targets) fix these
choices with hand-tuned heuristics: importance from activation statistics,
a size target per bucket, and a fixed depth profile.

**Voodoo replaces the heuristic with gradient descent.** The model itself,
distilling its own outputs, votes on how many bits every tensor deserves.

## 2. The core mechanism

### 2.1 Candidate sets and gates

For every selectable tensor (all `nn.Linear` weights and the embedding —
non-divisible input dims are zero-padded internally), Voodoo pre-computes K
candidate quantizations from a candidate menu, e.g.:

```
IQ1_S  IQ1_M  IQ2_XXS  IQ2_XS  IQ2_S  IQ3_XXS  IQ3_S  IQ4_XS  Q4_K  Q5_K  Q6_K  Q8_0
```

Each candidate is a full dequantized copy of the tensor *as llama.cpp would
store and reconstruct it* (§3). The training-time layer attaches K positive
scalar **gates** to the tensor; the effective weight during the forward pass is

```
W_eff = Σ_k  p_k · W_k        where  p = softmax(gates / τ)
```

`τ` is a temperature annealed geometrically from 1.0 → 0.01 over training.
Early on, all candidates blend (smooth gradient signal to every gate); as `τ`
falls, the softmax concentrates and the mixture approaches a hard argmax.

Optionally, **straight-through Gumbel hardening** (`--st_gumbel_fraction`,
default 0 = off) has each layer forward, with that per-step probability, a
one-hot Gumbel-softmax sample of its gates instead of the soft mixture — the
exact forward behavior of the deployed argmax model — while backward still
flows through the soft softmax (straight-through estimator, so gate gradients
stay exact). Sampling rather than argmax keeps exploration alive: an early
wrong choice still receives gradient and can be corrected on later steps.
`--st_gumbel_tau` controls the sample sharpness (lower = more argmax-like).

### 2.2 The loss

The only trainable parameters are the gates (~one scalar per tensor per
candidate). Training minimizes:

```
L = CE(student, tokens) + λ_KL · KL(teacher ‖ student) + λ_size · relu(|bytes/target − 1| − tol)²
```

- **Teacher**: the same model, frozen, at full (or Q8_0) precision. The KL
  term is the distillation signal that tells the gates *what quantization
  error costs* in output space, not in weight space.
- **Size term**: a dead-zone budget penalty. Inside the tolerance band the
  penalty is zero, so the optimizer is free to trade bytes between tensors
  without being nudged; outside it, the quadratic pulls the assignment back.
  The denominator is the *8-bit* model size — a Voodoo50 model is 50% of the
  8-bit original, with tied weights counted once.

After training each tensor is hard-assigned to its argmax candidate,
re-quantized from the original weights, dequantized back, and saved as a
normal HF state dict plus a `quant_assignments.json` sidecar.

### 2.3 Why this works: gradients through the mixture

Because `W_eff` is a convex-ish combination in *probability* space, the
gradient of the loss with respect to gate `k` is (up to the softmax jacobian)
the loss difference between using candidate `k` and the current blend —
measured on real data through the full model. That is exactly the quantity a
sensitivity analysis would try to estimate by ablation, obtained here for
every tensor simultaneously in one backward pass.

A custom autograd function mixes candidates one at a time on the compute
device, so the live GPU footprint stays at roughly one candidate per layer
regardless of menu size.

## 3. Exactness: bit-identical to llama.cpp

There is no approximate Python quantizer anywhere in this stack. Every
candidate, every final assignment, and every exported GGUF block is produced
by llama.cpp's own C++ routines (`quantize_iq4_xs`,
`dequantize_row_q2_K`, …) loaded from `libggml-base` via ctypes.

Consequences:

- The number the gates optimize is the number the runtime serves. No
  train/deploy quantization gap, ever.
- The exported GGUF's perplexity under `llama-perplexity` is explained
  entirely by the assignment, not by re-quantization drift.
- Quantization is CPU-side by design: the iterative IQ solvers
  (`make_qkx3`, `make_qp`, lattice search) depend on ggml's exact
  sequential float32 accumulation order, so a GPU port would not be
  bit-identical. The ctypes calls release the GIL, so candidate building
  parallelizes near-linearly across CPU cores.

The same guarantee flows through TP: because llama.cpp block quantization is
row-independent (and column-block-aligned slices keep block boundaries),
slicing a full-tensor candidate to a rank's shard is bit-identical to
quantizing the shard directly. One shared candidate cache, full-tensor keys.

## 4. Architecture-agnostic by construction

Voodoo touches only `nn.Linear` and `nn.Embedding` modules, discovered by
recursion over any `transformers` model. It has no knowledge of:

- attention vs SSM/linear attention (per-role candidate menus are pure
  name-pattern grouping — see §6.1),
- layer counts, head layouts, or tied-embedding topology (handled generically
  by storage-pointer dedup in the budget),
- vendor or device count (§5).

Anything `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
can load, Voodoo can quantize. Vendor-specific and model-specific concerns
live in separate adapter modules, never in the core.

## 5. Parallelism and hardware

Two independent scaling axes, both hardware-agnostic (CUDA or ROCm):

1. **Layer-wise sharding** (`--device_map cuda:0,cuda:1,...`): decoder layers
   round-robin across devices; activations hop devices per layer. The teacher
   shares the mapping. Simple, works for any model, no collectives.
2. **Full tensor parallelism** (`--tensor_parallel N` under `torchrun`):
   Megatron-style column/row sharding of every parallelizable Linear,
   vocab-parallel embeddings and head, exact cross-rank log-sum-exp for the
   full-vocab CE/KL, and SUM-synced gate gradients. The Q8 teacher is
   sharded identically and reloaded per step from the mmap'd source, keeping
   the teacher off VRAM between forwards.

The loss path uses per-token-chunk head backwards with one aggregated model
backward — CE and KL are token sums, so the decomposition is exact and peak
memory is one chunk of `[chunk, V]` logits instead of the whole sequence.

Post-training resilience: a rolling journal (`partial.pt`) snapshots gates +
argmax assignments every N steps, and `--finalize_from_partial` re-runs only
the bake after a crash. `--resume_from_partial` continues training.

## 6. What the experiments taught us

The following is distilled from controlled graft-and-retest attribution on a
27B hybrid SSM/attention model (Voodoo50 vs Unsloth UD-IQ4_XS, six
frankenstein builds; see the source treatise). These findings shape the
defaults.

### 6.1 Budget allocation dominates everything

The entire quality deficit of a mis-trained Voodoo run was concentrated in
mid/late-stack MLP under-provision. Attention precision was measured at
**zero** PPL effect (±0.001 across 24 tensors), calibration-data quality was
worth ~0.2%, and an embedding upgrade was dead weight. **Assignment is ~15×
more important than calibration.**

The root cause was menu structure, not the optimizer: attention's menu had no
mid rungs, so any "buy up" jumped a full +1.3 bpw cliff to Q5_K and the
attention family hoarded ~1 GB the MLP family needed. Hence:

- **Menus must have fine rungs in both directions** (Voodoo's per-role
  `--attention_candidates` / `--non_attention_candidates` defaults follow
  this).
- **Post-hoc guardrails** (`--tensor_upgrades`) bump matched tensor families
  up or down the ladder after training, funded by `--budget_reduction` —
  e.g. "no big MLP pinned to the menu floor; no attention pinned to the
  ceiling."
- **Calibration**: our imatrix stays; scaling imatrix volume buys ~0.2%,
  a polish, not a fix.

### 6.2 Long context punishes MLP starvation

The PPL gap between a well- and badly-allocated layout **widens** with
context (5.3% at 512 tokens → 9.2% at 8k). Train at the context you serve
(8192 is the default for large models), and select checkpoints at multiple
context lengths — training KL alone misses the widening.

### 6.3 Hybrid SSM/attention models need sequence length

Short sequences starve SSM-adjacent gradients (the recurrent state barely
warms up), biasing the optimizer to under-quantize the SSM projections.
512 tokens is the floor; 8192 is safe. The tiny per-head SSM state tensors
(`A_log`, `dt_bias`, `conv1d`) are kept F32 automatically.

### 6.4 What precision actually matters

| tensor family | sensitivity | consequence |
|---|---|---|
| MLP `gate/up/down` (mid/late stack) | dominant | spend your budget here |
| attention projections | ~zero | fine rungs, never a cliff |
| embedding | low | Q3_K/Q4_K class is enough |
| lm_head | moderate | Q5_K is the sweet spot at ~4 bpw budgets |
| SSM `alpha/beta`, norms, small state | keep exact | excluded from selection or F32 |

## 7. Naming and sizes

A `Voodoo{NN}` model is NN% of the original 8-bit size
(`--compression_ratio 0.45` → Voodoo45). Standard sizes: 25–80 in steps of 5.
Released GGUFs carry a UD-equivalent badge
(`<slug>.Voodoo45_IQ2_M.gguf`) so model cards index correctly; the label is a
fixed curated mapping, not derived from the assignment.

## 8. Reproducing the pipeline

```
1. voodoo data      # tokenize calibration text (or point at existing tokens)
2. voodoo train     # gate training + bake → .pt + quant_assignments.json
3. voodoo export    # re-quantize assignments → GGUF (llama.cpp-exact)
4. voodoo eval      # PPL + KL vs the frozen teacher
```

Every stage is resumable and journaled; the candidate cache makes re-runs at
different sizes start in seconds.

## 9. Sensitivity warm-start and knapsack polish

The gate loop structurally mis-attributes credit: logit-KL gives attention
tensors strong local gradients (they sit near the input of every downstream
path) while each MLP tensor's rung-up gain is small and diffuse. Two
mechanisms address this from opposite ends:

- **Warm start** (`--warm_start`): before training, one hooked forward pass
  captures the activations entering every selectable tensor;
  `s(t,q) = ‖(W_q−W)X_t‖²/‖W X_t‖²` is measured per candidate (W_q is the
  candidate cache — the expensive part is already done); a greedy knapsack
  over quality-per-byte produces the initial layout and the gate logits
  start concentrated on it. Training then only refines boundaries — the one
  job gradients are good at here. The measured table persists to
  `sensitivity.pkl` next to the journal.
- **Knapsack polish** (`--polish`): after the argmax freezes (same injection
  point as `--tensor_upgrades`), enumerate one-rung moves per tensor, rank
  by measured benefit-per-byte, greedily apply inside the size dead zone —
  spending slack on ranked upgrades and swapping low-value fat rungs down.
  At tau≈0.01 gradients are dead; this executes exactly the per-tensor swap
  the loop cannot.

**Measured caveat (0.8B, finalize-time experiment):** polish requires the
measured table from `--warm_start`; without it, polish skips itself. 

Measured ladder on Qwen3.5-0.8B Voodoo45 (same data/settings, 100-chunk
llama-perplexity, BF16 reference = 21.06):

| run | steps | PPL |
|---|---|---|
| cold start (25 journaled steps) | 25 | 21.33 |
| warm start | 10 | 23.80 |
| warm start + measured polish | 2 | 30.06 |

The mechanisms behave as specified — warm start lands exactly on budget
(338.6 MB), measured tables persist and feed polish, packing stays inside
tolerance — but at 0.8B scale with a handful of steps the warm start does
not yet beat a converged cold run; the original evidence for both mechanisms
is from 27B-scale layouts (WHITEPAPER §6), where budget allocation dominates
and the failure modes (menu-floor MLP collapse, attention rung hoarding) are
large. Treat 0.8B as the wiring test, not the efficacy proof.
