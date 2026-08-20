# Agent Notes — voodoo-dyn-quant

Learned per-tensor mixed-precision quantization ("Voodoo") for any LLM
architecture, llama.cpp-exact by construction. Read `WHITEPAPER.md` for how it
works before modifying training logic.

## Operating rules

- The user's workflow runs through the skills in `skills/`. For any install,
  data, train, export, or eval request, follow the matching skill in
  `skills/<task>/SKILL.md` — they encode the canonical commands and defaults.
- Never diverge from a skill's recipe without telling the user why.
- This project is hardware-agnostic. Vendor/device-specific code belongs ONLY
  in `voodoo_quant/hardware/` (env defaults, vendor attention backends) and in
  launcher glue. The core (`ggml`, `layers`, `parallel`, `training`) must
  never branch on vendor.
- Quantization exactness is the project's core promise: every quantized byte
  comes from llama.cpp's own quantizer via `voodoo_quant.ggml` (ctypes →
  `libggml-base`). NEVER add a Python/pure-torch approximate quantizer for
  weight storage. GPU dequant (`quants/gpu_dequant.py`) of ggml-produced bytes
  is fine and verified per-type.
- TP correctness invariant: candidate-cache keys are FULL-tensor names; ranks
  slice candidates after load (bit-identical for block-aligned slices). Do not
  break this.
- Architecture adapters are the ONLY place model topology is known
  (`voodoo_quant/arch/`): base.py defines the ArchAdapter contract (shard_layer,
  gguf_name, transpose_embedding, attention_segments), `__init__.py` is the
  registry. The core (`ggml`, `layers`, `parallel`, `training`) must never
  import a concrete adapter or branch on model_type. `parallel.tp_patch_model`
  dispatches via `adapter_for_model` and REFUSES unknown archs for TP.
  Adding an arch = one file + `@register`; add a mapping test in
  tests/test_arch.py.
- Do not run heavy training/eval jobs without checking GPU headroom first
  (`voodoo doctor`, or `nvidia-smi`/`rocm-smi`). Another process on the GPU
  causes mysterious OOMs and silent CPU fallbacks.
- Keep checkpoints/data/logs out of git (already in `.gitignore`). `third_party/`
  holds the llama.cpp checkout — never commit it.

## Environment

- One-time setup: `make bootstrap` (venv + package + `third_party/llama.cpp`
  libggml build). Re-run pieces individually: `make venv`, `make llamacpp`.
- Use the project venv: `.venv/bin/python` or `.venv/bin/voodoo`.
- `libggml-base` discovery: `VOODOO_GGML_LIB` env var → `third_party/llama.cpp/
  build/bin` → sibling `llama.cpp*` → system. `voodoo doctor` reports the
  resolved path.
- Hardware env defaults (allocator, NCCL, worker caps) come from
  `voodoo_quant.hardware.env_defaults()`. `voodoo tp` applies them
  automatically; manual torchrun launches should apply them too.
- A pinned llama.cpp ref can be set with `make llamacpp LLAMA_CPP_REF=<tag>`.

## Layout

```
voodoo_quant/
  ggml.py            llama.cpp-exact quant/dequant (ctypes), quant registry, ladder
  layers.py          MixedQuantLinear/Embedding + candidate cache (the gates)
  parallel.py        TP state, collectives, ShardSpec, sharding primitives
  arch/              DETACHED arch adapters: base.py + registry; qwen.py, lfm2.py
  quants/            GPU dequant kernels + LUTs for IQ/K formats
  hardware/          vendor detection, env defaults, ROCm flash-attention backend
  training/          trainer.py (gate training + bake), q8_teacher.py
  tools/             data.py, precache.py, export_gguf.py, evaluate.py, make_teacher.py,
                     sidecar.py (MTP nextn sidecar quant spec: default Q6_K, k/v Q8_0)
  naming.py          Voodoo{NN} size/label conventions
  imatrix.py         llama.cpp imatrix GGUF → HF tensor-name mapping
  cli.py             `voodoo` entrypoint (train/tp/export/eval/data/precache/make-teacher/doctor)
configs/             curated run configs (see skills/train)
skills/              agentic workflow skills (install/data/train/export/eval)
tests/               unit + smoke tests (CPU-runnable where possible)
```

## Conventions

- Env vars are `VOODOO_*`: `VOODOO_GGML_LIB`, `VOODOO_QUANT_WORKERS`,
  `VOODOO_FINALIZE_WORKERS`, `VOODOO_TINY_RECIPE` (tiny test config).
- Sizes: `Voodoo{NN}` = NN% of the 8-bit original; standard sizes 25–80 step 5;
  `--compression_ratio` = NN/100. GGUF quant badges come from
  `voodoo_quant.naming.VOODOO_QUANT_LABEL` (curated, not assignment-derived).
- The 8-bit candidate is `Q8_0` (never `Q8_K` — no vec_dot kernel exists).
- Default candidate set excludes ternary (`TQ1_0`/`TQ2_0` underperform).
- Post-hoc `--tensor_upgrades` accept negative levels (rungs down); first
  match wins; `--budget_reduction` funds upgrades without busting the target.
- Straight-through Gumbel hardening (`--st_gumbel_fraction`, 0 = off): each
  layer forwards a one-hot Gumbel sample of its gates with that per-step
  probability; backward stays soft/exact. `--st_gumbel_tau` sets sampling
  sharpness.
- Training runs journal `partial.pt` + `partial.meta.json` in the output dir;
  crash recovery is `--finalize_from_partial`, continue-training is
  `--resume_from_partial`. TP launchers auto-resume from the journal.
- Naming for artifacts: dash-joined checkpoints
  (`<slug>-Voodoo{NN}.pt`, `<slug>-Voodoo{NN}.quant_assignments.json`); the
  released GGUF alone uses `<slug>.Voodoo{NN}_{QUANT}.gguf`.

## Testing

- `make test` runs the suite. Tests must not require a GPU; guard GPU tests
  with `torch.cuda.is_available()` skips.
- `VOODOO_TINY_RECIPE=1` shrinks Qwen3.5-family configs for fast e2e runs.
- After touching `layers.py` or `parallel.py`, at minimum run
  `pytest tests/test_ggml.py tests/test_naming.py` and compile-check the tree.
