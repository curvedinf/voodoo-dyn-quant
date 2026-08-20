# Skill: eval

Evaluate a trained Voodoo checkpoint or exported GGUF. Report BOTH torch PPL
+ KL vs the frozen BF16 teacher (ground truth) and, for GGUFs, a
llama.cpp PPL (runtime check).

## 1. Torch eval (ground truth PPL/KL)

```bash
.venv/bin/voodoo eval \
    --model <model-id> \
    --checkpoint checkpoints/<family>/Voodoo{NN}/<slug>-Voodoo{NN}.pt \
    --data_dir data/<model> --data_name val_tokens.pt \
    --output_dir checkpoints/<family>/Voodoo{NN}/evals \
    --device cuda --dtype bfloat16 --no_compile --max_steps 50
```

- Omit `--base_checkpoint` so the teacher is the HF BF16 model (KLD is vs
  BF16; BF16 itself scores 0.0).
- MTP variants: pass `--base_checkpoint <converted-mtp-bf16.pt>`; those
  numbers are NOT comparable to non-MTP numbers.
- Write results into the variant's `evals/` folder, named by dataset
  (`fineweb.json`, `wikitext103.json`, ...).

## 2. llama.cpp PPL (runtime cross-check)

```bash
<path-to-llama.cpp>/build/bin/llama-perplexity \
    -m <model>.gguf -f data/<model>/val_text.txt \
    -c 512 --chunks 100 -ngl 99
```

- Fixed settings make runs cross-comparable. For long-context models also
  run `-c 8192` (fewer chunks) — allocation quality diverges MORE at long
  context, and a single 512-ctx score can hide it.
- torch and llama.cpp PPLs are on different scales for some architectures
  (known for hybrid SSM models); the RANKING should agree. If it does not,
  suspect a bad checkpoint, not the quant.

## 3. Rules

- Always report: file size (GB), torch PPL, KLD, llama PPL.
- Select/report checkpoints at multiple context lengths; judging by training
  KL alone misses widening long-context gaps.
- Multiple evals can run concurrently (each ~2–3 GB VRAM); check headroom
  first.
