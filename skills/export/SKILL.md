# Skill: export

Bake a trained assignment into a llama.cpp-exact GGUF.

## Command

```bash
.venv/bin/voodoo export \
    --checkpoint checkpoints/<family>/Voodoo{NN}/<slug>-Voodoo{NN}.pt \
    --quant-assignments checkpoints/<family>/Voodoo{NN}/<slug>-Voodoo{NN}.quant_assignments.json \
    --ref-gguf <bf16-reference.gguf> \
    --output checkpoints/<family>/Voodoo{NN}/<slug>-Voodoo{NN}.gguf
```

## Rules

- The reference GGUF supplies architecture metadata + tensor-name mapping
  only; weights come from the checkpoint via llama.cpp's quantizer.
- The output is auto-renamed to the canonical
  `<slug>.Voodoo{NN}_{QUANT}.gguf` (curated label from
  `voodoo_quant.naming.VOODOO_QUANT_LABEL`); pass `--quant-label` to override.
- **MTP sidecar**: tensors present only in the reference (`blk.N.nextn.*`)
  are quantized per `voodoo_quant.tools.sidecar` — large weights `Q6_K`
  (override with `--sidecar-quant <TYPE>`), attention k/v `Q8_0`, norms and
  per-head state F32 — by the same exact ggml quantizer. `--sidecar-quant
  none` copies through unquantized (F32/BF16), which costs ~25% file size at
  27B (~475 MiB); defaulting to Q6_K matches the measured reference layout.
- Tied embeddings (`tie_word_embeddings=True`): `output.weight` is omitted and
  llama.cpp reuses `token_embd.weight` — correct, leave it.
- Split reference GGUFs: the reference must be a single-file GGUF; a split
  ref cannot source MTP/nextn sidecar tensors. Use
  `llama-gguf-split --merge` on the ref first if needed.
- For MTP variants the reference MUST be the MTP BF16 GGUF so the nextn
  sidecar tensors are copied into the output.
- Export parallelizes on CPU (`VOODOO_FINALIZE_WORKERS`, default min(8, nproc)).

## Verify

- File size within a few % of the Voodoo target.
- Quick smoke: `llama-cli -m <gguf> -p "The capital of France is" -n 64 -ngl 99`
  (or `llama-perplexity` for a PPL check; see the eval skill).
