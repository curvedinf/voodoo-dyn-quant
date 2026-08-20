# Skill: install

Set up voodoo-dyn-quant from a clean checkout. Target: a working `voodoo` CLI
with a built `libggml-base` and a passing doctor report.

## Steps

1. Bootstrap everything (venv + editable package + torch + llama.cpp libggml):
   ```bash
   cd <repo>
   make bootstrap
   ```
   This creates `.venv/`, installs `.[torch,test]`, clones llama.cpp into
   `third_party/llama.cpp` (shallow, `LLAMA_CPP_REF`, default `master`) and
   builds `libggml-base` (CPU-only build; no GPU toolchain needed).

2. If the user has an existing ROCm or CUDA torch already (e.g. a prebuilt
   ROCm wheel), install it into the venv FIRST, then `make venv` skips torch:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install <torch-wheel> \
     && .venv/bin/pip install -e ".[test]"
   ```

3. If the user already has a llama.cpp checkout with `libggml-base.so` built,
   skip `make llamacpp` and set:
   ```bash
   export VOODOO_GGML_LIB=/path/to/llama.cpp/build/bin/libggml-base.so
   ```

4. Verify:
   ```bash
   make doctor     # or: .venv/bin/voodoo doctor
   ```
   SUCCESS = hardware line shows the GPU(s), torch imports, and
   `libggml-base` resolves to a real path. If libggml is NOT FOUND, run
   `make llamacpp`.

5. Sanity test: `make test`.

## Notes

- ROCm users: install the ROCm torch wheel before step 1's `pip install -e`
  (the `torch` extra pulls the default wheel, which is CUDA/CPU).
- No GPU is required for quantization itself (CPU ggml) or for the unit
  tests; training needs one (CUDA or ROCm).
- Never commit `.venv/` or `third_party/`.
