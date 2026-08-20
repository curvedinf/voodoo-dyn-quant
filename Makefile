# Voodoo dynamic quantization — preconfigured build & run defaults.
#
#   make bootstrap   one-shot: venv + package + llama.cpp libggml (nice defaults)
#   make llamacpp    clone+build llama.cpp's libggml-base into third_party/
#   make test        unit/smoke tests (CPU, no GPU needed)
#   make doctor      environment report
#   make clean       caches (keeps third_party and .venv)

PARENT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV        ?= $(PARENT_DIR).venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip

# Pinned llama.cpp commit providing the exact quantizers (libggml-base.so).
LLAMA_CPP_REF ?= master
THIRD_PARTY   := $(PARENT_DIR)third_party
LLAMA_DIR     := $(THIRD_PARTY)/llama.cpp
LLAMA_LIB     := $(LLAMA_DIR)/build/bin/libggml-base.so

.PHONY: bootstrap venv llamacpp test doctor clean

bootstrap: venv llamacpp
	$(PY) -m voodoo_quant.cli doctor || true
	@echo "bootstrap complete — try: $(VENV)/bin/voodoo train --help"

venv:
	@test -x $(PY) || { \
		python3 -m venv $(VENV) && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -e "$(PARENT_DIR)[torch,test]"; }

llamacpp: $(LLAMA_LIB)

$(LLAMA_LIB):
	mkdir -p $(THIRD_PARTY)
	git clone --depth 1 --branch $(LLAMA_CPP_REF) https://github.com/ggml-org/llama.cpp $(LLAMA_DIR)
	cmake -S $(LLAMA_DIR) -B $(LLAMA_DIR)/build \
		-DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF
	cmake --build $(LLAMA_DIR)/build --target ggml -j $$(nproc)
	@test -f $(LLAMA_LIB) || (echo "libggml-base.so missing after build" && exit 1)
	@echo "libggml-base built at $(LLAMA_LIB)"

test: venv
	$(PY) -m pytest tests/

doctor: venv
	$(PY) -m voodoo_quant.cli doctor

clean:
	rm -rf .cache .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
