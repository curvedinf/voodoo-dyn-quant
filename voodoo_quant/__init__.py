"""Voodoo — learned per-tensor mixed-precision quantization for any LLM.

The package is hardware-agnostic at its core:

- :mod:`voodoo_quant.ggml`      llama.cpp-exact quantization (ctypes -> libggml-base)
- :mod:`voodoo_quant.layers`    differentiable mixed-quant layers + candidate cache
- :mod:`voodoo_quant.parallel`  tensor-parallel sharding (generic + arch adapters)
- :mod:`voodoo_quant.training`  the gate trainer, losses, budget, journal
- :mod:`voodoo_quant.tools`     data prep, GGUF export, evaluation, precache
- :mod:`voodoo_quant.hardware`  accelerator detection + vendor-specific extras
"""

from voodoo_quant.cache import setup_caches  # noqa: F401  (import side-effect friendly)

__version__ = "0.1.0"
