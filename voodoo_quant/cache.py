"""Persistent compile-cache configuration.

Import this module (or :func:`setup_caches`) before any torch/triton compile so
kernels cache inside the project tree instead of ephemeral system temp dirs.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent.resolve()


def setup_caches() -> None:
    os.environ.setdefault("TRITON_CACHE_DIR", str(_ROOT / ".triton_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_ROOT / ".torchinductor_cache"))
    # Single-threaded compilation keeps first-compile latency and memory low.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


setup_caches()
