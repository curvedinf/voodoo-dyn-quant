"""Canonical naming for Voodoo deliverables.

Hugging Face only tags a GGUF on a repo's "GGUF" card when the filename
carries a recognizable quant badge, so released GGUFs are named::

    <model-slug>.Voodoo{NN}_{QUANT}.gguf      e.g.  Qwen3.5-0.8B.Voodoo45_IQ2_M.gguf

- ``{NN}`` is the Voodoo size: percent of the original 8-bit model size.
- ``{QUANT}`` is a curated UD-equivalent label per size (NOT derived from the
  per-tensor assignment distribution).

PyTorch checkpoints and sidecars keep the dash-joined form
``<model-slug>-Voodoo{NN}.pt`` / ``.quant_assignments.json``.
"""

from __future__ import annotations

import re
from pathlib import Path

# Voodoo size (percent of 8-bit) -> UD-equivalent quant label.
VOODOO_QUANT_LABEL: dict[int, str] = {
    25: "TQ1_0",
    30: "IQ1_S",
    35: "IQ1_M",
    40: "IQ2_XXS",
    45: "IQ2_M",
    50: "IQ3_XXS",
    55: "Q3_K_S",
    60: "IQ4_XS",
    65: "Q4_K_M",
    70: "Q5_K_S",
    75: "Q5_K_XL",
    80: "Q6_K",
}

STANDARD_SIZES: list[int] = list(range(25, 81, 5))

_VOOODO_RE = re.compile(r"^(?P<slug>.*?)[-.]Voodoo(?P<size>\d+)(?:_[^.]*)?\.gguf$")


def quant_label_for_size(size: int) -> str | None:
    """Curated quant label for a Voodoo size, or None if unknown."""
    return VOODOO_QUANT_LABEL.get(int(size))


def size_for_ratio(compression_ratio: float) -> int:
    """Round a compression ratio to the nearest standard Voodoo size."""
    return min(STANDARD_SIZES, key=lambda s: abs(s / 100.0 - compression_ratio))


def canonical_gguf_name(slug: str, size: int, label: str) -> str:
    return f"{slug}.Voodoo{int(size)}_{label}.gguf"


def canonicalize_gguf_path(path: str | Path, label_override: str | None = None) -> Path:
    """Return the canonical Voodoo GGUF path for ``path`` (rename target)."""
    p = Path(path)
    m = _VOOODO_RE.match(p.name)
    if not m:
        return p
    size = int(m.group("size"))
    label = label_override or quant_label_for_size(size)
    if not label:
        return p
    return p.with_name(canonical_gguf_name(m.group("slug"), size, label))
