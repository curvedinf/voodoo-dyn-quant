"""Naming conventions: Voodoo sizes, quant labels, canonical GGUF paths."""

from pathlib import Path

from voodoo_quant.naming import (
    VOODOO_QUANT_LABEL,
    canonical_gguf_name,
    canonicalize_gguf_path,
    quant_label_for_size,
    size_for_ratio,
)


def test_standard_labels_exist():
    for size in range(25, 81, 5):
        assert quant_label_for_size(size) is not None, size


def test_label_examples():
    assert VOODOO_QUANT_LABEL[45] == "IQ2_M"
    assert VOODOO_QUANT_LABEL[60] == "IQ4_XS"
    assert VOODOO_QUANT_LABEL[30] == "IQ1_S"


def test_canonical_gguf_name():
    assert canonical_gguf_name("Qwen3.5-0.8B", 45, "IQ2_M") == "Qwen3.5-0.8B.Voodoo45_IQ2_M.gguf"


def test_canonicalize_paths():
    # dot form (already released) and dash form both canonicalize
    p = canonicalize_gguf_path(Path("/x/Qwen3.5-0.8B-Voodoo45.gguf"))
    assert p.name == "Qwen3.5-0.8B.Voodoo45_IQ2_M.gguf"
    p = canonicalize_gguf_path("/x/model.Voodoo60_IQ4_XS.gguf")
    assert p.name == "model.Voodoo60_IQ4_XS.gguf"  # already canonical
    # non-voodoo names pass through untouched
    assert canonicalize_gguf_path("/x/plain.gguf").name == "plain.gguf"


def test_size_for_ratio():
    assert size_for_ratio(0.45) == 45
    assert size_for_ratio(0.301) == 30
    assert size_for_ratio(0.78) == 80
