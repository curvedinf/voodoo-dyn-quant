"""llama.cpp-exact quantization and dequantization for GGUF weight formats.

Every quantized byte this package produces comes from llama.cpp's own C++
quantizer (``libggml-base``), so candidates, checkpoints and exported GGUFs
are bit-identical to a stock ``llama-quantize`` run by construction. There is
no approximate Python re-implementation anywhere in the stack.

Library discovery order:

1. ``$VOODOO_GGML_LIB`` — explicit path to ``libggml-base.so``/``.dylib``
2. ``<repo>/third_party/llama.cpp/build/bin`` — the ``make llamacpp`` build
3. A sibling ``../llama.cpp*/build/bin`` next to the repo
4. Plain ``libggml-base.so`` via the system loader

Adding a new quant type is a one-line registry entry.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import torch

QK_K = 256  # elements per K/IQ super-block


class QuantTypeInfo:
    def __init__(
        self,
        name: str,
        ggml_type: int,
        block_size: int,
        block_bytes: int,
        quant_func: str,
        dequant_func: str,
    ):
        self.name = name
        self.ggml_type = ggml_type
        self.block_size = block_size
        self.block_bytes = block_bytes
        self.quant_func = quant_func
        self.dequant_func = dequant_func

    def bytes_per_weight(self) -> float:
        return self.block_bytes / self.block_size


_QUANT_REGISTRY: dict[str, QuantTypeInfo] = {
    "IQ1_S": QuantTypeInfo("IQ1_S", 19, 256, 50, "quantize_iq1_s", "dequantize_row_iq1_s"),
    "IQ1_M": QuantTypeInfo("IQ1_M", 29, 256, 56, "quantize_iq1_m", "dequantize_row_iq1_m"),
    "IQ2_XXS": QuantTypeInfo("IQ2_XXS", 16, 256, 66, "quantize_iq2_xxs", "dequantize_row_iq2_xxs"),
    "IQ2_XS": QuantTypeInfo("IQ2_XS", 17, 256, 74, "quantize_iq2_xs", "dequantize_row_iq2_xs"),
    "IQ2_S": QuantTypeInfo("IQ2_S", 22, 256, 82, "quantize_iq2_s", "dequantize_row_iq2_s"),
    "IQ3_XXS": QuantTypeInfo("IQ3_XXS", 18, 256, 98, "quantize_iq3_xxs", "dequantize_row_iq3_xxs"),
    "IQ3_S": QuantTypeInfo("IQ3_S", 21, 256, 110, "quantize_iq3_s", "dequantize_row_iq3_s"),
    "IQ4_XS": QuantTypeInfo("IQ4_XS", 23, 256, 136, "quantize_iq4_xs", "dequantize_row_iq4_xs"),
    "Q4_K": QuantTypeInfo("Q4_K", 12, 256, 144, "quantize_q4_K", "dequantize_row_q4_K"),
    "Q5_K": QuantTypeInfo("Q5_K", 13, 256, 176, "quantize_q5_K", "dequantize_row_q5_K"),
    "Q6_K": QuantTypeInfo("Q6_K", 14, 256, 210, "quantize_q6_K", "dequantize_row_q6_K"),
    "Q8_0": QuantTypeInfo("Q8_0", 8, 32, 34, "quantize_q8_0", "dequantize_row_q8_0"),
    "TQ1_0": QuantTypeInfo("TQ1_0", 34, 256, 54, "quantize_tq1_0", "dequantize_row_tq1_0"),
    "TQ2_0": QuantTypeInfo("TQ2_0", 35, 256, 66, "quantize_tq2_0", "dequantize_row_tq2_0"),
}

# Quality ladder, coarsest to finest. Used by post-hoc ``tensor_upgrades``.
QUANT_LADDER: list[str] = [
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S",
    "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
]

_LIB_NAME = "libggml-base.so"


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("VOODOO_GGML_LIB")
    if env:
        paths.append(Path(env))
    repo_root = Path(__file__).parent.parent
    paths.append(repo_root / "third_party" / "llama.cpp" / "build" / "bin" / _LIB_NAME)
    for sibling in sorted((repo_root.parent).glob("llama.cpp*")):
        paths.append(sibling / "build" / "bin" / _LIB_NAME)
    return paths


_LIB: "ctypes.CDLL | None" = None


def find_libggml() -> Path | None:
    for p in _candidate_paths():
        if p.exists():
            return p
    try:  # system loader fallback
        ctypes.CDLL(_LIB_NAME)
        return Path(_LIB_NAME)
    except OSError:
        return None


def _load_lib() -> "ctypes.CDLL":
    global _LIB
    if _LIB is not None:
        return _LIB

    explicit = find_libggml()
    if explicit is None:
        raise RuntimeError(
            "libggml-base.so not found. Run `make llamacpp` (builds into third_party/), "
            "or set VOODOO_GGML_LIB=/path/to/libggml-base.so, or install llama.cpp system-wide."
        )
    try:
        _LIB = ctypes.CDLL(str(explicit))
    except OSError:
        raise RuntimeError(
            f"Failed to load libggml-base from {explicit}. Rebuild with `make llamacpp` "
            "or point VOODOO_GGML_LIB at a valid libggml-base.so."
        ) from None

    _LIB.ggml_quantize_init.argtypes = [ctypes.c_int]
    _LIB.ggml_quantize_init.restype = None

    for info in _QUANT_REGISTRY.values():
        quant = getattr(_LIB, info.quant_func)
        quant.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        quant.restype = ctypes.c_size_t

        dequant = getattr(_LIB, info.dequant_func)
        dequant.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        dequant.restype = None

    return _LIB


def lib_path() -> str | None:
    p = find_libggml()
    return str(p) if p else None


def list_quant_types() -> list[str]:
    return list(_QUANT_REGISTRY.keys())


def default_candidate_types() -> list[str]:
    """All quant types except ternary (TQ1_0/TQ2_0 underperform IQ1/IQ2 in runs)."""
    return [qt for qt in list_quant_types() if not qt.startswith("TQ")]


def get_quant_info(quant_type: str) -> QuantTypeInfo:
    if quant_type not in _QUANT_REGISTRY:
        raise ValueError(f"Unknown quant type: {quant_type}. Supported: {list(_QUANT_REGISTRY.keys())}")
    return _QUANT_REGISTRY[quant_type]


def bytes_per_weight(quant_type: str) -> float:
    return get_quant_info(quant_type).bytes_per_weight()


def quantize_tensor(
    weight: torch.Tensor,
    quant_type: str,
    imatrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize a weight matrix with llama.cpp's own quantizer.

    Args:
        weight: [out_features, in_features] float tensor (padded to a multiple
            of the quant block size by the caller).
        imatrix: [in_features] float32 importance weights (llama.cpp imatrix).

    Returns:
        qweight: [nblocks, block_bytes] uint8 raw quantized bytes.
    """
    info = get_quant_info(quant_type)
    lib = _load_lib()
    lib.ggml_quantize_init(info.ggml_type)

    if weight.dtype != torch.float32:
        weight = weight.float()
    weight = weight.contiguous()

    out_features, in_features = weight.shape
    if in_features % info.block_size != 0:
        raise ValueError(
            f"{quant_type} requires in_features % {info.block_size} == 0, got {in_features}"
        )

    nblocks_per_row = in_features // info.block_size
    nblocks = out_features * nblocks_per_row
    out_bytes = nblocks * info.block_bytes

    if imatrix is None:
        imatrix = torch.ones(in_features, dtype=torch.float32)
    else:
        imatrix = imatrix.to(torch.float32).contiguous()
    if imatrix.numel() != in_features:
        raise ValueError(f"imatrix size {imatrix.numel()} != in_features {in_features}")

    dst = (ctypes.c_uint8 * out_bytes)()
    src_ptr = ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float))
    im_ptr = ctypes.cast(imatrix.data_ptr(), ctypes.POINTER(ctypes.c_float))

    quant = getattr(lib, info.quant_func)
    result = quant(src_ptr, ctypes.byref(dst), out_features, in_features, im_ptr)
    if result != out_bytes:
        raise RuntimeError(f"{info.quant_func} returned {result} bytes, expected {out_bytes}")

    flat = torch.from_numpy(np.frombuffer(bytes(dst), dtype=np.uint8).copy())
    return flat.reshape(nblocks, info.block_bytes)


def dequantize_tensor(
    qweight: torch.Tensor,
    quant_type: str,
    out_features: int,
    in_features: int,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Dequantize raw quantized bytes back to a float32 weight matrix.

    Args:
        qweight: [nblocks, block_bytes] uint8 raw bytes.
        out_features / in_features: expected matrix shape (in_features must be
            a multiple of the quant block size).
        pin_memory: allocate pinned CPU output for async H2D transfers.
    """
    info = get_quant_info(quant_type)
    lib = _load_lib()

    if in_features % info.block_size != 0:
        raise ValueError(
            f"{quant_type} requires in_features % {info.block_size} == 0, got {in_features}"
        )

    nblocks_per_row = in_features // info.block_size
    expected_blocks = out_features * nblocks_per_row
    if qweight.numel() != expected_blocks * info.block_bytes:
        raise ValueError(
            f"qweight size {qweight.numel()} does not match {expected_blocks} blocks "
            f"of {info.block_bytes} bytes for {quant_type}"
        )

    qweight = qweight.reshape(expected_blocks, info.block_bytes).contiguous()
    out = torch.empty(
        out_features * in_features,
        dtype=torch.float32,
        device=qweight.device,
        pin_memory=pin_memory and qweight.device.type == "cpu",
    )

    dequant = getattr(lib, info.dequant_func)
    dequant(
        ctypes.cast(qweight.data_ptr(), ctypes.c_void_p),
        ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        out_features * in_features,
    )
    return out.reshape(out_features, in_features)


def estimate_bytes(
    shape: tuple[int, ...],
    quant_type: str | None,
    source_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Estimate stored bytes for a tensor (quant type or source dtype)."""
    numel = int(np.prod(shape))
    if quant_type is None:
        return numel * torch.finfo(source_dtype).bits // 8
    return int(numel * bytes_per_weight(quant_type))
