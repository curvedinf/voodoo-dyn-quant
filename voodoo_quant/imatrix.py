"""
Load llama.cpp imatrix data from a GGUF file and map it to HuggingFace tensor names.

llama.cpp imatrix files contain, for each quantized weight tensor:
  - {name}.in_sum2  : sum of squared activations per input column
  - {name}.counts   : number of samples aggregated (usually a scalar)

The actual importance vector is in_sum2 / counts[0].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).parent.parent


def _ensure_ggufpy():
    """Add llama.cpp's gguf-py to sys.path if a local checkout is available.

    Search order: the repo's ``third_party/llama.cpp`` (created by
    ``make llamacpp``) first, then any sibling ``llama.cpp*`` checkout next to
    the repo (e.g. ``llama.cpp-upstream``). An installed ``gguf`` package
    (``pip install gguf``) always works too — the paths are only a fallback
    for running from source.
    """
    candidates = [_REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"]
    candidates += [sibling / "gguf-py" for sibling in sorted(_REPO_ROOT.parent.glob("llama.cpp*"))]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_ensure_ggufpy()
try:
    from gguf import GGUFReader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gguf package not found; pip install gguf, or build llama.cpp into "
        "third_party/ with `make llamacpp` (its gguf-py is used automatically)"
    ) from exc


# Mapping from HuggingFace parameter names to llama.cpp GGUF names.
# Applies to Qwen3.5-0.8B-Base.  MLP/attention names are standard; linear_attn
# (Mamba/SSD) names come from the llama.cpp convert_hf_to_gguf.py mapping.
HF_TO_GGUF_SUFFIXES = {
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
}


def hf_name_to_gguf(hf_name: str) -> str | None:
    """Map a HuggingFace parameter name to a llama.cpp GGUF tensor name."""
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_idx = int(parts[2])
        rest = ".".join(parts[3:])
        gguf_suffix = HF_TO_GGUF_SUFFIXES.get(rest)
        if gguf_suffix is None:
            return None
        return f"blk.{layer_idx}.{gguf_suffix}"
    gguf_suffix = HF_TO_GGUF_SUFFIXES.get(hf_name)
    if gguf_suffix is not None:
        return gguf_suffix
    return None


def load_llamacpp_imatrix(path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load a llama.cpp imatrix GGUF and return {hf_tensor_name: imatrix_vector}.

    Missing GGUF tensors are silently omitted; callers can fall back to a
    uniform imatrix for those weights.
    """
    path = Path(path)
    reader = GGUFReader(str(path))

    sum2_tensors = {
        t.name: torch.from_numpy(t.data.copy()).float()
        for t in reader.tensors
        if t.name.endswith(".in_sum2")
    }
    count_tensors = {
        t.name: torch.from_numpy(t.data.copy()).float()
        for t in reader.tensors
        if t.name.endswith(".counts")
    }

    gguf_to_imatrix: dict[str, torch.Tensor] = {}
    for name, in_sum2 in sum2_tensors.items():
        base = name[: -len(".in_sum2")]
        count_name = base + ".counts"
        counts = count_tensors.get(count_name)
        if counts is None or counts.numel() == 0:
            continue
        gguf_to_imatrix[base] = in_sum2 / counts[0]

    # Invert to HF names.  GGUF names are like "blk.0.ffn_gate.weight".
    hf_imatrix: dict[str, torch.Tensor] = {}
    for gguf_name, imatrix in gguf_to_imatrix.items():
        if not gguf_name.startswith("blk."):
            # Map top-level tensors like "token_embd.weight" directly.
            for hf_name, gguf_suffix in HF_TO_GGUF_SUFFIXES.items():
                if "." not in hf_name and gguf_suffix == gguf_name:
                    hf_imatrix[hf_name] = imatrix
                    break
            continue

        # "blk.0.ffn_gate.weight" -> layer_idx=0, suffix="ffn_gate.weight"
        parts = gguf_name.split(".")
        layer_idx = int(parts[1])
        suffix = ".".join(parts[2:])
        for hf_suffix, gguf_suffix in HF_TO_GGUF_SUFFIXES.items():
            if gguf_suffix == suffix:
                hf_name = f"model.layers.{layer_idx}.{hf_suffix}"
                hf_imatrix[hf_name] = imatrix
                break

    return hf_imatrix
