"""Accelerator detection and hardware-specific training defaults.

Voodoo's core is hardware-agnostic; this module is the one place that knows
about vendor quirks. :func:`detect` returns a :class:`Hardware` summary and
:func:`env_defaults` produces the environment exports a launcher should apply
(allocator settings, NCCL tweaks, worker caps). Nothing else in the package
branches on vendor.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass
class Hardware:
    vendor: str  # "nvidia" | "amd" | "cpu"
    torch_backend: str  # "cuda" | "cpu"
    gpu_count: int
    gpu_names: list[str] = field(default_factory=list)
    vram_total_gb: float = 0.0

    @property
    def is_cuda(self) -> bool:
        return self.torch_backend == "cuda"

    def describe(self) -> str:
        head = f"{self.vendor} x{self.gpu_count} ({self.torch_backend})"
        if self.gpu_names:
            head += f" — {', '.join(self.gpu_names[:4])}"
        if self.vram_total_gb:
            head += f", {self.vram_total_gb:.0f} GB total VRAM"
        return head


def _probe_torch() -> tuple[int, list[str]]:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            return n, names
    except Exception:
        pass
    return 0, []


def _rocm_probe() -> list[str]:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        names = []
        for line in out.stdout.splitlines():
            if "Card series" in line:
                names.append(line.split(",", 1)[1].strip())
        return names
    except Exception:
        return []


def _nvidia_probe() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return [l.split("(")[0].strip() for l in out.stdout.splitlines() if "GPU" in l]
    except Exception:
        return []


def detect() -> Hardware:
    count, torch_names = _probe_torch()
    if count:
        amd = os.environ.get("ROCM_HOME") or any(
            "amd" in n.lower() or "gfx" in n.lower() for n in torch_names
        )
        vendor = "amd" if amd else "nvidia"
        names = _rocm_probe() if vendor == "amd" else _nvidia_probe()
        names = names or torch_names
        try:
            import torch

            vram = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(count)
            ) / 1e9
        except Exception:
            vram = 0.0
        return Hardware(vendor, "cuda", count, names, vram)
    return Hardware("cpu", "cpu", 0)


def env_defaults(hw: Hardware | None = None) -> dict[str, str]:
    """Vendor-appropriate environment defaults as a printable dict."""
    hw = hw or detect()
    env: dict[str, str] = {
        # Host RAM hygiene for multi-worker quantization (any vendor).
        "MALLOC_ARENA_MAX": "2",
        "OMP_NUM_THREADS": "1",
        # Kernel-compile hygiene.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_PARALLEL_COMPILE": "1",
    }
    if hw.torch_backend == "cuda":
        if hw.vendor == "amd":
            # ROCm: expandable segments avoid fragmentation; P2P off works
            # around unreliable xGMI links on some MI boards.
            env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
            env["NCCL_P2P_DISABLE"] = "1"
        else:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def apply_env() -> dict[str, str]:
    """Apply :func:`env_defaults` to the current process and return them."""
    env = env_defaults()
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def shell_exports(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={v}" for k, v in env.items())


def quant_workers(hw: Hardware | None = None) -> int:
    """Default CPU worker count for the exact ggml quantizer (GIL-releasing)."""
    hw = hw or detect()
    cap = 8 if hw.is_cuda else 4  # leave cores for GPU feeding threads
    return int(os.environ.get("VOODOO_QUANT_WORKERS", min(cap, os.cpu_count() or 2)))


def has_ggml() -> bool:
    from voodoo_quant import ggml

    return ggml.find_libggml() is not None


def doctor() -> str:
    """Human-readable environment report for the CLI."""
    lines = []
    hw = detect()
    lines.append(f"hardware      : {hw.describe()}")
    try:
        import torch

        lines.append(f"torch         : {torch.__version__} (hip={getattr(torch.version, 'hip', None)})")
    except Exception as exc:
        lines.append(f"torch         : NOT IMPORTABLE ({exc})")
    try:
        import transformers

        lines.append(f"transformers  : {transformers.__version__}")
    except Exception as exc:
        lines.append(f"transformers  : NOT IMPORTABLE ({exc})")
    from voodoo_quant import ggml

    lines.append(f"libggml-base  : {ggml.lib_path() or 'NOT FOUND — run `make llamacpp` or set VOODOO_GGML_LIB'}")
    lines.append("env defaults  :")
    for k, v in env_defaults(hw).items():
        lines.append(f"  {k}={v}")
    return "\n".join(lines)
