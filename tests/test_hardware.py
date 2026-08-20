"""Hardware detection and env-default sanity (CPU-safe)."""

import os

from voodoo_quant.hardware import detect, env_defaults, quant_workers, shell_exports


def test_detect_returns_summary():
    hw = detect()
    assert hw.vendor in ("nvidia", "amd", "cpu")
    assert hw.torch_backend in ("cuda", "cpu")
    if hw.torch_backend == "cpu":
        assert hw.gpu_count == 0
    else:
        assert hw.gpu_count >= 1


def test_env_defaults_shape():
    env = env_defaults()
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MALLOC_ARENA_MAX"] == "2"
    if detect().is_cuda:
        alloc = env.get("PYTORCH_HIP_ALLOC_CONF") or env.get("PYTORCH_CUDA_ALLOC_CONF")
        assert alloc == "expandable_segments:True"


def test_shell_exports_printable():
    text = shell_exports(env_defaults())
    assert text.startswith("export ")
    assert "OMP_NUM_THREADS=1" in text


def test_quant_workers_env_override(monkeypatch):
    monkeypatch.setenv("VOODOO_QUANT_WORKERS", "3")
    assert quant_workers() == 3


def test_cache_dirs_setup():
    import voodoo_quant.cache as c

    c.setup_caches()
    assert os.path.isdir(os.environ["TRITON_CACHE_DIR"])
