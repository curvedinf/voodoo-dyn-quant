#!/usr/bin/env python3
"""
Per-model timing and resource-utilization logging.

Used to collect wall-clock time, CPU/RAM usage, and GPU memory/utilization
for every stage of Voodoo model creation (training, export, eval, perplexity,
example completions, etc.). Logs are written as JSONL to
``<model_dir>/run_stats.jsonl`` so each model has a single append-only record
of every script that touched it.

Example::

    from voodoo_quant.stats import log_stage

    with log_stage(
        stage="train",
        model_dir=Path(args.output_dir),
        script=Path(__file__).name,
        args=vars(args),
    ):
        run_training()
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INTERVAL = 2.0


def _parse_ps_output(text: str) -> tuple[float, float, int]:
    """Sum %CPU and RSS across a ``ps`` output block."""
    cpu_total = 0.0
    rss_total = 0.0
    n = 0
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                cpu_total += float(parts[0])
                rss_total += float(parts[1])
                n += 1
            except ValueError:
                continue
    return cpu_total, rss_total, n


def _rocm_smi() -> dict[str, Any] | None:
    """Return GPU memory (MB) and utilization (%) from rocm-smi, or None."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showuse"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = proc.stdout + proc.stderr
    except Exception:
        return None

    mem_lines = [l for l in out.splitlines() if "VRAM Total Used Memory" in l]
    util_lines = [l for l in out.splitlines() if "GPU use (%)" in l]

    mem_mb = None
    if mem_lines:
        try:
            mem_b = int(mem_lines[0].split(":")[-1].strip().split()[0])
            mem_mb = mem_b / 1024 / 1024
        except Exception:
            pass

    util = None
    if util_lines:
        try:
            util = float(util_lines[0].split(":")[-1].strip())
        except Exception:
            pass

    return {"gpu_mem_mb": mem_mb, "gpu_util_percent": util}


def _nvidia_smi() -> dict[str, Any] | None:
    """Return GPU memory (MB) and utilization (%) from nvidia-smi, or None."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = proc.stdout.strip()
    except Exception:
        return None

    mem_mb = None
    util = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mem_mb = float(parts[0])
                util = float(parts[1])
                break
            except Exception:
                continue
    return {"gpu_mem_mb": mem_mb, "gpu_util_percent": util}


def _gpu_sample() -> dict[str, Any]:
    """Try ROCm first, then NVIDIA."""
    sample = _rocm_smi()
    if sample is None or (sample["gpu_mem_mb"] is None and sample["gpu_util_percent"] is None):
        sample = _nvidia_smi()
    if sample is None:
        return {"gpu_mem_mb": None, "gpu_util_percent": None}
    return sample


class ResourceMonitor:
    """Sample CPU/RAM for the process group and GPU state on an interval."""

    def __init__(self, interval: float = DEFAULT_INTERVAL):
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pgid = os.getpgrp()

    def _sample(self) -> None:
        cpu_total = 0.0
        rss_total = 0.0
        n_procs = 0
        try:
            proc = subprocess.run(
                ["ps", "-o", "%cpu,rss", "-g", str(self.pgid), "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cpu_total, rss_total, n_procs = _parse_ps_output(proc.stdout)
        except Exception:
            pass

        gpu = _gpu_sample()

        self.samples.append({
            "time": time.time(),
            "cpu_percent": cpu_total,
            "rss_kb": rss_total,
            "n_procs": n_procs,
            "gpu_mem_mb": gpu["gpu_mem_mb"],
            "gpu_util_percent": gpu["gpu_util_percent"],
        })

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0}

        def _max(key: str):
            vals = [s[key] for s in self.samples if s[key] is not None]
            return max(vals) if vals else None

        def _avg(key: str):
            vals = [s[key] for s in self.samples if s[key] is not None]
            return sum(vals) / len(vals) if vals else None

        return {
            "samples": len(self.samples),
            "cpu_percent_max": _max("cpu_percent"),
            "cpu_percent_avg": _avg("cpu_percent"),
            "rss_mb_max": _max("rss_kb") / 1024 if _max("rss_kb") is not None else None,
            "rss_mb_avg": _avg("rss_kb") / 1024 if _avg("rss_kb") is not None else None,
            "gpu_mem_mb_max": _max("gpu_mem_mb"),
            "gpu_mem_mb_avg": _avg("gpu_mem_mb"),
            "gpu_util_percent_max": _max("gpu_util_percent"),
            "gpu_util_percent_avg": _avg("gpu_util_percent"),
        }


@contextmanager
def log_stage(
    stage: str,
    model_dir: str | Path,
    script: str | None = None,
    args: dict[str, Any] | None = None,
    interval: float = DEFAULT_INTERVAL,
):
    """Context manager that times a stage and appends resource stats to the model log."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run_stats.jsonl"

    start_time = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    monitor = ResourceMonitor(interval=interval)
    monitor.start()

    status = "completed"
    error: str | None = None
    try:
        yield monitor
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        elapsed = time.time() - start_time
        monitor.stop()
        record: dict[str, Any] = {
            "timestamp": start_iso,
            "stage": stage,
            "script": script,
            "args": args,
            "status": status,
            "duration_seconds": elapsed,
            "resources": monitor.summary(),
        }
        if error is not None:
            record["error"] = error
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def log_event(
    stage: str,
    model_dir: str | Path,
    script: str | None = None,
    args: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a lightweight event (no resource monitoring) for a model."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "run_stats.jsonl"
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "script": script,
        "args": args,
        "status": "event",
    }
    if metadata:
        record["metadata"] = metadata
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
