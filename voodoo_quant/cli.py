#!/usr/bin/env python3
"""Unified ``voodoo`` command-line entrypoint.

Subcommands:

- ``train``      full Voodoo trainer arg surface (delegates to
                 ``voodoo_quant.training.trainer``)
- ``tp``         launch tensor-parallel training under ``torch.distributed.run``
- ``doctor``     hardware / torch / libggml environment report
- ``export``     mixed-precision checkpoint -> llama.cpp-exact GGUF
- ``eval``       torch PPL + KL evaluation of a checkpoint
- ``data``       prepare calibration/eval token data
- ``precache``   pre-build the quant candidate cache
- ``make-teacher`` create a BF16 base .pt + Q8_0 teacher from an HF dir

Dispatch is deliberately shallow: this module parses only the subcommand
token and hands the rest of ``argv`` to the owning module's own argparse, so
each tool keeps its full argument surface and ``--help`` without duplicating
or re-parenting parser actions here.
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOL_MODULES = {
    "export": "export_gguf",
    "eval": "evaluate",
    "data": "data",
    "precache": "precache",
    "make-teacher": "make_teacher",
}
_TOOLS = tuple(_TOOL_MODULES)

_USAGE_EXAMPLES = {
    "train": "voodoo train --model Qwen/Qwen3.5-0.8B-Base --compression_ratio 0.45 ...",
    "tp": "voodoo tp --nproc 4 -- --tensor_parallel 4 --base_checkpoint <pt> ...",
    "export": "voodoo export --checkpoint <pt> --quant-assignments <json> --ref-gguf <gguf> --output <gguf>",
    "eval": "voodoo eval --checkpoint <pt> --data_dir data/<model> --max_steps 50",
    "data": "voodoo data --model <model-id> --output_dir ./data/<model>",
    "precache": "voodoo precache --checkpoint <pt> --model <model-id>",
    "make-teacher": "voodoo make-teacher --model_dir ./<model-dir> --output_dir checkpoints/<model>",
}


def _cmd_train(rest: list[str]) -> int:
    """Delegate to the trainer's own ``main()`` by intercepting ``sys.argv``.

    The trainer parser is owned by ``voodoo_quant.training.trainer``; rather
    than re-parenting its actions into a subparser (fragile with defaults and
    help formatting), we splice ``rest`` into ``sys.argv`` and call
    ``trainer.main()`` so ``voodoo train ...`` behaves exactly like running
    the trainer module directly — same argparse surface, same ``--help``,
    same run-stats stage logging. This is also the entrypoint ``voodoo tp``
    re-executes under torchrun.
    """
    from voodoo_quant.training import trainer

    argv_backup = sys.argv
    sys.argv = [argv_backup[0], *rest]
    try:
        trainer.main()
    finally:
        sys.argv = argv_backup
    return 0


def _cmd_tp(rest: list[str]) -> int:
    """Re-exec tensor-parallel training under ``torch.distributed.run``.

    Applies the hardware env defaults, prints the resolved command, then
    replaces this process with ``python -m torch.distributed.run
    --nproc_per_node N -m voodoo_quant.cli train <train args...>``.
    """
    parser = argparse.ArgumentParser(
        prog="voodoo tp",
        description="Launch tensor-parallel Voodoo training under torchrun.",
        epilog=f"example: {_USAGE_EXAMPLES['tp']}",
    )
    parser.add_argument("--nproc", type=int, required=True, help="torchrun --nproc_per_node (one rank per GPU)")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="trainer arguments forwarded verbatim (conventionally after a literal '--')",
    )
    args = parser.parse_args(rest)

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    from voodoo_quant.hardware import apply_env

    env = apply_env()
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(args.nproc),
        "-m", "voodoo_quant.cli",
        "train",
        *train_args,
    ]
    print("Applied hardware env defaults (user-set values win):")
    for k, v in env.items():
        print(f"  {k}={v}")
    print("Launching:", " ".join(cmd), flush=True)
    os.execvp(sys.executable, cmd)
    return 0  # unreachable after execvp; kept for the type checker


def _cmd_doctor(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="voodoo doctor",
        description="Report hardware, torch, and libggml-base discovery state.",
    )
    parser.parse_args(rest)
    from voodoo_quant.hardware import doctor

    print(doctor())
    return 0


def _cmd_tool(name: str, rest: list[str]) -> int:
    """Run a ``voodoo_quant.tools`` module's ``main(rest)`` with its own argparse."""
    import importlib

    main_fn = getattr(importlib.import_module(f"voodoo_quant.tools.{_TOOL_MODULES[name]}"), "main")
    return main_fn(rest) or 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voodoo",
        description="Voodoo: learned per-tensor mixed-precision quantization, llama.cpp-exact by construction.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.add_parser(
        "train",
        help="train Voodoo gates and bake a mixed-precision checkpoint (see `voodoo train --help` for the full arg surface)",
    )
    tp = sub.add_parser(
        "tp",
        help="launch tensor-parallel training under torchrun (`voodoo tp --nproc N -- <train args...>`)",
    )
    tp.add_argument("--nproc", type=int, help=argparse.SUPPRESS)
    tp.add_argument("train_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    sub.add_parser("doctor", help="environment / hardware / libggml report")
    for name in _TOOLS:
        sub.add_parser(
            name,
            help=f"`voodoo {name} --help` for the full arg surface (example: {_USAGE_EXAMPLES[name]})",
        )
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    command, rest = argv[0], argv[1:]

    if command == "train":
        return _cmd_train(rest)
    if command == "tp":
        return _cmd_tp(rest)
    if command == "doctor":
        return _cmd_doctor(rest)
    if command in _TOOLS:
        return _cmd_tool(command, rest)

    build_parser().error(f"unknown command {command!r} (choose from train, tp, doctor, {', '.join(_TOOLS)})")
    return 2


if __name__ == "__main__":
    main()
