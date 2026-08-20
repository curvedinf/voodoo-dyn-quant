"""Architecture adapter registry.

The registry maps models to :class:`~voodoo_quant.arch.base.ArchAdapter`
subclasses. The core never imports arch modules directly — it asks the
registry, which discovers every ``*_adapter``-registered module in this
package. Adding support for a new architecture means dropping one file in
``voodoo_quant/arch/`` and registering it here; nothing else changes.

When no adapter matches, a :class:`NullAdapter` is returned: single-GPU /
layer-wise training and export still work (they are arch-agnostic), but
tensor parallelism refuses to run rather than sharding blindly.
"""

from __future__ import annotations

from typing import Optional, Type

from voodoo_quant.arch.base import ArchAdapter, DEFAULT_ATTENTION_SEGMENTS  # noqa: F401


class NullAdapter(ArchAdapter):
    """Fallback: no arch-specific knowledge. TP is refused, not guessed."""

    @classmethod
    def shard_layer(cls, layer) -> list[str]:
        return []


_ADAPTERS: list[Type[ArchAdapter]] = []
_REGISTERED_NAMES: set[str] = set()


def register(adapter_cls: Type[ArchAdapter]) -> Type[ArchAdapter]:
    if adapter_cls.__name__ in _REGISTERED_NAMES:
        return adapter_cls
    _ADAPTERS.append(adapter_cls)
    _REGISTERED_NAMES.add(adapter_cls.__name__)
    return adapter_cls


def adapters() -> list[Type[ArchAdapter]]:
    if not _ADAPTERS:
        _autoload()
    return list(_ADAPTERS)


def _autoload() -> None:
    """Import adapter modules once so their register() calls run."""
    import importlib
    import pkgutil

    for mod in pkgutil.iter_modules(__path__):
        if mod.name in ("base", "__init__"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")


def adapter_for_config(config) -> Optional[Type[ArchAdapter]]:
    """Adapter matching a transformers config (by model_type), or None."""
    cfg = getattr(config, "text_config", config)
    for a in adapters():
        if a.matches_config(cfg):
            return a
    return None


def adapter_for_model(model) -> Optional[Type[ArchAdapter]]:
    """Adapter matching a model instance (by its config), or None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    return adapter_for_config(cfg)


def adapter_for_gguf_arch(arch: str) -> Optional[Type[ArchAdapter]]:
    """Adapter matching a llama.cpp ``general.architecture`` string, or None."""
    for a in adapters():
        if a.matches_gguf_arch(arch):
            return a
    return None
