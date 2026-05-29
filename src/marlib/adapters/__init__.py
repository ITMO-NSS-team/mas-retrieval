"""Adapters for different MAS auto-generator systems.

Each adapter self-registers via ``@register`` (see ``base.py``). Importing this
package walks the adapter modules and imports each **defensively**: an adapter
whose (often heavy / optional) dependencies are missing is skipped, not crashed
on and not commented out. ``available_adapters()`` therefore reflects exactly
what can run in the current environment.

Convention: a single-file adapter is its top-level module (e.g. ``naive_rag``);
a package adapter lives in ``<name>/adapter.py``. Drop in a new module/package
following that convention and it is discovered automatically.
"""

from __future__ import annotations

import importlib
import pkgutil

from marlib.adapters.base import (
    AbstractAdapter,
    available_adapters,
    get_adapter_class,
    register,
)
from marlib.log import logger

# Infrastructure modules that are not adapters.
_NON_ADAPTER = {"base", "tools", "registry"}


def _load_adapters() -> None:
    """Import every adapter module, skipping those with unmet dependencies."""
    for info in pkgutil.iter_modules(__path__):
        if info.name in _NON_ADAPTER:
            continue
        target = (
            f"{__name__}.{info.name}.adapter"
            if info.ispkg
            else f"{__name__}.{info.name}"
        )
        try:
            importlib.import_module(target)
        except Exception as e:  # missing optional deps, import-time errors, etc.
            logger.debug(f"Adapter '{info.name}' unavailable: {e!r}")


_load_adapters()

__all__ = [
    "AbstractAdapter",
    "available_adapters",
    "get_adapter_class",
    "register",
]
