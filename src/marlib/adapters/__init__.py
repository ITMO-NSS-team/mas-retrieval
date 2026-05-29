"""Adapter framework (harness).

This package provides the adapter *contract* and *registry* only — the concrete
adapters (the systems under test) live OUTSIDE the library, under
``experiments/systems/<name>/``, and are picked up at runtime by
``discover_adapters(root)`` (path-based discovery). Each external adapter
self-registers via ``@register`` (see ``base.py``); one whose dependencies are
missing is simply skipped.

``tools.py`` here are shared retrieval/calculator helpers that external adapters
import (content depends on harness, never the reverse).
"""

from __future__ import annotations

from marlib.adapters.base import (
    DEFAULT_SYSTEMS_ROOT,
    AbstractAdapter,
    available_adapters,
    discover_adapters,
    get_adapter_class,
    register,
)

__all__ = [
    "DEFAULT_SYSTEMS_ROOT",
    "AbstractAdapter",
    "available_adapters",
    "discover_adapters",
    "get_adapter_class",
    "register",
]
