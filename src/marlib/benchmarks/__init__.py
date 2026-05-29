"""Benchmark repository: directory-discovered specs + auto-registered builders.

Importing this package registers all built-in benchmark builders (one module
per benchmark). The set of *available* benchmarks is discovered from the data
directory (``discover``), while *how to build* each is looked up in the builder
registry (``get_builder``) — both keyed by the same name.
"""

from __future__ import annotations

from marlib.benchmarks.base import (
    DEFAULT_ROOT,
    BenchmarkBuilder,
    BenchmarkSpec,
    build_index,
    discover,
    get_builder,
    load_spec,
    registered_builders,
    slugify,
)

# Importing the builder modules triggers their @register side effects.
from marlib.benchmarks import financebench, hotpotqa  # noqa: E402,F401

__all__ = [
    "DEFAULT_ROOT",
    "BenchmarkBuilder",
    "BenchmarkSpec",
    "build_index",
    "discover",
    "get_builder",
    "load_spec",
    "registered_builders",
    "slugify",
]
