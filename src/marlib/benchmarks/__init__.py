"""Benchmark framework (harness).

This package provides the benchmark *contract*, *spec*, *registry* and shared
indexing only. The concrete benchmarks (recipe + data) live OUTSIDE the library,
under ``experiments/benchmarks/<name>/`` (``manifest.toml`` + ``builder.py`` +
generated data). ``discover(root)`` reads the manifests into specs and imports
each external ``builder.py`` so its ``@register`` fires.
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
