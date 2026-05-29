"""Build ChromaDB indexes via the builder registry.

Thin dispatcher around :func:`marlib.benchmarks.build_index` (the indexing
pipeline is identical for every benchmark). Selects benchmarks by name, or
``all`` discovered from the data directory, and indexes each one's corpus.
"""

from __future__ import annotations

import argparse

from marlib.benchmarks import build_index, discover, load_spec


def main() -> None:
    """CLI entry point for building indexes."""
    parser = argparse.ArgumentParser(description="Build ChromaDB indexes for benchmarks")
    parser.add_argument(
        "--benchmark",
        "--dataset",
        dest="benchmark",
        default="all",
        help="Benchmark name, or 'all' (default: every discovered benchmark).",
    )
    parser.add_argument(
        "--data-dir",
        default="experiments/data/benchmarks",
        help="Benchmark repository root (one subdir per benchmark).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="Embedder model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (BGE-M3 uses ~3-4GB VRAM, safe for 16GB GPUs).",
    )
    args = parser.parse_args()

    names = (
        list(discover(args.data_dir)) if args.benchmark == "all" else [args.benchmark]
    )
    for name in names:
        spec = load_spec(name, args.data_dir)
        build_index(spec, embedder_model=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
