"""Build retrieval corpora via the builder registry.

Thin dispatcher: selects benchmarks (by name, or ``all`` discovered from the
data directory) and calls each registered builder's ``build_corpus``. The actual
per-benchmark logic lives in ``marlib.benchmarks.<name>``.
"""

from __future__ import annotations

import argparse

from marlib.benchmarks import discover, get_builder, load_spec


def main() -> None:
    """CLI entry point for corpus preparation."""
    parser = argparse.ArgumentParser(description="Build benchmark retrieval corpora")
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
        "--max-paragraphs",
        type=int,
        default=None,
        help="Optional cap on corpus size (for testing).",
    )
    args = parser.parse_args()

    names = (
        list(discover(args.data_dir)) if args.benchmark == "all" else [args.benchmark]
    )
    for name in names:
        spec = load_spec(name, args.data_dir)
        get_builder(name).build_corpus(spec, max_paragraphs=args.max_paragraphs)


if __name__ == "__main__":
    main()
