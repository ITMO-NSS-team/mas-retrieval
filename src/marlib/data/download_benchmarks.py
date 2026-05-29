from __future__ import annotations

import argparse

from marlib.benchmarks import discover, get_builder, load_spec
from marlib.log import logger


def main() -> None:
    """CLI entry point for downloading benchmark data."""
    parser = argparse.ArgumentParser(description="Download benchmark data")
    parser.add_argument(
        "--benchmark",
        "--dataset",
        dest="benchmark",
        default="all",
        help="Benchmark name, or 'all' (default: every discovered benchmark).",
    )
    parser.add_argument(
        "--data-dir",
        default="experiments/benchmarks",
        help="Benchmark repository root (one subdir per benchmark).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if questions.jsonl exists."
    )
    args = parser.parse_args()

    names = (
        list(discover(args.data_dir)) if args.benchmark == "all" else [args.benchmark]
    )
    for name in names:
        spec = load_spec(name, args.data_dir)
        if spec.questions_path.exists() and not args.force:
            logger.info(f"skip download: {name} (questions.jsonl exists)")
            continue
        get_builder(name).download(spec)


if __name__ == "__main__":
    main()
