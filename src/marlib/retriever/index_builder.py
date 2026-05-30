from __future__ import annotations

import argparse

from marlib.benchmarks import build_index, discover, load_spec
from marlib.log import logger
from marlib.retriever.config import DEFAULT_EMBEDDER


def main() -> None:
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
        default="experiments/benchmarks",
        help="Benchmark repository root (one subdir per benchmark).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDER,
        help=f"Embedder model name (default: {DEFAULT_EMBEDDER}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (BGE-M3 uses ~3-4GB VRAM, safe for 16GB GPUs).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-index even if an index already exists."
    )
    args = parser.parse_args()

    names = (
        list(discover(args.data_dir)) if args.benchmark == "all" else [args.benchmark]
    )
    for name in names:
        spec = load_spec(name, args.data_dir)
        if not spec.corpus_path.exists():
            logger.info(f"skip index: {name} (no corpus.jsonl)")
            continue
        if (spec.index_path / spec.collection / "chroma.sqlite3").exists() and not args.force:
            logger.info(f"skip index: {name} (index already exists)")
            continue
        build_index(spec, embedder_model=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
