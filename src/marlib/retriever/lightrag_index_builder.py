from __future__ import annotations

import argparse

from marlib.benchmarks import build_lightrag_index, discover, load_spec
from marlib.log import logger
from marlib.retriever.config import DEFAULT_EMBEDDER, RetrieverSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LightRAG indexes for benchmarks")
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
        "--embedder-backend",
        default=RetrieverSettings.model_fields["embedder_backend"].default,
        choices=["local", "openai"],
        help=(
            "Embedding backend: local FlagEmbedding or OpenAI-compatible API "
            f"(default: {RetrieverSettings.model_fields['embedder_backend'].default})."
        ),
    )
    parser.add_argument(
        "--embedder-base-url",
        default=None,
        help="Base URL for OpenAI-compatible embeddings, e.g. http://localhost:8001/v1.",
    )
    parser.add_argument(
        "--embedder-api-key",
        default=None,
        help="API key for OpenAI-compatible embeddings (default: OPENAI_API_KEY or dummy).",
    )
    parser.add_argument(
        "--embedder-dim",
        type=int,
        default=RetrieverSettings.model_fields["embedder_dim"].default,
        help=f"Embedding dimension for LightRAG storage (default: {RetrieverSettings.model_fields['embedder_dim'].default}).",
    )
    parser.add_argument(
        "--llm-model",
        default=RetrieverSettings.model_fields["lightrag_llm_model"].default,
        help=(
            "LLM model used by LightRAG entity/relation extraction "
            f"(default: {RetrieverSettings.model_fields['lightrag_llm_model'].default})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of corpus documents passed to each LightRAG insert call.",
    )
    parser.add_argument(
        "--limit",
        "--max-documents",
        dest="max_documents",
        type=int,
        default=None,
        help="Index only the first N corpus documents, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-parallel-insert",
        type=int,
        default=2,
        help="LightRAG concurrent document insertion pipeline width.",
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
            logger.info(f"skip lightrag index: {name} (no corpus.jsonl)")
            continue

        index_path = spec.lightrag_index_path / spec.collection
        marker = index_path / "marlib_lightrag_index.json"
        if marker.exists() and not args.force:
            logger.info(f"skip lightrag index: {name} (index already exists)")
            continue

        build_lightrag_index(
            spec,
            embedder_model=args.model,
            embedder_backend=args.embedder_backend,
            embedder_base_url=args.embedder_base_url,
            embedder_api_key=args.embedder_api_key,
            embedder_dim=args.embedder_dim,
            llm_model=args.llm_model,
            batch_size=args.batch_size,
            max_documents=args.max_documents,
            max_parallel_insert=args.max_parallel_insert,
            clear_existing=args.force,
        )


if __name__ == "__main__":
    main()
