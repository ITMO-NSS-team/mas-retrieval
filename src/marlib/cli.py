from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from marlib.adapters import discover_adapters, get_adapter_class
from marlib.benchmarks import discover, load_spec
from marlib.log import logger
from marlib.retriever import Retriever, RetrieverSettings
from marlib.runner import run_system_on_benchmark, save_results
from marlib.tracing.schemas import SystemResults


def _git_sha() -> str | None:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    # Discovered from the default content roots, for the --help listings only.
    available = list(discover())
    systems = discover_adapters()

    parser = argparse.ArgumentParser(
        description="Run one benchmark across the given system(s) and save "
        "results + provenance."
    )
    parser.add_argument(
        "--benchmark",
        default="financebench",
        help=f"Benchmark (discovered from --data-dir). Available: {available}",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="LLM model identifier."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["fedotmas"],
        help=f"System(s) to run in this process. Available: {systems}",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Cap on number of questions (default: full set).",
    )
    parser.add_argument(
        "--generation-mode",
        default=None,
        help="Generation mode forwarded to the adapter (e.g. shared).",
    )
    # Defaults for these live in RetrieverSettings (single source of truth); a
    # flag left unset (None) keeps that default rather than overriding it.
    rs = RetrieverSettings.model_fields
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help=f"Candidates retrieved before rerank (default: {rs['retrieve_top_k'].default}).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=f"Documents kept after rerank (default: {rs['rerank_top_k'].default}).",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help=f"Embedder model name (default: {rs['embedder'].default}).",
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help=f"Reranker model name (default: {rs['reranker'].default}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/benchmarks"),
        help="Benchmark repository root (one subdir per benchmark).",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=Path("experiments/systems"),
        help="Systems (adapters) repository root (one subdir per system).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output root dir."
    )
    parser.add_argument(
        "--note", default="", help="Free-text note describing this run."
    )
    return parser


def main() -> None:
    """Console-script entry point (`bench`)."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        spec = load_spec(args.benchmark, args.data_dir)
    except ValueError as e:
        parser.error(str(e))
    available = discover_adapters(args.systems_dir)
    unknown = [s for s in args.systems if s not in available]
    if unknown:
        parser.error(f"Unknown system(s) {unknown}. Available: {available}")

    # Unique run id avoids collisions between concurrent runs on one machine.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.benchmark}_{uuid4().hex[:6]}"
    out_dir = args.results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting run {run_id}",
        benchmark=args.benchmark,
        model=args.model,
        systems=args.systems,
        note=args.note or None,
    )

    overrides = {
        k: v
        for k, v in {
            "embedder": args.embedder,
            "reranker": args.reranker,
            "retrieve_top_k": args.retrieve_top_k,
            "rerank_top_k": args.rerank_top_k,
        }.items()
        if v is not None
    }
    settings = RetrieverSettings(
        index_path=spec.index_path, collection=spec.collection, **overrides
    )
    # Export so any MCP server spawned by an adapter reconstructs this same config.
    settings.export_env()

    logger.info("Initializing retriever...")
    retriever = Retriever(settings)
    logger.info(
        f"Collection '{spec.collection}' loaded",
        documents=retriever.document_count,
    )

    questions = spec.load_questions(sample_n=args.sample_n)
    logger.info(f"Loaded {len(questions)} questions")
    sample_qs = [q.get("question", "") for q in questions[:5]]

    adapter_kwargs = {}
    if args.generation_mode is not None:
        adapter_kwargs["generation_mode"] = args.generation_mode

    all_results: list[SystemResults] = []
    summaries: list[dict] = []

    for system_name in args.systems:
        logger.info(f"Running system: {system_name}")
        adapter = get_adapter_class(system_name)(
            retriever=retriever, model=args.model, **adapter_kwargs
        )
        adapter.set_benchmark_context(args.benchmark, spec.description, sample_qs)

        results = run_system_on_benchmark(
            adapter=adapter,
            questions=questions,
            benchmark_name=args.benchmark,
            model=args.model,
            metrics=spec.metrics,
        )

        logger.success(
            f"Results for {system_name}/{args.benchmark}",
            **{k: round(v, 3) for k, v in results.avg_metrics.items()},
            tokens_per_q=round(results.avg_tokens_per_question),
            prompt_tokens_per_q=round(results.avg_prompt_tokens_per_question),
            completion_tokens_per_q=round(results.avg_completion_tokens_per_question),
            latency_ms=round(results.avg_latency_ms),
        )

        save_results(results, out_dir)
        all_results.append(results)
        summaries.append(
            {
                "system": results.system_name,
                "metrics": results.avg_metrics,
                "avg_tokens": results.avg_tokens_per_question,
                "failed": results.failed_questions,
                "total": results.total_questions,
            }
        )

    params = {
        "benchmark": args.benchmark,
        "model": args.model,
        "systems": args.systems,
        "sample_n": args.sample_n,
        "generation_mode": args.generation_mode,
        "retrieve_top_k": settings.retrieve_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "embedder": settings.embedder,
        "reranker": settings.reranker,
        "index_path": str(settings.index_path),
        "data_dir": str(args.data_dir),
        "systems_dir": str(args.systems_dir),
        "metrics": list(spec.metrics),
    }
    run_meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "argv": sys.argv,
        "git_sha": _git_sha(),
        "note": args.note,
        "params": params,
        "summaries": summaries,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Append a one-line index entry for at-a-glance history.
    index_entry = {
        "run_id": run_id,
        "dir": str(out_dir),
        "benchmark": args.benchmark,
        "model": args.model,
        "systems": args.systems,
        "note": args.note,
        "git_sha": run_meta["git_sha"],
        "summaries": summaries,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(index_entry) + "\n")

    logger.success(
        "Run complete",
        results_dir=str(out_dir),
        history_index=str(args.results_dir / "runs.jsonl"),
    )


if __name__ == "__main__":
    main()
