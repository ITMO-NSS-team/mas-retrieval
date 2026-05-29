"""Typer CLI for running MAS retrieval benchmarks.

Defaults for every run parameter live here as flag defaults; the canonical way
to launch (and the place common runs are named) is the justfile, e.g.
`just run --benchmark hotpotqa ...`. Each invocation is frozen into a per-run
provenance directory (run_meta.json) and appended to results/runs.jsonl, so the
history of "what was benchmarked" is recoverable without digging through logs.

Cross-platform: launched via `just run` (and `python -m retcapslib.cli`), which
work on Linux, macOS and Windows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer

from retcapslib.experiment import (
    ADAPTERS,
    BENCHMARKS,
    load_adapter,
    load_benchmark,
    run_system_on_benchmark,
    save_results,
)
from retcapslib.logging.schemas import SystemResults
from retcapslib.retriever.core import init_retriever


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


def run(
    benchmark: str = typer.Option(
        "financebench", help=f"Benchmark preset. One of: {list(BENCHMARKS)}"
    ),
    model: str = typer.Option("openai/gpt-4o-mini", help="LLM model identifier."),
    systems: list[str] = typer.Option(
        ["fedotmas"],
        help=f"System(s) to run in this process. Available: {list(ADAPTERS)}",
    ),
    sample_n: Optional[int] = typer.Option(
        None, help="Cap on number of questions (default: full set)."
    ),
    generation_mode: Optional[str] = typer.Option(
        None, help="Generation mode forwarded to the adapter (e.g. shared)."
    ),
    retrieve_top_k: int = typer.Option(20, help="Candidates retrieved before rerank."),
    rerank_top_k: int = typer.Option(10, help="Documents kept after rerank."),
    embedder: str = typer.Option("BAAI/bge-m3", help="Embedder model name."),
    reranker: str = typer.Option("BAAI/bge-reranker-v2-m3", help="Reranker model name."),
    index_path: Path = typer.Option(
        Path("experiments/data/chroma_index"), help="Base ChromaDB index dir."
    ),
    data_dir: Path = typer.Option(
        Path("experiments/data/benchmarks"), help="Benchmark jsonl dir."
    ),
    results_dir: Path = typer.Option(Path("results"), help="Output root dir."),
    seed: int = typer.Option(42, help="Random seed (recorded in provenance)."),
    note: str = typer.Option("", help="Free-text note describing this run."),
) -> None:
    """Run one benchmark across the given system(s) and save results + provenance."""
    if benchmark not in BENCHMARKS:
        raise typer.BadParameter(
            f"Unknown benchmark '{benchmark}'. Available: {list(BENCHMARKS)}"
        )
    unknown = [s for s in systems if s not in ADAPTERS]
    if unknown:
        raise typer.BadParameter(
            f"Unknown system(s) {unknown}. Available: {list(ADAPTERS)}"
        )

    spec = BENCHMARKS[benchmark]

    # Unique run id avoids collisions between concurrent runs on one machine.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{benchmark}_{uuid4().hex[:6]}"
    out_dir = Path(results_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Run {run_id}")
    print(f"  benchmark={benchmark}  model={model}  systems={systems}")
    if note:
        print(f"  note: {note}")
    print("=" * 60)

    retriever_config = {
        "embedder": embedder,
        "reranker": reranker,
        "retrieve_top_k": retrieve_top_k,
        "rerank_top_k": rerank_top_k,
        "index_path": str(index_path),
        "collection_name": spec.collection_name,
    }

    print("Initializing retriever...")
    retriever = init_retriever(retriever_config)
    retriever.set_collection(spec.collection_name)
    print(
        f"  Collection '{spec.collection_name}': "
        f"{retriever._collection.count()} documents"
    )

    questions = load_benchmark(benchmark, sample_n=sample_n, data_dir=data_dir)
    print(f"  Loaded {len(questions)} questions")
    sample_qs = [q.get("question", "") for q in questions[:5]]

    adapter_kwargs = {}
    if generation_mode is not None:
        adapter_kwargs["generation_mode"] = generation_mode

    all_results: list[SystemResults] = []
    summaries: list[dict] = []

    for system_name in systems:
        print(f"\n{'=' * 40}\nSystem: {system_name}\n{'=' * 40}")
        adapter = load_adapter(system_name, retriever, model, **adapter_kwargs)
        adapter.set_benchmark_context(benchmark, spec.description, sample_qs)

        results = run_system_on_benchmark(
            adapter=adapter,
            questions=questions,
            benchmark_name=benchmark,
            model=model,
        )

        print(f"\n  Results for {system_name}/{benchmark}:")
        print(f"    EM:  {results.avg_exact_match:.3f}")
        print(f"    F1:  {results.avg_f1:.3f}")
        print(f"    ACC: {results.avg_llm_accuracy:.3f}")
        print(f"    CR:  {results.avg_context_recall:.3f}")
        print(
            f"    Tokens/Q: {results.avg_tokens_per_question:.0f}"
            f" (in: {results.avg_prompt_tokens_per_question:.0f},"
            f" out: {results.avg_completion_tokens_per_question:.0f})"
        )
        print(f"    Latency/Q: {results.avg_latency_ms:.0f}ms")

        save_results(results, out_dir)
        all_results.append(results)
        summaries.append(
            {
                "system": results.system_name,
                "avg_llm_accuracy": results.avg_llm_accuracy,
                "avg_f1": results.avg_f1,
                "avg_exact_match": results.avg_exact_match,
                "avg_context_recall": results.avg_context_recall,
                "avg_tokens": results.avg_tokens_per_question,
                "failed": results.failed_questions,
                "total": results.total_questions,
            }
        )

    # --- Provenance: freeze the exact invocation + context into the run dir ---
    params = {
        "benchmark": benchmark,
        "model": model,
        "systems": systems,
        "sample_n": sample_n,
        "generation_mode": generation_mode,
        "retrieve_top_k": retrieve_top_k,
        "rerank_top_k": rerank_top_k,
        "embedder": embedder,
        "reranker": reranker,
        "index_path": str(index_path),
        "data_dir": str(data_dir),
        "seed": seed,
    }
    run_meta = {
        "run_id": run_id,
        "timestamp": timestamp,
        "argv": sys.argv,
        "git_sha": _git_sha(),
        "note": note,
        "params": params,
        "summaries": summaries,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Append a one-line index entry for at-a-glance history.
    index_entry = {
        "run_id": run_id,
        "dir": str(out_dir),
        "benchmark": benchmark,
        "model": model,
        "systems": systems,
        "note": note,
        "git_sha": run_meta["git_sha"],
        "summaries": summaries,
    }
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "runs.jsonl", "a") as f:
        f.write(json.dumps(index_entry) + "\n")

    print(f"\nResults + provenance saved to: {out_dir}")
    print(f"History index: {Path(results_dir) / 'runs.jsonl'}")


def main() -> None:
    """Console-script entry point (`bench`). Single command, no subcommand."""
    typer.run(run)


if __name__ == "__main__":
    main()
