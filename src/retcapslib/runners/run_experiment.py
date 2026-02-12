"""Main experiment runner for evaluating MAS auto-generators on retrieval tasks.

Orchestrates:
1. Loading benchmarks (HotpotQA, FinanceBench)
2. Initializing retriever with ChromaDB index
3. Running each system adapter on each benchmark
4. Computing metrics and saving results
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

# from retcapslib.adapters.automas import AutoMASAdapter
from retcapslib.adapters.base import AbstractAdapter
from retcapslib.adapters.ma_rag import MARagAdapter
from retcapslib.adapters.mas_zero import MASZeroAdapter
from retcapslib.adapters.meta_agent import MetaAgentAdapter
from retcapslib.adapters.naive_rag import NaiveRAGAdapter
from retcapslib.adapters.single_agent import SingleAgentAdapter
from retcapslib.adapters.swarm_agentic import SwarmAgenticAdapter
from retcapslib.evaluation.llm_judge import llm_accuracy
from retcapslib.evaluation.metrics import evaluate_question
from retcapslib.logging.schemas import QuestionLog, SystemResults
from retcapslib.retriever.core import Retriever, init_retriever


def _slugify(text: str) -> str:
    """Lowercase text and replace non-alphanumeric characters with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# Registry of available adapters
ADAPTERS: dict[str, type[AbstractAdapter]] = {
    "naive_rag": NaiveRAGAdapter,
    "single_agent": SingleAgentAdapter,
    "swarm_agentic": SwarmAgenticAdapter,
    # "automas": AutoMASAdapter,
    "meta_agent": MetaAgentAdapter,
    "mas_zero": MASZeroAdapter,
    "ma_rag": MARagAdapter,
}


_DEFAULT_BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    "hotpotqa": (
        "Multi-hop question answering over Wikipedia. Questions require "
        "finding and reasoning over 2+ documents to produce a short "
        "factual answer (entity, yes/no, number, or short phrase)."
    ),
    "financebench": (
        "Financial question answering over SEC filings and company reports. "
        "Questions require locating specific financial data and performing "
        "calculations or comparisons to produce precise numerical or factual answers."
    ),
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_benchmark(
    benchmark_name: str,
    config: dict[str, Any],
    data_dir: str | Path = "experiments/data/benchmarks",
) -> list[dict[str, Any]]:
    """Load a benchmark dataset.

    Args:
        benchmark_name: Name of the benchmark (hotpotqa, musique).
        config: Experiment configuration.
        data_dir: Directory containing benchmark files.

    Returns:
        List of question dictionaries.
    """
    data_dir = Path(data_dir)

    if benchmark_name == "hotpotqa":
        filepath = data_dir / "hotpotqa_sample.jsonl"
    elif benchmark_name == "financebench":
        filepath = data_dir / "financebench_sample.jsonl"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    questions = []
    with open(filepath) as f:
        for line in f:
            questions.append(json.loads(line))

    # Apply sample_n limit from benchmark config
    benchmark_config = config.get("benchmarks", {}).get(benchmark_name, {})
    sample_n = benchmark_config.get("sample_n")

    if sample_n and len(questions) > sample_n:
        questions = questions[:sample_n]

    return questions


def load_adapter(
    system_name: str,
    retriever: Retriever,
    model: str,
    **kwargs: Any,
) -> AbstractAdapter:
    """Load and initialize an adapter by name.

    Args:
        system_name: Name of the system (e.g., naive_rag, single_agent).
        retriever: Initialized Retriever instance.
        model: LLM model to use.
        **kwargs: Additional options (e.g., generation_mode).

    Returns:
        Initialized adapter instance.
    """
    if system_name not in ADAPTERS:
        raise ValueError(
            f"Unknown system: {system_name}. Available: {list(ADAPTERS.keys())}"
        )

    adapter_class = ADAPTERS[system_name]
    return adapter_class(retriever=retriever, model=model, **kwargs)


def extract_gold_paragraphs(question: dict[str, Any]) -> list[str]:
    """Extract gold supporting paragraphs from question metadata.

    Args:
        question: Question dictionary with benchmark-specific fields.

    Returns:
        List of gold paragraph identifiers (typically article titles).
    """
    # HotpotQA format
    if "supporting_facts" in question:
        return list(set(question["supporting_facts"]["titles"]))

    # FinanceBench format - evidence field contains supporting pages
    if "evidence" in question:
        evidence_list = question["evidence"]
        if not isinstance(evidence_list, list):
            evidence_list = [evidence_list]
        ids = []
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
            doc_name = ev.get("doc_name", question.get("doc_name", ""))
            page_num = ev.get("evidence_page_num")
            if doc_name and page_num is not None:
                ids.append(f"{_slugify(doc_name)}_p{page_num}")
        return ids

    return []


def run_system_on_benchmark(
    adapter: AbstractAdapter,
    questions: list[dict[str, Any]],
    benchmark_name: str,
    model: str,
) -> SystemResults:
    """Run a system adapter on a benchmark dataset.

    Args:
        adapter: Initialized adapter instance.
        questions: List of question dictionaries.
        benchmark_name: Name of the benchmark.
        model: LLM model being used.

    Returns:
        SystemResults with all question logs and aggregate metrics.
    """
    results = SystemResults(
        system_name=adapter.name,
        benchmark=benchmark_name,
        model=model,
        total_questions=len(questions),
    )

    question_logs: list[QuestionLog] = []

    for q in tqdm(questions, desc=f"{adapter.name}/{benchmark_name}"):
        try:
            question_id = q.get("id", "unknown")
            question_text = q.get("question", "")
            gold_answer = q.get("answer", "")

            # Execute the adapter
            predicted_answer, log = adapter.execute(
                question_id=question_id,
                question=question_text,
                gold_answer=gold_answer,
            )

            # Evaluate answer quality
            gold_paragraphs = extract_gold_paragraphs(q)
            retrieved_doc_ids = [tc.results for tc in log.tool_calls]
            all_retrieved = [
                doc_id for doc_ids in retrieved_doc_ids for doc_id in doc_ids
            ]

            metrics = evaluate_question(
                predicted_answer=predicted_answer,
                gold_answer=gold_answer,
                retrieved_doc_ids=all_retrieved,
                gold_paragraphs=gold_paragraphs,
            )

            # Update log with metrics
            log.exact_match = metrics["exact_match"]
            log.f1_score = metrics["f1"]
            log.context_recall = metrics.get("context_recall")

            # LLM-as-a-judge accuracy
            try:
                log.llm_accuracy = llm_accuracy(
                    question=question_text,
                    predicted=predicted_answer,
                    gold=gold_answer,
                    model_name=model,
                )
            except Exception as e:
                print(f"  LLM judge failed for {question_id}: {e}")
                log.llm_accuracy = None

            question_logs.append(log)

            if log.error:
                results.failed_questions += 1

        except Exception as e:
            q_id = q.get("id", "unknown") if isinstance(q, dict) else "unknown"
            print(f"  FAILED question {q_id}: {e}")
            error_log = QuestionLog(
                question_id=q_id,
                question=q.get("question", "") if isinstance(q, dict) else "",
                gold_answer=q.get("answer", "") if isinstance(q, dict) else "",
                predicted_answer="",
                error=str(e),
            )
            question_logs.append(error_log)
            results.failed_questions += 1

    # Compute aggregate metrics
    results.question_logs = question_logs

    if question_logs:
        results.avg_exact_match = sum(L.exact_match or 0 for L in question_logs) / len(
            question_logs
        )
        results.avg_f1 = sum(L.f1_score or 0 for L in question_logs) / len(
            question_logs
        )

        recall_scores = [
            L.context_recall for L in question_logs if L.context_recall is not None
        ]
        if recall_scores:
            results.avg_context_recall = sum(recall_scores) / len(recall_scores)

        # LLM-as-a-judge accuracy
        acc_scores = [
            L.llm_accuracy for L in question_logs if L.llm_accuracy is not None
        ]
        if acc_scores:
            results.avg_llm_accuracy = sum(acc_scores) / len(acc_scores)

        results.avg_tokens_per_question = sum(
            L.total_tokens for L in question_logs
        ) / len(question_logs)
        results.avg_prompt_tokens_per_question = sum(
            L.total_prompt_tokens for L in question_logs
        ) / len(question_logs)
        results.avg_completion_tokens_per_question = sum(
            L.total_completion_tokens for L in question_logs
        ) / len(question_logs)
        results.avg_retrieval_calls = sum(
            L.num_retrieval_calls for L in question_logs
        ) / len(question_logs)
        results.avg_llm_calls = sum(L.num_llm_calls for L in question_logs) / len(
            question_logs
        )
        results.avg_latency_ms = sum(L.total_latency_ms for L in question_logs) / len(
            question_logs
        )

    return results


def save_results(
    results: SystemResults,
    output_dir: str | Path,
) -> None:
    """Save system results to disk.

    Args:
        results: SystemResults to save.
        output_dir: Directory for output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    filename = f"{results.system_name}_{results.benchmark}_{results.model.replace('/', '_')}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(results.model_dump(), f, indent=2)

    print(f"Saved results to: {filepath}")


def run_experiment(config_path: str | Path) -> None:
    """Run the full experiment as defined in config.

    Args:
        config_path: Path to experiment config YAML.
    """
    config = load_config(config_path)

    print("=" * 60)
    print("SIGIR 2026 Experiment Runner")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Systems: {config['systems']}")
    print(f"Benchmarks: {list(config['benchmarks'].keys())}")
    print(f"Primary model: {config['models']['primary']}")
    print()

    # Initialize retriever (uses first benchmark's collection as starting point)
    print("Initializing retriever...")
    first_benchmark = next(iter(config["benchmarks"].values()))
    retriever_config = dict(config["retriever"])
    retriever_config["collection_name"] = first_benchmark.get(
        "collection_name", "wikipedia"
    )
    retriever = init_retriever(retriever_config)
    print(f"  Collection loaded with {retriever._collection.count()} documents")
    print()

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["output"]["results_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config to results dir
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Run each system on each benchmark
    model = config["models"]["primary"]
    all_results: list[SystemResults] = []

    for system_entry in config["systems"]:
        try:
            if isinstance(system_entry, str):
                system_name, system_opts = system_entry, {}
            else:
                system_opts = dict(system_entry)
                system_name = system_opts.pop("name")

            if system_name not in ADAPTERS:
                print(f"Skipping unknown system: {system_name}")
                continue

            print(f"\n{'=' * 40}")
            print(f"System: {system_name}")
            if system_opts:
                print(f"Options: {system_opts}")
            print("=" * 40)

            adapter = load_adapter(system_name, retriever, model, **system_opts)

            for benchmark_name, benchmark_cfg in config["benchmarks"].items():
                print(f"\nBenchmark: {benchmark_name}")

                # Switch to the benchmark-specific collection
                collection_name = benchmark_cfg.get("collection_name", "wikipedia")
                retriever.set_collection(collection_name)
                print(
                    f"  Collection: {collection_name} ({retriever._collection.count()} docs)"
                )

                questions = load_benchmark(benchmark_name, config)
                print(f"  Loaded {len(questions)} questions")

                # Provide benchmark context to adapter for shared-mode generation
                description = benchmark_cfg.get(
                    "description",
                    _DEFAULT_BENCHMARK_DESCRIPTIONS.get(benchmark_name, ""),
                )
                sample_qs = [q.get("question", "") for q in questions[:5]]
                adapter.set_benchmark_context(benchmark_name, description, sample_qs)

                results = run_system_on_benchmark(
                    adapter=adapter,
                    questions=questions,
                    benchmark_name=benchmark_name,
                    model=model,
                )

                # Print summary
                print(f"\n  Results for {system_name}/{benchmark_name}:")
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

                save_results(results, results_dir)
                all_results.append(results)

        except Exception as e:
            print(f"\nFATAL: System '{system_entry}' failed: {e}")
            continue

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(
            f"{r.system_name:15} | {r.benchmark:12}"
            f" | EM={r.avg_exact_match:.3f} | F1={r.avg_f1:.3f}"
            f" | ACC={r.avg_llm_accuracy:.3f}"
            f" | Tok={r.avg_tokens_per_question:.0f}"
            f" (in:{r.avg_prompt_tokens_per_question:.0f}"
            f" out:{r.avg_completion_tokens_per_question:.0f})"
        )

    print(f"\nResults saved to: {results_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SIGIR 2026 MAS evaluation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/retcapslib/cfg_test_financebench.yaml",
        help="Path to experiment config",
    )

    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
