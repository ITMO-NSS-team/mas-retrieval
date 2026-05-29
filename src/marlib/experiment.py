"""Evaluation engine for MAS auto-generators on retrieval tasks.

Reusable building blocks, decoupled from the CLI surface (`cli.py`) so they can
be imported and tested on their own:
- ADAPTERS registry (available systems)
- BENCHMARKS registry (single source of truth for benchmark presets)
- load_benchmark / load_adapter
- run_system_on_benchmark (per-question execution + metric aggregation)
- save_results
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from marlib.adapters.base import AbstractAdapter
from marlib.adapters.fedotmas import FedotMASAdapter
from marlib.evaluation.llm_judge import llm_accuracy
from marlib.evaluation.metrics import evaluate_question
from marlib.log import logger
from marlib.retriever.core import Retriever
from marlib.tracing.schemas import QuestionLog, SystemResults


def _slugify(text: str) -> str:
    """Lowercase text and replace non-alphanumeric characters with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# Registry of available adapters
ADAPTERS: dict[str, type[AbstractAdapter]] = {
    # "naive_rag": NaiveRAGAdapter,
    # "single_agent": SingleAgentAdapter,
    # "swarm_agentic": SwarmAgenticAdapter,
    # "automas": AutoMASAdapter,
    "fedotmas": FedotMASAdapter,
    # "meta_agent": MetaAgentAdapter,
    # "mas_zero": MASZeroAdapter,
    # "ma_rag": MARagAdapter,
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static description of a benchmark — the only "preset" we keep in code."""

    collection_name: str
    file: str  # jsonl filename under the benchmarks data dir
    description: str
    split: str | None = None


# Single source of truth for benchmark presets (replaces per-benchmark YAML).
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "hotpotqa": BenchmarkSpec(
        collection_name="hotpotqa",
        file="hotpotqa_sample.jsonl",
        split="fullwiki_dev",
        description=(
            "Multi-hop question answering over Wikipedia. Questions require "
            "finding and reasoning over 2+ documents to produce a short "
            "factual answer (entity, yes/no, number, or short phrase)."
        ),
    ),
    "financebench": BenchmarkSpec(
        collection_name="financebench",
        file="financebench_sample.jsonl",
        description=(
            "Financial question answering over SEC filings and company reports. "
            "Questions require locating specific financial data and performing "
            "calculations or comparisons to produce precise numerical or factual answers."
        ),
    ),
}


def load_benchmark(
    benchmark_name: str,
    sample_n: int | None = None,
    data_dir: str | Path = "experiments/data/benchmarks",
) -> list[dict[str, Any]]:
    """Load a benchmark dataset from its preset in BENCHMARKS.

    Args:
        benchmark_name: Key into BENCHMARKS (e.g. hotpotqa, financebench).
        sample_n: Optional cap on number of questions (None = full set).
        data_dir: Directory containing benchmark jsonl files.

    Returns:
        List of question dictionaries.
    """
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARKS.keys())}"
        )

    filepath = Path(data_dir) / BENCHMARKS[benchmark_name].file

    questions = []
    with open(filepath) as f:
        for line in f:
            questions.append(json.loads(line))

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
                logger.warning(f"LLM judge failed for {question_id}: {e}")
                log.llm_accuracy = None

            question_logs.append(log)

            if log.error:
                results.failed_questions += 1

        except Exception as e:
            q_id = q.get("id", "unknown") if isinstance(q, dict) else "unknown"
            logger.error(f"FAILED question {q_id}: {e}")
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

    logger.info(f"Saved results to: {filepath}")
