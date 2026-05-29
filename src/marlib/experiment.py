from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from marlib.adapters import AbstractAdapter, get_adapter_class
from marlib.benchmarks import BenchmarkSpec, slugify
from marlib.evaluation.llm_judge import llm_accuracy
from marlib.evaluation.metrics import evaluate_question
from marlib.log import logger
from marlib.retriever.core import Retriever
from marlib.tracing.schemas import QuestionLog, SystemResults


def load_benchmark(
    spec: BenchmarkSpec,
    sample_n: int | None = None,
) -> list[dict[str, Any]]:
    """Load a benchmark's questions from its directory (``spec.questions_path``).

    Args:
        spec: Benchmark spec (see :func:`marlib.benchmarks.load_spec`).
        sample_n: Optional cap on number of questions (None = full set).

    Returns:
        List of question dictionaries.
    """
    questions = []
    with open(spec.questions_path) as f:
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
    adapter_class = get_adapter_class(system_name)
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
                ids.append(f"{slugify(doc_name)}_p{page_num}")
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
