from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from marlib.adapters import AbstractAdapter
from marlib.evaluation import EvalContext, get_metric
from marlib.evaluation.base import RETRIEVAL_TOOLS
from marlib.log import logger
from marlib.tracing.schemas import QuestionLog, SystemResults


def run_system_on_benchmark(
    adapter: AbstractAdapter,
    questions: list[dict],
    benchmark_name: str,
    model: str,
    metrics: tuple[str, ...],
) -> SystemResults:
    """Run one system over a benchmark, scoring each question with the benchmark's
    declared ``metrics``. A metric returning ``None`` is omitted from the averages."""
    results = SystemResults(
        system_name=adapter.name,
        benchmark=benchmark_name,
        model=model,
        total_questions=len(questions),
    )
    question_logs: list[QuestionLog] = []

    for q in tqdm(questions, desc=f"{adapter.name}/{benchmark_name}"):
        try:
            question_text = q.get("question", "")
            gold_answer = q.get("answer", "")
            predicted_answer, log = adapter.execute(
                question_id=q.get("id", "unknown"),
                question=question_text,
                gold_answer=gold_answer,
            )

            retrieved = [
                doc_id
                for tc in log.tool_calls
                if tc.tool_name in RETRIEVAL_TOOLS
                for doc_id in tc.results
            ]
            ctx = EvalContext(
                question=question_text,
                predicted=predicted_answer,
                gold=gold_answer,
                model=model,
                retrieved_doc_ids=retrieved,
                gold_doc_ids=q.get("gold_doc_ids", []),
            )
            for name in metrics:
                try:
                    score = get_metric(name)(ctx)
                except Exception as e:
                    logger.warning(f"Metric '{name}' failed for {q.get('id')}: {e}")
                    score = None
                if score is not None:
                    log.metrics[name] = score

            question_logs.append(log)
            if log.error:
                results.failed_questions += 1

        except Exception as e:
            q_id = q.get("id", "unknown") if isinstance(q, dict) else "unknown"
            logger.error(f"FAILED question {q_id}: {e}")
            question_logs.append(
                QuestionLog(
                    question_id=q_id,
                    question=q.get("question", "") if isinstance(q, dict) else "",
                    gold_answer=q.get("answer", "") if isinstance(q, dict) else "",
                    predicted_answer="",
                    error=str(e),
                )
            )
            results.failed_questions += 1

    results.question_logs = question_logs
    _aggregate(results, metrics)
    return results


def _aggregate(results: SystemResults, metrics: tuple[str, ...]) -> None:
    """Fill ``results`` averages from its question logs (in place)."""
    logs = results.question_logs
    if not logs:
        return

    for name in metrics:
        scores = [L.metrics[name] for L in logs if name in L.metrics]
        if scores:
            results.avg_metrics[name] = sum(scores) / len(scores)

    n = len(logs)
    results.avg_tokens_per_question = sum(L.total_tokens for L in logs) / n
    results.avg_prompt_tokens_per_question = sum(L.total_prompt_tokens for L in logs) / n
    results.avg_completion_tokens_per_question = (
        sum(L.total_completion_tokens for L in logs) / n
    )
    results.avg_retrieval_calls = sum(L.num_retrieval_calls for L in logs) / n
    results.avg_llm_calls = sum(L.num_llm_calls for L in logs) / n
    results.avg_latency_ms = sum(L.total_latency_ms for L in logs) / n


def save_results(results: SystemResults, output_dir: str | Path) -> None:
    """Write ``results`` as JSON into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{results.system_name}_{results.benchmark}_{results.model.replace('/', '_')}.json"
    )
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(results.model_dump(), f, indent=2)
    logger.info(f"Saved results to: {filepath}")
