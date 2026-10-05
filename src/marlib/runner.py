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
    checkpoint_path: Path | None = None,
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

    for raw_question in tqdm(questions, desc=f"{adapter.name}/{benchmark_name}"):
        q = raw_question if isinstance(raw_question, dict) else {}
        question_text = q.get("question", "")
        gold_answer = q.get("answer", "")
        try:
            if not isinstance(raw_question, dict):
                raise ValueError("Question record must be an object")
            predicted_answer, log = adapter.execute(
                question_id=q.get("id", "unknown"), question=question_text,
                gold_answer=gold_answer,
            )
        except Exception as e:
            logger.error(f"FAILED question {q.get('id')}: {e}")
            predicted_answer = ""
            log = QuestionLog(question_id=q.get("id", "unknown"),
                              question=question_text, gold_answer=gold_answer,
                              predicted_answer="", error=str(e), status="failed",
                              failure_kind="unknown",
                              resource_summary={"partial_usage_available": False})
        if log.error and log.status == "completed":
            log.status = "failed"
        retrieved = [doc_id for tc in log.tool_calls
                     if tc.tool_name in RETRIEVAL_TOOLS for doc_id in tc.results]
        judge_usage = {"prompt": 0, "completion": 0}
        ctx = EvalContext(question=question_text, predicted=predicted_answer,
                          gold=gold_answer, model=model, retrieved_doc_ids=retrieved,
                          gold_doc_ids=q.get("gold_doc_ids", []), judge_usage=judge_usage)
        for name in metrics:
            try:
                # An empty failed answer is wrong end-to-end, without a judge call.
                if log.error and not predicted_answer.strip() and name in {"llm_accuracy", "exact_match", "f1"}:
                    score = 0.0
                    log.metric_status[name] = "failed_answer_zero"
                else:
                    score = get_metric(name)(ctx)
                    log.metric_status[name] = "scored" if score is not None else "not_applicable"
                if score is not None:
                    log.metrics[name] = score
            except Exception as e:
                log.metric_status[name] = "judge_missing" if name == "llm_accuracy" else "metric_error"
                log.metric_errors[name] = str(e)
                logger.warning(f"Metric '{name}' failed for {q.get('id')}: {e}")
        log.judge_prompt_tokens = judge_usage["prompt"]
        log.judge_completion_tokens = judge_usage["completion"]
        question_logs.append(log)
        if log.error:
            results.failed_questions += 1
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path.open("a") as f:
                f.write(log.model_dump_json() + "\n")
                f.flush()

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
        results.metric_denominators[name] = len(scores)
        results.metric_missing[name] = len(logs) - len(scores)
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
