from __future__ import annotations

from marlib.runner import _aggregate
from marlib.tracing.schemas import QuestionLog, SystemResults


def _log(qid: str, metrics: dict[str, float], **kw) -> QuestionLog:
    return QuestionLog(
        question_id=qid,
        question="q",
        gold_answer="g",
        predicted_answer="p",
        metrics=metrics,
        **kw,
    )


class TestAggregate:
    def test_no_logs_is_noop(self):
        results = SystemResults(system_name="s", benchmark="b", model="m")
        _aggregate(results, ("f1",))
        assert results.avg_metrics == {}

    def test_averages_metric(self):
        results = SystemResults(system_name="s", benchmark="b", model="m")
        results.question_logs = [
            _log("1", {"f1": 1.0}),
            _log("2", {"f1": 0.0}),
        ]
        _aggregate(results, ("f1",))
        assert results.avg_metrics["f1"] == 0.5

    def test_metric_missing_on_some_logs_is_skipped(self):
        # context_recall returned None on q2, so it is absent from that log;
        # the average is over only the logs that have it.
        results = SystemResults(system_name="s", benchmark="b", model="m")
        results.question_logs = [
            _log("1", {"context_recall": 1.0}),
            _log("2", {}),
        ]
        _aggregate(results, ("context_recall",))
        assert results.avg_metrics["context_recall"] == 1.0

    def test_metric_absent_everywhere_not_in_avg(self):
        results = SystemResults(system_name="s", benchmark="b", model="m")
        results.question_logs = [_log("1", {})]
        _aggregate(results, ("f1",))
        assert "f1" not in results.avg_metrics

    def test_token_and_call_averages(self):
        results = SystemResults(system_name="s", benchmark="b", model="m")
        results.question_logs = [
            _log("1", {}, total_tokens=100, num_llm_calls=2, num_retrieval_calls=1),
            _log("2", {}, total_tokens=200, num_llm_calls=4, num_retrieval_calls=3),
        ]
        _aggregate(results, ())
        assert results.avg_tokens_per_question == 150.0
        assert results.avg_llm_calls == 3.0
        assert results.avg_retrieval_calls == 2.0
