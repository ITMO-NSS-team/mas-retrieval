import json

from marlib.runner import run_system_on_benchmark
from marlib.tracing.schemas import LLMCall, QuestionLog, SystemResults
from marlib import reporting


class Adapter:
    name = "fake"

    def execute(self, question_id, question, gold_answer):
        failed = question_id == "failed"
        answer = "" if failed else "a"
        return answer, QuestionLog(question_id=question_id, question=question,
                                   gold_answer=gold_answer, predicted_answer=answer,
                                   error="budget" if failed else None,
                                   total_tokens=13, status="budget_exhausted" if failed else "completed")


def test_failed_empty_answer_zero_but_missing_judge_not_zero(tmp_path, monkeypatch):
    calls = []
    def metric(ctx):
        calls.append(ctx.question)
        raise ConnectionError("judge unavailable")
    monkeypatch.setattr("marlib.runner.get_metric", lambda name: metric)
    checkpoint = tmp_path / "questions.jsonl"
    result = run_system_on_benchmark(Adapter(), [
        {"id": "failed", "question": "first", "answer": "a"},
        {"id": "answered", "question": "second", "answer": "a"},
    ], "fake", "model", ("llm_accuracy",), checkpoint)
    assert calls == ["second"]
    assert result.metric_denominators == {"llm_accuracy": 1}
    assert result.metric_missing == {"llm_accuracy": 1}
    assert result.question_logs[1].metric_status["llm_accuracy"] == "judge_missing"
    assert result.avg_tokens_per_question == 13
    assert len([json.loads(s) for s in checkpoint.read_text().splitlines()]) == 2


def test_cost_uses_each_model_and_marks_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(reporting, "run_cost", lambda m, i, o: (calls.append((m, i, o)), i + o)[1])
    q = QuestionLog(question_id="q", question="q", gold_answer="g", predicted_answer="a",
                    llm_calls=[LLMCall(model="node", prompt_tokens=10, completion_tokens=3, latency_ms=1),
                               LLMCall(model="meta", prompt_tokens=20, completion_tokens=4, latency_ms=1)])
    result = SystemResults(system_name="s", benchmark="b", model="node", question_logs=[q])
    assert reporting.system_cost(result, "node") == 37
    assert calls == [("node", 10, 3), ("meta", 20, 4)]
    q.llm_calls[0].usage_known = False
    assert reporting.system_cost(result, "node") is None


def test_malformed_question_does_not_abort_remaining_questions():
    result = run_system_on_benchmark(Adapter(), [None, {"id": "valid", "question": "q", "answer": "a"}],
                                     "fake", "model", ("exact_match",))
    assert len(result.question_logs) == 2
    assert result.failed_questions == 1
    assert result.question_logs[1].metrics["exact_match"] == 1
