import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.systems.mas_zero.adapter import MASZeroAdapter
from experiments.systems.mas_zero import feedback


class Retriever:
    def retrieve(self, query, top_k):
        return [NS(doc_id="source_0", title="Title", text="Evidence", score=1.0)]

    def rerank(self, query, docs, top_k):
        return docs


def fake_model(monkeypatch, fitness="0.5", meta_code=None, fail_node=False):
    payloads = []
    def create(**kwargs):
        payloads.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "meticulous evaluator" in prompt:
            result = {"solvable": "true", "complete": "true", "fitness": fitness, "feedback": "Improve"}
        elif "judge selecting" in prompt:
            result = {"thinking": "Compare evidence", "selection": "1"}
        elif kwargs["model"] == "meta":
            result = {"name": "new", "thought": "try", "code": meta_code}
        else:
            if fail_node:
                raise ConnectionError("simulated infrastructure failure")
            result = {"thinking": "distinctive reasoning", "answer": "42"}
        return NS(usage=NS(prompt_tokens=10, completion_tokens=5),
                  choices=[NS(message=NS(content=json.dumps(result)))])
    def factory(**kw):
        assert kw["max_retries"] == 0
        return NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr("openai.OpenAI", factory)
    return payloads


def adapter(tmp_path, **kwargs):
    a = MASZeroAdapter(Retriever(), model="node", blocks=["COT"], **kwargs)
    a.set_benchmark_context("fake", "Description", ["unlabelled example"])
    a.set_run_context(artifact_dir=str(tmp_path), run_id="run", repeat=0, condition_id="test")
    return a


def test_native_ql_rejects_cl(tmp_path):
    with pytest.raises(ValueError, match="per_task only"):
        adapter(tmp_path, generation_mode="one_time")


def test_answer_extraction_uses_final_delimiter():
    assert MASZeroAdapter._extract_answer("First guess Answer: wrong\n\nAnswer:right\nsecond line") == "right\nsecond line"


def test_all_stages_no_gold_and_no_unused_meta_request(tmp_path, monkeypatch):
    # Deliberately omit the callback in generated code. BoundAgent supplies it.
    code = '''def forward(self, taskInfo):
    agent = LLMAgentBase(['thinking', 'answer'], 'generated')
    thinking, answer = agent([taskInfo], 'solve')
    return self.make_final_answer(thinking, answer)
'''
    payloads = fake_model(monkeypatch, meta_code=code)
    a = adapter(tmp_path, n_generation=1, meta_model="meta")
    answer, log = a.execute("../../q", "Question", "GOLD_SENTINEL_7391")
    assert answer == "42"
    assert "GOLD_SENTINEL_7391" not in json.dumps(payloads)
    assert log.gold_answer == "GOLD_SENTINEL_7391"
    assert len(payloads) == 6  # seed, feedback, proposal, generated, feedback, selection
    assert log.total_tokens == 90
    assert {call.phase for call in log.llm_calls} == {"construction", "answer_execution", "internal_verification"}
    assert sum(p["model"] == "meta" for p in payloads) == 1
    assert "distinctive reasoning" in payloads[-1]["messages"][1]["content"]
    trace = json.loads(Path(log.artifact_paths["trace"]).read_text())
    assert len(trace["candidates"]) == 2
    assert trace["selected_index"] == 1
    assert all(call.measurement == "api_attempt" for call in log.llm_calls)


def test_budget_preserves_answer_cost_and_unique_artifacts(tmp_path, monkeypatch):
    payloads = fake_model(monkeypatch)
    a = adapter(tmp_path, n_generation=1, resource_limits={"max_requests": 1})
    first_answer, first = a.execute("q", "Question", "gold")
    _, second = a.execute("q", "Question", "gold")
    assert first_answer == "42"
    assert first.status == "budget_exhausted"
    assert first.total_tokens == 15
    assert len(payloads) == 2  # one per question; no feedback/finalization after cap
    assert first.artifact_paths != second.artifact_paths
    assert Path(first.artifact_paths["trace"]).exists()
    assert first.resource_summary["candidate_failures"] == 1


def test_candidate_failure_is_visible(tmp_path, monkeypatch):
    fake_model(monkeypatch, fail_node=True)
    _, log = adapter(tmp_path, n_generation=0).execute("q", "Question", "gold")
    assert log.error
    assert log.resource_summary["candidate_failures"] == 1
    assert log.resource_summary["unknown_usage_attempts"] == 1
    assert log.status == "failed"


def test_invalid_selection_cannot_choose_empty_candidate(monkeypatch):
    monkeypatch.setattr(feedback, "_verifier_json", lambda *a, **k: {"selection": "0"})
    with pytest.raises(ValueError, match="unavailable"):
        feedback.self_verify("q", [{"answer": ""}, {"answer": "a"}, {"answer": "b"}], model="fake")
