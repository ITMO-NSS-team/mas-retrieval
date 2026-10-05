import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.systems.mas_zero.adapter import MASZeroAdapter
from experiments.systems.mas_zero import feedback
from experiments.systems.mas_zero.prompts import EXAMPLE


class Retriever:
    def retrieve(self, query, top_k):
        return [NS(doc_id="source_0", title="Title", text="CORPUS_EVIDENCE_SENTINEL", score=1.0)]

    def rerank(self, query, docs, top_k):
        return docs


def fake_model(monkeypatch, fitness="0.5", meta_code=None, fail_node=False, node_outputs=None):
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
            result = {"thinking": "distinctive reasoning", "answer": "42",
                      "feedback": "distinctive critique", "correct": "False"}
            if node_outputs is not None:
                result["thinking"] += f" {len(node_outputs)}"
                result["feedback"] += f" {len(node_outputs)}"
                node_outputs.append(result.copy())
        return NS(usage=NS(prompt_tokens=10, completion_tokens=5),
                  choices=[NS(message=NS(content=json.dumps(result)))])
    def factory(**kw):
        assert kw["max_retries"] == 0
        return NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr("openai.OpenAI", factory)
    class AsyncClient:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            async def request(**payload):
                return create(**payload)
            self.chat = NS(completions=NS(create=request))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr("openai.AsyncOpenAI", AsyncClient)
    return payloads


def adapter(tmp_path, **kwargs):
    kwargs.setdefault("blocks", ["COT"])
    a = MASZeroAdapter(Retriever(), model="node", **kwargs)
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


def test_prompt_example_retrieves_before_reranking_in_fresh_worker(tmp_path, monkeypatch):
    payloads = fake_model(monkeypatch, meta_code=EXAMPLE["code"])
    _, log = adapter(tmp_path, n_generation=1, meta_model="meta").execute("q", "Question", "GOLD_SENTINEL")
    assert log.status == "completed"
    assert [c.tool_name for c in log.tool_calls] == ["retrieve", "rerank", "retrieve", "rerank"]
    assert all(c.results == ["source_0"] for c in log.tool_calls)
    # The first node request after the proposal is the generated Evidence Agent.
    proposal_idx = next(i for i, p in enumerate(payloads) if p["model"] == "meta")
    evidence_prompt = json.dumps(payloads[proposal_idx + 1]["messages"])
    assert "CORPUS_EVIDENCE_SENTINEL" in evidence_prompt
    assert "No results found." not in evidence_prompt
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    trace = json.loads(Path(log.artifact_paths["trace"]).read_text())
    assert trace["candidates"][1]["agents"] and trace["candidates"][1]["sub_tasks"]


@pytest.mark.parametrize("block", ["COT", "COT_SC", "Reflexion", "LLM_debate"])
def test_seed_feedback_contains_every_agent_output(tmp_path, monkeypatch, block):
    outputs = []
    payloads = fake_model(monkeypatch, node_outputs=outputs)
    _, log = adapter(tmp_path, blocks=[block], n_generation=0, max_round=2, max_sc=3).execute(
        "q", "Question", "GOLD_SENTINEL")
    assert log.status == "completed"
    trace = json.loads(Path(log.artifact_paths["trace"]).read_text())
    seed = trace["candidates"][0]
    verifier = next(p for p in payloads if "meticulous evaluator" in p["messages"][0]["content"])
    verifier_text = json.dumps(verifier["messages"])
    assert "Whole-question task output" in seed["sub_tasks"]
    for p, output in zip(payloads, outputs):
        key = "feedback" if '"correct"' in p["messages"][0]["content"] else "thinking"
        assert output[key] in seed["agents"]
        assert output[key] in verifier_text
    assert "GOLD_SENTINEL" not in json.dumps(payloads)


@pytest.mark.parametrize("code,reason", [
    (None, "required string keys"),
    ("def wrong(): pass", "forward() signature"),
    ("def forward(self, taskInfo):\n    broken (", "syntax error"),
])
def test_rejected_proposals_keep_cost_and_diagnostics(tmp_path, monkeypatch, code, reason):
    fake_model(monkeypatch, meta_code=code)
    answer, log = adapter(tmp_path, n_generation=1, meta_model="meta").execute("q", "Question", "gold")
    assert answer == "42" and log.status == "completed"
    diagnostics = log.resource_summary["diagnostics"]
    assert len(diagnostics) == 2
    assert all(d["stage"] == "meta_proposal" and reason in d["error"] for d in diagnostics)
    calls = [c for c in log.llm_calls if c.phase == "construction"]
    assert len(calls) == 2
    assert sum(c.prompt_tokens + c.completion_tokens for c in calls) == 30
    events = [json.loads(line) for line in Path(log.artifact_paths["events"]).read_text().splitlines()]
    rejected = [e for e in events if e["kind"] == "proposal_rejected"]
    assert {e["logical_call_id"] for e in rejected} == {c.logical_call_id for c in calls}
