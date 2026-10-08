import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from experiments.systems.adas.adapter import ADASAdapter
from experiments.systems.adas import core
from experiments.systems.adas.blocks import RAG_BLOCKS
from experiments.systems.adas_compact_prompt.adapter import ADASCompactPromptAdapter, PROMPT_SUFFIX
from experiments.systems.generated_single_agent_legacy.adapter import GeneratedSingleAgentLegacyAdapter
from test_single_agent_accounting import Retriever, completion


@pytest.fixture
def wire(monkeypatch):
    """Keep real SDK retry logic and HTTP clients; intercept transport only."""
    replies, payloads, timeouts = [], [], []
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    monkeypatch.setattr(openai.OpenAI, "_calculate_retry_timeout", lambda *args: 0)
    monkeypatch.setattr(openai.AsyncOpenAI, "_calculate_retry_timeout", lambda *args: 0)

    def handle(request):
        assert request.url.host == "offline.invalid"
        payloads.append(json.loads(request.content))
        timeouts.append(request.extensions.get("timeout"))
        assert replies, "Unexpected model request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "test failure"}})
        return httpx.Response(200, json=reply)

    def sync_send(self, request):
        return handle(request)

    async def async_send(self, request):
        return handle(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_send)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    return replies, payloads, timeouts


def architecture(code=None):
    return completion(json.dumps({"name": "test", "thought": "plan", "code": code or RAG_BLOCKS[0]["code"]}))


def node(answer="42"):
    return completion(json.dumps({"thinking": "Evidence", "answer": answer}))


def adapter(cls, tmp_path, **kwargs):
    result = cls(Retriever(), model="openai/gpt-4o-mini", **kwargs)
    result.set_benchmark_context("fake", "Financial filings", ["Unlabelled example"])
    result.set_run_context(artifact_dir=str(tmp_path), run_id="first", repeat=0)
    return result


def test_adas_original_request_parity_and_exact_prompt_delta(tmp_path, wire, monkeypatch):
    replies, payloads, timeouts = wire
    monkeypatch.setattr(core, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="fixed_agent_id")))
    replies.extend([architecture(), node(), architecture(), node()])
    original, old_log = adapter(ADASAdapter, tmp_path / "original").execute("q", "Q", "GOLD_SENTINEL")
    compact, log = adapter(ADASCompactPromptAdapter, tmp_path / "compact").execute("q", "Q", "GOLD_SENTINEL")
    assert original == compact == "42"
    assert old_log.tool_calls[0].results == log.tool_calls[0].results == ["doc_0"]
    expected = json.loads(json.dumps(payloads[0]))
    expected["messages"][1]["content"] += PROMPT_SUFFIX
    assert payloads[2] == expected
    assert payloads[1] == payloads[3]  # full executor request, including generated temperature
    assert timeouts[:2] == timeouts[2:]
    assert not {"temperature", "max_tokens", "max_completion_tokens"} & payloads[2].keys()
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert log.total_tokens == 30  # construction + one node, never the old callback twice
    assert log.resource_summary["n_agents"] == 1
    assert [c.phase for c in log.llm_calls] == ["construction", "answer_execution"]


def test_adas_reuse_and_independent_runs(tmp_path, wire):
    replies, payloads, _ = wire
    replies.extend([architecture(), node(), node("43"), architecture(), node()])
    a = adapter(ADASCompactPromptAdapter, tmp_path)
    _, first = a.execute("q", "Q", "gold")
    _, second = a.execute("q", "Q", "gold")
    assert first.total_tokens == 30 and second.total_tokens == 15
    assert first.artifact_paths["construction"] == second.artifact_paths["construction"]
    assert first.artifact_paths["events"] != second.artifact_paths["events"]
    a.set_run_context(artifact_dir=str(tmp_path), run_id="second", repeat=1)
    _, third = a.execute("q", "Q", "gold")
    assert third.total_tokens == 30 and len(payloads) == 5
    assert first.artifact_paths["construction"] != third.artifact_paths["construction"]


@pytest.mark.parametrize("error", [500, httpx.ConnectError("offline connection failure")])
def test_adas_sdk_retries_observed_without_policy_change(tmp_path, wire, error):
    replies, _, _ = wire
    replies.extend([architecture(), error, node()])
    answer, log = adapter(ADASCompactPromptAdapter, tmp_path).execute("q", "Q", "g")
    assert answer == "42", log.error
    assert len(log.llm_calls) == 3 and log.total_tokens == 30
    assert log.resource_summary["unknown_usage_attempts"] == 1
    assert log.llm_calls[1].logical_call_id == log.llm_calls[2].logical_call_id
    events = [json.loads(line) for line in Path(log.artifact_paths["events"]).read_text().splitlines()]
    assert sum(e["kind"] == "llm_start" for e in events) == 3
    assert sum(e["kind"] == "llm_end" for e in events) == 3


def test_adas_syntax_debug_and_original_seed_fallback(tmp_path, wire):
    replies, payloads, _ = wire
    replies.extend([architecture("def forward(self, taskInfo):\n    bad syntax"), architecture(), node()])
    answer, log = adapter(ADASCompactPromptAdapter, tmp_path).execute("q", "Q", "g")
    assert answer == "42" and log.total_tokens == 45
    assert "Syntax error" in payloads[1]["messages"][-1]["content"]
    replies.extend([completion("not JSON")] * 3 + [node()])
    a = adapter(ADASCompactPromptAdapter, tmp_path / "fallback")
    answer, log = a.execute("q", "Q", "g")
    assert answer == "42" and log.total_tokens == 60
    assert a.generated_system == RAG_BLOCKS[0]


def test_adas_failure_retains_partial_usage_and_workflow(tmp_path, wire):
    replies, _, _ = wire
    code = RAG_BLOCKS[0]["code"].replace("    return final_answer", "    raise RuntimeError('generated failure')")
    replies.extend([architecture(code), node()])
    answer, log = adapter(ADASCompactPromptAdapter, tmp_path).execute("q", "Q", "g")
    assert answer == "" and log.status == "failed" and log.total_tokens == 30
    assert "generated failure" in log.error
    assert json.loads(Path(log.artifact_paths["construction"]).read_text())["workflow"]["code"] == code
    assert log.resource_summary["n_agents"] == 1


def test_adas_uncached_construction_error_preserves_original_next_question_retry(tmp_path, wire):
    replies, payloads, _ = wire
    replies.extend([500, 500, 500, architecture(), node()])
    a = adapter(ADASCompactPromptAdapter, tmp_path)
    _, failed = a.execute("q1", "Q", "g")
    assert failed.status == "failed" and failed.total_tokens == 0
    assert failed.resource_summary["unknown_usage_attempts"] == 3
    answer, recovered = a.execute("q2", "Q", "g")
    assert answer == "42" and recovered.total_tokens == 30
    assert len(payloads) == 5
    assert failed.artifact_paths["construction"] != recovered.artifact_paths["construction"]
    assert json.loads(Path(failed.artifact_paths["construction"]).read_text())["error"]


def test_generated_legacy_tool_calls_defaults_and_prompt_reuse(tmp_path, wire):
    replies, payloads, timeouts = wire
    replies.extend([completion("Use evidence."), completion(None, ("retrieve", {"query": "source"})),
                    completion(None, ("rerank", {"query": "source"})), completion(), completion("43")])
    a = adapter(GeneratedSingleAgentLegacyAdapter, tmp_path)
    answer, first = a.execute("q1", "FIRST_QUESTION", "GOLD_SENTINEL")
    assert answer == "42", first.error
    answer, second = a.execute("q2", "SECOND_QUESTION", "GOLD_SENTINEL")
    assert answer == "43", second.error
    assert first.total_tokens == 60 and second.total_tokens == 15
    assert first.tool_counts == {"retrieve": 1, "rerank": 1}
    assert first.artifact_paths["construction"] == second.artifact_paths["construction"]
    assert first.resource_summary["construction_charged_here"] is True
    assert second.resource_summary["construction_charged_here"] is False
    for request in payloads[1:]:
        assert not {"temperature", "max_tokens", "max_completion_tokens"} & request.keys()
        assert request["messages"][0]["content"] == "Use evidence."
    assert payloads[0]["temperature"] == 0.3 and payloads[0]["max_tokens"] == 4096
    assert timeouts[0]["read"] == 60 and timeouts[1]["read"] == 600
    assert "FIRST_QUESTION" not in json.dumps(payloads[0])
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert a.effective_config()["usage_limits"]["request_limit"] == 50


def test_generated_legacy_sdk_retry_and_failed_construction(tmp_path, wire):
    replies, payloads, _ = wire
    replies.extend([completion("Use evidence."), 500, completion()])
    answer, log = adapter(GeneratedSingleAgentLegacyAdapter, tmp_path).execute("q", "Q", "g")
    assert answer == "42", log.error
    assert len(log.llm_calls) == 3 and log.total_tokens == 30
    assert log.resource_summary["execution"]["unknown_usage_attempts"] == 1
    assert log.llm_calls[1].logical_call_id == log.llm_calls[2].logical_call_id
    replies.append(completion(""))
    a = adapter(GeneratedSingleAgentLegacyAdapter, tmp_path / "failed")
    _, first = a.execute("q1", "Q", "g")
    _, second = a.execute("q2", "Q", "g")
    assert first.status == second.status == "failed"
    assert first.total_tokens == 15 and second.total_tokens == 0
    assert len(payloads) == 4


def test_legacy_rejects_new_execution_caps(tmp_path):
    with pytest.raises(ValueError):
        adapter(GeneratedSingleAgentLegacyAdapter, tmp_path, temperature=0.1)
    with pytest.raises(ValueError):
        adapter(ADASCompactPromptAdapter, tmp_path, resource_limits={"max_requests": 10})


def test_generated_legacy_retains_framework_limit_and_partial_cost(tmp_path, wire):
    replies, payloads, _ = wire
    replies.extend([completion("Use evidence.")] +
                   [completion(None, ("calculate", {"expression": "1+1"}))] * 50)
    answer, log = adapter(GeneratedSingleAgentLegacyAdapter, tmp_path).execute("q", "Q", "g")
    assert answer == "" and log.status == "failed"
    assert "request_limit of 50" in log.error
    assert len(payloads) == 51 and log.total_tokens == 51 * 15
    assert log.resource_summary["execution"]["api_attempts"] == 50


@pytest.mark.parametrize("cls", [ADASCompactPromptAdapter, GeneratedSingleAgentLegacyAdapter])
def test_phase_cost_export_charges_construction_once(cls, tmp_path, wire):
    from scripts.phase_costs import summarize
    replies, _, _ = wire
    if cls is ADASCompactPromptAdapter:
        replies.extend([architecture(), node(), node()])
    else:
        replies.extend([completion("Use evidence."), completion(), completion()])
    a = adapter(cls, tmp_path)
    _, first = a.execute("q1", "Q", "g")
    _, second = a.execute("q2", "Q", "g")
    costs = summarize({"system_name": a.name, "question_logs": [first.model_dump(), second.model_dump()]},
                      {"model": "openai/gpt-4o-mini", "usd_per_million_input": .15,
                       "usd_per_million_output": .6, "price_verified_date": "fixed_R011_tariff"})
    assert costs["phases"]["construction"]["api_attempts"] == 1
    assert costs["phases"]["answer_execution"]["api_attempts"] == 2
    assert costs["system_usd"] == pytest.approx(.0000135)
    assert costs["calculated_CL_reuse_usd_per_answer"]["150"] == pytest.approx(.0000045 * (1 + 1/150))
