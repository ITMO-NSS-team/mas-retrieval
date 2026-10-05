import asyncio
import json
from pathlib import Path

import httpx
import openai
import pytest

from experiments.systems.single_agent.adapter import SingleAgentAdapter
from experiments.systems.generated_single_agent.adapter import GeneratedSingleAgentAdapter
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession
from marlib.tracing.tracker import TokenTracker


class Retriever:
    def retrieve(self, query, top_k):
        from types import SimpleNamespace
        return [SimpleNamespace(doc_id="doc_0", title="Source", text="Evidence", score=1)]

    def rerank(self, query, docs, top_k):
        return docs


def completion(content="42", tool=None, finish_reason=None):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{"id": "tool_1", "type": "function",
                                  "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return {"id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
            "model": "gpt-4o-mini", "choices": [{"index": 0, "message": message,
             "finish_reason": finish_reason or ("tool_calls" if tool else "stop")}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture
def http_models(monkeypatch):
    """Exercise the real SDK and Pydantic AI stack, intercepting all HTTP."""
    payloads = []
    replies = []

    def handle(request):
        payloads.append(json.loads(request.content))
        assert replies, "Unexpected extra model request"
        reply = replies.pop(0)
        if callable(reply):
            reply = reply(payloads[-1])
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json=reply)

    real_sync, real_async = openai.OpenAI, openai.AsyncOpenAI

    def sync_client(**kwargs):
        assert kwargs["max_retries"] == 0
        return real_sync(**{**kwargs, "api_key": "offline", "base_url": "https://offline.invalid/v1",
                            "http_client": httpx.Client(transport=httpx.MockTransport(handle))})

    def async_client(**kwargs):
        assert kwargs["max_retries"] == 0
        return real_async(**{**kwargs, "api_key": "offline", "base_url": "https://offline.invalid/v1",
                             "http_client": httpx.AsyncClient(transport=httpx.MockTransport(handle))})

    monkeypatch.setattr(openai, "OpenAI", sync_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", async_client)
    return replies, payloads


def adapter(tmp_path, generated=False, **kwargs):
    cls = GeneratedSingleAgentAdapter if generated else SingleAgentAdapter
    a = cls(Retriever(), model="gpt-4o-mini", **kwargs)
    a.set_benchmark_context("fake", "A financial corpus", ["Unlabelled example"])
    a.set_run_context(artifact_dir=str(tmp_path), run_id="run", repeat=0)
    return a


def test_baseline_actual_attempts_and_tools(tmp_path, http_models):
    replies, payloads = http_models
    replies.extend([completion(None, ("retrieve", {"query": "evidence"})),
                    completion(None, ("rerank", {"query": "evidence"})), completion()])
    answer, log = adapter(tmp_path).execute("../../q", "Question", "GOLD_SENTINEL")
    assert answer == "42", log.error
    assert log.num_llm_calls == 3 and log.total_tokens == 45
    assert log.tool_counts == {"retrieve": 1, "rerank": 1}
    assert log.tool_calls[0].results == ["doc_0"]
    assert all(c.measurement == "api_attempt" for c in log.llm_calls)
    assert all(p["max_completion_tokens"] == 4096 for p in payloads)
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert log.gold_answer == "GOLD_SENTINEL"


def test_request_cap_preserves_partial_usage(tmp_path, http_models):
    replies, payloads = http_models
    replies.append(completion(None, ("calculate", {"expression": "2+2"})))
    answer, log = adapter(tmp_path, resource_limits={"max_requests": 1}).execute("q", "Q", "g")
    assert answer == "" and log.status == "budget_exhausted", log.error
    assert len(payloads) == 1 and log.total_tokens == 15
    assert log.num_tool_calls == 1
    assert Path(log.artifact_paths["execution_events"]).exists()


def test_tool_cap_stops_before_second_tool(tmp_path, http_models):
    replies, payloads = http_models
    replies.extend([completion(None, ("calculate", {"expression": "2+2"})),
                    completion(None, ("calculate", {"expression": "3+3"}))])
    _, log = adapter(tmp_path, resource_limits={"max_tool_calls": 1}).execute("q", "Q", "g")
    assert log.status == "budget_exhausted", log.error
    assert log.num_tool_calls == 1
    assert len(payloads) == 2 and log.total_tokens == 30


def test_cl_prompt_reuse_separate_budgets_and_no_gold(tmp_path, http_models):
    replies, payloads = http_models
    replies.extend([completion("Use the corpus carefully."), completion(), completion(),
                    completion("New independent instruction."), completion()])
    a = adapter(tmp_path, generated=True, resource_limits={"max_requests": 1})
    answer, first = a.execute("q1", "FIRST_QUESTION", "GOLD_SENTINEL")
    assert answer == "42", first.error
    _, second = a.execute("q2", "SECOND_QUESTION", "GOLD_SENTINEL")
    assert first.total_tokens == 30 and second.total_tokens == 15
    assert first.resource_summary["execution"]["api_attempts"] == 1
    assert set(first.resource_summary["execution"]["phases"]) == {"answer_execution"}
    assert first.resource_summary["construction_charged_here"] is True
    assert second.resource_summary["construction_charged_here"] is False
    assert first.artifact_paths["construction"] == second.artifact_paths["construction"]
    assert first.artifact_paths["execution_events"] != second.artifact_paths["execution_events"]
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert "FIRST_QUESTION" not in json.dumps(payloads[0])
    assert "Unlabelled example" in json.dumps(payloads[0])
    assert payloads[1]["messages"][0]["content"] == payloads[2]["messages"][0]["content"]
    a.set_run_context(artifact_dir=str(tmp_path), run_id="run2", repeat=1)
    _, third = a.execute("q1", "FIRST_QUESTION", "GOLD_SENTINEL")
    assert third.total_tokens == 30
    assert third.artifact_paths["construction"] != first.artifact_paths["construction"]


def test_failed_construction_is_not_retried_on_each_question(tmp_path, http_models):
    replies, payloads = http_models
    replies.append(completion(""))
    a = adapter(tmp_path, generated=True)
    _, first = a.execute("q1", "Q", "g")
    _, second = a.execute("q2", "Q", "g")
    assert first.status == second.status == "failed"
    assert first.total_tokens == 15 and second.total_tokens == 0
    assert len(payloads) == 1
    artifact = json.loads(Path(first.artifact_paths["construction"]).read_text())
    assert artifact["error"] and artifact["system_prompt"] is None


def test_truncated_instruction_is_rejected(tmp_path, http_models):
    replies, _ = http_models
    replies.append(completion("Incomplete", finish_reason="length"))
    _, log = adapter(tmp_path, generated=True).execute("q", "Q", "g")
    assert log.status == "failed" and "truncated" in log.error


def test_async_cancellation_releases_reservation_and_keeps_unknown_usage(tmp_path):
    tracker = TokenTracker("q", "q", "")
    session = ResourceSession(tracker, ResourceLimits(), tmp_path / "events.jsonl")

    async def run():
        started = asyncio.Event()
        async def pending(**kwargs):
            started.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(session.create_async(pending, "answer_execution", "id", model="fake"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert session.reserved_output == 0
    assert session.summary()["unknown_usage_attempts"] == 1
    assert "CancelledError" in tracker._llm_calls[0].error


def test_scope_and_mode_validation(tmp_path):
    with pytest.raises(ValueError, match="one_time"):
        adapter(tmp_path, generated=True, generation_mode="per_task")
    with pytest.raises(ValueError, match="execution"):
        adapter(tmp_path, resource_limits={"scope": "full_answer"})


def test_async_sdk_retry_counts_attempts_once(tmp_path, http_models, monkeypatch):
    replies, payloads = http_models
    replies.extend([httpx.ReadTimeout("offline timeout"), completion()])
    async def no_sleep(_):
        pass
    monkeypatch.setattr("marlib.tracing.resources.asyncio.sleep", no_sleep)
    answer, log = adapter(tmp_path).execute("q", "Q", "g")
    assert answer == "42", log.error
    assert len(payloads) == 2
    assert log.total_tokens == 15 and log.num_llm_calls == 2
    assert log.llm_calls[0].usage_known is False
    assert log.llm_calls[0].logical_call_id == log.llm_calls[1].logical_call_id


def test_unknown_usage_prevents_retry_under_token_limit(tmp_path, http_models):
    replies, payloads = http_models
    replies.append(httpx.ReadTimeout("offline timeout"))
    _, log = adapter(tmp_path, resource_limits={"max_total_tokens": 100}).execute("q", "Q", "g")
    assert log.status == "budget_exhausted", log.error
    assert len(payloads) == 1 and log.num_llm_calls == 1
    assert log.resource_summary["execution"]["unknown_usage_attempts"] == 1


def test_invalid_rerank_is_still_a_counted_tool_call(tmp_path, http_models):
    replies, _ = http_models
    replies.extend([completion(None, ("rerank", {"query": "no retrieval yet"})), completion()])
    _, log = adapter(tmp_path).execute("q", "Q", "g")
    assert log.status == "completed", log.error
    assert log.tool_counts == {"rerank": 1}
    assert log.resource_summary["execution"]["tool_calls"] == 1
