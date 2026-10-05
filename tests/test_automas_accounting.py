import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from experiments.systems.automas.adapter import AutoMASAdapter
from experiments.systems.automas.runtime import BudgetedMCPServer, tool_hook
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession
from marlib.tracing.tracker import TokenTracker
from test_single_agent_accounting import http_models, completion, Retriever


@pytest.fixture
def framework(monkeypatch):
    if importlib.util.find_spec("automas") is None:
        pytest.skip("Local AutoMAS installation required")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)


def structured(value):
    def respond(payload):
        fn = payload["tools"][0]["function"]
        params = fn["parameters"]
        if "response" in params.get("properties", {}):
            result = {"response": value}
        else:
            result = value
        return completion(None, (fn["name"], result))
    return respond


def workflow_replies():
    return [structured([{"name": "Answerer", "instructions": "Answer with evidence",
                         "model": "openai/other-model", "mcp_tools": []}]),
            structured({"Answerer": []})]


def make_adapter(tmp_path, **kwargs):
    a = AutoMASAdapter(Retriever(), model="gpt-4o-mini", **kwargs)
    a.set_benchmark_context("fake", "Financial corpus", ["Unlabelled example"])
    a.set_run_context(artifact_dir=str(tmp_path), repeat=0, run_id="run")
    return a


def test_real_framework_cl_construction_reuse(framework, tmp_path, http_models):
    replies, payloads = http_models
    replies.extend([*workflow_replies(), completion(), completion()])
    a = make_adapter(tmp_path, generation_mode="one_time", resource_limits={"max_requests": 1})
    answer, first = a.execute("q1", "FIRST_QUESTION", "GOLD_SENTINEL")
    assert answer == "42", first.error
    _, second = a.execute("q2", "SECOND_QUESTION", "GOLD_SENTINEL")
    assert second.status == "completed", second.error
    assert first.total_tokens == 45 and second.total_tokens == 15
    assert [c.phase for c in first.llm_calls] == ["construction", "construction", "answer_execution"]
    assert first.artifact_paths["workflow"] == second.artifact_paths["workflow"]
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert "FIRST_QUESTION" not in json.dumps(payloads[:2])
    assert all(p["model"] == "openai/gpt-4o-mini" for p in payloads)
    assert first.resource_summary["execution"]["api_attempts"] == 1


def test_ql_budget_includes_construction_and_survives_pipeline_error(framework, tmp_path, http_models):
    replies, payloads = http_models
    replies.extend(workflow_replies())
    a = make_adapter(tmp_path, generation_mode="per_task", resource_limits={"max_requests": 2})
    answer, log = a.execute("q", "Question", "gold")
    assert answer == "" and log.status == "budget_exhausted", log.error
    assert len(payloads) == 2 and log.total_tokens == 30
    assert log.resource_summary["execution"]["stop_reason"] == "max_requests"


def test_construction_wall_excludes_execution_and_is_frozen_on_reuse(framework, tmp_path, http_models, monkeypatch):
    from types import SimpleNamespace
    import marlib.tracing.resources as resources

    clock = [0.0]
    monkeypatch.setattr(resources, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    replies, _ = http_models

    def delayed(response, seconds):
        def respond(payload):
            clock[0] += seconds
            return response(payload) if callable(response) else response
        return respond

    replies.extend([*(delayed(reply, 2) for reply in workflow_replies()),
                    delayed(completion(), 40), delayed(completion(), 30)])
    adapter = make_adapter(tmp_path, generation_mode="one_time")
    _, first = adapter.execute("q1", "Q1", "gold")
    _, second = adapter.execute("q2", "Q2", "gold")
    assert first.status == second.status == "completed"
    assert first.resource_summary["construction_reference"]["wall_seconds"] == 4
    assert second.resource_summary["construction_reference"] == first.resource_summary["construction_reference"]
    assert first.resource_summary["execution"]["wall_seconds"] == 40
    assert second.resource_summary["execution"]["wall_seconds"] == 30
    events = [json.loads(line) for line in Path(first.artifact_paths["construction_events"]).read_text().splitlines()]
    end = next(event for event in events if event["kind"] == "construction_end")
    assert end["resources"]["wall_seconds"] == 4 and end["error"] is None


def test_construction_budget_failure_is_not_retried(framework, tmp_path, http_models):
    replies, payloads = http_models
    replies.append(workflow_replies()[0])
    a = make_adapter(tmp_path, generation_mode="one_time", construction_limits={"max_requests": 1})
    _, first = a.execute("q1", "Q", "g")
    _, second = a.execute("q2", "Q", "g")
    assert first.status == "budget_exhausted", first.error
    assert second.status == "failed"
    assert len(payloads) == 1 and second.total_tokens == 0
    assert Path(first.artifact_paths["workflow"]).exists()  # partial pool survives
    assert first.resource_summary["construction_reference"]["api_attempts"] == 1
    assert first.resource_summary["construction_reference"]["stop_reason"] == "max_requests"


def test_real_mcp_subprocess_admission_and_source_ids(tmp_path):
    server = tmp_path / "server.py"
    server.write_text('''from types import SimpleNamespace as NS
import marlib.mcp_server as module
class Retriever:
    def retrieve(self, query, top_k):
        return [NS(doc_id="doc_0", title="Source", text="Evidence", score=1)]
    def rerank(self, query, docs, top_k):
        return docs
module._retriever = Retriever()
module.mcp.run(show_banner=False)
''')
    tracker = TokenTracker("q", "q", "")
    session = ResourceSession(tracker, ResourceLimits(max_tool_calls=2), tmp_path / "events.jsonl")
    tracker.tool_event_sink = session.write
    async def run():
        env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(Path("src").resolve()), "MARLIB_PRIMITIVE_TOOLS": "1"}
        mcp = BudgetedMCPServer(sys.executable, args=[str(server)], env=env, timeout=20)
        hook = tool_hook(session)
        async with mcp:
            assert {t.name for t in await mcp.list_tools()} == {"retrieve", "rerank", "calculate"}
            assert "Evidence" in await hook(None, mcp.direct_call_tool, "retrieve", {"query": "Q"})
            assert "Evidence" in await hook(None, mcp.direct_call_tool, "rerank", {"query": "Q"})
            with pytest.raises(BudgetExhausted):
                await hook(None, mcp.direct_call_tool, "calculate", {"expression": "1+1"})
    asyncio.run(run())
    log = tracker.to_question_log("")
    assert log.tool_counts == {"retrieve": 1, "rerank": 1}
    assert all(c.results == ["doc_0"] for c in log.tool_calls)
