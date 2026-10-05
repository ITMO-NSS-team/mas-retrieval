from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import json
import threading

import pytest

from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession, TrackedCompletion, completion_request
from marlib.tracing.tracker import TokenTracker


def session(tmp_path, **limits):
    tracker = TokenTracker("q", "question", "")
    return ResourceSession(tracker, ResourceLimits(**limits), tmp_path / "events.jsonl")


def client(fn):
    return NS(chat=NS(completions=NS(create=fn)))


def response(inp=10, out=3):
    return NS(usage=NS(prompt_tokens=inp, completion_tokens=out))


def test_counts_tools_separately():
    tracker = TokenTracker("q", "q", "g")
    for tool in ("retrieve", "rerank", "calculate", "search"):
        with tracker.track_tool(tool, "q", 1):
            pass
    log = tracker.to_question_log("a")
    assert log.num_tool_calls == 4
    assert log.num_retrieval_calls == 2
    assert log.tool_counts == dict.fromkeys(["retrieve", "rerank", "calculate", "search"], 1)


def test_concurrent_request_reservation(tmp_path):
    s = session(tmp_path, max_requests=2)
    barrier = threading.Barrier(2)
    def request(**kw):
        barrier.wait(timeout=3)
        return response()
    def call(i):
        try:
            s.create(client(request), "answer_execution", str(i), model="fake")
            return True
        except BudgetExhausted:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(call, range(8))) == 2
    assert s.summary()["api_attempts"] == 2
    assert s.tracker.to_question_log("").total_tokens == 26


def test_failed_request_keeps_partial_cost_and_unknown_usage(tmp_path):
    s = session(tmp_path, max_total_tokens=1000)
    s.create(client(lambda **k: response()), "construction", "one", model="fake")
    def fail(**k):
        raise ConnectionError("offline fault")
    with pytest.raises(ConnectionError):
        s.create(client(fail), "internal_verification", "two", model="fake")
    with pytest.raises(BudgetExhausted, match="usage_unknown"):
        s.create(client(fail), "answer_execution", "three", model="fake")
    assert s.summary()["unknown_usage_attempts"] == 1
    log = s.tracker.to_question_log("")
    assert log.total_tokens == 13
    assert len(log.llm_calls) == 2
    events = [json.loads(line) for line in s.journal.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["session", "llm_start", "llm_end", "llm_start", "llm_end", "budget_stop"]
    assert events[-2]["usage_known"] is False


def test_token_overshoot_is_explicit_and_stops_next_call(tmp_path):
    s = session(tmp_path, max_total_tokens=20, output_per_request=10)
    observed = []
    def request(**kw):
        observed.append(kw)
        return response(30, 5)
    s.create(client(request), "answer_execution", "one", model="fake")
    assert s.summary()["total_token_overshoot"] == 15
    assert observed[0]["max_tokens"] == 10
    with pytest.raises(BudgetExhausted):
        s.create(client(request), "answer_execution", "two", model="fake")


def test_retry_attempts_share_logical_id_without_double_usage(tmp_path, monkeypatch):
    import openai
    import httpx
    s = session(tmp_path)
    calls = []
    def request(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))
        return response()
    monkeypatch.setattr("marlib.tracing.resources.time.sleep", lambda _: None)
    completion_request(client(request), TrackedCompletion(s, "construction"), "logical", model="fake")
    assert s.summary()["api_attempts"] == 2
    assert s.summary()["logical_calls"] == 1
    assert s.tracker.to_question_log("").total_tokens == 13


def test_budget_stop_never_enters_retry(tmp_path):
    s = session(tmp_path, max_requests=1, max_tool_calls=1, wall_seconds=20)
    calls = []
    c = client(lambda **kw: (calls.append(kw), response())[1])
    cb = TrackedCompletion(s, "answer_execution")
    completion_request(c, cb, "one", model="fake")
    with pytest.raises(BudgetExhausted):
        completion_request(c, cb, "two", model="fake")
    assert len(calls) == 1
    with pytest.raises(BudgetExhausted, match="max_requests"):
        s.tool()  # A swallowed budget exception cannot resume execution.
    s = ResourceSession(TokenTracker("q", "q", ""), ResourceLimits(max_tool_calls=1))
    s.tool()
    with pytest.raises(BudgetExhausted, match="max_tool_calls"):
        s.tool()
    s = ResourceSession(TokenTracker("q", "q", ""), ResourceLimits(wall_seconds=20))
    s.started -= 21
    with pytest.raises(BudgetExhausted, match="wall_seconds"):
        s.check()
