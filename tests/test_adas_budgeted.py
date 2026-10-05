import json
import os
from pathlib import Path
import time

import pytest

from experiments.systems.adas_budgeted.adapter import ADASBudgetedAdapter
from experiments.systems.adas.blocks import RAG_BLOCKS
from marlib.adapters.code_worker import validate_code
from test_single_agent_accounting import http_models, completion, Retriever


def make_adapter(tmp_path, **limits):
    a = ADASBudgetedAdapter(Retriever(), resource_limits={"wall_seconds": 10, **limits})
    a.set_benchmark_context("fake", "Corpus", ["Unlabelled example"])
    a.set_run_context(artifact_dir=str(tmp_path))
    return a


def generated(code):
    return completion(json.dumps({"name": "test", "thought": "plan", "code": code}))


def assert_worker_dead(log):
    events = [json.loads(line) for line in Path(log.artifact_paths["execution_events"]).read_text().splitlines()]
    pid = next(e["pid"] for e in events if e["kind"] == "worker_start")
    assert any(e["kind"] == "worker_end" and e["pid"] == pid for e in events)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_seed_runs_in_worker_and_code_is_reused(tmp_path, http_models):
    replies, payloads = http_models
    replies.extend([generated(RAG_BLOCKS[0]["code"]),
                    completion(json.dumps({"thinking": "Reason", "answer": "42"})),
                    completion(json.dumps({"thinking": "Reason", "answer": "43"}))])
    a = make_adapter(tmp_path)
    answer, first = a.execute("q1", "FIRST_QUESTION", "GOLD_SENTINEL")
    assert answer == "42", first.error
    answer, second = a.execute("q2", "SECOND_QUESTION", "GOLD_SENTINEL")
    assert answer == "43", second.error
    assert first.total_tokens == 30 and second.total_tokens == 15
    assert first.tool_counts == {"retrieve": 1, "rerank": 1}
    assert "GOLD_SENTINEL" not in json.dumps(payloads)
    assert "FIRST_QUESTION" not in json.dumps(payloads[0])
    assert payloads[1]["temperature"] == 0.1  # seed's temperature cannot override the condition
    assert_worker_dead(first)
    assert_worker_dead(second)


NODE_CALL = """    agent = LLMAgentBase(['thinking', 'answer'], 'node', model='unapproved')
    thinking, answer = agent([taskInfo], 'solve')
"""


def test_hard_worker_deadline_preserves_prior_usage(tmp_path, http_models):
    replies, _ = http_models
    code = "def forward(self, taskInfo):\n" + NODE_CALL + "    while True:\n        pass\n"
    replies.extend([generated(code), completion(json.dumps({"thinking": "Reason", "answer": "42"}))])
    started = time.monotonic()
    _, log = make_adapter(tmp_path, wall_seconds=3).execute("q", "Q", "g")
    assert log.status == "budget_exhausted", log.error
    assert log.total_tokens == 30
    assert time.monotonic() - started < 8
    assert_worker_dead(log)


def test_worker_cannot_continue_after_request_cap(tmp_path, http_models):
    replies, payloads = http_models
    code = "def forward(self, taskInfo):\n" + NODE_CALL + "    thinking, answer = agent([taskInfo], 'again')\n    return self.make_final_answer(thinking, answer)\n"
    replies.extend([generated(code), completion(json.dumps({"thinking": "Reason", "answer": "42"}))])
    _, log = make_adapter(tmp_path, max_requests=1).execute("q", "Q", "g")
    assert log.status == "budget_exhausted", log.error
    assert len(payloads) == 2  # construction + one node call
    assert payloads[1]["model"] == "gpt-4o-mini"
    assert_worker_dead(log)


def test_restricted_import_fails_construction_without_fallback(tmp_path, http_models):
    replies, _ = http_models
    replies.append(generated("import os\ndef forward(self, taskInfo):\n    return taskInfo"))
    _, log = make_adapter(tmp_path).execute("q", "Q", "g")
    assert log.failure_kind == "generated_code_error"
    assert log.total_tokens == 15
    assert log.resource_summary["execution"] is None


def test_all_offered_seeds_satisfy_execution_policy():
    for block in RAG_BLOCKS:
        validate_code(block["code"])


@pytest.mark.parametrize("block", RAG_BLOCKS, ids=[b["name"] for b in RAG_BLOCKS])
def test_offered_seed_executes_with_restricted_helpers(block, tmp_path, http_models):
    replies, _ = http_models
    node = completion(json.dumps({"thinking": "Reason", "answer": "42", "feedback": "OK", "correct": "True"}))
    replies.extend([generated(block["code"])] + [node] * 20)
    answer, log = make_adapter(tmp_path).execute("q", "Q", "g")
    assert answer == "42", log.error
    assert_worker_dead(log)


def test_node_json_retries_share_logical_call(tmp_path, http_models):
    replies, _ = http_models
    replies.extend([generated(RAG_BLOCKS[0]["code"]), completion("not JSON"),
                    completion(json.dumps({"thinking": "Reason", "answer": "42"}))])
    answer, log = make_adapter(tmp_path).execute("q", "Q", "g")
    assert answer == "42", log.error
    node_calls = [c for c in log.llm_calls if c.phase == "answer_execution"]
    assert len(node_calls) == 2 and node_calls[0].logical_call_id == node_calls[1].logical_call_id
