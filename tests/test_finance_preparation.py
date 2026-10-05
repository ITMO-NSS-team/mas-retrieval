import asyncio
import json

import pytest

from scripts.phase_costs import summarize
from scripts.preflight_finance import inspect_data
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession, run_with_deadline
from marlib.tracing.tracker import TokenTracker


def test_preflight_keeps_all_questions_and_reports_missing_evidence(tmp_path):
    questions = [{"id": str(i), "gold_doc_ids": ["found" if i else "missing"]} for i in range(150)]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"doc_id": "found", "text": "Evidence"}))
    data, problems = inspect_data(tmp_path)
    assert data["question_count"] == 150 and data["missing_evidence"] == [{"id": "0", "doc_id": "missing"}]
    assert problems == ["Missing 1 evidence page references"]


def test_phase_costs_do_not_duplicate_cached_construction():
    def call(phase, inp):
        return dict(measurement="api_attempt", model="openai/gpt-4o-mini", phase=phase,
                    prompt_tokens=inp, completion_tokens=0, usage_known=True)
    logs = [dict(resource_summary={"construction_reference": {"input_tokens": 100}},
                 artifact_paths={"construction": "same.json"}, llm_calls=[call("answer_execution", 10)]) for _ in range(2)]
    logs[0]["llm_calls"].insert(0, call("construction", 100))
    run = dict(system_name="generated_single_agent_one_time", question_logs=logs)
    price = dict(model="gpt-4o-mini", usd_per_million_input=1, usd_per_million_output=1, price_verified_date="fixture")
    result = summarize(run, price, (2,))
    assert result["phases"]["construction"]["input_tokens"] == 100
    assert result["observed_mean_usd"] == pytest.approx(60 / 1e6)
    assert result["calculated_CL_reuse_usd_per_answer"]["2"] == pytest.approx(60 / 1e6)
    logs[1]["llm_calls"][0]["usage_known"] = False
    assert summarize(run, price)["system_usd"] is None
    assert summarize(run, price)["calculated_CL_reuse_usd_per_answer"] is None
    logs[1]["llm_calls"][0]["measurement"] = "legacy"
    with pytest.raises(ValueError, match="API-attempt"):
        summarize(run, price)


def test_async_deadline_cancels_but_does_not_mislabel_tool_timeout():
    async def run():
        session = ResourceSession(TokenTracker("q", "q", ""), ResourceLimits(wall_seconds=0.01))
        with pytest.raises(BudgetExhausted, match="wall_seconds"):
            await run_with_deadline(session, asyncio.sleep(10))
        other = ResourceSession(TokenTracker("q", "q", ""), ResourceLimits(wall_seconds=10))
        async def failing():
            raise TimeoutError("tool-specific failure")
        with pytest.raises(TimeoutError, match="tool-specific"):
            await run_with_deadline(other, failing())
        assert other.stop_reason is None
    asyncio.run(run())
