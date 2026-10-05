import asyncio
import json

import pytest

from scripts.phase_costs import summarize
from scripts.preflight_finance import inspect_data
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession, run_with_deadline
from marlib.tracing.tracker import TokenTracker


@pytest.fixture
def finance_builder():
    from marlib.benchmarks import discover, get_builder
    discover()
    return get_builder("financebench")


def test_finance_zero_based_evidence_maps_to_one_based_corpus(finance_builder):
    assert finance_builder.evidence_doc_ids({"doc_name": "fallback", "evidence": [
        {"doc_name": "3M_2023Q2_10Q", "evidence_page_num": 0},
        {"doc_name": "AES_2022_10K", "evidence_page_num": 131},
        {"evidence_doc_name": "Other_2022_10K", "evidence_page_num": 2},
        {"evidence_page_num": 4},
    ]}) == ["3m_2023q2_10q_p1", "aes_2022_10k_p132", "fallback_p5", "other_2022_10k_p3"]
    with pytest.raises(ValueError, match="Invalid zero-based"):
        finance_builder.evidence_doc_ids({"doc_name": "Doc", "evidence": [{"evidence_page_num": -1}]})


def test_finance_repair_preserves_questions_corpus_and_backup(tmp_path, finance_builder):
    questions = [{"id": str(i), "question": f"Q{i}", "answer": f"A{i}", "doc_name": "Doc",
                  "evidence": [{"evidence_page_num": i}], "gold_doc_ids": [f"doc_p{i}"]}
                 for i in range(150)]
    path = tmp_path / "questions.jsonl"
    original = "\n".join(json.dumps(q) for q in questions).encode()
    path.write_bytes(original)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("unchanged corpus")
    assert finance_builder.repair_evidence_ids(tmp_path)["changed_questions"] == 150
    assert path.read_bytes() == original
    result = finance_builder.repair_evidence_ids(tmp_path, apply=True)
    assert result["applied"]
    assert (tmp_path / "questions.before_evidence_fix.jsonl").read_bytes() == original
    corrected = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(corrected) == 150
    for before, after in zip(questions, corrected):
        assert after == {**before, "gold_doc_ids": [f"doc_p{int(before['id']) + 1}"]}
    assert corpus.read_text() == "unchanged corpus"
    assert not finance_builder.repair_evidence_ids(tmp_path, apply=True)["applied"]


def test_finance_repair_refuses_missing_evidence_and_existing_backup(tmp_path, finance_builder):
    path = tmp_path / "questions.jsonl"
    original = json.dumps({"id": "q", "gold_doc_ids": ["doc_p0"]})
    path.write_text(original)
    with pytest.raises(ValueError, match="No usable raw evidence"):
        finance_builder.repair_evidence_ids(tmp_path, apply=True)
    assert path.read_text() == original
    assert not (tmp_path / "questions.before_evidence_fix.jsonl").exists()
    original = json.dumps({"id": "q", "doc_name": "Doc", "evidence": [{"evidence_page_num": 0}]})
    path.write_text(original)
    (tmp_path / "questions.before_evidence_fix.jsonl").write_text("previous backup")
    with pytest.raises(FileExistsError):
        finance_builder.repair_evidence_ids(tmp_path, apply=True)
    assert path.read_text() == original


def test_preflight_detects_wrong_mapping_even_if_old_page_exists(tmp_path, finance_builder):
    questions = [{"id": str(i), "doc_name": "Doc", "evidence": [{"evidence_page_num": 1}],
                  "gold_doc_ids": ["doc_p1"]} for i in range(150)]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps({"doc_id": doc_id}) for doc_id in ["doc_p1", "doc_p2"]))
    data, problems = inspect_data(tmp_path)
    assert not data["missing_evidence"]
    assert len(data["evidence_mapping_mismatches"]) == 150
    assert any("Incorrect evidence page mapping" in p for p in problems)
    finance_builder.repair_evidence_ids(tmp_path, apply=True)
    data, problems = inspect_data(tmp_path)
    assert not problems and not data["evidence_mapping_mismatches"]


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
