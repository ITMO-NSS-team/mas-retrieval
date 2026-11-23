import json
from pathlib import Path

import pytest

from experiments.systems.automas import finance_topology as study
from scripts.run_metamas_topology import load_questions
from test_automas_accounting import framework, structured
from test_single_agent_accounting import completion, http_models


def candidate(name):
    return {"name": name, "instructions": f"Perform the generated role of {name}.",
            "model": "openai/other-model", "mcp_tools": []}


def factory(arm, folder):
    a = study.FinanceWorkflow(None, study.MODEL, workflow_topology=arm, generation_mode="one_time")
    a.set_benchmark_context("financebench", "financial QA", ["unlabelled example"])
    a.set_run_context(run_id=folder.name, artifact_dir=str(folder / "artifacts"))
    return a


def test_one_agent_retries_cardinality_without_dropping_a_generated_role(framework, http_models, tmp_path):
    replies, payloads = http_models
    replies.extend([structured([candidate("Searcher"), candidate("Calculator")]), structured([candidate("Analyst")]),
                    structured({"Analyst": []}), completion()])
    a = factory("one_agent", tmp_path)
    row = study.prepare_draw(a, tmp_path)
    answer, trace = a.execute("q", "Question", "GOLD")
    assert (answer, row["structure"]["roles"]) == ("42", ["Analyst"])
    assert row["construction"]["calls"] == 3 and trace.num_llm_calls == 1
    schema = payloads[0]["tools"][0]["function"]["parameters"]
    assert '"minItems": 1' in json.dumps(schema) and '"maxItems": 1' in json.dumps(schema)
    assert "generate exactly one agent" in str(payloads[0]["messages"])
    assert "generate exactly one agent" not in str(payloads[-2]["messages"])
    assert "GOLD" not in str(payloads)


def test_one_agent_and_judge_resume_with_real_framework(framework, http_models, monkeypatch, tmp_path):
    replies, payloads = http_models
    for _ in range(3):
        replies.extend([structured([candidate("Analyst")]), structured({"Analyst": []})])
    replies.extend(completion() for _ in range(3))
    verdicts = []
    def judge(question, predicted, gold, model, usage_sink):
        verdicts.append(predicted)
        if len(verdicts) == 1:
            raise RuntimeError("offline judge outage")
        assert model == study.MODEL and gold == "GOLD"
        usage_sink.update(prompt=5, completion=1)
        return 1.0
    monkeypatch.setattr(study, "llm_accuracy", judge)
    qs = [dict(id="q", question="Question", answer="GOLD", gold_doc_ids=["p1"])]
    with pytest.raises(RuntimeError, match="judge outage"):
        study.run(tmp_path, qs, factory)
    assert len(payloads) == 7  # Three pool/graph constructions, then one answer.
    summary = study.run(tmp_path, qs, factory)
    assert summary["status"] == "complete"
    assert set(summary["arms"]) == {"one_agent"}
    assert summary["standard_reference"]["accuracy"]["mean"] == 0.549
    assert summary["standard_reference"]["execution_usd"]["mean"] == 0.431
    assert summary["arms"]["one_agent"]["agents"] == [1, 1, 1]
    assert all(r["construction"]["calls"] == 2 for r in summary["draws"])
    assert summary["draws"][0]["execution"]["calls"] == 1
    assert summary["draws"][0]["construction"]["input_tokens"] == 20
    assert summary["draws"][0]["judge"]["input_tokens"] == 5
    assert summary["draws"][0]["judge_failures"] == 1
    assert len(payloads) == 9 and len(verdicts) == 4
    assert "GOLD" not in str(payloads) and not replies
    assert study.run(tmp_path, qs, factory) == summary and len(payloads) == 9
    for offset in (0, 2, 4):
        assert "generate exactly one agent" in str(payloads[offset]["messages"])


def test_partial_construction_reuses_original_pool(framework, http_models, tmp_path):
    a = factory("one_agent", tmp_path)
    old = tmp_path / "construction/0001/workflow.json"
    study.save(old, dict(task=a._build_task_description(), model=study.MODEL, graph=None,
                         pool=[dict(candidate("Frozen"), id="original-id", model=study.MODEL)],
                         proposed_models={"original-id": study.MODEL}))
    replies, payloads = http_models
    replies.append(structured({"Frozen": []}))
    row = study.prepare_draw(a, tmp_path)
    assert len(payloads) == 1 and "AGENTS IN POOL" in str(payloads[0]["messages"])
    assert row["structure"]["roles"] == ["Frozen"]
    assert study.read(tmp_path / row["workflow"])["pool"][0]["id"] == "original-id"


def test_invalid_draw_is_kept_and_never_resampled(framework, http_models, tmp_path):
    replies, payloads = http_models
    replies.extend([structured([candidate("A")]), structured({"A": ["A"]})])
    row = study.prepare_draw(factory("one_agent", tmp_path), tmp_path)
    assert row["status"] == "invalid_construction" and row["construction"]["calls"] == 2
    assert study.prepare_draw(factory("one_agent", tmp_path), tmp_path) == row and len(payloads) == 2


def test_random_standard_single_agent_is_retained(framework, http_models, tmp_path):
    replies, _ = http_models
    replies.extend([structured([candidate("Analyst")]), structured({"Analyst": []})])
    row = study.prepare_draw(factory("standard", tmp_path), tmp_path)
    assert row["structure"] == {"agents": 1, "edges": 0, "pool_size": 1, "roles": ["Analyst"]}


def test_official_ids_and_registration_order():
    ids = json.loads((Path(__file__).resolve().parents[1] / "configs/finance_prompt_ablation.ids.json").read_text())
    rows = [dict(id=i, question="q", answer="g") for i in reversed(ids)]
    from types import SimpleNamespace
    assert [q["id"] for q in load_questions(SimpleNamespace(load_questions=lambda: rows))] == ids
    with pytest.raises(ValueError, match="150"):
        load_questions(SimpleNamespace(load_questions=lambda: rows[:-1]))


def test_three_one_agent_draws_all_150_questions(framework, http_models, monkeypatch, tmp_path):
    from marlib.tracing.tracker import TokenTracker
    replies, payloads = http_models
    for _ in range(3):
        replies.extend([structured([candidate("Analyst")]), structured({"Analyst": []})])
    executions = []
    def execute(self, qid, question, gold):
        tracker = TokenTracker(qid, question, gold)
        tracker.log_llm_call(study.MODEL, 20, 5, 1, phase="answer_execution")
        log = tracker.to_question_log("ok")
        path = Path(self._run_context["artifact_dir"]) / "fake" / "question.json"
        log.artifact_paths["question"] = str(path)
        study.save(path, log.model_dump())
        executions.append((self._topology, self._run_context["run_id"], qid))
        return "ok", log
    def judge(question, answer, gold, model, usage_sink):
        usage_sink.update(prompt=10, completion=2)
        return 1.0
    monkeypatch.setattr(study.FinanceWorkflow, "execute", execute)
    monkeypatch.setattr(study, "llm_accuracy", judge)
    qs = [dict(id=f"q{i}", question=f"Question {i}", answer="GOLD") for i in range(150)]
    result = study.run(tmp_path, qs, factory)
    assert result["status"] == "complete"
    assert len(executions) == len(set(executions)) == 450
    assert {arm for arm, _, _ in executions} == {"one_agent"}
    assert not list(tmp_path.glob("standard-*"))
    for row in result["draws"]:
        assert row["questions"] == row["correct"] == 150
        assert row["execution"]["calls"] == row["judge"]["calls"] == 150
        assert row["execution"]["input_tokens"] == 3000 and row["execution"]["output_tokens"] == 750
        assert row["construction"]["calls"] == 2
    assert len(payloads) == 6


def test_recovery_after_retrieval_error_before_execution_session(framework, http_models, monkeypatch, tmp_path):
    from marlib.tracing.tracker import TokenTracker
    seen = []
    def execute(self, qid, question, gold):
        seen.append(qid)
        tracker = TokenTracker(qid, question, gold)
        if len(seen) == 1:
            tracker.set_error("CUDA out of memory")
        log = tracker.to_question_log("" if len(seen) == 1 else "ok")
        path = Path(self._run_context["artifact_dir"]) / str(len(seen)) / "question.json"
        log.artifact_paths["question"] = str(path)
        study.save(path, log.model_dump())
        return log.predicted_answer, log
    monkeypatch.setattr(study.FinanceWorkflow, "execute", execute)
    a = factory("standard", tmp_path)
    q = {"id": "q", "question": "Q", "answer": "GOLD"}
    with pytest.raises(RuntimeError, match="infrastructure"):
        study.answer_for(a, tmp_path, q)
    assert study.answer_for(a, tmp_path, q).predicted_answer == "ok"
    assert study.answer_for(a, tmp_path, q).predicted_answer == "ok"
    assert len(seen) == 2
