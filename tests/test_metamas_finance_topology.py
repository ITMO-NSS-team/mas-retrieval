import asyncio
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
    assert "retrieval_search(query, top_k=10, use_rerank=True)" in str(payloads[0]["messages"])
    assert "retrieve(query, top_k=20)" not in str(payloads[0]["messages"])
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


@pytest.mark.parametrize("arm", ["one_agent", "standard"])
def test_three_draws_all_150_questions(framework, http_models, monkeypatch, tmp_path, arm):
    from marlib.tracing.tracker import TokenTracker
    replies, payloads = http_models
    for repeat in range(3):
        graph = {"Analyst": []}
        if arm == "standard" and repeat:
            graph["Searcher"] = ["Analyst"]
        replies.extend([structured([candidate(name) for name in graph]), structured(graph)])
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
    reference = {"accuracy": {"mean": 0.551}, "evaluated": 3} if arm == "standard" else None
    result = study.run(tmp_path, qs, factory, arm=arm, one_agent_reference=reference)
    assert result["status"] == "complete"
    assert len(executions) == len(set(executions)) == 450
    assert {mode for mode, _, _ in executions} == {arm}
    assert list(result["arms"]) == [arm]
    assert result["arms"][arm]["agents"] == ([1, 2, 2] if arm == "standard" else [1, 1, 1])
    if arm == "standard":
        assert result["one_agent_reference"] == reference and "standard_reference" not in result
        assert all("generate exactly one agent" not in str(p["messages"]) for p in payloads)
        assert not list(tmp_path.glob("one_agent-*"))
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


def test_one_agent_ranked_search_interface_and_trace(tmp_path, monkeypatch):
    from marlib.tracing.resources import ResourceLimits, ResourceSession
    from marlib.tracing.tracker import TokenTracker

    backend = tmp_path / "fake_corpus.py"
    backend.write_text('''from types import SimpleNamespace as NS
import marlib.mcp_server as module
class Corpus:
    def search(self, query, top_k, use_rerank):
        ids = list(range(20))
        if use_rerank:
            ids.reverse()
        return [NS(doc_id=f"doc-{i}", title="Test", score=1.0, text=f"passage-{i}.") for i in ids[:top_k]]
module._retriever = Corpus()
module.mcp.run(show_banner=False)
''')
    # Even if the host uses primitive tools elsewhere, the experiment exports its own profile.
    monkeypatch.setenv("MARLIB_PRIMITIVE_TOOLS", "1")
    monkeypatch.setenv("PYTHONPATH", str(Path("src").resolve()))
    adapter = factory("one_agent", tmp_path)
    assert adapter.effective_config()["tools"] == ["retrieval_search", "calculate"]
    tracker = TokenTracker("q", "q", "")
    journal = tmp_path / "events.jsonl"
    session = ResourceSession(tracker, ResourceLimits(), journal)
    mcp, = adapter._toolsets(["retrieval"], session)
    mcp.args = [str(backend)]

    async def check_server():
        async with mcp:
            tools = {t.name: t for t in await mcp.list_tools()}
            assert set(tools) == {"retrieval_search", "calculate"}
            schema = tools["retrieval_search"].inputSchema["properties"]
            assert schema["top_k"]["default"] == 10 and schema["use_rerank"]["default"] is True
            for args, expected in [({}, list(range(19, 9, -1))),
                                   ({"top_k": 5}, list(range(19, 14, -1))),
                                   ({"top_k": 2, "use_rerank": False}, [0, 1]),
                                   ({"top_k": 40}, list(range(19, -1, -1)))]:
                text = await mcp.process_tool_call(None, mcp.direct_call_tool, "retrieval_search",
                                                   {"query": "Q", **args})
                assert text == "\n\n".join(f"[{n}] Test (score: 1.000)\npassage-{i}."
                                           for n, i in enumerate(expected, 1))
                assert tracker.to_question_log("").tool_calls[-1].results == [f"doc-{i}" for i in expected]
            assert await mcp.process_tool_call(None, mcp.direct_call_tool, "calculate",
                                               {"expression": "1+2"}) == "1+2 = 3.0"
    asyncio.run(check_server())
    log = tracker.to_question_log("")
    assert log.tool_counts == {"retrieve": 4, "calculate": 1}
    opts = [e for e in study.events(journal) if e["kind"] == "retrieval_search"]
    assert [(e["top_k"], e["use_rerank"]) for e in opts] == [(10, True), (5, True), (2, False), (40, True)]


def test_unavailable_tool_name_is_saved_in_failed_draw(framework, http_models, tmp_path):
    replies, _ = http_models
    replies.append(structured([dict(candidate("Analyst"), mcp_tools=["unavailable-server"])]))
    row = study.prepare_draw(factory("one_agent", tmp_path), tmp_path)
    assert row["status"] == "invalid_construction" and "unavailable-server" in row["error"]
    assert row["construction"]["calls"] == 1

def test_standard_charges_work_before_downstream_failure(framework, http_models, tmp_path):
    from dataclasses import replace
    replies, payloads = http_models
    replies.extend([structured([candidate("Searcher"), candidate("Analyst")]),
                    structured({"Searcher": ["Analyst"], "Analyst": []}), completion("evidence")])
    adapter = factory("standard", tmp_path)
    adapter._limits = replace(adapter._limits, max_requests=1)
    draw = study.prepare_draw(adapter, tmp_path)
    assert draw["structure"]["agents"] == 2
    result = study.evaluate(adapter, tmp_path, [dict(id="q", question="Question", answer="gold")])
    assert result["correct"] == 0 and result["judge"]["calls"] == 0
    assert len(payloads) == 3 and result["execution"]["calls"] == 1
    assert result["failed_answer_execution"] == result["execution"]
    assert result["execution"]["usd"] == pytest.approx((10 * .15 + 5 * .60) / 1e6)


def test_matched_reference_is_read_only_and_detects_protocol_drift(tmp_path, monkeypatch):
    from copy import deepcopy
    from scripts import run_metamas_topology as cli
    archive = tmp_path / "reference"
    old = {"conditions": {"one_agent": {"workflow_topology": "one_agent", "tools": ["retrieval_search", "calculate"]}},
           "question_ids": ["q"], "data_dir": "/old/data", "retrieval": {"index_path": "/old/index"},
           "source": {"source_sha256": {"/old/scripts/run_metamas_topology.py": "a",
                                        "/old/src/marlib/mcp_server.py": "b"}, "packages": {"openai": "1"}}}
    group = {"evaluated": 3, "accuracy": {"mean": 0.551}}
    study.save(archive / "manifest.json", old)
    study.save(archive / "summary.json", {"arms": {"one_agent": group}})
    monkeypatch.setattr(cli, "ONE_AGENT_MANIFEST", cli.file_hash(archive / "manifest.json"))
    monkeypatch.setattr(cli, "ONE_AGENT_SUMMARY", cli.file_hash(archive / "summary.json"))
    new = deepcopy(old)
    new["conditions"] = {"standard": {**old["conditions"]["one_agent"], "workflow_topology": "standard"}}
    new["source"]["source_sha256"] = {"/new/scripts/run_metamas_topology.py": "c", "/new/src/marlib/mcp_server.py": "b"}
    assert cli.verified_reference(archive, new)["accuracy"] == group["accuracy"]
    new["source"]["source_sha256"]["/new/src/marlib/mcp_server.py"] = "changed"
    with pytest.raises(ValueError, match="runtime_sources"):
        cli.verified_reference(archive, new)
    study.save(archive / "summary.json", {"status": "evaluating"})
    with pytest.raises(ValueError, match="summary.json"):
        cli.verified_reference(archive, old)
