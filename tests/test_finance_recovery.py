import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts.replay_finance_infrastructure import build_plan, execute, restore_adapter, validate_environment
from test_prompt_conditions import adapter, architecture, node, wire
from test_single_agent_accounting import completion, Retriever
from experiments.systems.adas_compact_prompt.adapter import ADASCompactPromptAdapter
from experiments.systems.generated_single_agent_legacy.adapter import GeneratedSingleAgentLegacyAdapter


@pytest.mark.parametrize("cls,system", [(ADASCompactPromptAdapter, "adas_compact_prompt"),
    (GeneratedSingleAgentLegacyAdapter, "generated_single_agent_legacy")])
def test_restore_exact_construction_without_generation_or_recharge(tmp_path, wire, cls, system):
    replies, payloads, _ = wire
    adas = system == "adas_compact_prompt"
    replies.extend([architecture(), node()] if adas else [completion("Use evidence."), completion()])
    original = adapter(cls, tmp_path / "original")
    _, first = original.execute("q", "Q", "g")
    path = Path(first.artifact_paths["construction"])
    restored = adapter(cls, tmp_path / "replay")
    restore_adapter(restored, system, path, tmp_path / "replay")
    assert restored.generated_system == original.generated_system
    assert Path(restored._construction_path).read_bytes() == path.read_bytes()
    replies.extend([node(), node()] if adas else [completion(), completion()])
    for _ in range(2):
        answer, log = restored.execute("q", "Q", "g")
        assert answer == "42", log.error
        assert len(log.llm_calls) == 1 and log.llm_calls[0].phase == "answer_execution"
        assert log.resource_summary["construction_charged_here"] is False
        assert log.resource_summary["construction_reference"] == first.resource_summary["construction_reference"]
        assert log.total_tokens == 15
    assert len(payloads) == 4
    # Defaults/messages of the single agent must remain exactly unchanged.
    if not adas:
        assert payloads[1] == payloads[2] == payloads[3]


def source_fixture(tmp_path, system="adas_compact_prompt", construction=None):
    root = tmp_path / "source_run"
    root.mkdir()
    relative = Path("artifacts") / system / "construction.json"
    (root / relative).parent.mkdir(parents=True)
    (root / relative).write_text(json.dumps(construction or {"error": None,
        "workflow": {"code": "def forward(self, taskInfo):\n    pass"}}))
    qs = [{"question_id": f"q{i}", "question": f"Question {i}", "gold_answer": "g",
        "status": "failed" if i < 2 else "completed", "predicted_answer": "" if i < 2 else "answer",
        "error": "CUDA out of memory" if i == 0 else "context_length_exceeded" if i == 1 else None,
        "tool_calls": [{"error": "CUDA out of memory"}] if i == 0 else [],
        "artifact_paths": {"construction": str(Path("results") / root.name / relative)}} for i in range(150)]
    params = {"systems": [system], "benchmark": "financebench", "generation_mode": "one_time",
        "model": "openai/gpt-4o-mini", "judge_model": "openai/gpt-4o-mini", "systems_dir": "experiments/systems",
        "data_dir": "experiments/benchmarks", "index_path": "experiments/benchmarks/financebench/index",
        "embedder": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3", "retrieve_top_k": 20, "rerank_top_k": 10}
    meta = {"run_id": root.name, "repeat": 2, "condition_id": "finance_" + system + "_cl_v1",
        "params": params, "question_ids": [q["question_id"] for q in qs],
        "source_sha256": {}, "data_sha256": {}, "packages": {}, "requested_adapter_config": {system: {"generation_mode": "one_time"}}}
    (root / "run_meta.json").write_text(json.dumps(meta))
    (root / f"{system}_one_time_financebench_openai_gpt-4o-mini.json").write_text(json.dumps({"question_logs": qs}))
    (root / f"{system}.questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    return root, meta, qs


def test_recovery_plan_selects_only_cuda_failures(tmp_path):
    root, _, _ = source_fixture(tmp_path)
    plan = build_plan(root)
    assert plan["question_ids"] == ["q0"]
    assert plan["retained_source_questions"] == 149
    assert plan["generation_calls"] == 0
    # A partial/different checkpoint must not silently choose replacement IDs.
    (root / "adas_compact_prompt.questions.jsonl").write_text("")
    with pytest.raises(ValueError, match="Checkpoint"):
        build_plan(root)


@pytest.mark.parametrize("section,key", [("source_sha256", "executor.py"), ("packages", "openai")])
def test_recovery_rejects_changed_execution_environment(section, key):
    meta = {"source_sha256": {}, "packages": {}, "data_sha256": {}}
    meta[section][key] = "original"
    with pytest.raises(ValueError, match="differ"):
        validate_environment(meta, {"source_sha256": {}, "packages": {}}, lambda p: None)


@pytest.mark.parametrize("preflight_error", [False, True])
def test_recovery_cli_path_uses_saved_prompt_and_records_history(tmp_path, wire, monkeypatch, preflight_error):
    from marlib import adapters, benchmarks, cli, provenance
    replies, payloads, _ = wire
    replies.extend([completion("Use evidence."), completion()])
    seed = adapter(GeneratedSingleAgentLegacyAdapter, tmp_path / "seed")
    _, first = seed.execute("q", "Q", "g")
    value = json.loads(Path(first.artifact_paths["construction"]).read_text())
    root, meta, original = source_fixture(tmp_path, "generated_single_agent_legacy", value)
    meta["effective_adapter_config"] = {"generated_single_agent_legacy": seed.effective_config()}
    (root / "run_meta.json").write_text(json.dumps(meta))
    spec = SimpleNamespace(name="financebench", collection="financebench", description="Financial filings", metrics=(),
        load_questions=lambda: [{"id": q["question_id"], "question": q["question"], "answer": q["gold_answer"]} for q in original])
    monkeypatch.setattr(benchmarks, "load_spec", lambda *a: spec)
    monkeypatch.setattr(provenance, "runtime_provenance", lambda *a: {"source_sha256": {}, "packages": {}})
    monkeypatch.setattr(adapters, "discover_adapters", lambda *a: ["generated_single_agent_legacy"])
    monkeypatch.setattr(adapters, "get_adapter_class", lambda *a: GeneratedSingleAgentLegacyAdapter)
    monkeypatch.setattr(cli, "_git_sha", lambda: "test")
    class FakeRetriever(Retriever):
        def __init__(self, settings):
            pass
        def rerank(self, query, docs, top_k):
            if preflight_error:
                raise RuntimeError("CUDA out of memory")
            return super().rerank(query, docs, top_k)
    monkeypatch.setitem(sys.modules, "marlib.retriever", SimpleNamespace(Retriever=FakeRetriever))
    replies.append(completion())
    out = tmp_path / "results"
    if preflight_error:
        with pytest.raises(RuntimeError, match="CUDA"):
            execute(root, build_plan(root), out)
        history = json.loads((out / "runs.jsonl").read_text())
        assert history["status"] == "recovery_error"
        assert len(payloads) == 2  # Source fixture only; replay never called a model.
        return
    assert execute(root, build_plan(root), out) == 0
    history = json.loads((out / "runs.jsonl").read_text())
    assert history["recovery"]["source_run_id"] == root.name
    assert history["recovery"]["question_ids"] == ["q0"]
    result_root = out / history["run_id"]
    result = json.loads(next(result_root.glob("*one_time*.json")).read_text())
    q = result["question_logs"][0]
    assert q["predicted_answer"] == "42" and q["total_tokens"] == 15
    assert not q["resource_summary"]["construction_charged_here"]
    assert len(payloads) == 3
