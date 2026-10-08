"""Replay only CUDA-failed FinanceBench answers with the exact saved CL construction.

Default is an offline plan. --execute loads retrieval, checks it without an LLM,
then runs paid answer/judge calls in this fresh process. Run once per source run.
This deliberately restores caches of the two frozen P006 adapters; their model,
tool and generated-code execution paths are unchanged. No generation is allowed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4


SYSTEMS = {
    "adas_compact_prompt": "finance_adas_compact_prompt_cl_v1",
    "generated_single_agent_legacy": "finance_generated_single_agent_legacy_cl_v1",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_plan(source: Path):
    meta = read(source / "run_meta.json")
    require(meta["run_id"] == source.name, "Source folder must retain its original run ID")
    systems = meta["params"]["systems"]
    require(len(systems) == 1 and systems[0] in SYSTEMS, "Only the two frozen P006 systems are supported")
    system = systems[0]
    require(meta["condition_id"] == SYSTEMS[system], "Expected the original full-series condition")
    require(meta["params"]["benchmark"] == "financebench" and meta["params"]["generation_mode"] == "one_time", "Expected FinanceBench CL")
    result_path = source / f"{system}_one_time_financebench_openai_gpt-4o-mini.json"
    logs = read(result_path)["question_logs"]
    checkpoint = source / f"{system}.questions.jsonl"
    require(logs == [json.loads(s) for s in checkpoint.read_text().splitlines() if s.strip()], "Checkpoint differs from final logs")
    ids = [q["question_id"] for q in logs]
    require(len(ids) == len(set(ids)) == 150 and ids == meta["question_ids"], "Expected 150 unique ordered source questions")
    targets = [q for q in logs if q["status"] == "failed" and "CUDA out of memory" in (q["error"] or "")]
    require(bool(targets), "No CUDA-failed answers to replay; method failures are not replacement candidates")
    require(all(not q["predicted_answer"] for q in targets), "Refusing to replace nonempty answers")
    require(all(any("CUDA out of memory" in (t["error"] or "") for t in q["tool_calls"]) for q in targets), "CUDA failure must be confirmed in tool records")
    paths = {q["artifact_paths"]["construction"] for q in logs}
    require(len(paths) == 1, "Expected one shared construction")
    relative = Path(next(iter(paths))).relative_to(Path("results") / source.name)
    require(".." not in relative.parts, "Invalid construction path")
    path = source / relative
    construction = read(path)
    require(construction["error"] is None, "Source construction failed")
    if system == "adas_compact_prompt":
        require(isinstance(construction.get("workflow", {}).get("code"), str), "Missing saved workflow")
        compile(construction["workflow"]["code"], "<restored-workflow>", "exec")
    else:
        require(isinstance(construction.get("system_prompt"), str) and construction["system_prompt"].strip(), "Missing saved instruction")
    return {"system": system, "source_run_id": source.name, "source_repeat": meta["repeat"],
        "source_condition_id": meta["condition_id"], "source_run_meta_sha256": digest(source / "run_meta.json"),
        "source_result_sha256": digest(result_path), "construction_relative_path": str(relative),
        "construction_sha256": digest(path), "question_ids": [q["question_id"] for q in targets],
        "retained_source_questions": 150 - len(targets), "generation_calls": 0,
        "selection": "All and only empty answers with recorded CUDA OOM in a retrieval tool; no accuracy selection",
        "cost_attribution": "Original construction/failed-attempt costs retained in source; replay charges execution/judge only"}


def restore_adapter(adapter, system, construction_path, artifact_dir):
    """Restore after benchmark/run context setup, before any execute() call.

    Private cache fields are intentionally limited to the two source-verified
    P006 adapters. Original generation and execution implementations stay intact.
    """
    value = read(construction_path)
    require(value["error"] is None, "Cannot restore failed construction")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / "restored_construction.json"
    require(not target.exists(), "Refusing to overwrite an existing construction")
    if system == "adas_compact_prompt":
        require(adapter._cached_system is None, "Adapter already has a workflow")
        require(value["generator_prompt_suffix"] == adapter.effective_config()["generator_prompt_suffix"], "Prompt suffix mismatch")
        require(isinstance(value.get("workflow", {}).get("code"), str), "Missing workflow")
        compile(value["workflow"]["code"], "<restored-workflow>", "exec")
        adapter._cached_system = value["workflow"]
        adapter._construction_seconds = 0.0
    elif system == "generated_single_agent_legacy":
        require(not adapter._construction_attempted, "Adapter already attempted construction")
        require(isinstance(value.get("system_prompt"), str) and value["system_prompt"].strip(), "Missing instruction")
        adapter._prompt = value["system_prompt"]
        adapter._construction_attempted = True
        adapter._construction_journal = None  # No construction attempt in this replay.
    else:
        raise ValueError("Unsupported replay system")
    shutil.copyfile(construction_path, target)
    adapter._construction_path = target
    adapter._construction_summary = value["resources"]


def validate_environment(meta, provenance, file_hash):
    # Execution code is unchanged by this script. Do not silently accept another
    # executor/library/data snapshot for an infrastructure replay.
    mismatches = [p for p, h in meta["source_sha256"].items() if provenance["source_sha256"].get(p) != h]
    require(not mismatches, "Source differs from original run: " + ", ".join(mismatches))
    packages = ("openai", "pydantic-ai", "pydantic-ai-slim", "httpx", "FlagEmbedding", "torch", "transformers", "chromadb")
    mismatches = [p for p in packages if provenance["packages"].get(p) != meta["packages"].get(p)]
    require(not mismatches, "Execution/retrieval package versions differ: " + ", ".join(mismatches))
    mismatches = [p for p, h in meta["data_sha256"].items() if file_hash(Path(p)) != h]
    require(not mismatches, "Benchmark/index provenance differs: " + ", ".join(mismatches))


def execute(source, plan, results_dir):
    from marlib.adapters import discover_adapters, get_adapter_class
    from marlib.benchmarks import load_spec
    from marlib.cli import _git_sha
    from marlib.provenance import file_hash, runtime_provenance
    from marlib.retriever.config import RetrieverSettings
    from marlib.runner import run_system_on_benchmark, save_results

    meta = read(source / "run_meta.json")
    params, system = meta["params"], plan["system"]
    os.environ["JUDGE_MODEL"] = params["judge_model"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    provenance = runtime_provenance(Path(params["systems_dir"]))
    validate_environment(meta, provenance, file_hash)
    spec = load_spec("financebench", Path(params["data_dir"]))
    by_id = {q["id"]: q for q in spec.load_questions()}
    original = read(source / f"{system}_one_time_financebench_openai_gpt-4o-mini.json")["question_logs"]
    require(all(by_id[q["question_id"]]["question"] == q["question"] and by_id[q["question_id"]]["answer"] == q["gold_answer"] for q in original), "Source question text/answers differ from local data")
    selected = [by_id[qid] for qid in plan["question_ids"]]
    settings = RetrieverSettings(index_path=Path(params["index_path"]), collection=spec.collection,
        **{k: params[k] for k in ("embedder", "reranker", "retrieve_top_k", "rerank_top_k")})
    settings.export_env()
    require(system in discover_adapters(Path(params["systems_dir"])), "Adapter dependencies unavailable")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_financebench_recovery_" + uuid4().hex[:6]
    out = results_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    artifact_dir = out / "artifacts" / system
    record = {"run_id": run_id, "git_sha": _git_sha(), "repeat": meta["repeat"],
        "timestamp": datetime.now().isoformat(), "argv": sys.argv,
        "condition_id": meta["condition_id"] + "_recovery_v1", "status": "started", "params": params,
        "question_ids": plan["question_ids"], "requested_adapter_config": meta["requested_adapter_config"],
        "data_sha256": meta["data_sha256"], "recovery": plan, **provenance,
        "recovery_script_sha256": digest(Path(__file__)), "failed_systems": []}

    def save():
        (out / "run_meta.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        from marlib.retriever import Retriever
        retriever = Retriever(settings)  # Exactly one model pair in a fresh process.
        # Retrieval-only technical check before any paid model/judge request.
        docs = retriever.retrieve(selected[0]["question"], top_k=params["retrieve_top_k"])
        ranked = retriever.rerank(selected[0]["question"], docs, top_k=params["rerank_top_k"])
        require(bool(docs) and bool(ranked), "Retrieval preflight returned no documents")
        record["retrieval_preflight"] = {"question_id": selected[0]["id"],
            "retrieved": [d.doc_id for d in docs], "reranked": [d.doc_id for d in ranked], "llm_calls": 0}
        adapter = get_adapter_class(system)(retriever=retriever, model=params["model"],
            **meta["requested_adapter_config"][system])
        adapter.set_benchmark_context(spec.name, spec.description, [q["question"] for q in original[:5]])
        adapter.set_run_context(run_id=run_id, repeat=meta["repeat"], condition_id=record["condition_id"], artifact_dir=str(artifact_dir))
        require(adapter.effective_config() == meta["effective_adapter_config"][system], "Effective adapter settings changed")
        restore_adapter(adapter, system, source / plan["construction_relative_path"], artifact_dir)
        record["effective_adapter_config"] = {system: adapter.effective_config()}
        save()
        result = run_system_on_benchmark(adapter=adapter, questions=selected, benchmark_name=spec.name,
            model=params["model"], metrics=spec.metrics, checkpoint_path=out / f"{system}.questions.jsonl")
        require(not any(c.phase == "construction" for q in result.question_logs for c in q.llm_calls), "Unexpected regeneration during replay")
        save_results(result, out)
        record["status"] = "infrastructure_error" if any("CUDA out of memory" in (q.error or "") for q in result.question_logs) else "completed"
        record["summaries"] = [{"system": result.system_name, "metrics": result.avg_metrics,
            "failed": result.failed_questions, "total": result.total_questions}]
        save()
        print(json.dumps({"output": str(out), "status": record["status"], "summaries": record["summaries"]}, indent=2))
        return 0 if record["status"] == "completed" else 1
    except BaseException as exc:
        record.update(status="recovery_error", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        with (results_dir / "runs.jsonl").open("a") as history:
            history.write(json.dumps({k: record.get(k) for k in (
                "run_id", "timestamp", "git_sha", "condition_id", "repeat", "status", "params", "summaries", "recovery")}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run retrieval preflight then paid execution/judge calls")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    plan = build_plan(args.source_run)
    print(json.dumps(plan, indent=2))
    if args.execute:
        raise SystemExit(execute(args.source_run, plan, args.results_dir))
