"""Evaluate three independently generated FinanceBench workflows for the selected MetaMAS topology."""
from __future__ import annotations

import argparse
import fcntl
from importlib.util import find_spec
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dotenv import load_dotenv
from experiments.systems.automas.finance_topology import MODEL, RATES, FinanceWorkflow, read, run, save
from marlib.benchmarks import load_spec
from marlib.provenance import file_hash, runtime_provenance
from marlib.retriever.config import DEFAULT_EMBEDDER, DEFAULT_RERANKER, RetrieverSettings
from scripts.preflight_finance import inspect_data, verify_existing_index

IDS = ROOT / "configs/finance_prompt_ablation.ids.json"


ONE_AGENT_RESULTS = ROOT / "results/metamas-financebench-ranked-search"
STANDARD_RESULTS = ROOT / "results/metamas-financebench-standard-ranked-search"
# Reviewed, complete Linux run at 7a101abb. Only its manifest and summary are read.
ONE_AGENT_MANIFEST = "66451b8431697a6dc023f6d2702b613c14abfe8142ef14d6dce0f465526a59ae"
ONE_AGENT_SUMMARY = "c9f757a640e5728c6e4446fa8d72575895590164c01a95693371f26b75595fd8"


def protocol(manifest):
    """Inputs that affect the experiment, independent of checkout and result locations."""
    settings = {k: v for k, v in manifest.items() if k not in ("source", "conditions", "data_dir", "retrieval")}
    settings["retrieval"] = {k: v for k, v in manifest["retrieval"].items() if k != "index_path"}
    condition, = manifest["conditions"].values()
    settings["condition"] = {k: v for k, v in condition.items() if k != "workflow_topology"}
    files = manifest["source"]["source_sha256"]
    runner, = (Path(k) for k in files if k.endswith("/scripts/run_metamas_topology.py"))
    checkout, data = runner.parents[1], Path(manifest["data_dir"])
    hashes = {}
    for name, digest in files.items():
        path = Path(name)
        if path.is_relative_to(data):
            name = "benchmark/" + str(path.relative_to(data))
        elif path.is_relative_to(checkout):
            name = str(path.relative_to(checkout))
        # These two files were edited to select the arm and label its output.
        if name not in ("scripts/run_metamas_topology.py", "experiments/systems/automas/finance_topology.py"):
            hashes[name] = digest
    settings["runtime_sources"] = hashes
    packages = manifest["source"]["packages"]
    settings["packages"] = {k: packages.get(k) for k in (
        "pydantic", "pydantic-ai-slim", "openai", "fastmcp", "mcp", "chromadb",
        "FlagEmbedding", "transformers", "torch", "sentence-transformers", "numpy")}
    return settings


def verified_reference(directory, current):
    for name, expected in (("manifest.json", ONE_AGENT_MANIFEST), ("summary.json", ONE_AGENT_SUMMARY)):
        if file_hash(directory / name) != expected:
            raise ValueError(f"Expected the completed ranked-search one-agent run in {directory} ({name})")
    old = protocol(read(directory / "manifest.json"))
    new = protocol(current)
    differences = [k for k in old.keys() | new.keys() if old.get(k) != new.get(k)]
    if differences:
        raise ValueError("Inputs differ from the one-agent run: " + ", ".join(sorted(differences)))
    summary = read(directory / "summary.json")
    return {"manifest_sha256": ONE_AGENT_MANIFEST, "summary_sha256": ONE_AGENT_SUMMARY,
            **summary["arms"]["one_agent"]}


def load_questions(spec):
    ids = read(IDS)
    rows = spec.load_questions()
    by_id = {q["id"]: q for q in rows}
    if len(rows) != 150 or len(ids) != 150 or set(ids) != set(by_id) or len(set(ids)) != 150:
        raise ValueError("Expected all 150 registered FinanceBench IDs")
    rows = [by_id[qid] for qid in ids]
    if any(not isinstance(q.get(k), str) or not q[k].strip() for q in rows for k in ("question", "answer")):
        raise ValueError("Missing FinanceBench question or reference answer")
    return rows


def check_retrieval(spec, settings, corpus_hash, corpus_count):
    """Runs in a short-lived child so the QA MCP server owns the GPU during evaluation."""
    from marlib.retriever.core import Retriever
    sidecar = spec.index_path / "index_provenance.json"
    if not (spec.index_path / spec.collection / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Missing prepared FinanceBench index under {spec.index_path}")
    origin = read(sidecar) if sidecar.is_file() else None
    if origin is not None and (origin.get("corpus_sha256") != corpus_hash or
                               origin.get("collection") != spec.collection or origin.get("embedder") != settings.embedder):
        raise ValueError("Index metadata does not match the FinanceBench corpus/embedding model")
    retriever = Retriever(settings)
    if retriever.document_count != corpus_count:
        raise ValueError("FinanceBench index and corpus have different document counts")
    if origin is None:
        origin = verify_existing_index(retriever._collection, spec.corpus_path, retriever._embedder)
    docs = retriever.search("reported revenue", top_k=10, use_rerank=True)
    if not docs:
        raise ValueError("FinanceBench retrieval/reranking returned no documents")
    return {"index_provenance": origin, "retrieved_ids": [d.doc_id for d in docs], "document_count": corpus_count}


def env_setup():
    load_dotenv(ROOT / ".env")
    router_url = "https://openrouter.ai/api/v1"
    key = os.getenv("OPENROUTER_API_KEY")
    if not key and os.getenv("OPENAI_BASE_URL", "").rstrip("/") == router_url:
        key = os.getenv("OPENAI_API_KEY")
    if key:
        os.environ.update(OPENAI_API_KEY=key, OPENROUTER_API_KEY=key, OPENAI_BASE_URL=router_url)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
    os.environ.update(AGENT_NODE_MODEL=MODEL, DEFAULT_META_MODEL=MODEL, LANGFUSE_TRACING_ENABLED="false")
    return bool(key)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "experiments/benchmarks",
                   help="Prepared benchmark root containing financebench/ (may be in another checkout)")
    p.add_argument("--arm", choices=("one_agent", "standard"), default="one_agent",
                   help="Select one arm. Standard lets MetaMAS choose the topology.")
    p.add_argument("--one-agent-results", type=Path, default=ONE_AGENT_RESULTS,
                   help="Completed ranked-search reference for the standard arm (read only)")
    p.add_argument("--output", type=Path, help="Defaults to a separate result directory for each arm")
    p.add_argument("--check", action="store_true", help="Validate data, framework and retrieval without LLM calls")
    args = p.parse_args(argv)
    out = (args.output or (STANDARD_RESULTS if args.arm == "standard" else ONE_AGENT_RESULTS)).resolve()
    reference_dir = args.one_agent_results.resolve()
    data_dir = args.data_dir.resolve()
    if args.arm == "standard" and (out.is_relative_to(reference_dir) or reference_dir.is_relative_to(out)):
        p.error("Choose separate, non-overlapping directories for the standard output and one-agent reference.")
    os.chdir(ROOT)
    key_available = env_setup()
    spec = load_spec("financebench", data_dir)
    data, issues = inspect_data(spec.root)
    if issues:
        p.exit(2, "\n".join(issues) + "\n")
    questions = load_questions(spec)
    if find_spec("automas") is None:
        p.error("Install the project's external automas framework in the experiment environment.")
    from automas.meta_agents import GraphGenerator, PoolGenerator  # noqa: F401
    from automas.pipeline import PipelineBuilder  # noqa: F401
    settings = RetrieverSettings(index_path=spec.index_path, collection=spec.collection,
                                 embedder=DEFAULT_EMBEDDER, reranker=DEFAULT_RERANKER,
                                 retrieve_top_k=20, rerank_top_k=10)
    settings.export_env()

    def make_adapter(arm, folder):
        adapter = FinanceWorkflow(None, model=MODEL, meta_model=MODEL, workflow_topology=arm,
                                  generation_mode="one_time", temperature=0.1, generator_temperature=0.3)
        adapter.set_benchmark_context(spec.name, spec.description, [q["question"] for q in questions[:5]])
        adapter.set_run_context(run_id=folder.name, artifact_dir=str(folder / "artifacts"))
        return adapter

    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.exit(2, f"Another run is using {out}\n")
        source = runtime_provenance(ROOT / "experiments/systems/automas")
        # Restrict the fingerprint to actual inputs of this experiment.
        source["source_sha256"].update({str(f): file_hash(f) for f in
            (Path(__file__).resolve(), ROOT / "scripts/preflight_finance.py", IDS, spec.root / "builder.py",
             spec.root / "manifest.toml")})
        manifest = {"draws_per_arm": 3, "seeds": None, "model": MODEL, "judge_model": MODEL,
                    "provider": "https://openrouter.ai/api/v1", "prices_per_million_usd": list(RATES),
                    "data_sha256": {k: data[k]["sha256"] for k in ("questions.jsonl", "corpus.jsonl")},
                    "index_provenance_sha256": file_hash(spec.index_path / "index_provenance.json"),
                    "question_ids": [q["id"] for q in questions], "data_dir": str(spec.root),
                    "retrieval": settings.model_dump(mode="json"), "source": source,
                    "conditions": {args.arm: make_adapter(args.arm, out).effective_config()}}
        path = out / "manifest.json"
        reference = None
        if args.arm == "standard":
            try:
                reference = verified_reference(reference_dir, manifest)
            except ValueError as exc:
                p.exit(2, f"{exc}\n")
        archived = (args.arm == "one_agent" and file_hash(path) == ONE_AGENT_MANIFEST
                    and file_hash(out / "summary.json") == ONE_AGENT_SUMMARY)
        if archived:
            if protocol(read(path)) != protocol(manifest):
                p.exit(2, "Inputs differ from the archived one-agent run.\n")
            if not args.check:
                print(f"Already complete: {out / 'summary.json'}")
                return
        elif path.exists() and read(path) != manifest:
            p.exit(2, "Run inputs or code differ from the saved manifest. Check the environment.\n")
        if not args.check and not key_available:
            p.exit(2, "Set OPENROUTER_API_KEY or the harness OpenRouter credentials in .env.\n")
        with multiprocessing.get_context("spawn").Pool(1) as worker:
            check = worker.apply(check_retrieval, (spec, settings, data["corpus.jsonl"]["sha256"], data["corpus_count"]))
        if not archived:
            save(out / "preflight.json", check)
        if args.check:
            print("FinanceBench data, framework and retrieval ready. Zero LLM calls.")
            return
        save(path, manifest)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if not (out / "checkout.json").exists():
            save(out / "checkout.json", {"git_commit": commit.stdout.strip() or None})
        summary = run(out, questions, make_adapter, arm=args.arm, one_agent_reference=reference)
        print(f"{summary['status']}: {out / 'summary.json'}")
        if summary["status"] != "complete":
            p.exit(2, "Some predeclared draws were invalid. See summary.json.\n")


if __name__ == "__main__":
    main()
