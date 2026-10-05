from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from marlib.adapters import discover_adapters, get_adapter_class
from marlib.benchmarks import BenchmarkSpec, discover, load_spec
from marlib.evaluation.llm_judge import judge_model
from marlib.reporting import render_summary
from marlib.log import logger
from marlib.retriever.config import RetrieverSettings
from marlib.runner import run_system_on_benchmark, save_results
from marlib.tracing.schemas import SystemResults
from marlib.provenance import file_hash, runtime_provenance


def _git_sha() -> str | None:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    # Discovered from the default content roots, for the --help listings only.
    available = list(discover())
    systems = discover_adapters()

    parser = argparse.ArgumentParser(
        description="Run one benchmark across the given system(s) and save "
        "results + provenance."
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["financebench"],
        help="Benchmark(s) to run, each in turn (like --systems takes several). "
        f"Discovered from --data-dir. Available: {available}",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4o-mini", help="LLM model identifier."
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["fedotmas"],
        help=f"System(s) to run in this process. Available: {systems}",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Cap on number of questions (default: full set).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the whole (benchmark x system) grid this many times; the final "
        "summary table reports mean +/- std over repeats (default: 1).",
    )
    parser.add_argument(
        "--generation-mode",
        default=None,
        choices=["one_time", "per_task"],
        help="When the system (re)generates its MAS: 'one_time' generates once "
        "and reuses it across the whole benchmark; 'per_task' regenerates for "
        "each question. Default is per-adapter.",
    )
    # Defaults for these live in RetrieverSettings (single source of truth); a
    # flag left unset (None) keeps that default rather than overriding it.
    rs = RetrieverSettings.model_fields
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help=f"Candidates retrieved before rerank (default: {rs['retrieve_top_k'].default}).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=f"Documents kept after rerank (default: {rs['rerank_top_k'].default}).",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help=f"Embedder model name (default: {rs['embedder'].default}).",
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help=f"Reranker model name (default: {rs['reranker'].default}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("experiments/benchmarks"),
        help="Benchmark repository root (one subdir per benchmark).",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=Path("experiments/systems"),
        help="Systems (adapters) repository root (one subdir per system).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output root dir."
    )
    parser.add_argument(
        "--note", default="", help="Free-text note describing this run."
    )
    parser.add_argument("--adapter-config", type=Path, help="JSON object keyed by system name; adapter kwargs")
    parser.add_argument("--question-ids", type=Path, help="JSON list of ordered question IDs; mutually exclusive with --sample-n")
    parser.add_argument("--condition-id", default="default", help="Protocol condition label saved in metadata")
    return parser


def main() -> None:
    """Console-script entry point (`bench`).

    Runs each requested benchmark in turn — ``--benchmark`` accepts several, just
    like ``--systems`` accepts several systems. Every benchmark gets its own
    retriever, results dir, and provenance entry.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if args.question_ids and args.sample_n is not None:
        parser.error("--question-ids and --sample-n are mutually exclusive")
    args.adapter_options = json.loads(args.adapter_config.read_text()) if args.adapter_config else {}
    if not isinstance(args.adapter_options, dict) or any(not isinstance(v, dict) for v in args.adapter_options.values()):
        parser.error("--adapter-config must be an object keyed by system name")
    available = discover_adapters(args.systems_dir)
    unknown = [s for s in args.systems if s not in available]
    if unknown:
        parser.error(f"Unknown system(s) {unknown}. Available: {available}")
    for name in args.systems:
        cls = get_adapter_class(name)
        cfg = args.adapter_options.get(name, {})
        if {"retriever", "model"} & cfg.keys():
            parser.error("Set the node model with --model; retriever is supplied by the harness")
        if "resource_limits" in cfg and not cls.supports_resource_limits:
            parser.error(f"Resource limits are not instrumented for: {name}")
        mode = args.generation_mode or cfg.get("generation_mode")
        if mode and cls.supported_generation_modes and mode not in cls.supported_generation_modes:
            parser.error(f"{name} supports only {cls.supported_generation_modes}")
    if args.repeats < 1 or (args.sample_n is not None and args.sample_n < 1):
        parser.error("--repeats and --sample-n must be positive")

    # Resolve (and validate) every benchmark up front so a typo fails fast,
    # before any heavy retriever load or model call.
    specs: list[BenchmarkSpec] = []
    for name in args.benchmark:
        try:
            specs.append(load_spec(name, args.data_dir))
        except ValueError as e:
            parser.error(str(e))

    # Run the grid `--repeats` times, collecting each successful system run so the
    # final table can report mean +/- std over repeats. Isolate each benchmark: a
    # crash in one (e.g. retriever/index load) is logged and skipped so the rest of
    # the grid still runs.
    collected: dict[tuple[str, str], list[SystemResults]] = {}
    failed_benchmarks: list[str] = []
    for rep in range(args.repeats):
        if args.repeats > 1:
            logger.info(f"=== repeat {rep + 1}/{args.repeats} ===")
        for spec in specs:
            try:
                results = _run_benchmark(spec, args, repeat=rep)
            except Exception as e:
                logger.error(f"Benchmark '{spec.name}' failed, skipping: {e}")
                failed_benchmarks.append(f"{spec.name} (repeat {rep + 1})")
                continue
            for r in results:
                collected.setdefault((r.system_name, r.benchmark), []).append(r)
    if failed_benchmarks:
        logger.warning(
            f"{len(failed_benchmarks)} benchmark run(s) failed: {failed_benchmarks}"
        )

    render_summary(
        collected,
        benchmarks=[spec.name for spec in specs],
        repeats=args.repeats,
        model=args.model,
        judge_model=judge_model(),
    )


def _run_benchmark(
    spec: BenchmarkSpec, args: argparse.Namespace, repeat: int = 0
) -> list[SystemResults]:
    """Run all requested systems on one benchmark and save results + provenance.

    Returns the successful ``SystemResults`` (failed systems are skipped), so the
    caller can aggregate them across repeats into the summary table.
    """
    # Unique run id avoids collisions between concurrent runs on one machine.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{spec.name}_{uuid4().hex[:6]}"
    out_dir = args.results_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting run {run_id}",
        benchmark=spec.name,
        model=args.model,
        systems=args.systems,
        note=args.note or None,
    )

    overrides = {
        k: v
        for k, v in {
            "embedder": args.embedder,
            "reranker": args.reranker,
            "retrieve_top_k": args.retrieve_top_k,
            "rerank_top_k": args.rerank_top_k,
        }.items()
        if v is not None
    }
    settings = RetrieverSettings(
        index_path=spec.index_path, collection=spec.collection, **overrides
    )
    # Export so any MCP server spawned by an adapter reconstructs this same config.
    settings.export_env()

    questions = spec.load_questions(sample_n=args.sample_n)
    if args.question_ids:
        ids = json.loads(args.question_ids.read_text())
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be a list of unique strings")
        by_id = {q["id"]: q for q in questions}
        if len(by_id) != len(questions):
            raise ValueError("Duplicate question IDs in benchmark")
        questions = [by_id[qid] for qid in ids]

    logger.info(f"Loaded {len(questions)} questions")
    sample_qs = [q.get("question", "") for q in questions[:5]]

    adapter_kwargs = {}
    if args.generation_mode is not None:
        adapter_kwargs["generation_mode"] = args.generation_mode

    all_results: list[SystemResults] = []
    summaries: list[dict] = []
    failed_systems: list[str] = []
    effective_configs = {}
    provenance = {"run_id": run_id, "git_sha": _git_sha(), "repeat": repeat,
                  "condition_id": args.condition_id, "question_ids": [q["id"] for q in questions],
                  "requested_adapter_config": args.adapter_options, "status": "started"}
    provenance.update(runtime_provenance(args.systems_dir))
    provenance["data_sha256"] = {str(p): file_hash(p) for p in (
        spec.questions_path, spec.corpus_path, spec.root / "manifest.toml",
        spec.root / "builder.py", spec.root / "corpus_provenance.json",
        spec.index_path / "index_provenance.json")}
    (out_dir / "run_meta.json").write_text(json.dumps(provenance, indent=2))

    try:
        from marlib.retriever import Retriever

        logger.info("Initializing retriever...")
        retriever = Retriever(settings)
        logger.info(
            f"Collection '{spec.collection}' loaded",
            documents=retriever.document_count,
        )

    except Exception as exc:
        provenance.update(status="infrastructure_error", error=str(exc))
        (out_dir / "run_meta.json").write_text(json.dumps(provenance, indent=2))
        raise

    for system_name in args.systems:
        logger.info(f"Running system: {system_name}")
        # Isolate each system: an adapter that fails to build or crashes mid-run
        # is logged and recorded, but the remaining systems still run.
        try:
            adapter = get_adapter_class(system_name)(
                retriever=retriever, model=args.model, **{**args.adapter_options.get(system_name, {}), **adapter_kwargs}
            )
            adapter.set_benchmark_context(spec.name, spec.description, sample_qs)
            adapter.set_run_context(run_id=run_id, repeat=repeat, condition_id=args.condition_id,
                                    artifact_dir=str(out_dir / "artifacts" / system_name))
            effective_configs[system_name] = adapter.effective_config()
            provenance["effective_adapter_config"] = effective_configs
            (out_dir / "run_meta.json").write_text(json.dumps(provenance, indent=2))

            results = run_system_on_benchmark(
                adapter=adapter,
                questions=questions,
                benchmark_name=spec.name,
                model=args.model,
                metrics=spec.metrics,
                checkpoint_path=out_dir / f"{system_name}.questions.jsonl",
            )
        except Exception as e:
            logger.error(
                f"System '{system_name}' on '{spec.name}' failed, skipping: {e}"
            )
            failed_systems.append(system_name)
            summaries.append({"system": system_name, "error": str(e)})
            continue

        logger.success(
            f"Results for {system_name}/{spec.name}",
            **{k: round(v, 3) for k, v in results.avg_metrics.items()},
            tokens_per_q=round(results.avg_tokens_per_question),
            prompt_tokens_per_q=round(results.avg_prompt_tokens_per_question),
            completion_tokens_per_q=round(results.avg_completion_tokens_per_question),
            latency_ms=round(results.avg_latency_ms),
        )

        save_results(results, out_dir)
        all_results.append(results)
        summaries.append(
            {
                "system": results.system_name,
                "metrics": results.avg_metrics,
                "avg_tokens": results.avg_tokens_per_question,
                "failed": results.failed_questions,
                "total": results.total_questions,
            }
        )

    params = {
        "benchmark": spec.name,
        "model": args.model,
        "systems": args.systems,
        "sample_n": args.sample_n,
        "repeat": repeat,
        "judge_model": judge_model(),
        "generation_mode": args.generation_mode,
        "retrieve_top_k": settings.retrieve_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "embedder": settings.embedder,
        "reranker": settings.reranker,
        "index_path": str(settings.index_path),
        "data_dir": str(args.data_dir),
        "systems_dir": str(args.systems_dir),
        "metrics": list(spec.metrics),
    }
    if failed_systems:
        logger.warning(
            f"{spec.name}: {len(failed_systems)} system(s) failed: {failed_systems}"
        )
    run_meta = {
        **provenance,
        "status": "completed_with_failures" if failed_systems else "completed",
        "run_id": run_id,
        "timestamp": timestamp,
        "argv": sys.argv,
        "git_sha": _git_sha(),
        "note": args.note,
        "params": params,
        "failed_systems": failed_systems,
        "summaries": summaries,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Append a one-line index entry for at-a-glance history.
    index_entry = {
        "run_id": run_id,
        "dir": str(out_dir),
        "benchmark": spec.name,
        "model": args.model,
        "systems": args.systems,
        "note": args.note,
        "git_sha": run_meta["git_sha"],
        "summaries": summaries,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "runs.jsonl", "a") as f:
        f.write(json.dumps(index_entry) + "\n")

    logger.success(
        "Run complete",
        results_dir=str(out_dir),
        history_index=str(args.results_dir / "runs.jsonl"),
    )
    return all_results


if __name__ == "__main__":
    main()
