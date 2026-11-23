"""Resumable FinanceBench evaluation of one-agent MetaMAS against the submitted paper’s reference."""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import AsyncExitStack
import json
from pathlib import Path
from statistics import mean, stdev
from uuid import uuid4

from marlib.evaluation.llm_judge import llm_accuracy
from marlib.evaluation.metrics import context_recall
from marlib.tracing.resources import BudgetExhausted
from marlib.tracing.tracker import TokenTracker
from .adapter import AutoMASAdapter

MODEL = "openai/gpt-4o-mini"
ARMS = ("one_agent",)
STANDARD_REFERENCE = {
    "source": "Submitted paper, main results table, FinanceBench, corpus-level MetaMAS",
    "runs": 3, "questions_per_run": 150,
    "accuracy": {"mean": 0.549, "sd": 0.037},
    "execution_usd": {"mean": 0.431, "sd": 0.041},
}
RATES = (0.15, 0.60)  # Study's nominal USD per million input/output tokens.


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def events(path):
    if path.is_file():
        with path.open() as f:
            for line in f:
                if line.endswith("\n"):
                    yield json.loads(line)


def cost(calls):
    calls = list(calls)
    inp = sum(c["prompt_tokens"] for c in calls)
    out = sum(c["completion_tokens"] for c in calls)
    return {"input_tokens": inp, "output_tokens": out, "calls": len(calls),
            "unknown_usage": sum(not c["usage_known"] for c in calls),
            "usd": (inp * RATES[0] + out * RATES[1]) / 1_000_000}


def failed_infrastructure(errors):
    markers = ("apiconnectionerror", "apitimeouterror", "authenticationerror", "permissiondeniederror",
               "ratelimiterror", "internalservererror", "notfounderror", "cuda out of memory", "cuda error",
               "device-side assert", "connection closed", "connection reset", "insufficient credits",
               "payment required", "error code: 402", "disk i/o error")
    return next((s for s in errors if s and any(m in s.lower() for m in markers)), None)


class FinanceWorkflow(AutoMASAdapter):
    async def construct(self, directory, partial=None):
        tracker = TokenTracker("construction", self._build_task_description(), "")
        with self._retrieval_registry():
            async with AsyncExitStack() as stack:
                await self._build_structure(tracker, directory, stack, "", resume_pool=partial)

    def restore(self, snapshot, path):
        from automas.pipeline import PipelineBuilder
        if snapshot["task"] != self._build_task_description() or snapshot["model"] != self._meta_model:
            raise ValueError("Frozen construction differs from this experiment")
        pool = self._pool_from_snapshot(snapshot)
        graph = snapshot["graph"]
        self._validate_topology(pool, graph)
        pipeline = PipelineBuilder().create_from_pool(pool, graph).build()
        nodes = list(pipeline.execution_order)
        if not nodes:
            raise ValueError("Generated graph has no executable node")
        self._cached_pool, self._cached_graph = pool, graph
        self._construction_path, self._construction_attempted = path, True
        return {"agents": len(nodes), "edges": sum(len(n.children) for n in nodes),
                "pool_size": len(pool), "roles": [n.name for n in nodes]}


def prepare_draw(adapter, folder):
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    root = folder / "construction"
    root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(p for p in root.iterdir() if p.is_dir())
    invalid = root / "invalid.json"
    partial = snapshot_path = None
    for attempt in attempts:
        history = list(events(attempt / "construction.jsonl"))
        for entry in history:
            error = entry.get("error") or ""
            if (entry["kind"] == "construction_end" and
                error.startswith(("ValueError:", "UnexpectedModelBehavior:", "BudgetExhausted:")) and
                not failed_infrastructure(e.get("error") for e in history)):
                if not invalid.exists():
                    save(invalid, {"error": error})
                break
        path = attempt / "workflow.json"
        if path.is_file():
            snapshot = read(path)
            if snapshot["graph"] is not None:
                snapshot_path = path
                break
            partial = snapshot
    if not invalid.exists() and snapshot_path is None:
        attempt = root / f"{len(attempts) + 1:04}"
        attempt.mkdir()
        try:
            asyncio.run(adapter.construct(attempt, partial))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors = [e.get("error") for e in events(attempt / "construction.jsonl")]
            save(attempt / "failure.json", {"error": message})
            if failed_infrastructure([message, *errors]) or not isinstance(
                    exc, (ValueError, BudgetExhausted, UnexpectedModelBehavior)):
                raise
            save(invalid, {"error": message})
        else:
            snapshot_path = attempt / "workflow.json"
    journals = list(root.glob("*/construction.jsonl"))
    data = [e for journal in journals for e in events(journal)]
    bill = cost(e for e in data if e["kind"] == "llm_end")
    bill["unfinished_calls"] = sum(e["kind"] == "llm_start" for e in data) - bill["calls"]
    if invalid.exists():
        return {"status": "invalid_construction", "error": read(invalid)["error"], "construction": bill}
    assert snapshot_path is not None
    snapshot = read(snapshot_path)
    structure = adapter.restore(snapshot, snapshot_path)
    return {"status": "constructed", "structure": structure, "construction": bill,
            "workflow": str(snapshot_path.relative_to(folder))}


def answer_for(adapter, folder, q):
    from marlib.tracing.schemas import QuestionLog
    answer = folder / "answer.json"
    if answer.exists():
        return QuestionLog.model_validate(read(answer))
    folder.mkdir(parents=True, exist_ok=True)
    # The native adapter writes a trace before handing control back. Recover that
    # trace if the process died between the two checkpoints.
    artifacts = folder / "attempts"
    for trace_path in sorted(artifacts.glob("*/question.json")):
        if not (trace_path.parent / "infrastructure.json").exists():
            trace = QuestionLog.model_validate(read(trace_path))
            break
    else:
        adapter._run_context["artifact_dir"] = str(artifacts)
        _, trace = adapter.execute(q["id"], q["question"], q["answer"])
        trace_path = Path(trace.artifact_paths["question"])
    if trace.question_id != q["id"] or trace.question != q["question"] or trace.gold_answer != q["answer"]:
        raise ValueError(f"Question checkpoint mismatch at {folder}")
    if any(call.phase == "construction" for call in trace.llm_calls):
        raise RuntimeError("Unexpected workflow regeneration during evaluation")
    cause = failed_infrastructure([trace.error, *(c.error for c in trace.tool_calls),
                                   *(c.error for c in trace.llm_calls if trace.error)])
    if cause:
        save(trace_path.parent / "infrastructure.json", {"error": cause})
        raise RuntimeError(f"Execution infrastructure failure for {q['id']}: {cause}")
    save(answer, trace.model_dump())
    return trace


def evaluate(adapter, root, questions):
    scores, traces = [], []
    for i, q in enumerate(questions, 1):
        folder = root / "questions" / f"{i:03}"
        trace = answer_for(adapter, folder, q)
        path = folder / "score.json"
        if not path.exists():
            usage = {}
            failed = bool(trace.error and not trace.predicted_answer.strip())
            try:
                accuracy = 0.0 if failed else llm_accuracy(q["question"], trace.predicted_answer,
                                                           q["answer"], MODEL, usage_sink=usage)
            except Exception as exc:
                save(folder / "judge_failures" / f"{uuid4().hex}.json", {"error": str(exc)})
                raise
            ids = [doc for call in trace.tool_calls if call.tool_name in ("retrieve", "rerank", "search")
                   for doc in call.results]
            gold = q.get("gold_doc_ids") or []
            save(path, {"id": q["id"], "accuracy": accuracy, "recall": context_recall(ids, gold) if gold else None,
                        "judge_called": not failed, "usage": usage})
        score = read(path)
        if score["id"] != q["id"]:
            raise ValueError(f"Judge checkpoint mismatch at {path}")
        traces.append(trace)
        scores.append(score)
        if i % 25 == 0:
            print(f"{root.name}: {i}/{len(questions)}", flush=True)
    recall = [s["recall"] for s in scores if s["recall"] is not None]
    usd = cost(c.model_dump() for t in traces for c in t.llm_calls if c.phase == "answer_execution")
    judge = cost({"prompt_tokens": s["usage"].get("prompt", 0), "completion_tokens": s["usage"].get("completion", 0),
                  "usage_known": any(s["usage"].values())} for s in scores if s["judge_called"])
    # Recoverable outages and interrupted requests are retained outside the method's cost.
    attempts = []
    for qdir, t in zip(sorted((root / "questions").iterdir()), traces, strict=True):
        kept = t.artifact_paths.get("execution_events")
        for journal in (qdir / "attempts").glob("*/execution.jsonl"):
            if str(journal) != kept:
                attempts.extend(events(journal))
    overhead = cost(e for e in attempts if e["kind"] == "llm_end")
    overhead["unfinished_calls"] = sum(e["kind"] == "llm_start" for e in attempts) - overhead["calls"]
    return {"status": "evaluated", "questions": len(scores), "correct": sum(s["accuracy"] for s in scores),
            "accuracy": mean(s["accuracy"] for s in scores), "context_recall": mean(recall) if recall else None,
            "n_context_recall": len(recall), "answer_status": dict(Counter(t.status for t in traces)),
            "tool_calls": dict(sum((Counter(t.tool_counts) for t in traces), Counter())),
            "execution": usd, "judge": judge, "interrupted_execution": overhead,
            "judge_failures": len(list((root / "questions").glob("*/judge_failures/*.json")))}


def run(output, questions, make_adapter):
    draws = []
    saved = {}
    for repeat in range(1, 4):
        for arm in ARMS:
            folder = output / f"{arm}-{repeat}"
            adapter = make_adapter(arm, folder)
            prepared = prepare_draw(adapter, folder)
            draws.append({"arm": arm, "repeat": repeat, **prepared})
            saved[(arm, repeat)] = adapter
            save(output / "summary.json", {"status": "designing", "draws": draws})
    save(output / "summary.json", {"status": "evaluating", "draws": draws})
    for row in draws:
        if row["status"] == "invalid_construction":
            continue
        arm, repeat = row["arm"], row["repeat"]
        folder = output / f"{arm}-{repeat}"
        row.update(evaluate(saved[(arm, repeat)], folder, questions))
        save(output / "summary.json", {"status": "evaluating", "draws": draws})
    groups = {}
    for arm in ARMS:
        valid = [r for r in draws if r["arm"] == arm and r["status"] == "evaluated"]
        group = {"attempted": 3, "evaluated": len(valid), "agents": [r["structure"]["agents"] for r in valid]}
        for key, values in (("accuracy", [r["accuracy"] for r in valid]),
                            ("execution_usd", [r["execution"]["usd"] for r in valid])):
            group[key] = {"mean": mean(values) if values else None, "sd": stdev(values) if len(values) > 1 else None}
        groups[arm] = group
    summary = {"status": "complete" if all(g["evaluated"] == 3 for g in groups.values()) else "incomplete",
               "arms": groups, "standard_reference": STANDARD_REFERENCE, "draws": draws}
    save(output / "summary.json", summary)
    return summary
