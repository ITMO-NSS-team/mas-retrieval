"""MAS-Zero adapter for RAG benchmark evaluation.

Per-question RAG adaptation of MAS-Zero (Designing Multi-Agent Systems
with Zero Supervision), with retrieval tools wired into the generated sub-MAS so
it is comparable to the RAG systems in this harness. The algorithm runs three
steps for every question, inference-time only, with NO gold answer used:

1. Initial archive: each block (COT / COT_SC / Reflexion / LLM_debate) is run on
   the question and self-scored by MAS-Feedback.
2. Meta-iterations: the meta-model decomposes the question into sub-tasks and
   wires a sub-MAS (code generation); each generation is executed, self-scored
   by MAS-Feedback (solvability + completeness -> fitness), and refined via a
   reflexion prompt that consumes the intermediate outputs + memory.
3. Self-Verification: a list-wise judge selects the best answer among ALL
   candidate solutions produced across the iteration.

Contrast with the `adas` system, which is the single-step code-generation
baseline (no decomposition, feedback loop, or self-verification).
"""

from __future__ import annotations

import json
import asyncio
import logging
import os
import re
import types
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable

import openai

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
from marlib.adapters.code_execution import run_generated
if TYPE_CHECKING:
    from marlib.retriever.core import Document, Retriever
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker
from marlib.tracing.resources import (
    BudgetExhausted, ResourceLimits, ResourceSession, TrackedCompletion, completion_request,
)

from .blocks import INIT_BLOCKS, get_init_archive
from .core import (
    ANSWER_PATTERN,
    TOO_HARD_MARK,
    AgentSystem,
    Info,
    LLMAgentBase,
)
from .feedback import mas_feedback, self_verify
from .prompts import (
    PROPOSE_SYSTEM_PROMPT,
    REFLECT_AFTER_EVAL_PROMPT,
    build_propose_prompt,
)
from .tracing import CandidateTrace, MASZeroTrace

logger = logging.getLogger(__name__)

_DEFAULT_COT_INSTRUCTION = (
    "Please think step by step and provide your answer. "
    "Think carefully about the question and the retrieved context."
)
_DEFAULT_DEBATE_ROLES = [
    "an analytical researcher",
    "a critical reviewer",
    "a creative problem solver",
]
_DEFAULT_BLOCKS = ["COT", "COT_SC", "Reflexion", "LLM_debate"]


@register("mas_zero")
class MASZeroAdapter(AbstractAdapter):
    """Full MAS-Zero meta-agent: decompose -> feedback loop -> self-verify."""

    supports_resource_limits = True
    supported_generation_modes = ("per_task",)

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode not in (None, "per_task"):
            raise ValueError("MAS-Zero supports per_task only; one_time would mislabel the algorithm")
        self._generation_mode = "per_task"
        self._isolate_code = self._config.get("isolate_generated_code", True)
        if not isinstance(self._isolate_code, bool):
            raise ValueError("isolate_generated_code must be boolean")
        self._limits = ResourceLimits(**{"wall_seconds": 900, **self._config.get("resource_limits", {})})
        if self._limits.scope != "full_answer":
            raise ValueError("MAS-Zero requires full_answer resource limits")
        if self._isolate_code and self._limits.wall_seconds is None:
            raise ValueError("Isolated MAS-Zero requires wall_seconds")

        self._meta_model: str = self._config.get("meta_model", self._model)
        # Zero-supervision verifier; defaults to the node model (no o3-mini needed).
        self._verifier_model: str = self._config.get("verifier_model", self._model)
        self._top_k: int = self._config.get("top_k", 20)
        self._n_generation: int = self._config.get("n_generation", 10)
        self._max_round: int = self._config.get("max_round", 2)
        self._max_sc: int = self._config.get("max_sc", 3)
        # Stop the search once a candidate reaches this self-assessed fitness.
        self._fitness_threshold: float = self._config.get("fitness_threshold", 1.0)
        self._cot_instruction: str = self._config.get(
            "cot_instruction", _DEFAULT_COT_INSTRUCTION
        )
        self._debate_roles: list[str] = (
            self._config.get("debate_roles") or list(_DEFAULT_DEBATE_ROLES)
        )
        self._block_names: list[str] = (
            self._config.get("blocks") or list(_DEFAULT_BLOCKS)
        )
        if set(self._block_names) - INIT_BLOCKS.keys():
            raise ValueError("Unknown MAS-Zero seed block")

        self._trace_enabled: bool = self._config.get("trace", True) or os.environ.get(
            "MAS_ZERO_TRACE", ""
        ).lower() in ("1", "true", "yes")
        if self._n_generation < 0 or self._max_round < 1 or self._max_sc < 1:
            raise ValueError("Invalid MAS-Zero iteration counts")
        if "debug_max" in self._config:
            raise ValueError("debug_max is not implemented in this RAG variant")

        logger.info(
            "MASZeroAdapter: meta=%s node=%s verifier=%s blocks=%s "
            "n_generation=%d max_round=%d max_sc=%d trace=%s",
            self._meta_model,
            self._model,
            self._verifier_model,
            self._block_names,
            self._n_generation,
            self._max_round,
            self._max_sc,
            self._trace_enabled,
        )

    @property
    def name(self) -> str:
        return "mas_zero"

    def effective_config(self) -> dict[str, Any]:
        return {"variant": "mas_zero_rag_self_feedback_v4" if self._isolate_code else "mas_zero_rag_self_feedback_v4_in_process", "generation_mode": "per_task",
                "isolate_generated_code": self._isolate_code, "security_sandbox": False,
                "node_temperature_policy": "generated", "worker_deadline": "remaining_full_answer_budget",
                "model": self._model, "meta_model": self._meta_model,
                "verifier_model": self._verifier_model, "n_generation": self._n_generation,
                "blocks": self._block_names, "max_round": self._max_round,
                "max_sc": self._max_sc, "fitness_threshold": self._fitness_threshold,
                "resource_limits": asdict(self._limits), "trace": self._trace_enabled,
                "stop_policy": "best_completed_candidate_by_self_fitness_without_new_call"}

    def generate_system(self, question: str) -> str:
        """MAS-Zero designs a fresh architecture per question inside execute().

        Exposed for API compatibility; returns a short description of the setup.
        """
        return (
            f"MAS-Zero per-question search (meta={self._meta_model}, "
            f"verifier={self._verifier_model}, blocks={self._block_names}, "
            f"n_generation={self._n_generation})"
        )

    # ── tool closures ─────────────────────────────────────────────────────────

    def _make_tool_closures(self, tracker: TokenTracker, session: ResourceSession) -> tuple[Any, Any, Any]:
        last_retrieved: list[Document] = []
        retriever = self._retriever

        def retrieve_fn(query: str, top_k: int = 20) -> str:
            nonlocal last_retrieved
            session.tool()
            with tracker.track_tool("retrieve", query, top_k) as results:
                docs, formatted = do_retrieve(retriever, query, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def rerank_fn(query: str, top_k: int = 10) -> str:
            nonlocal last_retrieved
            session.tool()
            with tracker.track_tool("rerank", query, top_k) as results:
                docs, formatted = do_rerank(retriever, query, last_retrieved, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def calc_fn(expression: str) -> str:
            session.tool()
            with tracker.track_tool("calculate", expression, 0):
                result = do_calculate(expression)
            return result

        return retrieve_fn, rerank_fn, calc_fn

    # ── meta-model (propose / reflexion) ──────────────────────────────────────

    def _call_meta(
        self,
        messages: list[dict],
        usage_callback: Callable[[int, int], None],
    ) -> dict | None:
        """Call the meta-model once; return a validated solution dict or None."""
        client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_retries=0,
            timeout=60.0,
        )
        logical_call_id = uuid4().hex
        response = completion_request(client, usage_callback, logical_call_id,
            model=self._meta_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        def reject(reason: str) -> None:
            diagnostic = {"stage": "meta_proposal", "error": reason,
                          "logical_call_id": logical_call_id}
            self._diagnostics.append(diagnostic)
            if isinstance(usage_callback, TrackedCompletion):
                usage_callback.session.write({"kind": "proposal_rejected", **diagnostic})
            logger.warning("Meta-model proposal rejected: %s", reason)

        try:
            solution = json.loads(text)
        except json.JSONDecodeError:
            reject("Meta-model returned invalid JSON")
            return None
        if not isinstance(solution, dict) or not all(isinstance(solution.get(k), str) for k in ("name", "thought", "code")):
            reject("Meta-model missing required string keys: name, thought, code")
            return None
        if "def forward(self, taskInfo):" not in solution["code"]:
            reject("Generated code missing forward() signature")
            return None
        try:
            compile(solution["code"], "<generated>", "exec")
        except SyntaxError as e:
            reject(f"Generated code has syntax error: {e}")
            return None
        return solution

    # ── forward execution ─────────────────────────────────────────────────────

    def _exec_forward(
        self,
        code: str,
        system: AgentSystem,
        agent_class: type,
        task_info: Info,
    ) -> Info:
        if self._isolate_code:
            session = system._usage_callback.session
            value = asyncio.run(run_generated(
                code=code, question=task_info.content,
                core_path=Path(__file__).with_name("core.py"),
                system_config={"node_model": system.node_model, "cot_instruction": system.cot_instruction,
                               "max_round": system.max_round, "max_sc": system.max_sc,
                               "debate_role": system.debate_role},
                session=session, retriever=self._retriever, model=self._model, temperature=None,
                artifacts=self._question_artifacts / f"worker_{uuid4().hex}"))
            return Info(**value)
        namespace: dict[str, Any] = {"LLMAgentBase": agent_class, "Info": Info,
                                     "__builtins__": __builtins__}
        exec(  # noqa: S102 — running model-generated architecture by design
            code,
            namespace,
            namespace,
        )
        forward_fn = namespace.get("forward")
        if not callable(forward_fn):
            callables = [v for v in namespace.values() if callable(v)]
            if not callables:
                raise RuntimeError("Generated code defined no callable")
            forward_fn = callables[0]
        system.forward = types.MethodType(forward_fn, system)
        return system.forward(task_info)

    @staticmethod
    def _extract_answer(content: str) -> str:
        content = content or ""
        # make_final_answer appends this delimiter after the reasoning, which
        # itself can contain an earlier "Answer:". Preserve multiline answers.
        if "\n\nAnswer:" in content:
            answer = content.rsplit("\n\nAnswer:", 1)[1].strip()
        else:
            match = re.search(ANSWER_PATTERN, content)
            answer = match.group(1).strip() if match else content.strip()
        if TOO_HARD_MARK in answer:
            answer = answer.split(TOO_HARD_MARK)[0].strip()
        return answer

    # ── main entry point ──────────────────────────────────────────────────────

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer="",  # No reference answer in objects reachable by generated code.
        )

        root = Path(self._run_context.get("artifact_dir", "logs/mas_zero"))
        identity = sha256(question_id.encode()).hexdigest()[:16]
        self._question_artifacts = root / f"{identity}_{uuid4().hex}"
        self._question_artifacts.mkdir(parents=True, exist_ok=False)
        session = ResourceSession(tracker, self._limits, self._question_artifacts / "events.jsonl",
                                  coverage="mas_zero_parent_calls_and_worker_rpc" if self._isolate_code else "mas_zero_in_process_helpers")
        tracker.tool_event_sink = session.write
        self._diagnostics: list[dict] = []
        trace: MASZeroTrace | None = None
        if self._trace_enabled:
            trace = MASZeroTrace(
                question_id=question_id,
                meta_model=self._meta_model,
                node_model=self._model,
                verifier_model=self._verifier_model,
                blocks_offered=list(self._block_names),
                n_generation=self._n_generation,
            )

        node_cb = TrackedCompletion(session, "answer_execution")
        meta_cb = TrackedCompletion(session, "construction")
        verifier_cb = TrackedCompletion(session, "internal_verification")

        retrieve_fn, rerank_fn, calc_fn = self._make_tool_closures(tracker, session)

        system = AgentSystem()
        system.node_model = self._model
        system.cot_instruction = self._cot_instruction
        system.max_round = self._max_round
        system.max_sc = self._max_sc
        system.debate_role = self._debate_roles
        system._retrieve_fn = retrieve_fn
        system._rerank_fn = rerank_fn
        system._calc_fn = calc_fn
        system._usage_callback = node_cb

        node_model = self._model

        class BoundAgent(LLMAgentBase):
            def __init__(self, *args, **kwargs):
                # Generated code may omit the callback/model; accounting still applies.
                kwargs["usage_callback"] = node_cb
                kwargs["model"] = node_model
                super().__init__(*args, **kwargs)

        agent_class = BoundAgent
        task_info = Info("task", "user", question, None, None, None, -1)

        candidates: list[dict] = []
        memory: list[dict] = []

        def evaluate(
            code: str, name: str, thought: str, stage: str, generation: int
        ) -> dict:
            """Run one architecture, self-score it, and record a candidate."""
            cand: dict = {
                "name": name,
                "code": code,
                "thought": thought,
                "stage": stage,
                "generation": generation,
                "answer": "",
                "fitness": 0.0,
                "feedback": "",
                "sub_tasks": None,
                "agents": None,
                "error": None,
                "error_type": None,
                "error_stage": None,
            }
            stopped = None
            execution_stage = "candidate_execution"
            try:
                session.check()
                result = self._exec_forward(code, system, agent_class, task_info)
                content = result.content if hasattr(result, "content") else ""
                cand["thinking"] = content
                cand["answer"] = self._extract_answer(content)
                cand["sub_tasks"] = getattr(result, "sub_tasks", None)
                cand["agents"] = getattr(result, "agents", None)
                execution_stage = "internal_feedback"
                cand["fitness"], cand["feedback"] = mas_feedback(
                    question,
                    cand["sub_tasks"],
                    cand["agents"],
                    cand["answer"],
                    model=self._verifier_model,
                    usage_callback=verifier_cb,
                )
            except BudgetExhausted as e:
                cand["error"] = str(e)
                cand["error_type"], cand["error_stage"] = type(e).__name__, execution_stage
                stopped = e
            except Exception as e:  # generated code / runtime failure
                logger.warning("Candidate '%s' failed: %s", name, e)
                cand["error"] = str(e)
                cand["error_type"], cand["error_stage"] = type(e).__name__, execution_stage
            candidates.append(cand)
            memory.append({cand["answer"]: round(cand["fitness"], 3)})
            if trace is not None:
                trace.candidates.append(
                    CandidateTrace(
                        stage=stage,
                        generation=generation,
                        name=name,
                        thought=thought,
                        code=code,
                        answer=cand["answer"],
                        thinking=cand.get("thinking", ""),
                        fitness=cand["fitness"],
                        feedback=cand["feedback"],
                        sub_tasks=cand["sub_tasks"],
                        agents=cand["agents"],
                        error=cand["error"],
                        error_type=cand["error_type"],
                        error_stage=cand["error_stage"],
                    )
                )
            session.write({"kind": "candidate", **cand})
            if trace is not None:
                self._save_trace(trace)
            if stopped:
                raise stopped
            return cand

        try:
            archive = get_init_archive(self._block_names)
            solved = False

            # 1. Initial archive evaluation.
            for block in archive:
                cand = evaluate(
                    block["code"], block["name"], block.get("thought", ""),
                    stage="initial", generation=-1,
                )
                block["fitness"] = round(cand["fitness"], 3)
                if cand["fitness"] >= self._fitness_threshold:
                    solved = True
                    if trace is not None:
                        trace.stopped_early = True
                    break

            # 2. Meta-iterations (decompose -> evaluate -> reflexion).
            # A meta-model failure here is non-fatal: we degrade to
            # self-verification over whatever candidates were collected.
            if not solved and self._n_generation > 0:
                try:
                    self._meta_iterations(
                        question, archive, evaluate, memory, meta_cb, trace
                    )
                except BudgetExhausted:
                    raise
                except Exception as e:
                    self._diagnostics.append({"stage": "meta_iteration", "error": str(e)})
                    logger.warning("Meta-iteration aborted: %s", e)

            # 3. Self-verification across all candidates.
            answer, best_idx = self._select_answer(question, candidates, verifier_cb)

            if trace is not None:
                trace.selected_index = best_idx
                trace.selected_answer = answer

        except BudgetExhausted as e:
            tracker.set_error(str(e))
            usable = [(i, c) for i, c in enumerate(candidates) if c["answer"].strip()]
            best_idx, best = max(usable, key=lambda pair: pair[1]["fitness"]) if usable else (-1, {"answer": ""})
            answer = best["answer"]
            self._diagnostics.append({"stage": "budget", "error": str(e), "selected_index": best_idx})
            if trace is not None:
                trace.selected_index, trace.selected_answer = best_idx, answer
                trace.execution_error = f"budget_exhausted: {e}"
        except Exception as e:
            logger.error("MAS-Zero execution failed: %s", e)
            tracker.set_error(str(e))
            if trace is not None:
                trace.execution_error = str(e)
            answer = ""

        if not answer.strip() and not tracker._error:
            tracker.set_error("No candidate produced a nonempty answer")
        if trace is not None:
            self._save_trace(trace)

        log = tracker.to_question_log(answer)
        log.gold_answer = gold_answer
        if any(d["stage"] == "budget" for d in self._diagnostics):
            log.status, log.failure_kind = "budget_exhausted", "budget_exhausted"
        elif log.error:
            log.failure_kind = "unknown"
        log.resource_summary = session.summary()
        log.resource_summary["worker_process_deadline"] = self._isolate_code
        log.resource_summary["security_sandbox"] = False
        log.resource_summary["candidate_failures"] = sum(bool(c["error"]) for c in candidates)
        log.resource_summary["diagnostics"] = self._diagnostics
        log.artifact_paths["events"] = str(session.journal)
        if trace is not None:
            log.artifact_paths["trace"] = str(self._question_artifacts / "trace.json")
        session.write({"kind": "question_end", "status": log.status, "resources": log.resource_summary})
        return answer, log

    def _meta_iterations(
        self,
        question: str,
        archive: list[dict],
        evaluate: Callable[..., dict],
        memory: list[dict],
        meta_cb: Callable[[int, int], None],
        trace: MASZeroTrace | None,
    ) -> None:
        """Run the decompose -> evaluate -> reflexion loop in place."""
        msg_list: list[dict] = [
            {"role": "system", "content": PROPOSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_propose_prompt(
                    archive,
                    question,
                    benchmark_description=self._benchmark_description,
                    sample_questions=self._sample_questions,
                ),
            },
        ]
        if self._isolate_code:
            msg_list[0]["content"] += (
                "\nExecution uses a supervised worker. Only math, random, statistics and collections "
                "imports are available. Use the supplied LLMAgentBase, Info and self.retrieve/rerank/calculate. "
                "No external clients, file access, classes, threads, subprocesses or private attributes "
                "except self._usage_callback. Return Info from forward(self, taskInfo)."
            )
        next_solution = self._call_meta(msg_list, meta_cb)

        for n in range(self._n_generation):
            if next_solution is None:
                # Re-propose from scratch if the meta-model stumbled.
                next_solution = self._call_meta(msg_list, meta_cb)
                if next_solution is None:
                    break

            cand = evaluate(
                next_solution["code"],
                next_solution.get("name", f"generation-{n + 1}"),
                next_solution.get("thought", ""),
                stage="generation",
                generation=n + 1,
            )
            if cand["fitness"] >= self._fitness_threshold:
                if trace is not None:
                    trace.stopped_early = True
                break

            # Reflexion: feed intermediate outputs + memory back in.
            assistant_payload = dict(next_solution)
            assistant_payload["sub_tasks"] = cand["sub_tasks"]
            assistant_payload["agents"] = cand["agents"]
            assistant_payload["final_response"] = cand["answer"]
            assistant_payload["fitness"] = round(cand["fitness"], 3)
            if cand["error"]:
                assistant_payload["error"] = cand["error"]
            msg_list.append(
                {"role": "assistant", "content": json.dumps(assistant_payload)}
            )
            reflect = REFLECT_AFTER_EVAL_PROMPT.format(last_round=n + 1, prev_round=n)
            reflect += f"\n\nVerifier feedback: {cand['feedback']}"
            reflect += f"\n\nmemory: {json.dumps(memory)}"
            msg_list.append({"role": "user", "content": reflect})

            if n + 1 < self._n_generation:
                next_solution = self._call_meta(msg_list, meta_cb)

    def _select_answer(
        self,
        question: str,
        candidates: list[dict],
        verifier_cb: Callable[[int, int], None],
    ) -> tuple[str, int]:
        """Self-verify across candidates; degrade gracefully on any failure."""
        if not candidates:
            return "", -1
        try:
            best_idx = self_verify(
                question, candidates,
                model=self._verifier_model, usage_callback=verifier_cb,
            )
        except BudgetExhausted:
            raise
        except Exception as e:
            self._diagnostics.append({"stage": "selection_fallback", "error": str(e)})
            logger.warning("Self-verification failed: %s", e)
            best_idx = max(
                range(len(candidates)),
                key=lambda i: (bool(candidates[i].get("answer", "").strip()), candidates[i].get("fitness", 0.0)),
            )
        return candidates[best_idx]["answer"], best_idx

    # ── tracing helpers ───────────────────────────────────────────────────────

    def _save_trace(self, trace: MASZeroTrace) -> None:
        logger.debug("MAS-Zero trace:\n%s", trace.summary())
        try:
            path = self._question_artifacts / "trace.json"
            with open(path, "w") as f:
                f.write(trace.model_dump_json(indent=2))
            logger.info("Trace saved to %s", path)
        except OSError as e:
            logger.warning("Failed to save trace: %s", e)
