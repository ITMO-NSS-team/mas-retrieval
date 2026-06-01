"""MAS-Zero adapter for RAG benchmark evaluation.

Faithful per-question implementation of MAS-Zero (Designing Multi-Agent Systems
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
import logging
import os
import re
import types
from typing import Any, Callable

import backoff
import openai

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
from marlib.retriever.core import Document, Retriever
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker

from .blocks import get_init_archive
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

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)

        self._meta_model: str = self._config.get("meta_model", self._model)
        # Zero-supervision verifier; defaults to the node model (no o3-mini needed).
        self._verifier_model: str = self._config.get("verifier_model", self._model)
        self._top_k: int = self._config.get("top_k", 20)
        self._n_generation: int = self._config.get("n_generation", 10)
        self._max_round: int = self._config.get("max_round", 2)
        self._max_sc: int = self._config.get("max_sc", 3)
        self._debug_max: int = self._config.get("debug_max", 3)
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

        self._trace_enabled: bool = self._config.get("trace", False) or os.environ.get(
            "MAS_ZERO_TRACE", ""
        ).lower() in ("1", "true", "yes")

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

    def _make_tool_closures(self, tracker: TokenTracker) -> tuple[Any, Any, Any]:
        last_retrieved: list[Document] = []
        retriever = self._retriever

        def retrieve_fn(query: str, top_k: int = 20) -> str:
            nonlocal last_retrieved
            with tracker.track_tool("retrieve", query, top_k) as results:
                docs, formatted = do_retrieve(retriever, query, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def rerank_fn(query: str, top_k: int = 10) -> str:
            nonlocal last_retrieved
            with tracker.track_tool("rerank", query, top_k) as results:
                docs, formatted = do_rerank(retriever, query, last_retrieved, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def calc_fn(expression: str) -> str:
            with tracker.track_tool("calculate", expression, 0):
                result = do_calculate(expression)
            return result

        return retrieve_fn, rerank_fn, calc_fn

    def _usage_cb(self, tracker: TokenTracker, model: str) -> Callable[[int, int], None]:
        def cb(prompt_tokens: int, completion_tokens: int) -> None:
            tracker.log_llm_call(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,
            )

        return cb

    # ── meta-model (propose / reflexion) ──────────────────────────────────────

    @backoff.on_exception(
        backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=3
    )
    def _call_meta(
        self,
        messages: list[dict],
        usage_callback: Callable[[int, int], None],
    ) -> dict | None:
        """Call the meta-model once; return a validated solution dict or None."""
        client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        response = client.chat.completions.create(
            model=self._meta_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        if usage:
            usage_callback(usage.prompt_tokens, usage.completion_tokens)

        text = response.choices[0].message.content or ""
        try:
            solution = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Meta-model returned invalid JSON")
            return None
        if not all(k in solution for k in ("name", "thought", "code")):
            logger.warning("Meta-model missing required keys")
            return None
        if "def forward(self, taskInfo):" not in solution["code"]:
            logger.warning("Generated code missing forward() signature")
            return None
        try:
            compile(solution["code"], "<generated>", "exec")
        except SyntaxError as e:
            logger.warning("Generated code has syntax error: %s", e)
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
        namespace: dict[str, Any] = {}
        exec(  # noqa: S102 — running model-generated architecture by design
            code,
            {"LLMAgentBase": agent_class, "Info": Info, "__builtins__": __builtins__},
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
        match = re.search(ANSWER_PATTERN, content or "")
        answer = match.group(1).strip() if match else (content or "").strip()
        if TOO_HARD_MARK in answer:
            answer = answer.split(TOO_HARD_MARK)[0].strip()
        if not match and "\n" in answer:
            answer = answer.split("\n")[-1].strip()
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
            gold_answer=gold_answer,
        )

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

        node_cb = self._usage_cb(tracker, self._model)
        meta_cb = self._usage_cb(tracker, self._meta_model)
        verifier_cb = self._usage_cb(tracker, self._verifier_model)

        retrieve_fn, rerank_fn, calc_fn = self._make_tool_closures(tracker)

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

        agent_class = self._traced_agent_class(trace) if trace is not None else LLMAgentBase
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
            }
            try:
                result = self._exec_forward(code, system, agent_class, task_info)
                content = result.content if hasattr(result, "content") else ""
                cand["answer"] = self._extract_answer(content)
                cand["sub_tasks"] = getattr(result, "sub_tasks", None)
                cand["agents"] = getattr(result, "agents", None)
                cand["fitness"], cand["feedback"] = mas_feedback(
                    question,
                    cand["sub_tasks"],
                    cand["agents"],
                    cand["answer"],
                    model=self._verifier_model,
                    usage_callback=verifier_cb,
                )
            except Exception as e:  # generated code / runtime failure
                logger.warning("Candidate '%s' failed: %s", name, e)
                cand["error"] = str(e)
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
                        fitness=cand["fitness"],
                        feedback=cand["feedback"],
                        sub_tasks=cand["sub_tasks"],
                        agents=cand["agents"],
                        error=cand["error"],
                    )
                )
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
                except Exception as e:
                    logger.warning("Meta-iteration aborted: %s", e)

            # 3. Self-verification across all candidates.
            answer, best_idx = self._select_answer(question, candidates, verifier_cb)

            if trace is not None:
                trace.selected_index = best_idx
                trace.selected_answer = answer

        except Exception as e:
            logger.error("MAS-Zero execution failed: %s", e)
            tracker.set_error(str(e))
            if trace is not None:
                trace.execution_error = str(e)
            answer = ""

        if trace is not None:
            self._save_trace(trace)

        return answer, tracker.to_question_log(answer)

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
        except Exception as e:
            logger.warning("Self-verification failed: %s", e)
            best_idx = max(
                range(len(candidates)),
                key=lambda i: candidates[i].get("fitness", 0.0),
            )
        return candidates[best_idx]["answer"], best_idx

    # ── tracing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _traced_agent_class(trace: MASZeroTrace) -> type:
        # Trace captured at architecture granularity (CandidateTrace); the agent
        # subclass is a hook point for finer tracing if needed later.
        return LLMAgentBase

    def _save_trace(self, trace: MASZeroTrace) -> None:
        logger.debug("MAS-Zero trace:\n%s", trace.summary())
        try:
            trace_dir = os.path.join("logs", "mas_zero")
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"trace_{trace.question_id}.json")
            with open(path, "w") as f:
                f.write(trace.model_dump_json(indent=2))
            logger.info("Trace saved to %s", path)
        except OSError as e:
            logger.warning("Failed to save trace: %s", e)
