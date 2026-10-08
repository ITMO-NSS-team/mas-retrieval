"""P006: one generator prompt addition, original ADAS execution policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import openai

from marlib.adapters.base import register
from marlib.tracing.observation import CallObserver
from marlib.tracing.tracker import TokenTracker

from ..adas.adapter import ADASAdapter
from ..adas.prompts import SYSTEM_PROMPT


COMPACTNESS_BLOCK = (
    "Prefer the smallest workflow sufficient to solve the task. "
    "Add an agent only when it serves a distinct necessary function or an independent subtask. "
    "Avoid redundant agents, repeated model calls, and coordination steps."
)
PROMPT_SUFFIX = "\n\n" + COMPACTNESS_BLOCK + "\n"


@register("adas_compact_prompt")
class ADASCompactPromptAdapter(ADASAdapter):
    supported_generation_modes = ("one_time",)

    def __init__(self, retriever, model="gpt-4o-mini", **kwargs):
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode != "one_time":
            raise ValueError("adas_compact_prompt supports one_time only")
        self._on_benchmark_change()

    @property
    def name(self):
        return "adas_compact_prompt_one_time"

    def _on_benchmark_change(self):
        super()._on_benchmark_change()
        self._construction_path = None
        self._construction_summary = None
        self._construction_seconds = 0.0

    def set_run_context(self, **context):
        if context != self._run_context:
            self._on_benchmark_change()
        super().set_run_context(**context)

    def effective_config(self):
        return {**super().effective_config(), "variant": "adas_compact_prompt_cl_v1",
                "generator_prompt_suffix": PROMPT_SUFFIX, "meta_model": self._meta_model,
                "blocks": [b["name"] for b in self._blocks], "max_round": self._max_round,
                "max_sc": self._max_sc, "debug_max": self._debug_max,
                "cot_instruction": self._cot_instruction, "debate_roles": self._debate_roles,
                "generator_temperature": "omitted", "node_temperature": "generated",
                "output_limit": "omitted", "request_timeout": openai.DEFAULT_TIMEOUT.as_dict(),
                "sdk_retries": openai.DEFAULT_MAX_RETRIES, "meta_backoff_max_tries": 3,
                "node_backoff_max_tries": 5, "node_json_attempts": 5,
                "resource_limits": None, "worker_isolation": False,
                "generation_fallback": "original_first_seed",
                "construction_exception_policy": "original_retry_on_next_question_if_uncached",
                "construction_cost_attribution": "question_on_which_attempt_occurred",
                "observation": "passive HTTP sends; original SDK retries and request options"}

    def _call_meta_model(self, prompt):
        modified = prompt + PROMPT_SUFFIX
        self._observer.write({"kind": "generator_prompt", "phase": "construction",
                              "original_user_prompt": prompt, "added_suffix": PROMPT_SUFFIX,
                              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": modified}]})
        return super()._call_meta_model(modified)

    def generate_system(self, question):
        if self._cached_system is not None:
            return super().generate_system(question)
        started = time.perf_counter()
        error = None
        # Each original-generation attempt is preserved, including exceptions
        # before a workflow is cached. Do not add a new retry/fallback policy.
        self._construction_path = self._artifacts / "construction.json"
        try:
            with self._observer.phase("construction"):
                return super().generate_system(question)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            self._construction_seconds += elapsed
            self._construction_summary = self._observer.summary("construction")
            self._construction_path.write_text(json.dumps({
                "workflow": self._cached_system, "error": error,
                "generator_prompt_suffix": PROMPT_SUFFIX, "wall_seconds": elapsed,
                "resources": self._construction_summary,
                "events": str(self._observer.journal),
            }, indent=2) + "\n")

    def _make_tool_closures(self, tracker):
        tracker.tool_event_sink = self._observer.write
        return super()._make_tool_closures(tracker)

    def _save_trace(self, trace):
        # Preserve the original optional trace wrapper, but never overwrite a
        # previous run/question. HTTP and agent-instance traces are always saved.
        path = self._artifacts / "trace.json"
        path.write_text(trace.model_dump_json(indent=2) + "\n")

    def execute(self, question_id, question, gold_answer):
        root = Path(self._run_context.get("artifact_dir", f"logs/{self.name}"))
        self._artifacts = root / f"{hashlib.sha256(question_id.encode()).hexdigest()[:12]}_{uuid4().hex}"
        self._artifacts.mkdir(parents=True, exist_ok=False)
        tracker = TokenTracker(question_id, question, "")
        self._observer = CallObserver(tracker, self._artifacts / "events.jsonl")
        construction_before = self._construction_seconds
        with self._observer.phase("answer_execution"):
            answer, original_log = super().execute(question_id, question, "")
        # Original ADAS usage callbacks remain active for behavioral parity, but
        # only HTTP attempt observations enter the new ledger (no double count).
        tracker._tool_calls = original_log.tool_calls
        if original_log.error:
            tracker.set_error(original_log.error)
        log = tracker.to_question_log(answer)
        log.gold_answer = gold_answer
        construction_seconds = self._construction_seconds - construction_before
        log.resource_summary = {
            **self._observer.summary(), "construction_reference": self._construction_summary,
            "construction_charged_here": any(c.phase == "construction" for c in log.llm_calls),
            "construction_seconds_here": construction_seconds,
            "answer_execution_seconds": max(0.0, log.total_latency_ms / 1000 - construction_seconds),
            "tool_calls": log.num_tool_calls,
            "n_agents": len({a["agent_id"] for a in self._observer.agents}),
            "agent_count_definition": "LLMAgentBase instances created during this answer, not model calls",
        }
        log.artifact_paths = {"events": str(self._observer.journal)}
        if self._construction_path:
            log.artifact_paths["construction"] = str(self._construction_path)
        trace_path = self._artifacts / "trace.json"
        if trace_path.exists():
            log.artifact_paths["trace"] = str(trace_path)
        self._observer.write({"kind": "question_end", "status": log.status,
                              "error": log.error, "resources": log.resource_summary})
        return answer, log
