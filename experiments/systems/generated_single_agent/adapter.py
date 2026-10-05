"""CL control: generate one instruction, then reuse the single-agent executor."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from uuid import uuid4

import openai

from marlib.adapters.base import register
from marlib.adapters.construction import task_description
from marlib.tracing.resources import ResourceLimits, ResourceSession, TrackedCompletion, completion_request
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker

from ..single_agent.adapter import SingleAgentAdapter


GENERATOR_INSTRUCTION = """Write a system instruction for ONE research assistant
that answers questions using the provided document corpus. Return only the
instruction, without commentary or a specific question's answer. This instruction
will be reused across the benchmark. There is no validation set or answer feedback.

Available tools:
- retrieve(query, top_k=20): dense search of the common document corpus.
- rerank(query, top_k=10): rerank the most recent retrieve results.
- calculate(expression): evaluate a mathematical expression.

Guide the agent to ground answers in retrieved evidence, rerank retrieved passages,
use the calculator for arithmetic, and give a concise final answer. Make economical
use of model and tool calls. Do not invent tools or delegate to other agents.
"""


@register("generated_single_agent")
class GeneratedSingleAgentAdapter(SingleAgentAdapter):
    supported_generation_modes = ("one_time",)

    def __init__(self, retriever, model="gpt-4o-mini", **kwargs):
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode not in (None, "one_time"):
            raise ValueError("generated_single_agent supports one_time only")
        self._generation_mode = "one_time"
        self._meta_model = self._config.get("meta_model", self._model)
        self._generator_temperature = self._config.get("generator_temperature", 0.3)
        if not isinstance(self._generator_temperature, (int, float)) or not 0 <= self._generator_temperature <= 2:
            raise ValueError("generator_temperature must be between 0 and 2")
        self._construction_limits = ResourceLimits(**{
            "scope": "construction", "max_requests": 3,
            **self._config.get("construction_limits", {})})
        if self._construction_limits.scope != "construction":
            raise ValueError("construction_limits must use construction scope")
        self._on_benchmark_change()

    @property
    def name(self):
        return "generated_single_agent_one_time"

    def _on_benchmark_change(self):
        self._prompt = None
        self._construction_attempted = False
        self._construction_summary = None
        self._construction_path = None
        self._construction_journal = None

    def set_run_context(self, **context):
        if context != self._run_context:
            self._on_benchmark_change()
        super().set_run_context(**context)

    def effective_config(self):
        config = super().effective_config()
        config.pop("system_prompt")
        return {**config, "variant": "generated_single_agent_cl_v1",
                "meta_model": self._meta_model,
                "generator_temperature": self._generator_temperature,
                "generator_instruction": GENERATOR_INSTRUCTION,
                "construction_limits": asdict(self._construction_limits),
                "construction_failure_policy": "fail_repeat_without_regeneration",
                "construction_cost_attribution": "first_attempted_question_only"}

    @property
    def generated_system(self):
        return {"system_prompt": self._prompt} if self._prompt else None

    def generate_system(self, question):
        if self._prompt is None:
            raise RuntimeError("Prompt construction is accounted inside execute()")
        return self._prompt

    def _prepare_prompt(self, tracker: TokenTracker, artifacts: Path):
        if self._construction_attempted:
            if self._prompt is None:
                raise RuntimeError("CL prompt construction failed; this repeat cannot execute")
            return self._prompt, {"construction_charged_here": False}

        self._construction_attempted = True
        self._construction_path = artifacts / "construction.json"
        session = ResourceSession(tracker, self._construction_limits,
                                  artifacts / "construction.jsonl", coverage="prompt_generator_chat_completions")
        self._construction_journal = session.journal
        messages = [{"role": "system", "content": GENERATOR_INSTRUCTION},
                    {"role": "user", "content": task_description(
                        self._benchmark_description, self._sample_questions)}]
        raw_output, error = None, None
        try:
            with openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"),
                               api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0) as client:
                response = completion_request(client, TrackedCompletion(session, "construction"),
                                              uuid4().hex, model=self._meta_model, messages=messages,
                                              temperature=self._generator_temperature)
            raw_output = response.choices[0].message.content
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise ValueError("Generator returned an empty system instruction")
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raise ValueError("Generator instruction was truncated by the output limit")
            self._prompt = raw_output.strip()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._construction_summary = session.summary()
            self._construction_path.write_text(json.dumps({
                "model": self._meta_model, "messages": messages, "raw_output": raw_output,
                "system_prompt": self._prompt, "error": error,
                "resources": self._construction_summary,
            }, ensure_ascii=False, indent=2) + "\n")
            session.write({"kind": "construction_end", "error": error})
        return self._prompt, {"construction_charged_here": True}

    def _attach_construction(self, log: QuestionLog):
        # Summary is a reference on reused questions. Only the first question's
        # llm_calls contain construction, so summing run costs cannot duplicate it.
        log.resource_summary["construction_reference"] = self._construction_summary
        log.resource_summary["construction_charged_here"] = any(
            call.phase == "construction" for call in log.llm_calls)
        if self._construction_path:
            log.artifact_paths["construction"] = str(self._construction_path)
        if self._construction_journal:
            log.artifact_paths["construction_events"] = str(self._construction_journal)
