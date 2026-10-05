"""Single-proposal CL code control, using ADAS seeds and a supervised worker."""
from __future__ import annotations

import json
from pathlib import Path

from marlib.adapters.base import register
from marlib.adapters.code_execution import GeneratedCodeError, run_generated
from marlib.adapters.code_worker import validate_code
from marlib.adapters.construction import task_description

from ..adas import core
from ..adas.blocks import RAG_BLOCKS
from ..adas.prompts import SYSTEM_PROMPT, build_meta_prompt
from ..generated_single_agent.adapter import GeneratedSingleAgentAdapter


EXECUTION_POLICY = """
Use model/tool calls economically. Every model and tool call shares the execution
budget. Use only self.retrieve, self.rerank, self.calculate, LLMAgentBase and Info.
The runtime fixes every node's model and temperature to the comparison settings.
Only math, random, statistics and collections imports are available. No external
clients, file access, private attributes (except self._usage_callback), classes,
threads or subprocesses. Return Info from forward(self, taskInfo).
"""


@register("adas_budgeted")
class ADASBudgetedAdapter(GeneratedSingleAgentAdapter):
    def __init__(self, retriever, model="gpt-4o-mini", **kwargs):
        kwargs["resource_limits"] = {"wall_seconds": 120, **kwargs.get("resource_limits", {})}
        super().__init__(retriever, model, **kwargs)
        if self._limits.wall_seconds is None:
            raise ValueError("adas_budgeted requires wall_seconds")

    @property
    def name(self):
        return "adas_budgeted_one_time"

    @property
    def generated_system(self):
        return {"code": self._prompt} if self._prompt else None

    def effective_config(self):
        cfg = super().effective_config()
        cfg.update(variant="adas_budgeted_cl_v1", generator_instruction=SYSTEM_PROMPT + EXECUTION_POLICY,
                   proposal_policy="one_proposal_no_repair_no_fallback", node_json_attempts=5,
                   max_round=2, max_sc=3, tool_execution="sequential",
                   worker_deadline="kill_process_group", security_sandbox=False)
        return cfg

    def _construction_messages(self):
        return [{"role": "system", "content": SYSTEM_PROMPT + EXECUTION_POLICY},
                {"role": "user", "content": build_meta_prompt(
                    list(RAG_BLOCKS), question=None,
                    benchmark_description=task_description(self._benchmark_description, self._sample_questions),
                    sample_questions=None)}]

    def _generation_options(self):
        return {"response_format": {"type": "json_object"}}

    def _artifact_content(self):
        return {"code": self._prompt}

    def _execution_coverage(self):
        return "supervised_code_model_and_tool_rpc"

    def _decode_generated(self, text):
        try:
            value = json.loads(text)
            if not isinstance(value, dict) or not isinstance(value.get("code"), str):
                raise ValueError("Generator must return a JSON object with code")
            validate_code(value["code"])
            return value["code"]
        except (ValueError, SyntaxError) as exc:
            raise GeneratedCodeError(str(exc)) from exc

    async def _answer(self, question, prompt, tracker, session):
        # SingleAgentAdapter created the session journal inside the question artifacts.
        artifacts = session.journal.parent
        value = await run_generated(code=prompt, question=question, core_path=core.__file__,
            system_config={"node_model": self._model, "cot_instruction": "Think step by step using retrieved evidence.",
                           "max_round": 2, "max_sc": 3,
                           "debate_role": ["an analytical researcher", "a critical reviewer", "a creative problem solver"]},
            session=session, retriever=self._retriever, model=self._model,
            temperature=self._temperature, artifacts=artifacts)
        content = value["content"] or ""
        return content.rsplit("\n\nAnswer:", 1)[-1].strip()

    def _failure_kind(self, error):
        return "generated_code_error" if isinstance(error, GeneratedCodeError) else "unknown"

    def _attach_construction(self, log):
        super()._attach_construction(log)
        events = log.artifact_paths.get("execution_events")
        if events:
            log.artifact_paths["worker_stderr"] = str(Path(events).parent / "worker.stderr")
            log.resource_summary["execution"]["worker_process_deadline"] = True
            log.resource_summary["execution"]["security_sandbox"] = False
