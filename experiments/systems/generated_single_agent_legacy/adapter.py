"""Generated prompt with single-agent request/tool policy from source 9b5926b.

Retain omitted execution model_settings, default UsageLimits and @agent.tool
scheduling. Construction reuses the existing generated-instruction implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import time
from typing import Any

import openai

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from marlib.adapters.base import register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
from marlib.tracing.observation import CallObserver, ObservedAsyncHttpClient, label_async_completions
from marlib.tracing.tracker import TokenTracker

from ..generated_single_agent.adapter import GeneratedSingleAgentAdapter, GENERATOR_INSTRUCTION


@dataclass
class LegacyDeps:
    retriever: Any
    tracker: TokenTracker
    _last_retrieved: list = field(default_factory=list)


def retrieve(ctx: RunContext[LegacyDeps], query: str, top_k: int = 20) -> str:
    """Retrieve candidate passages from the knowledge base via dense search."""
    with ctx.deps.tracker.track_tool("retrieve", query, top_k) as doc_ids:
        docs, formatted = do_retrieve(ctx.deps.retriever, query, top_k)
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


def rerank(ctx: RunContext[LegacyDeps], query: str, top_k: int = 10) -> str:
    """Re-rank recently retrieved passages using a cross-encoder model."""
    with ctx.deps.tracker.track_tool("rerank", query, top_k) as doc_ids:
        if not ctx.deps._last_retrieved:
            return "Error: No documents to rerank. Call retrieve() first."
        docs, formatted = do_rerank(ctx.deps.retriever, query, ctx.deps._last_retrieved, top_k)
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


def calculate(ctx: RunContext[LegacyDeps], expression: str) -> str:
    """Evaluate a mathematical expression (e.g. '1234.5 * 0.15', 'round(456.78 / 123, 2)')."""
    with ctx.deps.tracker.track_tool("calculate", expression, 0):
        result = do_calculate(expression)
    return result


@register("generated_single_agent_legacy")
class GeneratedSingleAgentLegacyAdapter(GeneratedSingleAgentAdapter):
    supports_resource_limits = False

    def __init__(self, retriever, model="gpt-4o-mini", **kwargs):
        if "temperature" in kwargs or "resource_limits" in kwargs:
            raise ValueError("Legacy execution preserves omitted temperature and default framework limits")
        super().__init__(retriever, model, **kwargs)

    @property
    def name(self):
        return "generated_single_agent_legacy_one_time"

    def effective_config(self):
        return {"variant": "generated_single_agent_legacy_cl_v1", "model": self._model,
                "generation_mode": "one_time", "meta_model": self._meta_model,
                "generator_temperature": self._generator_temperature,
                "generator_instruction": GENERATOR_INSTRUCTION,
                "construction_limits": asdict(self._construction_limits),
                "construction_failure_policy": "fail_repeat_without_regeneration",
                "construction_cost_attribution": "first_attempted_question_only",
                "execution_source_reference": "9b5926b:experiments/systems/single_agent/adapter.py",
                "execution_model_settings": "omitted", "output_limit": "omitted",
                "tool_execution": "default @agent.tool scheduling (sequential=False)",
                "usage_limits": vars(UsageLimits()), "usage_limit_scope": "Pydantic AI logical model requests",
                "sdk_retries": openai.DEFAULT_MAX_RETRIES,
                "request_timeout": cached_async_http_client(provider="openai").timeout.as_dict(),
                "stop_policy": "original empty answer on exception; no finalization",
                "observation": "passive HTTP sends; no aggregate-usage duplication",
                "transport_lifecycle": "dedicated per-answer client with cached provider default timeout/headers"}

    def _legacy_answer(self, question, prompt, tracker, observer):
        # Copy the provider's installed defaults to a dedicated client. Never
        # patch the provider's shared cached client or change execution kwargs.
        defaults = cached_async_http_client(provider="openai")
        http = ObservedAsyncHttpClient(observer, timeout=defaults.timeout, headers=defaults.headers)
        provider = OpenAIProvider(base_url=os.environ.get("OPENAI_BASE_URL"),
                                  api_key=os.environ.get("OPENAI_API_KEY"), http_client=http)
        label_async_completions(provider.client)
        model = OpenAIChatModel(self._model, provider=provider)
        agent = Agent(deps_type=LegacyDeps, system_prompt=prompt)
        for tool in (retrieve, rerank, calculate):
            agent.tool(tool)
        deps = LegacyDeps(retriever=self._retriever, tracker=tracker)

        # Close the dedicated client on run_sync's loop. Model/tool/usage
        # settings remain omitted, just as in the original executor.
        from pydantic_ai._utils import get_event_loop
        try:
            result = agent.run_sync(question, model=model, deps=deps)
            return result.output
        finally:
            get_event_loop().run_until_complete(provider.client.close())

    def execute(self, question_id, question, gold_answer):
        tracker = TokenTracker(question_id, question, "")
        artifacts = self._artifact_dir(question_id)
        observer = CallObserver(tracker, artifacts / "execution.jsonl")
        tracker.tool_event_sink = observer.write
        construction_started = time.perf_counter()
        execution_started = None
        answer = ""
        try:
            prompt, _ = self._prepare_prompt(tracker, artifacts)
            execution_started = time.perf_counter()
            with observer.phase("answer_execution"):
                answer = self._legacy_answer(question, prompt, tracker, observer)
        except Exception as exc:
            tracker.set_error(f"{type(exc).__name__}: {exc}")
        log = tracker.to_question_log(answer)
        log.gold_answer = gold_answer
        log.resource_summary = {"execution": observer.summary(),
                                "construction_seconds_here": (execution_started or time.perf_counter()) - construction_started,
                                "answer_execution_seconds": time.perf_counter() - execution_started if execution_started else 0.0,
                                "tool_calls": log.num_tool_calls, "n_agents": 1 if execution_started else 0,
                                "usage_limits": vars(UsageLimits())}
        log.artifact_paths = {"execution_events": str(observer.journal)}
        self._attach_construction(log)
        observer.write({"kind": "question_end", "status": log.status,
                        "error": log.error, "resources": log.resource_summary})
        return answer, log
