"""Single agent baseline adapter using pydantic-ai.

Iterative tool-calling agent that can make multiple retrieval calls
to gather evidence before answering.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
if TYPE_CHECKING:
    from marlib.retriever.core import Document, Retriever
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession, tracked_async_client, run_with_deadline
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker


@dataclass
class SingleAgentDeps:
    """Dependencies injected into the pydantic-ai agent tools."""

    retriever: Retriever
    tracker: TokenTracker
    session: ResourceSession
    _last_retrieved: list[Document] = field(default_factory=list)


DEFAULT_SYSTEM_PROMPT = (
        "You are a research assistant that answers questions using a document "
        "knowledge base and a calculator.\n\n"
        "Tools:\n"
        "- retrieve(query, top_k): Dense search for candidate passages. Use this first.\n"
        "- rerank(query, top_k): Re-score the most recent retrieve() results with a "
        "cross-encoder for better ranking. Always call after retrieve().\n"
        "- calculate(expression): Evaluate a math expression "
        '(e.g. "1234.5 * 0.15", "round(456.78 / 123, 2)").\n\n'
        "Workflow:\n"
        "1. Break complex questions into sub-queries\n"
        "2. For each: retrieve(query) -> rerank(query) to get best passages\n"
        "3. Use calculate() for numerical computations\n"
        "4. Synthesize evidence into a concise final answer\n\n"
        "Note: rerank() operates on results from your most recent retrieve() call."
)


def retrieve(ctx: RunContext[SingleAgentDeps], query: str, top_k: int = 20) -> str:
    """Retrieve candidate passages from the knowledge base via dense search."""
    ctx.deps.session.tool()
    with ctx.deps.tracker.track_tool("retrieve", query, top_k) as doc_ids:
        docs, formatted = do_retrieve(ctx.deps.retriever, query, top_k)
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


def rerank(ctx: RunContext[SingleAgentDeps], query: str, top_k: int = 10) -> str:
    """Re-rank recently retrieved passages using a cross-encoder model."""
    ctx.deps.session.tool()
    with ctx.deps.tracker.track_tool("rerank", query, top_k) as doc_ids:
        if not ctx.deps._last_retrieved:
            return "Error: No documents to rerank. Call retrieve() first."
        docs, formatted = do_rerank(
            ctx.deps.retriever, query, ctx.deps._last_retrieved, top_k
        )
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


def calculate(ctx: RunContext[SingleAgentDeps], expression: str) -> str:
    """Evaluate a mathematical expression (e.g. '1234.5 * 0.15', 'round(456.78 / 123, 2)')."""
    ctx.deps.session.tool()
    with ctx.deps.tracker.track_tool("calculate", expression, 0) as _doc_ids:
        result = do_calculate(expression)
    return result


@register("single_agent")
class SingleAgentAdapter(AbstractAdapter):
    """Single iterative agent with search tool, powered by pydantic-ai."""

    supports_resource_limits = True

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        self._limits = ResourceLimits(**{"scope": "execution", "max_requests": 50,
                                       **self._config.get("resource_limits", {})})
        if self._limits.scope != "execution":
            raise ValueError("Single-agent adapters require execution resource limits")
        self._temperature = self._config.get("temperature", 0.1)
        if not isinstance(self._temperature, (int, float)) or not 0 <= self._temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def effective_config(self) -> dict[str, Any]:
        return {**super().effective_config(), "variant": "single_agent_accounted_v1",
                "resource_limits": asdict(self._limits), "temperature": self._temperature,
                "output_per_request": self._limits.output_per_request,
                "validation_retries": 1, "tool_execution": "sequential",
                "stop_policy": "empty_answer_without_finalization",
                "system_prompt": DEFAULT_SYSTEM_PROMPT}

    def _artifact_dir(self, question_id: str) -> Path:
        root = Path(self._run_context.get("artifact_dir", f"logs/{self.name}"))
        path = root / f"{hashlib.sha256(question_id.encode()).hexdigest()[:12]}_{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _prepare_prompt(self, tracker: TokenTracker, artifacts: Path) -> tuple[str, dict]:
        return DEFAULT_SYSTEM_PROMPT, {}

    async def _answer(self, question: str, prompt: str, tracker: TokenTracker, session: ResourceSession) -> str:
        client = tracked_async_client(session, "answer_execution",
                                      base_url=os.environ.get("OPENAI_BASE_URL"),
                                      api_key=os.environ.get("OPENAI_API_KEY"))
        async with client:
            model = OpenAIChatModel(self._model, provider=OpenAIProvider(openai_client=client))
            agent = Agent(model=model, deps_type=SingleAgentDeps, system_prompt=prompt,
                          retries=1, tools=[Tool(fn, sequential=True) for fn in (retrieve, rerank, calculate)])
            deps = SingleAgentDeps(retriever=self._retriever, tracker=tracker, session=session)
            # Our limits apply to API attempts; disable Pydantic AI's hidden
            # default cap of 50 model requests so effective_config is complete.
            result = await agent.run(question, deps=deps,
                                     model_settings={"temperature": self._temperature},
                                     usage_limits=UsageLimits(request_limit=None))
            return result.output

    @property
    def name(self) -> str:
        return "single_agent"

    def generate_system(self, question: str) -> str:
        return "static: iterative search agent"

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer="",
        )
        artifacts = self._artifact_dir(question_id)
        session = None
        prompt_metadata = {}
        failure_kind = None
        try:
            prompt, prompt_metadata = self._prepare_prompt(tracker, artifacts)
            session = ResourceSession(tracker, self._limits, artifacts / "execution.jsonl",
                                      coverage=self._execution_coverage())
            tracker.tool_event_sink = session.write
            answer = asyncio.run(run_with_deadline(session, self._answer(question, prompt, tracker, session)))
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Agent returned an empty answer")
        except BudgetExhausted as e:
            tracker.set_error(str(e))
            failure_kind = "budget_exhausted"
            answer = ""
        except Exception as e:
            tracker.set_error(f"{type(e).__name__}: {e}")
            failure_kind = self._failure_kind(e)
            answer = ""
        log = tracker.to_question_log(answer)
        log.gold_answer = gold_answer
        log.failure_kind = failure_kind
        if failure_kind == "budget_exhausted":
            log.status = "budget_exhausted"
        log.resource_summary = {"execution": session.summary() if session else None,
                                **prompt_metadata}
        if session:
            log.artifact_paths["execution_events"] = str(session.journal)
            session.write({"kind": "question_end", "status": log.status})
        self._attach_construction(log)
        (artifacts / "question.json").write_text(log.model_dump_json(indent=2))
        return answer, log

    def _attach_construction(self, log: QuestionLog) -> None:
        """Hook for a generated prompt, including unsuccessful construction."""

    def _failure_kind(self, error: Exception) -> str:
        return "unknown"

    def _execution_coverage(self):
        return "single_agent_chat_completions_and_local_tools"
