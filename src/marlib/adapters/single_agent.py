"""Single agent baseline adapter using pydantic-ai.

Iterative tool-calling agent that can make multiple retrieval calls
to gather evidence before answering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker
from marlib.retriever.core import Document, Retriever


@dataclass
class SingleAgentDeps:
    """Dependencies injected into the pydantic-ai agent tools."""

    retriever: Retriever
    tracker: TokenTracker
    _last_retrieved: list[Document] = field(default_factory=list)


agent = Agent(
    deps_type=SingleAgentDeps,
    system_prompt=(
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
    ),
)


@agent.tool
def retrieve(ctx: RunContext[SingleAgentDeps], query: str, top_k: int = 20) -> str:
    """Retrieve candidate passages from the knowledge base via dense search."""
    with ctx.deps.tracker.track_tool("retrieve", query, top_k) as doc_ids:
        docs, formatted = do_retrieve(ctx.deps.retriever, query, top_k)
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


@agent.tool
def rerank(ctx: RunContext[SingleAgentDeps], query: str, top_k: int = 10) -> str:
    """Re-rank recently retrieved passages using a cross-encoder model."""
    if not ctx.deps._last_retrieved:
        return "Error: No documents to rerank. Call retrieve() first."
    with ctx.deps.tracker.track_tool("rerank", query, top_k) as doc_ids:
        docs, formatted = do_rerank(
            ctx.deps.retriever, query, ctx.deps._last_retrieved, top_k
        )
        ctx.deps._last_retrieved = docs
        doc_ids.extend([doc.doc_id for doc in docs])
    return formatted


@agent.tool
def calculate(ctx: RunContext[SingleAgentDeps], expression: str) -> str:
    """Evaluate a mathematical expression (e.g. '1234.5 * 0.15', 'round(456.78 / 123, 2)')."""
    with ctx.deps.tracker.track_tool("calculate", expression, 0) as _doc_ids:
        result = do_calculate(expression)
    return result


@register("single_agent")
class SingleAgentAdapter(AbstractAdapter):
    """Single iterative agent with search tool, powered by pydantic-ai."""

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)

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
            gold_answer=gold_answer,
        )

        try:
            deps = SingleAgentDeps(retriever=self._retriever, tracker=tracker)

            # Build model instance - strip "openai/" prefix if present
            model_name = self._model

            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")

            provider = OpenAIProvider(base_url=base_url, api_key=api_key)
            model = OpenAIChatModel(model_name, provider=provider)

            result = agent.run_sync(question, model=model, deps=deps)

            answer = result.output

            # Log aggregate LLM usage
            usage = result.usage()
            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
                latency_ms=0,  # latency captured by tracker._start_time
                function_calls=usage.tool_calls or 0,
            )

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
