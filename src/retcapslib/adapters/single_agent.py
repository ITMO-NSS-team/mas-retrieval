"""Single agent baseline adapter using pydantic-ai.

Iterative tool-calling agent that can make multiple retrieval calls
to gather evidence before answering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Retriever


@dataclass
class SingleAgentDeps:
    """Dependencies injected into the pydantic-ai agent tools."""

    retriever: Retriever
    tracker: TokenTracker


agent = Agent(
    deps_type=SingleAgentDeps,
    system_prompt=(
        "You are a helpful assistant that answers questions by searching for relevant information.\n\n"
        "You have access to a search tool that retrieves passages from a knowledge base.\n"
        "For complex questions that require multiple pieces of information, you should:\n"
        "1. Break down the question into sub-queries\n"
        "2. Search for each piece of information\n"
        "3. Combine the evidence to answer the question\n\n"
        "When you have gathered enough information, provide your final answer.\n"
        "Be concise and direct in your final answer."
    ),
)


@agent.tool
def search(ctx: RunContext[SingleAgentDeps], query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant passages.

    Args:
        query: The search query to find relevant passages.
        top_k: Number of passages to retrieve (default: 5).
    """
    with ctx.deps.tracker.track_tool("search", query, top_k) as doc_ids:
        docs = ctx.deps.retriever.search(query, top_k=top_k)
        doc_ids.extend([doc.doc_id for doc in docs])

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[{i}] {doc.title}: {doc.text}")

    return "\n\n".join(results) if results else "No results found."


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

            # Build model instance — strip "openai/" prefix if present
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
