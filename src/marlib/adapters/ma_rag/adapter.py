"""MA-RAG adapter: plan-execute-summarize multi-agent pipeline."""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from marlib.adapters.base import AbstractAdapter
from marlib.adapters.ma_rag.nodes import CreatePlan, ma_rag_graph
from marlib.adapters.ma_rag.state import MARagDeps, MARagState
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker
from marlib.retriever.core import Retriever


class MARagAdapter(AbstractAdapter):
    """Multi-agent RAG with plan → execute → summarize graph (MA-RAG port)."""

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)

    @property
    def name(self) -> str:
        return "ma_rag"

    def generate_system(self, question: str) -> str:
        return "static: plan-execute-summarize graph (MA-RAG)"

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
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            provider = OpenAIProvider(base_url=base_url, api_key=api_key)
            model = OpenAIChatModel(self._model, provider=provider)

            deps = MARagDeps(
                retriever=self._retriever,
                tracker=tracker,
                model=model,
                model_name=self._model,
            )
            state = MARagState(question=question)

            result = ma_rag_graph.run_sync(
                CreatePlan(), state=state, deps=deps,
            )
            answer = result.output

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
