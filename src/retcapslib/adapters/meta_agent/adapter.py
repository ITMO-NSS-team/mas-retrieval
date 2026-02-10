"""MetaAgent FSM-based multi-agent adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.adapters.meta_agent.fsm_gen import generate_mas
from retcapslib.adapters.meta_agent.multi_agent import MultiAgentSystem
from retcapslib.adapters.tools import do_calculate, do_rerank, do_retrieve
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Document, Retriever

logger = logging.getLogger(__name__)

_TASK_DESCRIPTION = (
    "Answer questions by retrieving relevant documents, optionally reranking "
    "them for precision, and computing numerical answers when needed. "
    "Available tools: retrieve (semantic search), rerank (cross-encoder "
    "reranking of retrieved documents), calculate (safe math evaluation)."
)


class MetaAgentAdapter(AbstractAdapter):
    """MetaAgent FSM-based multi-agent adapter.

    Generates an agent team + FSM once via LLM (lazy init),
    then reuses them for every question.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "shared"
        self._agents_json: list[dict] | None = None
        self._states_json: dict | None = None
        self._initialized = False

    def _init_team(self) -> None:
        """Generate agent team + FSM once (lazy)."""
        if self._initialized:
            return

        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")

        logger.info("Generating MetaAgent team for model=%s", self._model)
        agents_json, states_json = generate_mas(
            task=_TASK_DESCRIPTION,
            model=self._model,
            base_url=base_url,
            api_key=api_key,
        )
        self._agents_json = agents_json
        self._states_json = states_json
        self._initialized = True
        logger.info(
            "MetaAgent team ready: %d agents, %d states",
            len(agents_json),
            len(states_json["states"]),
        )

    @property
    def name(self) -> str:
        return f"meta_agent_{self._generation_mode}"

    def generate_system(self, question: str) -> str:
        self._init_team()
        assert self._agents_json is not None
        assert self._states_json is not None
        return (
            f"meta_agent team: {len(self._agents_json)} agents, "
            f"{len(self._states_json['states'])} states"
        )

    def _make_tool_executor(
        self, tracker: TokenTracker,
    ) -> Any:
        """Build a closure that dispatches tool calls to shared tools."""
        last_retrieved: list[Document] = []
        retriever = self._retriever

        def executor(name: str, **kwargs: Any) -> str:
            nonlocal last_retrieved

            if name == "retrieve":
                query = kwargs.get("query", "")
                top_k = int(kwargs.get("top_k", 20))
                with tracker.track_tool("retrieve", query, top_k) as results:
                    docs, formatted = do_retrieve(retriever, query, top_k)
                    last_retrieved = docs
                    results.extend([d.doc_id for d in docs])
                return formatted

            if name == "rerank":
                query = kwargs.get("query", "")
                top_k = int(kwargs.get("top_k", 10))
                with tracker.track_tool("rerank", query, top_k) as results:
                    docs, formatted = do_rerank(
                        retriever, query, last_retrieved, top_k,
                    )
                    last_retrieved = docs
                    results.extend([d.doc_id for d in docs])
                return formatted

            if name == "calculate":
                expression = kwargs.get("expression", "")
                with tracker.track_tool("calculate", expression, 0) as results:
                    result = do_calculate(expression)
                return result

            return f"Unknown tool: {name}"

        return executor

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        if self._generation_mode == "per_question":
            self._initialized = False
        self._init_team()
        assert self._agents_json is not None
        assert self._states_json is not None

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")

            tool_executor = self._make_tool_executor(tracker)

            mas = MultiAgentSystem(
                agents_json=self._agents_json,
                states_json=self._states_json,
                tool_executor=tool_executor,
                model=self._model,
                base_url=base_url,
                api_key=api_key,
                tracker=tracker,
            )

            answer, _cost = mas.start(question)

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
