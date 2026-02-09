"""SwarmAgentic multi-agent team adapter (zero-shot, no PSO)."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.adapters.swarm_agentic.func import get_forward, set_forward
from retcapslib.adapters.swarm_agentic.logger import setup_logger
from retcapslib.adapters.swarm_agentic.role import Team
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Retriever


class SwarmAgenticAdapter(AbstractAdapter):
    """SwarmAgentic multi-agent team adapter (zero-shot, no PSO).

    Generates a team of roles + forward function once via LLM,
    then reuses them for every question.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        self._team_dict: dict[str, Any] | None = None
        self._forward_code: str | None = None
        self._initialized = False

    def _make_llm(self) -> ChatOpenAI:
        """Create a ChatOpenAI instance with env-based config."""
        return ChatOpenAI(
            model=self._model,
            temperature=0.001,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _init_team(self) -> None:
        """Generate team + forward code once via LLM (lazy)."""
        if self._initialized:
            return

        llm_init = self._make_llm()
        logger = setup_logger("init")

        # 1. Generate team (roles + workflow) from task description
        team = Team(
            llm=llm_init, logger=logger,
            retriever=self._retriever, tracker=None,
        )
        team.init(llm=llm_init)
        team.inject_tool_roles()

        # 2. Generate forward code
        self._forward_code = get_forward(
            llm_init, logger, team.to_str(), team.workflow,
        )
        self._team_dict = team.save_into_dict()
        self._initialized = True

    @property
    def name(self) -> str:
        return "swarm_agentic"

    def generate_system(self, question: str) -> str:
        self._init_team()
        assert self._team_dict is not None
        return f"swarm team: {len(self._team_dict['roles'])} roles"

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        self._init_team()
        assert self._team_dict is not None
        assert self._forward_code is not None

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            llm = self._make_llm()

            team = Team(
                llm=llm, logger=None,
                retriever=self._retriever, tracker=tracker,
            )
            team.update(self._team_dict)
            team.inject_tool_roles()

            func = set_forward(self._forward_code)
            team.reset_task(question)
            answer = func(team)

            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                function_calls=0,
            )

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
