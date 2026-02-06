"""Abstract base class for MAS auto-generator adapters.

Each adapter wraps a specific system (e.g., ADAS, MAS-GPT, AutoAgents)
and provides a uniform interface for question execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from retcapslib.logging.schemas import QuestionLog
from retcapslib.retriever.core import Retriever


class AbstractAdapter(ABC):
    """Base class for MAS auto-generator system adapters.

    Subclasses must implement:
    - generate_system(): Create MAS code/config for a question
    - execute(): Run the generated system and return answer + log
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            retriever: Initialized Retriever instance for search.
            model: LLM model to use for the system.
            **kwargs: Additional system-specific configuration.
        """
        self._retriever = retriever
        self._model = model
        self._config = kwargs

    @property
    def name(self) -> str:
        """Return the system name for logging."""
        return self.__class__.__name__.replace("Adapter", "").lower()

    @abstractmethod
    def generate_system(self, question: str) -> str:
        """Generate MAS code/config for the question.

        This is the "meta" step where the auto-generator creates
        a system specifically for the given question.

        Args:
            question: The question to answer.

        Returns:
            Generated system code, configuration, or description.
            The format depends on the specific auto-generator.
        """

    @abstractmethod
    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        """Execute the system on a question and return results.

        This runs the full pipeline:
        1. Generate system (if applicable)
        2. Execute generated system
        3. Track all tool/LLM calls
        4. Return answer and execution log

        Args:
            question_id: Unique identifier for the question.
            question: The question to answer.
            gold_answer: Ground truth answer (for logging).

        Returns:
            Tuple of (predicted_answer, QuestionLog).
        """
