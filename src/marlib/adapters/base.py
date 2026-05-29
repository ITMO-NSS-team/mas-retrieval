from __future__ import annotations

import importlib
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from marlib.log import logger
from marlib.tracing.schemas import QuestionLog
from marlib.retriever.core import Retriever


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
        self._generation_mode = kwargs.pop("generation_mode", None)
        self._config = kwargs

        # Benchmark context (set by runner before each benchmark)
        self._benchmark_name: str | None = None
        self._benchmark_description: str | None = None
        self._sample_questions: list[str] = []

    def set_benchmark_context(
        self,
        benchmark_name: str,
        description: str,
        sample_questions: list[str] | None = None,
    ) -> None:
        """Provide benchmark context for shared-mode system generation.

        Called by the runner before each benchmark. Resets adapter caches.
        """
        self._benchmark_name = benchmark_name
        self._benchmark_description = description
        self._sample_questions = sample_questions or []
        self._on_benchmark_change()

    def _on_benchmark_change(self) -> None:
        """Hook for subclasses to reset caches when benchmark changes."""
        pass

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


# --- Adapter registry ---------------------------------------------------------
# Adapters self-register via @register. The package __init__ imports each
# adapter module defensively, so the registry reflects only the systems whose
# (often heavy / optional) dependencies are actually installed — there is no
# central dict to edit or comment out.

_ADAPTERS: dict[str, type[AbstractAdapter]] = {}


def register(name: str):
    """Class decorator registering an :class:`AbstractAdapter` under ``name``."""

    def deco(cls: type[AbstractAdapter]) -> type[AbstractAdapter]:
        if name in _ADAPTERS:
            raise ValueError(f"Adapter '{name}' already registered")
        _ADAPTERS[name] = cls
        return cls

    return deco


def available_adapters() -> list[str]:
    """Names of adapters that registered successfully (import the package first)."""
    return sorted(_ADAPTERS)


def get_adapter_class(name: str) -> type[AbstractAdapter]:
    """Return the registered adapter class for ``name``, or raise with the list."""
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown system '{name}'. Available: {available_adapters()}"
        )
    return _ADAPTERS[name]


# --- Path-based discovery -----------------------------------------------------
# Adapters live OUTSIDE this package (they are experiment content, not harness).
# discover_adapters() imports each system package from an external directory via
# a synthetic namespace package, so the systems' relative imports resolve and
# their @register side effects fire. The synthetic prefix avoids clashing with
# the frameworks the adapters wrap (e.g. installed `fedotmas` / `automas`).

DEFAULT_SYSTEMS_ROOT = Path("experiments/systems")
_SYSTEMS_NS = "_marlib_systems"


def _ensure_namespace(parent: str, search_dir: Path) -> None:
    """Register/refresh a synthetic namespace package rooted at ``search_dir``."""
    mod = sys.modules.get(parent)
    if mod is None:
        mod = types.ModuleType(parent)
        sys.modules[parent] = mod
    mod.__path__ = [str(search_dir)]  # type: ignore[attr-defined]


def discover_adapters(root: str | Path = DEFAULT_SYSTEMS_ROOT) -> list[str]:
    """Import external adapter packages from ``root``, registering what loads.

    Each ``<root>/<name>/`` package is imported defensively: a system whose
    (heavy / optional) dependencies are missing is skipped, not crashed on.
    Idempotent. Returns the names available after discovery.
    """
    root = Path(root)
    if not root.is_dir():
        return available_adapters()
    _ensure_namespace(_SYSTEMS_NS, root)
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if not (entry / "__init__.py").exists():
            continue
        try:
            importlib.import_module(f"{_SYSTEMS_NS}.{entry.name}")
        except Exception as e:  # missing optional deps, import-time errors, etc.
            logger.debug(f"Adapter '{entry.name}' unavailable: {e!r}")
    return available_adapters()
