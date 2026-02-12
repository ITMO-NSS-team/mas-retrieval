"""State and dependency dataclasses for the MA-RAG graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.models.openai import OpenAIChatModel

from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Document, Retriever


@dataclass
class StepResult:
    step_description: str
    task_type: str  # "qa" or "aggregate"
    query: str
    answer: str
    success: bool
    confidence: int  # 0-10
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class MARagState:
    question: str = ""
    plan: list[str] = field(default_factory=list)
    current_step_index: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    final_answer: str = ""


@dataclass
class MARagDeps:
    retriever: Retriever
    tracker: TokenTracker
    model: OpenAIChatModel
    model_name: str
    top_k_retrieve: int = 20
    top_k_rerank: int = 10
    max_steps: int = 8
    _last_retrieved: list[Document] = field(default_factory=list)
