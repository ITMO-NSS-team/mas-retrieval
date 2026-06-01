"""Tracing models for MAS-Zero executions.

Captures the per-question meta-iteration: each candidate architecture, its
intermediate outputs, MAS-Feedback fitness, and the final self-verification
choice. Activated via config ``trace: true`` or env var ``MAS_ZERO_TRACE=1``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateTrace(BaseModel):
    """One evaluated architecture (initial block or proposed generation)."""

    stage: str  # "initial" or "generation"
    generation: int = -1  # -1 for initial blocks
    name: str = ""
    thought: str = ""
    code: str = ""
    answer: str = ""
    fitness: float = 0.0
    feedback: str = ""
    sub_tasks: str | None = None
    agents: str | None = None
    error: str | None = None


class MASZeroTrace(BaseModel):
    """Full trace of one MAS-Zero question execution."""

    question_id: str
    meta_model: str = ""
    node_model: str = ""
    verifier_model: str = ""
    blocks_offered: list[str] = Field(default_factory=list)
    candidates: list[CandidateTrace] = Field(default_factory=list)
    selected_index: int = -1
    selected_answer: str = ""
    n_generation: int = 0
    stopped_early: bool = False
    execution_error: str | None = None

    def summary(self) -> str:
        lines = [
            f"=== MAS-Zero Trace: {self.question_id} ===",
            f"Models: meta={self.meta_model}, node={self.node_model}, "
            f"verifier={self.verifier_model}",
            f"Candidates ({len(self.candidates)}):",
        ]
        for i, c in enumerate(self.candidates):
            marker = " *" if i == self.selected_index else "  "
            tag = c.stage if c.generation < 0 else f"gen{c.generation}"
            lines.append(
                f"{marker}[{i}] {tag} {c.name} fitness={c.fitness:.2f} "
                f"answer={c.answer[:60]!r}"
            )
        lines.append(f"Selected: #{self.selected_index} -> {self.selected_answer[:80]!r}")
        if self.execution_error:
            lines.append(f"ERROR: {self.execution_error}")
        return "\n".join(lines)
