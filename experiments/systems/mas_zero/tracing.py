"""Tracing models for MAS-Zero generated architectures.

Captures the full lifecycle of a MAS-Zero execution: meta-model generation,
architecture structure, agent interactions, and execution results. Activated
via config ``trace: true`` or environment variable ``MAS_ZERO_TRACE=1``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    """Record of a single LLMAgentBase.query() invocation."""

    agent_name: str
    agent_id: str
    output_fields: list[str]
    role: str
    iteration_idx: int = -1
    input_summary: str = ""
    output: dict[str, str] = Field(default_factory=dict)


class MASZeroTrace(BaseModel):
    """Full trace of one MAS-Zero question execution."""

    question_id: str
    mode: str
    architecture_name: str = ""
    architecture_thought: str = ""
    generated_code: str = ""
    meta_model: str = ""
    node_model: str = ""
    blocks_offered: list[str] = Field(default_factory=list)
    agent_calls: list[AgentTrace] = Field(default_factory=list)
    forward_bound_to: str = ""
    execution_error: str | None = None

    def summary(self) -> str:
        lines = [
            f"=== MAS-Zero Trace: {self.question_id} ===",
            f"Mode: {self.mode} | Arch: {self.architecture_name}",
            f"Models: meta={self.meta_model}, node={self.node_model}",
            f"Blocks: {', '.join(self.blocks_offered)}",
            f"Agent calls ({len(self.agent_calls)}):",
        ]
        for i, ac in enumerate(self.agent_calls):
            lines.append(
                f"  [{i}] {ac.agent_name} ({ac.role}) -> {list(ac.output.keys())}"
            )
        if self.execution_error:
            lines.append(f"ERROR: {self.execution_error}")
        return "\n".join(lines)
