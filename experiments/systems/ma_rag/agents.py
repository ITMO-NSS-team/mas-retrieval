"""Pydantic-ai agents and structured output models for MA-RAG."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .prompts import (
    ANSWER_QA_SYSTEM_PROMPT,
    EXTRACT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    STEP_DEFINER_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

# ── Output models ────────────────────────────────────────────


class PlanOutput(BaseModel):
    analysis: str = Field(description="Your analysis. Think step-by-step")
    steps: list[str] = Field(
        description="Different steps to follow, should be in sorted order"
    )


class StepTask(BaseModel):
    type: str = Field(
        description="Type of task, one of [aggregate, question-answering]"
    )
    task: str = Field(description="The detail task to do in this step")


class QAAnswer(BaseModel):
    analysis: str = Field(
        description="Your thoughts, analysis about the question and the context. Think step-by-step"
    )
    answer: str = Field(description="The answer for the question")
    success: str = Field(
        description="binary output (Yes or No), indicate you can answer or not"
    )
    rating: int = Field(
        default=5,
        description="How confident, from 0 to 10. More evidence, more agreement, more confident",
    )


class SummaryOutput(BaseModel):
    output: str = Field(description="Your output, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confident score")


# ── Agents (model deferred — passed at call time) ───────────


plan_agent: Agent[None, PlanOutput] = Agent(
    system_prompt=PLAN_SYSTEM_PROMPT,
    output_type=PlanOutput,
    defer_model_check=True,
)

step_definer_agent: Agent[None, StepTask] = Agent(
    system_prompt=STEP_DEFINER_SYSTEM_PROMPT,
    output_type=StepTask,
    defer_model_check=True,
)

extract_agent: Agent[None, str] = Agent(
    system_prompt=EXTRACT_SYSTEM_PROMPT,
    output_type=str,
    defer_model_check=True,
)

answer_agent: Agent[None, QAAnswer] = Agent(
    system_prompt=ANSWER_QA_SYSTEM_PROMPT,
    output_type=QAAnswer,
    defer_model_check=True,
)

summary_agent: Agent[None, SummaryOutput] = Agent(
    system_prompt=SUMMARY_SYSTEM_PROMPT,
    output_type=SummaryOutput,
    defer_model_check=True,
)
