from __future__ import annotations

import os
from functools import cache
from typing import Any

from pydantic import BaseModel

from marlib.evaluation.base import EvalContext, register_metric

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating whether a predicted answer is \
semantically equivalent to the gold (reference) answer for a given question.

Rules:
- Focus on meaning, not surface form. Minor wording differences are acceptable.
- Numbers must match in value (e.g. "$1.2 billion" == "1,200 million dollars").
- If the predicted answer contains the correct information but also extra \
  irrelevant detail, still mark it correct.
- If the predicted answer is vague, incomplete, or contradicts the gold answer, \
  mark it incorrect.

Respond with a JSON object containing:
- "correct": true/false
- "reasoning": a brief explanation of your verdict
"""


class JudgeVerdict(BaseModel):
    correct: bool
    reasoning: str


@cache
def _judge_agent() -> Any:
    # pydantic-ai is imported lazily so building the metric registry stays cheap
    # for benchmarks that never use the judge.
    from pydantic_ai import Agent

    return Agent(system_prompt=_JUDGE_SYSTEM_PROMPT, output_type=JudgeVerdict)


def llm_accuracy(
    question: str,
    predicted: str,
    gold: str,
    model_name: str,
) -> float:
    """Return 1.0 if the judge deems the predicted answer correct, else 0.0.

    Args:
        question: The original question.
        predicted: The system's predicted answer.
        gold: The ground-truth (gold) answer.
        model_name: Model identifier for the judge LLM.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    prompt = (
        f"Question: {question}\n\nGold answer: {gold}\n\nPredicted answer: {predicted}"
    )
    provider = OpenAIProvider(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    model = OpenAIChatModel(model_name, provider=provider)

    result = _judge_agent().run_sync(prompt, model=model)
    return float(result.output.correct)


@register_metric("llm_accuracy")
def _llm_accuracy(ctx: EvalContext) -> float:
    return llm_accuracy(
        question=ctx.question,
        predicted=ctx.predicted,
        gold=ctx.gold,
        model_name=ctx.model,
    )
