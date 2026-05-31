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


# The judge model is kept fixed (independent of the system-under-test backbone)
# so accuracy is scored consistently across runs and there is no self-judging by
# whatever model a system happens to use. Override via the JUDGE_MODEL env var.
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
_JUDGE_MODEL_ENV = "JUDGE_MODEL"


def judge_model() -> str:
    """Model used by the LLM judge: ``$JUDGE_MODEL`` or the fixed default."""
    return os.environ.get(_JUDGE_MODEL_ENV) or DEFAULT_JUDGE_MODEL


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
    usage_sink: dict[str, int] | None = None,
) -> float:
    """1.0 if the judge deems the predicted answer correct, else 0.0.

    If ``usage_sink`` is given, the judge call's token usage is accumulated into
    it under the ``"prompt"`` / ``"completion"`` keys (so the caller can cost the
    judge separately from the system under test).
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
    if usage_sink is not None:
        usage = result.usage()
        usage_sink["prompt"] = usage_sink.get("prompt", 0) + (usage.input_tokens or 0)
        usage_sink["completion"] = usage_sink.get("completion", 0) + (
            usage.output_tokens or 0
        )
    return float(result.output.correct)


@register_metric("llm_accuracy")
def _llm_accuracy(ctx: EvalContext) -> float:
    # Judge model is fixed (env-overridable), not the system-under-test backbone.
    return llm_accuracy(
        question=ctx.question,
        predicted=ctx.predicted,
        gold=ctx.gold,
        model_name=judge_model(),
        usage_sink=ctx.judge_usage,
    )
