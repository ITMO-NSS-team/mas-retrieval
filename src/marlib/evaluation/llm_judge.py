from __future__ import annotations

import os

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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


judge_agent = Agent(
    system_prompt=_JUDGE_SYSTEM_PROMPT,
    output_type=JudgeVerdict,
)


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
    prompt = (
        f"Question: {question}\n\nGold answer: {gold}\n\nPredicted answer: {predicted}"
    )

    provider = OpenAIProvider(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    model = OpenAIChatModel("openai/gpt-4o-mini", provider=provider)

    result = judge_agent.run_sync(prompt, model=model)
    return float(result.output.correct)
