"""MAS-Feedback and Self-Verification for MAS-Zero.

Two zero-supervision verifier-model steps (no gold answer is used):

- ``mas_feedback`` evaluates a generated MAS on **solvability** and
  **completeness** from its intermediate outputs (sub-task answers, agent
  outputs) and returns a fitness in [0, 1] plus textual feedback that drives the
  reflexion loop. The deterministic ``[TOO_HARD]`` self-report caps fitness.

- ``self_verify`` is the final list-wise judge: given all candidate answers
  produced across the meta-iteration, it selects the best one.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import backoff
import openai

from .core import TOO_HARD_MARK

logger = logging.getLogger(__name__)

UsageCallback = Callable[[int, int], None] | None


@backoff.on_exception(
    backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=5
)
def _verifier_json(
    messages: list[dict],
    model: str,
    output_fields: list[str],
    usage_callback: UsageCallback = None,
) -> dict[str, Any]:
    """Call the verifier model and parse a JSON object with output_fields."""
    client = openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        if usage and usage_callback:
            usage_callback(usage.prompt_tokens, usage.completion_tokens)
        text = response.choices[0].message.content or ""
        try:
            data = json.loads(text)
            if all(k in data for k in output_fields):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _clamp01(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


_FEEDBACK_SYSTEM = (
    "You are a meticulous evaluator of multi-agent solutions. You never solve the "
    "task yourself; you judge whether the given decomposition and agent outputs "
    "are sound.\n\n"
    "Reply EXACTLY with the following JSON format.\n"
    '{"solvable": "true or false", "complete": "true or false", '
    '"fitness": "a number between 0 and 1", "feedback": "your detailed feedback"}\n'
    "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a "
    "well-formed JSON object!"
)


def mas_feedback(
    question: str,
    sub_tasks: str | None,
    agents: str | None,
    final_answer: str,
    *,
    model: str,
    usage_callback: UsageCallback = None,
) -> tuple[float, str]:
    """Self-assess a MAS run; return (fitness in [0,1], feedback text).

    No gold answer is used — this is MAS-Zero's meta-level self-supervision.
    """
    too_hard = bool(sub_tasks) and TOO_HARD_MARK in sub_tasks

    user = (
        f"# Original question\n{question}\n\n"
        f"# Sub-task outputs\n{sub_tasks or '(no decomposition recorded)'}\n\n"
        f"# Agent outputs\n{agents or '(none recorded)'}\n\n"
        f"# Final answer\n{final_answer}\n\n"
        "Evaluate the solution:\n"
        "1. Solvable: was each sub-task actually solved by its agent? A sub-task "
        f"answer containing {TOO_HARD_MARK} signals it was not solvable as posed.\n"
        "2. Complete: do the sub-tasks together cover all information the original "
        "question needs, with no critical fact missing from every sub-task?\n"
        "3. Fitness: your best estimate, in [0,1], that the final answer correctly "
        "and completely answers the original question. Be strict: reserve values "
        "near 1.0 for answers you are confident are correct and complete.\n"
        "In 'feedback', point out which sub-tasks/agents are weak and how to fix "
        "the decomposition or block wiring."
    )
    data = _verifier_json(
        [
            {"role": "system", "content": _FEEDBACK_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        output_fields=["solvable", "complete", "fitness", "feedback"],
        usage_callback=usage_callback,
    )
    if not data:
        logger.warning("mas_feedback: verifier returned no parseable JSON")
        return 0.0, "Verifier produced no feedback."

    fitness = _clamp01(data.get("fitness", 0.0))
    # A self-reported [TOO_HARD] sub-task means the design is not yet adequate.
    if too_hard:
        fitness = min(fitness, 0.5)
    feedback = str(data.get("feedback", ""))
    return fitness, feedback


_SELF_VERIFY_SYSTEM = (
    "You are a judge selecting the single best answer from a list of candidate "
    "answers. You do not solve the task yourself.\n\n"
    "Reply EXACTLY with the following JSON format.\n"
    '{"thinking": "your comparison", "selection": "the integer id of the best answer"}\n'
    "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a "
    "well-formed JSON object!"
)


def self_verify(
    question: str,
    candidates: list[dict],
    *,
    model: str,
    usage_callback: UsageCallback = None,
) -> int:
    """List-wise self-verification: pick the best candidate index.

    ``candidates`` is a list of dicts with at least an ``answer`` key (optionally
    ``thinking``/``response`` for context). Returns an index into ``candidates``.
    Falls back to the highest-fitness candidate (or 0) on any failure.
    """
    usable = [c for c in candidates if (c.get("answer") or "").strip()]
    if not usable:
        return _best_by_fitness(candidates)
    if len(usable) == 1:
        return candidates.index(usable[0])

    lines = []
    for c in usable:
        idx = candidates.index(c)
        thinking = (c.get("thinking") or "")[:1500]
        answer = (c.get("answer") or "")[:800]
        lines.append(f"Answer ID {idx}:\nReasoning: {thinking}\nFinal answer: {answer}")
    answer_list = "\n\n".join(lines)

    user = (
        "Given the problem and a list of candidate reasoning steps and final "
        "answers, do not solve the task yourself. Compare the candidates, identify "
        "erroneous reasoning in the weaker ones, and select the best final answer. "
        "In 'thinking', justify your choice against the others. In 'selection', "
        "return ONLY the integer answer id.\n\n"
        f"Problem:\n{question}\n\nAnswer list:\n{answer_list}"
    )
    data = _verifier_json(
        [
            {"role": "system", "content": _SELF_VERIFY_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        output_fields=["thinking", "selection"],
        usage_callback=usage_callback,
    )
    selection = data.get("selection")
    try:
        idx = int(str(selection).strip())
    except (TypeError, ValueError):
        logger.warning("self_verify: unparseable selection %r", selection)
        return _best_by_fitness(candidates)
    if 0 <= idx < len(candidates):
        return idx
    return _best_by_fitness(candidates)


def _best_by_fitness(candidates: list[dict]) -> int:
    if not candidates:
        return 0
    return max(range(len(candidates)), key=lambda i: candidates[i].get("fitness", 0.0))
