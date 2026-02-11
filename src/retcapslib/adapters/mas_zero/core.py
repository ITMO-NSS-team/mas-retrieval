"""Self-contained MAS-Zero runtime.

Minimal extraction from MAS-Zero's search.py and common.py, adapted to:
- Use OpenAI client directly (no global model_sampler_map)
- Support usage callbacks for token tracking
- Provide retrieve/rerank/calculate methods on AgentSystem
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import namedtuple
from typing import Any, Callable

import logging

import backoff
import openai

logger = logging.getLogger(__name__)

Info = namedtuple(
    "Info",
    ["name", "author", "content", "prompt", "sub_tasks", "agents", "iteration_idx"],
)

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def _pack_message(role: str, content: Any) -> dict[str, Any]:
    return {"role": str(role), "content": content}


class LLMAgentBase:
    """Simplified LLM agent that calls OpenAI directly.

    Differences from MAS-Zero original:
    - No global model_sampler_map; uses openai.OpenAI directly
    - Constructor takes model/temperature explicitly
    - Optional usage_callback(prompt_tokens, completion_tokens) for tracking
    - JSON response format only
    """

    def __init__(
        self,
        output_fields: list[str],
        agent_name: str,
        role: str = "helpful assistant",
        model: str | None = None,
        temperature: float | None = None,
        usage_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.usage_callback = usage_callback
        self.id = uuid.uuid4().hex[:8]

    def generate_prompt(
        self,
        input_infos: list,
        instruction: str,
        is_sub_task: bool = False,
    ) -> tuple[str, str]:
        output_fields_and_description = {
            key: (
                f"Your {key}."
                if "answer" not in key
                else f"Your {key}. Provide a concise, direct answer."
            )
            for key in self.output_fields
        }

        format_inst = (
            "Reply EXACTLY with the following JSON format.\n"
            + json.dumps(output_fields_and_description)
            + "\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response "
            "is a well-formed JSON object!"
        )

        system_prompt = f"You are a {self.role}.\n\n{format_inst}"

        input_infos_text = ""
        for input_info in input_infos:
            if not isinstance(input_info, Info):
                continue
            field_name, author, content, _prompt, _, _, iteration_idx = input_info
            if author == repr(self):
                author += " (yourself)"
            if field_name == "task":
                input_infos_text += f"{content}\n\n"
            elif iteration_idx != -1:
                input_infos_text += (
                    f"### {field_name} #{iteration_idx + 1} by {author}:\n"
                    f"{content}\n\n"
                )
            else:
                input_infos_text += (
                    f"### {field_name} by {author}:\n{content}\n\n"
                )

        if is_sub_task:
            prompt = (
                input_infos_text
                + f"Given the above, answer the following question: {instruction}\n\n"
                "If the question is too complicated or information is missing, "
                "you still need to give your best answer."
            )
        else:
            prompt = input_infos_text + instruction

        return system_prompt, prompt

    def query(
        self,
        input_infos: list,
        instruction: str,
        iteration_idx: int = -1,
        is_sub_task: bool = False,
    ) -> list[Info]:
        system_prompt, prompt = self.generate_prompt(
            input_infos, instruction, is_sub_task=is_sub_task,
        )

        messages = [
            _pack_message(role="system", content=system_prompt),
            _pack_message(role="user", content=prompt),
        ]

        response_json = _get_json_response(
            messages,
            model=self.model,
            output_fields=self.output_fields,
            temperature=self.temperature,
            usage_callback=self.usage_callback,
        )

        output_infos = []
        for key in self.output_fields:
            value = response_json.get(key, "")
            info = Info(key, repr(self), value, messages, None, None, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self) -> str:
        return f"{self.agent_name} {self.id}"

    def __call__(
        self,
        input_infos: list,
        instruction: str,
        iteration_idx: int = -1,
        is_sub_task: bool = False,
    ) -> list[Info]:
        return self.query(
            input_infos, instruction,
            iteration_idx=iteration_idx, is_sub_task=is_sub_task,
        )


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=5)
def _get_json_response(
    messages: list[dict],
    model: str | None,
    output_fields: list[str],
    temperature: float | None,
    usage_callback: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """Call OpenAI and parse a JSON response containing output_fields."""
    client = openai.OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    kwargs: dict[str, Any] = {"model": model or "gpt-4o-mini", "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["response_format"] = {"type": "json_object"}

    for _ in range(5):
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        if usage and usage_callback:
            usage_callback(usage.prompt_tokens, usage.completion_tokens)

        text = response.choices[0].message.content or ""
        try:
            json_dict = json.loads(text)
            if set(json_dict.keys()) >= set(output_fields):
                return {k: json_dict[k] for k in output_fields}
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: return empty fields
    logger.warning(
        "LLM failed to produce valid JSON with fields %s after 5 attempts (model=%s)",
        output_fields, model,
    )
    return {k: "" for k in output_fields}


class AgentSystem:
    """Hosts the dynamically generated forward() function.

    Attributes set by the adapter before execution:
        node_model, cot_instruction, max_round, max_sc, debate_role
        retrieve_fn, rerank_fn, calc_fn  (tool closures)
    """

    def __init__(self) -> None:
        self.node_model: str = "gpt-4o-mini"
        self.cot_instruction: str = ""
        self.max_round: int = 2
        self.max_sc: int = 3
        self.debate_role: list[str] = []
        self._retrieve_fn: Callable | None = None
        self._rerank_fn: Callable | None = None
        self._calc_fn: Callable | None = None

    def make_final_answer(
        self,
        thinking: Info,
        answer: Info | str,
        sub_tasks: list | None = None,
        agents: list | None = None,
    ) -> Info:
        name = thinking.name
        author = thinking.author
        prompt = thinking.prompt
        iteration_idx = thinking.iteration_idx

        answer_content = answer if isinstance(answer, str) else answer.content

        if agents is None and sub_tasks is not None:
            agents = sub_tasks
            sub_tasks = None

        if sub_tasks is None and agents is None:
            final = Info(
                name, author,
                f"{thinking.content}\n\nAnswer:{answer_content}",
                prompt, None, None, iteration_idx,
            )
        elif agents is not None and sub_tasks is None:
            final = Info(
                name, author,
                f"{thinking.content}\n\nAnswer:{answer_content}",
                prompt, None, "\n".join(agents), iteration_idx,
            )
        else:
            final = Info(
                name, author,
                f"{thinking.content}\n\nAnswer:{answer_content}",
                prompt, "\n".join(sub_tasks), "\n".join(agents), iteration_idx,
            )
        return final

    # ── RAG tool methods (called by generated forward code) ──

    def retrieve(self, query: str, top_k: int = 20) -> str:
        """Retrieve relevant documents for the given query."""
        if self._retrieve_fn is None:
            return "No retriever available."
        return self._retrieve_fn(query, top_k)

    def rerank(self, query: str, top_k: int = 10) -> str:
        """Rerank previously retrieved documents for the given query."""
        if self._rerank_fn is None:
            return "No reranker available."
        return self._rerank_fn(query, top_k)

    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        if self._calc_fn is None:
            return "No calculator available."
        return self._calc_fn(expression)
