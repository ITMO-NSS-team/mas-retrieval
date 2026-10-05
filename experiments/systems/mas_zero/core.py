"""Self-contained MAS-Zero runtime.

Local adaptation of MAS-Zero's LLMAgentBase / AgentSystem (search.py +
code_archive.py), adapted to:
- Use the OpenAI client directly (no global model_sampler_map / shared_vars)
- Support a usage callback for token tracking
- Provide retrieve/rerank/calculate tool methods on AgentSystem (RAG harness)

This preserves the parts MAS-Zero's algorithm depends on that the ADAS baseline
drops: the ``is_sub_task`` prompting path with the ``[TOO_HARD]`` self-report
mechanism and the sub-task / agent bookkeeping fed back into MAS-Feedback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable

import openai
from uuid import uuid4
from marlib.tracing.resources import completion_request

logger = logging.getLogger(__name__)

Info = namedtuple(
    "Info",
    ["name", "author", "content", "prompt", "sub_tasks", "agents", "iteration_idx"],
)

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
TOO_HARD_MARK = "[TOO_HARD]"

# Sub-task instruction suffix: agents must always give a best answer, but flag
# under-specified / too-hard sub-tasks so MAS-Feedback can drive re-decomposition.
_SUB_TASK_SUFFIX = (
    "\n\nIf the question is too complicated or information is missing, you still "
    "need to give your best answer but add (1) an additional mark "
    f"{TOO_HARD_MARK} on the next line of your final answer and (2) an "
    f"information request or decomposition suggestion on the next line after the "
    f"{TOO_HARD_MARK} mark, in the 'answer' entry (for example: 300\\n"
    f"{TOO_HARD_MARK}\\nSuggestion: ...), and justify why you think so in the "
    "'thinking' entry. Otherwise answer normally."
)

_SUB_QUESTION_PATTERN = re.compile(
    r"Given the above, answer the following question: \s*(.*?)\s*\n\n", re.DOTALL
)


def _pack_message(role: str, content: Any) -> dict[str, Any]:
    return {"role": str(role), "content": content}


class LLMAgentBase:
    """LLM agent that calls OpenAI directly and returns a list of Info.

    Mirrors MAS-Zero's LLMAgentBase, including the ``is_sub_task`` prompt path.
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

    @staticmethod
    def _extract_sub_question(prompt: list[dict] | None) -> str | None:
        """Recover the sub-question header from a prior agent's prompt."""
        if not prompt:
            return None
        content = prompt[-1].get("content", "")
        match = _SUB_QUESTION_PATTERN.search(content)
        return match.group(1) if match else None

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
        prev_sub_question = ""
        for input_info in input_infos:
            if not isinstance(input_info, Info):
                continue
            field_name, author, content, prompt, _, _, iteration_idx = input_info
            if author == repr(self):
                author += " (yourself)"

            if field_name == "task":
                if is_sub_task:
                    input_infos_text += (
                        f"Related original question:\n\n{content}.\n\n"
                        "Related sub-task questions and answers:\n\n"
                    )
                else:
                    input_infos_text += f"{content}\n\n"
                continue

            header = f"{field_name} #{iteration_idx + 1}" if iteration_idx != -1 else field_name
            sub_q = self._extract_sub_question(prompt) if is_sub_task else None
            if sub_q and sub_q != prev_sub_question:
                input_infos_text += (
                    f"### {sub_q}\n\n### {header} by {author}:\n{content}\n\n"
                )
                prev_sub_question = sub_q
            else:
                input_infos_text += f"### {header} by {author}:\n{content}\n\n"

        if is_sub_task:
            prompt = (
                input_infos_text
                + f"Given the above, answer the following question: {instruction}"
                + _SUB_TASK_SUFFIX
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
        system_prompt, user_prompt = self.generate_prompt(
            input_infos, instruction, is_sub_task=is_sub_task
        )

        messages = [
            _pack_message(role="system", content=system_prompt),
            _pack_message(role="user", content=user_prompt),
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
            input_infos,
            instruction,
            iteration_idx=iteration_idx,
            is_sub_task=is_sub_task,
        )


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
        max_retries=0,
        timeout=60.0,
    )

    kwargs: dict[str, Any] = {"model": model or "gpt-4o-mini", "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["response_format"] = {"type": "json_object"}

    logical_id = uuid4().hex
    for _ in range(5):
        response = completion_request(client, usage_callback, logical_id, **kwargs)

        text = response.choices[0].message.content or ""
        try:
            json_dict = json.loads(text)
            if set(json_dict.keys()) >= set(output_fields):
                return {k: json_dict[k] for k in output_fields}
        except (json.JSONDecodeError, KeyError):
            pass

    logger.warning(
        "LLM failed to produce valid JSON with fields %s after 5 attempts (model=%s)",
        output_fields,
        model,
    )
    raise ValueError(f"No valid JSON for required fields {output_fields} after 5 attempts")


class AgentSystem:
    """Hosts the dynamically generated forward() function.

    Attributes set by the adapter before execution:
        node_model, cot_instruction, max_round, max_sc, debate_role,
        _usage_callback, and the retrieve/rerank/calc tool closures.
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
        self._usage_callback: Callable[[int, int], None] | None = None

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

        # MAS-Zero quirk: when only one extra list is passed it is `agents`.
        if agents is None and sub_tasks is not None:
            agents = sub_tasks
            sub_tasks = None

        content = f"{thinking.content}\n\nAnswer:{answer_content}"
        sub_tasks_str = "\n".join(sub_tasks) if sub_tasks else None
        agents_str = "\n".join(agents) if agents else None
        return Info(name, author, content, prompt, sub_tasks_str, agents_str, iteration_idx)

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
