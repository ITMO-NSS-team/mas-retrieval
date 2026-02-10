"""Adapted LLM wrapper for MetaAgent FSM-based multi-agent system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from retcapslib.logging.tracker import TokenTracker


class LLM:
    """Thin wrapper around OpenAI chat completions with token tracking."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        tracker: TokenTracker | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        self.system_prompt = system_prompt
        self.token_cost = 0
        self._model = model
        self._tracker = tracker

    def chat(self, message: str, temperature: float = 0.2) -> str:
        self.messages.append({"role": "user", "content": message})

        if self._tracker is not None:
            with self._tracker.track_llm(self._model) as stats:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=self.messages,
                    temperature=temperature,
                )
                if response.usage:
                    stats["prompt_tokens"] = response.usage.prompt_tokens
                    stats["completion_tokens"] = response.usage.completion_tokens
        else:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=self.messages,
                temperature=temperature,
            )

        rsp = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": rsp})
        if response.usage:
            self.token_cost += response.usage.total_tokens
        return rsp

    def add_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_tool_message(self, message: str) -> None:
        self.messages.append(
            {"role": "user", "content": "[INFO] This is a tool message:\n" + message}
        )

    def get_token_cost(self) -> int:
        return self.token_cost

    def recount_token(self) -> None:
        self.token_cost = 0
