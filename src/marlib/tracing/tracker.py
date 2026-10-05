from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Generator

from marlib.tracing.schemas import LLMCall, QuestionLog, ToolCall


class TokenTracker:
    """Accumulates tool calls, LLM calls, and tokens for one question."""

    def __init__(self, question_id: str, question: str, gold_answer: str) -> None:
        self.question_id = question_id
        self.question = question
        self.gold_answer = gold_answer

        self._tool_calls: list[ToolCall] = []
        self._llm_calls: list[LLMCall] = []
        self._start_time = time.perf_counter()
        self._error: str | None = None
        self.tool_event_sink = None

    def log_tool_call(
        self,
        tool_name: str,
        query: str,
        top_k: int,
        results: list[str],
        latency_ms: float,
        phase: str = "answer_execution",
        error: str | None = None,
    ) -> None:
        call = ToolCall(
            tool_name=tool_name,
            query=query,
            top_k=top_k,
            results=results,
            latency_ms=latency_ms,
            phase=phase,
            error=error,
        )
        self._tool_calls.append(call)
        if self.tool_event_sink:
            self.tool_event_sink({"kind": "tool_end", **call.model_dump()})

    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        function_calls: int = 0,
        **event: Any,
    ) -> None:
        self._llm_calls.append(
            LLMCall(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                function_calls=function_calls,
                **event,
            )
        )

    def set_error(self, error: str) -> None:
        self._error = error

    @contextmanager
    def track_tool(
        self, tool_name: str, query: str, top_k: int, phase: str = "answer_execution"
    ) -> Generator[list[str], None, None]:
        """Time a tool call; append returned doc_ids to the yielded list."""
        results: list[str] = []
        start = time.perf_counter()
        error = None
        try:
            yield results
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.log_tool_call(tool_name, query, top_k, results, latency_ms, phase=phase, error=error)

    @contextmanager
    def track_llm(self, model: str) -> Generator[dict[str, Any], None, None]:
        """Time an LLM call; fill the yielded stats dict (prompt/completion tokens)."""
        stats: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "function_calls": 0,
        }
        start = time.perf_counter()
        try:
            yield stats
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.log_llm_call(
                model=model,
                prompt_tokens=stats["prompt_tokens"],
                completion_tokens=stats["completion_tokens"],
                latency_ms=latency_ms,
                function_calls=stats["function_calls"],
            )

    def to_question_log(self, predicted_answer: str) -> QuestionLog:
        total_latency_ms = (time.perf_counter() - self._start_time) * 1000
        total_prompt = sum(call.prompt_tokens for call in self._llm_calls)
        total_completion = sum(call.completion_tokens for call in self._llm_calls)

        return QuestionLog(
            question_id=self.question_id,
            question=self.question,
            gold_answer=self.gold_answer,
            predicted_answer=predicted_answer,
            tool_calls=self._tool_calls,
            llm_calls=self._llm_calls,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            total_latency_ms=total_latency_ms,
            num_retrieval_calls=sum(c.tool_name in {"retrieve", "search"} for c in self._tool_calls),
            num_tool_calls=len(self._tool_calls),
            tool_counts=dict(Counter(c.tool_name for c in self._tool_calls)),
            num_llm_calls=len(self._llm_calls),
            error=self._error,
            status="failed" if self._error else "completed",
        )
