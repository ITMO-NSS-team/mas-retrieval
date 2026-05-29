from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from marlib.tracing.schemas import LLMCall, QuestionLog, ToolCall


class TokenTracker:
    """Tracks tool calls, LLM calls, and tokens during question execution.

    Usage:
        tracker = TokenTracker(question_id="q1", question="...", gold_answer="...")
        tracker.log_tool_call(tool_name="search", query="...", ...)
        tracker.log_llm_call(model="gpt-4o-mini", prompt_tokens=100, ...)
        log = tracker.to_question_log(predicted_answer="...")
    """

    def __init__(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> None:
        """Initialize tracker for a question.

        Args:
            question_id: Unique identifier for the question.
            question: The question text.
            gold_answer: Ground truth answer.
        """
        self.question_id = question_id
        self.question = question
        self.gold_answer = gold_answer

        self._tool_calls: list[ToolCall] = []
        self._llm_calls: list[LLMCall] = []
        self._start_time = time.perf_counter()
        self._error: str | None = None

    def log_tool_call(
        self,
        tool_name: str,
        query: str,
        top_k: int,
        results: list[str],
        latency_ms: float,
    ) -> None:
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool (retrieve, rerank, search).
            query: Query string passed to the tool.
            top_k: Number of results requested.
            results: List of returned doc_ids.
            latency_ms: Tool execution latency in milliseconds.
        """
        self._tool_calls.append(
            ToolCall(
                tool_name=tool_name,
                query=query,
                top_k=top_k,
                results=results,
                latency_ms=latency_ms,
            )
        )

    def log_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        function_calls: int = 0,
    ) -> None:
        """Log an LLM API call.

        Args:
            model: Model identifier (e.g., gpt-4o-mini).
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            latency_ms: API call latency in milliseconds.
            function_calls: Number of function/tool calls in response.
        """
        self._llm_calls.append(
            LLMCall(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                function_calls=function_calls,
            )
        )

    def set_error(self, error: str) -> None:
        """Record an error that occurred during execution."""
        self._error = error

    @contextmanager
    def track_tool(
        self,
        tool_name: str,
        query: str,
        top_k: int,
    ) -> Generator[list[str], None, None]:
        """Context manager for tracking tool calls with automatic timing.

        Usage:
            with tracker.track_tool("search", query, top_k=10) as results:
                docs = retriever.search(query, top_k=10)
                results.extend([doc.doc_id for doc in docs])
        """
        results: list[str] = []
        start = time.perf_counter()
        try:
            yield results
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.log_tool_call(
                tool_name=tool_name,
                query=query,
                top_k=top_k,
                results=results,
                latency_ms=latency_ms,
            )

    @contextmanager
    def track_llm(
        self,
        model: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for tracking LLM calls with automatic timing.

        Usage:
            with tracker.track_llm("gpt-4o-mini") as stats:
                response = openai.chat.completions.create(...)
                stats["prompt_tokens"] = response.usage.prompt_tokens
                stats["completion_tokens"] = response.usage.completion_tokens
                stats["function_calls"] = len(response.choices[0].message.tool_calls or [])
        """
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
        """Create a QuestionLog from tracked data.

        Args:
            predicted_answer: The system's predicted answer.

        Returns:
            QuestionLog with all tracked calls and computed aggregates.
        """
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
            num_retrieval_calls=len(self._tool_calls),
            num_llm_calls=len(self._llm_calls),
            error=self._error,
        )
