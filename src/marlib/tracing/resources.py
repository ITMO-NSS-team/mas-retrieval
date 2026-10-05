"""Cooperative request/tool budgets and durable API-attempt accounting.

These guards cover explicitly instrumented calls in one process. Token limits
are observed-usage stop thresholds, not hard provider billing limits. Wall time
is checked at call boundaries; arbitrary generated Python needs process isolation.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from marlib.tracing.tracker import TokenTracker


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceLimits:
    max_requests: int | None = None
    max_tool_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    wall_seconds: float | None = None
    request_timeout: float = 60.0
    output_per_request: int = 4096
    scope: str = "full_answer"

    def __post_init__(self):
        if self.scope not in {"full_answer", "execution", "construction"}:
            raise ValueError("Unknown resource budget scope")
        for key, value in asdict(self).items():
            if key == "scope" or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key not in {"wall_seconds", "request_timeout"} and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")


class ResourceSession:
    def __init__(self, tracker: TokenTracker, limits: ResourceLimits,
                 journal: Path | None = None, *, coverage: str = "mas_zero_in_process_helpers"):
        self.tracker, self.limits, self.journal = tracker, limits, journal
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.requests = self.tools = self.input_tokens = self.output_tokens = 0
        self.reserved_output = self.unknown_usage = 0
        self.logical_ids: set[str] = set()
        self.coverage = coverage
        self._events: list[dict] = []
        if journal:
            journal.parent.mkdir(parents=True, exist_ok=True)
            # Unique attempt directories are supplied by the adapter.
            journal.touch(exist_ok=False)
        self.write({"kind": "session", "limits": asdict(limits),
                    "token_semantics": "observed_usage_stop_with_output_reservation",
                    "wall_semantics": "cooperative_call_boundary"})

    def write(self, event: dict):
        with self.lock:
            if self.journal:
                with self.journal.open("a") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
                    f.flush()

    def check(self):
        lim = self.limits
        if lim.wall_seconds and time.perf_counter() - self.started >= lim.wall_seconds:
            raise BudgetExhausted("wall_seconds")
        for key, used in (("max_input_tokens", self.input_tokens),
                          ("max_output_tokens", self.output_tokens),
                          ("max_total_tokens", self.input_tokens + self.output_tokens)):
            cap = getattr(lim, key)
            if cap is not None and used >= cap:
                raise BudgetExhausted(key)
        if self.unknown_usage and any((lim.max_input_tokens, lim.max_output_tokens, lim.max_total_tokens)):
            raise BudgetExhausted("usage_unknown_after_request")

    def tool(self):
        with self.lock:
            self.check()
            if self.limits.max_tool_calls is not None and self.tools >= self.limits.max_tool_calls:
                raise BudgetExhausted("max_tool_calls")
            self.tools += 1

    def _begin(self, phase: str, logical_call_id: str, kwargs: dict):
        # Streaming needs a different usage lifecycle; reject before any request.
        if kwargs.get("stream") is True:
            raise ValueError("ResourceSession requires non-streaming completions")
        with self.lock:
            self.check()
            lim = self.limits
            if lim.max_requests is not None and self.requests >= lim.max_requests:
                raise BudgetExhausted("max_requests")
            token_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
            requested = kwargs.get(token_key)
            output = min(requested, lim.output_per_request) if isinstance(requested, int) else lim.output_per_request
            if output <= 0:
                raise ValueError("Requested output token limit must be positive")
            for cap, used in ((lim.max_output_tokens, self.output_tokens),
                              (lim.max_total_tokens, self.input_tokens + self.output_tokens)):
                if cap is not None:
                    output = min(output, cap - used - self.reserved_output)
            if output <= 0:
                raise BudgetExhausted("output_reservation")
            timeout = lim.request_timeout
            if lim.wall_seconds:
                timeout = min(timeout, lim.wall_seconds - (time.perf_counter() - self.started))
                if timeout <= 0:
                    raise BudgetExhausted("wall_seconds")
            self.requests += 1
            self.logical_ids.add(logical_call_id)
            self.reserved_output += output
            event_id = uuid4().hex
            self.write({"kind": "llm_start", "event_id": event_id,
                        "logical_call_id": logical_call_id, "phase": phase,
                        "model": kwargs["model"], "output_reserved": output})
            event = {"model": kwargs["model"], "phase": phase, "event_id": event_id,
                     "logical_call_id": logical_call_id}
            payload = {**kwargs, token_key: output, "timeout": timeout}
            payload.pop("max_tokens" if token_key == "max_completion_tokens" else "max_completion_tokens", None)
            return event, output, payload, time.perf_counter()

    def _finish(self, event, output, started, response, error):
        usage = getattr(response, "usage", None)
        known = usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None
        inp = usage.prompt_tokens if known else 0
        out = usage.completion_tokens if known else 0
        event = dict(**event, prompt_tokens=inp, completion_tokens=out,
                     latency_ms=(time.perf_counter() - started) * 1000,
                     usage_known=known, error=error, measurement="api_attempt")
        with self.lock:
            self.input_tokens += inp
            self.output_tokens += out
            self.unknown_usage += int(not known)
            self.reserved_output -= output
            self._events.append(event)
            self.tracker.log_llm_call(**event)
            self.write({"kind": "llm_end", **event})

    def create(self, client, phase: str, logical_call_id: str, **kwargs):
        event, output, payload, started = self._begin(phase, logical_call_id, kwargs)
        response, error = None, None
        try:
            response = client.chat.completions.create(**payload)
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._finish(event, output, started, response, error)

    async def create_async(self, create, phase: str, logical_call_id: str, **kwargs):
        """One SDK attempt; the supplied client must have SDK retries disabled."""
        event, output, payload, started = self._begin(phase, logical_call_id, kwargs)
        response, error = None, None
        try:
            response = await create(**payload)
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._finish(event, output, started, response, error)

    def summary(self) -> dict:
        with self.lock:
            phases = {}
            for call in self._events:
                phase = phases.setdefault(call["phase"], {"input_tokens": 0, "output_tokens": 0,
                                                       "api_attempts": 0, "sum_latency_ms": 0.0})
                phase["input_tokens"] += call["prompt_tokens"]
                phase["output_tokens"] += call["completion_tokens"]
                phase["api_attempts"] += 1
                phase["sum_latency_ms"] += call["latency_ms"]
            return {"limits": asdict(self.limits), "api_attempts": self.requests,
                    "phases": phases,
                    "logical_calls": len(self.logical_ids), "tool_calls": self.tools,
                    "unknown_usage_attempts": self.unknown_usage,
                    "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_token_overshoot": max(0, self.input_tokens + self.output_tokens - self.limits.max_total_tokens)
                    if self.limits.max_total_tokens is not None else 0,
                    "wall_seconds": time.perf_counter() - self.started,
                    "coverage": self.coverage,
                    "hard_wall_timeout": False}


def tracked_async_client(session: ResourceSession, phase: str, **client_kwargs):
    """Create a dedicated SDK client for Pydantic AI Chat Completions.

    The wrapper is local to this client. It counts each explicit retry once and
    leaves other framework/provider clients untouched. Responses/streaming are
    outside this integration and must not be used with this client.
    """
    import openai

    client = openai.AsyncOpenAI(**{**client_kwargs, "max_retries": 0})
    raw_create = client.chat.completions.create

    async def create(**kwargs):
        logical_id = uuid4().hex
        for attempt in range(3):
            try:
                return await session.create_async(raw_create, phase, logical_id, **kwargs)
            except (openai.RateLimitError, openai.APITimeoutError):
                if attempt == 2:
                    raise
                session.check()
                await asyncio.sleep(0.5 * (attempt + 1))

    client.chat.completions.create = create
    return client


class TrackedCompletion:
    def __init__(self, session: ResourceSession, phase: str):
        self.session, self.phase = session, phase

    def create(self, client, logical_call_id: str, **kwargs):
        return self.session.create(client, self.phase, logical_call_id, **kwargs)


def completion_request(client, callback, logical_call_id: str, **kwargs):
    import openai

    for attempt in range(3):
        try:
            if isinstance(callback, TrackedCompletion):
                return callback.create(client, logical_call_id, **kwargs)
            response = client.chat.completions.create(**kwargs)
            break
        except (openai.RateLimitError, openai.APITimeoutError):
            if attempt == 2:
                raise
            if isinstance(callback, TrackedCompletion):
                callback.session.check()
            time.sleep(0.5 * (attempt + 1))
    if callback and response.usage:
        callback(response.usage.prompt_tokens, response.usage.completion_tokens)
    return response
