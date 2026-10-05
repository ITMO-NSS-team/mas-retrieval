"""Cooperative request/tool budgets and durable API-attempt accounting.

These guards cover explicitly instrumented calls in one process. Token limits
are observed-usage stop thresholds, not hard provider billing limits. Wall time
is checked at call boundaries; arbitrary generated Python needs process isolation.
"""
from __future__ import annotations

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
        if self.scope != "full_answer":
            raise ValueError("Only full_answer scope is implemented for MAS-Zero")
        for key, value in asdict(self).items():
            if key == "scope" or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key not in {"wall_seconds", "request_timeout"} and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")


class ResourceSession:
    def __init__(self, tracker: TokenTracker, limits: ResourceLimits,
                 journal: Path | None = None):
        self.tracker, self.limits, self.journal = tracker, limits, journal
        self.lock = threading.RLock()
        self.started = time.perf_counter()
        self.requests = self.tools = self.input_tokens = self.output_tokens = 0
        self.reserved_output = self.unknown_usage = 0
        self.logical_ids: set[str] = set()
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

    def create(self, client, phase: str, logical_call_id: str, **kwargs):
        with self.lock:
            self.check()
            lim = self.limits
            if lim.max_requests is not None and self.requests >= lim.max_requests:
                raise BudgetExhausted("max_requests")
            output = min(kwargs.get("max_tokens", lim.output_per_request), lim.output_per_request)
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
        started = time.perf_counter()
        response, error = None, None
        try:
            response = client.chat.completions.create(**{**kwargs, "max_tokens": output, "timeout": timeout})
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            usage = getattr(response, "usage", None)
            known = usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None
            inp = usage.prompt_tokens if known else 0
            out = usage.completion_tokens if known else 0
            event = dict(model=kwargs["model"], prompt_tokens=inp,
                         completion_tokens=out, latency_ms=(time.perf_counter() - started) * 1000,
                         phase=phase, event_id=event_id, logical_call_id=logical_call_id,
                         usage_known=known, error=error, measurement="api_attempt")
            with self.lock:
                self.input_tokens += inp
                self.output_tokens += out
                self.unknown_usage += int(not known)
                self.reserved_output -= output
                self.tracker.log_llm_call(**event)
                self.write({"kind": "llm_end", **event})

    def summary(self) -> dict:
        with self.lock:
            phases = {}
            for call in self.tracker._llm_calls:
                phase = phases.setdefault(call.phase, {"input_tokens": 0, "output_tokens": 0,
                                                       "api_attempts": 0, "sum_latency_ms": 0.0})
                phase["input_tokens"] += call.prompt_tokens
                phase["output_tokens"] += call.completion_tokens
                phase["api_attempts"] += 1
                phase["sum_latency_ms"] += call.latency_ms
            return {"limits": asdict(self.limits), "api_attempts": self.requests,
                    "phases": phases,
                    "logical_calls": len(self.logical_ids), "tool_calls": self.tools,
                    "unknown_usage_attempts": self.unknown_usage,
                    "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_token_overshoot": max(0, self.input_tokens + self.output_tokens - self.limits.max_total_tokens)
                    if self.limits.max_total_tokens is not None else 0,
                    "wall_seconds": time.perf_counter() - self.started,
                    "coverage": "mas_zero_in_process_helpers",
                    "hard_wall_timeout": False}


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
