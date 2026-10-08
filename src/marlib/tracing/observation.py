"""Passive Chat Completions accounting without changing SDK request policy.

Dedicated HTTP clients observe each send, including SDK retries. No admission
limits, decoding options, retry policy or timeouts are injected. This is separate
from ResourceSession, whose request wrapper deliberately enforces limits.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

import httpx
import openai

from marlib.tracing.tracker import TokenTracker


_active: ContextVar[tuple | None] = ContextVar("marlib_observation", default=None)
_logical_call: ContextVar[str | None] = ContextVar("marlib_sdk_call", default=None)


class CallObserver:
    def __init__(self, tracker: TokenTracker, journal: Path):
        self.tracker = tracker
        self.journal = journal
        self.started = time.perf_counter()
        self.lock = threading.RLock()
        self.calls: list[dict] = []
        self.agents: list[dict] = []
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.touch(exist_ok=False)
        self.write({"kind": "session", "policy": "observe_only", "limits": None})

    def write(self, event: dict) -> None:
        with self.lock, self.journal.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()

    @contextmanager
    def phase(self, name: str):
        token = _active.set((self, name))
        try:
            yield
        finally:
            _active.reset(token)

    def begin(self, request: httpx.Request):
        payload = json.loads(request.content)
        active = _active.get()
        event = {"event_id": uuid4().hex, "logical_call_id": _logical_call.get() or uuid4().hex,
                 "phase": active[1] if active and active[0] is self else "answer_execution",
                 "model": payload["model"]}
        self.write({"kind": "llm_start", **event, "request": payload,
                    "sdk_retry_count": request.headers.get("x-stainless-retry-count")})
        return event, time.perf_counter()

    def finish(self, event: dict, started: float, response, error: str | None) -> None:
        body = None
        if response is not None:
            try:
                body = response.json()
            except (ValueError, httpx.ResponseNotRead):
                pass
            if response.is_error:
                error = f"HTTP {response.status_code}"
        usage = body.get("usage") if isinstance(body, dict) else None
        known = isinstance(usage, dict) and all(isinstance(usage.get(k), int)
                                              for k in ("prompt_tokens", "completion_tokens"))
        call = {**event, "prompt_tokens": usage["prompt_tokens"] if known else 0,
                "completion_tokens": usage["completion_tokens"] if known else 0,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage_known": known, "error": error, "measurement": "api_attempt"}
        with self.lock:
            self.calls.append(call)
            self.tracker.log_llm_call(**call)
            self.write({"kind": "llm_end", **call, "response": body})

    def summary(self, phase: str | None = None) -> dict:
        calls = [c for c in self.calls if phase is None or c["phase"] == phase]
        phases = {}
        for call in calls:
            item = phases.setdefault(call["phase"], {"input_tokens": 0, "output_tokens": 0,
                                                      "api_attempts": 0, "unknown_usage_attempts": 0})
            item["input_tokens"] += call["prompt_tokens"]
            item["output_tokens"] += call["completion_tokens"]
            item["api_attempts"] += 1
            item["unknown_usage_attempts"] += int(not call["usage_known"])
        return {"coverage": "chat_completions_http_sends_including_sdk_retries",
                "policy": "observe_only", "limits": None, "phases": phases,
                "api_attempts": len(calls), "logical_calls": len({c["logical_call_id"] for c in calls}),
                "logical_call_definition": "SDK create invocation; SDK retries share one ID",
                "input_tokens": sum(c["prompt_tokens"] for c in calls),
                "output_tokens": sum(c["completion_tokens"] for c in calls),
                "unknown_usage_attempts": sum(not c["usage_known"] for c in calls)}


class ObservedHttpClient(openai.DefaultHttpxClient):
    def __init__(self, observer: CallObserver, **kwargs):
        super().__init__(**kwargs)
        self.observer = observer

    def send(self, request, **kwargs):
        event, started = self.observer.begin(request)
        response, error = None, None
        try:
            response = super().send(request, **kwargs)
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.observer.finish(event, started, response, error)


class ObservedAsyncHttpClient(httpx.AsyncClient):
    """Caller supplies the original provider's HTTP settings."""

    def __init__(self, observer: CallObserver, **kwargs):
        super().__init__(**kwargs)
        self.observer = observer

    async def send(self, request, **kwargs):
        event, started = self.observer.begin(request)
        response, error = None, None
        try:
            response = await super().send(request, **kwargs)
            return response
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.observer.finish(event, started, response, error)


def label_async_completions(client) -> None:
    create = client.chat.completions.create

    async def observed_create(*args, **kwargs):
        token = _logical_call.set(uuid4().hex)
        try:
            return await create(*args, **kwargs)
        finally:
            _logical_call.reset(token)

    client.chat.completions.create = observed_create


@contextmanager
def completion_client(**kwargs):
    """Original OpenAI defaults; passive HTTP observation only when activated."""
    active = _active.get()
    if active is None:
        with openai.OpenAI(**kwargs) as client:
            yield client
        return
    with ObservedHttpClient(active[0]) as http:
        with openai.OpenAI(**kwargs, http_client=http) as client:
            create = client.chat.completions.create

            def observed_create(*args, **options):
                token = _logical_call.set(uuid4().hex)
                try:
                    return create(*args, **options)
                finally:
                    _logical_call.reset(token)

            client.chat.completions.create = observed_create
            yield client


def observe_agent(agent) -> None:
    active = _active.get()
    if active:
        observer, phase = active
        event = {"kind": "agent_created", "phase": phase, "agent_id": agent.id,
                 "agent_name": agent.agent_name, "model": agent.model,
                 "temperature": agent.temperature, "role": agent.role,
                 "output_fields": agent.output_fields}
        observer.agents.append(event)
        observer.write(event)
