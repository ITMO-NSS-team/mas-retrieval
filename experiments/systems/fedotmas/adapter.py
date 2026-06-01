from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fedotmas import MAW, MAWConfig
from fedotmas.mcp.registry import MCPServerConfig, StdioMCPServer
from marlib.adapters.base import AbstractAdapter, register
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker


def _run_async(coro: Any) -> Any:
    """Run *coro* on a fresh event loop, then drain background cleanup before
    closing it.

    We call this once per question. The ADK + LiteLLM/httpx stack leaves
    keep-alive connection pools that close lazily via finalizers; with a bare
    ``asyncio.run`` those finalizers fire ``loop.call_soon`` *after* the loop is
    already closed, producing harmless but noisy ``RuntimeError: Event loop is
    closed`` / "Task exception was never retrieved" teardown tracebacks. We (a)
    install an exception handler that swallows exactly that error, and (b)
    cancel + await any still-pending tasks so the pools close on a live loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _ignore_loop_closed(loop: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_ignore_loop_closed)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@register("fedotmas")
class FedotMAWAdapter(AbstractAdapter):
    def __init__(
        self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "one_time"

        self._cached_config: MAWConfig | None = None

    def _on_benchmark_change(self) -> None:
        self._cached_config = None

    @property
    def name(self) -> str:
        return f"fedotmas_{self._generation_mode}"

    def _build_mcp_registry(self) -> dict[str, MCPServerConfig]:
        return {
            "retrieval": StdioMCPServer(
                command=sys.executable,
                args=(
                    "-m",
                    "marlib.mcp_server",
                ),
                timeout=30,
                description=(
                    "Search a document corpus and calculate mathematical expressions. "
                ),
                tags=("retrieval", "math"),
            ),
        }

    def _build_task_description(self) -> str:
        parts = []
        if self._benchmark_description:
            parts.append(self._benchmark_description)
        else:
            parts.append(
                "Answer questions accurately using retrieval-augmented generation."
            )
        if self._sample_questions:
            examples = "\n".join(f"- {q}" for q in self._sample_questions[:3])
            parts.append(f"\nExample questions from the benchmark:\n{examples}")
        return "\n".join(parts)

    async def _generate_config(
        self, question: str
    ) -> tuple[MAWConfig, MAW | None]:
        if self._generation_mode == "one_time" and self._cached_config is not None:
            return self._cached_config, None

        registry = self._build_mcp_registry()
        maw = MAW(
            meta_model=self._model, worker_models=[self._model], mcp_servers=registry
        )

        if self._generation_mode == "one_time":
            task_description = self._build_task_description()
        else:
            task_description = question

        config = await maw.generate_config(task_description)

        if self._generation_mode == "one_time":
            self._cached_config = config

        return config, maw

    def generate_system(self, question: str) -> str:
        config, _ = _run_async(self._generate_config(question))
        return config.model_dump_json(indent=2)

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        docids_file = Path(f"/tmp/marlib_docids_{question_id}.jsonl")
        if docids_file.exists():
            docids_file.unlink()
        os.environ["MARLIB_DOCIDS_FILE"] = str(docids_file)

        try:
            config, meta_maw = _run_async(self._generate_config(question))

            if meta_maw is not None:
                tracker.log_llm_call(
                    model=self._model,
                    prompt_tokens=meta_maw.meta_prompt_tokens,
                    completion_tokens=meta_maw.meta_completion_tokens,
                    latency_ms=meta_maw.meta_elapsed * 1000,
                )

            registry = self._build_mcp_registry()
            maw = MAW(
                meta_model=self._model,
                worker_models=[self._model],
                mcp_servers=registry,
            )
            result = _run_async(maw.build_and_run(config, question))

            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=maw.total_prompt_tokens,
                completion_tokens=maw.total_completion_tokens,
                latency_ms=maw.elapsed * 1000,
            )

            answer = self._extract_answer(result)
            self._log_tool_calls(tracker, docids_file)

        except Exception as e:
            tracker.set_error(str(e))
            import traceback

            traceback.print_exc()
            answer = ""

        if docids_file.exists():
            docids_file.unlink()

        return answer, tracker.to_question_log(answer)

    @staticmethod
    def _extract_answer(state: dict[str, Any]) -> str:
        if not state or not isinstance(state, dict):
            return ""

        for key in reversed(list(state.keys())):
            if key == "user_query":
                continue
            value = state[key]
            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    @staticmethod
    def _log_tool_calls(tracker: TokenTracker, docids_file: Path) -> None:
        if not docids_file.exists():
            return

        with open(docids_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    tracker.log_tool_call(
                        tool_name=entry.get("tool", "retrieve"),
                        query=entry.get("query", ""),
                        top_k=len(entry.get("doc_ids", [])),
                        results=entry.get("doc_ids", []),
                        latency_ms=0,
                    )
                except json.JSONDecodeError:
                    continue
