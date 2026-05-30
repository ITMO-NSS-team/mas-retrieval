from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from automas.meta_agents import GraphGenerator, PoolGenerator
from automas.pipeline import PipelineBuilder
from marlib.adapters.base import AbstractAdapter, register
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker


@register("automas")
class AutoMASAdapter(AbstractAdapter):
    def __init__(
        self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "per_task"

        self._cached_pool: Any = None
        self._cached_graph: Any = None

    def _on_benchmark_change(self) -> None:
        self._cached_pool = None
        self._cached_graph = None

    def _build_task_description(self) -> str:
        """Build a generic task description from benchmark context for one_time mode."""
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

    @property
    def name(self) -> str:
        return f"automas_{self._generation_mode}"

    def _setup_mcp_registry(self) -> None:
        pass

    def _set_llm_env(self) -> None:
        model = self._model
        os.environ.setdefault("AGENT_NODE_MODEL", model)
        os.environ.setdefault("DEFAULT_META_MODEL", model)

    def _init_framework(self) -> None:
        self._set_llm_env()
        self._setup_mcp_registry()

    def generate_system(self, question: str) -> str:
        return "AutoMAS auto-generated multi-agent pipeline (per-task)"

    async def _ensure_structure(self, question: str) -> tuple[Any, Any]:
        if (
            self._generation_mode == "one_time"
            and self._cached_pool is not None
            and self._cached_graph is not None
        ):
            return self._cached_pool, self._cached_graph

        pool_gen = PoolGenerator()
        graph_gen = GraphGenerator()

        # Use generic benchmark description for one_time mode,
        # specific question for per-task mode
        if self._generation_mode == "one_time":
            task_description = self._build_task_description()
        else:
            task_description = question

        pool = await pool_gen.create_pool(task_description)
        graph = await graph_gen.create_graph(pool, task_description)

        if self._generation_mode == "one_time":
            self._cached_pool = pool
            self._cached_graph = graph

        return pool, graph

    async def _execute_async(self, question: str) -> tuple[Any, Any]:
        pool, graph = await self._ensure_structure(question)

        # PipelineBuilder.create_from_pool() deep-copies agents internally,
        # so pool/graph templates can be reused directly.
        # Shallow-copy graph dict as a safety measure.
        builder = PipelineBuilder()
        pipeline = builder.create_from_pool(
            pool, {k: list(v) for k, v in graph.items()}
        ).build()
        result = await pipeline.ainvoke(question)
        return result, pipeline

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        self._init_framework()

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
            result, pipeline = asyncio.run(self._execute_async(question))
            answer = self._extract_answer(result)

            prompt_tokens = getattr(pipeline, "input_tokens", 0) or 0
            completion_tokens = getattr(pipeline, "output_tokens", 0) or 0

            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,
                function_calls=0,
            )

            self._log_tool_calls(tracker, docids_file)

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        if docids_file.exists():
            docids_file.unlink()

        return answer, tracker.to_question_log(answer)

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        if result is None:
            return ""

        if isinstance(result, dict):
            answer = result.get("answer")
            if answer is not None:
                return str(answer).strip()
            return str(result)

        return str(result).strip()

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
