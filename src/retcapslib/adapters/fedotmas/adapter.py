from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from fedotmas import MAS, PipelineConfig
from fedotmas.mcp.registry import MCPServerConfig, StdioMCPServer

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker


class FedotMASAdapter(AbstractAdapter):
    def __init__(
        self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "shared"

        self._cached_config: PipelineConfig | None = None

    def _on_benchmark_change(self) -> None:
        self._cached_config = None

    @property
    def name(self) -> str:
        return f"fedotmas_{self._generation_mode}"

    def _set_retriever_env(self) -> None:
        cfg = self._retriever._config
        os.environ["RETCAP_INDEX_PATH"] = str(cfg["index_path"])
        os.environ["RETCAP_COLLECTION"] = self._retriever._collection_name
        os.environ["RETCAP_EMBEDDER"] = str(cfg.get("embedder", "BAAI/bge-m3"))
        os.environ["RETCAP_RERANKER"] = str(
            cfg.get("reranker", "BAAI/bge-reranker-v2-m3")
        )

    def _build_mcp_registry(self) -> dict[str, MCPServerConfig]:
        return {
            "retcap-retrieval": StdioMCPServer(
                command=sys.executable,
                args=(
                    "-m",
                    "retcapslib.adapters.automas.mcp_retrieval_server",
                ),
                timeout=30,
                description=(
                    "Search a document corpus and calculate mathematical expressions. "
                    "Use 'retrieval_search' to find relevant passages by semantic query "
                    "and 'calculate' to evaluate arithmetic."
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

    async def _generate_config(self, question: str) -> PipelineConfig:
        if self._generation_mode == "shared" and self._cached_config is not None:
            return self._cached_config

        registry = self._build_mcp_registry()
        mas = MAS(
            meta_model=self._model, worker_models=[self._model], mcp_servers=registry
        )

        if self._generation_mode == "shared":
            task_description = self._build_task_description()
        else:
            task_description = question

        config = await mas.generate_config(task_description)

        if self._generation_mode == "shared":
            self._cached_config = config

        return config

    def generate_system(self, question: str) -> str:
        config = asyncio.run(self._generate_config(question))
        return config.model_dump_json(indent=2)

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        self._set_retriever_env()

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        docids_file = Path(f"/tmp/retcap_docids_{question_id}.jsonl")
        if docids_file.exists():
            docids_file.unlink()
        os.environ["RETCAP_DOCIDS_FILE"] = str(docids_file)

        try:
            config = asyncio.run(self._generate_config(question))

            registry = self._build_mcp_registry()
            mas = MAS(meta_model=self._model, mcp_servers=registry)
            result = asyncio.run(mas.build_and_run(config, question))

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
