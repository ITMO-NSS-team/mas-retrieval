"""AutoMAS adapter: auto-generated multi-agent pipeline with our retrieval tools.

Uses AutoMAS as a library. For each question:
1. PoolGenerator meta-agent creates an agent pool from MCP tool descriptions
2. GraphGenerator meta-agent builds a DAG connecting agents
3. Pipeline executes the DAG and returns the answer

MCP registry is monkey-patched to expose only our retrieval + calculator tools.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker


class AutoMASAdapter(AbstractAdapter):
    """AutoMAS adapter: auto-generated multi-agent pipeline with our retrieval tools."""

    def __init__(self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "per_question"
        self._automas: Any = None
        self._cached_pipeline: Any = None

    @property
    def name(self) -> str:
        return f"automas_{self._generation_mode}"

    def _setup_mcp_registry(self) -> None:
        """Replace AutoMAS MCP servers with our retrieval+calculator server."""
        import automas.mcp.external_descriptions as ext_desc
        import automas.mcp.registry as reg
        from automas.mcp.server_config import MCPServerConfig

        server_script = str(Path(__file__).parent / "mcp_retrieval_server.py")

        # Clear all existing MCP servers and register only ours
        reg.MCP_SERVERS.clear()
        reg.MCP_SERVERS["retrieval-search"] = MCPServerConfig(
            command=sys.executable,
            args=(server_script,),
            timeout=60,
            retries=2,
            module_path=None,
        )

        # Add description for the meta-agent planner
        ext_desc.EXTERNAL_SERVER_DESCRIPTIONS.clear()
        ext_desc.EXTERNAL_SERVER_DESCRIPTIONS["retrieval-search"] = (
            "MCP server for document knowledge base retrieval and calculation.\n"
            "Tools:\n"
            "- retrieval_search(query, top_k, use_rerank): Search document knowledge base "
            "for relevant passages using dense retrieval + optional cross-encoder reranking.\n"
            "- calculate(expression): Safely evaluate mathematical expressions "
            "(supports +, -, *, /, **, round, abs, min, max)."
        )

    def _set_retriever_env(self) -> None:
        """Set env vars for MCP server subprocess to initialize retriever."""
        cfg = self._retriever._config
        os.environ["RETCAP_INDEX_PATH"] = str(cfg["index_path"])
        os.environ["RETCAP_COLLECTION"] = self._retriever._collection_name
        os.environ["RETCAP_EMBEDDER"] = str(cfg.get("embedder", "BAAI/bge-m3"))
        os.environ["RETCAP_RERANKER"] = str(cfg.get("reranker", "BAAI/bge-reranker-v2-m3"))

    def _set_llm_env(self) -> None:
        """Set env vars for AutoMAS LLM configuration."""
        model = self._model
        # AutoMAS uses OpenRouter format (e.g., "openai/gpt-4o-mini")
        if "/" not in model:
            model = f"openai/{model}"
        os.environ.setdefault("AGENT_NODE_MODEL", model)
        os.environ.setdefault("DEFAULT_META_MODEL", model)

    def _init_framework(self) -> None:
        if self._automas is not None:
            return

        self._set_llm_env()
        self._set_retriever_env()
        self._setup_mcp_registry()

        from automas.main import AutoMAS

        self._automas = AutoMAS()

    def generate_system(self, question: str) -> str:
        return "AutoMAS auto-generated multi-agent pipeline (per-question)"

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

        # Set doc_ids tracking file for this question
        docids_file = Path(f"/tmp/retcap_docids_{question_id}.jsonl")
        if docids_file.exists():
            docids_file.unlink()
        os.environ["RETCAP_DOCIDS_FILE"] = str(docids_file)

        try:
            if self._generation_mode == "shared" and self._cached_pipeline is not None:
                result = self._cached_pipeline.execute(query=question)
            else:
                result = self._automas.run(query=question)
                if self._generation_mode == "shared":
                    self._cached_pipeline = self._automas.pipeline

            answer = self._extract_answer(result)

            # Log LLM usage from pipeline execution
            pipeline = self._cached_pipeline or self._automas.pipeline
            prompt_tokens = getattr(pipeline, "input_tokens", 0) or 0
            completion_tokens = getattr(pipeline, "output_tokens", 0) or 0
            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,
                function_calls=0,
            )

            # Log tool calls from doc_ids file
            self._log_tool_calls(tracker, docids_file)

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        # Cleanup
        if docids_file.exists():
            docids_file.unlink()

        return answer, tracker.to_question_log(answer)

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        """Extract final answer from AutoMAS result dict."""
        if result is None:
            return ""
        answer = result.get("answer")
        if answer is not None:
            return str(answer).strip()
        return str(result)

    @staticmethod
    def _log_tool_calls(tracker: TokenTracker, docids_file: Path) -> None:
        """Read doc_ids file written by MCP server and log tool calls."""
        if not docids_file.exists():
            return
        with open(docids_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tracker.log_tool_call(
                    tool_name=entry.get("tool", "retrieve"),
                    query=entry.get("query", ""),
                    top_k=len(entry.get("doc_ids", [])),
                    results=entry.get("doc_ids", []),
                    latency_ms=0,
                )
