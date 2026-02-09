"""CG-MAS adapter: FEDOT.MAS code-generation framework with our retrieval tools.

Uses FEDOT.MAS GraphMASFramework as a library. For each question:
1. Meta-agent plans a workflow graph with MCP tools
2. Coder-agent generates Python code (async run_workflow())
3. Validator executes code, with debug loop on errors
4. Returns final answer

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


class CGMASAdapter(AbstractAdapter):
    """CG-MAS adapter: FEDOT.MAS code-generation framework with our retrieval tools."""

    def __init__(self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(retriever, model, **kwargs)
        self._framework: Any = None

    @property
    def name(self) -> str:
        return "cg_mas"

    def _setup_mcp_registry(self) -> None:
        """Replace FEDOT.MAS MCP servers with our retrieval+calculator server."""
        import fedotmas.mcp.external_descriptions as ext_desc
        import fedotmas.mcp.registry as reg
        from fedotmas.mcp.server_config import MCPServerConfig

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
        """Set env vars for FEDOT.MAS LLM configuration."""
        os.environ.setdefault("PLANNER_LLM_MODEL", self._model)
        os.environ.setdefault("CODER_LLM_MODEL", self._model)
        os.environ.setdefault("DEBUG_LLM_MODEL", self._model)
        os.environ.setdefault("SUPERVISOR_LLM_MODEL", self._model)

    def _init_framework(self) -> None:
        if self._framework is not None:
            return

        self._set_llm_env()
        self._set_retriever_env()
        self._setup_mcp_registry()

        from fedotmas.graph_framework import GraphMASFramework

        self._framework = GraphMASFramework(
            max_execution_cycles=3,
            emit_anyway=True,
            track_timing=True,
        )

    def generate_system(self, question: str) -> str:
        return "FEDOT.MAS code-generation framework (per-question)"

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

        # Set output dir for generated code
        output_dir = Path(f"generated_cg_mas/{question_id}")
        from fedotmas.agents.code_emitter import GraphEmitterAgentConfig

        self._framework.emitter_agent.config = GraphEmitterAgentConfig(
            output_dir=output_dir,
        )

        # Snapshot FEDOT.MAS LLM usage before
        from fedotmas.llm import get_total_llm_usage

        usage_before = get_total_llm_usage()

        try:
            result = self._framework.run(user_request=question, emit_anyway=True)

            answer = self._extract_answer(result)

            # Log LLM call (FEDOT.MAS tracks cost in USD, not individual tokens)
            _ = get_total_llm_usage() - usage_before  # cost delta in USD
            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=0,
                completion_tokens=0,
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
        """Extract final answer from FEDOT.MAS result dict."""
        final_response = result.get("final_response")
        if final_response is None:
            return ""
        if isinstance(final_response, dict):
            answer = final_response.get("final_answer")
            if answer is not None:
                return str(answer).strip()
            for v in final_response.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return str(final_response)
        return str(final_response).strip()

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
