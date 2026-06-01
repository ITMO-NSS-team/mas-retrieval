from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from marlib.adapters.base import AbstractAdapter, register
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker

# Description surfaced to AutoMAS' meta-agent (PoolGenerator) so it knows the
# corpus-retrieval server exists and is the way to ground answers. Without this
# the meta-agent only sees AutoMAS' built-in servers (web-search, e2b-sandbox,
# ...) and never touches the benchmark corpus.
_RETRIEVAL_DESCRIPTION = """\
Local document-corpus retrieval for the active benchmark. This is the ONLY way
to access the benchmark's knowledge base — always use it to gather evidence
before answering corpus/document questions; do not rely on web search or prior
knowledge for them.

Tools:
- retrieval_search(query, top_k=10, use_rerank=True): semantic search over the
  benchmark corpus; returns ranked passages with titles and scores.
- calculate(expression): evaluate a math expression safely (e.g. ratios).

Use cases: financial-report QA, multi-hop document QA, factual lookup grounded
in the provided corpus.
"""


def _normalize_openrouter_model(model: str) -> str:
    """AutoMAS routes through OpenRouter, whose model ids are namespaced
    (``provider/model``). marlib passes bare ids like ``gpt-4o-mini``; prefix an
    ``openai/`` namespace when none is present so OpenRouter accepts it."""
    if not model or "/" in model:
        return model
    return f"openai/{model}"


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
        self._framework_ready = False

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

    def _set_llm_env(self) -> None:
        """Populate the env AutoMAS reads. Must run *before* AutoMAS is imported:
        its default model ids (``AGENT_NODE_MODEL`` / ``DEFAULT_META_MODEL``) are
        captured at module-import time, and ``AgentNode``/``BaseMetaAgent`` require
        ``OPENROUTER_API_KEY`` in the environment (AutoMAS routes via OpenRouter)."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            # The repo already routes its OpenAI-compatible calls through
            # OpenRouter (OPENAI_BASE_URL=https://openrouter.ai/api/v1), so the
            # existing OPENAI_API_KEY *is* an OpenRouter key — reuse it rather
            # than demanding a second secret.
            base = os.environ.get("OPENAI_BASE_URL", "")
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and "openrouter" in base:
                os.environ["OPENROUTER_API_KEY"] = openai_key
            else:
                raise RuntimeError(
                    "AutoMAS routes through OpenRouter but OPENROUTER_API_KEY is "
                    "not set (and OPENAI_API_KEY is not an OpenRouter key). Set "
                    "OPENROUTER_API_KEY in the environment (e.g. .env) before "
                    "running the 'automas' system."
                )
        model = _normalize_openrouter_model(self._model)
        os.environ.setdefault("AGENT_NODE_MODEL", model)
        os.environ.setdefault("DEFAULT_META_MODEL", model)

    def _setup_mcp_registry(self) -> None:
        """Make marlib's retrieval MCP server the *only* server AutoMAS can use.

        AutoMAS ships servers for web search (SearXNG), browser, e2b sandbox,
        etc. On a corpus-grounded benchmark those are wrong (and unconfigured —
        the meta-agent kept routing nodes to a non-running SearXNG), so we clear
        the registry and leave only ``retrieval``. AutoMAS launches MCP servers
        as stdio subprocesses and forwards ``os.environ`` to them, so MARLIB_*
        env (incl. ``MARLIB_DOCIDS_FILE``) reaches ``marlib.mcp_server`` for
        doc-id tracking. ``marlib.mcp_server.__main__`` already silences its
        stdout console so the JSON-RPC stream stays clean."""
        import marlib.mcp_server
        from automas.mcp import external_descriptions
        from automas.mcp import registry as automas_registry
        from automas.mcp.server_config import MCPServerConfig

        server_path = marlib.mcp_server.__file__

        # command=sys.executable + a real script path passes AutoMAS'
        # validate_server_config (it treats args[0] as a path that must exist;
        # `-m module` would fail that check).
        retrieval_cfg = MCPServerConfig(
            command=sys.executable,
            args=(server_path,),
            timeout=30,
            module_path=None,
        )
        automas_registry.MCP_SERVERS.clear()
        automas_registry.MCP_SERVERS["retrieval"] = retrieval_cfg
        external_descriptions.EXTERNAL_SERVER_DESCRIPTIONS.clear()
        external_descriptions.EXTERNAL_SERVER_DESCRIPTIONS["retrieval"] = (
            _RETRIEVAL_DESCRIPTION
        )

    def _init_framework(self) -> None:
        # Order matters: env before any AutoMAS import, then register the MCP
        # server (which imports AutoMAS submodules).
        self._set_llm_env()
        self._setup_mcp_registry()
        self._framework_ready = True

    def generate_system(self, question: str) -> str:
        return "AutoMAS auto-generated multi-agent pipeline (per-task)"

    async def _ensure_structure(self, question: str) -> tuple[Any, Any]:
        from automas.meta_agents import GraphGenerator, PoolGenerator

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
        from automas.pipeline import PipelineBuilder

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
            import traceback

            traceback.print_exc()
            answer = ""

        if docids_file.exists():
            docids_file.unlink()

        return answer, tracker.to_question_log(answer)

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        if result is None:
            return ""

        if isinstance(result, dict):
            for key in ("answer", "output", "final_output", "result"):
                value = result.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return str(result).strip()

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
