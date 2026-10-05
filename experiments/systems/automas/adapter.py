from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
from uuid import uuid4
from contextlib import AsyncExitStack
from typing import Any

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.construction import task_description
from marlib.provenance import file_hash
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker
from marlib.tracing.resources import BudgetExhausted, ResourceLimits, ResourceSession, tracked_async_client, run_with_deadline

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
- retrieve(query, top_k=20): dense search over the shared benchmark corpus.
- rerank(query, top_k=10): re-score this agent's most recent retrieve results.
  Always call after retrieve. Use model/tool calls economically.
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
    supports_resource_limits = True
    supported_generation_modes = ("one_time", "per_task")
    def __init__(
        self, retriever: Any, model: str = "gpt-4o-mini", **kwargs: Any
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "per_task"

        if self._generation_mode not in self.supported_generation_modes:
            raise ValueError("Invalid MetaMAS generation mode")
        scope = "execution" if self._generation_mode == "one_time" else "full_answer"
        self._limits = ResourceLimits(**{"scope": scope, "max_requests": 50, **self._config.get("resource_limits", {})})
        if self._limits.scope != scope:
            raise ValueError(f"MetaMAS {self._generation_mode} requires {scope} scope")
        self._construction_limits = ResourceLimits(**{"scope": "construction", "max_requests": 10,
            **self._config.get("construction_limits", {})})
        if self._construction_limits.scope != "construction":
            raise ValueError("construction_limits requires construction scope")
        self._meta_model = _normalize_openrouter_model(self._config.get("meta_model", self._model))
        self._temperature = self._config.get("temperature", 0.1)
        self._generator_temperature = self._config.get("generator_temperature", 0.3)
        for value in (self._temperature, self._generator_temperature):
            if not isinstance(value, (int, float)) or not 0 <= value <= 2:
                raise ValueError("Invalid temperature")
        self._on_benchmark_change()

    def _on_benchmark_change(self) -> None:
        self._cached_pool = None
        self._cached_graph = None
        self._construction_attempted = False
        self._construction_session = None
        self._construction_summary = None
        self._construction_path = None

    def _build_task_description(self) -> str:
        """Build a generic task description from benchmark context for one_time mode."""
        return task_description(self._benchmark_description, self._sample_questions)

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

    def generate_system(self, question: str) -> str:
        return f"AutoMAS accounted workflow ({self._generation_mode})"

    def effective_config(self):
        from importlib.util import find_spec
        spec = find_spec("automas")
        external = Path(spec.origin).parent if spec and spec.origin else None
        return {**super().effective_config(), "variant": "automas_accounted_v1",
                "framework_source_sha256": {str(p.relative_to(external)): file_hash(p)
                                             for p in sorted(external.rglob("*.py"))} if external else {},
                "model": _normalize_openrouter_model(self._model), "meta_model": self._meta_model,
                "temperature": self._temperature, "generator_temperature": self._generator_temperature,
                "resource_limits": asdict(self._limits), "construction_limits": asdict(self._construction_limits),
                "tools": ["retrieve", "rerank", "calculate"], "validation_retries": 3,
                "stop_policy": "empty_answer_without_finalization",
                "construction_cost_attribution": "first_question_for_CL_all_questions_for_QL"}

    def set_run_context(self, **context):
        if context != self._run_context:
            self._on_benchmark_change()
        super().set_run_context(**context)

    async def _model_for(self, stack, session, phase, model):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        client = await stack.enter_async_context(tracked_async_client(
            session, phase, base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]))
        return OpenAIChatModel(model, provider=OpenRouterProvider(openai_client=client),
                               settings={"temperature": self._generator_temperature if phase == "construction" else self._temperature})

    def _toolsets(self, names, session):
        from .runtime import BudgetedMCPServer, tool_hook
        if any(name != "retrieval" for name in names):
            raise ValueError("Workflow requested a tool server outside the shared corpus")
        env = {k: v for k, v in os.environ.items()
               if k.startswith("MARLIB_") or k in {"PATH", "PYTHONPATH", "HOME", "TMPDIR"}}
        env["MARLIB_PRIMITIVE_TOOLS"] = "1"
        env.pop("MARLIB_DOCIDS_FILE", None)
        server = Path(__file__).resolve().parents[3] / "src/marlib/mcp_server.py"
        return [BudgetedMCPServer(sys.executable, args=[str(server)], env=env, timeout=60,
                                 process_tool_call=tool_hook(session), max_retries=3)
                for _ in names]

    async def _ensure_structure(self, question, session, stack):
        from automas.meta_agents import GraphGenerator, PoolGenerator
        from pydantic_ai.usage import UsageLimits
        if self._generation_mode == "one_time" and self._construction_attempted:
            if self._cached_pool is None:
                raise RuntimeError("CL workflow construction failed; this repeat cannot execute")
            return self._cached_pool, self._cached_graph
        self._construction_attempted = True
        model = await self._model_for(stack, session, "construction", self._meta_model)
        # Preserve upstream schema, prompts and validation; inject the tracked
        # model at construction, before any generator can create a private client.
        class TrackedGenerator:
            def _create_model(self):
                return model
            async def _run_agent(self, prompt):
                result = await self.agent.run(prompt, usage_limits=UsageLimits(request_limit=None))
                self._usage = result.usage()
                return result.output
        class Pool(TrackedGenerator, PoolGenerator):
            pass
        class Graph(TrackedGenerator, GraphGenerator):
            pass
        pool_gen = Pool(model=self._meta_model, temperature=self._generator_temperature)
        graph_gen = Graph(model=self._meta_model, temperature=self._generator_temperature)
        task = self._build_task_description() if self._generation_mode == "one_time" else question
        pool = await pool_gen.create_pool(task)
        proposed_models = {node.id: node.model for node in pool}
        for node in pool:
            node.model = _normalize_openrouter_model(self._model)
            if any(name != "retrieval" for name in node.mcp_tools):
                raise ValueError("Generated pool requested an unavailable tool server")
        self._construction_path.write_text(json.dumps({"task": task, "pool": pool.full_agents_data,
            "graph": None, "proposed_models": proposed_models, "model": self._meta_model}, indent=2))
        graph = await graph_gen.create_graph(pool, task)
        self._construction_path.write_text(json.dumps({"task": task, "pool": pool.full_agents_data,
            "graph": graph, "proposed_models": proposed_models,
            "model": self._meta_model}, ensure_ascii=False, indent=2))
        if self._generation_mode == "one_time":
            self._cached_pool, self._cached_graph = pool, graph
        return pool, graph

    async def _execute_async(self, question, tracker, artifacts):
        from automas.pipeline import PipelineBuilder
        from pydantic_ai import Agent
        from pydantic_ai.usage import UsageLimits
        async with AsyncExitStack() as stack:
            combined = self._limits.scope == "full_answer"
            needs_construction = self._generation_mode == "per_task" or not self._construction_attempted
            if combined:
                self._execution_session = ResourceSession(tracker, self._limits, artifacts / "answer.jsonl",
                                                          coverage="automas_chat_completions_and_mcp_primitives")
            if needs_construction:
                self._construction_session = self._execution_session if combined else ResourceSession(
                    tracker, self._construction_limits, artifacts / "construction.jsonl",
                    coverage="automas_generators_chat_completions")
                self._construction_path = artifacts / "workflow.json"
            if needs_construction:
                construction_error = None
                try:
                    pool, graph = await run_with_deadline(self._construction_session,
                        self._ensure_structure(question, self._construction_session, stack))
                except BaseException as exc:
                    construction_error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    if not combined:
                        # Freeze before execution starts, including failed generation.
                        self._construction_summary = self._construction_session.summary()
                        self._construction_session.write({"kind": "construction_end",
                            "error": construction_error, "resources": self._construction_summary})
            else:
                pool, graph = await self._ensure_structure(question, self._construction_session, stack)
            if not combined:
                self._execution_session = ResourceSession(tracker, self._limits, artifacts / "execution.jsonl",
                                                          coverage="automas_chat_completions_and_mcp_primitives")
            session = self._execution_session
            tracker.tool_event_sink = session.write
            model = await self._model_for(stack, session, "answer_execution", _normalize_openrouter_model(self._model))
            pipeline = PipelineBuilder().create_from_pool(pool, {k: list(v) for k,v in graph.items()}).build()
            self._pipeline = pipeline
            for node in pipeline.execution_order:
                agent = Agent(name=node.name, model=model, instructions=node.instructions,
                              toolsets=self._toolsets(node.mcp_tools, session), retries=3,
                              model_settings={"temperature": self._temperature})
                class Runner:
                    def __init__(self, agent):
                        self.agent = agent
                    async def run(self, value):
                        return await self.agent.run(value, usage_limits=UsageLimits(request_limit=None))
                runner = Runner(agent)
                node.build_agent = lambda runner=runner: runner
            result = await run_with_deadline(session, pipeline.ainvoke(question))
            if session.stop_reason:
                raise BudgetExhausted(session.stop_reason)
            return self._answer_from_pipeline(pipeline, result)

    def execute(self, question_id, question, gold_answer):
        tracker = TokenTracker(question_id, question, "")
        root = Path(self._run_context.get("artifact_dir", "logs/automas"))
        artifacts = root / f"{sha256(question_id.encode()).hexdigest()[:12]}_{uuid4().hex}"
        artifacts.mkdir(parents=True, exist_ok=False)
        self._execution_session = None
        self._pipeline = None
        if self._generation_mode == "per_task":
            self._on_benchmark_change()
        # A cached construction summary is frozen at the end of its first attempt.
        charged_here = self._generation_mode == "per_task" or not self._construction_attempted
        answer = ""
        saved_registry = None
        try:
            self._set_llm_env()
            from automas.mcp import registry, external_descriptions
            saved_registry = (dict(registry.MCP_SERVERS), dict(external_descriptions.EXTERNAL_SERVER_DESCRIPTIONS))
            self._setup_mcp_registry()
            answer = asyncio.run(self._execute_async(question, tracker, artifacts))
            if not answer.strip():
                raise ValueError("Workflow returned an empty answer")
        except Exception as exc:
            tracker.set_error(f"{type(exc).__name__}: {exc}")
        finally:
            if saved_registry is not None:
                registry.MCP_SERVERS.clear()
                registry.MCP_SERVERS.update(saved_registry[0])
                external_descriptions.EXTERNAL_SERVER_DESCRIPTIONS.clear()
                external_descriptions.EXTERNAL_SERVER_DESCRIPTIONS.update(saved_registry[1])
        sessions = [s for s in (self._construction_session if charged_here else None, self._execution_session) if s]
        stopped = next((s.stop_reason for s in sessions if s.stop_reason), None)
        log = tracker.to_question_log(answer)
        log.gold_answer = gold_answer
        log.failure_kind = "budget_exhausted" if stopped else ("unknown" if log.error else None)
        if stopped:
            log.status = "budget_exhausted"
        log.resource_summary = {
            "execution": self._execution_session.summary() if self._execution_session else None,
            "construction_reference": self._construction_summary,
            "construction_charged_here": charged_here,
            "budget_scope": self._limits.scope,
            "workflow_nodes": len(self._pipeline.execution_order) if self._pipeline else None,
        }
        for key, session in (("construction", self._construction_session), ("execution", self._execution_session)):
            if session:
                log.artifact_paths[key + "_events"] = str(session.journal)
        if self._construction_path and self._construction_path.exists():
            log.artifact_paths["workflow"] = str(self._construction_path)
        trace = getattr(self._pipeline, "_trace", None)
        if trace is not None:
            trace_path = artifacts / "pipeline_trace.json"
            trace_path.write_text(trace.model_dump_json(indent=2))
            log.artifact_paths["pipeline_trace"] = str(trace_path)
        for session in set(sessions):
            session.write({"kind": "question_end", "status": log.status, "error": log.error})
        (artifacts / "question.json").write_text(log.model_dump_json(indent=2))
        return answer, log

    # A generated workflow may end in a stage that reviews the answer instead of
    # producing one. AutoMAS' pipeline returns the *last* node's output
    # (``pipeline.ainvoke``), so for such workflows the reviewer's verdict, not
    # the answer, would be scored. Node names matching this pattern are treated
    # as non-answering and skipped when reading the answer off the pipeline.
    _NON_ANSWERING_NODE = re.compile(
        r"quality|assess|verif|valid|critic|review|evaluat|judge", re.IGNORECASE
    )

    @classmethod
    def _answer_from_pipeline(cls, pipeline: Any, result: Any) -> str:
        """Read the final answer off an executed pipeline.

        Walks the execution order backwards and returns the last node that
        actually answers, so a trailing review/verification stage does not
        replace the answer with its own verdict. Falls back to the pipeline's
        own return value when no such node is found, when the pipeline does not
        expose its per-node outputs, or when every node looks non-answering.

        This is a no-op for single-node workflows (the one node is both the last
        and the only candidate), so it cannot change results for runs whose
        generated workflow has one agent.
        """
        fallback = cls._extract_answer(result)

        order = getattr(pipeline, "execution_order", None)
        session = getattr(pipeline, "node_session", None)
        executions = getattr(session, "node_executions", None)
        if not order or not executions:
            return fallback

        for node in reversed(list(order)):
            if cls._NON_ANSWERING_NODE.search(getattr(node, "name", "") or ""):
                continue
            execution = executions.get(getattr(node, "id", None))
            if execution is None:
                continue
            answer = cls._extract_answer(getattr(execution, "output", None))
            if answer:
                return answer

        return fallback

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
