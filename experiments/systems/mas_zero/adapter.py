"""MAS-Zero adapter for RAG benchmark evaluation.

Uses MAS-Zero's meta-agent concept (LLM designs agent architectures from
building blocks) with RAG tools, without importing from the MAS-Zero directory.

Supports 'one_time' (generate once, reuse) and 'per_task' modes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import backoff
import openai

from marlib.adapters.base import AbstractAdapter, register
from marlib.adapters.tools import do_calculate, do_rerank, do_retrieve
from marlib.log import logger
from marlib.retriever.core import Document, Retriever
from marlib.tracing.schemas import QuestionLog
from marlib.tracing.tracker import TokenTracker

from .blocks import RAG_BLOCKS
from .core import (
    ANSWER_PATTERN,
    AgentSystem,
    Info,
    LLMAgentBase,
)
from .prompts import SYSTEM_PROMPT, build_meta_prompt
from .tracing import AgentTrace, MASZeroTrace

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_COT_INSTRUCTION = (
    "Please think step by step and provide your answer. "
    "Think carefully about the question and the retrieved context."
)
_DEFAULT_DEBATE_ROLES = [
    "an analytical researcher",
    "a critical reviewer",
    "a creative problem solver",
]


@register("mas_zero")
class MASZeroAdapter(AbstractAdapter):
    """MAS-Zero meta-agent adapter for RAG benchmarks.

    The meta-model generates an architecture (forward function) that combines
    reasoning blocks with retrieval tools. The architecture is then executed
    per-question with access to retrieve(), rerank(), and calculate().
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(retriever, model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "one_time"
        if self._generation_mode not in ("one_time", "per_task"):
            raise ValueError(
                f"Invalid generation_mode '{self._generation_mode}'; "
                "use 'one_time' or 'per_task'"
            )

        self._n_generation: int = self._config.get("n_generation", 1)
        self._meta_model: str = self._config.get("meta_model", self._model)
        self._top_k: int = self._config.get("top_k", 20)
        self._max_round: int = self._config.get("max_round", 2)
        self._max_sc: int = self._config.get("max_sc", 3)
        self._debug_max: int = self._config.get("debug_max", 3)
        self._cot_instruction: str = self._config.get(
            "cot_instruction",
            _DEFAULT_COT_INSTRUCTION,
        )
        self._debate_roles: list[str] = self._config.get(
            "debate_roles",
            _DEFAULT_DEBATE_ROLES,
        )
        if not self._debate_roles:
            logger.warning("debate_roles empty; using defaults")
            self._debate_roles = list(_DEFAULT_DEBATE_ROLES)
        blocks_config = self._config.get(
            "blocks",
            ["RAG_COT", "RAG_REFLEXION", "RAG_DEBATE", "RAG_COT_SC"],
        )
        block_map = {b["name"]: b for b in RAG_BLOCKS}
        block_name_map = {
            "RAG_COT": "RAG-Chain-of-Thought",
            "RAG_REFLEXION": "RAG-Reflexion",
            "RAG_DEBATE": "RAG-LLM-Debate",
            "RAG_COT_SC": "RAG-Self-Consistency-COT",
        }
        self._blocks: list[dict] = []
        for b in blocks_config:
            name = block_name_map.get(b, b)
            if name in block_map:
                self._blocks.append(block_map[name])
        if not self._blocks:
            logger.warning(
                "No blocks matched config %s; using all RAG_BLOCKS", blocks_config
            )
            self._blocks = list(RAG_BLOCKS)

        # Cached architecture (for one_time mode)
        self._cached_system: dict | None = None

        # Tracing
        self._trace_enabled: bool = self._config.get("trace", False) or os.environ.get(
            "MAS_ZERO_TRACE", ""
        ).lower() in ("1", "true", "yes")

        logger.info(
            "MASZeroAdapter: mode=%s, meta_model=%s, model=%s, blocks=%d, trace=%s",
            self._generation_mode,
            self._meta_model,
            self._model,
            len(self._blocks),
            self._trace_enabled,
        )

    def _on_benchmark_change(self) -> None:
        self._cached_system = None

    @property
    def name(self) -> str:
        return f"mas_zero_{self._generation_mode}"

    @property
    def generated_system(self) -> dict | None:
        """Access the full generated system dict (thought, name, code).

        Returns None if no system has been generated yet.
        """
        return self._cached_system

    def generate_system(self, question: str) -> str:
        """Generate a MAS architecture via the meta-model.

        In one_time mode, generates once and caches. In per_task mode,
        generates a fresh architecture for each question.

        Returns:
            Multi-line description including architecture name, reasoning,
            and the generated forward() code.
        """
        if self._generation_mode == "one_time" and self._cached_system is not None:
            return self._format_system_description(self._cached_system)

        question_for_prompt = (
            question if self._generation_mode == "per_task" else None
        )
        archive = list(self._blocks)

        prompt = build_meta_prompt(
            archive,
            question=question_for_prompt,
            benchmark_description=self._benchmark_description,
            sample_questions=(
                self._sample_questions if self._generation_mode == "one_time" else None
            ),
        )
        solution = self._call_meta_model(prompt)

        if solution is not None:
            self._cached_system = solution
            desc = self._format_system_description(solution)
            logger.info("MAS-Zero generated architecture:\n%s", desc)
            return desc

        # Fallback: use first block directly
        logger.warning("Meta-model failed to generate; falling back to first block")
        self._cached_system = (
            self._blocks[0]
            if self._blocks
            else {
                "name": "fallback-cot",
                "code": RAG_BLOCKS[0]["code"],
            }
        )
        return self._format_system_description(self._cached_system)

    @staticmethod
    def _format_system_description(system: dict) -> str:
        """Format a system dict into a readable multi-line description."""
        parts = [f"Architecture: {system.get('name', 'unknown')}"]
        if "thought" in system:
            parts.append(f"Reasoning: {system['thought']}")
        if "code" in system:
            parts.append(f"Code:\n{system['code']}")
        return "\n".join(parts)

    @backoff.on_exception(
        backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=3
    )
    def _call_meta_model(self, prompt: str) -> dict | None:
        """Call the meta-model to generate an architecture."""
        client = openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self._debug_max):
            response = client.chat.completions.create(
                model=self._meta_model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content or ""
            try:
                solution = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "Meta-model returned invalid JSON (attempt %d)", attempt + 1
                )
                continue

            if not all(k in solution for k in ("name", "thought", "code")):
                logger.warning(
                    "Meta-model missing required keys (attempt %d)", attempt + 1
                )
                continue

            if "def forward(self, taskInfo):" not in solution["code"]:
                logger.warning(
                    "Generated code missing forward() signature (attempt %d)",
                    attempt + 1,
                )
                continue

            # Syntax check
            try:
                compile(solution["code"], "<generated>", "exec")
            except SyntaxError as e:
                logger.warning(
                    "Generated code has syntax error (attempt %d): %s", attempt + 1, e
                )
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Syntax error in your code: {e}\n"
                            "Please fix the code and return the corrected version "
                            "in the same JSON format."
                        ),
                    }
                )
                continue

            return solution

        return None

    def _make_tool_closures(
        self,
        tracker: TokenTracker,
    ) -> tuple[Any, Any, Any]:
        """Build tool closures that track calls via TokenTracker."""
        last_retrieved: list[Document] = []
        retriever = self._retriever

        def retrieve_fn(query: str, top_k: int = 20) -> str:
            nonlocal last_retrieved
            with tracker.track_tool("retrieve", query, top_k) as results:
                docs, formatted = do_retrieve(retriever, query, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def rerank_fn(query: str, top_k: int = 10) -> str:
            nonlocal last_retrieved
            with tracker.track_tool("rerank", query, top_k) as results:
                docs, formatted = do_rerank(retriever, query, last_retrieved, top_k)
                last_retrieved = docs
                results.extend([d.doc_id for d in docs])
            return formatted

        def calc_fn(expression: str) -> str:
            with tracker.track_tool("calculate", expression, 0) as results:
                result = do_calculate(expression)
            return result

        return retrieve_fn, rerank_fn, calc_fn

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        if self._generation_mode == "per_task":
            self._cached_system = None

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        # Point A: Init trace
        trace: MASZeroTrace | None = None
        if self._trace_enabled:
            trace = MASZeroTrace(
                question_id=question_id,
                mode=self._generation_mode,
                meta_model=self._meta_model,
                node_model=self._model,
                blocks_offered=[b["name"] for b in self._blocks],
            )

        try:
            # 1. Build tool closures
            retrieve_fn, rerank_fn, calc_fn = self._make_tool_closures(tracker)

            # 2. Usage callback for LLM token tracking
            def usage_callback(prompt_tokens: int, completion_tokens: int) -> None:
                tracker.log_llm_call(
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=0,  # timing handled per-call in core
                )

            # 3. Generate system (architecture)
            self.generate_system(question)
            if self._cached_system is None:
                raise RuntimeError("generate_system() failed to produce a system")

            forward_code = self._cached_system.get("code")
            if not forward_code:
                raise RuntimeError(
                    f"System '{self._cached_system.get('name', '?')}' has no 'code' field"
                )

            logger.info(self._format_system_description(self._cached_system))

            # Point B: Capture architecture details
            if trace is not None:
                trace.architecture_name = self._cached_system.get("name", "")
                trace.architecture_thought = self._cached_system.get("thought", "")
                trace.generated_code = forward_code

            # 4. Build AgentSystem with tools and config
            system = AgentSystem()
            system.node_model = self._model
            system.cot_instruction = self._cot_instruction
            system.max_round = self._max_round
            system.max_sc = self._max_sc
            system.debate_role = self._debate_roles
            system._retrieve_fn = retrieve_fn
            system._rerank_fn = rerank_fn
            system._calc_fn = calc_fn
            system._usage_callback = usage_callback

            # 5. exec() forward function and set on the system
            # Use traced agent class if tracing is enabled
            if trace is not None:
                _collector = trace.agent_calls
                _OrigAgent = LLMAgentBase

                class _TracedAgent(_OrigAgent):
                    def query(
                        self_agent,
                        input_infos,
                        instruction,
                        iteration_idx=-1,
                        is_sub_task=False,
                    ):
                        result = super().query(
                            input_infos,
                            instruction,
                            iteration_idx=iteration_idx,
                            is_sub_task=is_sub_task,
                        )
                        _collector.append(
                            AgentTrace(
                                agent_name=self_agent.agent_name,
                                agent_id=self_agent.id,
                                output_fields=self_agent.output_fields,
                                role=self_agent.role,
                                iteration_idx=iteration_idx,
                                input_summary=instruction[:200],
                                output={
                                    info.name: (info.content or "")[:500]
                                    for info in result
                                },
                            )
                        )
                        return result

                exec_agent_class = _TracedAgent
            else:
                exec_agent_class = LLMAgentBase

            namespace: dict[str, Any] = {}
            exec(
                forward_code,
                {
                    "LLMAgentBase": exec_agent_class,
                    "Info": Info,
                    "__builtins__": __builtins__,
                },
                namespace,
            )

            if "forward" in namespace and callable(namespace["forward"]):
                forward_fn = namespace["forward"]
                bound_name = "forward"
            else:
                func_names = [k for k, v in namespace.items() if callable(v)]
                if not func_names:
                    raise RuntimeError("Generated code did not define any callable")
                logger.warning("No 'forward' in namespace; using '%s'", func_names[0])
                forward_fn = namespace[func_names[0]]
                bound_name = func_names[0]

            import types

            system.forward = types.MethodType(forward_fn, system)

            # Point C: Capture forward binding
            if trace is not None:
                trace.forward_bound_to = bound_name

            # 6. Create taskInfo and run forward
            task_info = Info(
                "task",
                "user",
                question,
                None,
                None,
                None,
                -1,
            )
            result = system.forward(task_info)

            # 7. Extract answer
            answer = ""
            if result and hasattr(result, "content"):
                match = re.search(ANSWER_PATTERN, result.content)
                if match:
                    answer = match.group(1).strip()
                else:
                    # Take the content after last newline or full content
                    answer = result.content.strip()
                    if "\n" in answer:
                        answer = answer.split("\n")[-1].strip()

        except Exception as e:
            logger.error("MAS-Zero execution failed: %s", e)
            tracker.set_error(str(e))
            if trace is not None:
                trace.execution_error = str(e)
            answer = ""

        # Point D: Save trace
        if trace is not None:
            self._save_trace(trace)

        return answer, tracker.to_question_log(answer)

    def _save_trace(self, trace: MASZeroTrace) -> None:
        """Save trace to logs/mas_zero/ as JSON and log summary."""
        logger.debug("MAS-Zero trace:\n%s", trace.summary())
        try:
            trace_dir = os.path.join("logs", "mas_zero")
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"trace_{trace.question_id}.json")
            with open(path, "w") as f:
                f.write(trace.model_dump_json(indent=2))
            logger.info("Trace saved to %s", path)
        except OSError as e:
            logger.warning("Failed to save trace: %s", e)
