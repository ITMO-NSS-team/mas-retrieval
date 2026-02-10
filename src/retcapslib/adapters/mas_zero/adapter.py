"""MAS-Zero adapter for RAG benchmark evaluation.

Uses MAS-Zero's meta-agent concept (LLM designs agent architectures from
building blocks) with RAG tools, without importing from the MAS-Zero directory.

Supports 'shared' (generate once, reuse) and 'per_question' modes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import backoff
import openai

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.adapters.mas_zero.blocks import RAG_BLOCKS
from retcapslib.adapters.mas_zero.core import (
    ANSWER_PATTERN,
    AgentSystem,
    Info,
    LLMAgentBase,
)
from retcapslib.adapters.mas_zero.prompts import SYSTEM_PROMPT, build_meta_prompt
from retcapslib.adapters.tools import do_calculate, do_rerank, do_retrieve
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Document, Retriever

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
            self._generation_mode = "shared"

        self._n_generation: int = self._config.get("n_generation", 1)
        self._meta_model: str = self._config.get("meta_model", self._model)
        self._top_k: int = self._config.get("top_k", 20)
        self._max_round: int = self._config.get("max_round", 2)
        self._max_sc: int = self._config.get("max_sc", 3)
        self._debug_max: int = self._config.get("debug_max", 3)
        self._cot_instruction: str = self._config.get(
            "cot_instruction", _DEFAULT_COT_INSTRUCTION,
        )
        self._debate_roles: list[str] = self._config.get(
            "debate_roles", _DEFAULT_DEBATE_ROLES,
        )
        blocks_config = self._config.get(
            "blocks", ["RAG_COT", "RAG_REFLEXION", "RAG_DEBATE", "RAG_COT_SC"],
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

        # Cached architecture (for shared mode)
        self._cached_system: dict | None = None

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

        In shared mode, generates once and caches. In per_question mode,
        generates a fresh architecture for each question.

        Returns:
            Multi-line description including architecture name, reasoning,
            and the generated forward() code.
        """
        if self._generation_mode == "shared" and self._cached_system is not None:
            return self._format_system_description(self._cached_system)

        question_for_prompt = question if self._generation_mode == "per_question" else None
        archive = list(self._blocks)

        prompt = build_meta_prompt(archive, question=question_for_prompt)
        solution = self._call_meta_model(prompt)

        if solution is not None:
            self._cached_system = solution
            desc = self._format_system_description(solution)
            logger.info("MAS-Zero generated architecture:\n%s", desc)
            return desc

        # Fallback: use first block directly
        logger.warning("Meta-model failed to generate; falling back to first block")
        self._cached_system = self._blocks[0] if self._blocks else {
            "name": "fallback-cot",
            "code": RAG_BLOCKS[0]["code"],
        }
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

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError), max_tries=3)
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
                logger.warning("Meta-model returned invalid JSON (attempt %d)", attempt + 1)
                continue

            if not all(k in solution for k in ("name", "thought", "code")):
                logger.warning("Meta-model missing required keys (attempt %d)", attempt + 1)
                continue

            if "def forward(self, taskInfo):" not in solution["code"]:
                logger.warning("Generated code missing forward() signature (attempt %d)", attempt + 1)
                continue

            # Syntax check
            try:
                compile(solution["code"], "<generated>", "exec")
            except SyntaxError as e:
                logger.warning("Generated code has syntax error (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Syntax error in your code: {e}\n"
                        "Please fix the code and return the corrected version "
                        "in the same JSON format."
                    ),
                })
                continue

            return solution

        return None

    def _make_tool_closures(
        self, tracker: TokenTracker,
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
        if self._generation_mode == "per_question":
            self._cached_system = None

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
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
            assert self._cached_system is not None

            forward_code = self._cached_system["code"]

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
            namespace: dict[str, Any] = {}
            exec(forward_code, {
                "LLMAgentBase": LLMAgentBase,
                "Info": Info,
                "__builtins__": __builtins__,
            }, namespace)

            func_names = [k for k, v in namespace.items() if callable(v)]
            if not func_names:
                raise RuntimeError("Generated code did not define any callable")

            import types
            system.forward = types.MethodType(namespace[func_names[0]], system)

            # 6. Create taskInfo and run forward
            task_info = Info(
                "task", "user", question, None, None, None, -1,
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
            answer = ""

        return answer, tracker.to_question_log(answer)
