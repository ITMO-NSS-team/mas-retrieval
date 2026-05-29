"""Pydantic-graph nodes for the MA-RAG plan-execute-summarize pipeline."""

import logging
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from marlib.adapters.ma_rag.agents import (
    QAAnswer,
    answer_agent,
    extract_agent,
    plan_agent,
    step_definer_agent,
    summary_agent,
)
from marlib.adapters.ma_rag.prompts import ANSWER_AGGREGATE_SYSTEM_PROMPT
from marlib.adapters.ma_rag.state import MARagDeps, MARagState, StepResult
from marlib.adapters.tools import do_rerank, do_retrieve

logger = logging.getLogger(__name__)


def _log_usage(deps: MARagDeps, result) -> None:
    """Log token usage from a pydantic-ai RunResult."""
    usage = result.usage()
    deps.tracker.log_llm_call(
        model=deps.model_name,
        prompt_tokens=usage.input_tokens or 0,
        completion_tokens=usage.output_tokens or 0,
        latency_ms=0,
        function_calls=0,
    )


# ── CreatePlan ───────────────────────────────────────────────


@dataclass
class CreatePlan(BaseNode[MARagState, MARagDeps, str]):
    async def run(
        self, ctx: GraphRunContext[MARagState, MARagDeps],
    ) -> Union["ExecuteStep", End[str]]:
        question = ctx.state.question
        logger.info("Creating plan for: %s", question)

        result = await plan_agent.run(
            f"Question: {question}",
            model=ctx.deps.model,
        )
        _log_usage(ctx.deps, result)

        ctx.state.plan = result.output.steps
        logger.info("Plan (%d steps): %s", len(ctx.state.plan), ctx.state.plan)
        return ExecuteStep()


# ── ExecuteStep ──────────────────────────────────────────────


@dataclass
class ExecuteStep(BaseNode[MARagState, MARagDeps, str]):
    async def run(
        self, ctx: GraphRunContext[MARagState, MARagDeps],
    ) -> Union["ExecuteStep", "Summarize", End[str]]:
        state = ctx.state
        deps = ctx.deps
        idx = state.current_step_index

        # Stop conditions: all steps done, safety cap, or last step failed
        if idx >= len(state.plan) or idx >= deps.max_steps:
            return Summarize()
        if state.step_results and not state.step_results[-1].success:
            return Summarize()

        step_text = state.plan[idx]
        logger.info("Step %d/%d: %s", idx + 1, len(state.plan), step_text)

        # Build memory string from previous results
        memory = ""
        for i, sr in enumerate(state.step_results):
            memory += f"Task: {state.plan[i]}\nAnswer: {sr.answer}\n\n"

        plan_str = "[" + ", ".join(f'"{s}"' for s in state.plan) + "]"
        definer_prompt = (
            f"Plan: {plan_str}\n"
            f"Current step: {step_text}\n"
            f"Results of finished steps:\n{memory if memory else 'None yet'}"
        )
        definer_result = await step_definer_agent.run(
            definer_prompt, model=deps.model,
        )
        _log_usage(deps, definer_result)

        task = definer_result.output
        task_type = "aggregate" if "aggregate" in task.type.lower() else "qa"
        query = task.task
        logger.info("  type=%s, query=%s", task_type, query)

        if task_type == "qa":
            qa_answer, doc_ids = await self._execute_qa(deps, query)
        else:
            qa_answer = await self._execute_aggregate(deps, query)
            doc_ids = []

        success = qa_answer.success.lower().startswith("y")
        state.step_results.append(
            StepResult(
                step_description=step_text,
                task_type=task_type,
                query=query,
                answer=qa_answer.answer,
                success=success,
                confidence=qa_answer.rating,
                doc_ids=doc_ids,
            )
        )
        state.current_step_index += 1
        return ExecuteStep()

    async def _execute_qa(
        self, deps: MARagDeps, query: str,
    ) -> tuple[QAAnswer, list[str]]:
        """Retrieve, rerank, extract per-doc notes, then answer."""
        with deps.tracker.track_tool("retrieve", query, deps.top_k_retrieve) as rid:
            docs, _ = do_retrieve(deps.retriever, query, deps.top_k_retrieve)
            deps._last_retrieved = docs
            rid.extend([d.doc_id for d in docs])

        with deps.tracker.track_tool("rerank", query, deps.top_k_rerank) as rid:
            docs, _ = do_rerank(
                deps.retriever, query, deps._last_retrieved, deps.top_k_rerank,
            )
            deps._last_retrieved = docs
            rid.extend([d.doc_id for d in docs])

        doc_ids = [d.doc_id for d in docs]

        # Extract step: one LLM call per document (matches original MA-RAG)
        notes = []
        for doc in docs:
            extract_result = await extract_agent.run(
                f"Passage:\n###\n{doc.text}\n###\n\nQuery: {query}?",
                model=deps.model,
            )
            _log_usage(deps, extract_result)
            notes.append(extract_result.output)

        # Build context from extracted notes (original format: doc_{id}: [note])
        context_parts = []
        for doc, note in zip(docs, notes):
            context_parts.append(f"doc_{doc.doc_id}: [{note}]")
        context = "\n\n".join(context_parts)

        user_msg = (
            f"Retrieved documents:\n{context}\n\n"
            f"Question: {query}"
        )
        result = await answer_agent.run(user_msg, model=deps.model)
        _log_usage(deps, result)
        return result.output, doc_ids

    async def _execute_aggregate(
        self, deps: MARagDeps, query: str,
    ) -> QAAnswer:
        """Answer an aggregate question (no retrieval)."""
        result = await answer_agent.run(
            query,
            model=deps.model,
            instructions=ANSWER_AGGREGATE_SYSTEM_PROMPT,
        )
        _log_usage(deps, result)
        return result.output


# ── Summarize ────────────────────────────────────────────────


@dataclass
class Summarize(BaseNode[MARagState, MARagDeps, str]):
    async def run(
        self, ctx: GraphRunContext[MARagState, MARagDeps],
    ) -> End[str]:
        state = ctx.state
        deps = ctx.deps

        plan_str = "[" + ", ".join(f'"{s}"' for s in state.plan) + "]"
        memory = ""
        for i, sr in enumerate(state.step_results):
            memory += (
                f"Task: {state.plan[i]}\n"
                f"Question: {sr.query}\n"
                f"Answer: {sr.answer}\n"
                f"Confident score: {sr.confidence}\n\n"
            )

        user_msg = (
            f"Original Question: {state.question}\n"
            f"Plan: {plan_str}\n"
            f"Output of steps:\n{memory}\n"
            f"Original Question: {state.question}"
        )

        result = await summary_agent.run(user_msg, model=deps.model)
        _log_usage(deps, result)

        answer = result.output.answer
        state.final_answer = answer
        logger.info("Final answer: %s", answer)
        return End(answer)


# ── Graph instance ───────────────────────────────────────────

ma_rag_graph = Graph(
    nodes=[CreatePlan, ExecuteStep, Summarize],
)
