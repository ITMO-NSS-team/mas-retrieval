"""Meta-agent prompts for MAS-Zero (RAG-adapted, JSON).

Ported from MAS-Zero's prompts/plan/propose.py and reflect_after_eval.py:
  - PROPOSE: task decomposition + wiring sub-MAS from archive blocks
  - REFLECT_AFTER_EVAL: MAS-Feedback reflexion (solvability / completeness /
    fitness) used to refine the architecture across meta-iterations

`Info` and `LLMAgentBase` are injected into the exec() globals by the adapter,
so generated forward() bodies reference them directly without any import.
"""

from __future__ import annotations

import json

# Meta-model response schema (propose + reflexion both use these fields).
PROPOSE_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "Reply EXACTLY with the following JSON format.\n"
    '{"reflection": "Your reflection (if applicable).", "thought": "Your thought.", '
    '"name": "Your name.", "code": "Your code."}\n'
    "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a "
    "well-formed JSON object!"
)

# Condensed, RAG-adapted version of MAS-Zero's util_code + wrong_implementation
# guidance — the rules the generated forward() must obey.
_CODE_GUIDANCE = """# Coding interface and rules
Your `code` is the body of a single function `def forward(self, taskInfo):` that
returns `self.make_final_answer(thinking, answer, sub_tasks, agents)`.

Available in scope (DO NOT import these):
- `Info(name, author, content, prompt, sub_tasks, agents, iteration_idx)` namedtuple.
- `LLMAgentBase(output_fields, agent_name, role=..., model=self.node_model, temperature=..., usage_callback=self._usage_callback)`.
  Calling an agent returns a list of `Info` (one per output field); e.g.
  `thinking, answer = agent([taskInfo, ...], instruction, is_sub_task=True)`.
- Tools on `self`: `self.retrieve(query, top_k=20)`, `self.rerank(query, top_k=10)`,
  `self.calculate(expression)` — each returns a formatted string. Wrap a tool
  result in an Info to pass it to an agent, e.g.
  `ctx = Info('retrieved_context', 'retriever', self.rerank(q, 10), None, None, None, -1)`.
- Config on `self`: `self.node_model`, `self.cot_instruction`, `self.max_round`,
  `self.max_sc`, `self.debate_role`, `self._usage_callback`.

Hard rules:
1. ACTUALLY IMPLEMENT each block you use (write the for-loops for COT_SC / Debate
   / Reflexion). Naming an agent "debate" does not implement debate.
2. You may NOT call archive blocks by name — re-implement them from their `code`.
   You may only change how blocks connect and their settings (instruction, role,
   temperature). Do not invent new block types.
3. When an agent handles a decomposed sub-task, pass `is_sub_task=True`.
4. Never create an `Info` from another agent's returned `Info`; pass returned
   Infos directly into the next agent's input list. Never print. Never return an
   error string — always return your best `make_final_answer(...)`.
5. Track bookkeeping for MAS-Feedback:
   - init `sub_tasks = []` and `agents = []` at the top of forward.
   - append each sub-task's outcome to `sub_tasks`, e.g.
     `sub_tasks.append(f"Sub-task 1 output: thinking - {t.content}; answer - {a.content}")`.
   - append each agent's outcome to `agents`, e.g.
     `agents.append(f"CoT agent {agent.id} for sub-task 1: thinking {t.content}; answer {a.content}")`.
   - end with `return self.make_final_answer(thinking, answer, sub_tasks, agents)`.
6. Each sub-task instruction must start with its ID and dependencies, e.g.
   "Sub-task 3: Based on the outputs of sub-task 1 and sub-task 2, ...". Pass the
   prerequisite sub-tasks' thinking/answer Infos into the agent's input list. Do
   NOT leak or hard-code the final answer in a sub-task instruction.
"""

EXAMPLE = {
    "reflection": "N/A for the initial round.",
    "thought": (
        "**Decomposition:** sub-task 1 = identify the entities the question asks "
        "about and retrieve evidence for each; sub-task 2 = based on sub-task 1, "
        "reason over the combined evidence to produce the final answer. Each is "
        "solvable by a retrieval-grounded CoT block and together they yield the "
        "answer.\n\n"
        "**Overall Architecture:** COT (address sub-task 1) -> COT (address "
        "sub-task 2)."
    ),
    "name": "Decomposed-RAG-CoT",
    "code": (
        "def forward(self, taskInfo):\n"
        "    sub_tasks = []\n"
        "    agents = []\n"
        "    q = taskInfo.content\n"
        "    ctx1 = Info('retrieved_context', 'retriever', self.rerank(q, 10) or self.retrieve(q, 20), None, None, None, -1)\n"
        "    a1 = LLMAgentBase(['thinking', 'answer'], 'Evidence Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)\n"
        "    t1, ans1 = a1([taskInfo, ctx1], 'Sub-task 1: Identify the entities the question concerns and summarise the retrieved evidence about each.', is_sub_task=True)\n"
        "    agents.append(f'Evidence agent {a1.id} for sub-task 1: {t1.content}; answer {ans1.content}')\n"
        "    sub_tasks.append(f'Sub-task 1 output: thinking - {t1.content}; answer - {ans1.content}')\n"
        "    a2 = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)\n"
        "    t2, ans2 = a2([taskInfo, t1, ans1], 'Sub-task 2: Based on the output of sub-task 1, reason over the evidence and give the final answer.', is_sub_task=True)\n"
        "    agents.append(f'Reasoning agent {a2.id} for sub-task 2: {t2.content}; answer {ans2.content}')\n"
        "    sub_tasks.append(f'Sub-task 2 output: thinking - {t2.content}; answer - {ans2.content}')\n"
        "    return self.make_final_answer(t2, ans2, sub_tasks, agents)\n"
    ),
}


def _benchmark_context(
    description: str | None,
    sample_questions: list[str] | None,
) -> str:
    if not description:
        return ""
    parts = [f"# Benchmark Context\n{description}\n"]
    if sample_questions:
        examples = "\n".join(f"- {q}" for q in sample_questions[:3])
        parts.append(
            f"\nExample questions from this benchmark:\n{examples}\n"
            "Design for this TYPE of question, not just these examples.\n"
        )
    return "\n".join(parts) + "\n"


def build_propose_prompt(
    archive: list[dict],
    question: str,
    benchmark_description: str | None = None,
    sample_questions: list[str] | None = None,
) -> str:
    """Build the MAS-Design (decompose + wire sub-MAS) prompt for one question."""
    archive_str = "[" + ",\n".join(json.dumps(sol) for sol in archive) + "]"

    return f"""# Overview
You are an expert researcher designing agentic systems for retrieval-augmented
question answering. Given an archive of architectures (blocks) and a question,
your objective has two parts.

(1) **Task decomposition.** Decompose the question into sub-task 1, sub-task 2,
..., sub-task n, each easy enough that a single archive block can solve it.
- Do NOT solve the task yourself and do NOT leak the answer in any sub-task
  instruction (a short-cut like "output exactly ..." is leakage).
- Each sub-task instruction must include its ID and dependencies, e.g.
  "Sub-task 2: Based on the outputs of sub-task 1, ...".
- The answer to the LAST sub-task must equal the answer to the original question.
- Justify how the sub-tasks compose into the final answer.

(2) **Wire the sub-MAS.** Assign archive blocks (as nodes) to address each
sub-task and connect them into a multi-layered network using '->'. Example:
"COT (address sub-task 1) -> LLM_debate (address sub-task 2)". Do NOT assign one
block to do all sub-tasks. Only change connections/settings — reuse the archive
blocks' implementations as-is (re-implemented in code; you cannot call them by
name). Retrieval (self.retrieve / self.rerank) should be used to ground the
sub-tasks in the corpus.

If your previous attempts in the archive have fitness 0, the sub-tasks were too
hard for their blocks — decompose further into easier sub-tasks.

{_CODE_GUIDANCE}

# Discovered architecture archive
The fitness value (when present) is the self-assessed correctness on this
question; your GOAL is to maximize it.

{archive_str}

# Output format
Return JSON with keys "reflection", "thought", "name", "code". In "thought"
include a **Decomposition** section (the final sub-tasks + justification) and an
**Overall Architecture** section (the block connections using '->'). Here is an
example of the expected format:

{json.dumps(EXAMPLE)}

{_benchmark_context(benchmark_description, sample_questions)}# Your task
Below is the question to solve:

{question}
"""


# MAS-Feedback reflexion prompt (ported & condensed from reflect_after_eval.py).
# `{last_round}` / `{prev_round}` are filled per iteration by the adapter.
REFLECT_AFTER_EVAL_PROMPT = """Round {last_round} feedback. Carefully review the \
proposed architecture ("code"), the per-sub-task answers ("sub_tasks"), the \
per-agent outputs ("agents"), the final response ("final_response"), the fitness \
("fitness", the self-assessed correctness on this question), and the running \
"memory" (previous final answers and their fitness) across the whole history. \
Reflect on:

1. **Solvable**: Is each sub-task solvable by its block? Check each sub-task's
   answer. If an answer contains [TOO_HARD], that sub-task must be decomposed
   further — follow the 'Suggestion:' after the mark. If a sub-task answer looks
   wrong, decide whether (a) the sub-task is still too hard (decompose further,
   keeping every sub-task specific, self-contained, and connected to its
   prerequisites) or (b) a block/agent is the wrong tool (swap the block or
   adjust its instruction/role/temperature). Justify (a) and/or (b).

2. **Completeness**: Do the sub-tasks together carry ALL information the original
   question needs? No critical fact may be missing from every sub-task. Ensure
   each sub-task receives its prerequisites' outputs.

3. **Fitness**: Your goal is to raise fitness. If it is low the final answer is
   likely wrong. Read "memory" and explicitly steer the last sub-task away from
   previously wrong answers (e.g. "It is known that <wrong answers with fitness
   0> are not correct.").

Then improve the implementation or propose a revised architecture.

Add these keys to your previous JSON answer:
- "reflection": your analysis of Solvable / Completeness / Fitness (which
  sub-tasks are wrong? which agent/block malfunctioned?) and concrete fixes.
- "thought": the revised decomposition and architecture (same format as before:
  **Decomposition** + **Overall Architecture** with '->'). Do not leak answers.
- "name": a name for the revised architecture (no words like "new"/"improved").
- "code": the COMPLETE updated forward() implementing every improvement. Obey all
  the coding rules from Round 0. Return only self.make_final_answer(...). Avoid
  syntax errors (e.g. if a string contains ', wrap it in ")."""
