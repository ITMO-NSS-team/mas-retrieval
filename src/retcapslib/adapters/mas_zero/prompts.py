"""Meta-agent prompts for MAS-Zero RAG architecture generation.

Adapted from MAS-Zero's prompts/plan/propose.py for RAG tasks.
The meta-model uses these prompts to generate a forward() function
that combines retrieval tools with reasoning blocks.
"""

from __future__ import annotations

import json


SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "Reply EXACTLY with the following JSON format.\n"
    '{"thought": "Your thought.", "name": "Your name.", "code": "Your code."}\n'
    "DO NOT MISS ANY REQUEST FIELDS and ensure that your response "
    "is a well-formed JSON object!"
)

EXAMPLE = {
    "thought": (
        "**Insights:** For RAG question-answering, we need to first retrieve "
        "relevant documents, then reason over them. A chain-of-thought approach "
        "with retrieval gives the LLM grounded context.\n\n"
        "**Overall Architecture:** retrieve -> rerank -> CoT reasoning\n\n"
        "**Implementation:** 1. Retrieve documents using self.retrieve(). "
        "2. Rerank for precision using self.rerank(). "
        "3. Pass context to a CoT agent for step-by-step reasoning."
    ),
    "name": "RAG-Chain-of-Thought",
    "code": (
        "def forward(self, taskInfo):\n"
        "    from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase\n"
        "    question = taskInfo.content\n"
        "    context = self.retrieve(question, top_k=20)\n"
        "    context_reranked = self.rerank(question, top_k=10)\n"
        "    context_info = Info('retrieved_context', 'retriever', context_reranked, None, None, None, -1)\n"
        "    cot_instruction = self.cot_instruction + ' Use the retrieved documents as context.'\n"
        "    cot_agent = LLMAgentBase(['thinking', 'answer'], 'RAG-CoT Agent', model=self.node_model, temperature=0.0, usage_callback=self._usage_callback)\n"
        "    thinking, answer = cot_agent([taskInfo, context_info], cot_instruction)\n"
        "    final_answer = self.make_final_answer(thinking, answer)\n"
        "    return final_answer\n"
    ),
}


def _benchmark_context(
    description: str | None,
    sample_questions: list[str] | None,
) -> str:
    """Build an optional benchmark context section for the meta-prompt."""
    if not description:
        return ""
    parts = [f"# Benchmark Context\n{description}\n"]
    if sample_questions:
        examples = "\n".join(f"- {q}" for q in sample_questions[:3])
        parts.append(
            f"\nExample questions from this benchmark:\n{examples}\n"
            "\nDesign your architecture to handle this TYPE of questions effectively, "
            "not just these specific examples.\n"
        )
    return "\n".join(parts) + "\n"


def build_meta_prompt(
    archive: list[dict],
    question: str | None = None,
    benchmark_description: str | None = None,
    sample_questions: list[str] | None = None,
) -> str:
    """Build the meta-agent prompt for generating RAG architectures.

    Args:
        archive: List of block dicts with {thought, name, code}.
        question: Optional question to include for per_question mode.
        benchmark_description: Description of the benchmark task type.
        sample_questions: Example questions from the benchmark.

    Returns:
        The user prompt string.
    """
    archive_str = ",\n".join([json.dumps(sol) for sol in archive])
    archive_str = f"[{archive_str}]"

    prompt = f"""# Overview
You are an expert machine learning researcher designing agentic RAG systems.
Given a set of RAG architectures in the archive, design a new architecture that
combines retrieval with reasoning to answer questions effectively.

# Available Tools
Your generated forward() function has access to these methods on `self`:

1. `self.retrieve(query, top_k=20)` - Semantic search over a document corpus.
   Returns a formatted string of retrieved documents with titles and scores.

2. `self.rerank(query, top_k=10)` - Cross-encoder reranking of previously
   retrieved documents for higher precision. Returns formatted reranked docs.

3. `self.calculate(expression)` - Safe mathematical expression evaluation.
   Returns the result as a string (e.g., "2+3 = 5").

# Available Components
- `LLMAgentBase(output_fields, agent_name, model=self.node_model, role=..., temperature=..., usage_callback=self._usage_callback)` - Creates an LLM agent
- `Info(name, author, content, prompt, sub_tasks, agents, iteration_idx)` - Message container
- `self.make_final_answer(thinking, answer)` - Creates the final answer Info
- `self.node_model` - The model to use for agents
- `self.cot_instruction` - Base instruction for chain-of-thought
- `self.max_round` - Max iterations for reflexion/debate
- `self.max_sc` - Number of agents for self-consistency
- `self.debate_role` - List of roles for debate agents
- `self._usage_callback` - Callback for token tracking (pass to LLMAgentBase)

IMPORTANT: Always import from retcapslib.adapters.mas_zero.core:
  `from retcapslib.adapters.mas_zero.core import Info, LLMAgentBase`

# Discovered Architecture Archive
{archive_str}

# Output Format
Here is an example:

{json.dumps(EXAMPLE)}

Your code must define exactly one function: `def forward(self, taskInfo):`
- taskInfo is an Info namedtuple with the question in taskInfo.content
- Must return self.make_final_answer(thinking, answer)
- The answer Info must contain the final answer text

{_benchmark_context(benchmark_description, sample_questions)}# Your Task
Design an optimal RAG architecture that combines retrieval with reasoning
blocks for question answering. You can:
- Combine blocks (e.g., retrieve -> debate -> reflexion)
- Use iterative retrieval (retrieve, reason, re-retrieve with refined query)
- Use self-consistency with retrieval
- Create novel combinations

Observe the discovered architectures carefully and think about what insights
can be learned from them. Draw inspiration from the literature to propose
effective combinations.
"""

    if question:
        prompt += f"\nBelow is the question to solve:\n\n{question}"

    return prompt
