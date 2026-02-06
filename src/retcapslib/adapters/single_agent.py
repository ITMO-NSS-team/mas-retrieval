"""Single agent baseline adapter.

Iterative tool-calling agent that can make multiple retrieval calls
to gather evidence before answering.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Retriever

# Tool definition for the agent
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the knowledge base for relevant passages. Use this to find information needed to answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant passages.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (default: 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


class SingleAgentAdapter(AbstractAdapter):
    """Single iterative agent with search tool."""

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize single agent adapter.

        Args:
            retriever: Retriever instance.
            model: LLM model for the agent.
            max_iterations: Maximum number of search iterations.
        """
        super().__init__(retriever, model, **kwargs)
        self._max_iterations = max_iterations
        self._client = OpenAI()

    @property
    def name(self) -> str:
        return "single_agent"

    def generate_system(self, question: str) -> str:
        """Single agent has no dynamic system generation."""
        return "static: iterative search agent"

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        """Execute iterative agent loop.

        The agent can call search multiple times to gather evidence,
        then provides a final answer.
        """
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            system_prompt = """You are a helpful assistant that answers questions by searching for relevant information.

You have access to a search tool that retrieves passages from a knowledge base.
For complex questions that require multiple pieces of information, you should:
1. Break down the question into sub-queries
2. Search for each piece of information
3. Combine the evidence to answer the question

When you have gathered enough information, provide your final answer.
Be concise and direct in your final answer."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            # Iterative tool-calling loop
            for _ in range(self._max_iterations):
                with tracker.track_llm(self._model) as stats:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        tools=[SEARCH_TOOL],
                        tool_choice="auto",
                        max_tokens=512,
                        temperature=0,
                    )
                    stats["prompt_tokens"] = response.usage.prompt_tokens
                    stats["completion_tokens"] = response.usage.completion_tokens

                assistant_message = response.choices[0].message
                messages.append(assistant_message.model_dump())

                # Check if agent wants to use tools
                if not assistant_message.tool_calls:
                    # No tool calls - agent is done
                    answer = assistant_message.content or ""
                    break

                stats["function_calls"] = len(assistant_message.tool_calls)

                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "search":
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query", question)
                        top_k = args.get("top_k", 5)

                        # Execute search
                        with tracker.track_tool("search", query, top_k) as doc_ids:
                            docs = self._retriever.search(query, top_k=top_k)
                            doc_ids.extend([doc.doc_id for doc in docs])

                        # Format results for the agent
                        results = []
                        for i, doc in enumerate(docs, 1):
                            results.append(f"[{i}] {doc.title}: {doc.text}")

                        tool_result = (
                            "\n\n".join(results) if results else "No results found."
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            }
                        )
            else:
                # Max iterations reached - force final answer
                with tracker.track_llm(self._model) as stats:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages
                        + [
                            {
                                "role": "user",
                                "content": "Please provide your final answer now based on the information gathered.",
                            }
                        ],
                        max_tokens=256,
                        temperature=0,
                    )
                    stats["prompt_tokens"] = response.usage.prompt_tokens
                    stats["completion_tokens"] = response.usage.completion_tokens

                answer = response.choices[0].message.content or ""

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
