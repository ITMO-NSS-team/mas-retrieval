"""Naive RAG baseline adapter.

Simple single-shot retrieve-then-generate pipeline:
1. Search for relevant passages
2. Generate answer from retrieved context
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from retcapslib.adapters.base import AbstractAdapter
from retcapslib.logging.schemas import QuestionLog
from retcapslib.logging.tracker import TokenTracker
from retcapslib.retriever.core import Retriever

load_dotenv()


class NaiveRAGAdapter(AbstractAdapter):
    """Naive RAG baseline: single retrieve + generate."""

    def __init__(
        self,
        retriever: Retriever,
        model: str = "gpt-4o-mini",
        top_k: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize naive RAG adapter.

        Args:
            retriever: Retriever instance.
            model: LLM model for answer generation.
            top_k: Number of passages to retrieve.
        """
        super().__init__(retriever, model, **kwargs)
        self._top_k = top_k
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
        )

    @property
    def name(self) -> str:
        return "naive_rag"

    def generate_system(self, question: str) -> str:
        """Naive RAG has no dynamic system generation."""
        return "static: retrieve + generate"

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        """Execute naive RAG pipeline.

        1. Retrieve top-k passages
        2. Generate answer from context
        """
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            # Step 1: Retrieve passages
            with tracker.track_tool("search", question, self._top_k) as doc_ids:
                docs = self._retriever.search(question, top_k=self._top_k)
                doc_ids.extend([doc.doc_id for doc in docs])

            # Build context from retrieved passages
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[{i}] {doc.title}\n{doc.text}")
            context = "\n\n".join(context_parts)

            # Step 2: Generate answer
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Answer the question using only the information in the context. Be concise and direct.
If the answer cannot be found in the context, say "I don't know"."""

            user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

            with tracker.track_llm(self._model) as stats:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=256,
                    temperature=0,
                )
                stats["prompt_tokens"] = response.usage.prompt_tokens
                stats["completion_tokens"] = response.usage.completion_tokens

            answer = response.choices[0].message.content.strip()

        except Exception as e:
            tracker.set_error(str(e))
            answer = ""

        return answer, tracker.to_question_log(answer)
