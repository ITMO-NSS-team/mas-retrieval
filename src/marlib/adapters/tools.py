from __future__ import annotations

from marlib.adapters.calc import do_calculate, safe_eval
from marlib.retriever.core import Document, Retriever

__all__ = [
    "format_docs",
    "do_retrieve",
    "do_rerank",
    "do_calculate",
    "safe_eval",
]


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No results found."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.title} (score: {doc.score:.3f})\n{doc.text}")
    return "\n\n".join(parts)


def do_retrieve(
    retriever: Retriever,
    query: str,
    top_k: int = 20,
) -> tuple[list[Document], str]:
    docs = retriever.retrieve(query, top_k=top_k)
    return docs, format_docs(docs)


def do_rerank(
    retriever: Retriever,
    query: str,
    docs: list[Document],
    top_k: int = 10,
) -> tuple[list[Document], str]:
    reranked = retriever.rerank(query, docs, top_k=top_k)
    return reranked, format_docs(reranked)
