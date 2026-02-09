"""MCP server providing retrieval and calculator tools for CG-MAS.

Runs as a subprocess spawned by FEDOT.MAS via MCPServerStdio.
Reads retriever config from environment variables.
Writes doc_ids to RETCAP_DOCIDS_FILE for tracking.
"""

from __future__ import annotations

import json
import os

from fastmcp import FastMCP

from retcapslib.adapters.tools import safe_eval
from retcapslib.retriever.core import Retriever

mcp = FastMCP("retrieval")

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        config = {
            "embedder": os.environ.get("RETCAP_EMBEDDER", "BAAI/bge-m3"),
            "reranker": os.environ.get("RETCAP_RERANKER", "BAAI/bge-reranker-v2-m3"),
            "index_path": os.environ["RETCAP_INDEX_PATH"],
            "collection_name": os.environ.get("RETCAP_COLLECTION", "financebench"),
        }
        _retriever = Retriever(config)
    return _retriever


def _log_doc_ids(tool_name: str, query: str, doc_ids: list[str]) -> None:
    """Append doc_ids to tracking file for context_recall metric."""
    path = os.environ.get("RETCAP_DOCIDS_FILE")
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps({"tool": tool_name, "query": query, "doc_ids": doc_ids}) + "\n")


@mcp.tool()
def retrieval_search(query: str, top_k: int = 10, use_rerank: bool = True) -> str:
    """Search the document knowledge base for relevant passages.

    Use to find information, evidence, and facts from the corpus.

    Args:
        query: Natural language search query.
        top_k: Number of passages to return (default 10, max 20).
        use_rerank: Apply neural reranking for better relevance (default True).

    Returns:
        Formatted ranked passages with titles and scores.
    """
    top_k = min(top_k, 20)
    retriever = _get_retriever()
    docs = retriever.search(query, top_k=top_k, use_rerank=use_rerank)
    doc_ids = [doc.doc_id for doc in docs]
    _log_doc_ids("retrieve", query, doc_ids)

    if not docs:
        return "No results found."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.title} (score: {doc.score:.3f})\n{doc.text}")
    return "\n\n".join(parts)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, %, round(), abs(), min(), max(), pi, e.

    Args:
        expression: Math expression to evaluate (e.g. "revenue / shares", "round(456.78 / 123, 2)").

    Returns:
        Result string like "expression = value".
    """
    try:
        result = safe_eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


if __name__ == "__main__":
    mcp.run()
