from __future__ import annotations

import json
import os

from fastmcp import FastMCP

from marlib.adapters.calc import safe_eval
from marlib.retriever import Retriever, RetrieverSettings

mcp = FastMCP("marlib-tools")

# Built once in __main__ before the server starts (this module is only ever run
# as `python -m marlib.mcp_server`, spawned by the retrieval adapters).
_retriever: Retriever | None = None


def _log_doc_ids(tool_name: str, query: str, doc_ids: list[str]) -> None:
    """Append doc_ids to the tracking file for the context_recall metric."""
    path = os.environ.get("MARLIB_DOCIDS_FILE")
    if not path:
        return
    with open(path, "a") as f:
        f.write(
            json.dumps({"tool": tool_name, "query": query, "doc_ids": doc_ids}) + "\n"
        )


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
    assert _retriever is not None, "retriever not initialized (run as __main__)"
    top_k = min(top_k, 20)
    docs = _retriever.search(query, top_k=top_k, use_rerank=use_rerank)
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
    # This process speaks JSON-RPC over stdout (MCP stdio transport). marlib's
    # logger (logly) writes its console sink to stdout too, which corrupts the
    # protocol stream ("Failed to parse JSONRPC message from server"). Silence
    # the stdout console before anything logs; optionally tee to a file so the
    # subprocess logs remain inspectable.
    from marlib.log import logger

    logger.configure(level=os.environ.get("MARLIB_LOG_LEVEL", "INFO").upper(),
                     color=False, console=False)
    log_file = os.environ.get("MARLIB_LOG_FILE")
    if log_file:
        logger.add(sink=log_file)

    # Fields are populated from MARLIB_* env at runtime, not constructor args.
    _retriever = Retriever(RetrieverSettings())  # ty: ignore[missing-argument]
    mcp.run(show_banner=False)
