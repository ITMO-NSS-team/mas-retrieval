"""FastMCP wrapper for the retrieval API.

Used exclusively by CG-MAS (FEDOT.MAS), which discovers tools via MCP.
Other systems use the Python retriever functions directly.
"""

from __future__ import annotations

from fastmcp import FastMCP

from retcapslib.retriever.core import Document, get_retriever


def format_results(docs: list[Document]) -> str:
    """Format documents into a readable string.

    Args:
        docs: List of Document objects.

    Returns:
        Formatted string with titles, scores, and text.
    """
    if not docs:
        return "No results found."

    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc.title} (score: {doc.score:.3f})\n{doc.text}")

    return "\n\n".join(parts)


def create_retrieval_mcp_server() -> FastMCP:
    """Create a FastMCP server exposing retrieval tools.

    Tools exposed:
        - retrieval_search(query, top_k, use_rerank) -> list of passages

    Returns:
        A FastMCP server instance ready to be registered in
        FEDOT.MAS's MCP registry.

    Integration point:
        Register in FEDOT.MAS/src/fedotmas/mcp/registry.py
        alongside existing Tavily and browser MCP servers.
    """
    mcp = FastMCP("retrieval")

    @mcp.tool()
    def retrieval_search(
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> str:
        """Search the Wikipedia knowledge base for relevant passages.

        Use this tool to find information about topics, entities, events,
        or facts. The knowledge base contains Wikipedia paragraphs that
        can help answer questions.

        Args:
            query: Natural language search query describing what you're looking for.
            top_k: Number of passages to return (default 10, max 20).
            use_rerank: Whether to apply neural reranking for better relevance (default True).

        Returns:
            Formatted string of ranked passages with titles and relevance scores.
        """
        top_k = min(top_k, 20)  # Cap at 20 results

        retriever = get_retriever()
        docs = retriever.search(query, top_k=top_k, use_rerank=use_rerank)

        return format_results(docs)

    return mcp


# For direct execution as MCP server
if __name__ == "__main__":
    import sys

    from retcapslib.retriever.core import init_retriever

    # Initialize retriever (requires config)
    # In practice, this would be loaded from config.yaml
    config = {
        "embedder": "BAAI/bge-m3",
        "reranker": "jinaai/jina-reranker-v3",
        "index_path": "experiments/data/index/",
        "corpus_path": "experiments/data/corpus/wiki_paragraphs.jsonl",
    }

    try:
        init_retriever(config)
    except FileNotFoundError:
        print(
            "Error: Index or corpus not found. Run data preparation first.",
            file=sys.stderr,
        )
        print("  1. python -m experiments.data.prepare_corpus", file=sys.stderr)
        print(
            "  2. python -m experiments.retriever.index_builder --corpus ... --output ...",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp = create_retrieval_mcp_server()
    mcp.run()
