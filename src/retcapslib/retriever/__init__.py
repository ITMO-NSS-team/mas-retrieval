"""Unified retrieval infrastructure for all systems.

Provides dense retrieval over a shared Wikipedia corpus using
Jina embeddings v4 + ChromaDB, with Jina reranker v3 for re-ranking.
"""

from retcapslib.retriever.core import (
    Document,
    Retriever,
    get_retriever,
    init_retriever,
    rerank,
    retrieve,
    search,
)
from retcapslib.retriever.embedder import BGEM3Embedder
from retcapslib.retriever.reranker import JinaReranker

__all__ = [
    "Document",
    "Retriever",
    "BGEM3Embedder",
    "JinaReranker",
    "init_retriever",
    "get_retriever",
    "retrieve",
    "rerank",
    "search",
]
