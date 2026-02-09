"""Unified retrieval infrastructure for all systems.

Provides dense retrieval over a shared Wikipedia corpus using
BGE-M3 embeddings + ChromaDB, with BGE reranker for re-ranking.
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
from retcapslib.retriever.reranker import BGEReranker

__all__ = [
    "Document",
    "Retriever",
    "BGEM3Embedder",
    "BGEReranker",
    "init_retriever",
    "get_retriever",
    "retrieve",
    "rerank",
    "search",
]
