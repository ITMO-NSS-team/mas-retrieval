"""Core retrieval API used by all systems.

Canonical functions:
- retrieve(query, top_k) -> dense ChromaDB search
- rerank(query, docs, top_k) -> Jina reranker v3 re-scoring
- search(query, top_k, rerank) -> retrieve + optional rerank (default pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from retcapslib.retriever.embedder import BGEM3Embedder
from retcapslib.retriever.reranker import JinaReranker


@dataclass
class Document:
    """A retrieved passage with metadata."""

    doc_id: str
    title: str
    text: str
    score: float = 0.0


class Retriever:
    """Stateful retriever wrapping ChromaDB collection + embedder + reranker."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize retriever from config.

        Args:
            config: Retriever section from config.yaml with keys:
                - embedder: model name for BGEM3Embedder
                - reranker: model name for JinaReranker
                - index_path: path to ChromaDB persistent storage directory
                - collection_name: ChromaDB collection name (default "wikipedia")
                - retrieve_top_k: default candidates for retrieval (default 20)
                - rerank_top_k: default results after reranking (default 10)
        """
        self._config = config

        # Initialize embedder
        embedder_model = config.get("embedder", "BAAI/bge-m3")
        self._embedder = BGEM3Embedder(model_name=embedder_model)

        # Initialize reranker
        reranker_model = config.get("reranker", "jinaai/jina-reranker-v3")
        self._reranker = JinaReranker(model_name=reranker_model)

        # Initialize ChromaDB client and get collection
        index_path = Path(config["index_path"])
        self._client = chromadb.PersistentClient(path=str(index_path))

        collection_name = config.get("collection_name", "wikipedia")
        self._collection = self._client.get_collection(collection_name)

        self._collection_name = collection_name

        # Store default parameters
        self._default_retrieve_k = config.get("retrieve_top_k", 20)
        self._default_rerank_k = config.get("rerank_top_k", 10)

    def set_collection(self, name: str) -> None:
        """Switch to a different ChromaDB collection.

        Keeps the embedder and reranker loaded — only the collection changes.

        Args:
            name: Name of the ChromaDB collection to switch to.
        """
        if name != self._collection_name:
            self._collection = self._client.get_collection(name)
            self._collection_name = name

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Dense retrieval via ChromaDB.

        Args:
            query: Search query string.
            top_k: Number of candidates to return (default from config).

        Returns:
            List of Documents sorted by dense similarity score.
        """
        if top_k is None:
            top_k = self._default_retrieve_k

        # Encode query
        query_embedding = self._embedder.encode_queries([query])

        # Ensure embedding is float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Query ChromaDB collection
        results = self._collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Build Document list
        documents = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = (
                results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            )
            distances = (
                results["distances"][0] if results["distances"] else [0.0] * len(ids)
            )

            for doc_id, text, metadata, distance in zip(
                ids, docs, metadatas, distances
            ):
                # Convert distance to score (for cosine: smaller distance = higher similarity)
                # ChromaDB cosine distance is in [0, 2], we convert to score in [0, 1]
                score = 1.0 - (distance / 2.0)

                documents.append(
                    Document(
                        doc_id=doc_id,
                        title=metadata.get("title", ""),
                        text=text or "",
                        score=score,
                    )
                )

        return documents

    def rerank(
        self, query: str, docs: list[Document], top_k: int | None = None
    ) -> list[Document]:
        """Re-rank documents using Jina reranker v3.

        Args:
            query: Original query for relevance scoring.
            docs: Candidate documents from retrieve().
            top_k: Number of documents to keep after reranking (default from config).

        Returns:
            Re-ranked and truncated list of Documents.
        """
        if top_k is None:
            top_k = self._default_rerank_k

        return self._reranker.rerank(query, docs, top_k)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        """Full retrieval pipeline: retrieve top candidates → rerank to top-k.

        This is the canonical function exposed to all systems.

        Args:
            query: Search query string.
            top_k: Final number of documents to return (default from config).
            use_rerank: Whether to apply reranking (default True).

        Returns:
            List of Documents after retrieval and optional reranking.
        """
        if top_k is None:
            top_k = self._default_rerank_k

        # Retrieve more candidates for reranking
        retrieve_k = self._default_retrieve_k if use_rerank else top_k
        candidates = self.retrieve(query, top_k=retrieve_k)

        if use_rerank and candidates:
            return self.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]


# Module-level convenience functions (use a global Retriever instance)

_retriever: Retriever | None = None


def init_retriever(config: dict) -> Retriever:
    """Initialize the global retriever instance."""
    global _retriever
    _retriever = Retriever(config)
    return _retriever


def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    assert _retriever is not None, "Call init_retriever() first"
    return _retriever


def retrieve(query: str, top_k: int = 20) -> list[Document]:
    """Module-level retrieve using global retriever."""
    assert _retriever is not None, "Call init_retriever() first"
    return _retriever.retrieve(query, top_k)


def rerank(query: str, docs: list[Document], top_k: int = 10) -> list[Document]:
    """Module-level rerank using global retriever."""
    assert _retriever is not None, "Call init_retriever() first"
    return _retriever.rerank(query, docs, top_k)


def search(query: str, top_k: int = 10, use_rerank: bool = True) -> list[Document]:
    """Module-level search using global retriever."""
    assert _retriever is not None, "Call init_retriever() first"
    return _retriever.search(query, top_k, use_rerank)
