from __future__ import annotations

import threading
from dataclasses import dataclass

import chromadb
import numpy as np

from marlib.retriever.config import RetrieverSettings
from marlib.retriever.embedder import BGEM3Embedder
from marlib.retriever.reranker import BGEReranker


@dataclass
class Document:
    """A retrieved passage with metadata."""

    doc_id: str
    title: str
    text: str
    score: float = 0.0


class Retriever:
    """ChromaDB collection + embedder + reranker behind a search() pipeline."""

    def __init__(self, settings: RetrieverSettings) -> None:
        self.settings = settings

        self._embedder = BGEM3Embedder(model_name=settings.embedder)
        self._reranker = BGEReranker(model_name=settings.reranker)

        # Each dataset's collection lives in its own subdirectory of index_path.
        index_path = settings.index_path / settings.collection
        self._client = chromadb.PersistentClient(path=str(index_path))
        self._collection = self._client.get_collection(settings.collection)

        # Serializes embedder/reranker access (their tokenizers aren't thread-safe).
        self._lock = threading.Lock()

        self._default_retrieve_k = settings.retrieve_top_k
        self._default_rerank_k = settings.rerank_top_k

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Dense retrieval, sorted by similarity."""
        if top_k is None:
            top_k = self._default_retrieve_k

        with self._lock:
            query_embedding = self._embedder.encode_queries([query])
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        results = self._collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

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

            for doc_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
                # Cosine distance is in [0, 2]; map to a [0, 1] similarity score.
                score = 1.0 - (distance / 2.0)
                documents.append(
                    Document(
                        doc_id=doc_id,
                        title=str(metadata.get("title", "")),
                        text=text or "",
                        score=score,
                    )
                )

        return documents

    def rerank(
        self, query: str, docs: list[Document], top_k: int | None = None
    ) -> list[Document]:
        """Cross-encoder re-rank, truncated to top_k."""
        if top_k is None:
            top_k = self._default_rerank_k
        with self._lock:
            return self._reranker.rerank(query, docs, top_k)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        """Retrieve candidates then rerank to top_k — the entry point for systems."""
        if top_k is None:
            top_k = self._default_rerank_k

        retrieve_k = self._default_retrieve_k if use_rerank else top_k
        candidates = self.retrieve(query, top_k=retrieve_k)

        if use_rerank and candidates:
            return self.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
