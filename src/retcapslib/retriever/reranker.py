"""Jina reranker v3 wrapper for passage re-ranking.

Uses local inference via the Jina reranker v3 model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import AutoModel

if TYPE_CHECKING:
    from retcapslib.retriever.core import Document


class JinaReranker:
    """Wrapper around Jina reranker v3 for cross-encoder re-ranking."""

    def __init__(self, model_name: str = "jinaai/jina-reranker-v3") -> None:
        """Load the reranker model.

        Args:
            model_name: HuggingFace model identifier.
        """
        self._model_name = model_name
        self._model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",  # Automatic precision selection
            trust_remote_code=True,
        )
        self._model.eval()

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 10,
    ) -> list[Document]:
        """Re-rank documents by relevance to the query.

        The Jina reranker v3 can process up to 64 documents simultaneously
        in a single forward pass using its 131K token context window.

        Args:
            query: The search query.
            documents: Candidate documents to re-rank.
            top_k: Number of top documents to return.

        Returns:
            Re-ranked list of Documents, truncated to top_k,
            with updated scores from the reranker.
        """
        if not documents:
            return []

        # Extract texts for reranking
        doc_texts = [doc.text for doc in documents]

        # Rerank using the model's built-in method
        # Returns list of dicts: {"document": str, "relevance_score": float, "index": int}
        results = self._model.rerank(query, doc_texts, top_n=top_k)

        # Build result list with updated scores
        reranked = []
        for result in results:
            original_idx = result["index"]
            doc = documents[original_idx]
            # Create a copy with updated score
            from retcapslib.retriever.core import Document

            reranked.append(
                Document(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=result["relevance_score"],
                )
            )

        return reranked
