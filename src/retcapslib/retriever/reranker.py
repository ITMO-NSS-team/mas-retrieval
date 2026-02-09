"""BGE reranker wrapper for passage re-ranking.

Uses local inference via the BAAI/bge-reranker-v2-m3 model
through the FlagEmbedding library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from retcapslib.retriever.core import Document


class BGEReranker:
    """Wrapper around BGE-Reranker for cross-encoder re-ranking."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import FlagReranker

        self._model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"Loading BGE reranker on {device}...")
        self._model = FlagReranker(
            model_name,
            use_fp16=use_fp16 and device == "cuda",
            device=device,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 10,
    ) -> list[Document]:
        if not documents:
            return []

        from retcapslib.retriever.core import Document

        sentence_pairs = [[query, doc.text] for doc in documents]
        scores = self._model.compute_score(sentence_pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            Document(
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.text,
                score=float(score),
            )
            for doc, score in scored
        ]
