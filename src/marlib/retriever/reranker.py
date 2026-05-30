from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from marlib.log import logger
from marlib.retriever.config import DEFAULT_RERANKER

if TYPE_CHECKING:
    from marlib.retriever.core import Document


class BGEReranker:
    """Cross-encoder re-ranking with BGE-Reranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import FlagReranker

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        logger.info(f"Loading BGE reranker on {device}...")
        self._model = FlagReranker(
            model_name,
            use_fp16=use_fp16 and device != "cpu",
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

        from marlib.retriever.core import Document

        sentence_pairs = [(query, doc.text) for doc in documents]
        # normalize=True applies a sigmoid, keeping scores in [0, 1] so they are
        # comparable with the cosine similarities produced by retrieve().
        scores = self._model.compute_score(sentence_pairs, normalize=True)
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
