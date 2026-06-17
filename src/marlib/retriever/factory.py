from __future__ import annotations

from marlib.retriever.config import RetrieverSettings
from marlib.retriever.core import ChromaRetriever, RetrieverProtocol


def make_retriever(settings: RetrieverSettings) -> RetrieverProtocol:
    if settings.retriever == "chroma":
        return ChromaRetriever(settings)
    if settings.retriever == "lightrag":
        from marlib.retriever.lightrag_core import LightRAGRetriever

        return LightRAGRetriever(settings)
    raise ValueError(f"Unknown retriever backend: {settings.retriever}")
