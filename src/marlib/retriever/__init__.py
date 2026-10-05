from marlib.retriever.config import RetrieverSettings


def __getattr__(name):
    # Configuration and offline tests do not need torch, Chroma, or model weights.
    from importlib import import_module

    modules = {"Document": "core", "Retriever": "core",
               "BGEM3Embedder": "embedder", "BGEReranker": "reranker"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"marlib.retriever.{modules[name]}"), name)

__all__ = [
    "Document",
    "Retriever",
    "RetrieverSettings",
    "BGEM3Embedder",
    "BGEReranker",
]
