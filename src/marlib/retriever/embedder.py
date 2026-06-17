from __future__ import annotations

import os
from typing import Any, Literal, Protocol

import numpy as np
import torch

from marlib.log import logger
from marlib.retriever.config import DEFAULT_EMBEDDER


class EmbedderProtocol(Protocol):
    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray: ...

    def encode_documents(
        self, documents: list[str], batch_size: int = 32
    ) -> np.ndarray: ...


class BGEM3Embedder:
    """Dense encoding with BGE-M3."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDER,
        device: str | None = None,
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        logger.info(f"Loading BGE-M3 embedder on {device}...")
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and device != "cpu",
            device=device,
        )

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries to (n, 1024) dense vectors."""
        output = self._model.encode(
            queries,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        if not isinstance(dense_vecs, np.ndarray):
            dense_vecs = np.array(dense_vecs)
        return dense_vecs

    def encode_documents(
        self, documents: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """Encode passages to (n, 1024) dense vectors."""
        output = self._model.encode(
            documents,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        if self._device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()

        dense_vecs = output["dense_vecs"]
        if not isinstance(dense_vecs, np.ndarray):
            dense_vecs = np.array(dense_vecs)
        return dense_vecs


class OpenAIEmbedder:
    """Dense encoding through an OpenAI-compatible embeddings API."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDER,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "OpenAI-compatible embeddings require the optional "
                    "'api_embedder' dependency group. Install it with "
                    "`uv sync --group api_embedder`."
                ) from e

            client = OpenAI(
                base_url=self._base_url,
                api_key=api_key or os.environ.get("OPENAI_API_KEY") or "dummy",
            )
        self._client = client
        logger.info(
            f"Loading OpenAI-compatible embedder API: {model_name}",
            base_url=self._base_url,
        )

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(queries, batch_size=batch_size)

    def encode_documents(
        self, documents: list[str], batch_size: int = 32
    ) -> np.ndarray:
        return self._encode(documents, batch_size=batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            logger.debug(
                "OpenAI embedder request",
                model=self._model_name,
                base_url=self._base_url,
                batch_size=len(batch),
            )
            response = self._client.embeddings.create(
                model=self._model_name,
                input=batch,
            )
            data = list(response.data)
            data.sort(key=lambda item: getattr(item, "index", 0))
            embeddings.extend([list(item.embedding) for item in data])
        return np.array(embeddings, dtype=np.float32)


def make_embedder(
    model_name: str = DEFAULT_EMBEDDER,
    backend: Literal["local", "openai"] = "local",
    base_url: str | None = None,
    api_key: str | None = None,
) -> EmbedderProtocol:
    if backend == "local":
        return BGEM3Embedder(model_name=model_name)
    if backend == "openai":
        return OpenAIEmbedder(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    raise ValueError(f"Unknown embedder backend: {backend}")
