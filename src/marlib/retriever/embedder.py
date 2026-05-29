"""BGE-M3 embeddings wrapper for query and document encoding.

Uses local inference via the BAAI/bge-m3 model with FlagEmbedding library.
Supports dense retrieval mode (1024-dim embeddings).
"""

from __future__ import annotations

import numpy as np
import torch

from marlib.log import logger


class BGEM3Embedder:
    """Wrapper around BGE-M3 for dense encoding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        use_fp16: bool = True,
    ) -> None:
        """Load the embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on ("cuda", "cpu", or None for auto).
            use_fp16: Use half precision for memory efficiency (default True).
        """
        from FlagEmbedding import BGEM3FlagModel

        self._model_name = model_name

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        # Load model with FlagEmbedding
        logger.info(f"Loading BGE-M3 embedder on {device}...")
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and device != "cpu",
            device=device,
        )

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode search queries.

        Args:
            queries: List of query strings.
            batch_size: Encoding batch size.

        Returns:
            numpy array of shape (len(queries), 1024).
        """
        output = self._model.encode(
            queries,
            batch_size=batch_size,
            max_length=512,  # Queries are typically short
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        if not isinstance(dense_vecs, np.ndarray):
            dense_vecs = np.array(dense_vecs)
        return dense_vecs

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode document passages for indexing.

        Args:
            documents: List of passage texts.
            batch_size: Encoding batch size.

        Returns:
            numpy array of shape (len(documents), 1024).
        """
        output = self._model.encode(
            documents,
            batch_size=batch_size,
            max_length=8192,  # BGE-M3 supports long documents
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        # Free GPU memory after batch
        if self._device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()

        dense_vecs = output["dense_vecs"]
        if not isinstance(dense_vecs, np.ndarray):
            dense_vecs = np.array(dense_vecs)
        return dense_vecs

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return 1024  # BGE-M3 fixed dimension
