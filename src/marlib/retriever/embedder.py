from __future__ import annotations

import numpy as np
import torch

from marlib.log import logger


class BGEM3Embedder:
    """Dense encoding with BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
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
