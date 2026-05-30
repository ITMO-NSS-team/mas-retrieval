from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models, referenced wherever a model name is needed (settings, the
# index builder, the embedder/reranker classes) so indexing and retrieval can
# never silently default to different models.
DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class RetrieverSettings(BaseSettings):
    """Single source of truth for retriever configuration.

    Defaults live here and nowhere else. The CLI builds an instance from its
    flags; the MCP server (a separate process) reconstructs one from the
    ``MARLIB_*`` environment variables the CLI exported via :meth:`export_env`.
    """

    model_config = SettingsConfigDict(
        env_prefix="MARLIB_", env_file=".env", extra="ignore"
    )

    # No defaults: there is no safe fallback for *where the corpus lives*, so a
    # missing value fails loudly at construction instead of silently querying
    # the wrong collection.
    index_path: Path
    collection: str

    embedder: str = DEFAULT_EMBEDDER
    reranker: str = DEFAULT_RERANKER
    retrieve_top_k: int = 20
    rerank_top_k: int = 10

    def export_env(self) -> None:
        """Serialize settings into ``MARLIB_*`` env vars for a spawned subprocess."""
        prefix = self.model_config["env_prefix"]
        for key, value in self.model_dump().items():
            os.environ[f"{prefix}{key}".upper()] = str(value)
