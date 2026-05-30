from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from marlib.retriever.config import (
    DEFAULT_EMBEDDER,
    DEFAULT_RERANKER,
    RetrieverSettings,
)


@pytest.fixture(autouse=True)
def _clear_marlib_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MARLIB_"):
            monkeypatch.delenv(key, raising=False)


def _settings(**kw) -> RetrieverSettings:
    # _env_file=None isolates the test from any .env in the working directory.
    return RetrieverSettings(_env_file=None, **kw)


class TestRetrieverSettings:
    def test_required_fields_missing_raises(self):
        with pytest.raises(ValidationError):
            _settings()

    def test_defaults(self):
        s = _settings(index_path=Path("/tmp/idx"), collection="c")
        assert s.embedder == DEFAULT_EMBEDDER
        assert s.reranker == DEFAULT_RERANKER
        assert s.retrieve_top_k == 20
        assert s.rerank_top_k == 10

    def test_overrides(self):
        s = _settings(
            index_path=Path("/tmp/idx"),
            collection="c",
            retrieve_top_k=5,
            embedder="custom/model",
        )
        assert s.retrieve_top_k == 5
        assert s.embedder == "custom/model"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("MARLIB_INDEX_PATH", "/tmp/from_env")
        monkeypatch.setenv("MARLIB_COLLECTION", "env_coll")
        monkeypatch.setenv("MARLIB_RETRIEVE_TOP_K", "42")
        s = RetrieverSettings(_env_file=None)
        assert s.index_path == Path("/tmp/from_env")
        assert s.collection == "env_coll"
        assert s.retrieve_top_k == 42


class TestExportEnvRoundTrip:
    def test_round_trip(self, monkeypatch):
        original = _settings(
            index_path=Path("/data/index"),
            collection="financebench",
            retrieve_top_k=15,
            rerank_top_k=7,
            embedder="custom/embed",
            reranker="custom/rerank",
        )
        original.export_env()

        # A spawned subprocess reconstructs settings from the exported env.
        reconstructed = RetrieverSettings(_env_file=None)
        assert reconstructed.model_dump() == original.model_dump()

    def test_export_sets_prefixed_uppercase_vars(self, monkeypatch):
        _settings(index_path=Path("/x"), collection="c").export_env()
        assert os.environ["MARLIB_INDEX_PATH"] == "/x"
        assert os.environ["MARLIB_COLLECTION"] == "c"
