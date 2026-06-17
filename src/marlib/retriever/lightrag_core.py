from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from marlib.log import logger
from marlib.retriever.config import RetrieverSettings
from marlib.retriever.core import Document
from marlib.retriever.embedder import make_embedder


@dataclass(frozen=True)
class LightRAGUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def diff(self, before: "LightRAGUsage") -> "LightRAGUsage":
        return LightRAGUsage(
            prompt_tokens=self.prompt_tokens - before.prompt_tokens,
            completion_tokens=self.completion_tokens - before.completion_tokens,
            total_tokens=self.total_tokens - before.total_tokens,
            calls=self.calls - before.calls,
        )


class _LightRAGTokenTracker:
    """Adapter for LightRAG's token_tracker.add_usage(...) callback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = LightRAGUsage()

    def add_usage(self, token_counts: dict[str, Any]) -> None:
        with self._lock:
            self._usage = LightRAGUsage(
                prompt_tokens=self._usage.prompt_tokens
                + int(token_counts.get("prompt_tokens") or 0),
                completion_tokens=self._usage.completion_tokens
                + int(token_counts.get("completion_tokens") or 0),
                total_tokens=self._usage.total_tokens
                + int(token_counts.get("total_tokens") or 0),
                calls=self._usage.calls + 1,
            )

    def snapshot(self) -> LightRAGUsage:
        with self._lock:
            return self._usage


def _provider_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _run_async(coro: Any) -> Any:
    if not inspect.isawaitable(coro):
        return coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as e:
            result["error"] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _get_field(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _coerce_references(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        for key in ("references", "chunks", "contexts", "retrieved_contexts"):
            refs = value.get(key)
            if isinstance(refs, list):
                return refs
    return []


def documents_from_lightrag_result(result: Any, limit: int | None) -> list[Document]:
    """Convert common LightRAG query result shapes into this repo's Document type."""
    refs = []
    for name in ("references", "chunks", "contexts", "retrieved_contexts"):
        refs = _coerce_references(_get_field(result, (name,)))
        if refs:
            break
    if not refs:
        refs = _coerce_references(result)

    docs: list[Document] = []
    limited_refs = refs if limit is None else refs[:limit]
    for i, ref in enumerate(limited_refs):
        text = _get_field(
            ref,
            ("content", "chunk_content", "text", "document", "context"),
            "",
        )
        doc_id = _get_field(
            ref,
            ("doc_id", "source_id", "reference_id", "id", "file_path"),
            f"lightrag:{i}",
        )
        title = _get_field(ref, ("title", "file_path", "source_id"), str(doc_id))
        score = _get_field(ref, ("score", "similarity", "distance"), 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        docs.append(
            Document(
                doc_id=str(doc_id),
                title=str(title),
                text=str(text or ""),
                score=score,
            )
        )

    if docs:
        return docs

    if isinstance(result, str) and result.strip():
        return [
            Document(
                doc_id="lightrag_context",
                title="LightRAG context",
                text=result.strip(),
                score=0.0,
            )
        ]
    return []


def answer_from_lightrag_result(result: Any) -> str:
    """Extract a generated answer from common LightRAG query result shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()

    for name in ("answer", "response", "result", "output", "content"):
        value = _get_field(result, (name,))
        if isinstance(value, str) and value.strip():
            return value.strip()

    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    return str(result).strip()


class LightRAGRetriever:
    """LightRAG backend adapted to the repo's synchronous retriever contract."""

    def __init__(self, settings: RetrieverSettings) -> None:
        self.settings = settings
        self._default_retrieve_k = settings.retrieve_top_k
        self._default_rerank_k = settings.rerank_top_k
        self._rag_lock = threading.Lock()
        self._embedder_lock = threading.Lock()
        self._token_tracker = _LightRAGTokenTracker()

        try:
            from lightrag import LightRAG
            from lightrag.llm.openai import openai_complete_if_cache
            from lightrag.utils import wrap_embedding_func_with_attrs
        except ImportError as e:
            raise RuntimeError(
                "LightRAG retriever requires the optional 'lightrag' dependency "
                "group. Install it with `uv sync --group lightrag`."
            ) from e

        working_dir = settings.index_path / settings.collection
        if not working_dir.exists():
            raise FileNotFoundError(
                f"LightRAG index not found at {working_dir}. Build it before "
                "running with `--retriever lightrag`."
            )
        self._working_dir = working_dir

        self._embedder = make_embedder(
            model_name=settings.embedder,
            backend=settings.embedder_backend,
            base_url=settings.embedder_base_url,
            api_key=settings.embedder_api_key,
        )

        async def embedding_func(texts: list[str]) -> np.ndarray:
            with self._embedder_lock:
                embeddings = self._embedder.encode_documents(
                    texts, batch_size=self._default_retrieve_k
                )
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            return embeddings

        embedding_func = wrap_embedding_func_with_attrs(
            embedding_dim=settings.embedder_dim,
            max_token_size=8192,
            model_name=settings.embedder,
        )(embedding_func)

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await openai_complete_if_cache(
                _provider_model_name(settings.lightrag_llm_model),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                token_tracker=self._token_tracker,
                **kwargs,
            )

        logger.info(f"Initializing LightRAG at: {working_dir}")
        self._rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model_func,
            llm_model_name=_provider_model_name(settings.lightrag_llm_model),
            embedding_func=embedding_func,
            enable_llm_cache=False,
        )
        _run_async(self._rag.initialize_storages())
        self._document_count = self._read_document_count()

    @property
    def document_count(self) -> int:
        return self._document_count

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        return self.search(query, top_k=top_k, use_rerank=False)

    def rerank(
        self, query: str, docs: list[Document], top_k: int | None = None
    ) -> list[Document]:
        if top_k is None:
            top_k = self._default_rerank_k
        return docs[:top_k]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> list[Document]:
        if top_k is None:
            top_k = self._default_rerank_k

        param = self._query_param(top_k=top_k, use_rerank=use_rerank)
        with self._rag_lock:
            if hasattr(self._rag, "query"):
                result = self._rag.query(query, param=param)
            else:
                result = _run_async(self._rag.aquery(query, param=param))
        result = _run_async(result)
        return documents_from_lightrag_result(result, limit=top_k)

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> tuple[str, list[Document], LightRAGUsage]:
        """Ask LightRAG to generate the final answer, returning answer + references + usage."""
        param = self._query_param(
            top_k=top_k,
            use_rerank=use_rerank,
            only_need_context=None,
        )
        with self._rag_lock:
            before_usage = self._token_tracker.snapshot()
            if hasattr(self._rag, "query"):
                result = self._rag.query(query, param=param)
            else:
                result = _run_async(self._rag.aquery(query, param=param))
            result = _run_async(result)
            usage = self._token_tracker.snapshot().diff(before_usage)
        return (
            answer_from_lightrag_result(result),
            documents_from_lightrag_result(result, limit=top_k),
            usage,
        )

    def close(self) -> None:
        finalize = getattr(self._rag, "finalize_storages", None)
        if finalize is not None:
            _run_async(finalize())

    def _query_param(
        self,
        top_k: int | None,
        use_rerank: bool | None,
        only_need_context: bool | None = True,
    ) -> Any:
        from lightrag import QueryParam

        kwargs = {"mode": self.settings.lightrag_mode}
        if top_k is not None:
            kwargs["top_k"] = top_k
            kwargs["chunk_top_k"] = top_k
        if only_need_context is not None:
            kwargs["only_need_context"] = only_need_context
        if use_rerank is not None:
            kwargs["enable_rerank"] = use_rerank
        while kwargs:
            try:
                return QueryParam(**kwargs)
            except TypeError as e:
                message = str(e)
                removed = False
                for key in list(kwargs):
                    if key in message:
                        kwargs.pop(key)
                        removed = True
                        break
                if not removed:
                    kwargs.popitem()
        return QueryParam()

    def _read_document_count(self) -> int:
        marker = self._working_dir / "marlib_lightrag_index.json"
        if not marker.exists():
            return 0
        try:
            data = json.loads(marker.read_text())
        except (OSError, json.JSONDecodeError):
            return 0
        for key in ("document_count", "documents", "corpus_count"):
            value = data.get(key)
            if isinstance(value, int):
                return value
        return 0
