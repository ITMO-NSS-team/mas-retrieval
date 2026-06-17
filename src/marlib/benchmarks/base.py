from __future__ import annotations

import importlib
import json
import re
import sys
import tomllib
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from marlib.log import logger

if TYPE_CHECKING:
    from chromadb.api.types import Metadata

# Default location of the benchmark repository (one subdirectory per benchmark).
DEFAULT_ROOT = Path("experiments/benchmarks")
_BENCH_NS = "_marlib_benchmarks"

# Metrics applied when a manifest does not declare its own ``metrics`` list.
DEFAULT_METRICS: tuple[str, ...] = ("exact_match", "f1")


def slugify(text: str) -> str:
    """Lowercase text and replace runs of non-alphanumeric chars with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class BenchmarkSpec:
    """A benchmark loaded from its ``manifest.toml``; paths are conventions under ``root``."""

    name: str
    root: Path
    description: str
    collection: str
    split: str | None = None
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def questions_path(self) -> Path:
        """JSONL of evaluation questions (produced by ``download``)."""
        return self.root / "questions.jsonl"

    def load_questions(self, sample_n: int | None = None) -> list[dict[str, Any]]:
        """Read the question set from ``questions_path`` (optionally capped)."""
        with open(self.questions_path) as f:
            questions = [json.loads(line) for line in f]
        if sample_n is not None:
            return questions[:sample_n]
        return questions

    @property
    def corpus_path(self) -> Path:
        """JSONL retrieval corpus (produced by ``build_corpus``)."""
        return self.root / "corpus.jsonl"

    @property
    def index_path(self) -> Path:
        """Base ChromaDB directory; the collection lives in ``index/<collection>``."""
        return self.root / "index"

    @property
    def lightrag_index_path(self) -> Path:
        """Base LightRAG directory; the collection lives in ``lightrag_index/<collection>``."""
        return self.root / "lightrag_index"

    @property
    def source_dir(self) -> Path:
        """Raw inputs needed to build the corpus (e.g. downloaded PDFs)."""
        return self.root / "source"


def _spec_from_manifest(manifest_path: Path) -> BenchmarkSpec:
    """Parse a single ``manifest.toml`` into a :class:`BenchmarkSpec`."""
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    root = manifest_path.parent
    name = data.get("name") or root.name
    metrics = data.get("metrics")
    return BenchmarkSpec(
        name=name,
        root=root,
        description=(data.get("description") or "").strip(),
        collection=data.get("collection") or name,
        split=data.get("split"),
        metrics=tuple(metrics) if metrics else DEFAULT_METRICS,
    )


def _ensure_namespace(parent: str, search_dir: Path) -> None:
    """Register/refresh a synthetic namespace package rooted at ``search_dir``."""
    mod = sys.modules.get(parent)
    if mod is None:
        mod = types.ModuleType(parent)
        sys.modules[parent] = mod
    mod.__path__ = [str(search_dir)]  # type: ignore[attr-defined]


def discover(root: str | Path = DEFAULT_ROOT) -> dict[str, BenchmarkSpec]:
    """Scan ``<root>/*/manifest.toml`` for benchmarks, importing each builder.py so
    its ``@register`` fires. The available set is exactly the dirs with a manifest."""
    root = Path(root)
    specs: dict[str, BenchmarkSpec] = {}
    if not root.is_dir():
        return specs
    _ensure_namespace(_BENCH_NS, root)
    for manifest_path in sorted(root.glob("*/manifest.toml")):
        spec = _spec_from_manifest(manifest_path)
        specs[spec.name] = spec
        if (manifest_path.parent / "builder.py").exists():
            try:
                importlib.import_module(f"{_BENCH_NS}.{spec.name}.builder")
            except Exception as e:  # builder import should be light; warn if not
                logger.warning(f"Builder for '{spec.name}' failed to load: {e!r}")
    return specs


def load_spec(name: str, root: str | Path = DEFAULT_ROOT) -> BenchmarkSpec:
    """Load one benchmark spec by name, raising with the available list on miss."""
    specs = discover(root)
    if name not in specs:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available in {root}: {sorted(specs)}"
        )
    return specs[name]


class BenchmarkBuilder(ABC):
    """Fetch and prepare one benchmark's data.

    Import heavy deps (``datasets``, ``pymupdf``, ...) lazily inside methods so
    importing the registry stays dependency-free.
    """

    @abstractmethod
    def download(self, spec: BenchmarkSpec) -> None:
        """Download questions (and raw source files) into the benchmark dir."""

    @abstractmethod
    def build_corpus(
        self, spec: BenchmarkSpec, max_paragraphs: int | None = None
    ) -> None:
        """Build the retrieval corpus (``spec.corpus_path``) from downloaded data."""


_BUILDERS: dict[str, type[BenchmarkBuilder]] = {}


def register(name: str):
    """Class decorator registering a :class:`BenchmarkBuilder` under ``name``."""

    def deco(cls: type[BenchmarkBuilder]) -> type[BenchmarkBuilder]:
        if name in _BUILDERS:
            raise ValueError(f"Benchmark builder '{name}' already registered")
        _BUILDERS[name] = cls
        return cls

    return deco


def registered_builders() -> list[str]:
    """Names of all registered builders (import the package to populate)."""
    return sorted(_BUILDERS)


def get_builder(name: str) -> BenchmarkBuilder:
    """Instantiate the registered builder for ``name``, raising with the list on miss."""
    if name not in _BUILDERS:
        raise ValueError(
            f"No builder registered for '{name}'. Registered: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name]()


# Identical for every benchmark, so it lives here rather than in each builder.
def build_index(
    spec: BenchmarkSpec,
    embedder_model: str | None = None,
    embedder_backend: Literal["local", "openai"] = "local",
    embedder_base_url: str | None = None,
    embedder_api_key: str | None = None,
    batch_size: int = 32,
) -> None:
    """Encode ``corpus.jsonl`` into a ChromaDB collection under
    ``spec.index_path / spec.collection`` (the layout Retriever expects).

    ``embedder_model`` defaults to the same model Retriever queries with, so an
    index is never built with a different model than retrieval uses.
    """
    import gc
    import json

    import chromadb
    import numpy as np
    import torch
    from tqdm import tqdm

    from marlib.log import logger
    from marlib.retriever.config import DEFAULT_EMBEDDER
    from marlib.retriever.embedder import make_embedder

    embedder_model = embedder_model or DEFAULT_EMBEDDER

    corpus_path = spec.corpus_path
    chroma_path = spec.index_path / spec.collection
    chroma_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing ChromaDB at: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=spec.collection,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Loading embedder: {embedder_model}")
    embedder = make_embedder(
        model_name=embedder_model,
        backend=embedder_backend,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )

    logger.info(f"Loading corpus from: {corpus_path}")
    docs = []
    with open(corpus_path) as f:
        for line in f:
            docs.append(json.loads(line))
    logger.info(f"Corpus size: {len(docs)} documents")

    logger.info("Encoding and indexing documents...")
    for batch_start in tqdm(range(0, len(docs), batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]

        texts = [d["text"] for d in batch_docs]
        doc_ids = [d["doc_id"] for d in batch_docs]

        embeddings = embedder.encode_documents(texts, batch_size=batch_size)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        metadatas: list[Metadata] = []
        for d in batch_docs:
            meta = {"title": str(d["title"]), "doc_id": str(d["doc_id"])}
            for key in ("company", "doc_type", "doc_period", "gics_sector"):
                if d.get(key):
                    meta[key] = str(d[key])
            metadatas.append(meta)

        collection.add(
            ids=doc_ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        del embeddings, texts, doc_ids, metadatas, batch_docs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info(f"Collection size: {collection.count()} documents")


def build_lightrag_index(
    spec: BenchmarkSpec,
    embedder_model: str | None = None,
    embedder_backend: Literal["local", "openai"] = "local",
    embedder_base_url: str | None = None,
    embedder_api_key: str | None = None,
    embedder_dim: int = 1024,
    llm_model: str = "openai/gpt-4o-mini",
    batch_size: int = 8,
    max_documents: int | None = None,
    max_parallel_insert: int = 2,
    clear_existing: bool = False,
) -> None:
    """Insert ``corpus.jsonl`` into a LightRAG workspace under
    ``spec.lightrag_index_path / spec.collection``.

    LightRAG indexing extracts entities/relations with an LLM, so this builder
    keeps the model explicit and records a marker file after successful insert.
    """
    import gc
    import os
    import shutil
    import threading
    from datetime import UTC, datetime

    import numpy as np
    import torch
    from tqdm import tqdm

    from marlib.log import logger
    from marlib.retriever.config import DEFAULT_EMBEDDER
    from marlib.retriever.embedder import make_embedder
    from marlib.retriever.lightrag_core import _provider_model_name, _run_async

    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as e:
        raise RuntimeError(
            "LightRAG index builder requires the optional 'lightrag' dependency "
            "group. Install it with `uv sync --group lightrag`."
        ) from e

    embedder_model = embedder_model or DEFAULT_EMBEDDER
    corpus_path = spec.corpus_path
    index_path = spec.lightrag_index_path / spec.collection

    if clear_existing and index_path.exists():
        logger.info(f"Clearing existing LightRAG index at: {index_path}")
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading corpus from: {corpus_path}")
    docs = []
    with open(corpus_path) as f:
        for line in f:
            docs.append(json.loads(line))
            if max_documents is not None and len(docs) >= max_documents:
                break
    logger.info(f"Corpus size: {len(docs)} documents")

    logger.info(f"Loading embedder: {embedder_model}")
    embedder = make_embedder(
        model_name=embedder_model,
        backend=embedder_backend,
        base_url=embedder_base_url,
        api_key=embedder_api_key,
    )
    embedder_lock = threading.Lock()

    async def embedding_func(texts: list[str]) -> np.ndarray:
        with embedder_lock:
            embeddings = embedder.encode_documents(texts, batch_size=batch_size)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    embedding_func = wrap_embedding_func_with_attrs(
        embedding_dim=embedder_dim,
        max_token_size=8192,
        model_name=embedder_model,
    )(embedding_func)

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> str:
        return await openai_complete_if_cache(
            _provider_model_name(llm_model),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            **kwargs,
        )

    logger.info(f"Initializing LightRAG at: {index_path}")
    rag = LightRAG(
        working_dir=str(index_path),
        llm_model_func=llm_model_func,
        llm_model_name=_provider_model_name(llm_model),
        embedding_func=embedding_func,
        embedding_func_max_async=1,
        max_parallel_insert=max_parallel_insert,
    )
    _run_async(rag.initialize_storages())

    try:
        logger.info("Inserting documents into LightRAG...")
        progress = tqdm(
            total=len(docs),
            desc=f"LightRAG insert ({batch_size}/batch)",
            unit="doc",
        )
        with progress:
            for batch_start in range(0, len(docs), batch_size):
                batch_docs = docs[batch_start : batch_start + batch_size]
                texts = [
                    f"Title: {d.get('title', '')}\n\n{d['text']}".strip()
                    for d in batch_docs
                ]
                doc_ids = [str(d["doc_id"]) for d in batch_docs]
                _run_async(rag.insert(texts, ids=doc_ids))
                progress.update(len(batch_docs))

                del texts, doc_ids, batch_docs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        marker = {
            "benchmark": spec.name,
            "collection": spec.collection,
            "document_count": len(docs),
            "corpus_path": str(corpus_path),
            "embedder": embedder_model,
            "embedder_backend": embedder_backend,
            "embedder_base_url": embedder_base_url,
            "embedder_dim": embedder_dim,
            "llm_model": llm_model,
            "batch_size": batch_size,
            "max_parallel_insert": max_parallel_insert,
            "max_documents": max_documents,
            "built_at": datetime.now(UTC).isoformat(),
        }
        (index_path / "marlib_lightrag_index.json").write_text(
            json.dumps(marker, indent=2) + "\n"
        )
        logger.info(f"LightRAG index ready: {index_path}")
    finally:
        finalize = getattr(rag, "finalize_storages", None)
        if finalize is not None:
            _run_async(finalize())
