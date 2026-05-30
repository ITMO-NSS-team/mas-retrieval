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
from typing import TYPE_CHECKING, Any

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
    from marlib.retriever.embedder import BGEM3Embedder

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
    embedder = BGEM3Embedder(model_name=embedder_model)

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
