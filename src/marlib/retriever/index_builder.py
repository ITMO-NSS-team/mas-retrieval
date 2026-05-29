"""Build ChromaDB index from a corpus.

Reads the preprocessed corpus (JSONL), encodes passages with
BGE-M3 embeddings, and stores in a ChromaDB persistent collection.

Supports per-dataset presets via --dataset flag:
  hotpotqa     → corpus/hotpotqa_paragraphs.jsonl,     collection "hotpotqa"
  financebench → corpus/financebench_paragraphs.jsonl,  collection "financebench"
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import chromadb
import numpy as np
import torch
from tqdm import tqdm

from marlib.log import logger
from marlib.retriever.embedder import BGEM3Embedder

# ChromaDB batch limit (SQLite constraint)
CHROMA_MAX_BATCH = 41666

# Per-dataset presets: (corpus_path, collection_name)
# When using --dataset, the ChromaDB storage is automatically placed in a
# per-dataset subdirectory: {output}/{collection_name}/
DATASET_PRESETS = {
    "hotpotqa": (
        "experiments/data/corpus/hotpotqa_paragraphs.jsonl",
        "hotpotqa",
    ),
    "financebench": (
        "experiments/data/corpus/financebench_paragraphs.jsonl",
        "financebench",
    ),
}


def build_index(
    corpus_path: str | Path,
    chroma_path: str | Path,
    embedder_model: str = "BAAI/bge-m3",
    batch_size: int = 32,
    collection_name: str = "wikipedia",
) -> None:
    """Build a ChromaDB collection from the Wikipedia corpus.

    Args:
        corpus_path: Path to wiki_paragraphs.jsonl (one JSON object per line
            with fields: doc_id, title, text).
        chroma_path: Directory for ChromaDB persistent storage.
        embedder_model: Model identifier for BGE-M3 embeddings.
        batch_size: Number of passages to encode per batch.
        collection_name: Name of the ChromaDB collection.
    """
    corpus_path = Path(corpus_path)
    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB client
    logger.info(f"Initializing ChromaDB at: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Create collection with cosine similarity metric
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Initialize embedder
    logger.info(f"Loading embedder: {embedder_model}")
    embedder = BGEM3Embedder(model_name=embedder_model)

    # Load corpus from JSONL
    logger.info(f"Loading corpus from: {corpus_path}")
    docs = []
    with open(corpus_path) as f:
        for line in f:
            docs.append(json.loads(line))

    logger.info(f"Corpus size: {len(docs)} documents")

    # Process in batches for encoding
    logger.info("Encoding and indexing documents...")
    for batch_start in tqdm(range(0, len(docs), batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, len(docs))
        batch_docs = docs[batch_start:batch_end]

        # Extract fields
        texts = [d["text"] for d in batch_docs]
        doc_ids = [d["doc_id"] for d in batch_docs]
        titles = [d["title"] for d in batch_docs]

        # Encode batch
        logger.debug(f"Encoding batch {batch_start}-{batch_end}")
        embeddings = embedder.encode_documents(texts, batch_size=batch_size)

        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Build metadata — include extra fields if present in corpus
        metadatas = []
        for d in batch_docs:
            meta = {"title": d["title"], "doc_id": d["doc_id"]}
            for key in ("company", "doc_type", "doc_period", "gics_sector"):
                if key in d and d[key]:
                    meta[key] = str(d[key])
            metadatas.append(meta)

        # Add to ChromaDB collection
        logger.debug("Adding batch to ChromaDB")
        collection.add(
            ids=doc_ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        # Free memory
        del embeddings, texts, doc_ids, titles, metadatas, batch_docs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info(f"Collection size: {collection.count()} documents")


def main() -> None:
    """CLI entry point for building the index."""
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from Wikipedia corpus"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotqa", "financebench"],
        default=None,
        help="Dataset preset (sets corpus path and collection name automatically)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus JSONL (overrides --dataset preset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/data/chroma_index",
        help="Output directory for ChromaDB storage",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedder model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (BGE-M3 uses ~3-4GB VRAM, safe for 16GB GPUs)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (overrides --dataset preset)",
    )

    args = parser.parse_args()

    # Resolve corpus path and collection name from preset or explicit args
    if args.dataset:
        preset_corpus, preset_collection = DATASET_PRESETS[args.dataset]
        corpus = args.corpus or preset_corpus
        collection = args.collection or preset_collection
    else:
        corpus = args.corpus or "experiments/data/corpus/wiki_paragraphs.jsonl"
        collection = args.collection or "wikipedia"

    # Place each dataset in its own subdirectory to avoid collision
    chroma_path = f"{args.output}/{collection}"

    build_index(
        corpus_path=corpus,
        chroma_path=chroma_path,
        embedder_model=args.model,
        batch_size=args.batch_size,
        collection_name=collection,
    )


if __name__ == "__main__":
    main()
