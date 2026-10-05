"""Check the FinanceBench runtime/data without LLM calls or automatic downloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform


def digest(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_data(root):
    problems, metadata = [], {}
    for name in ("questions.jsonl", "corpus.jsonl"):
        path = root / name
        metadata[name] = {"sha256": digest(path)}
        if not path.is_file():
            problems.append(f"Missing {path}")
    if problems:
        return metadata, problems
    questions = [json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines() if line.strip()]
    # Presence alone cannot detect the historical off-by-one: most wrong IDs
    # still name a real corpus page. Check against the original evidence too.
    mapping_mismatches = []
    if any(q.get("evidence") for q in questions):
        from marlib.benchmarks import discover, get_builder
        discover()
        builder = get_builder("financebench")
        for q in questions:
            if q.get("evidence"):
                expected = builder.evidence_doc_ids(q)
                if sorted(q.get("gold_doc_ids") or []) != expected:
                    mapping_mismatches.append({"id": q.get("id"), "expected": expected,
                                               "actual": q.get("gold_doc_ids")})
    if mapping_mismatches:
        problems.append(f"Incorrect evidence page mapping for {len(mapping_mismatches)} questions; run scripts/repair_finance_evidence.py")
    ids, seen_docs = set(), set()
    for q in questions:
        if not q.get("id") or q["id"] in ids:
            problems.append("Missing or duplicate question ID")
        ids.add(q.get("id"))
        if not q.get("gold_doc_ids"):
            problems.append(f"No evidence mapping for {q.get('id')}")
    with (root / "corpus.jsonl").open() as f:
        for line in f:
            doc = json.loads(line)
            if not doc.get("doc_id") or doc["doc_id"] in seen_docs:
                problems.append("Missing or duplicate corpus ID")
            seen_docs.add(doc.get("doc_id"))
    missing = [{"id": q["id"], "doc_id": doc} for q in questions
               for doc in q.get("gold_doc_ids", []) if doc not in seen_docs]
    if missing:
        problems.append(f"Missing {len(missing)} evidence page references")
    if len(questions) != 150:
        problems.append(f"Expected 150 FinanceBench questions; found {len(questions)}")
    metadata.update(question_count=len(questions), corpus_count=len(seen_docs), missing_evidence=missing,
                    evidence_mapping_mismatches=mapping_mismatches,
                    pilot_ids=[q["id"] for q in questions[:5]], ordered_question_ids=[q["id"] for q in questions])
    return metadata, problems


def verify_existing_index(collection, corpus_path, embedder):
    """Check all stored texts/IDs, then test sampled embedding compatibility.

    This cannot recover a historical model revision or prove that unsampled
    vectors were built with the same model. Record that limit in the sidecar.
    """
    documents = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    expected = {d["doc_id"]: d["text"] for d in documents}
    if not expected or len(expected) != len(documents):
        raise ValueError("Empty corpus or duplicate document IDs")
    if collection.count() != len(expected):
        raise ValueError("Index document count differs from corpus")
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        raise ValueError("Cannot confirm cosine distance in the existing index metadata")
    ids = sorted(expected)
    for start in range(0, len(ids), 400):
        batch = ids[start:start + 400]
        stored = collection.get(ids=batch, include=["documents"])
        if set(stored["ids"]) != set(batch):
            raise ValueError("Index document IDs differ from corpus")
        texts = stored.get("documents")
        if texts is None or len(texts) != len(stored["ids"]):
            raise ValueError("Index has missing document texts")
        for doc_id, text in zip(stored["ids"], texts):
            if text != expected[doc_id]:
                raise ValueError(f"Index text differs from corpus: {doc_id}")

    # Independent of evidence labels and accuracy; stable across machines/runs.
    sample = sorted(ids, key=lambda doc_id: hashlib.sha256(doc_id.encode()).hexdigest())[:16]
    stored = collection.get(ids=sample, include=["embeddings"])
    vectors = stored.get("embeddings")
    if set(stored["ids"]) != set(sample) or vectors is None or len(vectors) != len(sample):
        raise ValueError("Index has missing sample embeddings")
    vectors = dict(zip(stored["ids"], vectors))
    fresh = embedder.encode_documents([expected[doc_id] for doc_id in sample], batch_size=2)
    if len(fresh) != len(sample):
        raise ValueError("Embedder returned an unexpected number of vectors")
    similarities = []
    for doc_id, new in zip(sample, fresh):
        old = vectors[doc_id]
        if len(old) != 1024 or len(new) != 1024:
            raise ValueError(f"Embedding dimension differs from BGE-M3: {doc_id}")
        old, new = list(map(float, old)), list(map(float, new))
        if not all(math.isfinite(v) for v in old + new):
            raise ValueError(f"Non-finite embedding: {doc_id}")
        denominator = math.sqrt(sum(v*v for v in old)) * math.sqrt(sum(v*v for v in new))
        similarity = sum(a*b for a, b in zip(old, new)) / denominator if denominator else 0.0
        if similarity < 0.999:
            raise ValueError(f"Stored embedding incompatible with current BGE-M3 for {doc_id}: cosine={similarity:.6f}")
        similarities.append({"doc_id": doc_id, "cosine_similarity": similarity})
    return {"all_ids_and_texts_match": True, "document_count": len(expected),
            "embedding_sample_size": len(sample), "minimum_cosine_similarity": 0.999,
            "sample_selection": "first 16 document IDs ordered by SHA-256 of ID",
            "embedding_checks": similarities,
            "limitations": ["Embedding compatibility is sampled; unsampled vectors are not verified.",
                            "Historical model revision and build environment are unknown."]}


def check(root, load_retriever=False, register_existing_index=False):
    packages, problems = {}, []
    for name in ("torch", "chromadb", "FlagEmbedding", "pydantic-ai-slim", "openai", "fastmcp", "automas"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
            problems.append(f"Missing package: {name}")
    try:
        data, data_problems = inspect_data(root)
        problems.extend(data_problems)
    except (ValueError, KeyError, TypeError) as exc:
        data = {}
        problems.append(f"Invalid benchmark data: {exc}")
    index = root / "index" / "financebench"
    if not (index / "chroma.sqlite3").is_file():
        problems.append(f"Missing Chroma index: {index}")
    provenance = root / "index" / "index_provenance.json"
    needs_registration = not provenance.is_file() and register_existing_index
    if not provenance.is_file() and not needs_registration:
        problems.append("Missing index_provenance.json; verify the index before experiments")
    elif provenance.is_file():
        try:
            index_meta = json.loads(provenance.read_text())
            if index_meta.get("corpus_sha256") != digest(root / "corpus.jsonl"):
                problems.append("Index provenance does not match the current corpus")
            if index_meta.get("embedder") != "BAAI/bge-m3" or index_meta.get("collection") != "financebench":
                problems.append("Index embedder/collection differs from the comparison configuration")
        except (ValueError, AttributeError) as exc:
            problems.append(f"Invalid index provenance: {exc}")
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        problems.append("Required torch>=2.10 has no macOS x86_64 wheel; use the Linux run host")
    runtime = None
    verification = None
    if (load_retriever or register_existing_index) and not problems:
        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        previous = {k: os.environ.get(k) for k in keys}
        try:
            os.environ.update({k: "1" for k in keys})
            from marlib.retriever import Retriever, RetrieverSettings
            retriever = Retriever(RetrieverSettings(index_path=root / "index", collection="financebench",
                                                   embedder="BAAI/bge-m3", reranker="BAAI/bge-reranker-v2-m3"))
            if needs_registration:
                verification = verify_existing_index(retriever._collection, root / "corpus.jsonl", retriever._embedder)
            docs = retriever.search("What was the reported revenue?", top_k=2)
            runtime = {"document_count": retriever.document_count, "retrieved_ids": [d.doc_id for d in docs]}
            if not docs or retriever.document_count != data["corpus_count"]:
                problems.append("Retriever returned no passages or index document count differs from corpus")
            if needs_registration and not problems:
                if digest(root / "corpus.jsonl") != data["corpus.jsonl"]["sha256"]:
                    raise ValueError("Corpus changed during verification")
                metadata = {"corpus_sha256": data["corpus.jsonl"]["sha256"],
                            "embedder": "BAAI/bge-m3", "collection": "financebench", "distance": "cosine",
                            "document_count": retriever.document_count, "model_revision": None,
                            "provenance_kind": "verified_existing_index",
                            "verified_at": datetime.now(timezone.utc).isoformat(),
                            "verification_packages": packages, "verification": verification}
                # Never overwrite build-time metadata or an earlier verification.
                with provenance.open("x") as f:
                    f.write(json.dumps(metadata, indent=2) + "\n")
        except Exception as exc:
            problems.append(f"Retrieval runtime failed: {type(exc).__name__}: {exc}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return {"platform": platform.platform(), "python": platform.python_version(), "packages": packages,
            "data": data, "retrieval_smoke": runtime, "existing_index_verification": verification, "problems": problems,
            "ready": not problems and runtime is not None,
            "llm_calls": 0, "providers_checked": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/benchmarks/financebench"))
    parser.add_argument("--output", type=Path, default=Path("reports/finance_preflight.json"))
    parser.add_argument("--load-retriever", action="store_true", help="Load existing weights/index and run retrieval; no LLM calls")
    parser.add_argument("--register-existing-index", action="store_true",
                        help="Verify all stored texts/IDs and up to 16 embeddings, run retrieval, then write missing provenance; no LLM calls")
    args = parser.parse_args()
    report = check(args.root, args.load_retriever, args.register_existing_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready": report["ready"], "problems": report["problems"], "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["ready"] else 1)
