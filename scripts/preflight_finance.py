"""Check the FinanceBench runtime/data without LLM calls or automatic downloads."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
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


def check(root, load_retriever=False):
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
    if not provenance.is_file():
        problems.append("Missing index_provenance.json; verify the index before experiments")
    else:
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
    if load_retriever and not problems:
        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        previous = {k: os.environ.get(k) for k in keys}
        try:
            os.environ.update({k: "1" for k in keys})
            from marlib.retriever import Retriever, RetrieverSettings
            retriever = Retriever(RetrieverSettings(index_path=root / "index", collection="financebench"))
            docs = retriever.search("What was the reported revenue?", top_k=2)
            runtime = {"document_count": retriever.document_count, "retrieved_ids": [d.doc_id for d in docs]}
            if not docs or retriever.document_count != data["corpus_count"]:
                problems.append("Retriever returned no passages or index document count differs from corpus")
        except Exception as exc:
            problems.append(f"Retrieval runtime failed: {type(exc).__name__}: {exc}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return {"platform": platform.platform(), "python": platform.python_version(), "packages": packages,
            "data": data, "retrieval_smoke": runtime, "problems": problems,
            "ready": not problems and runtime is not None,
            "llm_calls": 0, "providers_checked": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/benchmarks/financebench"))
    parser.add_argument("--output", type=Path, default=Path("reports/finance_preflight.json"))
    parser.add_argument("--load-retriever", action="store_true", help="Load existing weights/index and run retrieval; no LLM calls")
    args = parser.parse_args()
    report = check(args.root, args.load_retriever)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready": report["ready"], "problems": report["problems"], "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["ready"] else 1)
