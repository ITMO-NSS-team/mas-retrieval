"""Prepare corpora for HotpotQA and FinanceBench benchmarks.

HotpotQA:     Extracts paragraphs from HotpotQA fullwiki context (~5M paragraphs).
FinanceBench: Extracts unique evidence pages from the downloaded benchmark JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Default output paths per dataset
DEFAULT_PATHS = {
    "hotpotqa": "experiments/data/corpus/hotpotqa_paragraphs.jsonl",
    "financebench": "experiments/data/corpus/financebench_paragraphs.jsonl",
}


def _slugify(text: str) -> str:
    """Lowercase text and replace non-alphanumeric characters with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def prepare_hotpotqa_corpus(
    output_path: str | Path,
    max_paragraphs: int | None = None,
) -> None:
    """Prepare Wikipedia corpus from HotpotQA fullwiki.

    HotpotQA fullwiki provides ~5M paragraphs from Wikipedia.
    Each paragraph includes title and sentences.

    Args:
        output_path: Path to write hotpotqa_paragraphs.jsonl.
        max_paragraphs: Optional limit on number of paragraphs (for testing).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA fullwiki dataset...")
    # Load both train and validation to get maximum paragraph coverage
    train_ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    val_ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

    # Collect unique paragraphs from context
    seen_ids = set()
    paragraphs = []

    def process_context(context: dict) -> None:
        """Extract paragraphs from HotpotQA context field."""
        titles = context["title"]
        sentences_list = context["sentences"]

        for title, sentences in zip(titles, sentences_list):
            # Join sentences into paragraph text
            text = " ".join(sentences)

            # Create unique ID from title + text hash
            content_hash = hashlib.md5((title + text).encode()).hexdigest()[:12]
            doc_id = f"{title.replace(' ', '_')}_{content_hash}"

            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            paragraphs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                }
            )

            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                return

    print("Processing training set contexts...")
    for example in tqdm(train_ds, desc="Train"):
        process_context(example["context"])
        if max_paragraphs and len(paragraphs) >= max_paragraphs:
            break

    if not max_paragraphs or len(paragraphs) < max_paragraphs:
        print("Processing validation set contexts...")
        for example in tqdm(val_ds, desc="Validation"):
            process_context(example["context"])
            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

    print(f"Collected {len(paragraphs)} unique paragraphs")

    # Save as JSONL
    print(f"Saving corpus to: {output_path}")
    with open(output_path, "w") as f:
        for para in tqdm(paragraphs, desc="Writing"):
            f.write(json.dumps(para) + "\n")

    print("HotpotQA corpus preparation complete!")


def prepare_financebench_corpus(
    output_path: str | Path,
    benchmark_path: str | Path = "experiments/data/benchmarks/financebench_sample.jsonl",
    max_paragraphs: int | None = None,
) -> None:
    """Prepare corpus from FinanceBench evidence pages.

    Extracts unique pages from the evidence field across all 150 questions.
    Deduplicates by (doc_name, evidence_page_num) tuple.

    Args:
        output_path: Path to write financebench_paragraphs.jsonl.
        benchmark_path: Path to the downloaded financebench_sample.jsonl.
        max_paragraphs: Optional limit on number of pages (for testing).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_path)

    print(f"Loading FinanceBench benchmark from: {benchmark_path}")
    questions = []
    with open(benchmark_path) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions")

    # Extract unique evidence pages
    seen_pages: set[tuple[str, int]] = set()
    paragraphs = []

    for q in tqdm(questions, desc="Extracting evidence pages"):
        evidence_list = q.get("evidence", [])
        if not isinstance(evidence_list, list):
            evidence_list = [evidence_list]

        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue

            doc_name = ev.get("doc_name", q.get("doc_name", ""))
            page_num = ev.get("evidence_page_num")
            text = ev.get("evidence_text_full_page", "")

            if not doc_name or page_num is None or not text.strip():
                continue

            key = (doc_name, int(page_num))
            if key in seen_pages:
                continue
            seen_pages.add(key)

            doc_id = f"{_slugify(doc_name)}_p{page_num}"
            title = f"{doc_name} (p. {page_num})"

            paragraphs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                }
            )

            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

        if max_paragraphs and len(paragraphs) >= max_paragraphs:
            break

    print(f"Collected {len(paragraphs)} unique evidence pages")

    # Save as JSONL
    print(f"Saving corpus to: {output_path}")
    with open(output_path, "w") as f:
        for para in tqdm(paragraphs, desc="Writing"):
            f.write(json.dumps(para) + "\n")

    print("FinanceBench corpus preparation complete!")


def main() -> None:
    """CLI entry point for corpus preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare corpus for HotpotQA and/or FinanceBench"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotqa", "financebench", "all"],
        default="all",
        help="Which dataset corpus to prepare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSONL corpus (overrides default per-dataset path)",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=None,
        help="Optional limit on paragraphs (for testing)",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help="Path to financebench_sample.jsonl (for financebench corpus)",
    )

    args = parser.parse_args()

    if args.dataset in ("hotpotqa", "all"):
        out = args.output if args.dataset == "hotpotqa" else None
        out = out or DEFAULT_PATHS["hotpotqa"]
        prepare_hotpotqa_corpus(output_path=out, max_paragraphs=args.max_paragraphs)

    if args.dataset in ("financebench", "all"):
        out = args.output if args.dataset == "financebench" else None
        out = out or DEFAULT_PATHS["financebench"]
        kwargs = {"output_path": out, "max_paragraphs": args.max_paragraphs}
        if args.benchmark_path:
            kwargs["benchmark_path"] = args.benchmark_path
        prepare_financebench_corpus(**kwargs)


if __name__ == "__main__":
    main()
