"""Prepare corpora for HotpotQA and FinanceBench benchmarks.

HotpotQA:     Extracts paragraphs from HotpotQA fullwiki context (~5M paragraphs).
FinanceBench: Extracts all pages from SEC filing PDFs using PyMuPDF (~15K-25K pages).
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
    pdf_dir: str | Path = "experiments/data/financebench_pdfs",
    benchmark_path: str | Path = "experiments/data/benchmarks/financebench_sample.jsonl",
    max_paragraphs: int | None = None,
) -> None:
    """Prepare corpus from FinanceBench SEC filing PDFs.

    Extracts text from every page of each PDF in pdf_dir using PyMuPDF,
    creating a realistic retrieval corpus (~15K-25K pages) instead of only
    the ~150 evidence pages from the benchmark dataset.

    Args:
        output_path: Path to write financebench_paragraphs.jsonl.
        pdf_dir: Directory containing downloaded FinanceBench PDFs.
        benchmark_path: Path to the downloaded financebench_sample.jsonl
            (used for metadata enrichment).
        max_paragraphs: Optional limit on number of pages (for testing).
    """
    import pymupdf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(pdf_dir)
    benchmark_path = Path(benchmark_path)

    # Build metadata mapping from benchmark JSONL: doc_name -> metadata
    metadata_map: dict[str, dict] = {}
    if benchmark_path.exists():
        print(f"Loading metadata from: {benchmark_path}")
        with open(benchmark_path) as f:
            for line in f:
                entry = json.loads(line)
                doc_name = entry.get("doc_name")
                if doc_name and doc_name not in metadata_map:
                    metadata_map[doc_name] = {
                        "company": entry.get("company"),
                        "doc_type": entry.get("doc_type"),
                        "doc_period": entry.get("doc_period"),
                        "gics_sector": entry.get("gics_sector"),
                    }
        print(f"Loaded metadata for {len(metadata_map)} documents")

    # Iterate PDFs and extract text page-by-page
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: no PDF files found in {pdf_dir}")
        return

    print(f"Processing {len(pdf_files)} PDFs from: {pdf_dir}")

    seen_ids: set[str] = set()
    paragraphs = []
    skipped_encrypted = 0
    skipped_empty = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        doc_name = pdf_path.stem

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            print(f"\n  Warning: could not open {pdf_path.name}: {e}")
            continue

        if doc.is_encrypted:
            print(f"\n  Warning: skipping encrypted PDF: {pdf_path.name}")
            skipped_encrypted += 1
            doc.close()
            continue

        meta = metadata_map.get(doc_name, {})

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")

            # Skip near-empty pages (covers, image-only pages, etc.)
            if len(text.strip()) < 50:
                skipped_empty += 1
                continue

            # 1-based page numbering to match HF dataset evidence_page_num
            page_number = page_idx + 1
            doc_id = f"{_slugify(doc_name)}_p{page_number}"

            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            title = f"{doc_name} (p. {page_number})"

            entry = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
            }
            # Add metadata if available
            for key in ("company", "doc_type", "doc_period", "gics_sector"):
                if meta.get(key):
                    entry[key] = meta[key]

            paragraphs.append(entry)

            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

        doc.close()

        if max_paragraphs and len(paragraphs) >= max_paragraphs:
            break

    print(f"Collected {len(paragraphs)} pages from {len(pdf_files)} PDFs")
    if skipped_encrypted:
        print(f"  Skipped {skipped_encrypted} encrypted PDFs")
    if skipped_empty:
        print(f"  Skipped {skipped_empty} near-empty pages")

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
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="experiments/data/financebench_pdfs",
        help="Directory containing FinanceBench PDFs (for financebench corpus)",
    )

    args = parser.parse_args()

    if args.dataset in ("hotpotqa", "all"):
        out = args.output if args.dataset == "hotpotqa" else None
        out = out or DEFAULT_PATHS["hotpotqa"]
        prepare_hotpotqa_corpus(output_path=out, max_paragraphs=args.max_paragraphs)

    if args.dataset in ("financebench", "all"):
        out = args.output if args.dataset == "financebench" else None
        out = out or DEFAULT_PATHS["financebench"]
        kwargs: dict = {
            "output_path": out,
            "max_paragraphs": args.max_paragraphs,
            "pdf_dir": args.pdf_dir,
        }
        if args.benchmark_path:
            kwargs["benchmark_path"] = args.benchmark_path
        prepare_financebench_corpus(**kwargs)


if __name__ == "__main__":
    main()
