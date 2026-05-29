"""Download and preprocess HotpotQA and FinanceBench benchmarks.

Creates samples for evaluation:
- HotpotQA: 500 questions stratified by type (bridge/comparison)
- FinanceBench: 150 financial QA examples from SEC filings (full dataset)
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from marlib.log import logger


def download_hotpotqa(
    output_dir: str | Path,
    sample_n: int = 500,
    seed: int = 42,
) -> None:
    """Download HotpotQA fullwiki and create stratified sample.

    Args:
        output_dir: Directory to save processed benchmark.
        sample_n: Total number of questions to sample.
        seed: Random seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HotpotQA fullwiki validation set...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

    # Separate by type (bridge vs comparison)
    bridge_questions = [ex for ex in ds if ex["type"] == "bridge"]
    comparison_questions = [ex for ex in ds if ex["type"] == "comparison"]

    logger.info(f"Total questions: {len(ds)}")
    logger.info(f"Bridge: {len(bridge_questions)}, Comparison: {len(comparison_questions)}")

    # Stratified sampling: 250 each
    import random

    random.seed(seed)

    n_per_type = sample_n // 2
    sampled_bridge = random.sample(
        bridge_questions, min(n_per_type, len(bridge_questions))
    )
    sampled_comparison = random.sample(
        comparison_questions, min(n_per_type, len(comparison_questions))
    )

    sampled = sampled_bridge + sampled_comparison
    random.shuffle(sampled)

    # Convert to standard format
    processed = []
    for ex in sampled:
        processed.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "supporting_facts": {
                    "titles": ex["supporting_facts"]["title"],
                    "sent_ids": ex["supporting_facts"]["sent_id"],
                },
            }
        )

    # Save
    output_file = output_dir / "hotpotqa_sample.jsonl"
    logger.info(f"Saving {len(processed)} questions to: {output_file}")
    with open(output_file, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    logger.success("HotpotQA download complete!")


def download_financebench(output_dir: str | Path) -> None:
    """Download FinanceBench dataset (all 150 examples).

    FinanceBench contains 150 financial QA examples with evidence from
    SEC filings (10-K, 10-Q reports).

    Args:
        output_dir: Directory to save processed benchmark.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FinanceBench dataset...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    logger.info(f"Total examples: {len(ds)}")

    # Convert to standard format
    processed = []
    for ex in ds:
        processed.append(
            {
                "id": ex["financebench_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "evidence": ex["evidence"],
                "company": ex.get("company"),
                "doc_name": ex.get("doc_name"),
                "question_type": ex.get("question_type"),
                "question_reasoning": ex.get("question_reasoning"),
                "justification": ex.get("justification"),
                "gics_sector": ex.get("gics_sector"),
                "doc_type": ex.get("doc_type"),
                "doc_period": ex.get("doc_period"),
            }
        )

    # Save
    output_file = output_dir / "financebench_sample.jsonl"
    logger.info(f"Saving {len(processed)} questions to: {output_file}")
    with open(output_file, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    logger.success("FinanceBench download complete!")


def download_financebench_pdfs(
    pdf_dir: str | Path = "experiments/data/financebench_pdfs",
    benchmark_path: str | Path = "experiments/data/benchmarks/financebench_sample.jsonl",
) -> None:
    """Download SEC filing PDFs referenced by the FinanceBench dataset.

    Reads the benchmark JSONL to get unique doc_name values, then downloads
    each PDF from the official patronus-ai/financebench GitHub repository.
    Files that already exist locally are skipped.

    Args:
        pdf_dir: Directory to save downloaded PDFs.
        benchmark_path: Path to the downloaded financebench_sample.jsonl.
    """
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(benchmark_path)

    # Read benchmark to get unique doc_name values
    logger.info(f"Loading benchmark from: {benchmark_path}")
    doc_names: set[str] = set()
    with open(benchmark_path) as f:
        for line in f:
            entry = json.loads(line)
            doc_name = entry.get("doc_name")
            if doc_name:
                doc_names.add(doc_name)

    logger.info(f"Found {len(doc_names)} unique documents to download")

    base_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"

    # Optional GitHub token for rate limits
    github_token = os.environ.get("GITHUB_TOKEN")

    downloaded = 0
    skipped = 0
    failed = 0

    for doc_name in tqdm(sorted(doc_names), desc="Downloading PDFs"):
        pdf_path = pdf_dir / f"{doc_name}.pdf"

        if pdf_path.exists():
            skipped += 1
            continue

        encoded_name = urllib.parse.quote(f"{doc_name}.pdf")
        url = f"{base_url}/{encoded_name}"

        try:
            req = urllib.request.Request(url)
            if github_token:
                req.add_header("Authorization", f"token {github_token}")
            with urllib.request.urlopen(req, timeout=60) as resp:
                pdf_path.write_bytes(resp.read())
            downloaded += 1
        except Exception as e:
            logger.warning(f"Failed to download {doc_name}: {e}")
            failed += 1

    logger.success(
        f"Done: {downloaded} downloaded, {skipped} skipped, {failed} failed"
    )


def main() -> None:
    """CLI entry point for downloading benchmarks."""
    parser = argparse.ArgumentParser(description="Download and preprocess benchmarks")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/data/benchmarks",
        help="Output directory for benchmark files",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=500,
        help="Number of questions per benchmark (HotpotQA only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["hotpotqa", "financebench", "financebench-pdfs", "all"],
        default="all",
        help="Which benchmark to download",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="experiments/data/financebench_pdfs",
        help="Directory to save FinanceBench PDFs",
    )

    args = parser.parse_args()

    if args.benchmark in ("hotpotqa", "all"):
        download_hotpotqa(args.output_dir, args.sample_n, args.seed)

    if args.benchmark in ("financebench", "all"):
        download_financebench(args.output_dir)

    if args.benchmark == "financebench-pdfs":
        benchmark_path = Path(args.output_dir) / "financebench_sample.jsonl"
        download_financebench_pdfs(
            pdf_dir=args.pdf_dir,
            benchmark_path=benchmark_path,
        )


if __name__ == "__main__":
    main()
