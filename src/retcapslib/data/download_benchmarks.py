"""Download and preprocess HotpotQA and MuSiQue benchmarks.

Creates stratified samples for evaluation:
- HotpotQA: 500 questions stratified by type (bridge/comparison)
- MuSiQue: 500 questions stratified by hop_count (2/3/4)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


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

    print("Loading HotpotQA fullwiki validation set...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

    # Separate by type (bridge vs comparison)
    bridge_questions = [ex for ex in ds if ex["type"] == "bridge"]
    comparison_questions = [ex for ex in ds if ex["type"] == "comparison"]

    print(f"Total questions: {len(ds)}")
    print(f"Bridge: {len(bridge_questions)}, Comparison: {len(comparison_questions)}")

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
    print(f"Saving {len(processed)} questions to: {output_file}")
    with open(output_file, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    print("HotpotQA download complete!")


def download_musique(
    output_dir: str | Path,
    sample_n: int = 500,
    seed: int = 42,
) -> None:
    """Download MuSiQue and create stratified sample by hop count.

    Args:
        output_dir: Directory to save processed benchmark.
        sample_n: Total number of questions to sample.
        seed: Random seed for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MuSiQue validation set...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    # Group by hop count
    by_hops: dict[int, list] = {}
    for ex in ds:
        # MuSiQue has n_hops field indicating reasoning depth
        n_hops = len(ex["question_decomposition"])
        if n_hops not in by_hops:
            by_hops[n_hops] = []
        by_hops[n_hops].append(ex)

    print(f"Total questions: {len(ds)}")
    for hops, questions in sorted(by_hops.items()):
        print(f"  {hops}-hop: {len(questions)}")

    # Stratified sampling proportional to original distribution
    import random

    random.seed(seed)

    total = sum(len(q) for q in by_hops.values())
    sampled = []

    for hops, questions in by_hops.items():
        # Proportional allocation
        n_sample = int(sample_n * len(questions) / total)
        n_sample = max(1, min(n_sample, len(questions)))
        sampled.extend(random.sample(questions, n_sample))

    # If we have fewer than target, sample more from largest group
    while len(sampled) < sample_n:
        largest_hops = max(by_hops.keys(), key=lambda h: len(by_hops[h]))
        remaining = [q for q in by_hops[largest_hops] if q not in sampled]
        if remaining:
            sampled.append(random.choice(remaining))
        else:
            break

    random.shuffle(sampled)
    sampled = sampled[:sample_n]

    # Convert to standard format
    processed = []
    for ex in sampled:
        processed.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "answer_aliases": ex.get("answer_aliases", []),
                "hop_count": len(ex["question_decomposition"]),
                "question_decomposition": ex["question_decomposition"],
                "paragraphs": ex.get("paragraphs", []),
            }
        )

    # Save
    output_file = output_dir / "musique_sample.jsonl"
    print(f"Saving {len(processed)} questions to: {output_file}")
    with open(output_file, "w") as f:
        for item in processed:
            f.write(json.dumps(item) + "\n")

    print("MuSiQue download complete!")


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
        help="Number of questions per benchmark",
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
        choices=["hotpotqa", "musique", "all"],
        default="all",
        help="Which benchmark to download",
    )

    args = parser.parse_args()

    if args.benchmark in ("hotpotqa", "all"):
        download_hotpotqa(args.output_dir, args.sample_n, args.seed)

    if args.benchmark in ("musique", "all"):
        download_musique(args.output_dir, args.sample_n, args.seed)


if __name__ == "__main__":
    main()
