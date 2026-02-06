"""Prepare Wikipedia corpora for HotpotQA and MuSiQue benchmarks.

HotpotQA: Extracts paragraphs from HotpotQA fullwiki context (~5M paragraphs).
MuSiQue:  Extracts and deduplicates paragraphs from MuSiQue dataset splits
          following IRCoT methodology (~139K paragraphs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Default output paths per dataset
DEFAULT_PATHS = {
    "hotpotqa": "experiments/data/corpus/hotpotqa_paragraphs.jsonl",
    "musique": "experiments/data/corpus/musique_paragraphs.jsonl",
}


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


def prepare_musique_corpus(
    output_path: str | Path,
    max_paragraphs: int | None = None,
) -> None:
    """Prepare Wikipedia corpus from MuSiQue dataset paragraphs.

    Following IRCoT methodology: extract and deduplicate paragraphs from
    the MuSiQue dataset's 'paragraphs' field across all available splits.
    This produces ~139K unique Wikipedia paragraphs.

    Args:
        output_path: Path to write musique_paragraphs.jsonl.
        max_paragraphs: Optional limit on number of paragraphs (for testing).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading MuSiQue dataset...")
    # Load all available splits to maximize paragraph coverage
    splits = ["train", "validation"]
    all_examples = []
    for split in splits:
        try:
            ds = load_dataset("dgslibisey/MuSiQue", split=split)
            all_examples.extend(ds)
            print(f"  Loaded {split}: {len(ds)} examples")
        except Exception as e:
            print(f"  Skipping {split}: {e}")

    # Extract and deduplicate paragraphs (IRCoT approach)
    seen_hashes = set()
    paragraphs = []

    for example in tqdm(all_examples, desc="Extracting paragraphs"):
        for para in example.get("paragraphs", []):
            title = para.get("title", "")
            text = para.get("paragraph_text", "")

            if not text.strip():
                continue

            # Hash title+text for deduplication (matching IRCoT)
            content_hash = hashlib.md5((title + text).encode()).hexdigest()

            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # Use hash prefix as doc_id (consistent, unique)
            doc_id = f"{title.replace(' ', '_')}_{content_hash[:12]}"

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

    print(f"Collected {len(paragraphs)} unique paragraphs")

    # Save as JSONL
    print(f"Saving corpus to: {output_path}")
    with open(output_path, "w") as f:
        for para in tqdm(paragraphs, desc="Writing"):
            f.write(json.dumps(para) + "\n")

    print("MuSiQue corpus preparation complete!")


# Keep backward-compatible alias
prepare_corpus = prepare_hotpotqa_corpus


def main() -> None:
    """CLI entry point for corpus preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia corpus for HotpotQA and/or MuSiQue"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpotqa", "musique", "all"],
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

    args = parser.parse_args()

    if args.dataset in ("hotpotqa", "all"):
        out = args.output if args.dataset == "hotpotqa" else None
        out = out or DEFAULT_PATHS["hotpotqa"]
        prepare_hotpotqa_corpus(output_path=out, max_paragraphs=args.max_paragraphs)

    if args.dataset in ("musique", "all"):
        out = args.output if args.dataset == "musique" else None
        out = out or DEFAULT_PATHS["musique"]
        prepare_musique_corpus(output_path=out, max_paragraphs=args.max_paragraphs)


if __name__ == "__main__":
    main()
