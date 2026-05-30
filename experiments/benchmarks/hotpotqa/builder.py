"""HotpotQA benchmark builder.

Multi-hop QA over Wikipedia (fullwiki). ``download`` creates a stratified
(bridge/comparison) question sample; ``build_corpus`` extracts unique Wikipedia
paragraphs from the HotpotQA contexts as the retrieval corpus.
"""

from __future__ import annotations

import hashlib
import json
import random

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register
from marlib.log import logger


@register("hotpotqa")
class HotpotQABuilder(BenchmarkBuilder):
    """Builder for the HotpotQA fullwiki benchmark."""

    def download(
        self,
        spec: BenchmarkSpec,
        sample_n: int = 500,
        seed: int = 42,
    ) -> None:
        """Download HotpotQA fullwiki and write a stratified question sample.

        Args:
            spec: Benchmark spec (questions written to ``spec.questions_path``).
            sample_n: Total number of questions to sample (split evenly by type).
            seed: Random seed for reproducibility.
        """
        from datasets import load_dataset

        spec.questions_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading HotpotQA fullwiki validation set...")
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

        bridge = [ex for ex in ds if ex["type"] == "bridge"]
        comparison = [ex for ex in ds if ex["type"] == "comparison"]
        logger.info(f"Total questions: {len(ds)}")
        logger.info(f"Bridge: {len(bridge)}, Comparison: {len(comparison)}")

        random.seed(seed)
        n_per_type = sample_n // 2
        sampled = random.sample(bridge, min(n_per_type, len(bridge))) + random.sample(
            comparison, min(n_per_type, len(comparison))
        )
        random.shuffle(sampled)

        processed = [
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
                # Gold evidence in the corpus id space (doc_id = "<Title>_<hash>"),
                # so context_recall hits via the title prefix.
                "gold_doc_ids": sorted(
                    {t.replace(" ", "_") for t in ex["supporting_facts"]["title"]}
                ),
            }
            for ex in sampled
        ]

        logger.info(f"Saving {len(processed)} questions to: {spec.questions_path}")
        with open(spec.questions_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        logger.success("HotpotQA download complete!")

    def build_corpus(
        self,
        spec: BenchmarkSpec,
        max_paragraphs: int | None = None,
    ) -> None:
        """Build the Wikipedia paragraph corpus from HotpotQA fullwiki contexts.

        Args:
            spec: Benchmark spec (corpus written to ``spec.corpus_path``).
            max_paragraphs: Optional cap on number of paragraphs (for testing).
        """
        from datasets import load_dataset

        spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading HotpotQA fullwiki dataset...")
        train_ds = load_dataset("hotpot_qa", "fullwiki", split="train")
        val_ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

        seen_ids: set[str] = set()
        paragraphs: list[dict] = []

        def process_context(context: dict) -> None:
            for title, sentences in zip(context["title"], context["sentences"]):
                text = " ".join(sentences)
                content_hash = hashlib.md5((title + text).encode()).hexdigest()[:12]
                doc_id = f"{title.replace(' ', '_')}_{content_hash}"
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                paragraphs.append({"doc_id": doc_id, "title": title, "text": text})

        from tqdm import tqdm

        logger.info("Processing training set contexts...")
        for example in tqdm(train_ds, desc="Train"):
            process_context(example["context"])
            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

        if not max_paragraphs or len(paragraphs) < max_paragraphs:
            logger.info("Processing validation set contexts...")
            for example in tqdm(val_ds, desc="Validation"):
                process_context(example["context"])
                if max_paragraphs and len(paragraphs) >= max_paragraphs:
                    break

        if max_paragraphs:
            paragraphs = paragraphs[:max_paragraphs]

        logger.info(f"Collected {len(paragraphs)} unique paragraphs")
        logger.info(f"Saving corpus to: {spec.corpus_path}")
        with open(spec.corpus_path, "w") as f:
            for para in tqdm(paragraphs, desc="Writing"):
                f.write(json.dumps(para) + "\n")
        logger.success("HotpotQA corpus preparation complete!")
