"""MuSiQue benchmark builder.

Strict multi-hop QA over Wikipedia (MuSiQue-Answerable). ``download`` draws a
*stratified-by-hop-count* question sample (2/3/4 hops, like the HotpotQA builder
stratifies bridge/comparison) and records the gold supporting paragraphs as
``gold_doc_ids``. ``build_corpus`` collects every paragraph attached to the
sampled questions — the 2-4 supporting ones plus MuSiQue's own per-question
distractors — as the retrieval corpus, so ``doc_id`` matches between gold and
corpus and ``context_recall`` hits exactly.

Data access note
-----------------
MuSiQue is loaded from the ``dgslibisey/MuSiQue`` HuggingFace mirror of the
official StonyBrookNLP/musique (TACL 2022) release. Each example carries 20
paragraphs flagged with ``is_supporting``, the final ``answer`` + ``answer_aliases``,
and an ``id`` whose prefix encodes the hop count (``2hop__``, ``3hop1__``, ...).
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register
from marlib.log import logger

_HF_DATASET = "dgslibisey/MuSiQue"

# Hop counts to sample, split evenly (cf. HotpotQA bridge/comparison split).
HOPS: tuple[int, ...] = (2, 3, 4)
SAMPLE_N = 500
SEED = 42

# Example ids look like "2hop__292995_8796" / "3hop1__..." / "4hop3__...".
_HOP_RE = re.compile(r"^(\d+)hop")


def _hop_count(qid: str) -> int | None:
    """Parse the leading hop count from a MuSiQue example id."""
    m = _HOP_RE.match(qid)
    return int(m.group(1)) if m else None


def _doc_id(title: str, text: str) -> str:
    """Stable corpus id for a paragraph (same scheme as the HotpotQA builder).

    Content-hashed so the *same* paragraph appearing under several questions
    dedups to one corpus entry, and so ``download`` and ``build_corpus`` derive
    identical ids for gold/corpus matching.
    """
    content_hash = hashlib.md5((title + text).encode()).hexdigest()[:12]
    return f"{title.replace(' ', '_')}_{content_hash}"


@register("musique")
class MuSiQueBuilder(BenchmarkBuilder):
    """Builder for the MuSiQue-Answerable multi-hop benchmark."""

    def download(
        self,
        spec: BenchmarkSpec,
        sample_n: int = SAMPLE_N,
        seed: int = SEED,
        hops: tuple[int, ...] = HOPS,
    ) -> None:
        """Write a per-hop-count stratified question sample.

        Args:
            spec: Benchmark spec (questions written to ``spec.questions_path``).
            sample_n: Total questions, split evenly across ``hops``.
            seed: Random seed for reproducibility.
            hops: Hop counts to keep (default: 2/3/4-hop questions).
        """
        from datasets import load_dataset

        spec.questions_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading MuSiQue-Answerable validation set...")
        ds = load_dataset(_HF_DATASET, split="validation")

        by_hop: dict[int, list[dict]] = {h: [] for h in hops}
        for ex in ds:
            h = _hop_count(ex["id"])
            if h in by_hop:
                by_hop[h].append(ex)
        for h, exs in by_hop.items():
            logger.info(f"{h}-hop: {len(exs)} available")

        random.seed(seed)
        n_per_hop = sample_n // len(hops)
        sampled: list[dict] = []
        for h in hops:
            pool = by_hop[h]
            sampled += random.sample(pool, min(n_per_hop, len(pool)))
        random.shuffle(sampled)

        processed: list[dict] = []
        for ex in sampled:
            gold_doc_ids = sorted(
                {
                    _doc_id(p["title"], p["paragraph_text"])
                    for p in ex["paragraphs"]
                    if p["is_supporting"]
                }
            )
            processed.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "answer_aliases": list(ex.get("answer_aliases") or []),
                    "hops": _hop_count(ex["id"]),
                    # Gold evidence in corpus id space (doc_id == paragraph id).
                    "gold_doc_ids": gold_doc_ids,
                }
            )

        logger.info(f"Saving {len(processed)} questions to: {spec.questions_path}")
        with open(spec.questions_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        logger.success("MuSiQue download complete!")

    def build_corpus(
        self,
        spec: BenchmarkSpec,
        max_paragraphs: int | None = None,
    ) -> None:
        """Collect the sampled questions' paragraphs as the retrieval corpus.

        The corpus is the union of all paragraphs (supporting + MuSiQue's own
        per-question distractors) across the sampled questions, so every
        ``gold_doc_id`` is guaranteed present and other questions' paragraphs act
        as cross-question distractors.

        Args:
            spec: Benchmark spec (corpus written to ``spec.corpus_path``).
            max_paragraphs: Optional cap on number of paragraphs (for testing).
        """
        from datasets import load_dataset
        from tqdm import tqdm

        if not spec.questions_path.exists():
            logger.warning(
                "musique: no questions.jsonl (download skipped). Skipping corpus."
            )
            return
        spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(spec.questions_path) as f:
            sampled_ids = {json.loads(line)["id"] for line in f}
        logger.info(
            f"Collecting paragraphs for {len(sampled_ids)} sampled questions..."
        )

        ds = load_dataset(_HF_DATASET, split="validation")
        seen_ids: set[str] = set()
        paragraphs: list[dict] = []
        for ex in tqdm(ds, desc="Collecting"):
            if ex["id"] not in sampled_ids:
                continue
            for p in ex["paragraphs"]:
                doc_id = _doc_id(p["title"], p["paragraph_text"])
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                paragraphs.append(
                    {
                        "doc_id": doc_id,
                        "title": p["title"],
                        "text": p["paragraph_text"],
                    }
                )
            if max_paragraphs and len(paragraphs) >= max_paragraphs:
                break

        if max_paragraphs:
            paragraphs = paragraphs[:max_paragraphs]

        logger.info(f"Collected {len(paragraphs)} unique paragraphs")
        logger.info(f"Saving corpus to: {spec.corpus_path}")
        with open(spec.corpus_path, "w") as f:
            for para in paragraphs:
                f.write(json.dumps(para) + "\n")
        logger.success("MuSiQue corpus preparation complete!")
