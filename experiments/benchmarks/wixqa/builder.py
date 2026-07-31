"""WixQA benchmark builder.

Enterprise RAG over the Wix Help Center knowledge base (Cohen et al. 2025,
arXiv:2505.08643). ``download`` writes the WixQA-ExpertWritten split — 200
questions asked by real users and answered by Wix support experts — and
``build_corpus`` writes the accompanying knowledge base (6221 help articles) as
the retrieval corpus.

Unlike the Wikipedia benchmarks, the corpus is *not* derived from the sampled
questions: it is the whole knowledge base a support agent would search, and most
of it is a distractor for any given question. That is the point — it is the
enterprise retrieval setting FinanceBench also tests, in a non-financial domain.

Data access note
----------------
The dataset ships gold evidence as ``article_ids`` that are already the knowledge
base's own article ids, so ``gold_doc_ids`` are passed through verbatim and match
``doc_id`` by construction — no id derivation is involved on either side.

Every config's single split is named ``train`` on the Hub (the underlying files
are named ``test.jsonl``, but the config declares no split mapping).
"""

from __future__ import annotations

import json

from marlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register
from marlib.log import logger

_HF_DATASET = "Wix/WixQA"

# ExpertWritten only: 200 real user questions with expert-written answers.
# WixQA-Simulated (200, LLM-simulated user turns) and WixQA-Synthetic (6221,
# generated from articles) are deliberately left out — for an industrial claim
# real queries carry the argument, and the synthetic set leaks the corpus.
_QA_CONFIG = "wixqa_expertwritten"
_KB_CONFIG = "wix_kb_corpus"
_SPLIT = "train"


@register("wixqa")
class WixQABuilder(BenchmarkBuilder):
    """Builder for the WixQA enterprise support benchmark."""

    def download(self, spec: BenchmarkSpec) -> None:
        """Write the WixQA-ExpertWritten questions with their gold article ids.

        Args:
            spec: Benchmark spec (questions written to ``spec.questions_path``).
        """
        from datasets import load_dataset

        spec.questions_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {_HF_DATASET}/{_QA_CONFIG}...")
        ds = load_dataset(_HF_DATASET, _QA_CONFIG, split=_SPLIT)
        logger.info(f"Total examples: {len(ds)}")

        # The dataset carries no question id, so index within the split is the
        # id; the split is a fixed published file, so this is stable across runs.
        processed = [
            {
                "id": f"expertwritten_{i:04d}",
                "question": ex["question"],
                "answer": ex["answer"],
                # Already knowledge-base article ids — same space as doc_id.
                "gold_doc_ids": sorted(set(ex["article_ids"])),
            }
            for i, ex in enumerate(ds)
        ]

        logger.info(f"Saving {len(processed)} questions to: {spec.questions_path}")
        with open(spec.questions_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")
        logger.success("WixQA download complete!")

    def build_corpus(
        self,
        spec: BenchmarkSpec,
        max_paragraphs: int | None = None,
    ) -> None:
        """Write the Wix knowledge base as the retrieval corpus, one article per doc.

        Articles are kept whole rather than chunked: ``contents`` is plain text
        (the HTML original is dropped) and fits the embedder's 8192-token
        document window, so gold matching stays at article granularity, which is
        the granularity the dataset annotates.

        Args:
            spec: Benchmark spec (corpus written to ``spec.corpus_path``).
            max_paragraphs: Optional cap on number of articles (for testing).
        """
        from datasets import load_dataset

        spec.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {_HF_DATASET}/{_KB_CONFIG}...")
        kb = load_dataset(_HF_DATASET, _KB_CONFIG, split=_SPLIT)

        seen: set[str] = set()
        articles: list[dict] = []
        for art in kb:
            doc_id = art["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            articles.append(
                {
                    "doc_id": doc_id,
                    "title": art["title"],
                    "text": art["contents"],
                }
            )
            if max_paragraphs and len(articles) >= max_paragraphs:
                break

        logger.info(f"Collected {len(articles)} unique articles")
        logger.info(f"Saving corpus to: {spec.corpus_path}")
        with open(spec.corpus_path, "w") as f:
            for art in articles:
                f.write(json.dumps(art) + "\n")

        _check_gold_coverage(spec, seen)
        logger.success("WixQA corpus preparation complete!")


def _check_gold_coverage(spec: BenchmarkSpec, corpus_ids: set[str]) -> None:
    """Warn if any gold article id is absent from the corpus.

    ``context_recall`` is silently zero when gold and corpus id spaces disagree,
    so the mismatch is reported at build time rather than found in the results.
    """
    if not spec.questions_path.exists():
        return
    with open(spec.questions_path) as f:
        questions = [json.loads(line) for line in f]

    missing = {g for q in questions for g in q["gold_doc_ids"] if g not in corpus_ids}
    if missing:
        logger.warning(
            f"WixQA: {len(missing)} gold article id(s) are not in the corpus — "
            "context_recall will be understated. Example: "
            f"{sorted(missing)[0]}"
        )
    else:
        logger.info(f"WixQA: all gold ids of {len(questions)} questions are in corpus")
