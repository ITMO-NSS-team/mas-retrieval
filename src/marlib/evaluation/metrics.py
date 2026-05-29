from __future__ import annotations

import re
import string
from collections import Counter

from marlib.evaluation.base import EvalContext, register_metric


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison.

    Following HotpotQA/SQuAD normalization:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Collapse whitespace

    Args:
        text: Raw answer text.

    Returns:
        Normalized answer string.
    """
    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = " ".join(text.split())

    return text


def exact_match(pred: str, gold: str) -> float:
    """Compute exact match score.

    Args:
        pred: Predicted answer.
        gold: Gold answer.

    Returns:
        1.0 if normalized answers match exactly, 0.0 otherwise.
    """
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Compute token-level F1 score.

    Args:
        pred: Predicted answer.
        gold: Gold answer.

    Returns:
        F1 score between 0 and 1.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)

    if not pred_tokens:
        return 0.0

    # Count common tokens
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def context_recall(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """Fraction of gold doc_ids surfaced by the retriever.

    A gold id "hits" if some retrieved id equals it or is one of its parts
    (``retrieved.startswith(gold + "_")``). This covers both exact ids (e.g.
    FinanceBench ``slug_p12``) and grouped ids (e.g. HotpotQA ``Title_<hash>``,
    where the gold is the title prefix) without the metric knowing which
    benchmark it is scoring — the builder emits ``gold_doc_ids`` in the corpus's
    own id space.

    Args:
        retrieved_doc_ids: Document IDs returned by retrieval tool calls.
        gold_doc_ids: Gold supporting document IDs for the question.

    Returns:
        Recall score between 0 and 1.
    """
    retrieved = set(retrieved_doc_ids)
    hits = sum(
        any(r == gold or r.startswith(gold + "_") for r in retrieved)
        for gold in gold_doc_ids
    )
    return hits / len(gold_doc_ids)


# --- Registered metrics (uniform metric(ctx) -> float | None signature) -------


@register_metric("exact_match")
def _exact_match(ctx: EvalContext) -> float:
    return exact_match(ctx.predicted, ctx.gold)


@register_metric("f1")
def _f1(ctx: EvalContext) -> float:
    return f1_score(ctx.predicted, ctx.gold)


@register_metric("context_recall")
def _context_recall(ctx: EvalContext) -> float | None:
    if not ctx.gold_doc_ids:
        return None  # not applicable without gold evidence
    return context_recall(ctx.retrieved_doc_ids, ctx.gold_doc_ids)
