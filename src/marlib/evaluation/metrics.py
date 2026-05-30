from __future__ import annotations

import re
import string
from collections import Counter

from marlib.evaluation.base import EvalContext, register_metric


def normalize_answer(text: str) -> str:
    """HotpotQA/SQuAD normalization: lowercase, drop articles & punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between normalized answers."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def context_recall(
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
) -> float:
    """Fraction of gold doc_ids surfaced by the retriever.

    A gold id hits when a retrieved id equals it or is one of its parts
    (``retrieved.startswith(gold + "_")``) — covering both exact ids
    (``slug_p12``) and grouped ids (``Title_<hash>``) in one benchmark-agnostic rule.
    """
    retrieved = set(retrieved_doc_ids)
    hits = sum(
        any(r == gold or r.startswith(gold + "_") for r in retrieved)
        for gold in gold_doc_ids
    )
    return hits / len(gold_doc_ids)


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
