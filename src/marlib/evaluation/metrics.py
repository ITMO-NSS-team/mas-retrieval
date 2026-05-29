"""Evaluation metrics for multi-hop QA.

Implements standard metrics from HotpotQA/MuSiQue evaluation:
- Exact Match (EM): Normalized string equality
- F1 Score: Token-level overlap
- Context Recall: Fraction of gold supporting paragraphs retrieved
"""

from __future__ import annotations

import re
import string
from collections import Counter


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
    gold_paragraphs: list[str],
) -> float:
    """Compute context recall: fraction of gold paragraphs retrieved.

    For multi-hop QA, gold paragraphs are the supporting facts needed
    to answer the question. This metric measures how well the retriever
    surfaces the required evidence.

    Args:
        retrieved_doc_ids: List of retrieved document IDs.
        gold_paragraphs: List of gold supporting paragraph identifiers
            (typically Wikipedia article titles from HotpotQA).

    Returns:
        Recall score between 0 and 1.
    """
    if not gold_paragraphs:
        return 1.0  # No gold paragraphs needed

    retrieved_set = set(retrieved_doc_ids)

    # For HotpotQA, gold paragraphs are article titles
    # We check if any retrieved doc starts with the gold title
    hits = 0
    for gold in gold_paragraphs:
        gold_normalized = gold.replace(" ", "_")
        for doc_id in retrieved_set:
            # doc_id format: Title_hash, so check if it starts with gold title
            if doc_id.startswith(gold_normalized + "_") or doc_id == gold_normalized:
                hits += 1
                break

    return hits / len(gold_paragraphs)


def evaluate_question(
    predicted_answer: str,
    gold_answer: str,
    retrieved_doc_ids: list[str] | None = None,
    gold_paragraphs: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a single question across all metrics.

    Args:
        predicted_answer: System's predicted answer.
        gold_answer: Ground truth answer.
        retrieved_doc_ids: Optional list of retrieved document IDs.
        gold_paragraphs: Optional list of gold supporting paragraphs.

    Returns:
        Dictionary with em, f1, and optionally context_recall scores.
    """
    results = {
        "exact_match": exact_match(predicted_answer, gold_answer),
        "f1": f1_score(predicted_answer, gold_answer),
    }

    if retrieved_doc_ids is not None and gold_paragraphs is not None:
        results["context_recall"] = context_recall(retrieved_doc_ids, gold_paragraphs)

    return results
