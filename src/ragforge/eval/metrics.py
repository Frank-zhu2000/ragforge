"""Standard information retrieval and RAG evaluation metrics."""

import math
import re
from typing import List, Set, Union


def hit_rate(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Returns 1.0 if at least one ground-truth item is in top-k retrieved, else 0.0."""
    top_k = retrieved[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Precision@k = (Number of relevant items in top-k) / k."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item in ground_truth)
    return relevant_count / float(k)


def recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Recall@k = (Number of relevant items in top-k) / (Total relevant items)."""
    if not ground_truth:
        return 0.0
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item in ground_truth)
    return relevant_count / float(len(ground_truth))


def mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
    """Mean Reciprocal Rank (MRR): 1.0 / first_relevant_rank."""
    for rank, item in enumerate(retrieved, 1):
        if item in ground_truth:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not ground_truth or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += rel / math.log2(i + 2)  # index 0 -> log2(2) = 1.0

    # Ideal DCG: all possible relevant items ranked first
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ground_truth))))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def context_relevance(query: str, context: str) -> float:
    """
    Calculates context relevance score based on lexical coverage
    and question token overlap in the retrieved context.
    """
    if not query.strip() or not context.strip():
        return 0.0

    def extract_keywords(text: str) -> Set[str]:
        words = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "what", "which", "how",
            "why", "in", "on", "at", "to", "for", "of", "with", "by", "的", "了", "是", "在"
        }
        return {w for w in words if w not in stopwords and len(w) > 1}

    q_keywords = extract_keywords(query)
    if not q_keywords:
        return 1.0

    c_keywords = extract_keywords(context)
    overlap = q_keywords.intersection(c_keywords)
    return len(overlap) / float(len(q_keywords))


def faithfulness(answer: str, context: str) -> float:
    """
    Approximates hallucination score / faithfulness by verifying
    what proportion of noun/entity phrases in the generated answer
    are grounded in the provided context.
    """
    if not answer.strip():
        return 1.0
    if not context.strip():
        return 0.0

    # Extract substantive tokens (length >= 3)
    answer_tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]{2,}", answer.lower())
    if not answer_tokens:
        return 1.0

    context_lower = context.lower()
    grounded_count = sum(1 for token in answer_tokens if token in context_lower)
    return grounded_count / float(len(answer_tokens))
