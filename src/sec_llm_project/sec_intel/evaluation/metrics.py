"""Information-retrieval and grounding metrics (plan item 9).

All functions are pure and dependency-free so they can be unit-tested directly.
Relevance is matched against a set of *relevant ids* (chunk ids or item numbers,
depending on how the gold set is keyed).
"""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    # Set intersection so duplicate ids in the top-k cannot inflate recall > 1.
    found = set(retrieved[:k]) & rel
    return len(found) / len(rel)


def precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if k <= 0:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in rel)
    return hits / min(k, len(retrieved)) if retrieved else 0.0


def reciprocal_rank(retrieved: list[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, cid in enumerate(retrieved):
        if cid in rel:
            return 1.0 / (i + 1)
    return 0.0


def mrr(rankings: list[list[str]], relevants: list[Iterable[str]]) -> float:
    if not rankings:
        return 0.0
    total = sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants, strict=False))
    return total / len(rankings)


def average_precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    score = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in rel:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(rel), k)


def citation_correctness(cited_ids: Iterable[str], relevant: Iterable[str]) -> float:
    """Fraction of an answer's citations that point at relevant evidence."""
    cited = list(cited_ids)
    if not cited:
        return 0.0
    rel = set(relevant)
    return sum(1 for cid in cited if cid in rel) / len(cited)


def token_recall(chunk_text: str, passage: str) -> float:
    """Fraction of passage tokens present in chunk_text (passage-level recall).

    Used when gold relevance is expressed as a verbatim evidence passage rather
    than a SEC item number (e.g. FinDER / FinanceBench datasets).
    """
    p_toks = set(passage.lower().split())
    c_toks = set(chunk_text.lower().split())
    if not p_toks:
        return 0.0
    return len(p_toks & c_toks) / len(p_toks)


def passage_hit(chunk_text: str, passages: list[str], threshold: float = 0.5) -> bool:
    """True if chunk_text covers at least one gold passage at the given threshold."""
    return any(token_recall(chunk_text, p) >= threshold for p in passages)
