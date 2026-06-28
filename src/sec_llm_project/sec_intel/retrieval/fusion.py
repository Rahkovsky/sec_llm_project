"""Rank fusion utilities (plan item 3)."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked id lists.

    RRF is robust because it ignores raw score scales and only uses rank, which
    makes BM25 and cosine scores comparable without normalization.
    """
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights, strict=False):
        for rank, cid in enumerate(ids):
            fused[cid] = fused.get(cid, 0.0) + weight * (1.0 / (k + rank + 1))
    return fused


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {cid: (s - lo) / (hi - lo) for cid, s in scores.items()}


def weighted_fusion(
    score_maps: list[dict[str, float]], weights: list[float]
) -> dict[str, float]:
    """Min-max normalize each score map, then combine with weights."""
    fused: dict[str, float] = {}
    for scores, weight in zip(score_maps, weights, strict=False):
        for cid, s in _min_max(scores).items():
            fused[cid] = fused.get(cid, 0.0) + weight * s
    return fused
