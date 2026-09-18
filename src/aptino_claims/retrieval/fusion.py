"""Reciprocal Rank Fusion of dense + sparse result lists."""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several (chunk_id, score) rank lists via RRF.

    RRF is used (rather than a raw weighted score blend) because dense
    cosine similarities and BM25 scores live on incomparable scales; RRF
    only needs each list's *rank order*, which makes fusion robust without
    hand-tuned score normalization.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:top_k] if top_k else ordered
