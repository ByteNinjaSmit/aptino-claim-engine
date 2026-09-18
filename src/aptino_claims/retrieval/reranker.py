"""Cross-encoder reranking stage (BGE reranker via fastembed, local/free)."""
from __future__ import annotations

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ..config import settings

_reranker_cache: dict[str, TextCrossEncoder] = {}


def _get_reranker(model_name: str) -> TextCrossEncoder:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = TextCrossEncoder(model_name=model_name)
    return _reranker_cache[model_name]


def rerank(query: str, candidates: list[tuple[str, str]], model_name: str | None = None) -> list[tuple[str, float]]:
    """candidates: list of (chunk_id, text). Returns (chunk_id, rerank_score) sorted desc."""
    if not candidates:
        return []
    reranker = _get_reranker(model_name or settings.rerank_model)
    ids = [c[0] for c in candidates]
    texts = [c[1] for c in candidates]
    scores = list(reranker.rerank(query, texts))
    paired = list(zip(ids, (float(s) for s in scores)))
    return sorted(paired, key=lambda x: x[1], reverse=True)
