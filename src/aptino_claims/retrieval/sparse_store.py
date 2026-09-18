"""BM25 lexical retrieval over the policy chunks."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseStore:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        self._corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]
