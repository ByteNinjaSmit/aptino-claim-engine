"""Brute-force cosine-similarity dense store.

The policy corpus is on the order of ~100 chunks, so an exact numpy scan is
faster and simpler than standing up FAISS/Qdrant, while remaining a drop-in
swap (`search`) if the corpus grows and an ANN index becomes worthwhile.
"""
from __future__ import annotations

import numpy as np


class DenseStore:
    def __init__(self, chunk_ids: list[str], vectors: np.ndarray):
        self.chunk_ids = chunk_ids
        self.vectors = vectors  # (N, D), L2-normalized

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores = self.vectors @ query_vector  # cosine similarity (both normalized)
        ranked = np.argsort(-scores)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]
