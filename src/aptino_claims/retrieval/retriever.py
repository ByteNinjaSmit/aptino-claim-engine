"""HybridRetriever: dense + BM25 -> RRF fusion -> cross-encoder rerank.

This is the single entry point the Policy Evidence Agent calls. It returns
fully-annotated results (per-stage rank/score) so retrieval quality and
citation correctness can be measured, and so the API/trace can show
reviewer-facing evidence provenance without exposing any hidden reasoning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import settings
from .dense_store import DenseStore
from .embeddings import embed_query, embed_texts
from .fusion import reciprocal_rank_fusion
from .reranker import rerank as rerank_fn
from .sparse_store import SparseStore


@dataclass
class RetrievalResult:
    chunk_id: str
    section: str
    label: str
    page_start: int
    page_end: int
    text: str
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    fused_score: float | None
    rerank_score: float | None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section": self.section,
            "label": self.label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "source": "policy.pdf",
            "scores": {
                "dense_rank": self.dense_rank,
                "dense_score": self.dense_score,
                "sparse_rank": self.sparse_rank,
                "sparse_score": self.sparse_score,
                "fused_score": self.fused_score,
                "rerank_score": self.rerank_score,
            },
        }


class HybridRetriever:
    def __init__(self, chunk_meta: list[dict], dense_vectors: np.ndarray):
        self.chunk_meta = {c["chunk_id"]: c for c in chunk_meta}
        self._order = [c["chunk_id"] for c in chunk_meta]
        self._texts = [c["text"] for c in chunk_meta]
        self.dense = DenseStore(self._order, dense_vectors)
        self.sparse = SparseStore(self._order, self._texts)
        self._dense_vectors = dense_vectors

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict]) -> "HybridRetriever":
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        return cls(chunks, vectors)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "dense_vectors.npy", self._dense_vectors)
        with (index_dir / "chunk_meta.json").open("w", encoding="utf-8") as f:
            json.dump([self.chunk_meta[cid] for cid in self._order], f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path) -> "HybridRetriever":
        vectors = np.load(index_dir / "dense_vectors.npy")
        with (index_dir / "chunk_meta.json").open(encoding="utf-8") as f:
            chunk_meta = json.load(f)
        return cls(chunk_meta, vectors)

    @classmethod
    def load_or_build(cls, index_dir: Path, chunks_path: Path) -> "HybridRetriever":
        if (index_dir / "dense_vectors.npy").exists() and (index_dir / "chunk_meta.json").exists():
            return cls.load(index_dir)
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        retriever = cls.build(chunks)
        retriever.save(index_dir)
        return retriever

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_fused: int | None = None,
        top_k_final: int | None = None,
    ) -> list[RetrievalResult]:
        top_k_dense = top_k_dense or settings.top_k_dense
        top_k_sparse = top_k_sparse or settings.top_k_sparse
        top_k_fused = top_k_fused or settings.top_k_fused
        top_k_final = top_k_final or settings.top_k_final

        q_vec = embed_query(query)
        dense_hits = self.dense.search(q_vec, top_k_dense)
        sparse_hits = self.sparse.search(query, top_k_sparse)

        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, start=1)}
        dense_score = dict(dense_hits)
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse_hits, start=1)}
        sparse_score = dict(sparse_hits)

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k_fused)
        fused_score = dict(fused)
        if not fused:
            return []

        candidates = [(cid, self.chunk_meta[cid]["text"]) for cid, _ in fused]
        reranked = rerank_fn(query, candidates)[:top_k_final]

        results: list[RetrievalResult] = []
        for chunk_id, rerank_score in reranked:
            meta = self.chunk_meta[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    section=meta["section"],
                    label=meta["label"],
                    page_start=meta["page_start"],
                    page_end=meta["page_end"],
                    text=meta["text"],
                    dense_rank=dense_rank.get(chunk_id),
                    dense_score=dense_score.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    sparse_score=sparse_score.get(chunk_id),
                    fused_score=fused_score.get(chunk_id),
                    rerank_score=rerank_score,
                )
            )
        return results
