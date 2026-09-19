"""Policy Evidence Agent: hybrid retrieval + reranking per decision dimension."""
from __future__ import annotations

import time

from ..config import settings
from ..retrieval.retriever import HybridRetriever
from .dimensions import DIMENSIONS, to_evidence_items
from .state import CaseState


def run(state: CaseState, retriever: HybridRetriever) -> CaseState:
    """Retrieve evidence for every applicable dimension.

    On a validation-triggered retry (`state.widen_retrieval`), the candidate
    pool and the number of chunks handed to the reasoning agents are doubled,
    so a clause that ranked just outside the first top-k can still be found.
    """
    started = time.perf_counter()
    total_hits = 0
    by_key = {d.key: d for d in DIMENSIONS}
    factor = 2 if state.widen_retrieval else 1

    for dim_key in state.dimensions:
        query = by_key[dim_key].query(state.facts)
        results = retriever.search(
            query,
            top_k_dense=settings.top_k_dense * factor,
            top_k_sparse=settings.top_k_sparse * factor,
            top_k_fused=settings.top_k_fused * factor,
            top_k_final=settings.top_k_final * factor,
        )
        state.evidence_by_dimension[dim_key] = to_evidence_items(results)
        total_hits += len(results)

    mode = f"widened x{factor} (retry)" if factor > 1 else "standard"
    state.log(
        "PolicyEvidenceAgent",
        "hybrid_retrieve_and_rerank",
        f"{mode}: ran {len(state.dimensions)} queries, {total_hits} evidence chunks returned after fusion+rerank.",
        retrieval_count=total_hits,
        started_at=started,
        reads=["facts", "dimensions"],
        writes=["evidence_by_dimension"],
    )
    state.hand_off("PolicyEvidenceAgent", "CoverageExclusionAgent", f"{total_hits} ranked evidence chunks with page/section/chunk metadata")
    return state
