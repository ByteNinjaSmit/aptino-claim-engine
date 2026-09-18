"""Policy Evidence Agent: hybrid retrieval + reranking per decision dimension."""
from __future__ import annotations

import time

from ..retrieval.retriever import HybridRetriever
from .dimensions import DIMENSIONS, to_evidence_items
from .state import CaseState


def run(state: CaseState, retriever: HybridRetriever) -> CaseState:
    started = time.perf_counter()
    total_hits = 0
    by_key = {d.key: d for d in DIMENSIONS}

    for dim_key in state.dimensions:
        dim = by_key[dim_key]
        query = dim.query(state.facts)
        results = retriever.search(query)
        state.evidence_by_dimension[dim_key] = to_evidence_items(results)
        total_hits += len(results)

    state.log(
        "PolicyEvidenceAgent",
        "hybrid_retrieve_and_rerank",
        f"Ran {len(state.dimensions)} queries, {total_hits} evidence chunks returned after fusion+rerank.",
        retrieval_count=total_hits,
        started_at=started,
    )
    return state
