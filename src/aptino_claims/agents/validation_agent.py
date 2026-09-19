"""Validation Agent.

Turns every material statement into an auditable chain

    decision claim -> citation -> retrieved policy chunk -> evidence check -> verdict

and records SUPPORTED / UNSUPPORTED / CONTRADICTED per claim (see
`verification.py` for how each assertion type is checked). The overall
status is FAIL if any claim is UNSUPPORTED or CONTRADICTED.

On FAIL the orchestrator gets one chance to repair the problem with a
widened re-retrieval (a clause may simply have ranked outside the first
top-k). If it still fails, the decision is downgraded to NEEDS_REVIEW: the
system abstains rather than presenting a claim its own evidence check
could not verify.
"""
from __future__ import annotations

import time

from ..retrieval.retriever import HybridRetriever
from .state import CaseState, DecisionStatus, ValidationOutcome
from .verification import verify_state


def run(state: CaseState, retriever: HybridRetriever, final_attempt: bool = True) -> CaseState:
    started = time.perf_counter()
    corpus = {cid: meta["text"] for cid, meta in retriever.chunk_meta.items()}
    rows = verify_state(state, corpus)

    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
    for r in rows:
        counts[r.verdict] += 1
    problems = [r for r in rows if r.verdict != "SUPPORTED"]
    unsupported = [f"[{r.dimension}] {r.verdict}: {r.claim} ({r.chunk_id or 'no chunk'}) - {r.reason}" for r in problems]
    status = "FAIL" if problems else "PASS"

    state.validation = ValidationOutcome(
        status=status, unsupported_claims=unsupported, verifications=rows, counts=counts, attempts=state.attempt,
        notes=f"{counts['SUPPORTED']}/{len(rows)} claims verified against retrieved policy text.",
    )

    if status == "FAIL" and final_attempt and state.decision != DecisionStatus.NEEDS_REVIEW:
        state.decision = DecisionStatus.NEEDS_REVIEW
        state.confidence = min(state.confidence, 0.3)
        state.rationale += (
            "\n\nDowngraded to NEEDS_REVIEW: evidence verification could not confirm "
            f"{len(problems)} claim(s) even after a widened re-retrieval."
        )

    state.log(
        "ValidationAgent",
        "verify_claims_against_evidence",
        f"status={status}; {counts['SUPPORTED']} supported, {counts['UNSUPPORTED']} unsupported, {counts['CONTRADICTED']} contradicted "
        f"of {len(rows)} claims",
        started_at=started,
        reads=["findings", "applicable_limits", "decision", "evidence_by_dimension"],
        writes=["validation"] + (["decision"] if status == "FAIL" and final_attempt else []),
    )
    return state
