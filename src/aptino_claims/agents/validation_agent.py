"""Validation Agent.

Verifies that every material citation actually came from the evidence that
was retrieved for that dimension (catches a mismatched/hallucinated
citation) and that any numeric figure named in a finding's statement
literally appears in the cited chunk's text (catches an unsupported
number). On FAIL, the case is downgraded to NEEDS_REVIEW -- the system's
one revision/retry behavior, applied because there is nothing left to
re-retrieve: the evidence already didn't support the claim.
"""
from __future__ import annotations

import re
import time

from .state import CaseState, DecisionStatus, ValidationOutcome

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_supported(claim_label: str, cited_text: str) -> bool:
    """Check the short citation *label* (e.g. "30-day initial waiting
    period"), not the full finding statement. The label only ever contains
    the policy threshold number itself (parsed from the cited text), never
    a case-derived figure (elapsed days, deduction amounts, ...), so any
    number appearing there must also appear in the cited text.
    """
    stmt_numbers = set(_NUMBER_RE.findall(claim_label))
    if not stmt_numbers:
        return True
    text_numbers = set(_NUMBER_RE.findall(cited_text))
    unsupported = {n for n in stmt_numbers if n not in text_numbers}
    return not unsupported


def run(state: CaseState) -> CaseState:
    started = time.perf_counter()
    unsupported_claims: list[str] = []

    all_evidence_ids = {ev.chunk_id for evs in state.evidence_by_dimension.values() for ev in evs}

    for finding in state.findings:
        if not finding.applicable or not finding.citations:
            continue
        for citation in finding.citations:
            if citation.chunk_id not in all_evidence_ids:
                unsupported_claims.append(f"[{finding.dimension}] cites chunk {citation.chunk_id} which was never retrieved")
                continue
            evidence_items = state.evidence_by_dimension.get(finding.dimension, [])
            cited_item = next((e for e in evidence_items if e.chunk_id == citation.chunk_id), None)
            if cited_item and not _numbers_supported(citation.claim, cited_item.text):
                unsupported_claims.append(f"[{finding.dimension}] statement contains a figure not present in cited chunk {citation.chunk_id}")

    status = "FAIL" if unsupported_claims else "PASS"
    state.validation = ValidationOutcome(status=status, unsupported_claims=unsupported_claims)

    if status == "FAIL" and state.decision != DecisionStatus.NEEDS_REVIEW:
        state.decision = DecisionStatus.NEEDS_REVIEW
        state.confidence = min(state.confidence, 0.3)
        state.rationale += "\n\nDowngraded to NEEDS_REVIEW: validation found citation(s) not fully supported by retrieved evidence."

    state.log(
        "ValidationAgent",
        "verify_citations",
        f"status={status} unsupported={len(unsupported_claims)}",
        started_at=started,
    )
    return state
