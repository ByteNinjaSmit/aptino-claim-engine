"""Coverage & Exclusion Agent: interprets retrieved evidence against case facts."""
from __future__ import annotations

import time

from .dimensions import DIMENSIONS
from .state import CaseState

_LAST = "unmodelled_policy_risk"


def run(state: CaseState) -> CaseState:
    started = time.perf_counter()
    by_key = {d.key: d for d in DIMENSIONS}

    # The unknown-dimension check runs last: it only looks at policy clauses
    # that no modelled dimension has already used.
    ordered = [k for k in state.dimensions if k != _LAST] + [k for k in state.dimensions if k == _LAST]

    for dim_key in ordered:
        if dim_key == _LAST:
            state.facts["_cited_chunk_ids"] = sorted({c.chunk_id for f in state.findings for c in f.citations})
        evidence = state.evidence_by_dimension.get(dim_key, [])
        finding, limits, missing = by_key[dim_key].evaluate(state.facts, evidence)
        state.findings.append(finding)
        state.applicable_limits.extend(limits)
        state.missing_fields.extend(missing)

    n_exclusion = sum(1 for f in state.findings if f.status == "SUPPORTS_EXCLUSION")
    n_insufficient = sum(1 for f in state.findings if f.status == "INSUFFICIENT_EVIDENCE")
    n_limit = sum(1 for f in state.findings if f.status == "SUPPORTS_LIMIT")
    state.log(
        "CoverageExclusionAgent",
        "assess_dimensions",
        f"{len(state.findings)} findings: {n_exclusion} exclusion-supporting, {n_limit} limit-supporting, "
        f"{n_insufficient} insufficient-evidence; {len(state.applicable_limits)} limit(s) computed.",
        started_at=started,
        reads=["facts", "evidence_by_dimension"],
        writes=["findings", "applicable_limits", "missing_fields"],
    )
    state.hand_off("CoverageExclusionAgent", "DecisionAgent",
                   f"{len(state.findings)} cited findings, {len(state.applicable_limits)} limit(s), {len(state.missing_fields)} gap(s)")
    return state
