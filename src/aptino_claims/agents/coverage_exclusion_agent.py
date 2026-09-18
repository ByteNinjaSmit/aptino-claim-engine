"""Coverage & Exclusion Agent: interprets retrieved evidence against case facts."""
from __future__ import annotations

import time

from .dimensions import DIMENSIONS
from .state import CaseState


def run(state: CaseState) -> CaseState:
    started = time.perf_counter()
    by_key = {d.key: d for d in DIMENSIONS}

    for dim_key in state.dimensions:
        dim = by_key[dim_key]
        evidence = state.evidence_by_dimension.get(dim_key, [])
        finding, limits, missing = dim.evaluate(state.facts, evidence)
        state.findings.append(finding)
        state.applicable_limits.extend(limits)
        state.missing_fields.extend(missing)

    n_exclusion = sum(1 for f in state.findings if f.status == "SUPPORTS_EXCLUSION")
    n_insufficient = sum(1 for f in state.findings if f.status == "INSUFFICIENT_EVIDENCE")
    n_limit = sum(1 for f in state.findings if f.status == "SUPPORTS_LIMIT")
    state.log(
        "CoverageExclusionAgent",
        "assess_dimensions",
        f"{len(state.findings)} findings: {n_exclusion} exclusion-supporting, {n_limit} limit-supporting, {n_insufficient} insufficient-evidence.",
        started_at=started,
    )
    return state
