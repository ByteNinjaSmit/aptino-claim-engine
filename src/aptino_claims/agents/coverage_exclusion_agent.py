"""Coverage & Exclusion Agent: interprets retrieved evidence against case facts."""
from __future__ import annotations

import time

from ..config import settings
from . import llm_interpretation
from .dimensions import DIMENSIONS
from .state import CaseState

_LAST = "unmodelled_policy_risk"


def _llm_detail(state: CaseState) -> str:
    r = state.llm_interpretation
    if not r.get("enabled"):
        return ""
    if r.get("error"):
        return f" LLM interpretation: {r['error']}."
    return f" LLM interpretation: {len(r['accepted'])} verified observation(s) accepted, {len(r['rejected'])} rejected."


def run(state: CaseState, llm=None, interpret: bool | None = None, retriever=None) -> CaseState:
    """Rule-based assessment of every dimension, then (optionally) an LLM interpretation pass.

    `interpret=None` follows the LLM_INTERPRETATION setting (default off).
    """
    started = time.perf_counter()
    interpret = settings.llm_interpretation if interpret is None else interpret
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

    if interpret and llm is not None:
        llm_interpretation.run(state, llm, retriever)

    n_exclusion = sum(1 for f in state.findings if f.status == "SUPPORTS_EXCLUSION")
    n_insufficient = sum(1 for f in state.findings if f.status == "INSUFFICIENT_EVIDENCE")
    n_limit = sum(1 for f in state.findings if f.status == "SUPPORTS_LIMIT")
    state.log(
        "CoverageExclusionAgent",
        "assess_dimensions",
        f"{len(state.findings)} findings: {n_exclusion} exclusion-supporting, {n_limit} limit-supporting, "
        f"{n_insufficient} insufficient-evidence; {len(state.applicable_limits)} limit(s) computed."
        + _llm_detail(state),
        started_at=started,
        reads=["facts", "evidence_by_dimension"] + (["llm"] if state.llm_interpretation.get("enabled") else []),
        writes=["findings", "applicable_limits", "missing_fields"],
    )
    state.hand_off("CoverageExclusionAgent", "DecisionAgent",
                   f"{len(state.findings)} cited findings, {len(state.applicable_limits)} limit(s), {len(state.missing_fields)} gap(s)")
    return state
