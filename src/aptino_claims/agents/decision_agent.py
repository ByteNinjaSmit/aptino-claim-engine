"""Decision Agent: combines specialist findings into the final decision.

The combination logic is deliberately simple and auditable rather than
LLM-driven: it is a small precedence rule over the *already-cited*
per-dimension findings produced by the Coverage & Exclusion Agent. An LLM
is only ever used, optionally, to turn the resulting findings into a more
fluent rationale paragraph -- never to decide the outcome itself.
"""
from __future__ import annotations

import time

from ..llm.base import LLMClient
from .state import CaseState, DecisionStatus

_WHOLE_CLAIM_EXCLUSION_DIMS = {
    "initial_waiting_period",
    "named_disease_first_year_waiting_period",
    "pre_existing_disease_waiting_period",
    "domiciliary_treatment_conditions",
    "cosmetic_exclusion",
}


def _template_rationale(state: CaseState) -> str:
    lines = [f"- [{f.dimension}] {f.statement}" for f in state.findings if f.applicable]
    return f"Decision {state.decision.value} for case {state.case_id}, based on:\n" + "\n".join(lines)


def run(state: CaseState, llm: LLMClient) -> CaseState:
    started = time.perf_counter()
    findings = state.findings

    insufficient = [f for f in findings if f.status == "INSUFFICIENT_EVIDENCE"]
    whole_claim_exclusions = [f for f in findings if f.status == "SUPPORTS_EXCLUSION" and f.dimension in _WHOLE_CLAIM_EXCLUSION_DIMS]
    time_window_limits = [f for f in findings if f.status == "SUPPORTS_LIMIT" and f.dimension == "pre_post_hospitalization_window"]
    sub_limits = [l for l in state.applicable_limits if l.dimension == "category_sub_limits" and (l.deduction_inr or 0) > 0]
    time_window_deductions = [l for l in state.applicable_limits if l.dimension == "pre_post_hospitalization_window" and (l.deduction_inr or 0) > 0]

    missing_dates = any(m.field == "policy_start_date/claim_date" for m in state.missing_fields)
    unresolved = bool(state.facts.get("_explicit_unknowns")) or bool(insufficient)

    # Precedence: without dates nothing can be decided; otherwise one verified
    # ground for rejection is enough (unresolved side-questions cannot rescue
    # an excluded claim); only then does unresolved evidence force a review.
    if missing_dates:
        state.decision = DecisionStatus.NEEDS_REVIEW
        state.confidence = 0.25
    elif whole_claim_exclusions:
        state.decision = DecisionStatus.NOT_ADMISSIBLE
        state.confidence = 0.85 if not unresolved else 0.75
    elif unresolved:
        state.decision = DecisionStatus.NEEDS_REVIEW
        state.confidence = max(0.25, 0.4 - 0.03 * len(insufficient))
    elif time_window_limits and time_window_deductions:
        state.decision = DecisionStatus.PARTIALLY_ADMISSIBLE
        state.confidence = 0.75
    elif sub_limits:
        state.decision = DecisionStatus.ADMISSIBLE_WITH_LIMITS
        state.confidence = 0.8
    else:
        state.decision = DecisionStatus.ADMISSIBLE
        state.confidence = 0.85

    state.rationale = _template_rationale(state)

    if llm is not None:
        prompt = (
            f"Case {state.case_id}. Decision: {state.decision.value}.\n"
            f"Findings (each already evidence-cited -- do not add new facts):\n{state.rationale}\n\n"
            "Rewrite as one concise reviewer-facing paragraph that states the decision "
            f"({state.decision.value.replace('_', ' ').lower()}) explicitly."
        )
        llm_text = llm.generate_rationale(prompt)
        # Accept LLM prose only if it stays proportionate and still names the
        # decision; otherwise keep the deterministic, fully-cited template.
        normalized = llm_text.lower().replace("_", " ") if llm_text else ""
        if llm_text and len(llm_text) < 4 * len(state.rationale) + 200 and state.decision.value.lower().replace("_", " ") in normalized:
            state.rationale = llm_text

    state.log(
        "DecisionAgent",
        "combine_findings",
        f"decision={state.decision.value} confidence={state.confidence} "
        f"(insufficient={len(insufficient)}, exclusions={len(whole_claim_exclusions)}, "
        f"time_window_limits={len(time_window_deductions)}, sub_limits={len(sub_limits)})",
        started_at=started,
        reads=["findings", "applicable_limits", "missing_fields"],
        writes=["decision", "confidence", "rationale"],
    )
    state.hand_off("DecisionAgent", "ValidationAgent", f"decision {state.decision.value} to verify against its citations")
    return state
