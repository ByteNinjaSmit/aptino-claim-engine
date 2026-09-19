"""Decision Agent: combines specialist findings into the final decision.

The combination logic is deliberately simple and auditable rather than
LLM-driven: it is a small precedence rule over the *already-cited*
per-dimension findings produced by the Coverage & Exclusion Agent. An LLM
is only ever used, optionally, to turn the resulting findings into a more
fluent rationale paragraph -- never to decide the outcome itself.
"""
from __future__ import annotations

import json
import re
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


# Language that argues for changing the outcome. A legitimate rationale explains the decision; it never
# tells the reviewer to overrule it, so text like this on a non-approving decision came from injected input.
_OVERRIDE = re.compile(r"\b(overrid\w*|overrul\w*|pre-?approved|disregard\w*|ignore (?:the|all|any|previous)|paid in full|must be paid|should be approved)\b", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _norm_number(n: str) -> str:
    n = n.rstrip("0").rstrip(".") if "." in n else n
    return n.lstrip("0") or "0"


def _numbers(text: str) -> set[str]:
    return {_norm_number(n) for n in _NUMBER.findall(text.replace(",", ""))}


def _guard_llm_rationale(state: CaseState, llm_text: str | None) -> tuple[str | None, str]:
    """Accept LLM prose only if it is proportionate, names the decision, and introduces no new number.

    Every figure in the paragraph must already appear in the cited findings, the
    computed limits/amounts, or the case itself; otherwise the model has asserted
    something the evidence does not support and the cited template is kept.
    """
    if not llm_text:
        return None, "no LLM text"
    if len(llm_text) >= 4 * len(state.rationale) + 200:
        return None, "LLM text disproportionately long"
    if state.decision.value.lower().replace("_", " ") not in llm_text.lower().replace("_", " "):
        return None, "LLM text does not name the decision"
    if not state.decision.value.startswith("ADMISSIBLE") and _OVERRIDE.search(llm_text):
        return None, "LLM text argues against the decision"
    allowed = _numbers(" ".join([
        state.rationale, json.dumps(state.raw_case, default=str),
        " ".join(l.description for l in state.applicable_limits), json.dumps(state._amounts(state.decision.value)),
    ]))
    invented = sorted(_numbers(llm_text) - allowed)
    if invented:
        return None, f"LLM text contains figures not in the evidence: {', '.join(invented[:5])}"
    return llm_text, "accepted"


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
        accepted, guard = _guard_llm_rationale(state, llm_text)
        if accepted:
            state.rationale, state.rationale_source = accepted, "llm"
        elif llm_text:
            state.rationale_source = f"template (LLM text rejected: {guard})"

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
