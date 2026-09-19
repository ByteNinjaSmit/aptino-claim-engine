"""Case Analysis Agent.

Extracts normalized facts from the raw case JSON, flags missing/unknown
fields, and builds the investigation checklist (which decision dimensions
apply to this case). Deterministic: the input is already structured JSON,
so "extraction" here means normalization and dimension-detection rather
than free-text parsing.
"""
from __future__ import annotations

import time
from datetime import date

from .dimensions import DIMENSIONS
from .state import CaseState, MissingEvidence

_KNOWN_TOP_LEVEL_FIELDS = {
    "case_id", "policy_id", "policy_start_date", "claim_date", "sum_insured_inr",
    "continuous_coverage_months", "prior_insurer_continuous_years", "patient", "hospital",
    "treatment", "expenses_inr", "documents", "task", "evidence_context", "expense_timing", "prior_policy",
}


_KNOWN_NESTED_FIELDS = {
    "patient": {"age"},
    "hospital": {"name", "network_provider"},
    "treatment": {"type", "admission_hours", "diagnosis", "procedure", "pre_existing", "experimental",
                  "package_charges_agreed", "hospital_room_unavailable", "patient_cannot_be_moved"},
}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def build_facts(raw_case: dict) -> tuple[dict, list[MissingEvidence]]:
    missing: list[MissingEvidence] = []
    treatment = raw_case.get("treatment", {}) or {}
    hospital = raw_case.get("hospital", {}) or {}
    patient = raw_case.get("patient", {}) or {}
    expenses = raw_case.get("expenses_inr", {}) or {}

    policy_start = _parse_date(raw_case.get("policy_start_date"))
    claim_date = _parse_date(raw_case.get("claim_date"))
    if policy_start is None or claim_date is None:
        missing.append(MissingEvidence(field="policy_start_date/claim_date", reason="Missing or unparseable policy start / claim date."))
        days_since_start = 10**6  # treat as "long-standing policy" only if we truly cannot tell; downstream also checks continuity
    else:
        days_since_start = (claim_date - policy_start).days

    if "sum_insured_inr" not in raw_case:
        missing.append(MissingEvidence(field="sum_insured_inr", reason="Sum insured not supplied; sub-limit amounts cannot be computed."))

    facts = {
        "age": patient.get("age"),
        "sum_insured": raw_case.get("sum_insured_inr", 0),
        "continuous_coverage_months": raw_case.get("continuous_coverage_months", 0),
        "prior_insurer_continuous_years": raw_case.get("prior_insurer_continuous_years", 0),
        "prior_policy": raw_case.get("prior_policy"),
        "days_since_policy_start": days_since_start,
        "treatment_type": treatment.get("type", "inpatient"),
        "admission_hours": treatment.get("admission_hours", 24),
        "diagnosis": treatment.get("diagnosis", ""),
        "procedure": treatment.get("procedure", ""),
        "pre_existing": treatment.get("pre_existing", False),
        "experimental": treatment.get("experimental", False),
        "package_charges_agreed": bool(treatment.get("package_charges_agreed", False)),
        "hospital_room_unavailable": treatment.get("hospital_room_unavailable"),
        "patient_cannot_be_moved": treatment.get("patient_cannot_be_moved"),
        "hospital_network_provider": hospital.get("network_provider", False),
        "hospital_name": hospital.get("name", "the treating facility"),
        "expenses": expenses,
        "documents": raw_case.get("documents", []),
        "evidence_context": raw_case.get("evidence_context") or {},
        "expense_timing": raw_case.get("expense_timing"),
    }

    # Reliability scenario: input attributes irrelevant to the policy decision
    # (kept, but never fed into any dimension's reasoning).
    unrecognized = set(raw_case.keys()) - _KNOWN_TOP_LEVEL_FIELDS
    for group, known in _KNOWN_NESTED_FIELDS.items():
        unrecognized |= {f"{group}.{k}" for k in (raw_case.get(group) or {}) if k not in known}
    facts["_unrecognized_fields"] = sorted(unrecognized)

    # Reliability scenario: the case author explicitly flags a fact as
    # unresolved (`null`) rather than omitting the key entirely -- e.g.
    # `{"hospital_registered": null}`. That is a deliberate "investigated
    # but unknown" signal, distinct from a field simply being absent, and
    # must not be silently overridden by an unrelated proxy signal (such as
    # network-provider status) elsewhere in the case. It always forces
    # NEEDS_REVIEW regardless of which dimension it relates to.
    explicit_unknowns = [k for k, v in (raw_case.get("evidence_context") or {}).items() if v is None]
    for key in explicit_unknowns:
        missing.append(MissingEvidence(
            field=f"evidence_context.{key}",
            reason=f"Case explicitly flags '{key}' as unresolved (null); a confident decision cannot assume a value for this fact.",
        ))
    facts["_explicit_unknowns"] = explicit_unknowns

    return facts, missing


def run(state: CaseState) -> CaseState:
    started = time.perf_counter()
    facts, missing = build_facts(state.raw_case)
    state.facts = facts
    state.missing_fields.extend(missing)

    applicable = [d for d in DIMENSIONS if d.applies(facts)]
    state.dimensions = [d.key for d in applicable]
    state.investigation_checklist = [d.key.replace("_", " ") for d in applicable]

    detail = f"{len(applicable)} decision dimension(s) apply: {', '.join(state.dimensions)}"
    if facts.get("_unrecognized_fields"):
        detail += f" | ignored non-policy-relevant fields: {facts['_unrecognized_fields']}"
    state.log("CaseAnalysisAgent", "extract_facts_and_plan", detail, started_at=started,
              reads=["raw_case"], writes=["facts", "dimensions", "investigation_checklist", "missing_fields"])
    state.hand_off("CaseAnalysisAgent", "PolicyEvidenceAgent", f"{len(state.dimensions)} policy questions to research")
    return state
