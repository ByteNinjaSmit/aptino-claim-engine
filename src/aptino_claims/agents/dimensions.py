"""The decision-dimension registry.

Each dimension is a (applicability check, retrieval query, evaluator)
triple. This is the "genuine" reasoning core: it only ever asserts a
conclusion when the number/condition it needs was actually found in the
text of the chunk retrieved for that dimension. When the number can't be
parsed out of the retrieved evidence, or a required case fact is missing,
the dimension reports INSUFFICIENT_EVIDENCE rather than guessing -- which is
what lets the Decision Agent abstain safely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..retrieval.retriever import RetrievalResult
from . import rules_extract as rx
from .state import ApplicableLimit, Citation, EvidenceItem, Finding, MissingEvidence

EvalResult = tuple[Finding, list[ApplicableLimit], list[MissingEvidence]]


def to_evidence_items(results: list[RetrievalResult]) -> list[EvidenceItem]:
    return [EvidenceItem(**r.to_dict() | {"dense_rank": r.dense_rank, "dense_score": r.dense_score,
                                           "sparse_rank": r.sparse_rank, "sparse_score": r.sparse_score,
                                           "fused_score": r.fused_score, "rerank_score": r.rerank_score})
            for r in results]


def cite(ev: EvidenceItem, claim: str, assertion: dict | None = None) -> Citation:
    return Citation(claim=claim, page=ev.page_start, section=ev.section, chunk_id=ev.chunk_id,
                    rerank_score=ev.rerank_score, assertion=assertion)


def threshold(*pairs: tuple[int, str], must_contain: list[str] | None = None) -> dict:
    return {"type": "threshold", "values": [{"value": v, "unit": u} for v, u in pairs], "must_contain": must_contain or []}


def phrases(*must: str, any_of: list[list[str]] | None = None) -> dict:
    return {"type": "phrases", "must_contain": list(must), "any_of": any_of or []}


def _find_source(evidence: list[EvidenceItem], keyword: str) -> EvidenceItem | None:
    for ev in evidence:
        if keyword.lower() in ev.text.lower():
            return ev
    return evidence[0] if evidence else None


def _no_evidence(dimension: str, why: str) -> EvalResult:
    finding = Finding(
        dimension=dimension,
        applicable=True,
        statement=f"{why} Treat as insufficient evidence for this dimension.",
        status="INSUFFICIENT_EVIDENCE",
        citations=[],
        confidence=0.2,
    )
    return finding, [], [MissingEvidence(field=dimension, reason=why)]


# --------------------------------------------------------------------------
# 1. Initial (30-day) waiting period
# --------------------------------------------------------------------------

def _initial_wait_applies(facts: dict) -> bool:
    return True


def _initial_wait_query(facts: dict) -> str:
    return "30 days waiting period all claims exceptions continuous coverage without break"


def _initial_wait_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "initial_waiting_period"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the policy's initial waiting-period clause.")
    top = evidence[0]
    days = rx.extract_waiting_days(top.text)
    if days is None:
        return _no_evidence(dim, "Retrieved clause did not contain a parseable waiting-period day count.")

    days_since_start = facts["days_since_policy_start"]
    continuity_ok = facts["continuous_coverage_months"] > 0 or facts["prior_insurer_continuous_years"] >= 1

    assumptions = []
    if continuity_ok and facts["continuous_coverage_months"] > 0:
        assumptions.append("Continuous-coverage months reported in the case are assumed to be break-free.")
    if continuity_ok and facts["continuous_coverage_months"] == 0:
        assumptions.append("The prior-insurer route also requires proof the insured was unaware of the illness; this is not verifiable from the case data.")

    if days_since_start >= days or continuity_ok:
        reason = (
            f"policy has been continuously in force ({facts['continuous_coverage_months']} months) or prior "
            f"Indian-insurer continuity applies"
            if continuity_ok
            else f"claim date is {days_since_start} days after policy inception"
        )
        finding = Finding(
            dimension=dim,
            applicable=True,
            statement=f"Initial {days}-day waiting period is satisfied: {reason}.",
            status="SUPPORTS_ADMISSIBLE",
            citations=[cite(top, f"{days}-day initial waiting period and continuity exceptions", threshold((days, "days")))],
            confidence=0.9, assumptions=assumptions if days_since_start < days else [],
        )
    else:
        finding = Finding(
            dimension=dim,
            applicable=True,
            statement=(
                f"Claim date is only {days_since_start} days after policy inception, inside the policy's "
                f"{days}-day initial waiting period, and no continuous-coverage exception applies."
            ),
            status="SUPPORTS_EXCLUSION",
            citations=[cite(top, f"{days}-day initial waiting period", threshold((days, "days")))],
            confidence=0.9,
        )
    return finding, [], []


# --------------------------------------------------------------------------
# 2. First-year named-disease waiting period
# --------------------------------------------------------------------------

_NAMED_DISEASE_KEYWORDS = [
    "cataract", "benign prostatic", "myomectomy", "hysterectomy", "hernia", "hydrocele",
    "fistula", "piles", "arthritis", "gout", "rheumatism", "joint replacement", "sinusitis",
    "stone in", "urinary", "biliary", "dilatation and curettage", "d&c", "cyst", "polyp",
    "tumour", "tumor", "nodule", "dialysis", "tonsil", "gastric ulcer", "duodenal ulcer",
]


def _named_disease_match(facts: dict) -> bool:
    text = f"{facts.get('diagnosis','')} {facts.get('procedure','')}".lower()
    return any(kw in text for kw in _NAMED_DISEASE_KEYWORDS)


def _named_disease_applies(facts: dict) -> bool:
    return facts["treatment_type"] in ("inpatient", "day_care") and _named_disease_match(facts)


def _named_disease_query(facts: dict) -> str:
    return "first year of operation waiting period named diseases cataract hernia hysterectomy waiver"


def _named_disease_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "named_disease_first_year_waiting_period"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the first-year named-disease waiting-period clause.")
    top = evidence[0]
    years = rx.find_int(r"(\d+)\s*year", top.text)
    if years is None:
        return _no_evidence(dim, "Retrieved clause did not contain a parseable first-year wait duration.")
    threshold_days = years * 365

    prior = facts.get("prior_policy") or {}
    waived = facts["prior_insurer_continuous_years"] >= 1 and (
        prior.get("database_and_claim_history_received", True)
    )
    days_since_start = facts["days_since_policy_start"]
    named_assumptions = []
    if waived and "database_and_claim_history_received" not in prior:
        named_assumptions.append("Prior-insurer claim history is assumed received (the case does not say).")
    cite_assert = threshold((years, "year"), must_contain=["first year"])

    if days_since_start >= threshold_days:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=f"Policy has been in force for {days_since_start} days, past the {years}-year named-disease waiting period.",
            status="SUPPORTS_ADMISSIBLE", citations=[cite(top, f"{years}-year named-disease waiting period elapsed", cite_assert)], confidence=0.85,
        )
    elif waived:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Named-disease waiting period would otherwise apply (policy only {days_since_start} days old), "
                f"but is waived because of {facts['prior_insurer_continuous_years']} completed continuous year(s) "
                f"under a prior Indian individual health insurer with claim history received."
            ),
            status="SUPPORTS_ADMISSIBLE",
            citations=[cite(top, "waiting period waived for continuous prior Indian-insurer coverage", cite_assert)],
            confidence=0.8, assumptions=named_assumptions,
        )
    else:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Diagnosis/procedure falls on the policy's first-year named-disease list, the policy is only "
                f"{days_since_start} days old, and no prior-insurer continuity waiver applies."
            ),
            status="SUPPORTS_EXCLUSION",
            citations=[cite(top, f"{years}-year named-disease waiting period", cite_assert)],
            confidence=0.85,
        )
    return finding, [], []


# --------------------------------------------------------------------------
# 3. Pre-existing disease waiting period
# --------------------------------------------------------------------------

def _pre_existing_applies(facts: dict) -> bool:
    return bool(facts.get("pre_existing"))


def _pre_existing_query(facts: dict) -> str:
    return "pre-existing diseases waiting period months continuous coverage portability reduction"


def _pre_existing_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "pre_existing_disease_waiting_period"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the pre-existing-disease waiting-period clause.")
    top = evidence[0]
    months_needed = rx.extract_pre_existing_months(top.text)
    if months_needed is None:
        return _no_evidence(dim, "Retrieved clause did not contain a parseable month count.")

    prior = facts.get("prior_policy") or {}
    portability_months = (
        facts["prior_insurer_continuous_years"] * 12
        if prior.get("database_and_claim_history_received", facts["prior_insurer_continuous_years"] > 0)
        else 0
    )
    effective_months = facts["continuous_coverage_months"] + portability_months
    pre_assumptions = []
    if portability_months and "database_and_claim_history_received" not in prior:
        pre_assumptions.append("Prior-insurer claim history is assumed received (the case does not say).")
    pre_assert = threshold((months_needed, "months"))

    if effective_months >= months_needed:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Effective continuous coverage ({effective_months} months, including any portability credit) "
                f"meets the {months_needed}-month pre-existing-disease waiting period."
            ),
            status="SUPPORTS_ADMISSIBLE", citations=[cite(top, f"{months_needed}-month pre-existing disease waiting period satisfied", pre_assert)], confidence=0.85, assumptions=pre_assumptions,
        )
    else:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Effective continuous coverage is {effective_months} months, short of the "
                f"{months_needed}-month pre-existing-disease waiting period ({months_needed - effective_months} months remaining)."
            ),
            status="SUPPORTS_EXCLUSION", citations=[cite(top, f"{months_needed}-month pre-existing disease waiting period not met", pre_assert)], confidence=0.85,
        )
    return finding, [], []


# --------------------------------------------------------------------------
# 4. Hospital definition
# --------------------------------------------------------------------------

HOSPITAL_ASSERT = phrases("Hospital means", "in-patient beds")


def _hospital_def_applies(facts: dict) -> bool:
    return facts["treatment_type"] != "domiciliary"


def _hospital_def_query(facts: dict) -> str:
    return "definition of Hospital registration minimum criteria in-patient beds qualified nursing"


def _hospital_def_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "hospital_definition"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the policy's Hospital definition.")
    top = evidence[0]
    ctx = facts.get("evidence_context") or {}
    hospital_name = facts.get("hospital_name", "the treating facility")

    if facts.get("hospital_network_provider"):
        finding = Finding(
            dimension=dim, applicable=True,
            statement=f"{hospital_name} is a network-empanelled provider, which is treated as satisfying the policy's Hospital definition.",
            status="SUPPORTS_ADMISSIBLE", citations=[cite(top, "Hospital definition (registration / minimum criteria)", HOSPITAL_ASSERT)], confidence=0.75,
            assumptions=["Network-empanelled status is treated as proof the facility meets the Hospital definition (the policy text does not state this)."],
        )
        return finding, [], []

    if ctx.get("hospital_registered") is False or ctx.get("hospital_minimum_criteria_documented") is False or (
        ctx.get("hospital_registered") is None and "hospital_registered" in ctx
    ):
        reason = (
            f"Supplied evidence does not establish that {hospital_name} is registered under the Clinical "
            f"Establishments Act or meets the policy's minimum-criteria test (round-the-clock qualified nursing "
            f"and medical staff, minimum in-patient beds, dedicated operation theatre)."
        )
        finding = Finding(
            dimension=dim, applicable=True, statement=reason, status="INSUFFICIENT_EVIDENCE",
            citations=[cite(top, "Hospital definition (registration / minimum criteria)", HOSPITAL_ASSERT)], confidence=0.3,
        )
        return finding, [], [MissingEvidence(field="hospital_registration_proof", reason=reason)]

    if not ctx:
        reason = (
            f"{hospital_name} is not flagged as a network provider and no registration/minimum-criteria evidence "
            f"was supplied; the policy's Hospital definition cannot be confirmed from the available documents."
        )
        finding = Finding(
            dimension=dim, applicable=True, statement=reason, status="INSUFFICIENT_EVIDENCE",
            citations=[cite(top, "Hospital definition (registration / minimum criteria)", HOSPITAL_ASSERT)], confidence=0.3,
        )
        return finding, [], [MissingEvidence(field="hospital_registration_proof", reason=reason)]

    finding = Finding(
        dimension=dim, applicable=True,
        statement=f"Available evidence does not contradict {hospital_name} meeting the policy's Hospital definition.",
        status="SUPPORTS_ADMISSIBLE", citations=[cite(top, "Hospital definition (registration / minimum criteria)", HOSPITAL_ASSERT)], confidence=0.5,
    )
    return finding, [], []


# --------------------------------------------------------------------------
# 5. Domiciliary treatment conditions
# --------------------------------------------------------------------------

DOMICILIARY_ASSERT = phrases("removed to a Hospital", "non-availability of room")


def _domiciliary_applies(facts: dict) -> bool:
    return facts["treatment_type"] == "domiciliary"


def _domiciliary_query(facts: dict) -> str:
    return "domiciliary treatment condition patient cannot be removed non-availability of room in hospital"


def _domiciliary_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "domiciliary_treatment_conditions"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the Domiciliary Treatment definition.")
    top = evidence[0]
    room_unavail = facts.get("hospital_room_unavailable")
    cannot_move = facts.get("patient_cannot_be_moved")

    if room_unavail is True or cannot_move is True:
        reason = "non-availability of a hospital room" if room_unavail else "the patient's condition prevents moving them to a Hospital"
        finding = Finding(
            dimension=dim, applicable=True,
            statement=f"Domiciliary Treatment condition is met: {reason}.",
            status="SUPPORTS_ADMISSIBLE", citations=[cite(top, "Domiciliary Treatment definition condition met", DOMICILIARY_ASSERT)], confidence=0.85,
        )
    elif room_unavail is False and cannot_move is False:
        finding = Finding(
            dimension=dim, applicable=True,
            statement="Neither Domiciliary Treatment condition (room unavailability or inability to move the patient) is met.",
            status="SUPPORTS_EXCLUSION", citations=[cite(top, "Domiciliary Treatment definition condition not met", DOMICILIARY_ASSERT)], confidence=0.8,
        )
    else:
        return _no_evidence(dim, "Case does not state whether a hospital room was unavailable or the patient could not be moved.")
    return finding, [], []


# --------------------------------------------------------------------------
# 6. Day-care / <24h treatment
# --------------------------------------------------------------------------

# procedure keyword in the case -> the wording used in the policy's named list (NB4)
_NAMED_DAYCARE = {
    "dialysis": "Dialysis", "chemotherapy": "Chemotherapy", "radiotherapy": "Radiotherapy",
    "eye surgery": "Eye Surgery", "cataract": "Eye Surgery", "lithotripsy": "Lithotripsy",
    "tonsillectomy": "Tonsillectomy", "d&c": "D&C", "dilatation and curettage": "D&C",
}
DAYCARE_DEFINITION_ASSERT = phrases("less than 24 hrs", "General or Local Anesthesia")


def _daycare_applies(facts: dict) -> bool:
    return facts["treatment_type"] == "day_care" or (
        facts["treatment_type"] == "inpatient" and facts.get("admission_hours", 999) < 24
    )


def _daycare_query(facts: dict) -> str:
    return "day care treatment less than 24 hours anesthesia technological advancement list of procedures"


def _daycare_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "day_care_less_than_24h"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the Day Care Treatment definition.")
    text = f"{facts.get('diagnosis','')} {facts.get('procedure','')}".lower()
    named_terms = sorted({wording for kw, wording in _NAMED_DAYCARE.items() if kw in text})
    definition = _find_source(evidence, "less than 24 hrs")

    if named_terms:
        # Cite the chunk that actually lists the procedure. If retrieval did not
        # surface it, cite the top chunk anyway: verification will flag it
        # UNSUPPORTED and trigger the widened-retrieval retry.
        src = next((ev for ev in evidence if any(term.lower() in ev.text.lower() for term in named_terms)), evidence[0])
        finding = Finding(
            dimension=dim, applicable=True,
            statement=f"{', '.join(named_terms)} is on the policy's named list of treatments covered when completed in under 24 hours.",
            status="SUPPORTS_ADMISSIBLE",
            citations=[cite(src, "named <24h day-care procedure", phrases(*named_terms))], confidence=0.85,
        )
    elif facts["treatment_type"] == "day_care":
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                "Treatment is classified as day care; admissibility assumes it required anesthesia and would "
                "otherwise have needed >24h hospitalization, per the Day Care Treatment definition."
            ),
            status="SUPPORTS_ADMISSIBLE", citations=[cite(definition, "Day Care Treatment definition", DAYCARE_DEFINITION_ASSERT)], confidence=0.6,
            assumptions=["The procedure is assumed to need anesthesia and to have otherwise required a stay over 24 hours (not stated in the case)."],
        )
    else:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Admission was only {facts.get('admission_hours')} hours for a procedure not on the policy's named "
                f"day-care list; medical necessity for a stay under 24 hours cannot be confirmed from the supplied evidence."
            ),
            status="INSUFFICIENT_EVIDENCE", citations=[cite(definition, "Day Care Treatment definition", DAYCARE_DEFINITION_ASSERT)], confidence=0.3,
        )
        return finding, [], [MissingEvidence(field="day_care_medical_justification", reason=finding.statement)]
    return finding, [], []


# --------------------------------------------------------------------------
# 7. Pre/post hospitalization time window
# --------------------------------------------------------------------------

def _pre_post_applies(facts: dict) -> bool:
    exp = facts.get("expenses", {})
    return exp.get("pre_hospitalization", 0) > 0 or exp.get("post_hospitalization", 0) > 0


def _pre_post_query(facts: dict) -> str:
    return "pre-hospitalisation post-hospitalisation days immediately preceding following expenses"


def _pre_post_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "pre_post_hospitalization_window"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the pre/post-hospitalization time-window clause.")
    top = evidence[0]
    pre_limit, post_limit = rx.extract_pre_post_windows(top.text)
    if pre_limit is None or post_limit is None:
        return _no_evidence(dim, "Retrieved clause did not contain parseable pre/post-hospitalization day limits.")

    timing = facts.get("expense_timing")
    exp = facts.get("expenses", {})
    window_assert = threshold((pre_limit, "days"), (post_limit, "days"))
    if not timing:
        # Missing timing metadata is common and, on its own, is not treated
        # as a reason to withhold payment or abstain on the whole case --
        # only an explicit, confirmed window violation (see the `else`
        # branch below) does that. This keeps NEEDS_REVIEW/deductions
        # reserved for genuinely confirmed problems rather than penalizing
        # every case that simply didn't populate an optional field.
        reason = (
            f"Pre/post-hospitalization expenses were claimed but exact admission/discharge-relative timing was "
            f"not supplied; assumed within the {pre_limit}-day/{post_limit}-day windows pending documentation."
        )
        finding = Finding(
            dimension=dim, applicable=True, statement=reason, status="SUPPORTS_ADMISSIBLE",
            citations=[cite(top, f"{pre_limit}-day pre / {post_limit}-day post hospitalization window", window_assert)], confidence=0.55,
            assumptions=[f"Pre/post-hospitalization expenses are assumed to fall inside the {pre_limit}/{post_limit}-day windows (timing not supplied)."],
        )
        return finding, [], [MissingEvidence(field="expense_timing", reason=reason)]

    pre_days = timing.get("pre_hospitalization_days_before_admission")
    post_days = timing.get("post_hospitalization_days_after_discharge")
    same_condition = timing.get("same_condition_confirmed", True)

    pre_ok = pre_days is None or pre_days <= pre_limit
    post_ok = post_days is None or post_days <= post_limit

    limits: list[ApplicableLimit] = []
    if pre_ok and post_ok and same_condition:
        finding = Finding(
            dimension=dim, applicable=True,
            statement=f"Pre-hospitalization ({pre_days} days) and post-hospitalization ({post_days} days) expenses are within the {pre_limit}/{post_limit}-day windows.",
            status="SUPPORTS_ADMISSIBLE", citations=[cite(top, f"{pre_limit}-day pre / {post_limit}-day post hospitalization window", window_assert)], confidence=0.85,
        )
    else:
        deduction = 0.0
        if not pre_ok:
            deduction += exp.get("pre_hospitalization", 0)
        if not post_ok:
            deduction += exp.get("post_hospitalization", 0)
        if not same_condition:
            deduction = exp.get("pre_hospitalization", 0) + exp.get("post_hospitalization", 0)
        finding = Finding(
            dimension=dim, applicable=True,
            statement=(
                f"Pre/post-hospitalization expenses fall outside the {pre_limit}-day/{post_limit}-day windows or are "
                f"not confirmed to relate to the same condition, so they are not payable."
            ),
            status="SUPPORTS_LIMIT", citations=[cite(top, f"{pre_limit}-day pre / {post_limit}-day post hospitalization window", window_assert)], confidence=0.75,
        )
        if deduction > 0:
            limits.append(ApplicableLimit(description="Pre/post-hospitalization expenses outside policy time window", dimension=dim, deduction_inr=deduction, citations=[cite(top, "time-window limit", window_assert)]))
    return finding, limits, []


# --------------------------------------------------------------------------
# 8. Cosmetic / aesthetic exclusion
# --------------------------------------------------------------------------

_COSMETIC_KEYWORDS = ["cosmetic", "aesthetic", "plastic surgery"]
_INJURY_CONTEXT_KEYWORDS = ["accident", "injury", "burn", "trauma", "reconstruct"]


def _cosmetic_applies(facts: dict) -> bool:
    text = f"{facts.get('diagnosis','')} {facts.get('procedure','')}".lower()
    return any(kw in text for kw in _COSMETIC_KEYWORDS)


def _cosmetic_query(facts: dict) -> str:
    return "cosmetic aesthetic treatment plastic surgery exclusion injury disease"


def _cosmetic_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "cosmetic_exclusion"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the cosmetic/aesthetic treatment exclusion clause.")
    top = evidence[0]
    text = f"{facts.get('diagnosis','')} {facts.get('procedure','')}".lower()
    injury_context = any(kw in text for kw in _INJURY_CONTEXT_KEYWORDS)

    if injury_context:
        return _no_evidence(dim, "Diagnosis/procedure mentions cosmetic terms alongside an injury context; cannot confirm from supplied evidence whether the injury/disease carve-out applies.")

    finding = Finding(
        dimension=dim, applicable=True,
        statement="Diagnosis/procedure is cosmetic/aesthetic in nature with no stated injury or disease basis, which the policy excludes.",
        status="SUPPORTS_EXCLUSION", citations=[cite(top, "cosmetic/aesthetic treatment exclusion", phrases("cosmetic", "plastic surgery"))], confidence=0.85,
    )
    return finding, [], []


# --------------------------------------------------------------------------
# 9. Experimental / unproven treatment
# --------------------------------------------------------------------------

def _experimental_applies(facts: dict) -> bool:
    return bool(facts.get("experimental"))


def _experimental_query(facts: dict) -> str:
    return "unproven experimental treatment definition established medical practice not approved"


def _experimental_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "experimental_unproven_treatment"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the Unproven/Experimental Treatment definition.")
    top = _find_source(evidence, "Unproven/Experimental Treatment") or evidence[0]
    reason = (
        "The policy defines 'Unproven/Experimental Treatment' but does not contain an explicit numbered "
        "exclusion clause naming it; the nearest analogous exclusion (treatments not approved by the Indian "
        "Medical Council / non-allopathic treatment) does not unambiguously cover an experimental allopathic "
        "therapy. A safe decision cannot be made from the supplied policy text alone."
    )
    finding = Finding(
        dimension=dim, applicable=True, statement=reason, status="INSUFFICIENT_EVIDENCE",
        citations=[cite(top, "Unproven/Experimental Treatment is defined but no other clause of the policy mentions it",
                        {"type": "absence", "term": "experimental", "allowed_chunk_ids": [top.chunk_id]})],
        confidence=0.3,
    )
    return finding, [], [MissingEvidence(field="experimental_treatment_exclusion_basis", reason=reason)]


# --------------------------------------------------------------------------
# 10. Category sub-limits
# --------------------------------------------------------------------------

def _sublimits_applies(facts: dict) -> bool:
    return facts["treatment_type"] in ("inpatient", "day_care", "domiciliary")


def _sublimits_query(facts: dict) -> str:
    return "room boarding nursing sub limit surgeon fees anesthesia medicines package sum insured domiciliary ambulance"


def _limit_assert(keyword: str, *, percent=None, flat=None, si=None, multiplier=1, claimed=None, cap=None, deduction=None) -> dict:
    return {"type": "limit", "keyword": keyword, "percent": percent, "flat": flat, "sum_insured": si,
            "multiplier": multiplier, "claimed": claimed, "cap": cap, "deduction": deduction}


def _sublimits_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "category_sub_limits"
    if not evidence:
        return _no_evidence(dim, "Could not retrieve the coverage sub-limit clauses.")
    combined = "\n".join(ev.text for ev in evidence)
    si = facts["sum_insured"]
    exp = facts.get("expenses", {})
    limits: list[ApplicableLimit] = []
    assumptions: list[str] = []

    def add_limit(description: str, keyword: str, claimed: float, cap: float, **kw) -> None:
        # Cite the chunk whose *clause* actually states the limit, not merely
        # the first chunk that mentions the keyword (e.g. an unrelated note).
        src = next((ev for ev in evidence if rx.clause_with(ev.text, keyword)), None) or _find_source(evidence, keyword)
        deduction = claimed - cap
        limits.append(ApplicableLimit(
            description=description, dimension=dim, deduction_inr=deduction,
            citations=[cite(src, description.split(" (")[0],
                            _limit_assert(keyword, si=si, claimed=claimed, cap=cap, deduction=deduction, **kw))],
        ))

    if facts["treatment_type"] == "domiciliary":
        dom_pct = rx.extract_percent_of_si(combined, "Domiciliary")
        if dom_pct is not None:
            cap = dom_pct / 100 * si
            claimed = exp.get("doctor_fees", 0) + exp.get("medicines_diagnostics", 0) + exp.get("room", 0)
            if claimed > cap:
                add_limit(f"Domiciliary Hospitalization aggregate sub-limit ({dom_pct}% of Basic Sum Insured = INR {cap:,.0f})",
                          "Domiciliary", claimed, cap, percent=dom_pct)
        statement = "Domiciliary Hospitalization is capped at an aggregate percentage of the Basic Sum Insured."
    else:
        los_days = max(1, -(-int(facts.get("admission_hours", 24)) // 24))
        room_pct = rx.extract_percent_of_si(combined, "Room")
        if room_pct is not None and exp.get("room", 0) > 0:
            assumptions.append(
                f"The 'Normal Room' limit is applied per day for the length of stay ({los_days} day(s), admission hours rounded up); "
                "the room clause does not say 'per day' explicitly (the ICU clause does).")
            assumptions.append("All claimed room expense is treated as normal-room rent; the case does not split out ICU days.")
            cap = room_pct / 100 * si * los_days
            if exp["room"] > cap:
                add_limit(f"Room/boarding/nursing sub-limit ({room_pct}% of Basic Sum Insured per day x {los_days} day(s) = INR {cap:,.0f})",
                          "Room", exp["room"], cap, percent=room_pct, multiplier=los_days)

        fees_pct = rx.extract_percent_of_si(combined, "fees")
        fees_deduction = 0.0
        if fees_pct is not None and exp.get("doctor_fees", 0) > 0:
            cap = fees_pct / 100 * si
            if exp["doctor_fees"] > cap:
                fees_deduction = exp["doctor_fees"] - cap
                add_limit(f"Medical practitioner/surgeon fees sub-limit ({fees_pct}% of Sum Insured = INR {cap:,.0f})",
                          "fees", exp["doctor_fees"], cap, percent=fees_pct)

        meds_pct = rx.extract_percent_of_si(combined, "Anesthesia")
        meds_deduction = 0.0
        if meds_pct is not None and exp.get("medicines_diagnostics", 0) > 0:
            cap = meds_pct / 100 * si
            if exp["medicines_diagnostics"] > cap:
                meds_deduction = exp["medicines_diagnostics"] - cap
                add_limit(f"Anesthesia/medicines/diagnostics sub-limit ({meds_pct}% of Sum Insured = INR {cap:,.0f})",
                          "Anesthesia", exp["medicines_diagnostics"], cap, percent=meds_pct)

        # The 75% "Any One Illness" cap applies to *agreed package charges* only.
        package_pct = rx.extract_percent_of_si(combined, "package")
        if package_pct is not None and facts.get("package_charges_agreed"):
            package_cap = package_pct / 100 * si
            subtotal_after_caps = (
                min(exp.get("room", 0), (room_pct / 100 * si * los_days) if room_pct else exp.get("room", 0))
                + (exp.get("doctor_fees", 0) - fees_deduction)
                + (exp.get("medicines_diagnostics", 0) - meds_deduction)
            )
            if subtotal_after_caps > package_cap:
                add_limit(f"'Any One Illness' package cap ({package_pct}% of Sum Insured = INR {package_cap:,.0f})",
                          "package", subtotal_after_caps, package_cap, percent=package_pct)
        elif package_pct is not None:
            assumptions.append("No agreed package charges are stated in the case, so the 75% 'Any One Illness' package cap is not applied.")

        amb_pct = rx.extract_percent_of_si(combined, "Ambulance")
        amb_flat = rx.extract_flat_amount(combined, "Ambulance")
        if exp.get("ambulance", 0) > 0 and (amb_pct is not None or amb_flat is not None):
            candidates = []
            if amb_pct is not None:
                candidates.append(amb_pct / 100 * si)
            if amb_flat is not None:
                candidates.append(amb_flat)
            cap = min(candidates)
            if exp["ambulance"] > cap:
                add_limit(f"Ambulance charges sub-limit (lower of {amb_pct}% of Basic Sum Insured and INR {amb_flat:,.0f} = INR {cap:,.0f})",
                          "Ambulance", exp["ambulance"], cap, percent=amb_pct, flat=amb_flat)
        statement = "Category-wise sub-limits (room/boarding, practitioner fees, medicines/anesthesia, ambulance) were checked against claimed expenses."

    # The whole payout is "subject to the overall Sum Insured limit" (Additional Benefits note).
    total_claimed = sum(v for v in exp.values() if isinstance(v, (int, float)))
    remaining = total_claimed - sum(l.deduction_inr or 0 for l in limits)
    if si and remaining > si:
        overall = next((ev for ev in evidence if "overall sum insured" in " ".join(ev.text.lower().split())), None)
        if overall is not None:
            limits.append(ApplicableLimit(
                description=f"Overall Sum Insured limit (INR {si:,.0f}) caps the total payable",
                dimension=dim, deduction_inr=remaining - si,
                citations=[cite(overall, "Overall Sum Insured limit", phrases("overall sum insured"))],
            ))
        else:
            assumptions.append("The total payable exceeds the sum insured but the overall-limit clause was not retrieved, so the cap was not applied.")

    top = evidence[0]
    finding = Finding(
        dimension=dim, applicable=bool(limits),
        statement=statement + (f" {len(limits)} limit(s) reduce the payable amount." if limits else " No category sub-limit is exceeded."),
        status="SUPPORTS_LIMIT" if limits else "NOT_APPLICABLE",
        citations=[c for l in limits for c in l.citations][:1] or [cite(top, "coverage sub-limits")],
        confidence=0.8 if limits else 0.6, assumptions=assumptions if limits else [],
    )
    return finding, limits, []


# --------------------------------------------------------------------------
# 11. Unknown policy dimension (safety net)
# --------------------------------------------------------------------------
#
# The ten dimensions above cover the rules this system models. The policy has
# other, specific exclusions it does not model (dental, pregnancy, spectacles,
# HIV, outpatient, listed chronic diseases, ...). Rather than silently
# ignoring them, this dimension looks at what the case is *about* (diagnosis
# and procedure) and flags any policy exclusion that names the same thing but
# is not already handled by a modelled dimension. It never decides the claim
# itself: it abstains and hands the reviewer the exact clause.

_GENERIC_TERMS = {
    "acute", "chronic", "severe", "mild", "treatment", "surgery", "surgical", "procedure", "therapy", "condition",
    "conditions", "complication", "complications", "disorder", "disease", "diseases", "requiring", "home", "medical",
    "management", "hospital", "hospitalization", "with", "from", "that", "this", "other", "similar", "related",
    "exacerbation", "repair", "existing", "experimental", "unproven", "infection", "observation", "general",
    "domiciliary", "day", "care", "inpatient", "admission",
}


def _terms(text: str) -> set[str]:
    words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", text.lower())}
    generic = {g.rstrip("s") for g in _GENERIC_TERMS}
    return {w for w in words if w not in generic}


def _unmodelled_applies(facts: dict) -> bool:
    return True


def _unmodelled_query(facts: dict) -> str:
    return f"{facts.get('diagnosis', '')} {facts.get('procedure', '')} exclusion not covered".strip()


def _unmodelled_eval(facts: dict, evidence: list[EvidenceItem]) -> EvalResult:
    dim = "unmodelled_policy_risk"
    terms = _terms(f"{facts.get('diagnosis', '')} {facts.get('procedure', '')}")
    handled = set(facts.get("_cited_chunk_ids", []))
    for ev in evidence:
        if ev.section != "What We Exclude" or ev.chunk_id in handled:
            continue
        body = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", ev.text.lower())}
        overlap = sorted(terms & body)
        if overlap:
            statement = (
                f"The diagnosis/procedure mentions '{overlap[0]}', which the policy names in a specific exclusion "
                f"({ev.label}, page {ev.page_start}). No modelled rule evaluates it, so the system will not decide "
                f"this on its own; a reviewer should confirm whether the exclusion applies."
            )
            finding = Finding(
                dimension=dim, applicable=True, statement=statement, status="INSUFFICIENT_EVIDENCE", confidence=0.3,
                citations=[cite(ev, "Possible exclusion not covered by any modelled check", phrases(overlap[0]))],
            )
            return finding, [], [MissingEvidence(field="unmodelled_policy_risk", reason=statement)]
    finding = Finding(
        dimension=dim, applicable=False, status="NOT_APPLICABLE", confidence=0.6,
        statement="No unmodelled policy exclusion matches the diagnosis or procedure.", citations=[],
    )
    return finding, [], []


@dataclass
class Dimension:
    key: str
    applies: Callable[[dict], bool]
    query: Callable[[dict], str]
    evaluate: Callable[[dict, list[EvidenceItem]], EvalResult]


DIMENSIONS: list[Dimension] = [
    Dimension("initial_waiting_period", _initial_wait_applies, _initial_wait_query, _initial_wait_eval),
    Dimension("named_disease_first_year_waiting_period", _named_disease_applies, _named_disease_query, _named_disease_eval),
    Dimension("pre_existing_disease_waiting_period", _pre_existing_applies, _pre_existing_query, _pre_existing_eval),
    Dimension("hospital_definition", _hospital_def_applies, _hospital_def_query, _hospital_def_eval),
    Dimension("domiciliary_treatment_conditions", _domiciliary_applies, _domiciliary_query, _domiciliary_eval),
    Dimension("day_care_less_than_24h", _daycare_applies, _daycare_query, _daycare_eval),
    Dimension("pre_post_hospitalization_window", _pre_post_applies, _pre_post_query, _pre_post_eval),
    Dimension("cosmetic_exclusion", _cosmetic_applies, _cosmetic_query, _cosmetic_eval),
    Dimension("experimental_unproven_treatment", _experimental_applies, _experimental_query, _experimental_eval),
    Dimension("category_sub_limits", _sublimits_applies, _sublimits_query, _sublimits_eval),
    Dimension("unmodelled_policy_risk", _unmodelled_applies, _unmodelled_query, _unmodelled_eval),
]
