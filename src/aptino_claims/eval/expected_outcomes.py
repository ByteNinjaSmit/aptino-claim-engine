"""Hand-labeled expected outcomes used by the evaluation script.

How labels were established: each case's `task` and JSON facts were read
against the specific policy clauses that govern that fact pattern, and the
outcome (and, where limits apply, the exact rupee deduction) was derived by
hand from the clause text -- e.g. 1% of sum insured per day for room, 25% for
practitioner fees, 40% for medicines/diagnostics, "lower of 1% or Rs 1000"
for ambulance. The policy is small and fixed, so every label is independently
checkable by reading the cited clause; there is no external answer key.

Two honest caveats (repeated in the report): the labels are the author's
reading of the policy, not an insurer's adjudication, and several limit
computations rest on stated interpretive assumptions (see `assumptions` in
the API response, e.g. room limit applied per day of stay).

`primary_dimensions`: the decision dimension(s) that are the crux of the case.
`expected_deduction_inr`: hand-derived total limit deduction for decisive
cases (None when the case is not payable or the system should abstain).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpectedOutcome:
    decision: str
    primary_dimensions: list[str]
    rationale: str
    expected_deduction_inr: float | None = None


def _o(decision, dims, why, ded=None):
    return ExpectedOutcome(decision, dims, why, ded)


EXPECTED: dict[str, ExpectedOutcome] = {
    "PUB-001": _o("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"],
                  "Room INR 30,000 vs 1% x 500,000 x 4 days = 20,000 (-10,000); ambulance INR 1,200 vs lower of 1% (5,000) and Rs 1,000 (-200).", 10200),
    "PUB-002": _o("NOT_ADMISSIBLE", ["initial_waiting_period"],
                  "Claim falls 19 days after inception, inside the 30-day initial waiting period, with no continuity exception.", 0),
    "PUB-003": _o("NOT_ADMISSIBLE", ["pre_existing_disease_waiting_period"],
                  "27 months of continuous coverage is short of the 48-month pre-existing-disease waiting period.", 0),
    "PUB-004": _o("ADMISSIBLE_WITH_LIMITS", ["domiciliary_treatment_conditions", "category_sub_limits"],
                  "Room unavailability satisfies Domiciliary Treatment; 25,000 + 80,000 = 105,000 exceeds the 20% aggregate cap of 100,000 (-5,000).", 5000),
    "PUB-005": _o("ADMISSIBLE", ["day_care_less_than_24h"],
                  "Cataract/eye surgery is explicitly named as a covered <24h day-care procedure; no cap is exceeded.", 0),
    "PUB-006": _o("NEEDS_REVIEW", ["hospital_definition"],
                  "Case explicitly marks hospital_registered and medical_necessity_confirmed as unresolved (null)."),
    "PUB-007": _o("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"],
                  "SI 1,000,000, 4 days: room 80,000 vs 40,000 (-40,000); fees 300,000 vs 250,000 (-50,000); medicines 500,000 vs 400,000 (-100,000); ambulance 1,500 vs 1,000 (-500).", 190500),
    "PUB-008": _o("NOT_ADMISSIBLE", ["cosmetic_exclusion"],
                  "Cosmetic surgery with no injury/disease basis is explicitly excluded.", 0),
    "PUB-009": _o("ADMISSIBLE", ["pre_post_hospitalization_window"],
                  "30-day pre / 60-day post windows are both satisfied and same-condition is confirmed; no cap is exceeded.", 0),
    "PUB-010": _o("ADMISSIBLE", ["named_disease_first_year_waiting_period"],
                  "Cataract falls in the first policy year, but 1 completed year with a prior Indian insurer (history received) waives the named-disease wait.", 0),
    "PUB-011": _o("NEEDS_REVIEW", ["hospital_definition"],
                  "Non-network facility whose registration and minimum criteria are explicitly undocumented; the Hospital definition cannot be confirmed."),
    "PUB-012": _o("NEEDS_REVIEW", ["experimental_unproven_treatment"],
                  "Policy defines Unproven/Experimental Treatment but has no clause that unambiguously excludes it."),
    "CUST-001": _o("NOT_ADMISSIBLE", ["named_disease_first_year_waiting_period"],
                   "Hernia within the first policy year, no prior-insurer continuity to waive the named-disease wait.", 0),
    "CUST-002": _o("ADMISSIBLE", ["named_disease_first_year_waiting_period"],
                   "Hysterectomy within the first year, but 2 completed prior-insurer years (history received) waive the wait; no cap is exceeded.", 0),
    "CUST-003": _o("PARTIALLY_ADMISSIBLE", ["pre_post_hospitalization_window", "category_sub_limits"],
                   "Pre-hospitalization incurred 45 days before admission (outside 30) -> -12,000; room 25,000 vs 20,000 -> -5,000.", 17000),
    "CUST-004": _o("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"],
                   "SI 300,000, 4 days: room 20,000 vs 12,000 (-8,000); irrelevant attributes are ignored.", 8000),
    "CUST-005": _o("NEEDS_REVIEW", ["experimental_unproven_treatment"],
                   "Experimental stem-cell therapy: same abstention basis as PUB-012."),
    "CUST-006": _o("NEEDS_REVIEW", ["domiciliary_treatment_conditions", "unmodelled_policy_risk"],
                   "Neither Domiciliary Treatment condition is stated, and asthma is named in a specific exclusion no modelled check evaluates."),
    "CUST-007": _o("ADMISSIBLE", ["pre_existing_disease_waiting_period"],
                   "Pre-existing condition: 14 months + 4 completed prior-insurer years = 62 >= 48 months; no cap is exceeded.", 0),
    "CUST-008": _o("NEEDS_REVIEW", ["unmodelled_policy_risk"],
                   "Dental treatment is named in a specific exclusion that no modelled check evaluates; surface it for a reviewer."),
    "CUST-009": _o("NEEDS_REVIEW", ["unmodelled_policy_risk"],
                   "Pregnancy/childbirth is named in a specific exclusion that no modelled check evaluates; surface it for a reviewer."),
    "CUST-010": _o("NEEDS_REVIEW", ["day_care_less_than_24h"],
                   "12-hour stay for a procedure not on the named short-stay list; medical justification is not supplied."),
    "CUST-011": _o("NOT_ADMISSIBLE", ["initial_waiting_period", "cosmetic_exclusion"],
                   "Two independent grounds: 14 days after inception (inside the 30-day wait) and a cosmetic procedure.", 0),
    "CUST-012": _o("NOT_ADMISSIBLE", ["cosmetic_exclusion"],
                   "The cosmetic exclusion is established; the unresolved hospital-registration question cannot rescue an excluded claim.", 0),
    "CUST-013": _o("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"],
                   "SI 200,000, 2 days: room 12,000 vs 4,000 (-8,000); fees 60,000 vs 50,000 (-10,000); medicines 90,000 vs 80,000 (-10,000); ambulance 1,500 vs 1,000 (-500).", 28500),
    "CUST-014": _o("NOT_ADMISSIBLE", ["named_disease_first_year_waiting_period"],
                   "First-year cataract; the prior-insurer reduction applies only if the previous insurer's database and claim history were received (they were not).", 0),
}

# Cases the assignment brief specifically requires to resolve as NEEDS_REVIEW,
# plus the abstention cases added here.
REQUIRED_ABSTENTIONS = [cid for cid, o in EXPECTED.items() if o.decision == "NEEDS_REVIEW"]

DECISION_CLASSES = ["ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE", "NEEDS_REVIEW"]
