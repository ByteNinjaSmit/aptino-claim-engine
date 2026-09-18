"""Hand-labeled expected outcomes used by the evaluation script.

How the expected outcome was established: each case's `task` field plus its
JSON facts were checked by hand against the specific policy clauses that
govern that fact pattern (see ARCHITECTURE.md / README "Decision Quality"
section for the clause-by-clause reasoning, e.g. the 30-day initial wait,
the 48-month pre-existing-disease wait, the 20%/25%/40%/75% sub-limits, the
named-disease first-year wait, and the two explicit `null` evidence-context
cases). This is a small, fixed policy document, so the correct outcome is
independently verifiable by reading the cited clause -- there is no external
"answer key"; the label is derived the same way a human claims reviewer
would derive it.

`primary_dimensions`: which decision dimension(s) are the crux of the case,
used for the retrieval-quality (evidence keyword recall) metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpectedOutcome:
    decision: str
    primary_dimensions: list[str]
    rationale: str


EXPECTED: dict[str, ExpectedOutcome] = {
    "PUB-001": ExpectedOutcome("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"], "Room expense exceeds the 1%/day sub-limit; otherwise a clean admissible appendectomy."),
    "PUB-002": ExpectedOutcome("NOT_ADMISSIBLE", ["initial_waiting_period"], "Claim falls 19 days after inception, inside the 30-day initial waiting period, with no continuity exception."),
    "PUB-003": ExpectedOutcome("NOT_ADMISSIBLE", ["pre_existing_disease_waiting_period"], "27 months of continuous coverage is short of the 48-month pre-existing-disease waiting period."),
    "PUB-004": ExpectedOutcome("ADMISSIBLE_WITH_LIMITS", ["domiciliary_treatment_conditions", "category_sub_limits"], "Room unavailability satisfies Domiciliary Treatment; claimed amount exceeds the 20% aggregate sub-limit."),
    "PUB-005": ExpectedOutcome("ADMISSIBLE", ["day_care_less_than_24h"], "Cataract/eye surgery is explicitly named as a covered <24h day-care procedure."),
    "PUB-006": ExpectedOutcome("NEEDS_REVIEW", ["hospital_definition"], "Case explicitly marks hospital_registered and medical_necessity_confirmed as unresolved (null)."),
    "PUB-007": ExpectedOutcome("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"], "Surgeon fees, medicines/diagnostics, and the 'Any One Illness' package cap all bind for this large cancer-treatment bill."),
    "PUB-008": ExpectedOutcome("NOT_ADMISSIBLE", ["cosmetic_exclusion"], "Cosmetic surgery with no injury/disease basis is explicitly excluded."),
    "PUB-009": ExpectedOutcome("ADMISSIBLE", ["pre_post_hospitalization_window"], "30-day pre / 60-day post windows are both satisfied and same-condition is confirmed."),
    "PUB-010": ExpectedOutcome("ADMISSIBLE", ["named_disease_first_year_waiting_period"], "Cataract falls in the first policy year, but 1 completed year with a prior Indian insurer (history received) waives the named-disease wait."),
    "PUB-011": ExpectedOutcome("NEEDS_REVIEW", ["hospital_definition"], "Non-network, unregistered-status facility with minimum-criteria explicitly undocumented -- Hospital definition cannot be confirmed."),
    "PUB-012": ExpectedOutcome("NEEDS_REVIEW", ["experimental_unproven_treatment"], "Policy defines Unproven/Experimental Treatment but has no clause unambiguously excluding it."),
    "CUST-001": ExpectedOutcome("NOT_ADMISSIBLE", ["named_disease_first_year_waiting_period"], "Hernia within the first policy year, no prior-insurer continuity to waive the named-disease wait."),
    "CUST-002": ExpectedOutcome("ADMISSIBLE", ["named_disease_first_year_waiting_period"], "Hysterectomy within the first policy year, but 2 completed prior-insurer years waive the named-disease wait."),
    "CUST-003": ExpectedOutcome("PARTIALLY_ADMISSIBLE", ["pre_post_hospitalization_window"], "Pre-hospitalization expenses were incurred 45 days before admission, outside the 30-day window; the hospitalization itself is admissible."),
    "CUST-004": ExpectedOutcome("ADMISSIBLE_WITH_LIMITS", ["category_sub_limits"], "Room expense exceeds sub-limit; blood type / parking / room-type preference are correctly ignored."),
    "CUST-005": ExpectedOutcome("NEEDS_REVIEW", ["experimental_unproven_treatment"], "Experimental stem-cell therapy -- same abstention basis as PUB-012."),
    "CUST-006": ExpectedOutcome("NEEDS_REVIEW", ["domiciliary_treatment_conditions"], "Neither Domiciliary Treatment condition (room unavailability / cannot be moved) is stated."),
}

# Cases the assignment brief specifically requires to resolve as NEEDS_REVIEW.
REQUIRED_ABSTENTIONS = ["PUB-006", "PUB-011", "PUB-012", "CUST-005", "CUST-006"]

# Keyword(s) that must appear in at least one retrieved evidence chunk for a
# given dimension for retrieval to be considered "on target" -- the
# retrieval-quality proxy used in place of exact chunk-id gold labels (which
# would be brittle to re-chunking) or an external relevance judgment set
# (which does not exist for a bespoke policy document).
DIMENSION_EVIDENCE_KEYWORDS: dict[str, list[str]] = {
    "initial_waiting_period": ["30 days"],
    "named_disease_first_year_waiting_period": ["first year"],
    "pre_existing_disease_waiting_period": ["48 months"],
    "hospital_definition": ["Hospital means", "in-patient beds"],
    "domiciliary_treatment_conditions": ["Domiciliary Treatment means"],
    "day_care_less_than_24h": ["24 hrs", "24 hours"],
    "pre_post_hospitalization_window": ["immediately preceding", "immediately following"],
    "cosmetic_exclusion": ["cosmetic"],
    "experimental_unproven_treatment": ["Unproven", "Experimental"],
    "category_sub_limits": ["Sum Insured", "Sum Assured"],
}
