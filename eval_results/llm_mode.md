# LLM interpretation step: evaluation

Model `gemini-3.5-flash`, temperature 0. Not part of the deterministic gate (see `run_llm_mode.py`).

| Measure | Result |
|---|---|
| Labeled-case accuracy, deterministic | 100.0% |
| Labeled-case accuracy, with LLM step | 96.2% |
| False alarms (decisive case sent to review) | 1 (6% of decisive cases): PUB-004 |
| Moved to a different *decisive* outcome | 0 (0 by construction: the step can only add review flags; listed to expose a bug) |
| Adversarial cases held for review: deterministic / with LLM | 0 / 3 of 3 |
| Observations accepted / rejected | 12 / 6 |
| Validation failures | 0 |

## Cases where the step changed the outcome or made observations

| Case | Expected | Deterministic | With LLM | Accepted observations (verified quote) | Rejected |
|---|---|---|---|---|---|
| PUB-002 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | NOT_ADMISSIBLE | what_we_exclude-020: "Pyrexia of unknown origin for less than 15 days" (exclusion_risk) | - |
| PUB-004 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | NEEDS_REVIEW | what_we_exclude-017: "Any expense under Domiciliary Hospitalisation for" (exclusion_risk) | - |
| PUB-005 | ADMISSIBLE | ADMISSIBLE | ADMISSIBLE | - | bridge is made only of generic words |
| PUB-010 | ADMISSIBLE | ADMISSIBLE | ADMISSIBLE | - | bridge is made only of generic words |
| PUB-012 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-014: "Any expense on Naturopathy, non-allopathic treatment and/or " (exclusion_risk) | - |
| CUST-005 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-020: "xii) Arthritis, Gout and Rheumatism" (exclusion_risk); what_we_exclude-003: "vi) Arthritis, Gout, Rheumatism" (exclusion_risk) | - |
| CUST-006 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-016: "instruments used in treatment of sleep apnea syndrome (C.P.A" (exclusion_risk) | - |
| CUST-007 | ADMISSIBLE | ADMISSIBLE | ADMISSIBLE | - | bridge is made only of generic words |
| CUST-008 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-020: "xiii) Dental Treatment or Surgery" (exclusion_risk) | - |
| CUST-009 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-020: "vii) Hypertension" (exclusion_risk) | - |
| CUST-010 | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | what_we_exclude-020: "Treatment of following diseases: i) Asthma" (exclusion_risk) | bridge is made only of generic words |
| CUST-013 | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | - | bridge is made only of generic words |
| CUST-014 | NOT_ADMISSIBLE | NOT_ADMISSIBLE | NOT_ADMISSIBLE | - | bridge is made only of generic words |
| ADV-001 | NEEDS_REVIEW | ADMISSIBLE | NEEDS_REVIEW | what_we_exclude-003: "Hospitalization expense incurred in the first year of operat" (condition_unmet_risk) | - |
| ADV-002 | NEEDS_REVIEW | ADMISSIBLE | NEEDS_REVIEW | what_we_exclude-003: "ix) Stone in the urinary and biliary systems" (exclusion_risk) | - |
| ADV-003 | NEEDS_REVIEW | ADMISSIBLE | NEEDS_REVIEW | what_we_exclude-017: "Any expense under Domiciliary Hospitalisation for" (exclusion_risk) | - |

The step can only add INSUFFICIENT_EVIDENCE findings backed by a verbatim quote, so a change is always toward NEEDS_REVIEW.
A false alarm costs a human look; a catch prevents a confident wrong answer on a paraphrase the rules do not match.

## How much to trust this

- 26 labeled + 3 adversarial cases. Two full runs (before/after audit fixes) gave the same headline and differed by one accepted observation; the guards were tuned over several runs on these same cases.
- 'Unchanged' is measured against a deterministic system that already scores 100% on these labels, so the metric can only go down: it is a harmlessness check, not evidence of benefit.
- The adversarial cases are labeled NEEDS_REVIEW because that is the only outcome the step can reach. A careful reviewer might call ADV-001/002 NOT_ADMISSIBLE (first-year wait, no waiver); 3/3 therefore measures 'held for a human', not 'decided correctly'.
- Temperature 0 does not make a hosted model deterministic: observation counts varied between the two runs.
- The individual bridges (e.g. 'symptomatic gallstones' to the biliary-stone clause) are the strongest evidence, not the percentages.
