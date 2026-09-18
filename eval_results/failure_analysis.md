# Failure Analysis

Four failures found during development (via direct pipeline smoke-testing
against the supplied cases, not just eyeballing code), their root causes,
and the fix applied. All four are still guarded by the unit tests in
`tests/` so they can't silently regress.

## 1. Chunker: a 3-letter connector word was detected as a section heading

**Symptom**: `WHAT WE EXCLUDE` produced a single 11-line chunk instead of
21 separate exclusion items.

**Root cause**: the all-caps heading detector (`_detect_heading` in
`ingestion/chunker.py`) matched any all-caps line of length ≥3. Item 1's
body text contains a standalone line `"AND"` (the PDF wraps a sentence
across lines and a stray connector lands alone), which the regex accepted
as a new section heading, prematurely closing the "What We Exclude"
section after item 1's first few lines. Items 2–21 were then silently
dropped from that section.

**Fix**: added a stopword list (`AND`, `OR`, `NB`, `IF`, ...) and a
minimum heading length of 5 characters. Covered by
`tests/test_chunker.py::test_heading_stopwords_are_not_treated_as_sections`.

## 2. Chunker: nested numbering mistaken for new top-level clauses

**Symptom**: "Standard Terms and Conditions" (13 real numbered items)
produced 25 chunks, with labels like "Item 1" and "Item 2" appearing more
than once for unrelated clauses.

**Root cause**: item 10 (Portability) contains its own nested list
restarting at "1." and "2." (`"1. If You wish to exercise..."`). The
numbered-clause chunker treated *any* line matching `^\d+\.\s` as a new
top-level item, so Portability's internal sub-points were split out as
fake "Item 1" / "Item 2" chunks, colliding with the real items 1 and 2
(Notice, Mis-description) in the reviewer-facing label.

**Fix**: a numbered line is only treated as a new top-level clause if its
number is a strict `+1` continuation of the current item; anything else is
treated as a continuation of the current clause. Covered by
`tests/test_chunker.py::test_numbered_section_splits_top_level_items_only`.

## 3. Reasoning: a network-provider hospital silently overrode an explicit "unresolved" evidence flag

**Symptom**: PUB-006 (`evidence_context: {"hospital_registered": null,
"medical_necessity_confirmed": null}`, task: *"identify what additional
evidence is required before a final decision"*) initially resolved to
`ADMISSIBLE_WITH_LIMITS` instead of the clearly-intended `NEEDS_REVIEW`.

**Root cause**: the `hospital_definition` dimension checked
`hospital_network_provider` *before* looking at `evidence_context`. Prime
Hospital is a network provider, so the dimension short-circuited to
"presumed to satisfy the Hospital definition" and never inspected the
explicit `null` flags the case was deliberately supplying. More broadly,
the system had no generic way to treat an explicit `null` in
`evidence_context` as a "deliberately unresolved" signal — it only reacted
to specific `False` values inside one dimension.

**Fix**: `case_analysis_agent.py` now scans `evidence_context` for any key
with value `null` and records it as an `_explicit_unknowns` fact,
independent of which dimension it relates to. `decision_agent.py` treats a
non-empty `_explicit_unknowns` list as a hard, first-priority reason to
return `NEEDS_REVIEW`, regardless of what any individual dimension
concluded. This is a *generic* rule (not a PUB-006-specific patch), so it
also correctly reinforces PUB-011's `hospital_registered: null`.
Covered by `tests/test_decision_agent.py::test_explicit_null_evidence_forces_needs_review_even_if_no_finding_is_insufficient`.

## 4. Reasoning: missing (optional) timing metadata over-triggered abstention/deductions

**Symptom**: in an early version, any case with nonzero
`pre_hospitalization`/`post_hospitalization` expenses but no
`expense_timing` block (most of the supplied cases) resolved to either
`NEEDS_REVIEW` or `PARTIALLY_ADMISSIBLE`, even though the case gave no
actual reason to doubt those expenses — `expense_timing` is an optional
field the schema doc explicitly lists, and only one supplied case
(PUB-009) populates it.

**Root cause**: the `pre_post_hospitalization_window` dimension treated
"timing not supplied" the same as "timing supplied and out of window" —
either full abstention or a full deduction of the pre/post amount. This
conflated "we weren't told" with "we were told it's a problem", which is
too aggressive: it would have made the majority of otherwise-clean public
cases abstain or lose money for no case-specific reason, diluting the
signal of the genuinely evidence-deficient cases (PUB-006, PUB-011,
PUB-012).

**Fix**: missing timing metadata now only produces a transparency note in
`missing_evidence` (visible to the reviewer) with no effect on the
decision or payable amount — the claim is assumed within the 30/60-day
window pending documentation. Only a *confirmed* window violation
(`expense_timing` present and out of range, exercised by the custom case
`CUST-003`) produces a deduction/`PARTIALLY_ADMISSIBLE`. This is the
proportionality principle documented in `ARCHITECTURE.md`: abstention and
deductions are reserved for confirmed problems, not merely-optional gaps.

## Residual limitations (not failures, but worth naming)

- The system has no eleventh "catch-all" dimension for a policy question
  outside the ten modeled ones (see `ARCHITECTURE.md` §6) — a genuinely
  novel fact pattern would need a new dimension added, not be caught
  generically.
- Sub-limit math does not separate ICU-day room charges from ward-day room
  charges because the input schema only supplies one `room` figure.
