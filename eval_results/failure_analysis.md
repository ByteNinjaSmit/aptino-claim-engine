# Failure Analysis

Fourteen failures found during development, their root causes, and the fix applied. #1-#4 came from smoke-testing
the supplied cases; #5-#10 were found by the stronger evaluation (hand-derived payable amounts, the claim -> citation ->
chunk verifier, new edge cases); #11 records limitations found by reading the policy closely; #12-#14 came from building and measuring the optional LLM step. Fixed items are guarded by
unit tests and by gates in the evaluation, so they cannot silently regress.

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
also correctly reinforces PUB-011's `hospital_registered: null`. (Precedence was later refined: see #9 -- a verified exclusion outranks it.)
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

## 5. Payable amounts were wrong even though every decision label was right

**Found by**: the new payable-amount check (hand-derived expected deductions, `eval/expected_outcomes.py`). Decision accuracy alone was 100% and would have hidden this.

**Symptom**: PUB-007 (large cancer bill) reported deductions of INR 130,000 instead of INR 190,500: the 40% cap on medicines/diagnostics was never applied, and a 75% "package" cap was applied in its place.

**Root cause**: two independent extraction defects in `agents/rules_extract.py`. (a) The regex required the wording "40% **of** Sum Insured" but the policy says "40% Sum Insured" for that clause, so it never matched. (b) Keywords were bound to numbers by a 120-character window, which can straddle two neighbouring clauses and pick up the wrong figure.

**Fix**: percentages and flat caps are now extracted **within a single clause** (split on numbered items, `NB` notes and `a)`/`b)` sub-points), with "of" optional. Covered by `tests/test_extraction_and_unknown.py::test_every_category_limit_is_extracted_from_the_real_policy_text`.

## 6. The ambulance cap ("lower of 1% and Rs 1000") was silently ignored

**Found by**: same amount check (PUB-001 should lose INR 200 on a INR 1,200 ambulance bill; it lost nothing).

**Root cause**: the rupee figure sat more than the old window's distance from the word "Ambulance", so only the percentage (INR 5,000) was used and the "whichever is less" flat cap never applied.

**Fix**: the clause-level extraction above; the cap is now `min(1% of SI, Rs 1000)`. Locked in by the same test plus the amount-exactness gate in the evaluation (11/11 hand-derived deductions must match).

## 7. A limit was cited to the wrong policy chunk

**Found by**: the new claim -> citation -> chunk verifier (SUPPORTED / UNSUPPORTED / CONTRADICTED). The old validator only checked that a cited chunk id had been retrieved, so it could never notice this.

**Symptom**: the ambulance limit cited `extensions-001`, a chunk that merely *mentions* ambulances, while the limit itself lives in `what_we_cover-005`. The verifier reported `UNSUPPORTED: no percentage limit found near 'Ambulance' in the chunk`, the retry loop re-retrieved, still failed, and the case was (correctly, given the citation) downgraded to NEEDS_REVIEW.

**Root cause**: `_find_source()` returned the first retrieved chunk containing the keyword anywhere.

**Fix**: limit citations now pick the chunk whose *clause* actually states the limit. After the fix all 128 claims across 26 cases verify, and gold-citation precision is 100%.

## 8. The 75% "Any One Illness" package cap was applied to ordinary claims

**Found by**: reading the clause while hand-deriving PUB-007's expected deduction.

**Root cause**: NB3 restricts expenses "under **agreed package charges**" to 75% of the sum insured. No supplied case says a package was agreed, yet the cap was applied to every bill.

**Fix**: the cap is applied only when the case states `package_charges_agreed`; otherwise the response lists an explicit assumption ("no agreed package charges stated, so the 75% cap is not applied"). Covered by `test_package_cap_only_applies_when_a_package_was_agreed`.

## 9. An excluded claim could be turned into NEEDS_REVIEW by an unrelated open question

**Found by**: designing CUST-012 (cosmetic surgery, established exclusion, plus an unresolved hospital-registration flag).

**Root cause**: decision precedence put "anything unresolved" ahead of "a verified exclusion", so a claim with a definite ground for rejection was reported as undecidable.

**Fix**: precedence is now missing-dates > verified exclusion > unresolved evidence > limits > admissible. A confirmed exclusion still wins, at reduced confidence (0.75) and with the open questions listed. Covered by `test_confirmed_exclusion_outranks_unresolved_side_questions`.

## 10. The unknown-policy-dimension check produced a false alarm

**Found by**: running all 18 cases after adding the check; PUB-004 flipped to NEEDS_REVIEW.

**Root cause**: the check matched the word "domiciliary" in the procedure ("Domiciliary treatment") against exclusion item 17, but "domiciliary" is a *treatment mode* already handled by a modelled dimension, not a disease named by an exclusion.

**Fix**: treatment-mode words (domiciliary, day, care, inpatient, ...) and generic clinical words (infection, acute, therapy, ...) are excluded from matching, and clauses already cited by a modelled dimension are skipped. Covered by `test_generic_words_do_not_trigger_the_unknown_dimension` and `test_clauses_already_used_by_a_modelled_check_are_not_reflagged`.

## 11. Two limitations discovered by reading the policy more closely (not fixed)

- **Exclusion items 18-20 belong to item 17.** In the PDF, "17. Any expense under Domiciliary Hospitalisation for" is followed by items 18 (pre/post hospitalisation), 19 ("treatment not exceeding three days") and 20 (the disease list: asthma, bronchitis, diabetes, ...). Read together they restrict *domiciliary* treatment only. The chunker splits them into separate top-level items. The unknown-dimension check still surfaces item 20 for an asthma claim (CUST-006), but a re-chunk that keeps 17-20 together would model this correctly. Not done: the numbering is the only signal and a generic rule for it is fragile.
- **The "Normal Room" limit is ambiguous.** The ICU clause says "per day"; the room clause does not. The system applies it per day of stay and says so in `assumptions` on every affected answer. If the insurer meant per stay, PUB-001's deduction would be INR 25,000, not INR 10,200.

## 12. Keyword matching used substrings, not words

**Found by**: building the paraphrase test cases for the LLM step (ADV-002, "symptomatic gallstones / laparoscopic cholecystectomy").

**Symptom**: the case was rejected as a first-year named-disease claim, which is the right outcome but for the wrong reason: the keyword `cyst` matched *inside* "chole**cyst**ectomy".

**Fix**: policy terms match at the start of a word (`dimensions.has_term`): "cyst" matches "cysts" but not "cholecystectomy". Covered by `test_terms_match_at_word_start_not_inside_other_words`. This removed one spurious claim from CUST-004 (127 claims verified instead of 128); no decision changed.

## 13. The first LLM-interpretation design was net-negative with a weak model

**Found by**: running the step against a real model (`eval/run_llm_mode.py`), which the unit tests with a scripted model could not reveal.

**Symptom**: with `gemini-2.5-flash-lite`, labeled accuracy fell from 100% to 65% because the model flagged clauses that plainly did not apply: it re-raised sub-limits the rules already compute, called a 96-hour stay "unmet", and paired "inpatient" with "outpatient" and "Cancer" with "infertility".

**Root causes / fixes, in order**: (a) it was shown retrieval-ranked chunks from every section, so it commented on definitions and limits: it is now shown the full exclusions list only and `limit_risk` was removed; (b) a self-reported `applies` flag did nothing (the model always answered "yes") and was dropped; (c) the model reasoned about durations, which is the rules' job: numeric fields were removed from its scope; (d) it now must show a **verbatim bridge** (`case_span` from the case, `clause_term` from the clause), and bridges made only of generic words are rejected. That moved the weak model to 73%; a stronger model (`gemini-3.5-flash`) reached 96% with 3/3 paraphrase catches. The remaining lesson is recorded in the README: the guards bound the damage but do not make a weak model accurate.

## 14. A verbatim quote proves the text exists, not that the reasoning is right

**Found by**: reading the accepted observations for the weak model: it "caught" ADV-002 by pairing *cholecystectomy* with *hysterectomy* (both "-ectomy"), and the quote check passed because the quote was genuine.

**Consequence**: the quote/bridge verification guarantees grounding (no invented clauses or figures), not semantic correctness. That is why the step can only add review flags, and why its measured false-alarm rate is reported instead of assumed to be zero.

## Residual limitations (not failures, but worth naming)

- The system has no eleventh "catch-all" dimension for a policy question
  outside the ten modeled ones (see `ARCHITECTURE.md` §6) — a genuinely
  novel fact pattern would need a new dimension added, not be caught
  generically.
- Sub-limit math does not separate ICU-day room charges from ward-day room
  charges because the input schema only supplies one `room` figure.
