from aptino_claims.agents.state import ApplicableLimit, CaseState, Citation, DecisionStatus, EvidenceItem, Finding
from aptino_claims.agents.verification import verify_assertion, verify_state

WAIT = "2. 30 days Waiting Period A waiting period of 30 days will apply to all claims unless continuously covered."
LIMITS = ("a) Normal Room expenses: 1.0% of Basic Sum Insured. "
          "3. Expenses on Anesthesia, Blood, Oxygen, Medicines and similar expenses subject to a limit of 40% Sum Insured . "
          "b) Ambulance charges in connection with any admissible claim limited to 1.0% of the Basic Sum Insured or Rupees 1000/- whichever is less for each claim.")


def _v(assertion, text, corpus=None, cid="c1"):
    return verify_assertion(assertion, text, corpus or {cid: text}, cid)[0]


def test_threshold_supported_contradicted_unsupported():
    a = {"type": "threshold", "values": [{"value": 30, "unit": "days"}], "must_contain": []}
    assert _v(a, WAIT) == "SUPPORTED"
    a["values"][0]["value"] = 45
    assert _v(a, WAIT) == "CONTRADICTED"          # chunk states a different day count
    assert _v(a, "no numbers of that kind here") == "UNSUPPORTED"


def test_phrases_require_all():
    assert _v({"type": "phrases", "must_contain": ["waiting period", "30 days"]}, WAIT) == "SUPPORTED"
    assert _v({"type": "phrases", "must_contain": ["waiting period", "cosmetic"]}, WAIT) == "UNSUPPORTED"


def test_limit_percent_flat_and_arithmetic():
    ok = {"type": "limit", "keyword": "Ambulance", "percent": 1.0, "flat": 1000, "sum_insured": 500000,
          "multiplier": 1, "claimed": 1200, "cap": 1000, "deduction": 200}
    assert _v(ok, LIMITS) == "SUPPORTED"
    assert _v({**ok, "percent": 2.0}, LIMITS) == "CONTRADICTED"       # wrong percentage
    assert _v({**ok, "deduction": 5000}, LIMITS) == "CONTRADICTED"    # arithmetic does not add up


def test_limit_without_of_in_policy_wording():
    # the policy writes "40% Sum Insured" (no "of"); the old extractor missed it
    a = {"type": "limit", "keyword": "Anesthesia", "percent": 40.0, "sum_insured": 1_000_000, "claimed": 500000,
         "cap": 400000, "deduction": 100000}
    assert _v(a, LIMITS) == "SUPPORTED"


def test_absence_is_contradicted_when_term_appears_elsewhere():
    corpus = {"def": "Unproven/Experimental Treatment means ...", "excl": "Any experimental therapy is excluded."}
    a = {"type": "absence", "term": "experimental", "allowed_chunk_ids": ["def"]}
    assert verify_assertion(a, corpus["def"], corpus, "def")[0] == "CONTRADICTED"
    assert verify_assertion(a, corpus["def"], {"def": corpus["def"]}, "def")[0] == "SUPPORTED"


def _state(decision, findings=(), limits=(), evidence_text=WAIT):
    s = CaseState(case_id="T", raw_case={})
    s.facts = {"_explicit_unknowns": []}
    s.decision = DecisionStatus(decision)
    s.findings = list(findings)
    s.applicable_limits = list(limits)
    s.evidence_by_dimension = {"initial_waiting_period": [
        EvidenceItem(chunk_id="c1", section="What We Exclude", label="x", page_start=9, page_end=9, text=evidence_text)]}
    return s


def _finding(status, chunk="c1", assertion=None):
    cit = Citation(claim="30-day wait", page=9, section="What We Exclude", chunk_id=chunk,
                   assertion=assertion or {"type": "threshold", "values": [{"value": 30, "unit": "days"}], "must_contain": []})
    return Finding(dimension="initial_waiting_period", applicable=True, statement="s", status=status, citations=[cit])


def test_citation_to_unretrieved_chunk_is_unsupported():
    s = _state("NOT_ADMISSIBLE", [_finding("SUPPORTS_EXCLUSION", chunk="never-retrieved")])
    rows = verify_state(s, {"c1": WAIT})
    assert rows[0].verdict == "UNSUPPORTED" and "never retrieved" in rows[0].reason


def test_decision_claims_follow_findings():
    supported = _finding("SUPPORTS_EXCLUSION")
    rows = verify_state(_state("NOT_ADMISSIBLE", [supported]), {"c1": WAIT})
    assert rows[-1].kind == "decision_claim" and rows[-1].verdict == "SUPPORTED"
    # calling an excluded claim admissible is contradicted by its own findings
    rows = verify_state(_state("ADMISSIBLE", [supported]), {"c1": WAIT})
    assert rows[-1].verdict == "CONTRADICTED"
    # a limits decision with no verified limit is unsupported
    rows = verify_state(_state("ADMISSIBLE_WITH_LIMITS", [_finding("SUPPORTS_ADMISSIBLE")]), {"c1": WAIT})
    assert rows[-1].verdict == "UNSUPPORTED"


def test_admissible_with_a_deduction_is_contradicted():
    lim = ApplicableLimit(description="d", dimension="category_sub_limits", deduction_inr=500.0)
    rows = verify_state(_state("ADMISSIBLE", [_finding("SUPPORTS_ADMISSIBLE")], [lim]), {"c1": WAIT})
    assert rows[-1].verdict == "CONTRADICTED"
