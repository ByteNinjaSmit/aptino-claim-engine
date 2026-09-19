import json

import pytest

from aptino_claims.agents import case_analysis_agent, decision_agent
from aptino_claims.agents import dimensions as dim
from aptino_claims.agents.state import ApplicableLimit, CaseState, DecisionStatus, EvidenceItem, Finding
from aptino_claims.config import settings

CHUNKS = [json.loads(l) for l in (settings.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
COVER = "\n".join(c["text"] for c in CHUNKS if c["chunk_id"].startswith("what_we_cover"))


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def generate_rationale(self, prompt):
        return self.text


def _limits_state():
    s = CaseState(case_id="T", raw_case={"expenses_inr": {"room": 30000}, "sum_insured_inr": 500000})
    s.facts = {"_explicit_unknowns": []}
    s.findings = [Finding(dimension="category_sub_limits", applicable=True, statement="Room exceeds the cap.",
                          status="SUPPORTS_LIMIT", confidence=0.8)]
    s.applicable_limits = [ApplicableLimit(description="Room sub-limit (INR 20,000)", dimension="category_sub_limits", deduction_inr=10200.0)]
    return s


def test_faithful_llm_rationale_is_accepted():
    s = _limits_state()
    text = "The claim is admissible with limits: room spend of 30,000 exceeds the 20,000 cap, reducing payment by 10,200."
    decision_agent.run(s, FakeLLM(text))
    assert s.decision == DecisionStatus.ADMISSIBLE_WITH_LIMITS
    assert s.rationale == text and s.rationale_source == "llm"


def test_llm_rationale_with_an_invented_figure_is_rejected():
    s = _limits_state()
    decision_agent.run(s, FakeLLM("The claim is admissible with limits, reducing payment by 12,345 for the room."))
    assert s.rationale_source.startswith("template (LLM text rejected")
    assert "12345" in s.rationale_source and "12,345" not in s.rationale


def test_llm_rationale_that_drops_the_decision_is_rejected():
    s = _limits_state()
    decision_agent.run(s, FakeLLM("Everything looks fine to me, payment reduced by 10,200."))
    assert s.rationale_source.startswith("template") and "does not name the decision" in s.rationale_source


def test_payable_is_capped_at_the_sum_insured():
    facts = {"treatment_type": "inpatient", "sum_insured": 100_000, "admission_hours": 96,
             "expenses": {"room": 4_000, "doctor_fees": 25_000, "medicines_diagnostics": 40_000,
                          "pre_hospitalization": 60_000, "post_hospitalization": 50_000}}
    evidence = [EvidenceItem(chunk_id="wc", section="What We Cover", label="x", page_start=8, page_end=8, text=COVER)]
    finding, limits, _ = dim._sublimits_eval(facts, evidence)
    overall = [l for l in limits if "Overall Sum Insured" in l.description]
    assert len(overall) == 1
    claimed = sum(facts["expenses"].values())
    assert claimed - sum(l.deduction_inr for l in limits) == 100_000     # payable == sum insured
    assert overall[0].citations[0].assertion == {"type": "phrases", "must_contain": ["overall sum insured"], "any_of": []}


def test_no_overall_cap_when_within_the_sum_insured():
    facts = {"treatment_type": "inpatient", "sum_insured": 500_000, "admission_hours": 96,
             "expenses": {"room": 4_000, "doctor_fees": 10_000, "medicines_diagnostics": 20_000}}
    evidence = [EvidenceItem(chunk_id="wc", section="What We Cover", label="x", page_start=8, page_end=8, text=COVER)]
    _, limits, _ = dim._sublimits_eval(facts, evidence)
    assert not any("Overall Sum Insured" in l.description for l in limits)


def test_irrelevant_attributes_are_detected_at_nested_levels_too():
    raw = {"case_id": "X", "policy_start_date": "2024-01-01", "claim_date": "2026-01-01", "favourite_colour": "blue",
           "patient": {"age": 30, "zodiac": "leo"}, "hospital": {"name": "H", "network_provider": True, "parking": True},
           "treatment": {"type": "inpatient", "diagnosis": "d", "procedure": "p", "mood": "ok"}}
    facts, _ = case_analysis_agent.build_facts(raw)
    assert facts["_unrecognized_fields"] == ["favourite_colour", "hospital.parking", "patient.zodiac", "treatment.mood"]


def test_internal_failure_returns_500_without_leaking_the_exception(monkeypatch):
    from fastapi.testclient import TestClient
    from aptino_claims.api import main

    def boom(*a, **k):
        raise RuntimeError("secret internal detail /etc/passwd")

    with TestClient(main.app) as client:
        monkeypatch.setattr(main, "analyze_case", boom)
        resp = client.post("/analyze", json={"case_id": "X", "policy_start_date": "2024-01-01", "claim_date": "2026-01-01",
                                             "treatment": {"type": "inpatient", "diagnosis": "d"}})
    assert resp.status_code == 500
    assert "secret internal detail" not in resp.text and "passwd" not in resp.text
