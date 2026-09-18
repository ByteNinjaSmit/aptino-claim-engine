from aptino_claims.agents import decision_agent
from aptino_claims.agents.state import CaseState, DecisionStatus, Finding
from aptino_claims.llm.providers import OfflineLLMClient


def _state_with_findings(findings, explicit_unknowns=None) -> CaseState:
    state = CaseState(case_id="T-1", raw_case={})
    state.findings = findings
    state.facts = {"_explicit_unknowns": explicit_unknowns or []}
    return state


def test_explicit_null_evidence_forces_needs_review_even_if_no_finding_is_insufficient():
    findings = [
        Finding(dimension="hospital_definition", applicable=True, statement="ok", status="SUPPORTS_ADMISSIBLE", confidence=0.8),
    ]
    state = _state_with_findings(findings, explicit_unknowns=["hospital_registered"])
    decision_agent.run(state, OfflineLLMClient())
    assert state.decision == DecisionStatus.NEEDS_REVIEW


def test_insufficient_finding_forces_needs_review():
    findings = [Finding(dimension="x", applicable=True, statement="?", status="INSUFFICIENT_EVIDENCE", confidence=0.2)]
    state = _state_with_findings(findings)
    decision_agent.run(state, OfflineLLMClient())
    assert state.decision == DecisionStatus.NEEDS_REVIEW


def test_whole_claim_exclusion_yields_not_admissible():
    findings = [Finding(dimension="cosmetic_exclusion", applicable=True, statement="excluded", status="SUPPORTS_EXCLUSION", confidence=0.85)]
    state = _state_with_findings(findings)
    decision_agent.run(state, OfflineLLMClient())
    assert state.decision == DecisionStatus.NOT_ADMISSIBLE


def test_clean_admissible_case_with_no_limits():
    findings = [Finding(dimension="initial_waiting_period", applicable=True, statement="ok", status="SUPPORTS_ADMISSIBLE", confidence=0.9)]
    state = _state_with_findings(findings)
    decision_agent.run(state, OfflineLLMClient())
    assert state.decision == DecisionStatus.ADMISSIBLE
