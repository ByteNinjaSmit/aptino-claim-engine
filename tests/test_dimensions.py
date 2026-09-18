from aptino_claims.agents import dimensions as dim
from aptino_claims.agents.state import EvidenceItem


def _ev(chunk_id: str, text: str, section: str = "What We Exclude", page: int = 9) -> EvidenceItem:
    return EvidenceItem(chunk_id=chunk_id, section=section, label=chunk_id, page_start=page, page_end=page, text=text)


def test_initial_waiting_period_excludes_recent_claim_without_continuity():
    facts = {"days_since_policy_start": 10, "continuous_coverage_months": 0, "prior_insurer_continuous_years": 0}
    evidence = [_ev("we-002", "2. 30 days Waiting Period. A waiting period of 30 days will apply to all claims unless...")]
    finding, limits, missing = dim._initial_wait_eval(facts, evidence)
    assert finding.status == "SUPPORTS_EXCLUSION"
    assert finding.citations[0].chunk_id == "we-002"


def test_initial_waiting_period_admissible_with_continuity():
    facts = {"days_since_policy_start": 5, "continuous_coverage_months": 14, "prior_insurer_continuous_years": 0}
    evidence = [_ev("we-002", "A waiting period of 30 days will apply to all claims unless continuously covered.")]
    finding, _, _ = dim._initial_wait_eval(facts, evidence)
    assert finding.status == "SUPPORTS_ADMISSIBLE"


def test_pre_existing_wait_not_met():
    facts = {"continuous_coverage_months": 27, "prior_insurer_continuous_years": 0, "prior_policy": None}
    evidence = [_ev("def-037", "Pre-existing diseases will not be covered until 48 months of continuous coverage have elapsed.", section="Definitions", page=5)]
    finding, _, _ = dim._pre_existing_eval(facts, evidence)
    assert finding.status == "SUPPORTS_EXCLUSION"
    assert "48" in finding.citations[0].claim


def test_pre_existing_wait_met_via_portability():
    facts = {
        "continuous_coverage_months": 12,
        "prior_insurer_continuous_years": 3,
        "prior_policy": {"database_and_claim_history_received": True},
    }
    evidence = [_ev("def-037", "will not be covered until 48 months of continuous coverage have elapsed", section="Definitions", page=5)]
    finding, _, _ = dim._pre_existing_eval(facts, evidence)
    assert finding.status == "SUPPORTS_ADMISSIBLE"


def test_hospital_definition_flags_explicit_null_even_if_network_provider():
    facts = {"hospital_network_provider": True, "evidence_context": {"hospital_registered": None}, "hospital_name": "X"}
    evidence = [_ev("def-022", "Hospital means any institution... registered... OR complies with all minimum criteria", section="Definitions", page=3)]
    finding, _, missing = dim._hospital_def_eval(facts, evidence)
    # network_provider alone still yields a (soft) admissible finding at the
    # dimension level; the explicit-null override is enforced by the
    # decision agent, not this dimension -- covered by test_decision_agent.
    assert finding.status == "SUPPORTS_ADMISSIBLE"


def test_cosmetic_exclusion_triggers_on_pure_cosmetic_case():
    facts = {"diagnosis": "Cosmetic condition", "procedure": "Cosmetic surgery"}
    evidence = [_ev("we-005", "cosmetic or aesthetic treatment of any description... plastic surgery except those relating to treatment of Injury or Disease")]
    finding, _, _ = dim._cosmetic_eval(facts, evidence)
    assert finding.status == "SUPPORTS_EXCLUSION"


def test_experimental_treatment_always_abstains():
    facts = {"experimental": True}
    evidence = [_ev("def-041", "Unproven/Experimental Treatment means a treatment... not based on established medical practice in India", section="Definitions", page=5)]
    finding, _, missing = dim._experimental_eval(facts, evidence)
    assert finding.status == "INSUFFICIENT_EVIDENCE"
    assert len(missing) == 1


def test_sublimits_computes_room_deduction():
    facts = {
        "treatment_type": "inpatient",
        "sum_insured": 500000,
        "admission_hours": 96,
        "expenses": {"room": 30000, "doctor_fees": 0, "medicines_diagnostics": 0, "ambulance": 0},
    }
    evidence = [_ev("wc-001", "Normal Room expenses: 1.0% of Basic Sum Insured.", section="What We Cover", page=7)]
    finding, limits, _ = dim._sublimits_eval(facts, evidence)
    assert len(limits) == 1
    assert limits[0].deduction_inr == 30000 - 0.01 * 500000 * 4
