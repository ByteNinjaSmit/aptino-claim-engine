import pytest
from fastapi.testclient import TestClient

from aptino_claims.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] > 0


def test_analyze_valid_case(client):
    case = {
        "case_id": "TEST-001",
        "policy_id": "USGIC-CSC-2017-2018",
        "policy_start_date": "2024-01-01",
        "claim_date": "2026-01-01",
        "sum_insured_inr": 500000,
        "continuous_coverage_months": 24,
        "prior_insurer_continuous_years": 0,
        "patient": {"age": 30},
        "hospital": {"name": "Test Hospital", "network_provider": True},
        "treatment": {
            "type": "inpatient", "admission_hours": 96, "diagnosis": "Acute appendicitis",
            "procedure": "Appendectomy", "pre_existing": False, "experimental": False,
        },
        "expenses_inr": {"room": 10000, "doctor_fees": 10000, "medicines_diagnostics": 20000, "pre_hospitalization": 0, "post_hospitalization": 0, "ambulance": 0},
        "documents": ["claim_form"],
    }
    resp = client.post("/analyze", json=case)
    assert resp.status_code == 200
    body = resp.json()
    assert body["case_id"] == "TEST-001"
    assert body["decision"] in {"ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE", "NEEDS_REVIEW"}
    assert isinstance(body["citations"], list)
    assert "trace" in body


def test_analyze_rejects_malformed_case(client):
    resp = client.post("/analyze", json={"case_id": "BAD-1"})  # missing required fields
    assert resp.status_code == 422


def test_analyze_tolerates_unknown_fields(client):
    case = {
        "case_id": "TEST-002",
        "policy_start_date": "2024-01-01",
        "claim_date": "2026-01-01",
        "treatment": {"type": "inpatient", "diagnosis": "Fever"},
        "some_irrelevant_field": "should not break validation",
    }
    resp = client.post("/analyze", json=case)
    assert resp.status_code == 200


def test_llm_interpretation_is_opt_in_per_request(client, monkeypatch):
    import json
    from aptino_claims.api import main

    calls = []

    class SpyLLM:
        def interpret(self, system, user):
            calls.append(1)
            return json.dumps({"observations": []})

        def generate_rationale(self, prompt):
            return None

    case = {"case_id": "OPT-1", "policy_start_date": "2024-01-01", "claim_date": "2026-01-01", "sum_insured_inr": 500000,
            "continuous_coverage_months": 24, "hospital": {"name": "H", "network_provider": True},
            "treatment": {"type": "inpatient", "admission_hours": 96, "diagnosis": "Acute appendicitis", "procedure": "Appendectomy"},
            "expenses_inr": {"room": 10000, "doctor_fees": 10000, "medicines_diagnostics": 20000}}
    monkeypatch.setitem(main._state, "llm", SpyLLM())
    off = client.post("/analyze", json=case).json()
    assert calls == [] and off["llm_interpretation"] == {"enabled": False}
    on = client.post("/analyze?llm_interpretation=true", json=case).json()
    assert calls == [1] and on["llm_interpretation"]["enabled"] and on["llm_interpretation"]["accepted"] == []


def test_explicit_false_turns_interpretation_off_even_when_the_server_default_is_on(client, monkeypatch):
    import json
    from types import SimpleNamespace
    from aptino_claims.agents import coverage_exclusion_agent
    from aptino_claims.api import main

    calls = []

    class SpyLLM:
        def interpret(self, system, user):
            calls.append(1)
            return json.dumps({"observations": []})

        def generate_rationale(self, prompt):
            return None

    case = {"case_id": "OPT-2", "policy_start_date": "2024-01-01", "claim_date": "2026-01-01", "sum_insured_inr": 500000,
            "continuous_coverage_months": 24, "hospital": {"name": "H", "network_provider": True},
            "treatment": {"type": "inpatient", "admission_hours": 96, "diagnosis": "Acute appendicitis", "procedure": "Appendectomy"},
            "expenses_inr": {"room": 10000, "doctor_fees": 10000, "medicines_diagnostics": 20000}}
    monkeypatch.setitem(main._state, "llm", SpyLLM())
    monkeypatch.setattr(coverage_exclusion_agent, "settings", SimpleNamespace(llm_interpretation=True))
    client.post("/analyze?llm_interpretation=false", json=case)
    assert calls == []
    client.post("/analyze", json=case)                      # omitted -> follows the (patched) server setting
    assert calls == [1]
    assert client.post("/analyze?llm_interpretation=maybe", json=case).status_code == 422
