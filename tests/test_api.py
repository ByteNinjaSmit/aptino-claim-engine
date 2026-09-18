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
