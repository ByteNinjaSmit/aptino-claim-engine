"""Request/response models for the public API.

`extra="allow"` on every nested model is intentional: the schema note
supplied with the assignment says the system "should tolerate
unknown/non-critical fields" -- so an attribute the policy doesn't care
about (see the Case Analysis Agent's `_unrecognized_fields` tracking)
passes through validation instead of causing a 422.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int | None = None


class HospitalIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    network_provider: bool = False


class TreatmentIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "inpatient"
    admission_hours: float = 24
    diagnosis: str = ""
    procedure: str = ""
    pre_existing: bool = False
    experimental: bool = False
    hospital_room_unavailable: bool | None = None
    patient_cannot_be_moved: bool | None = None


class ClaimCaseIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    policy_id: str | None = None
    policy_start_date: str
    claim_date: str
    sum_insured_inr: float = 0
    continuous_coverage_months: int = 0
    prior_insurer_continuous_years: int = 0
    patient: PatientIn = Field(default_factory=PatientIn)
    hospital: HospitalIn = Field(default_factory=HospitalIn)
    treatment: TreatmentIn
    expenses_inr: dict = Field(default_factory=dict)
    documents: list[str] = Field(default_factory=list)
    task: str | None = None
    evidence_context: dict | None = None
    expense_timing: dict | None = None
    prior_policy: dict | None = None


class HealthOut(BaseModel):
    status: str
    chunk_count: int
    llm_provider: str
