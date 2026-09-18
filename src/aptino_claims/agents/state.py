"""Pydantic models shared across agents and returned by the API.

Agents exchange `CaseState` (structured, typed) rather than free-form text;
`CaseState.to_response()` projects the final state onto the public decision
contract from the assignment brief.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Citation(BaseModel):
    claim: str
    source: str = "policy.pdf"
    page: int
    section: str
    chunk_id: str
    rerank_score: float | None = None


class EvidenceItem(BaseModel):
    chunk_id: str
    section: str
    label: str
    page_start: int
    page_end: int
    text: str
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class Finding(BaseModel):
    dimension: str
    applicable: bool
    statement: str
    status: Literal["SUPPORTS_ADMISSIBLE", "SUPPORTS_LIMIT", "SUPPORTS_EXCLUSION", "INSUFFICIENT_EVIDENCE", "NOT_APPLICABLE"]
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.5


class ApplicableLimit(BaseModel):
    description: str
    dimension: str
    deduction_inr: float | None = None
    citations: list[Citation] = Field(default_factory=list)


class MissingEvidence(BaseModel):
    field: str
    reason: str


class ValidationOutcome(BaseModel):
    status: Literal["PASS", "FAIL"]
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = ""


class TraceEvent(BaseModel):
    agent: str
    action: str
    detail: str = ""
    elapsed_ms: float = 0.0
    retrieval_count: int | None = None


class CaseState(BaseModel):
    """Structured state threaded through the agent pipeline."""

    case_id: str
    raw_case: dict[str, Any]
    facts: dict[str, Any] = Field(default_factory=dict)
    dimensions: list[str] = Field(default_factory=list)
    investigation_checklist: list[str] = Field(default_factory=list)
    missing_fields: list[MissingEvidence] = Field(default_factory=list)

    evidence_by_dimension: dict[str, list[EvidenceItem]] = Field(default_factory=dict)

    findings: list[Finding] = Field(default_factory=list)
    applicable_limits: list[ApplicableLimit] = Field(default_factory=list)

    decision: DecisionStatus | None = None
    confidence: float = 0.0
    rationale: str = ""

    validation: ValidationOutcome | None = None
    trace: list[TraceEvent] = Field(default_factory=list)

    def log(self, agent: str, action: str, detail: str = "", retrieval_count: int | None = None, started_at: float | None = None) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000 if started_at is not None else 0.0
        self.trace.append(
            TraceEvent(agent=agent, action=action, detail=detail, elapsed_ms=round(elapsed_ms, 2), retrieval_count=retrieval_count)
        )

    def all_citations(self) -> list[Citation]:
        seen: dict[tuple[str, str], Citation] = {}
        for finding in self.findings:
            for c in finding.citations:
                seen[(c.chunk_id, c.claim)] = c
        for limit in self.applicable_limits:
            for c in limit.citations:
                seen[(c.chunk_id, c.claim)] = c
        return list(seen.values())

    def to_response(self) -> dict:
        key_findings = [f.statement for f in self.findings if f.applicable]
        return {
            "case_id": self.case_id,
            "decision": self.decision.value if self.decision else DecisionStatus.NEEDS_REVIEW.value,
            "confidence": round(self.confidence, 2),
            "key_findings": key_findings,
            "applicable_limits": [
                {"description": l.description, "deduction_inr": l.deduction_inr, "dimension": l.dimension}
                for l in self.applicable_limits
            ],
            "missing_evidence": [m.model_dump() for m in self.missing_fields],
            "citations": [c.model_dump() for c in self.all_citations()],
            "validation": self.validation.model_dump() if self.validation else {"status": "PASS", "unsupported_claims": []},
            "rationale": self.rationale,
            "trace": [t.model_dump() for t in self.trace],
        }
