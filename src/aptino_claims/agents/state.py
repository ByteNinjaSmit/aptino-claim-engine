"""Pydantic models shared across agents and returned by the API.

Agents exchange `CaseState` (structured, typed) rather than free-form text;
`CaseState.to_response()` projects the final state onto the public decision
contract from the assignment brief.

Every material citation carries a machine-checkable `assertion` (what the
finding claims the cited chunk says). The Validation Agent verifies each
assertion against the retrieved chunk text and records a SUPPORTED /
UNSUPPORTED / CONTRADICTED verdict per claim.
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
    assertion: dict[str, Any] | None = None


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
    assumptions: list[str] = Field(default_factory=list)


class ApplicableLimit(BaseModel):
    description: str
    dimension: str
    deduction_inr: float | None = None
    citations: list[Citation] = Field(default_factory=list)


class MissingEvidence(BaseModel):
    field: str
    reason: str


Verdict = Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ClaimVerification(BaseModel):
    """One row of the decision-claim -> citation -> chunk -> verdict audit trail."""

    claim_id: str
    kind: Literal["policy_claim", "limit_claim", "decision_claim"]
    dimension: str
    claim: str
    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    verdict: Verdict
    reason: str
    checks: list[CheckResult] = Field(default_factory=list)
    excerpt: str | None = None


class ValidationOutcome(BaseModel):
    status: Literal["PASS", "FAIL"]
    unsupported_claims: list[str] = Field(default_factory=list)
    notes: str = ""
    verifications: list[ClaimVerification] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    attempts: int = 1


class TraceEvent(BaseModel):
    agent: str
    action: str
    detail: str = ""
    elapsed_ms: float = 0.0
    retrieval_count: int | None = None
    attempt: int = 1
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    snapshot: dict[str, Any] = Field(default_factory=dict)


class Handoff(BaseModel):
    """A visible state transition between two agents (including retries)."""

    from_agent: str
    to_agent: str
    payload: str
    kind: Literal["forward", "retry"] = "forward"


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
    rationale_source: str = "template"

    validation: ValidationOutcome | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    handoffs: list[Handoff] = Field(default_factory=list)
    attempt: int = 1
    widen_retrieval: bool = False

    # -- state bookkeeping --------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Compact counters describing what the shared state holds right now."""
        return {
            "facts": len(self.facts),
            "dimensions": len(self.dimensions),
            "evidence_chunks": sum(len(v) for v in self.evidence_by_dimension.values()),
            "findings": len(self.findings),
            "limits": len(self.applicable_limits),
            "missing": len(self.missing_fields),
            "decision": self.decision.value if self.decision else None,
            "validation": self.validation.status if self.validation else None,
        }

    def log(
        self,
        agent: str,
        action: str,
        detail: str = "",
        retrieval_count: int | None = None,
        started_at: float | None = None,
        reads: list[str] | None = None,
        writes: list[str] | None = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started_at) * 1000 if started_at is not None else 0.0
        self.trace.append(
            TraceEvent(
                agent=agent,
                action=action,
                detail=detail,
                elapsed_ms=round(elapsed_ms, 2),
                retrieval_count=retrieval_count,
                attempt=self.attempt,
                reads=reads or [],
                writes=writes or [],
                snapshot=self.snapshot(),
            )
        )

    def hand_off(self, from_agent: str, to_agent: str, payload: str, kind: Literal["forward", "retry"] = "forward") -> None:
        self.handoffs.append(Handoff(from_agent=from_agent, to_agent=to_agent, payload=payload, kind=kind))

    def reset_analysis(self) -> None:
        """Clear everything derived from evidence so a retry recomputes it cleanly."""
        self.evidence_by_dimension = {}
        self.findings = []
        self.applicable_limits = []
        self.missing_fields = [m for m in self.missing_fields if m.field.startswith("evidence_context.") or m.field in {
            "policy_start_date/claim_date", "sum_insured_inr"}]
        self.decision = None
        self.confidence = 0.0
        self.rationale = ""
        self.validation = None

    # -- projections ----------------------------------------------------------
    def all_citations(self) -> list[Citation]:
        seen: dict[tuple[str, str], Citation] = {}
        for finding in self.findings:
            for c in finding.citations:
                seen[(c.chunk_id, c.claim)] = c
        for limit in self.applicable_limits:
            for c in limit.citations:
                seen[(c.chunk_id, c.claim)] = c
        return list(seen.values())

    def _excerpt_for(self, chunk_id: str, limit: int = 600) -> str | None:
        for evidence in self.evidence_by_dimension.values():
            for item in evidence:
                if item.chunk_id == chunk_id:
                    text = item.text
                    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."
        return None

    def _amounts(self, decision: str) -> dict:
        expenses = {k: v for k, v in (self.raw_case.get("expenses_inr") or {}).items() if isinstance(v, (int, float))}
        claimed = float(sum(expenses.values()))
        deductions = float(sum((l.deduction_inr or 0) for l in self.applicable_limits))
        if decision == DecisionStatus.NEEDS_REVIEW.value:
            payable = None
        elif decision == DecisionStatus.NOT_ADMISSIBLE.value:
            payable = 0.0
        else:
            payable = max(0.0, claimed - deductions)
        return {
            "claimed_total_inr": claimed,
            "total_deductions_inr": deductions,
            "estimated_payable_inr": payable,
            "expense_breakdown_inr": expenses,
        }

    def to_response(self) -> dict:
        key_findings = [f.statement for f in self.findings if f.applicable]
        decision = self.decision.value if self.decision else DecisionStatus.NEEDS_REVIEW.value
        validation = self.validation.model_dump() if self.validation else {"status": "PASS", "unsupported_claims": []}
        return {
            "case_id": self.case_id,
            "decision": decision,
            "confidence": round(self.confidence, 2),
            "key_findings": key_findings,
            "findings": [
                {
                    "dimension": f.dimension,
                    "status": f.status,
                    "statement": f.statement,
                    "confidence": f.confidence,
                    "applicable": f.applicable,
                    "chunk_ids": [c.chunk_id for c in f.citations],
                    "assumptions": f.assumptions,
                }
                for f in self.findings
            ],
            "assumptions": [
                {"dimension": f.dimension, "assumption": a} for f in self.findings for a in f.assumptions
            ],
            "unmodelled_policy_risks": [
                {"statement": f.statement, "chunk_ids": [c.chunk_id for c in f.citations]}
                for f in self.findings
                if f.dimension == "unmodelled_policy_risk" and f.applicable
            ],
            "amounts": self._amounts(decision),
            "applicable_limits": [
                {"description": l.description, "deduction_inr": l.deduction_inr, "dimension": l.dimension}
                for l in self.applicable_limits
            ],
            "missing_evidence": [m.model_dump() for m in self.missing_fields],
            "citations": [{**c.model_dump(), "excerpt": self._excerpt_for(c.chunk_id)} for c in self.all_citations()],
            "validation": validation,
            "rationale": self.rationale,
            "rationale_source": self.rationale_source,
            "trace": [t.model_dump() for t in self.trace],
            "handoffs": [h.model_dump() for h in self.handoffs],
        }
