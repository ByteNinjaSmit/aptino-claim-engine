"""The validation-triggered retry: a needed clause ranks outside the first top-k."""
import json

from aptino_claims.agents.orchestrator import analyze_case
from aptino_claims.config import settings
from aptino_claims.llm.providers import OfflineLLMClient
from aptino_claims.retrieval.retriever import RetrievalResult
from aptino_claims.retrieval.sparse_store import SparseStore

CHUNKS = [json.loads(l) for l in (settings.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


class StubRetriever:
    """BM25-only retriever that buries the named short-stay list at rank 7.

    With the default top_k_final=5 the clause listing 'Eye Surgery' is not in
    the evidence, so the day-care citation cannot be verified; the widened
    retry (top_k_final=10) brings it back.
    """

    def __init__(self):
        self.chunk_meta = {c["chunk_id"]: c for c in CHUNKS}
        self._sparse = SparseStore([c["chunk_id"] for c in CHUNKS], [c["text"] for c in CHUNKS])

    def search(self, query, top_k_dense=None, top_k_sparse=None, top_k_fused=None, top_k_final=5):
        ranked = [cid for cid, _ in self._sparse.search(query, 40)]
        buried = next(c["chunk_id"] for c in CHUNKS if "eye surgery, lithotripsy" in " ".join(c["text"].lower().split()))
        if "day care" in query:
            ranked = [c for c in ranked if c != buried]
            ranked.insert(6, buried)
        out = []
        for i, cid in enumerate(ranked[:top_k_final], start=1):
            m = self.chunk_meta[cid]
            out.append(RetrievalResult(cid, m["section"], m["label"], m["page_start"], m["page_end"], m["text"],
                                       None, None, i, 1.0, 1.0, float(top_k_final - i)))
        return out


CATARACT_DAYCARE = {
    "case_id": "RETRY-1", "policy_start_date": "2024-01-01", "claim_date": "2026-06-01", "sum_insured_inr": 500000,
    "continuous_coverage_months": 29, "prior_insurer_continuous_years": 0, "patient": {"age": 60},
    "hospital": {"name": "Eye Hospital", "network_provider": True},
    "treatment": {"type": "day_care", "admission_hours": 8, "diagnosis": "Cataract", "procedure": "Eye surgery",
                  "pre_existing": False, "experimental": False},
    "expenses_inr": {"room": 0, "doctor_fees": 10000, "medicines_diagnostics": 20000},
}


def test_failed_verification_triggers_one_widened_retry_that_repairs_the_citation():
    state = analyze_case(CATARACT_DAYCARE, StubRetriever(), OfflineLLMClient())
    assert state.attempt == 2
    assert state.validation.status == "PASS"
    assert state.decision.value == "ADMISSIBLE"
    retries = [h for h in state.handoffs if h.kind == "retry"]
    assert len(retries) == 1 and retries[0].from_agent == "ValidationAgent" and retries[0].to_agent == "PolicyEvidenceAgent"
    assert "day_care_less_than_24h" in retries[0].payload
    assert [t.attempt for t in state.trace if t.agent == "PolicyEvidenceAgent"] == [1, 2]


def test_every_agent_records_reads_writes_and_state_snapshot():
    state = analyze_case(CATARACT_DAYCARE, StubRetriever(), OfflineLLMClient())
    for event in state.trace:
        assert event.reads and event.writes and "findings" in event.snapshot
    forward = [(h.from_agent, h.to_agent) for h in state.handoffs if h.kind == "forward" and h.to_agent != "Response"]
    assert forward[:4] == [("CaseAnalysisAgent", "PolicyEvidenceAgent"), ("PolicyEvidenceAgent", "CoverageExclusionAgent"),
                           ("CoverageExclusionAgent", "DecisionAgent"), ("DecisionAgent", "ValidationAgent")]
