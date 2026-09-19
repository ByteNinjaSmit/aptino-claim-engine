"""Custom state-machine orchestrator wiring the five agents together.

    Case Analysis -> Policy Evidence -> Coverage & Exclusion -> Decision -> Validation
                            ^                                                   |
                            +---------- retry (widened retrieval, once) --------+

Every hand-off is recorded on the state (`state.handoffs`) and every agent
records what it read, what it wrote and a snapshot of the shared state
(`state.trace`), so the workflow is inspectable rather than implied.

The only loop is the validation-triggered retry: if evidence verification
finds an UNSUPPORTED/CONTRADICTED claim, the case is re-run once with
retrieval widened. A plain function pipeline is used instead of a graph
framework because the flow is otherwise a fixed sequence; swapping in
LangGraph would only mean replacing this module.
"""
from __future__ import annotations

from ..llm.base import LLMClient
from ..retrieval.retriever import HybridRetriever
from . import case_analysis_agent, coverage_exclusion_agent, decision_agent, policy_evidence_agent, validation_agent
from .state import CaseState

MAX_ATTEMPTS = 2


def analyze_case(raw_case: dict, retriever: HybridRetriever, llm: LLMClient, interpret: bool | None = None) -> CaseState:
    state = CaseState(case_id=str(raw_case.get("case_id", "UNKNOWN")), raw_case=raw_case)
    state = case_analysis_agent.run(state)

    while True:
        state = policy_evidence_agent.run(state, retriever)
        state = coverage_exclusion_agent.run(state, llm, interpret, retriever)
        state = decision_agent.run(state, llm)
        last_chance = state.attempt >= MAX_ATTEMPTS
        state = validation_agent.run(state, retriever, final_attempt=last_chance)

        if state.validation.status == "PASS" or last_chance:
            state.hand_off("ValidationAgent", "Response", f"validation {state.validation.status}; decision {state.decision.value}")
            return state

        failing = sorted({r.dimension for r in state.validation.verifications if r.verdict != "SUPPORTED"})
        state.hand_off(
            "ValidationAgent", "PolicyEvidenceAgent",
            f"retry {state.attempt}: {len(state.validation.unsupported_claims)} claim(s) failed verification ({', '.join(failing)}); widening retrieval",
            kind="retry",
        )
        state.attempt += 1
        state.widen_retrieval = True
        state.reset_analysis()
