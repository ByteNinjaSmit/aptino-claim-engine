"""Custom state-machine orchestrator wiring the five agents together.

Case Analysis -> Policy Evidence -> Coverage & Exclusion -> Decision -> Validation

A plain Python function pipeline is used instead of a graph framework: the
flow here is a fixed sequence (no branching/looping beyond the single
validation-triggered downgrade handled inside ValidationAgent), so a state
machine library would add indirection without adding capability. Swapping
in LangGraph would only mean replacing this module.
"""
from __future__ import annotations

from ..llm.base import LLMClient
from ..retrieval.retriever import HybridRetriever
from . import case_analysis_agent, coverage_exclusion_agent, decision_agent, policy_evidence_agent, validation_agent
from .state import CaseState


def analyze_case(raw_case: dict, retriever: HybridRetriever, llm: LLMClient) -> CaseState:
    state = CaseState(case_id=str(raw_case.get("case_id", "UNKNOWN")), raw_case=raw_case)
    state = case_analysis_agent.run(state)
    state = policy_evidence_agent.run(state, retriever)
    state = coverage_exclusion_agent.run(state)
    state = decision_agent.run(state, llm)
    state = validation_agent.run(state)
    return state
