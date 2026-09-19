"""Negative controls: how good is the citation verifier at catching bad claims?

Measuring the verifier only on its own agent's (correct) output tells us it
does not raise false alarms; it tells us nothing about whether it would catch
a real error. So we inject known faults into otherwise-correct states and
count how many are caught:

    wrong_chunk        point a citation at a different *retrieved* chunk that
                       does not satisfy the claim
    wrong_number       change the threshold / percentage the claim asserts
    arithmetic_tamper  inflate a reported deduction so cap/deduction math breaks
    decision_flip      change the final decision to a different class

A control counts as "detected" when the affected claim's verdict is no longer
SUPPORTED.
"""
from __future__ import annotations

from ..agents.state import CaseState
from ..agents.verification import verify_assertion, verify_state

FLIP_TARGET = {
    "ADMISSIBLE": "NOT_ADMISSIBLE",
    "ADMISSIBLE_WITH_LIMITS": "ADMISSIBLE",
    "PARTIALLY_ADMISSIBLE": "ADMISSIBLE",
    "NOT_ADMISSIBLE": "ADMISSIBLE",
    "NEEDS_REVIEW": "ADMISSIBLE",
}


def _all_citations(state: CaseState):
    for f in state.findings:
        if f.applicable:
            for c in f.citations:
                yield f.dimension, c
    for lim in state.applicable_limits:
        for c in lim.citations:
            yield lim.dimension, c


def _verdicts(state: CaseState, corpus: dict[str, str], claim_key: tuple[str, str, str | None]):
    dimension, claim, _ = claim_key
    return [r.verdict for r in verify_state(state, corpus) if r.dimension == dimension and r.claim == claim]


def run_controls(states: list[CaseState], corpus: dict[str, str]) -> dict:
    results = {k: {"injected": 0, "detected": 0, "missed": []} for k in
               ("wrong_chunk", "wrong_number", "arithmetic_tamper", "decision_flip")}

    def record(kind: str, ok: bool, label: str) -> None:
        results[kind]["injected"] += 1
        if ok:
            results[kind]["detected"] += 1
        else:
            results[kind]["missed"].append(label)

    for state in states:
        retrieved = {ev.chunk_id: ev for evs in state.evidence_by_dimension.values() for ev in evs}

        # --- claim-level mutations -------------------------------------------
        for dimension, cit in list(_all_citations(state)):
            if not cit.assertion:
                continue
            label = f"{state.case_id}:{dimension}:{cit.claim[:40]}"
            kind = cit.assertion.get("type")

            # wrong chunk: swap in a retrieved chunk that genuinely fails the assertion
            for alt_id, ev in retrieved.items():
                if alt_id == cit.chunk_id:
                    continue
                if verify_assertion(cit.assertion, ev.text, corpus, alt_id)[0] == "SUPPORTED":
                    continue
                mutated = state.model_copy(deep=True)
                for _, mc in _all_citations(mutated):
                    if mc.chunk_id == cit.chunk_id and mc.claim == cit.claim:
                        mc.chunk_id, mc.section, mc.page = alt_id, ev.section, ev.page_start
                verdicts = [r.verdict for r in verify_state(mutated, corpus)
                            if r.dimension == dimension and r.claim == cit.claim and r.chunk_id == alt_id]
                record("wrong_chunk", bool(verdicts) and all(v != "SUPPORTED" for v in verdicts), label)
                break

            # wrong number
            if kind in ("threshold", "limit"):
                mutated = state.model_copy(deep=True)
                changed = False
                for _, mc in _all_citations(mutated):
                    if mc.chunk_id == cit.chunk_id and mc.claim == cit.claim and mc.assertion:
                        if kind == "threshold" and mc.assertion.get("values"):
                            mc.assertion["values"][0]["value"] += 17
                            changed = True
                        elif kind == "limit" and mc.assertion.get("percent") is not None:
                            mc.assertion["percent"] *= 2
                            changed = True
                if changed:
                    verdicts = [r.verdict for r in verify_state(mutated, corpus) if r.dimension == dimension and r.claim == cit.claim]
                    record("wrong_number", bool(verdicts) and all(v != "SUPPORTED" for v in verdicts), label)

            # arithmetic tamper
            if kind == "limit" and cit.assertion.get("deduction") is not None:
                mutated = state.model_copy(deep=True)
                for _, mc in _all_citations(mutated):
                    if mc.chunk_id == cit.chunk_id and mc.claim == cit.claim and mc.assertion:
                        mc.assertion["deduction"] = mc.assertion["deduction"] + 1000
                verdicts = [r.verdict for r in verify_state(mutated, corpus) if r.dimension == dimension and r.claim == cit.claim]
                record("arithmetic_tamper", bool(verdicts) and all(v != "SUPPORTED" for v in verdicts), label)

        # --- decision flip -----------------------------------------------------
        if state.decision is not None:
            mutated = state.model_copy(deep=True)
            original = state.decision.value
            mutated.decision = type(state.decision)(FLIP_TARGET[original])
            verdict = next(r.verdict for r in verify_state(mutated, corpus) if r.kind == "decision_claim")
            record("decision_flip", verdict != "SUPPORTED", f"{state.case_id}:{original}->{FLIP_TARGET[original]}")

    total_inj = sum(v["injected"] for v in results.values())
    total_det = sum(v["detected"] for v in results.values())
    for v in results.values():
        v["detection_rate"] = round(v["detected"] / v["injected"], 3) if v["injected"] else None
    return {"by_type": results, "injected": total_inj, "detected": total_det,
            "detection_rate": round(total_det / total_inj, 3) if total_inj else None}
