"""Evidence verification: does the cited chunk actually say what the claim says?

Pipeline per material claim:

    Decision claim -> Citation -> Retrieved policy chunk -> Evidence check -> verdict

Verdicts:
    SUPPORTED     the cited chunk contains what the claim asserts
    UNSUPPORTED   the chunk does not contain it (or was never retrieved)
    CONTRADICTED  the chunk contains a *different* figure/fact than the claim
                  asserts, or the claim's own arithmetic does not add up

Each citation carries a machine-checkable `assertion` (see `dimensions.py`).
Supported assertion types:

    threshold  numbers with a unit (days/months/years) that must appear
    phrases    literal phrases that must all appear
    limit      "<pct>% of Sum Insured [or Rs <flat>]" bound to a keyword clause,
               plus the cap/deduction arithmetic derived from it
    absence    a term that must appear ONLY in the allowed chunks corpus-wide
               (backs claims of the form "the policy has no explicit clause on X")

No LLM is involved: verification is deterministic and reproducible.
"""
from __future__ import annotations

import math
import re

from . import rules_extract as rx
from .state import CaseState, CheckResult, ClaimVerification, Verdict

_UNIT_STEM = {"days": "day", "day": "day", "months": "month", "month": "month", "years": "year", "year": "year"}
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower())


def _numbers_with_unit(text: str, unit: str) -> set[int]:
    stem = _UNIT_STEM.get(unit, unit)
    return {int(n) for n in re.findall(rf"(\d+)\s*-?\s*{stem}s?\b", text, flags=re.I)}


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.5)


def _combine(results: list[tuple[Verdict, CheckResult]]) -> tuple[Verdict, list[CheckResult]]:
    checks = [c for _, c in results]
    verdicts = [v for v, _ in results]
    if "CONTRADICTED" in verdicts:
        return "CONTRADICTED", checks
    if "UNSUPPORTED" in verdicts:
        return "UNSUPPORTED", checks
    return "SUPPORTED", checks


# --------------------------------------------------------------------------
# assertion checkers: each returns (verdict, CheckResult)
# --------------------------------------------------------------------------

def _check_threshold(a: dict, text: str) -> list[tuple[Verdict, CheckResult]]:
    out: list[tuple[Verdict, CheckResult]] = []
    for item in a.get("values", []):
        want, unit = item["value"], item["unit"]
        found = _numbers_with_unit(text, unit)
        name = f"chunk states {want} {unit}"
        if want in found:
            out.append(("SUPPORTED", CheckResult(name=name, passed=True, detail=f"found {want} {unit} in cited text")))
        elif found:
            out.append(("CONTRADICTED", CheckResult(name=name, passed=False,
                                                    detail=f"claim says {want} {unit} but the chunk only states {sorted(found)} {unit}")))
        else:
            out.append(("UNSUPPORTED", CheckResult(name=name, passed=False, detail=f"no '{unit}' figure in the cited chunk")))
    for phrase in a.get("must_contain", []):
        ok = _norm(phrase) in _norm(text)
        out.append(("SUPPORTED" if ok else "UNSUPPORTED",
                    CheckResult(name=f"chunk contains '{phrase}'", passed=ok, detail="" if ok else "phrase not present")))
    return out


def _check_phrases(a: dict, text: str) -> list[tuple[Verdict, CheckResult]]:
    norm = _norm(text)
    out: list[tuple[Verdict, CheckResult]] = []
    for phrase in a.get("must_contain", []):
        ok = _norm(phrase) in norm
        out.append(("SUPPORTED" if ok else "UNSUPPORTED",
                    CheckResult(name=f"chunk contains '{phrase}'", passed=ok, detail="" if ok else "phrase not present")))
    for group in a.get("any_of", []):
        ok = any(_norm(p) in norm for p in group)
        out.append(("SUPPORTED" if ok else "UNSUPPORTED",
                    CheckResult(name=f"chunk contains one of {group}", passed=ok, detail="" if ok else "none present")))
    return out


def _check_limit(a: dict, text: str) -> list[tuple[Verdict, CheckResult]]:
    out: list[tuple[Verdict, CheckResult]] = []
    kw = a["keyword"]
    clause = rx.clause_with(text, kw)
    stated_pct = rx.extract_percent_of_si(text, kw)
    stated_flat = rx.extract_flat_amount(text, kw)

    if a.get("percent") is not None:
        name = f"clause for '{kw}' states {a['percent']}% of Sum Insured"
        if stated_pct is None:
            out.append(("UNSUPPORTED", CheckResult(name=name, passed=False, detail=f"no percentage limit found near '{kw}' in the chunk")))
        elif _close(stated_pct, a["percent"]):
            out.append(("SUPPORTED", CheckResult(name=name, passed=True, detail=f"chunk states {stated_pct}%")))
        else:
            out.append(("CONTRADICTED", CheckResult(name=name, passed=False, detail=f"chunk states {stated_pct}%, claim uses {a['percent']}%")))

    if a.get("flat") is not None:
        name = f"clause for '{kw}' states flat cap INR {a['flat']:,.0f}"
        if stated_flat is None:
            out.append(("UNSUPPORTED", CheckResult(name=name, passed=False, detail="no flat rupee cap found in the clause")))
        elif _close(stated_flat, a["flat"]):
            out.append(("SUPPORTED", CheckResult(name=name, passed=True, detail=f"chunk states INR {stated_flat:,.0f}")))
        else:
            out.append(("CONTRADICTED", CheckResult(name=name, passed=False, detail=f"chunk states INR {stated_flat:,.0f}, claim uses INR {a['flat']:,.0f}")))

    # arithmetic consistency of the derived cap and deduction
    if a.get("sum_insured") is not None and a.get("cap") is not None:
        candidates = []
        if a.get("percent") is not None:
            candidates.append(a["percent"] / 100 * a["sum_insured"] * a.get("multiplier", 1))
        if a.get("flat") is not None:
            candidates.append(a["flat"])
        expected_cap = min(candidates) if candidates else a["cap"]
        ok_cap = _close(expected_cap, a["cap"])
        out.append(("SUPPORTED" if ok_cap else "CONTRADICTED",
                    CheckResult(name="cap = limit x sum insured (x days)", passed=ok_cap,
                                detail=f"recomputed cap INR {expected_cap:,.0f} vs claimed INR {a['cap']:,.0f}")))
    if a.get("claimed") is not None and a.get("cap") is not None and a.get("deduction") is not None:
        expected_ded = max(0.0, a["claimed"] - a["cap"])
        ok_ded = _close(expected_ded, a["deduction"])
        out.append(("SUPPORTED" if ok_ded else "CONTRADICTED",
                    CheckResult(name="deduction = claimed - cap", passed=ok_ded,
                                detail=f"INR {a['claimed']:,.0f} - INR {a['cap']:,.0f} = INR {expected_ded:,.0f} vs reported INR {a['deduction']:,.0f}")))
    if clause is None and not out:
        out.append(("UNSUPPORTED", CheckResult(name="limit clause located", passed=False, detail=f"no clause mentions '{kw}'")))
    return out


def _check_absence(a: dict, text: str, corpus: dict[str, str], cited_chunk_id: str) -> list[tuple[Verdict, CheckResult]]:
    term = a["term"].lower()
    out: list[tuple[Verdict, CheckResult]] = []
    in_cited = term in text.lower()
    out.append(("SUPPORTED" if in_cited else "UNSUPPORTED",
                CheckResult(name=f"cited chunk mentions '{term}'", passed=in_cited, detail="" if in_cited else "term absent from the cited chunk")))
    allowed = set(a.get("allowed_chunk_ids", [])) | {cited_chunk_id}
    elsewhere = sorted(cid for cid, body in corpus.items() if term in body.lower() and cid not in allowed)
    out.append(("SUPPORTED" if not elsewhere else "CONTRADICTED",
                CheckResult(name=f"'{term}' appears nowhere else in the policy", passed=not elsewhere,
                            detail="scanned the whole policy: only the cited definition mentions it" if not elsewhere
                            else f"also mentioned in {elsewhere}, so 'no explicit clause' is not established")))
    return out


def verify_assertion(assertion: dict | None, text: str, corpus: dict[str, str], chunk_id: str) -> tuple[Verdict, str, list[CheckResult]]:
    if not assertion:
        return "UNSUPPORTED", "citation carries no checkable assertion", []
    kind = assertion.get("type")
    if kind == "threshold":
        results = _check_threshold(assertion, text)
    elif kind == "phrases":
        results = _check_phrases(assertion, text)
    elif kind == "limit":
        results = _check_limit(assertion, text)
    elif kind == "absence":
        results = _check_absence(assertion, text, corpus, chunk_id)
    elif kind == "quote":
        ok = _norm(assertion.get("quote", "")) in _norm(text) and bool(assertion.get("quote", "").strip())
        results = [("SUPPORTED" if ok else "UNSUPPORTED",
                    CheckResult(name="quoted text appears verbatim in the cited chunk", passed=ok,
                                detail="" if ok else "the quote is not in the cited chunk"))]
    else:
        return "UNSUPPORTED", f"unknown assertion type {kind!r}", []
    if not results:
        return "UNSUPPORTED", "assertion produced no checks", []
    verdict, checks = _combine(results)
    failed = [c.detail or c.name for c in checks if not c.passed]
    reason = "all evidence checks passed" if verdict == "SUPPORTED" else "; ".join(failed)
    return verdict, reason, checks


def _excerpt(text: str, assertion: dict | None, limit: int = 260) -> str:
    keyword = (assertion or {}).get("keyword")
    if keyword:
        clause = rx.clause_with(text, keyword)
        if clause:
            return clause.strip()[:limit]
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


# --------------------------------------------------------------------------
# state-level verification
# --------------------------------------------------------------------------

def verify_state(state: CaseState, corpus: dict[str, str]) -> list[ClaimVerification]:
    retrieved: dict[str, tuple[str, str, int]] = {}
    for evidence in state.evidence_by_dimension.values():
        for ev in evidence:
            retrieved[ev.chunk_id] = (ev.text, ev.section, ev.page_start)

    rows: list[ClaimVerification] = []
    n = 0

    def add(kind: str, dimension: str, citation, statement: str) -> None:
        nonlocal n
        n += 1
        cid = citation.chunk_id
        if cid not in retrieved:
            rows.append(ClaimVerification(
                claim_id=f"C{n}", kind=kind, dimension=dimension, claim=citation.claim, chunk_id=cid,
                page=citation.page, section=citation.section, verdict="UNSUPPORTED",
                reason="cited chunk was never retrieved for this case", checks=[
                    CheckResult(name="cited chunk was retrieved", passed=False, detail="citation points outside the retrieved evidence")]))
            return
        text, section, page = retrieved[cid]
        verdict, reason, checks = verify_assertion(citation.assertion, text, corpus, cid)
        checks.insert(0, CheckResult(name="cited chunk was retrieved", passed=True, detail=f"{section}, page {page}"))
        rows.append(ClaimVerification(
            claim_id=f"C{n}", kind=kind, dimension=dimension, claim=citation.claim, chunk_id=cid,
            page=page, section=section, verdict=verdict, reason=reason, checks=checks,
            excerpt=_excerpt(text, citation.assertion)))

    for finding in state.findings:
        if not finding.applicable:
            continue
        for citation in finding.citations:
            add("policy_claim", finding.dimension, citation, finding.statement)
    for limit in state.applicable_limits:
        for citation in limit.citations:
            add("limit_claim", limit.dimension, citation, limit.description)

    rows.extend(_verify_decision(state, rows, start=n))
    return rows


def _verify_decision(state: CaseState, rows: list[ClaimVerification], start: int) -> list[ClaimVerification]:
    """Does the final decision follow from findings whose evidence checks passed?"""
    decision = state.decision.value if state.decision else None
    supported_dims = {r.dimension for r in rows if r.verdict == "SUPPORTED" and r.kind == "policy_claim"}
    supported_limit_dims = {r.dimension for r in rows if r.verdict == "SUPPORTED" and r.kind == "limit_claim"}
    exclusions = [f for f in state.findings if f.status == "SUPPORTS_EXCLUSION"]
    insufficient = [f for f in state.findings if f.status == "INSUFFICIENT_EVIDENCE"]
    checks: list[CheckResult] = []
    verdict: Verdict = "SUPPORTED"
    reason = "decision follows from verified findings"

    if decision == "NOT_ADMISSIBLE":
        ok = any(f.dimension in supported_dims for f in exclusions)
        checks.append(CheckResult(name="a verified exclusion finding backs the rejection", passed=ok))
        if not ok:
            verdict, reason = "UNSUPPORTED", "rejection is not backed by a verified exclusion citation"
    elif decision in ("ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE"):
        ok = bool(supported_limit_dims)
        checks.append(CheckResult(name="a verified limit/deduction backs the reduced payout", passed=ok))
        if not ok:
            verdict, reason = "UNSUPPORTED", "limit-based decision has no verified limit citation"
    elif decision == "ADMISSIBLE":
        deductions = [l for l in state.applicable_limits if (l.deduction_inr or 0) > 0]
        unknowns = bool(state.facts.get("_explicit_unknowns"))
        checks.append(CheckResult(name="no exclusion finding contradicts admissibility", passed=not exclusions))
        checks.append(CheckResult(name="no unresolved finding or open question remains", passed=not insufficient and not unknowns))
        checks.append(CheckResult(name="no limit reduces the payable amount", passed=not deductions))
        if exclusions or insufficient or unknowns or deductions:
            verdict, reason = "CONTRADICTED", "an exclusion, unresolved question or limit exists but the claim was called fully admissible"
    elif decision == "NEEDS_REVIEW":
        has_reason = bool(insufficient) or bool(state.facts.get("_explicit_unknowns")) or any(
            m.field == "policy_start_date/claim_date" for m in state.missing_fields)
        checks.append(CheckResult(name="a stated reason justifies abstaining", passed=has_reason))
        if not has_reason:
            verdict, reason = "UNSUPPORTED", "abstained without a recorded unresolved finding"

    return [ClaimVerification(
        claim_id=f"C{start + 1}", kind="decision_claim", dimension="decision", claim=f"Decision is {decision}",
        verdict=verdict, reason=reason, checks=checks)]
