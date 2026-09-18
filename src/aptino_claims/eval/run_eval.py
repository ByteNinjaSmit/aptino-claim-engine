"""Reproducible end-to-end evaluation.

    python -m aptino_claims.eval.run_eval

Runs every supplied public case plus every custom case through the full
agent pipeline, scores decision accuracy against the hand-labeled expected
outcomes, checks that retrieval surfaced on-topic evidence for each case's
primary dimension, reports citation-validation ("citation correctness")
pass rate, and confirms the required NEEDS_REVIEW/abstention cases actually
abstain. Writes eval_results/metrics.json and eval_results/report.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..agents.orchestrator import analyze_case
from ..config import settings, REPO_ROOT
from ..llm.providers import get_llm_client
from ..retrieval.retriever import HybridRetriever
from .expected_outcomes import DIMENSION_EVIDENCE_KEYWORDS, EXPECTED, REQUIRED_ABSTENTIONS

PUBLIC_CASES_PATH = REPO_ROOT / "data" / "candidate_data" / "public_test_cases.json"
CUSTOM_CASES_PATH = REPO_ROOT / "data" / "custom_cases" / "custom_test_cases.json"
RESULTS_DIR = REPO_ROOT / "eval_results"


def _load_cases() -> list[dict]:
    public = json.loads(PUBLIC_CASES_PATH.read_text(encoding="utf-8"))
    custom = json.loads(CUSTOM_CASES_PATH.read_text(encoding="utf-8")) if CUSTOM_CASES_PATH.exists() else []
    return public + custom


def _evidence_hit(evidence_by_dimension: dict, dimension: str) -> bool:
    keywords = DIMENSION_EVIDENCE_KEYWORDS.get(dimension, [])
    if not keywords:
        return True
    texts = " ".join(ev["text"] for ev in evidence_by_dimension.get(dimension, []))
    return any(kw.lower() in texts.lower() for kw in keywords)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever.load_or_build(settings.index_dir, settings.index_dir / "chunks.jsonl")
    llm = get_llm_client()
    cases = _load_cases()

    per_case: list[dict] = []
    for raw_case in cases:
        case_id = raw_case["case_id"]
        started = time.perf_counter()
        state = analyze_case(raw_case, retriever, llm)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        response = state.to_response()

        expected = EXPECTED.get(case_id)
        decision_correct = expected is not None and response["decision"] == expected.decision

        evidence_by_dim = {k: [e.model_dump() for e in v] for k, v in state.evidence_by_dimension.items()}
        retrieval_hits = [
            _evidence_hit(evidence_by_dim, dim) for dim in (expected.primary_dimensions if expected else [])
        ]

        per_case.append({
            "case_id": case_id,
            "expected_decision": expected.decision if expected else None,
            "actual_decision": response["decision"],
            "decision_correct": decision_correct,
            "confidence": response["confidence"],
            "n_citations": len(response["citations"]),
            "n_applicable_limits": len(response["applicable_limits"]),
            "n_missing_evidence": len(response["missing_evidence"]),
            "validation_status": response["validation"]["status"],
            "unsupported_claims": response["validation"]["unsupported_claims"],
            "primary_dimensions": expected.primary_dimensions if expected else [],
            "retrieval_hit_rate": (sum(retrieval_hits) / len(retrieval_hits)) if retrieval_hits else None,
            "elapsed_ms": elapsed_ms,
        })

    n = len(per_case)
    decision_accuracy = sum(1 for c in per_case if c["decision_correct"]) / n
    validation_pass_rate = sum(1 for c in per_case if c["validation_status"] == "PASS") / n
    hit_rates = [c["retrieval_hit_rate"] for c in per_case if c["retrieval_hit_rate"] is not None]
    retrieval_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else None

    abstention_results = {
        case_id: next((c["actual_decision"] for c in per_case if c["case_id"] == case_id), None) == "NEEDS_REVIEW"
        for case_id in REQUIRED_ABSTENTIONS
    }
    abstention_pass = all(abstention_results.values())

    metrics = {
        "n_cases": n,
        "decision_accuracy": round(decision_accuracy, 3),
        "citation_validation_pass_rate": round(validation_pass_rate, 3),
        "retrieval_evidence_hit_rate": round(retrieval_hit_rate, 3) if retrieval_hit_rate is not None else None,
        "required_abstentions": abstention_results,
        "required_abstentions_pass": abstention_pass,
        "per_case": per_case,
    }

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(metrics)

    print(f"Decision accuracy: {decision_accuracy:.1%} ({sum(1 for c in per_case if c['decision_correct'])}/{n})")
    print(f"Citation validation pass rate: {validation_pass_rate:.1%}")
    print(f"Retrieval evidence hit rate: {retrieval_hit_rate:.1%}" if retrieval_hit_rate is not None else "Retrieval evidence hit rate: n/a")
    print(f"Required abstentions all correct: {abstention_pass} ({abstention_results})")
    print(f"Wrote {RESULTS_DIR / 'metrics.json'} and {RESULTS_DIR / 'report.md'}")

    if decision_accuracy < 1.0 or not abstention_pass:
        sys.exit(1)


def _write_report(metrics: dict) -> None:
    lines = [
        "# Evaluation Report",
        "",
        f"- Cases evaluated: {metrics['n_cases']}",
        f"- Decision accuracy: **{metrics['decision_accuracy']:.1%}**",
        f"- Citation validation pass rate: **{metrics['citation_validation_pass_rate']:.1%}**",
        f"- Retrieval evidence hit rate: **{metrics['retrieval_evidence_hit_rate']:.1%}**" if metrics["retrieval_evidence_hit_rate"] is not None else "- Retrieval evidence hit rate: n/a",
        f"- Required NEEDS_REVIEW cases all abstained: **{metrics['required_abstentions_pass']}**",
        "",
        "| Case | Expected | Actual | Correct | Confidence | Validation | Retrieval hit |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in metrics["per_case"]:
        hit = f"{c['retrieval_hit_rate']:.0%}" if c["retrieval_hit_rate"] is not None else "-"
        lines.append(
            f"| {c['case_id']} | {c['expected_decision']} | {c['actual_decision']} | "
            f"{'✅' if c['decision_correct'] else '❌'} | {c['confidence']:.2f} | {c['validation_status']} | {hit} |"
        )
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
