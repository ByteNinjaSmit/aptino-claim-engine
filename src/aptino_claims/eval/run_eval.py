"""Reproducible end-to-end evaluation.

    python -m aptino_claims.eval.run_eval

Measures, over every supplied public case plus every custom case:

  * decision quality   accuracy, confusion matrix, per-class precision/recall/F1,
                       unsafe-decision rate, abstention precision/recall
  * payable amounts    hand-derived expected deductions vs computed deductions
  * retrieval quality  recall@k and MRR against phrase-defined gold evidence, with an
                       ablation: dense only / BM25 only / RRF fusion / fusion + rerank
  * citation quality   gold-citation precision, coverage, and the verifier's
                       SUPPORTED / UNSUPPORTED / CONTRADICTED counts
  * verifier power     negative controls: inject known faults, measure detection
  * reproducibility    two independent runs (cache cleared) must be byte-identical
  * latency            per-case p50 / p95

Writes eval_results/metrics.json and eval_results/report.md. The LLM is
deliberately forced offline: it only rephrases the rationale and never
influences a decision, so the evaluation does not depend on an API key.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time

from ..agents.dimensions import DIMENSIONS
from ..agents.orchestrator import analyze_case
from ..agents.state import CaseState
from ..config import REPO_ROOT, settings
from ..llm.providers import OfflineLLMClient
from ..retrieval.retriever import HybridRetriever
from .controls import run_controls
from .expected_outcomes import DECISION_CLASSES, EXPECTED, REQUIRED_ABSTENTIONS
from .gold import gold_chunks, requirements_for
from .metrics import confusion_matrix, mean, per_class_prf, percentile, recall_at_k, reciprocal_rank

PUBLIC_CASES_PATH = REPO_ROOT / "data" / "candidate_data" / "public_test_cases.json"
CUSTOM_CASES_PATH = REPO_ROOT / "data" / "custom_cases" / "custom_test_cases.json"
RESULTS_DIR = REPO_ROOT / "eval_results"
DECISIVE = {"ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE"}
KS = (1, 3, 5)


def _load_cases() -> list[dict]:
    public = json.loads(PUBLIC_CASES_PATH.read_text(encoding="utf-8"))
    custom = json.loads(CUSTOM_CASES_PATH.read_text(encoding="utf-8")) if CUSTOM_CASES_PATH.exists() else []
    return public + custom


def _run_all(cases: list[dict], retriever: HybridRetriever) -> tuple[list[CaseState], list[dict], list[float]]:
    llm = OfflineLLMClient()
    states, responses, latencies = [], [], []
    for raw in cases:
        t0 = time.perf_counter()
        state = analyze_case(raw, retriever, llm)
        latencies.append((time.perf_counter() - t0) * 1000)
        states.append(state)
        responses.append(state.to_response())
    return states, responses, latencies


def _canonical(response: dict) -> str:
    r = {k: v for k, v in response.items() if k not in ("trace",)}
    return json.dumps(r, sort_keys=True, default=str)


# --------------------------------------------------------------------------
# metric blocks
# --------------------------------------------------------------------------

def decision_block(cases, responses) -> dict:
    pairs = [(EXPECTED[c["case_id"]].decision, r["decision"]) for c, r in zip(cases, responses)]
    correct = sum(1 for e, p in pairs if e == p)
    unsafe = [c["case_id"] for c, (e, p) in zip(cases, pairs) if p in DECISIVE and p != e]
    abstain_tp = sum(1 for e, p in pairs if e == "NEEDS_REVIEW" and p == "NEEDS_REVIEW")
    abstain_pred = sum(1 for _, p in pairs if p == "NEEDS_REVIEW")
    abstain_true = sum(1 for e, _ in pairs if e == "NEEDS_REVIEW")
    return {
        "accuracy": round(correct / len(pairs), 3),
        "correct": correct, "n": len(pairs),
        "confusion_matrix": confusion_matrix(pairs, DECISION_CLASSES),
        "per_class": per_class_prf(pairs, DECISION_CLASSES),
        "unsafe_decisions": unsafe,
        "unsafe_decision_rate": round(len(unsafe) / len(pairs), 3),
        "abstention_precision": round(abstain_tp / abstain_pred, 3) if abstain_pred else None,
        "abstention_recall": round(abstain_tp / abstain_true, 3) if abstain_true else None,
    }


def amounts_block(cases, responses) -> dict:
    rows, ok = [], 0
    for c, r in zip(cases, responses):
        exp = EXPECTED[c["case_id"]]
        if exp.expected_deduction_inr is None or exp.decision not in ("ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE"):
            continue
        actual = r["amounts"]["total_deductions_inr"]
        match = abs(actual - exp.expected_deduction_inr) <= 0.5 and r["decision"] == exp.decision
        ok += match
        rows.append({"case_id": c["case_id"], "expected_deduction_inr": exp.expected_deduction_inr,
                     "actual_deduction_inr": actual, "match": match})
    return {"exact_match_rate": round(ok / len(rows), 3) if rows else None, "n": len(rows), "rows": rows}


def retrieval_block(states: list[CaseState], retriever: HybridRetriever, corpus: dict[str, str]) -> dict:
    by_key = {d.key: d for d in DIMENSIONS}
    systems = ("dense", "sparse", "fused", "reranked")
    acc = {s: {**{f"recall@{k}": [] for k in KS}, "mrr": []} for s in systems}
    by_dim: dict[str, list[float]] = {}
    pairs = 0

    for state in states:
        for dim_key in state.dimensions:
            reqs = requirements_for(dim_key, state.facts)
            if not reqs:
                continue
            gold = gold_chunks(reqs, corpus)
            stages = retriever.search_stages(by_key[dim_key].query(state.facts))
            pairs += 1
            for s in systems:
                ranked = [cid for cid, _ in stages[s]]
                for k in KS:
                    acc[s][f"recall@{k}"].append(recall_at_k(ranked, gold, k))
                acc[s]["mrr"].append(reciprocal_rank(ranked, gold))
            by_dim.setdefault(dim_key, []).append(recall_at_k([c for c, _ in stages["reranked"]], gold, 5))

    table = {s: {m: round(mean(v), 3) for m, v in acc[s].items()} for s in systems}
    return {
        "n_case_dimension_pairs": pairs,
        "systems": table,
        "hybrid_rerank_recall@5_by_dimension": {d: round(mean(v), 3) for d, v in sorted(by_dim.items())},
    }


def citation_block(states: list[CaseState], corpus: dict[str, str]) -> dict:
    total = correct = covered = applicable = 0
    wrong: list[str] = []
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
    claims = 0
    for state in states:
        for r in state.validation.verifications:
            counts[r.verdict] += 1
            claims += 1
        cites: list[tuple[str, str, str]] = []
        for f in state.findings:
            if f.applicable and f.status != "NOT_APPLICABLE":
                cites.extend((f.dimension, c.chunk_id, c.claim) for c in f.citations)
        cites.extend((l.dimension, c.chunk_id, c.claim) for l in state.applicable_limits for c in l.citations)
        gold_by_dim = {d: set().union(*gold_chunks(requirements_for(d, state.facts), corpus).values() or [set()])
                       for d in {c[0] for c in cites} if requirements_for(d, state.facts)}
        for dim, cid, claim in cites:
            if dim in gold_by_dim:
                total += 1
                if cid in gold_by_dim[dim]:
                    correct += 1
                else:
                    wrong.append(f"{state.case_id}:{dim}:{cid}")
        for f in state.findings:
            if f.applicable and f.status != "NOT_APPLICABLE" and f.dimension in gold_by_dim:
                applicable += 1
                covered += any(c.chunk_id in gold_by_dim[f.dimension] for c in f.citations)
    return {
        "gold_citation_precision": round(correct / total, 3) if total else None,
        "gold_citations_checked": total, "wrong_citations": wrong,
        "finding_citation_coverage": round(covered / applicable, 3) if applicable else None,
        "verifier_claims": claims,
        "verifier_supported_rate": round(counts["SUPPORTED"] / claims, 3) if claims else None,
        "verifier_counts": counts,
    }


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def write_report(m: dict) -> None:
    d, a, r, c, ctl = m["decision"], m["amounts"], m["retrieval"], m["citations"], m["controls"]
    L = ["# Evaluation Report", "",
         f"Generated by `python -m aptino_claims.eval.run_eval` ({m['n_cases']} cases: {m['n_public']} supplied public + {m['n_custom']} candidate-authored). "
         "Fully offline and deterministic: no API key is used and the LLM never influences a decision.", "",
         "## Headline", ""]
    L += _md_table(["Measure", "Result"], [
        ["Decision accuracy", f"{d['correct']}/{d['n']} ({_pct(d['accuracy'])})"],
        ["Unsafe decisions (confident and wrong)", f"{len(d['unsafe_decisions'])} ({_pct(d['unsafe_decision_rate'])})"],
        ["Abstention precision / recall", f"{_pct(d['abstention_precision'])} / {_pct(d['abstention_recall'])}"],
        ["Payable-amount exact match (limit cases)", f"{_pct(a['exact_match_rate'])} of {a['n']}"],
        ["Retrieval recall@5 (hybrid + rerank)", _pct(r["systems"]["reranked"]["recall@5"])],
        ["Gold-citation precision", _pct(c["gold_citation_precision"])],
        ["Verifier: claims SUPPORTED", f"{c['verifier_counts']['SUPPORTED']}/{c['verifier_claims']}"],
        ["Verifier fault-injection detection", f"{ctl['detected']}/{ctl['injected']} ({_pct(ctl['detection_rate'])})"],
        ["Deterministic across two cold runs", str(m["reproducibility"]["identical"])],
        ["Latency p50 / p95 (cold cache)", f"{m['latency_ms']['p50']:.0f} ms / {m['latency_ms']['p95']:.0f} ms"],
    ])

    L += ["", "## Decision quality", "",
          "Expected outcomes are hand-derived from the policy text (see `eval/expected_outcomes.py` for each case's rationale).", ""]
    L += _md_table(["Case", "Expected", "Actual", "OK", "Deduction exp / act (INR)", "Claims verified", "Attempts"],
                   [[p["case_id"], p["expected_decision"], p["actual_decision"], "yes" if p["decision_correct"] else "NO",
                     "-" if p["expected_deduction_inr"] is None else f"{p['expected_deduction_inr']:,.0f} / {p['actual_deduction_inr']:,.0f}",
                     f"{p['claims_supported']}/{p['claims_total']}", p["attempts"]] for p in m["per_case"]])
    L += ["", "Confusion matrix (rows = expected, columns = predicted):", ""]
    cls = DECISION_CLASSES
    L += _md_table(["expected \\ predicted"] + cls, [[e] + [d["confusion_matrix"][e][p] for p in cls] for e in cls])
    L += ["", "Per-class precision / recall / F1:", ""]
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    L += _md_table(["Class", "Support", "Precision", "Recall", "F1"],
                   [[k, v["support"], fmt(v["precision"]), fmt(v["recall"]), fmt(v["f1"])] for k, v in d["per_class"].items()])

    L += ["", "## Retrieval quality (ablation)", "",
          f"{r['n_case_dimension_pairs']} (case, decision-dimension) queries. Gold evidence is defined by phrase in `eval/gold.py` "
          "(written from the PDF, independent of retriever output). recall@k = fraction of required policy clauses present in the top-k; MRR = reciprocal rank of the first gold chunk.", ""]
    names = {"dense": "Dense only (BGE-small)", "sparse": "BM25 only", "fused": "Dense + BM25 (RRF)", "reranked": "RRF + cross-encoder rerank (system)"}
    L += _md_table(["Retriever", "recall@1", "recall@3", "recall@5", "MRR"],
                   [[names[s], *[f"{r['systems'][s][f'recall@{k}']:.3f}" for k in KS], f"{r['systems'][s]['mrr']:.3f}"] for s in names])
    L += ["", "Hybrid + rerank recall@5 by dimension:", ""]
    L += _md_table(["Dimension", "recall@5"], [[k, f"{v:.3f}"] for k, v in r["hybrid_rerank_recall@5_by_dimension"].items()])

    L += ["", "## Citation correctness", "",
          f"- Gold-citation precision: {_pct(c['gold_citation_precision'])} of {c['gold_citations_checked']} citations point at a chunk that satisfies the gold phrase spec for that dimension."
          + (f" Wrong: {', '.join(c['wrong_citations'])}." if c["wrong_citations"] else ""),
          f"- Finding citation coverage: {_pct(c['finding_citation_coverage'])} of applicable findings cite at least one gold chunk.",
          f"- Evidence verifier: {c['verifier_counts']['SUPPORTED']} SUPPORTED, {c['verifier_counts']['UNSUPPORTED']} UNSUPPORTED, "
          f"{c['verifier_counts']['CONTRADICTED']} CONTRADICTED across {c['verifier_claims']} claims (claim -> citation -> chunk -> verdict).", ""]

    L += ["## Can the verifier catch bad citations? (negative controls)", "",
          "Known faults are injected into correct states; a control is *detected* when the affected claim is no longer SUPPORTED.", ""]
    L += _md_table(["Fault injected", "Injected", "Detected", "Rate"],
                   [[k.replace("_", " "), v["injected"], v["detected"], _pct(v["detection_rate"])] for k, v in ctl["by_type"].items()])
    missed = [f"{k}: {x}" for k, v in ctl["by_type"].items() for x in v["missed"]]
    if missed:
        L += ["", "Missed: " + "; ".join(missed)]

    L += ["", "## Abstention", "",
          f"Cases whose correct behaviour is NEEDS_REVIEW: {', '.join(REQUIRED_ABSTENTIONS)}. "
          f"Abstained correctly: {sum(1 for cid in REQUIRED_ABSTENTIONS if m['abstention'][cid])}/{len(REQUIRED_ABSTENTIONS)}. "
          f"Confident-and-wrong decisions: {len(d['unsafe_decisions'])}.", ""]

    L += ["## Reproducibility", "",
          f"Two independent cold runs (retrieval cache cleared between them) produced {'byte-identical' if m['reproducibility']['identical'] else 'DIFFERENT'} "
          f"responses (sha256 {m['reproducibility']['sha256_run1'][:16]}... vs {m['reproducibility']['sha256_run2'][:16]}...). "
          f"Environment: Python {m['environment']['python']}, dense={m['environment']['dense_model']}, rerank={m['environment']['rerank_model']}, "
          f"top-k dense/sparse/fused/final = {m['environment']['top_k']}.", ""]

    L += ["## What this evaluation does not prove", "",
          "- Labels are the author's reading of the policy, not an insurer's adjudication; hidden reviewer labels may differ, especially where the policy text is ambiguous (see each case's `assumptions`).",
          f"- {m['n_cases']} cases is small; 100% here is evidence the pipeline behaves as designed on these fact patterns, not a claim about all claims.",
          "- Several custom cases were written after the system existed to cover branches the public set does not; they test coverage, not generalisation to unseen wording.",
          "- Gold evidence is phrase-based; a chunk that supports a claim in different words would not count as gold.",
          "- Fault-injection measures the verifier against the fault types listed above, not arbitrary errors."]
    (RESULTS_DIR / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever.load_or_build(settings.index_dir, settings.index_dir / "chunks.jsonl")
    corpus = {cid: meta["text"] for cid, meta in retriever.chunk_meta.items()}
    cases = _load_cases()
    missing = [c["case_id"] for c in cases if c["case_id"] not in EXPECTED]
    if missing:
        raise SystemExit(f"cases without a labeled expected outcome: {missing}")

    retriever.clear_cache()
    states, responses, latencies = _run_all(cases, retriever)
    retriever.clear_cache()
    _, responses2, _ = _run_all(cases, retriever)
    h1 = hashlib.sha256("\n".join(_canonical(r) for r in responses).encode()).hexdigest()
    h2 = hashlib.sha256("\n".join(_canonical(r) for r in responses2).encode()).hexdigest()

    per_case = []
    for c, s, r, ms in zip(cases, states, responses, latencies):
        exp = EXPECTED[c["case_id"]]
        counts = r["validation"]["counts"]
        per_case.append({
            "case_id": c["case_id"], "expected_decision": exp.decision, "actual_decision": r["decision"],
            "decision_correct": exp.decision == r["decision"], "confidence": r["confidence"],
            "expected_deduction_inr": exp.expected_deduction_inr, "actual_deduction_inr": r["amounts"]["total_deductions_inr"],
            "estimated_payable_inr": r["amounts"]["estimated_payable_inr"],
            "claims_total": sum(counts.values()), "claims_supported": counts.get("SUPPORTED", 0),
            "validation_status": r["validation"]["status"], "attempts": s.attempt,
            "n_assumptions": len(r["assumptions"]), "n_unmodelled_risks": len(r["unmodelled_policy_risks"]),
            "primary_dimensions": exp.primary_dimensions, "latency_ms": round(ms, 1),
        })

    decision = decision_block(cases, responses)
    metrics = {
        "n_cases": len(cases), "n_public": sum(1 for c in cases if c["case_id"].startswith("PUB")),
        "n_custom": sum(1 for c in cases if not c["case_id"].startswith("PUB")),
        "decision": decision,
        "amounts": amounts_block(cases, responses),
        "retrieval": retrieval_block(states, retriever, corpus),
        "citations": citation_block(states, corpus),
        "controls": run_controls(states, corpus),
        "abstention": {cid: next(p["actual_decision"] for p in per_case if p["case_id"] == cid) == "NEEDS_REVIEW" for cid in REQUIRED_ABSTENTIONS},
        "reproducibility": {"identical": h1 == h2, "sha256_run1": h1, "sha256_run2": h2},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "mean": mean(latencies)},
        "environment": {"python": platform.python_version(), "dense_model": settings.dense_model, "rerank_model": settings.rerank_model,
                        "top_k": [settings.top_k_dense, settings.top_k_sparse, settings.top_k_fused, settings.top_k_final]},
        "per_case": per_case,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(metrics)

    print(f"Decision accuracy: {decision['accuracy']:.1%} ({decision['correct']}/{decision['n']}) | unsafe: {len(decision['unsafe_decisions'])}")
    print(f"Payable-amount exact match: {metrics['amounts']['exact_match_rate']} of {metrics['amounts']['n']}")
    sysm = metrics["retrieval"]["systems"]
    print("Retrieval recall@5 -> " + " | ".join(f"{k}: {v['recall@5']:.3f}" for k, v in sysm.items()))
    print(f"Gold-citation precision: {metrics['citations']['gold_citation_precision']} | verifier supported: {metrics['citations']['verifier_counts']}")
    print(f"Fault-injection detection: {metrics['controls']['detected']}/{metrics['controls']['injected']}")
    print(f"Reproducible: {metrics['reproducibility']['identical']} | latency p50/p95: {metrics['latency_ms']['p50']:.0f}/{metrics['latency_ms']['p95']:.0f} ms")
    print(f"Wrote {RESULTS_DIR / 'metrics.json'} and {RESULTS_DIR / 'report.md'}")

    failed = (decision["accuracy"] < 1.0 or not all(metrics["abstention"].values()) or not metrics["reproducibility"]["identical"]
              or (metrics["controls"]["detection_rate"] or 0) < 1.0 or (metrics["amounts"]["exact_match_rate"] or 0) < 1.0)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
