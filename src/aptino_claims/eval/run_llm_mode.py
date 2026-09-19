"""Evaluate the optional LLM interpretation step against a real model.

    LLM_PROVIDER=openai_compatible OPENAI_API_KEY=... python -m aptino_claims.eval.run_llm_mode

Runs every labeled case twice -- deterministic pipeline and with the LLM
interpretation step on -- and reports:

  * what the step changes (decision before -> after) and why (verified quotes)
  * false alarms: cases whose labeled outcome is decisive but that became
    NEEDS_REVIEW (the step is one-directional, so this is its only failure mode)
  * catches: adversarial cases (paraphrases the keyword rules miss) that the
    step flagged
  * rejected observations (hallucinated chunk ids / non-verbatim quotes)

Unlike `run_eval`, this run is not part of the deterministic gate: it needs a
provider, costs API calls, and model output can vary between runs.
"""
from __future__ import annotations

import json
import sys
import time

from ..agents.orchestrator import analyze_case
from ..config import REPO_ROOT, settings
from ..llm.providers import OfflineLLMClient, get_llm_client
from ..retrieval.retriever import HybridRetriever
from .expected_outcomes import EXPECTED

ADVERSARIAL_PATH = REPO_ROOT / "data" / "custom_cases" / "adversarial_cases.json"
RESULTS_DIR = REPO_ROOT / "eval_results"
DECISIVE = {"ADMISSIBLE", "ADMISSIBLE_WITH_LIMITS", "PARTIALLY_ADMISSIBLE", "NOT_ADMISSIBLE"}
# What a careful reviewer would do with the adversarial cases: hold them for review.
ADVERSARIAL_EXPECTED = "NEEDS_REVIEW"


def _cases() -> list[dict]:
    from .run_eval import _load_cases
    adversarial = json.loads(ADVERSARIAL_PATH.read_text(encoding="utf-8")) if ADVERSARIAL_PATH.exists() else []
    return _load_cases() + adversarial


def main() -> None:
    if settings.llm_provider != "openai_compatible" or not settings.openai_api_key:
        raise SystemExit("Set LLM_PROVIDER=openai_compatible and OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) to run the LLM-mode evaluation.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever.load_or_build(settings.index_dir, settings.index_dir / "chunks.jsonl")
    llm, offline = get_llm_client(), OfflineLLMClient()

    rows = []
    for c in _cases():
        cid = c["case_id"]
        expected = EXPECTED[cid].decision if cid in EXPECTED else ADVERSARIAL_EXPECTED
        before = analyze_case(c, retriever, offline, interpret=False)
        t0 = time.perf_counter()
        after = analyze_case(c, retriever, llm, interpret=True)
        report = after.llm_interpretation
        rows.append({
            "case_id": cid, "adversarial": cid.startswith("ADV"), "expected": expected,
            "deterministic": before.decision.value, "with_llm": after.decision.value,
            "changed": before.decision != after.decision,
            "accepted": report.get("accepted", []), "rejected": report.get("rejected", []), "error": report.get("error"),
            "validation": after.validation.status, "latency_ms": round((time.perf_counter() - t0) * 1000),
        })
        print(f"{cid:9} {before.decision.value:24} -> {after.decision.value:24} +{len(report.get('accepted', []))} accepted, "
              f"{len(report.get('rejected', []))} rejected {('| ' + report['error']) if report.get('error') else ''}")

    labeled = [r for r in rows if not r["adversarial"]]
    adv = [r for r in rows if r["adversarial"]]
    false_alarms = [r["case_id"] for r in labeled if r["expected"] in DECISIVE and r["with_llm"] == "NEEDS_REVIEW"]
    regressions = [r["case_id"] for r in labeled if r["deterministic"] == r["expected"] and r["with_llm"] != r["expected"]
                   and r["with_llm"] != "NEEDS_REVIEW"]
    summary = {
        "model": settings.openai_interpretation_model, "n_labeled": len(labeled), "n_adversarial": len(adv),
        "labeled_accuracy_deterministic": round(sum(r["deterministic"] == r["expected"] for r in labeled) / len(labeled), 3),
        "labeled_accuracy_with_llm": round(sum(r["with_llm"] == r["expected"] for r in labeled) / len(labeled), 3),
        "false_alarms": false_alarms, "false_alarm_rate_on_decisive_cases":
            round(len(false_alarms) / max(1, sum(r["expected"] in DECISIVE for r in labeled)), 3),
        "moved_to_a_different_decisive_outcome": regressions,
        "adversarial_deterministic_needs_review": sum(r["deterministic"] == ADVERSARIAL_EXPECTED for r in adv),
        "adversarial_with_llm_needs_review": sum(r["with_llm"] == ADVERSARIAL_EXPECTED for r in adv),
        "observations_accepted": sum(len(r["accepted"]) for r in rows), "observations_rejected": sum(len(r["rejected"]) for r in rows),
        "validation_failures": [r["case_id"] for r in rows if r["validation"] != "PASS"],
    }
    (RESULTS_DIR / "llm_mode.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    _write_md(summary, rows)
    print(json.dumps(summary, indent=2))


def _write_md(s: dict, rows: list[dict]) -> None:
    L = ["# LLM interpretation step: evaluation", "",
         f"Model `{s['model']}`, temperature 0. Not part of the deterministic gate (see `run_llm_mode.py`).", "",
         "| Measure | Result |", "|---|---|",
         f"| Labeled-case accuracy, deterministic | {s['labeled_accuracy_deterministic']:.1%} |",
         f"| Labeled-case accuracy, with LLM step | {s['labeled_accuracy_with_llm']:.1%} |",
         f"| False alarms (decisive case sent to review) | {len(s['false_alarms'])} ({s['false_alarm_rate_on_decisive_cases']:.0%} of decisive cases): {', '.join(s['false_alarms']) or '-'} |",
         f"| Moved to a different *decisive* outcome | {len(s['moved_to_a_different_decisive_outcome'])} (must be 0: the step is one-directional) |",
         f"| Adversarial cases held for review: deterministic / with LLM | {s['adversarial_deterministic_needs_review']} / {s['adversarial_with_llm_needs_review']} of {s['n_adversarial']} |",
         f"| Observations accepted / rejected | {s['observations_accepted']} / {s['observations_rejected']} |",
         f"| Validation failures | {len(s['validation_failures'])} |", "",
         "## Cases where the step changed the outcome or made observations", "",
         "| Case | Expected | Deterministic | With LLM | Accepted observations (verified quote) | Rejected |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["changed"] or r["accepted"] or r["rejected"]:
            acc = "; ".join(f"{a['chunk_id']}: \"{a['quote'][:60]}\" ({a['type']})" for a in r["accepted"]) or "-"
            rej = "; ".join(x["reason"] for x in r["rejected"]) or "-"
            L.append(f"| {r['case_id']} | {r['expected']} | {r['deterministic']} | {r['with_llm']} | {acc} | {rej} |")
    L += ["", "The step can only add INSUFFICIENT_EVIDENCE findings backed by a verbatim quote, so a change is always toward NEEDS_REVIEW.",
          "A false alarm costs a human look; a catch prevents a confident wrong answer on a paraphrase the rules do not match."]
    (RESULTS_DIR / "llm_mode.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
