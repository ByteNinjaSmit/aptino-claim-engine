"""Gold evidence for retrieval and citation evaluation.

For each decision dimension, the policy text that a correct answer must rest
on is described as a list of *requirements*; a requirement is a list of
*alternatives*; an alternative is a list of phrases that must all appear in a
chunk. A chunk is "gold" for a requirement if it satisfies any alternative.

Describing gold by phrase (rather than by chunk id) keeps the labels valid if
the policy is re-chunked, and the labels were written by reading the PDF, not
by looking at what the retriever returned.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")

GOLD: dict[str, dict[str, list[list[str]]]] = {
    "initial_waiting_period": {"waiting period": [["waiting period of 30 days"]]},
    "named_disease_first_year_waiting_period": {"first-year list": [["first year of operation", "cataract"]]},
    "pre_existing_disease_waiting_period": {"48-month wait": [["48 months of continuous coverage"]]},
    "hospital_definition": {"hospital definition": [["hospital means", "in-patient beds"]]},
    "domiciliary_treatment_conditions": {"domiciliary definition": [["domiciliary treatment means"]]},
    "day_care_less_than_24h": {"day-care basis": [["undertaken under general or local anesthesia"], ["eye surgery", "lithotripsy"]]},
    "pre_post_hospitalization_window": {"pre/post windows": [["immediately preceding hospitalisation"]]},
    "cosmetic_exclusion": {"cosmetic exclusion": [["cosmetic or aesthetic"]]},
    "experimental_unproven_treatment": {"experimental definition": [["unproven/experimental treatment means"]]},
    "category_sub_limits": {
        "room": [["normal room expenses"]],
        "fees": [["surgeons fees"]],
        "medicines": [["anesthesia, blood"]],
        "ambulance": [["ambulance charges"]],
        "domiciliary": [["domiciliary hospitalization will be paid"]],
    },
}


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower())


def requirements_for(dimension: str, facts: dict) -> dict[str, list[list[str]]]:
    """Requirements that apply to *this* case (e.g. only the caps for expense heads actually claimed)."""
    reqs = GOLD.get(dimension, {})
    if dimension != "category_sub_limits":
        return reqs
    exp = facts.get("expenses", {})
    if facts.get("treatment_type") == "domiciliary":
        return {"domiciliary": reqs["domiciliary"]}
    wanted = {"room": exp.get("room", 0) > 0, "fees": exp.get("doctor_fees", 0) > 0,
              "medicines": exp.get("medicines_diagnostics", 0) > 0, "ambulance": exp.get("ambulance", 0) > 0}
    return {name: alts for name, alts in reqs.items() if wanted.get(name)}


def gold_chunks(requirements: dict[str, list[list[str]]], corpus: dict[str, str]) -> dict[str, set[str]]:
    """Map each requirement to the set of chunk ids that satisfy it."""
    normed = {cid: _norm(text) for cid, text in corpus.items()}
    out: dict[str, set[str]] = {}
    for name, alternatives in requirements.items():
        out[name] = {cid for cid, body in normed.items() if any(all(p in body for p in alt) for alt in alternatives)}
    return out
