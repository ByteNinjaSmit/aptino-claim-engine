"""Numeric-threshold extraction from retrieved policy text.

The admissibility logic must stay provably tied to what was actually
retrieved rather than a hardcoded lookup table that could drift from the
source PDF. These helpers pull the specific numbers a dimension needs
(a day count, a month count, a percentage) directly out of the text of the
evidence chunk that was retrieved for that dimension, using a targeted
regex. If the pattern is not found in the retrieved text, the caller must
treat the threshold as unknown (-> INSUFFICIENT_EVIDENCE), never guess.
"""
from __future__ import annotations

import re


def find_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else None


def find_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None


def extract_waiting_days(text: str) -> int | None:
    return find_int(r"waiting period of (\d+)\s*days", text) or find_int(r"(\d+)\s*days?\s*waiting period", text)


def extract_pre_existing_months(text: str) -> int | None:
    return find_int(r"(\d+)\s*months of continuous coverage", text)


_CLAUSE_SPLIT = re.compile(r"(?=\b\d{1,2}\.\s)|(?=NB\d+:)|(?=\b[a-c]\)\s)")
_PERCENT_RE = re.compile(r"([\d.]+)\s*%\s*(?:of\s*)?(?:the\s*)?(?:Basic\s*)?Sum\s*(?:Insured|Assured)", re.I)
_FLAT_RE = re.compile(r"(?:Rupees|Rs\.?)\s*(\d+)", re.I)


def clauses(text: str) -> list[str]:
    """Split policy text into clause-level segments (numbered items, NB notes, a)/b)/c) sub-points).

    Limits are always stated inside a single clause, so binding a keyword to
    a number *within the same clause* is far more reliable than a character
    window that can straddle two neighbouring clauses.
    """
    return [c for c in _CLAUSE_SPLIT.split(text) if c.strip()]


def clause_with(text: str, near: str) -> str | None:
    for clause in clauses(text):
        if near.lower() in clause.lower() and (_PERCENT_RE.search(clause) or _FLAT_RE.search(clause)):
            return clause
    return None


def extract_percent_of_si(text: str, near: str) -> float | None:
    """The `<pct>% of ... Sum Insured/Assured` figure in the clause mentioning `near`."""
    clause = clause_with(text, near)
    m = _PERCENT_RE.search(clause) if clause else None
    return float(m.group(1)) if m else None


def extract_first_percent(text: str) -> float | None:
    m = _PERCENT_RE.search(text)
    return float(m.group(1)) if m else None


def extract_flat_amount(text: str, near: str) -> float | None:
    """A flat rupee cap ("Rupees 1000/-") in the clause mentioning `near`."""
    clause = clause_with(text, near)
    m = _FLAT_RE.search(clause) if clause else None
    return float(m.group(1)) if m else None


def extract_pre_post_windows(text: str) -> tuple[int | None, int | None]:
    pre = find_int(r"maximum of (\d+)\s*days? immediately preceding", text)
    post = find_int(r"maximum of (\d+)\s*days? immediately following", text)
    return pre, post


def extract_min_beds(text: str) -> tuple[int | None, int | None]:
    small_town = find_int(r"at least (\d+)\s*in-?patient beds in towns", text)
    other = find_int(r"at least (\d+)\s*\n?\s*in-?patient beds in all other", text)
    return small_town, other
