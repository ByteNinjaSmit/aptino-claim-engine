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


def extract_percent_of_si(text: str, near: str) -> float | None:
    """Find a `<pct>% of ... Sum Insured/Assured` figure appearing near a keyword."""
    for m in re.finditer(r"([\d.]+)\s*%\s*of\s*(?:the\s*)?(?:Basic\s*)?Sum\s*(?:Insured|Assured)", text, re.I):
        window = text[max(0, m.start() - 120) : m.start()]
        if near.lower() in window.lower() or near.lower() in text[m.start() : m.end() + 20].lower():
            return float(m.group(1))
    return None


def extract_first_percent(text: str) -> float | None:
    m = re.search(r"([\d.]+)\s*%\s*of\s*(?:the\s*)?(?:Basic\s*)?Sum\s*(?:Insured|Assured)", text, re.I)
    return float(m.group(1)) if m else None


def extract_flat_amount(text: str, near: str) -> float | None:
    for m in re.finditer(r"Rs\.?\s*(\d+)/?-?|Rupees\s*(\d+)", text, re.I):
        val = m.group(1) or m.group(2)
        window = text[max(0, m.start() - 80) : m.start() + 40]
        if near.lower() in window.lower():
            return float(val)
    return None


def extract_pre_post_windows(text: str) -> tuple[int | None, int | None]:
    pre = find_int(r"maximum of (\d+)\s*days? immediately preceding", text)
    post = find_int(r"maximum of (\d+)\s*days? immediately following", text)
    return pre, post


def extract_min_beds(text: str) -> tuple[int | None, int | None]:
    small_town = find_int(r"at least (\d+)\s*in-?patient beds in towns", text)
    other = find_int(r"at least (\d+)\s*\n?\s*in-?patient beds in all other", text)
    return small_town, other
