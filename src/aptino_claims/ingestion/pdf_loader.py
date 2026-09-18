"""Extract per-page text from the policy PDF, stripping running headers/footers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

_NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s+CSC-\s*Individual Health Insurance-Policy Wording.*IRDAI Reg No:\d+\s*$", re.I),
    re.compile(r"^\s*UNIVERSAL SOMPO GENERAL INSURANCE CO\.?\s*LTD\.?\s*$", re.I),
]


@dataclass(frozen=True)
class PageLine:
    page_number: int  # 1-indexed, matches the PDF's printed/physical page
    text: str


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _NOISE_PATTERNS)


def _clean(line: str) -> str:
    # Wingdings/Symbol bullet glyphs land in the Unicode Private Use Area
    # when extracted as text; normalize them to a plain hyphen bullet.
    return re.sub(r"[-]", "-", line)


def load_policy_lines(pdf_path: Path) -> list[PageLine]:
    """Return the document as an ordered list of (page_number, line) pairs.

    Running page headers/footers (page-number + document title, company
    letterhead) are dropped so they never leak into a citation chunk.
    """
    lines: list[PageLine] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            for line in raw.split("\n"):
                if _is_noise(line):
                    continue
                lines.append(PageLine(page_number=page.page_number, text=_clean(line)))
    return lines
