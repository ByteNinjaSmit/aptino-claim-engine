"""Structure-aware chunking of the policy document.

Rather than a fixed-size sliding window, the chunker walks the document's own
structure:

1. Top-level SECTION headings (DEFINITIONS, WHAT WE COVER, WHAT WE EXCLUDE,
   EXTENSIONS, CLAIMS PROCEDURE, STANDARD TERMS AND CONDITIONS, ...) are
   detected from all-caps / curated title-case heading lines.
2. Inside DEFINITIONS, each "<Term> means ..." paragraph becomes its own
   chunk, so a citation can point at exactly the defined term.
3. Inside numbered-clause sections (WHAT WE COVER, WHAT WE EXCLUDE, STANDARD
   TERMS AND CONDITIONS, the Critical Illness list, ...) each top-level
   numbered item -- together with its lettered/roman sub-points and NB notes
   -- becomes one chunk, since splitting a clause from its own sub-limits or
   notes would make the clause unsafe to cite.
4. Everything else is chunked by blank-line-delimited paragraph, merging
   paragraphs that are too short to stand alone and splitting ones that run
   long, so no chunk is a meaningless fragment.

Every chunk keeps the page range it was drawn from, its section heading, a
stable chunk_id, and the running item/definition label so retrieval metadata
can be traced straight back to the source PDF.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_loader import PageLine

# Section headings as they literally appear in this policy (case-sensitive
# where the source itself is not all-caps, e.g. "Critical Illness").
_KNOWN_HEADINGS = {
    "DEFINITIONS": "Definitions",
    "SCOPE OF COVER": "Scope of Cover",
    "WHAT WE COVER": "What We Cover",
    "WHAT WE EXCLUDE": "What We Exclude",
    "EXTENSIONS": "Extensions",
    "CLAIMS PROCEDURE": "Claims Procedure",
    "STANDARD TERMS AND CONDITIONS": "Standard Terms and Conditions",
    "STANDARD TERMS AND CONDITIONS:": "Standard Terms and Conditions",
    "CRITICAL ILLNESS": "Critical Illness Benefit (Optional Cover)",
}

_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z ,&/'\-]{2,60}:?$")
_HEADING_STOPWORDS = {"AND", "OR", "NB", "IF", "BUT", "THEN", "OF", "OTHER", "TO"}
_DEFINITION_START_RE = re.compile(r"^([A-Z][A-Za-z0-9()/,.'’\- ]{1,70}?)\s+means\b")
_TOP_NUMBERED_RE = re.compile(r"^(\d{1,2})\.\s+\S")

_PREAMBLE_SECTION = "Preamble"
_MAX_PARAGRAPH_WORDS = 160
_MIN_PARAGRAPH_WORDS = 12


@dataclass
class PolicyChunk:
    chunk_id: str
    section: str
    label: str
    page_start: int
    page_end: int
    text: str
    order: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section": self.section,
            "label": self.label,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "order": self.order,
            "source": "policy.pdf",
        }


def _detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped in _KNOWN_HEADINGS:
        return _KNOWN_HEADINGS[stripped]
    if stripped.rstrip(":") in _KNOWN_HEADINGS:
        return _KNOWN_HEADINGS[stripped.rstrip(":")]
    if (
        _ALLCAPS_HEADING_RE.match(stripped)
        and 1 <= len(stripped.split()) <= 6
        and stripped.rstrip(":") not in _HEADING_STOPWORDS
        and len(stripped.rstrip(":")) >= 5
    ):
        # Guard against short all-caps noise (connectors, "TPA", ...) by
        # requiring the *entire* line to be a plausible multi-letter heading.
        return stripped.rstrip(":").title()
    return None


@dataclass
class _Section:
    name: str
    lines: list[PageLine] = field(default_factory=list)


def _split_into_sections(lines: list[PageLine]) -> list[_Section]:
    sections: list[_Section] = [_Section(name=_PREAMBLE_SECTION)]
    for pl in lines:
        heading = _detect_heading(pl.text)
        if heading:
            sections.append(_Section(name=heading))
            continue
        sections[-1].lines.append(pl)
    return [s for s in sections if any(l.text.strip() for l in s.lines)]


def _page_span(lines: list[PageLine]) -> tuple[int, int]:
    pages = [l.page_number for l in lines] or [0]
    return min(pages), max(pages)


def _chunk_definitions(section: _Section, start_order: int) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    current: list[PageLine] = []
    current_term = "Preliminary"

    def flush(term: str, buf: list[PageLine]) -> None:
        text = " ".join(l.text.strip() for l in buf if l.text.strip())
        if not text:
            return
        p_start, p_end = _page_span(buf)
        idx = len(chunks) + 1
        chunks.append(
            PolicyChunk(
                chunk_id=f"definitions-{idx:03d}",
                section="Definitions",
                label=f"Definitions — {term}",
                page_start=p_start,
                page_end=p_end,
                text=text,
                order=start_order + idx,
            )
        )

    for pl in section.lines:
        m = _DEFINITION_START_RE.match(pl.text.strip())
        if m and current:
            flush(current_term, current)
            current = [pl]
            current_term = m.group(1).strip()
        else:
            if m and not current:
                current_term = m.group(1).strip()
            current.append(pl)
    flush(current_term, current)
    return chunks


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _chunk_numbered(section: _Section, start_order: int) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    current: list[PageLine] = []
    current_item = 0
    slug = _slugify(section.name)

    def flush(item_no: int, buf: list[PageLine]) -> None:
        text = " ".join(l.text.strip() for l in buf if l.text.strip())
        if not text:
            return
        p_start, p_end = _page_span(buf)
        idx = len(chunks) + 1
        label = f"{section.name} — Item {item_no}" if item_no else f"{section.name} — Preamble"
        chunks.append(
            PolicyChunk(
                chunk_id=f"{slug}-{idx:03d}",
                section=section.name,
                label=label,
                page_start=p_start,
                page_end=p_end,
                text=text,
                order=start_order + idx,
            )
        )

    for pl in section.lines:
        m = _TOP_NUMBERED_RE.match(pl.text.strip())
        # Only treat this as the *next* top-level clause if its number is a
        # strict continuation (1, 2, 3, ...). A nested list restarting at
        # "1." inside a clause (e.g. the Portability sub-points) must not be
        # mistaken for a new top-level item.
        if m and int(m.group(1)) == current_item + 1:
            flush(current_item, current)
            current = [pl]
            current_item = int(m.group(1))
        else:
            current.append(pl)
    flush(current_item, current)
    return chunks


def _chunk_paragraphs(section: _Section, start_order: int) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    paragraphs: list[list[PageLine]] = [[]]
    for pl in section.lines:
        if not pl.text.strip():
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        paragraphs[-1].append(pl)
    paragraphs = [p for p in paragraphs if p]

    merged: list[list[PageLine]] = []
    for para in paragraphs:
        word_count = sum(len(l.text.split()) for l in para)
        if merged and word_count < _MIN_PARAGRAPH_WORDS:
            merged[-1].extend(para)
        else:
            merged.append(list(para))

    slug = _slugify(section.name)
    for para in merged:
        words: list[str] = []
        buf: list[PageLine] = []
        for l in para:
            buf.append(l)
            words.extend(l.text.split())
            if len(words) >= _MAX_PARAGRAPH_WORDS:
                text = " ".join(x.text.strip() for x in buf if x.text.strip())
                p_start, p_end = _page_span(buf)
                idx = len(chunks) + 1
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"{slug}-{idx:03d}",
                        section=section.name,
                        label=f"{section.name} — Part {idx}",
                        page_start=p_start,
                        page_end=p_end,
                        text=text,
                        order=start_order + idx,
                    )
                )
                buf, words = [], []
        if buf:
            text = " ".join(x.text.strip() for x in buf if x.text.strip())
            p_start, p_end = _page_span(buf)
            idx = len(chunks) + 1
            chunks.append(
                PolicyChunk(
                    chunk_id=f"{slug}-{idx:03d}",
                    section=section.name,
                    label=f"{section.name} — Part {idx}",
                    page_start=p_start,
                    page_end=p_end,
                    text=text,
                    order=start_order + idx,
                )
            )
    return chunks


_NUMBERED_SECTIONS = {
    "What We Cover",
    "What We Exclude",
    "Standard Terms and Conditions",
    "Critical Illness Benefit (Optional Cover)",
}


def build_chunks(lines: list[PageLine]) -> list[PolicyChunk]:
    sections = _split_into_sections(lines)
    chunks: list[PolicyChunk] = []
    for section in sections:
        order = len(chunks)
        if section.name == "Definitions":
            chunks.extend(_chunk_definitions(section, order))
        elif section.name in _NUMBERED_SECTIONS:
            chunks.extend(_chunk_numbered(section, order))
        else:
            chunks.extend(_chunk_paragraphs(section, order))
    return chunks
