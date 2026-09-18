from aptino_claims.ingestion.chunker import build_chunks
from aptino_claims.ingestion.pdf_loader import PageLine


def test_definitions_split_by_term():
    lines = [
        PageLine(1, "DEFINITIONS"),
        PageLine(1, "Accident means a sudden unforeseen event."),
        PageLine(1, ""),
        PageLine(1, "Hospital means an institution registered as a Hospital."),
    ]
    chunks = build_chunks(lines)
    defs = [c for c in chunks if c.section == "Definitions"]
    assert len(defs) == 2
    assert "Accident" in defs[0].label
    assert "Hospital" in defs[1].label


def test_numbered_section_splits_top_level_items_only():
    lines = [
        PageLine(9, "WHAT WE EXCLUDE"),
        PageLine(9, "1. Pre-existing diseases"),
        PageLine(9, "Some explanatory text."),
        PageLine(9, "i. a nested point"),
        PageLine(9, "OR"),
        PageLine(9, "ii. another nested point"),
        PageLine(9, "2. 30 days Waiting Period"),
        PageLine(9, "More text."),
    ]
    chunks = build_chunks(lines)
    items = [c for c in chunks if c.section == "What We Exclude"]
    assert len(items) == 2
    assert items[0].text.startswith("1. Pre-existing diseases")
    assert "nested point" in items[0].text  # nested numbering stays attached to item 1
    assert items[1].text.startswith("2. 30 days Waiting Period")


def test_heading_stopwords_are_not_treated_as_sections():
    lines = [
        PageLine(1, "WHAT WE EXCLUDE"),
        PageLine(1, "1. Something"),
        PageLine(1, "AND"),
        PageLine(1, "more text that must stay in item 1"),
    ]
    chunks = build_chunks(lines)
    items = [c for c in chunks if c.section == "What We Exclude"]
    assert len(items) == 1
    assert "more text that must stay in item 1" in items[0].text


def test_pages_are_tracked_across_wrapped_lines():
    lines = [
        PageLine(1, "EXTENSIONS"),
        PageLine(1, "Some text on page 1."),
        PageLine(2, "Continued text on page 2."),
    ]
    chunks = build_chunks(lines)
    ext = [c for c in chunks if c.section == "Extensions"][0]
    assert ext.page_start == 1
    assert ext.page_end == 2
