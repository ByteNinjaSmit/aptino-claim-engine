"""CLI: ingest the policy PDF into chunks.jsonl (and build the retrieval index).

    python -m aptino_claims.ingestion.build_index
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import settings
from .chunker import build_chunks
from .pdf_loader import load_policy_lines


def ingest_to_chunks(pdf_path: Path) -> list[dict]:
    lines = load_policy_lines(pdf_path)
    chunks = build_chunks(lines)
    return [c.to_dict() for c in chunks]


def main() -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    chunks = ingest_to_chunks(settings.policy_pdf_path)

    out_path = settings.index_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks -> {out_path}", file=sys.stderr)
    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c["section"]] = by_section.get(c["section"], 0) + 1
    for section, count in by_section.items():
        print(f"  {section}: {count} chunks", file=sys.stderr)

    # Build and persist the dense embedding index up front so first API
    # request doesn't pay the embedding-model download/inference cost.
    from ..retrieval.retriever import HybridRetriever

    retriever = HybridRetriever.build(chunks)
    retriever.save(settings.index_dir)
    print(f"Saved retrieval index -> {settings.index_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
