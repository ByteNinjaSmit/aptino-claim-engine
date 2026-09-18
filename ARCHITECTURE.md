# Architecture / Design Note

## 1. Problem framing

The system must turn a structured health-insurance claim case into a
policy-grounded admissibility decision, where every material statement is
traceable to a specific page/section/chunk of the supplied policy PDF, and
the system abstains (`NEEDS_REVIEW`) rather than guesses when the policy
text or the case facts don't support a safe conclusion.

Two things shaped every downstream choice:

1. **The input is already structured JSON**, not free narrative text. So
   "case analysis" is fact normalization and dimension-detection, not
   NLP extraction — an LLM call here would add latency and hallucination
   risk without adding capability.
2. **The policy is one small, fixed 17-page document.** Correctness is
   independently verifiable by reading the cited clause. That makes it
   possible — and, given the assignment's explicit warning against
   "confident unsupported decisions", *preferable* — to keep the
   admissibility logic deterministic and evidence-grounded rather than
   asking an LLM to reason freely over retrieved text.

## 2. Agent boundaries and state flow

```
raw case JSON
     │
     ▼
┌─────────────────────┐   facts, applicable dimensions,
│ Case Analysis Agent  │──▶ investigation checklist, missing/
└─────────────────────┘   irrelevant-field notes
     │
     ▼
┌─────────────────────┐   per-dimension query → hybrid retrieval
│ Policy Evidence Agent│──▶ (dense + BM25 → RRF fusion → cross-encoder
└─────────────────────┘    rerank) → ranked, metadata-rich evidence
     │
     ▼
┌─────────────────────┐   per-dimension Finding (status + citation),
│ Coverage & Exclusion │──▶ ApplicableLimit (deduction + citation),
│       Agent          │    MissingEvidence
└─────────────────────┘
     │
     ▼
┌─────────────────────┐   precedence rule over findings → decision +
│   Decision Agent     │──▶ confidence + rationale (LLM-phrased only,
└─────────────────────┘    never LLM-decided)
     │
     ▼
┌─────────────────────┐   citation ⊆ retrieved-evidence check,
│  Validation Agent    │──▶ PASS/FAIL, downgrade-to-NEEDS_REVIEW retry
└─────────────────────┘
     │
     ▼
 decision contract JSON
```

Agents exchange a single typed `CaseState` (Pydantic model) — never raw
strings — so each agent's contribution is a structured, independently
inspectable field (`facts`, `evidence_by_dimension`, `findings`,
`applicable_limits`, `decision`, `validation`, `trace`), not another turn
of free text. This is what "genuinely specialized" agents means here: each
agent owns a distinct part of that schema and cannot touch the others.

A plain Python function pipeline (`agents/orchestrator.py`) implements the
sequence instead of a graph framework — the flow is a fixed sequence with
one localized retry (Validation Agent downgrading the decision when a
citation doesn't hold up), so LangGraph would add indirection without
adding capability at this scale. Swapping in LangGraph later only means
replacing that one module.

## 3. Retrieval design

- **Ingestion/chunking** (`ingestion/chunker.py`) walks the document's own
  structure rather than splitting on a fixed size: DEFINITIONS is split
  one term per chunk, numbered sections (What We Cover / What We Exclude /
  Standard Terms and Conditions / the Critical Illness list) are split one
  top-level clause per chunk — with nested sub-points (a/b/c, i/ii/iii)
  kept attached to their parent clause, since separating a sub-limit from
  its parent coverage item would make it uncitable on its own — and
  everything else falls back to paragraph chunking. Every chunk carries
  its page range, section heading, and a stable `chunk_id`.
- **Hybrid retrieval** (`retrieval/`): dense embeddings via a local ONNX
  model (`BAAI/bge-small-en-v1.5`, through `fastembed` — no API key, no
  GPU) and BM25 (`rank_bm25`) run independently; results are combined with
  **Reciprocal Rank Fusion** rather than a weighted score blend, because
  cosine similarity and BM25 scores live on incomparable scales and RRF
  only needs rank order. The fused top-N is then reranked with a
  cross-encoder (`BAAI/bge-reranker-base`, also local/free) before the top
  few chunks reach any agent.
- **Exact search, not ANN.** The corpus is ~115 chunks. A FAISS/Qdrant
  index would add a dependency and zero benefit at this scale; `DenseStore`
  is a 20-line numpy cosine scan behind the same `search()` interface, so
  swapping in a real ANN index later is a one-file change.
- Every retrieval result carries per-stage rank/score
  (`dense_rank/score`, `sparse_rank/score`, `fused_score`,
  `rerank_score`), which is what lets the evaluation script and the API
  response expose retrieval/citation quality instead of a black box.

## 4. Why the admissibility logic is deterministic, not LLM-free-text

Each of the ten decision dimensions (`agents/dimensions.py`) is an
(applicability check, retrieval query, evaluator) triple. The evaluator
never hardcodes a policy number — it **parses the actual number out of the
text of the chunk that was retrieved** (`agents/rules_extract.py`), e.g. it
regex-extracts "30" from "...waiting period of 30 days..." rather than
assuming 30. If the pattern isn't found in what was retrieved, the
dimension reports `INSUFFICIENT_EVIDENCE` — it never falls back to
outside knowledge. This keeps the reasoning provably tied to retrieval
(if the PDF changed, the numbers used would change with it) while
remaining fully deterministic and unit-testable without any LLM.

The **LLM client is real and pluggable** (`llm/providers.py`,
`LLM_PROVIDER=openai_compatible` + any OpenAI-compatible endpoint — Groq,
OpenAI, Together, local Ollama) but is only ever asked to rephrase an
*already-computed, already-cited* finding set into one fluent paragraph —
never to decide the outcome. If no key is configured
(`LLM_PROVIDER=offline`, the default) or the call fails or looks
suspicious (wildly longer than the input), the system falls back to a
deterministic templated rationale built directly from the findings. This
was a deliberate trade-off given the assignment's explicit red flag for
"confident unsupported decisions": free-form LLM reasoning over retrieved
clauses is exactly the failure mode that produces a fluent, wrong,
hard-to-audit answer, and there was no way to guarantee a live API key
would be available for grading. The interpretation step (`Coverage &
Exclusion Agent`) is where a real deployment would most plausibly add LLM
reasoning on top of the same evidence, with the Validation Agent as the
existing safety net against citation drift.

## 5. Validation / retry behavior

The Validation Agent checks two things about every material citation: (1)
the cited `chunk_id` was actually among the evidence retrieved for that
dimension (catches a mismatched citation), and (2) any policy-threshold
number named in the citation's short claim label literally appears in the
cited chunk's text (catches a number that isn't backed by what was
retrieved). On `FAIL`, the case is downgraded to `NEEDS_REVIEW` — this is
the system's one retry/revision behavior. There's nothing left to
re-retrieve at that point (the evidence already didn't support the claim),
so "retry" here means "fail safe to abstention" rather than "try again",
which is the correct behavior for an insurance decision system.

## 6. Trade-offs and known limitations

- **Ten hand-designed dimensions, not a general reasoner.** This covers
  every reliability scenario in the assignment brief and all 18 evaluation
  cases, but a genuinely novel policy question outside these ten axes
  would currently fall through with no dimension flagged, rather than
  being caught generically. A production system would want an eleventh
  "catch-all" dimension or an LLM-based gap-detector.
- **Sub-limit math assumes one occupancy tier.** `expenses_inr.room` is
  treated as normal room rent; the policy's separate ICU/day sub-limit
  isn't applied because the input schema doesn't distinguish ICU days from
  ward days.
- **Regex-based number extraction is exact-text-dependent.** It works
  because this is a fixed, known PDF; a differently-formatted policy PDF
  would need a review of the extraction patterns (this is explicitly
  favored over a hardcoded number table for the reasons in §4, but it's
  not a general-purpose numeric-clause parser).
- **PARTIALLY_ADMISSIBLE vs ADMISSIBLE_WITH_LIMITS** is decided by whether
  a distinctly-named claim component (currently: pre/post-hospitalization
  expenses on a confirmed window violation) is fully excluded, vs. a
  proportional cap reducing but not zeroing a component. This line is a
  judgment call documented here rather than dictated by the policy text.
