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
┌─────────────────────┐   claim -> citation -> chunk -> verdict
│  Validation Agent    │──▶ (SUPPORTED/UNSUPPORTED/CONTRADICTED);
└─────────────────────┘    on FAIL: widened re-retrieval once, then abstain
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
one localized retry loop (Validation -> Policy Evidence, once, with widened retrieval), so LangGraph would add indirection without
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

Each of the ten modelled decision dimensions (`agents/dimensions.py`) is an
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

## 5. Evidence verification and the retry loop

Every material statement is turned into an auditable chain:

```
decision claim -> citation -> retrieved policy chunk -> evidence check -> SUPPORTED | UNSUPPORTED | CONTRADICTED
```

Each citation a finding produces carries a machine-checkable *assertion*
(`agents/verification.py`): a threshold ("30 days"), required phrases, a
limit ("1% of sum insured, cap = 1% x SI x days, deduction = claimed - cap"),
or an *absence* claim ("no other clause of the policy mentions
'experimental'", checked by scanning the whole corpus). The Validation Agent
checks each assertion against the text of the cited chunk:

- **SUPPORTED** - the chunk contains what the claim asserts.
- **UNSUPPORTED** - it does not, or the chunk was never retrieved.
- **CONTRADICTED** - the chunk states a *different* figure, the claim's own
  arithmetic fails, or the final decision conflicts with its own findings
  (e.g. "admissible" while a verified exclusion or a deduction exists).

Verification is deterministic (no LLM), so it is reproducible and cheap.
It is measured, not assumed: the evaluation injects wrong chunks, wrong
numbers, tampered arithmetic and flipped decisions and reports how many are
caught (`eval_results/report.md`).

**Retry.** If any claim fails, the orchestrator loops back to the Policy
Evidence Agent once with retrieval widened 2x (a clause may simply have
ranked just outside the first top-k), then recomputes findings, decision and
validation. If it still fails, the decision is downgraded to `NEEDS_REVIEW`.
The loop, the reason for it, and each agent's reads/writes are recorded on
the state and shown in the UI's *Agent workflow* tab.

## 6. Unknown policy dimension (safety net)

The ten modelled dimensions cannot cover every rule in a policy. The policy
has specific exclusions the system does not model (dental, pregnancy,
spectacles, HIV, outpatient, listed chronic diseases, ...). An eleventh check,
`unmodelled_policy_risk`, looks at what the case is *about* (diagnosis and
procedure), retrieves the exclusions that name the same thing, and - if a
clause not already used by a modelled dimension matches - abstains with the
exact clause cited. It never decides a claim itself; it turns "silently
ignored" into "surfaced to a reviewer". Generic clinical words and treatment
modes are excluded from matching to keep false alarms low (see failure #10).

## 7. Reviewer support

The UI's reviewer panel is built from the same state: what needs attention
first (unsupported claims, unmodelled rules, missing evidence, retries), the
claim audit table with per-check detail and the cited excerpt, every
**assumption** the system made (e.g. "room limit applied per day of stay",
"network hospital presumed to meet the Hospital definition"), and a form to
record agree / override / escalate with notes as a downloadable JSON audit
record. Reviews are client-side; the API is stateless.

## 8. Trade-offs and known limitations

- **Eleven hand-designed dimensions, not a general reasoner.** The unknown-
  dimension check catches lexical overlap with a specific exclusion; it would
  miss a rule that applies for reasons not named in the diagnosis/procedure.
- **Sub-limit math rests on stated assumptions.** Room limit per day of stay;
  all room spend treated as normal-room (the input has no ICU split); the 75%
  package cap only when a package is stated. Each appears in the response's
  `assumptions`.
- **Regex-based number extraction is exact-text-dependent.** Extraction is
  clause-level and verified by the evidence checker, but a differently
  formatted policy would need its patterns reviewed.
- **Chunking limitation.** Exclusion items 18-20 are sub-items of 17
  (domiciliary-only) in the PDF but are chunked as top-level items (see
  `eval_results/failure_analysis.md` #11).
- **PARTIALLY_ADMISSIBLE vs ADMISSIBLE_WITH_LIMITS** is decided by whether a
  distinctly-named claim component (pre/post-hospitalization expenses on a
  confirmed window violation) is fully excluded, versus a proportional cap.
  This line is a documented judgment call, not dictated by the policy text.
- **Evaluation is the author's own.** Labels are the author's reading of the
  policy and 26 cases is small; see "What this evaluation does not prove" in
  `eval_results/report.md`.
