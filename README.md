# Aptino Claim Decision Engine

Policy-aware multi-agent RAG system that decides health-insurance claim
admissibility against a supplied policy PDF (Universal Sompo CSC —
Individual Health Insurance, `UNIHLIP18004V011718`), with hybrid retrieval,
inspectable citations, and explicit abstention when the evidence doesn't
support a safe decision.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design note (agent
boundaries, state flow, retrieval design, trade-offs).

## 1. Architecture at a glance

```
Case Analysis Agent → Policy Evidence Agent (hybrid retrieval + rerank)
   → Coverage & Exclusion Agent → Decision Agent → Validation Agent
```

Agents exchange a single typed state object end-to-end (see
`src/aptino_claims/agents/state.py`); nothing is passed between them as
free-form text. Full detail in `ARCHITECTURE.md`.

## 2. Repository layout

```
data/
  policy/                  supplied policy PDF
  candidate_data/          supplied 12 public test cases (unmodified)
  custom_cases/            5+ candidate-authored test cases
  index/                   generated: chunks.jsonl, dense_vectors.npy, chunk_meta.json
src/aptino_claims/
  ingestion/               PDF loading + structure-aware chunking
  retrieval/               dense/sparse/fusion/rerank + HybridRetriever
  agents/                  state models, dimension registry, 5 agents, orchestrator
  llm/                     pluggable LLM client (offline-safe by default)
  api/                     FastAPI app (/analyze, /health)
  eval/                    expected outcomes + reproducible evaluation script
frontend/streamlit_app.py  reviewer console
tests/                     pytest unit + API tests
eval_results/              metrics.json, report.md, failure_analysis.md
```

## 3. Setup (local)

Requires Python 3.12+ (pinned numpy 2.5 needs it).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # defaults run fully offline, no API key needed

# Build the policy index (chunk + embed + cache); the API also does this
# automatically on first run if the index is missing.
python -m aptino_claims.ingestion.build_index

# Run the API
uvicorn aptino_claims.api.main:app --reload --port 8000

# In a second terminal: run the frontend
API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

First run downloads two small local models (~130MB dense embedder + ~280MB
reranker, both from Hugging Face, both cached under `~/.cache/fastembed`
afterward) — no API key is required for retrieval.

By default `LLM_PROVIDER=offline`: the admissibility decision itself is
always deterministic and evidence-grounded (see `ARCHITECTURE.md` §4) and
never needs an LLM. To let the Decision Agent additionally phrase the
rationale through a real LLM, set in `.env`:

```
LLM_PROVIDER=openai_compatible
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.groq.com/openai/v1   # or api.openai.com/v1, etc.
OPENAI_MODEL=llama-3.1-8b-instant
```

## 4. API

### `GET /health`
```json
{"status": "ok", "chunk_count": 116, "llm_provider": "offline"}
```

### `POST /analyze`
Request body — see `data/claim_case_schema.md` / `data/candidate_data/public_test_cases.json`
for full examples; unknown/non-critical fields are tolerated, not rejected.

```json
{
  "case_id": "PUB-001",
  "policy_start_date": "2025-01-01",
  "claim_date": "2026-03-14",
  "sum_insured_inr": 500000,
  "continuous_coverage_months": 14,
  "prior_insurer_continuous_years": 0,
  "patient": {"age": 34},
  "hospital": {"name": "Sunrise Multispeciality", "network_provider": true},
  "treatment": {"type": "inpatient", "admission_hours": 96, "diagnosis": "Acute appendicitis",
                "procedure": "Appendectomy", "pre_existing": false, "experimental": false},
  "expenses_inr": {"room": 30000, "doctor_fees": 30000, "medicines_diagnostics": 90000,
                    "pre_hospitalization": 5000, "post_hospitalization": 7000, "ambulance": 1200},
  "documents": ["claim_form", "discharge_summary", "itemized_bill", "doctor_prescription"]
}
```

Response (trimmed):
```json
{
  "case_id": "PUB-001",
  "decision": "ADMISSIBLE_WITH_LIMITS",
  "confidence": 0.8,
  "key_findings": ["..."],
  "applicable_limits": [{"description": "Room/boarding/nursing sub-limit (...)", "deduction_inr": 10000.0, "dimension": "category_sub_limits"}],
  "missing_evidence": [{"field": "expense_timing", "reason": "..."}],
  "citations": [{"claim": "...", "source": "policy.pdf", "page": 7, "section": "What We Cover", "chunk_id": "what_we_cover-002", "rerank_score": 1.2}],
  "validation": {"status": "PASS", "unsupported_claims": []},
  "rationale": "...",
  "trace": [{"agent": "CaseAnalysisAgent", "action": "extract_facts_and_plan", "detail": "...", "elapsed_ms": 0.3, "retrieval_count": null}]
}
```

Malformed requests (missing required fields, wrong types) return `422`
with FastAPI's standard validation error body. Unexpected internal errors
return `500` with `{"error": "internal_error", ...}` — never a raw
stack trace.

## 5. Evaluation

```bash
python -m aptino_claims.eval.run_eval
```

Runs all 12 supplied public cases plus all 5+ candidate cases through the
full pipeline and writes `eval_results/metrics.json` +
`eval_results/report.md`, reporting:

- **Decision accuracy** against hand-labeled expected outcomes (see
  `src/aptino_claims/eval/expected_outcomes.py`, which documents how each
  label was derived from the policy text).
- **Retrieval evidence hit rate** — did retrieval surface a chunk
  containing the on-topic policy language for each case's primary
  dimension.
- **Citation validation pass rate** — the Validation Agent's citation/claim
  cross-check (see `ARCHITECTURE.md` §5).
- **Required-abstention check** — confirms the cases the assignment
  requires to resolve as `NEEDS_REVIEW` actually do.

See `eval_results/failure_analysis.md` for documented failure cases, root
causes, and the fixes applied.

**Current results** (`eval_results/report.md`, 12 supplied + 6 candidate cases):

| Metric | Result |
|---|---|
| Decision accuracy | 18/18 (100%) |
| Citation validation pass rate | 100% |
| Retrieval evidence hit rate | 100% |
| Required NEEDS_REVIEW cases correctly abstained | 5/5 |

## 6. Tests

```bash
pytest
```

## 7. Deployment

Fully Dockerized. Every push to `main` triggers GitHub Actions
(`.github/workflows/deploy.yml`) to build the backend and frontend images,
push them to Docker Hub, and roll them out on a VPS with
`docker compose` (`docker-compose.yml`). No GPU required (all retrieval
models are small ONNX models run on CPU). See [`DEPLOY.md`](DEPLOY.md).

- API: `http://<vps>:8000`
- Frontend: `http://<vps>:8501`

Run the same stack locally: `DOCKERHUB_USERNAME=local docker compose up --build`.

## 8. Known limitations

See `ARCHITECTURE.md` §6.
