"""FastAPI backend: POST /analyze, GET /health."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..agents.orchestrator import analyze_case
from ..config import settings
from ..llm.providers import get_llm_client
from ..retrieval.retriever import HybridRetriever
from .schemas import ClaimCaseIn, HealthOut

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        from ..ingestion.build_index import main as build_index_main

        build_index_main()
    _state["retriever"] = HybridRetriever.load_or_build(settings.index_dir, chunks_path)
    _state["llm"] = get_llm_client()
    try:
        # Load the embedding + reranker models now rather than on the first user request,
        # so /health only reports "ok" once the service can actually answer quickly.
        _state["retriever"].search("warm-up query")
    except Exception:  # noqa: BLE001 - a warm-up failure must not stop the service starting
        logger.exception("Model warm-up failed; the first request will load the models instead")
    yield
    _state.clear()


app = FastAPI(title="Aptino Claim Decision Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "The request could not be processed."})


@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    retriever: HybridRetriever | None = _state.get("retriever")
    return HealthOut(
        status="ok" if retriever is not None else "starting",
        chunk_count=len(retriever.chunk_meta) if retriever else 0,
        llm_provider=settings.llm_provider,
    )


@app.post("/analyze")
async def analyze(case: ClaimCaseIn, llm_interpretation: bool | None = Query(
        default=None, description="Run the optional LLM interpretation step (true), skip it (false), or follow the server setting (omitted). Needs a configured LLM provider; adds ~10 s."),
) -> dict:
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retrieval index is still initializing; retry shortly.")
    try:
        state = analyze_case(case.model_dump(), retriever, _state["llm"], interpret=llm_interpretation)
    except Exception as exc:  # defensive: never let a case-specific failure surface a stack trace
        logger.exception("Analysis failed for case %s", case.case_id)
        raise HTTPException(status_code=500, detail=f"Internal error while analyzing case {case.case_id}.") from exc
    return state.to_response()
