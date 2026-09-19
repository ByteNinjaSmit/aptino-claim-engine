"""Central configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(env_key: str, default: str) -> Path:
    value = os.getenv(env_key, default)
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _int(env_key: str, default: int) -> int:
    return int(os.getenv(env_key, str(default)))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "offline"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gemini-2.5-flash-lite"))
    openai_interpretation_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_INTERPRETATION_MODEL") or os.getenv("OPENAI_MODEL", "gemini-2.5-flash-lite")
    )
    llm_interpretation: bool = field(
        default_factory=lambda: os.getenv("LLM_INTERPRETATION", "off").strip().lower() in ("on", "true", "1", "yes")
    )

    policy_pdf_path: Path = field(
        default_factory=lambda: _path(
            "POLICY_PDF_PATH", "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
        )
    )
    index_dir: Path = field(default_factory=lambda: _path("INDEX_DIR", "data/index"))

    dense_model: str = field(default_factory=lambda: os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"))

    top_k_dense: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K_DENSE", 15))
    top_k_sparse: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K_SPARSE", 15))
    top_k_fused: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K_FUSED", 10))
    top_k_final: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K_FINAL", 5))

    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _int("API_PORT", 8000))


settings = Settings()
