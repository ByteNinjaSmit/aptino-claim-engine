"""Concrete LLM providers + factory."""
from __future__ import annotations

import logging

import httpx

from ..config import settings
from .base import LLMClient

logger = logging.getLogger(__name__)


class OfflineLLMClient:
    """No external calls. Always defers to the deterministic template."""

    def generate_rationale(self, prompt: str) -> str | None:
        return None


class OpenAICompatibleClient:
    """Works with any OpenAI-compatible chat completions endpoint

    (Groq, OpenAI, Together, local Ollama via its OpenAI-compat route, ...).
    Configured entirely through environment variables so no code change is
    needed to switch providers.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate_rationale(self, prompt: str) -> str | None:
        if not self.api_key:
            return None
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You rewrite pre-computed insurance claim findings into one "
                                "concise, reviewer-facing paragraph. You must not add any "
                                "fact, number, or policy interpretation that is not already "
                                "present in the findings you are given."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 350,
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 - any provider failure must degrade gracefully
            logger.warning("LLM rationale generation failed, falling back to template: %s", exc)
            return None


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(settings.openai_api_key, settings.openai_base_url, settings.openai_model)
    return OfflineLLMClient()
