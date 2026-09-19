"""Concrete LLM providers + factory."""
from __future__ import annotations

import logging

import httpx

from ..config import settings
from .base import LLMClient

logger = logging.getLogger(__name__)

_RATIONALE_SYSTEM = (
    "You rewrite pre-computed insurance claim findings into one concise, reviewer-facing paragraph. "
    "You must not add any fact, number, or policy interpretation that is not already present in the "
    "findings you are given."
)


class OfflineLLMClient:
    """No external calls. Everything defers to the deterministic pipeline."""

    def generate_rationale(self, prompt: str) -> str | None:
        return None

    def interpret(self, system: str, user: str) -> str | None:
        return None


class OpenAICompatibleClient:
    """Any OpenAI-compatible chat-completions endpoint (Gemini, OpenAI, Groq, Together, Ollama, ...).

    Configured entirely through environment variables so switching provider
    needs no code change.
    """

    def __init__(self, api_key: str, base_url: str, model: str, interpretation_model: str | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # The interpretation step is judgement-heavy, so it may use a stronger model than rationale phrasing.
        self.interpretation_model = interpretation_model or model

    def _chat(self, system: str, user: str, max_tokens: int, timeout: float, json_mode: bool = False,
              model: str | None = None) -> str | None:
        if not self.api_key:
            return None
        body: dict = {
            "model": model or self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = httpx.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                              json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 - any provider failure must degrade gracefully
            logger.warning("LLM call failed, falling back to the deterministic pipeline: %s", exc)
            return None

    def generate_rationale(self, prompt: str) -> str | None:
        return self._chat(_RATIONALE_SYSTEM, prompt, max_tokens=350, timeout=20.0)

    def interpret(self, system: str, user: str) -> str | None:
        # generous budget: reasoning models spend part of it on thinking before the JSON
        return self._chat(system, user, max_tokens=6000, timeout=120.0, json_mode=True, model=self.interpretation_model)


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(settings.openai_api_key, settings.openai_base_url, settings.openai_model,
                                      settings.openai_interpretation_model)
    return OfflineLLMClient()
