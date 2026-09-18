"""LLM client abstraction.

Admissibility logic itself is deterministic and evidence-grounded (see
`agents/dimensions.py`) -- an LLM is never allowed to invent a policy
conclusion. The LLM client is used only to (optionally) rephrase an
already-computed, already-cited finding set into a fluent rationale. If no
provider is configured, or the call fails, or the LLM output is not backed
by the supplied findings, the caller falls back to a deterministic
templated rationale. This keeps the system correct-by-construction even
with zero API keys configured (LLM_PROVIDER=offline).
"""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def generate_rationale(self, prompt: str) -> str | None:
        """Return a rationale string, or None if unavailable/unusable."""
        ...
