"""LLM client abstraction.

Admissibility logic is deterministic and evidence-grounded (see
`agents/dimensions.py`); an LLM is never allowed to invent a policy
conclusion. It is used in two bounded ways:

* `generate_rationale` - rephrase an already-computed, already-cited finding
  set into one paragraph (checked for unsupported figures before use).
* `interpret` - optional reading of the retrieved policy chunks to spot
  clauses the rule-based checks did not model. Its output is untrusted: each
  observation must quote a retrieved chunk verbatim, is verified, and can only
  make the outcome more cautious (see `agents/llm_interpretation.py`).

If no provider is configured or a call fails, both return None and the
system behaves exactly as the deterministic pipeline (LLM_PROVIDER=offline).
"""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def generate_rationale(self, prompt: str) -> str | None:
        """Return a rationale string, or None if unavailable/unusable."""
        ...

    def interpret(self, system: str, user: str) -> str | None:
        """Return the model's raw (expected JSON) reply, or None if unavailable."""
        ...
