"""Sanitising of case-derived free text before it is echoed into findings or LLM prompts.

Claim fields (diagnosis, procedure, hospital name) are untrusted input. They
end up in finding statements, and finding statements feed the optional
rationale-phrasing prompt, so markup and control characters are stripped and
length is capped. This narrows (it cannot eliminate) instruction-style text
hidden in a claim; decisions and amounts are protected structurally instead
(see `decision_agent` and `llm_interpretation`).
"""
from __future__ import annotations

import re

_UNSAFE = re.compile(r"[\x00-\x1f\x7f<>`]")
_WS = re.compile(r"\s+")


def clean(text: object, cap: int = 200) -> str:
    return _WS.sub(" ", _UNSAFE.sub(" ", str(text or ""))).strip()[:cap]
