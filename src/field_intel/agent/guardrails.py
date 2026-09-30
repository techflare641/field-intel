"""Input/output guardrails for the agent loop.

* ``redact_pii``: strip emails, phone numbers, and SSN-shaped strings from *inbound* user
  text before it reaches any LLM provider (grower questions sometimes paste contacts).
* ``sanitize_tool_output``: tool results are data, not instructions. Field names and
  reasons are user-editable strings, so we neutralise common prompt-injection phrasing
  and control characters before they are echoed back to the model.
* ``truncate``: cap tool result size so a large table cannot blow the context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions?", re.I),
    re.compile(r"disregard (?:all |any )?(?:previous|prior|above) instructions?", re.I),
    re.compile(r"you are now (?:an?|the) ", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"<\s*/?\s*(?:system|assistant|tool|instructions?)\s*>", re.I),
]
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True, slots=True)
class Redaction:
    text: str
    count: int


def redact_pii(text: str) -> Redaction:
    count = 0
    for pattern, token in ((_EMAIL, "[email]"), (_SSN, "[ssn]"), (_PHONE, "[phone]")):
        text, n = pattern.subn(token, text)
        count += n
    return Redaction(text=text, count=count)


def sanitize_tool_output(text: str) -> str:
    text = _CONTROL_CHARS.sub("", text)
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[removed]", text)
    return text


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = f"... [truncated {len(text) - max_chars} chars]"
    return text[: max(0, max_chars - len(marker))] + marker, True
