"""Minimal prompt-injection screening at the /query boundary.

Scope — deliberately narrow. This module is NOT a full guardrails engine.
It is a boundary check that catches the most common, low-effort injection
patterns. Sophisticated adversarial inputs will still get through; this
is a hygiene layer, not an authorization boundary.

A proper guardrails engine (semantic classifier, taxonomy of intents,
policy evaluation) is a separate future project. Explicitly out of scope
for the rag-harness codebase per Phase 9 in docs/ROADMAP.md.

Detection is regex-based against a handful of well-known patterns
(``ignore previous instructions``, ``system prompt``, ``you are now``,
``disregard the above``, ``forget your instructions``). Reasoning: cheap
CPU-only screening at the request boundary is a fast fail; anything more
sophisticated has to weigh false positives against real user queries and
belongs in the future engine.
"""

import re

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(previous|prior|all|the\s+above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(previous|prior|all|the\s+above|your)\s+", re.IGNORECASE),
    re.compile(r"\bforget\s+(your|all|the\s+above|previous|prior)\s+", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+\w+", re.IGNORECASE),
    re.compile(r"\b(system|assistant|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"<\s*(system|assistant|user|developer)\s*>", re.IGNORECASE),
]


def screen_for_injection(text: str) -> str | None:
    """Return a short reason string if *text* looks like prompt injection.

    Returns ``None`` when no known pattern matches.
    """
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return f"prompt-injection pattern matched: {match.group(0)!r}"
    return None
