"""Shared AsyncOpenAI client factory with retries and timeouts configured.

Every module that needs an AsyncOpenAI client should build one via
``build_async_client()`` so the retry policy and timeout are consistent
across the codebase and cheap to tune from ``settings``.

The SDK's built-in retry logic already applies exponential backoff with
jitter and only retries on the errors that should be retried (429, 5xx,
connection errors). We just set the caps.
"""

from openai import AsyncOpenAI

from rag_harness.config import settings


def build_async_client(api_key: str | None = None) -> AsyncOpenAI:
    """Return an ``AsyncOpenAI`` with the configured retries and timeout.

    Callers typically don't pass *api_key* — it defaults to
    ``settings.openai_api_key``. Tests may override.
    """
    return AsyncOpenAI(
        api_key=api_key or settings.openai_api_key,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout_seconds,
    )
