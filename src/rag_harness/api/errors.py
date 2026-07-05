"""Typed exception hierarchy for the FastAPI service.

Every user-facing error path raises one of these concrete subclasses so
the FastAPI exception handler in ``server.py`` can map it to:

- the correct HTTP status code
- a structured JSON response body with ``error_type``, ``message``, and
  ``detail`` fields
- a structured log event with matching fields (via ``logger.warning`` with
  ``extra=``) so ops can filter on error type across the log stream

Nothing here catches raw ``Exception`` — that stays 500 Internal Server
Error and surfaces the traceback in logs so a genuine bug isn't hidden.
"""

from dataclasses import dataclass


@dataclass
class RagHarnessError(Exception):
    """Base class for every user-facing API error.

    Subclasses set :attr:`status_code` and override :attr:`error_type`.
    """

    message: str
    detail: str | None = None
    status_code: int = 500
    error_type: str = "internal_error"

    def __str__(self) -> str:
        return self.message


class GuardrailRejection(RagHarnessError):
    """Request rejected at the input-guardrail boundary."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=422,
            error_type="guardrail_rejection",
        )


class RetrievalError(RagHarnessError):
    """Retrieval boundary failed (ChromaDB down, vector store unreachable, etc.)."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=503,
            error_type="retrieval_error",
        )


class GenerationError(RagHarnessError):
    """LLM boundary failed after retries (used only when refusal path is unsuitable)."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=503,
            error_type="generation_error",
        )


class NotReadyError(RagHarnessError):
    """A dependency (ChromaDB, LLM key, reranker) is not reachable at startup or check."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(
            message=message,
            detail=detail,
            status_code=503,
            error_type="not_ready",
        )
