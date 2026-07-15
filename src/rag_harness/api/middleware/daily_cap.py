"""Global daily-request-cap middleware for the public demo.

Consumes one slot from a shared :class:`DailyBudget` on every POST to
``/query``. When the day's slots are exhausted, returns HTTP 429 with a
``demo_daily_limit_reached`` body - the same shape as other structured
errors so the demo UI (and any client) can pattern-match uniformly.

Scoped to ``POST /query`` deliberately: ``/health``, ``/ready``, and
``/metrics`` must remain reachable for operators regardless of budget
state, and non-POST methods on ``/query`` are already 405 upstream.
"""

import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag_harness.api.budget import Budget

logger = logging.getLogger(__name__)


class DailyCapMiddleware(BaseHTTPMiddleware):
    """Enforce a global daily request cap on POST /query."""

    def __init__(self, app: object, budget: Budget) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._budget = budget

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method == "POST" and request.url.path == "/query":
            if not self._budget.check_and_increment():
                logger.warning(
                    "daily request cap reached - returning 429",
                    extra={"error_type": "demo_daily_limit_reached"},
                )
                body = {
                    "error_type": "demo_daily_limit_reached",
                    "message": (
                        "The demo has hit its daily request cap. Please try again after 00:00 UTC."
                    ),
                    "detail": None,
                }
                return Response(
                    content=json.dumps(body),
                    status_code=429,
                    media_type="application/json",
                )
        return await call_next(request)
