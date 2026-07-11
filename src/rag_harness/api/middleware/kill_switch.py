"""Kill-switch middleware - return 503 for /query when the demo is off.

Set ``DEMO_ENABLED=false`` in the Cloud Run environment (or a local
``.env``) to shut off the demo instantly without redeploying. Health,
readiness, and metrics probes stay reachable so operators can still see
the state.
"""

import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rag_harness.config import settings

logger = logging.getLogger(__name__)


class KillSwitchMiddleware(BaseHTTPMiddleware):
    """Reject POST /query with 503 when ``settings.demo_enabled`` is False."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == "/query" and not settings.demo_enabled:
            logger.warning(
                "kill switch engaged - /query returning 503",
                extra={"error_type": "demo_disabled"},
            )
            body = {
                "error_type": "demo_disabled",
                "message": "The public demo is temporarily disabled.",
                "detail": None,
            }
            return Response(
                content=json.dumps(body),
                status_code=503,
                media_type="application/json",
            )
        return await call_next(request)
