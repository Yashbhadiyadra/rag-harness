"""Security response headers on every route.

Browser-facing hardening for the public demo (the API itself is consumed
by non-browser clients, but the demo UI at ``/`` is a real web page and
the headers cost nothing on JSON responses):

- ``X-Content-Type-Options: nosniff`` - stop MIME sniffing.
- ``X-Frame-Options: DENY`` - the demo UI has no legitimate embedding
  use case, so deny framing outright (clickjacking).
- ``Strict-Transport-Security`` - one year; Cloud Run terminates TLS so
  every production request is HTTPS already, this pins browsers to it.
  Harmless over plain-HTTP local dev (browsers ignore HSTS on HTTP).
- ``Content-Security-Policy`` - strict same-origin. The demo UI loads
  only ``/static`` assets and calls ``/query`` on the same origin; there
  are no inline scripts or styles, so ``'self'`` needs no carve-outs.
  ``frame-ancestors 'none'`` is the CSP-level mirror of X-Frame-Options.
- ``Referrer-Policy: no-referrer`` - query text must never leak into a
  third-party Referer header.

The values are constants, not settings: there is no deployment of this
service where weaker headers are correct, so making them configurable
would only invite drift.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the security header set to every response, including errors."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)
        for name, value in _HEADERS.items():
            response.headers[name] = value
        return response
