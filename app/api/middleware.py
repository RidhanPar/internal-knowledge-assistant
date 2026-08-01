"""Request middleware: assign a request id and log each request.

Sets a short request id (from an inbound X-Request-ID header if present, else a
new one) into the logging context, records it on the response header, and logs
one line per request with the method, path, status, and how long it took. This
is the minimum tracing you want in a deployed service.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.context import request_id_var
from app.core.logging import get_logger

logger = get_logger("request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            # Log while the request id is still set on the context, so the access
            # line carries the same id as everything logged during the request.
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "%s %s -> %s in %dms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)
