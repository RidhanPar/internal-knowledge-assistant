"""Application errors and their HTTP handlers.

Two goals:

1. Turn failures into clean JSON the caller can act on, with a stable shape, a
   machine-readable code, and the request id for support. Never leak a stack
   trace or internal detail to the client.
2. Separate expected upstream failures (Bedrock or the database being
   unreachable or throttling) from unexpected bugs, so the first return a 503
   the caller can retry and the second return a 500.

The exception classes have no web framework imports, so the rag and ingestion
layers can raise them without depending on FastAPI. The handlers are registered
onto the app in `register_exception_handlers`.
"""

from __future__ import annotations

from app.core.context import request_id_var
from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base for errors we deliberately turn into an HTTP response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UpstreamError(AppError):
    """A dependency we call (Bedrock, the database) failed or was unreachable.

    Returned as 503 because retrying later may succeed, unlike a bug in our code.
    """

    status_code = 503
    code = "upstream_unavailable"

    def __init__(self, dependency: str, message: str) -> None:
        super().__init__(message)
        self.dependency = dependency


def register_exception_handlers(app) -> None:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    def _body(code: str, message: str) -> dict:
        return {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        # Log the detail server-side; return a safe message to the client.
        logger.warning("app_error code=%s dependency=%s: %s", exc.code,
                       getattr(exc, "dependency", "-"), exc.message)
        client_message = (
            f"A dependency ({exc.dependency}) is unavailable. Please retry."
            if isinstance(exc, UpstreamError)
            else exc.message
        )
        return JSONResponse(status_code=exc.status_code, content=_body(exc.code, client_message))

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_body("internal_error", "An internal error occurred."),
        )
