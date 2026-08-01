"""Error handling and request middleware, tested on a minimal app.

We build a bare FastAPI app (not the real one) so these tests need no database
or Bedrock. They check the response contract: upstream failures become a 503 with
a retriable message, unexpected errors become a 500 that leaks nothing, and every
response carries a request id.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import RequestContextMiddleware
from app.core.errors import UpstreamError, register_exception_handlers


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/boom-upstream")
    async def boom_upstream():
        raise UpstreamError("bedrock-generation", "throttled: too many requests")

    @app.get("/boom-bug")
    async def boom_bug():
        raise ValueError("secret internal detail")

    return app


def test_ok_route_sets_request_id_header():
    client = TestClient(_app())
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID")


def test_inbound_request_id_is_echoed():
    client = TestClient(_app())
    resp = client.get("/ok", headers={"X-Request-ID": "trace-123"})
    assert resp.headers.get("X-Request-ID") == "trace-123"


def test_upstream_error_returns_503_and_hides_detail():
    client = TestClient(_app())
    resp = client.get("/boom-upstream")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "upstream_unavailable"
    assert "bedrock-generation" in body["error"]["message"]
    # The raw upstream detail must not leak to the client.
    assert "throttled" not in body["error"]["message"]
    assert body["error"]["request_id"]


def test_unexpected_error_returns_500_and_leaks_nothing():
    client = TestClient(_app(), raise_server_exceptions=False)
    resp = client.get("/boom-bug")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert "secret internal detail" not in str(body)
