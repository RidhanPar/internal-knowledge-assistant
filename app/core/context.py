"""Request-scoped context.

`request_id_var` holds a short id for the current request. The logging filter
reads it so every log line emitted while handling a request carries the same id,
which is what lets you trace one request through the logs.
"""

from __future__ import annotations

from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
