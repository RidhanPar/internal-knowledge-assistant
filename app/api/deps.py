"""FastAPI dependencies.

The pool, repository, and RAG service are created once at startup and stored on
`app.state` (see main.py). These providers just hand them to route functions,
which keeps handlers free of construction logic and trivially overridable in
tests.
"""

from __future__ import annotations

from fastapi import Request

from app.db.repository import Repository
from app.rag.service import RagService


def get_repository(request: Request) -> Repository:
    return request.app.state.repository


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service
