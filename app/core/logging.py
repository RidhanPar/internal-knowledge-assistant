"""Structured-ish logging setup.

Kept intentionally small: a single configuration entrypoint so the API process
and the ingestion CLI log the same way. In a larger deployment this is where you
would wire JSON logging / correlation IDs.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Boto is chatty at INFO; keep it at WARNING unless we're debugging.
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
