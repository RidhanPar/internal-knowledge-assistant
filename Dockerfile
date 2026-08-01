# Multi-stage build. The builder installs dependencies into a virtualenv, and
# the final image copies only that venv plus the application code, so the runtime
# image stays small and contains no build tools.

FROM python:3.11-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.11-slim AS runtime
# Run as a non-root user. Containers should not run as root in production.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JSON_LOGS=true

COPY --from=builder /opt/venv /opt/venv
# Copy the code and the data the service needs at runtime. The same image also
# runs the one-off ingestion job (python -m scripts.ingest), so the corpus and
# scripts are included.
COPY app ./app
COPY scripts ./scripts
COPY corpus ./corpus
COPY data ./data
COPY db ./db
COPY eval ./eval

USER appuser
EXPOSE 8080

# App Runner and most platforms inject PORT. Fall back to 8080 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
