# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build.
#  - builder: install all Python deps into an isolated virtualenv.
#  - runtime: copy just the venv + app onto a clean slim base (no pip cache,
#    no build toolchain, no bytecode) → smaller, reproducible, easy to scan.
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps live ONLY in this stage; they never reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Trim the venv: bytecode caches and pip's own metadata aren't needed at runtime.
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name '*.pyc' -delete

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# OCI labels so the image is traceable back to the project/source/version.
ARG APP_VERSION=dev
LABEL org.opencontainers.image.title="eks-kit-rag-demo" \
      org.opencontainers.image.description="RAG demo for the EKS Production Kit: hybrid retrieval (pgvector + BM25/RRF), incremental indexing, bring-your-own-LLM answers." \
      org.opencontainers.image.source="https://github.com/smobinulla/eks-production-kit" \
      org.opencontainers.image.version="${APP_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE=/app/models \
    APP_VERSION=${APP_VERSION}

# Runtime-only system deps: libgomp1 (onnxruntime/fastembed), curl (HEALTHCHECK).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app

# Non-root with a writable model cache (matches the kit's security ethos).
# The embedding model (~130MB) downloads into /app/models on first startup.
RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-native health signal (compose/orchestrators read this).
HEALTHCHECK --interval=15s --timeout=3s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
