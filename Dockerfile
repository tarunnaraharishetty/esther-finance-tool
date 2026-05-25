# syntax=docker/dockerfile:1.7
#
# Multi-stage container for Esther.
#
# Stage 1: build the Vite frontend (web/) into web/dist/ — node only
#          lives in the build image, never in the runtime layer.
# Stage 2: install Python deps + copy backend source + built frontend.
#          Runs as a non-root user, binds to the port supplied by the
#          deploy host via $PORT (Railway / Render / Fly all use this
#          convention).
#
# Image size is dominated by torch + transformers (FinBERT). We pin
# python:3.12-slim rather than alpine because torch's manylinux wheels
# fail to install against musl.

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=20

# ---------------------------------------------------------------------
# 1. Frontend build — Node + npm
# ---------------------------------------------------------------------
FROM node:${NODE_VERSION}-slim AS frontend-build

WORKDIR /app/web
# Copy manifests first so docker layer-caches the install whenever
# package.json / package-lock.json are unchanged.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------------
# 2. Runtime — Python 3.12 + backend + bundled frontend
# ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# System deps required by transformers / torch wheels at install time.
# Kept narrow on purpose — every line here lands in the runtime layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root. UID 10001 is high enough to avoid colliding with
# common host accounts when the image is run with a bind-mounted
# data volume.
RUN useradd --create-home --shell /bin/bash --uid 10001 esther
WORKDIR /app

# Install Python deps first so they stay cached when only source
# changes. Copy only the install manifests, not the full source.
COPY pyproject.toml README.md ./
COPY src/__init__.py ./src/__init__.py
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -e .

# Copy backend source.
COPY src/ ./src/
COPY config/ ./config/

# Copy the built frontend assets from the build stage.
COPY --from=frontend-build /app/web/dist/ ./web/dist/

# Data + logs directories so the writable paths exist at runtime even
# before the first file lands. The default $PROJECT_ROOT-relative
# settings point here — operators wanting external persistence should
# bind-mount over these paths.
RUN mkdir -p /app/data /app/logs && chown -R esther:esther /app

USER esther

# Containerised deployments must explicitly opt out of the default dev
# session key + flip secure cookies on. Both checks live in the
# Settings model validator and refuse to start in non-dev mode without
# proper overrides — keep these set in the deploy env, not baked here.
ENV APP_ENV=prod \
    LOG_LEVEL=INFO \
    LOG_JSON=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check uses the cheap /api/health endpoint (no controller
# fetch). 30s grace period for FinBERT warm-up on first request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:${PORT:-8000}/api/health || exit 1

EXPOSE 8000

# Shell form so $PORT expands at runtime. ``esther serve`` is the
# canonical entry point and binds the FastAPI app.
CMD esther serve --host 0.0.0.0 --port ${PORT:-8000}
