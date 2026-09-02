# VigilVoice — Production Container Image
# Multi-stage build: React/Vite frontend build -> FastAPI runtime.
# The compiled frontend is emitted into /app/static and served by FastAPI,
# so the existing single-container deployment (and /api/* contracts) are preserved.

# ── Stage 1: Build the React frontend ─────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /build/frontend

# Copy dependency specifications first for better Docker layer caching
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund

# Copy frontend source and build (vite outDir emitting to /build/static)
COPY frontend ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS audio/codec runtime dependencies (ffmpeg/libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend /app/backend
COPY models /app/models
COPY data /app/data

# Copy the compiled frontend (index.html + assets) served by FastAPI
COPY --from=frontend /build/static /app/static

# Security: Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/reports/incidents && \
    chown -R appuser:appuser /app

USER appuser

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

EXPOSE 8000

# Run uvicorn server — PORT is injected by Render/Railway/Fly at runtime
SHELL ["/bin/sh", "-c"]
CMD python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
