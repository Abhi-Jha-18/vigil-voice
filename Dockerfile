# Stage 1: Build the frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
# Install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Build the app
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Container Image (Phase 14)
# Lightweight Python 3.11 build for secure CPU execution
FROM python:3.11-slim as base

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
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist
COPY models /app/models
COPY data /app/data

# Security: Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Healthcheck configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

EXPOSE 8000

# Run uvicorn server — PORT is injected by Render/Railway/Fly at runtime
SHELL ["/bin/sh", "-c"]
CMD python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
