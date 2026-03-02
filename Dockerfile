# ============================================================
# Multi-stage Dockerfile for the Financial Risk Intelligence Copilot
# ============================================================

FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies Stage ──
FROM base as dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application Stage ──
FROM dependencies as application

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application code
COPY config/ config/
COPY ml_engine/ ml_engine/
COPY rag_engine/ rag_engine/
COPY llm_layer/ llm_layer/
COPY evaluation/ evaluation/
COPY api/ api/
COPY utils/ utils/

# Create necessary directories
RUN mkdir -p artifacts data/raw data/processed data/regulatory_docs mlruns logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
