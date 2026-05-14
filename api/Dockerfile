# =============================================================================
# Credit Scoring API — production image
# =============================================================================
FROM python:3.11.9-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps required by xgboost (libgomp1) and pandas wheel install
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so layers cache when only source changes
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# Copy application source + data prep module + trained model
COPY api/ /app/api/
COPY src/ /app/src/
COPY models/best_model.joblib /app/models/best_model.joblib
COPY models/metadata.json /app/models/metadata.json

# Non-root user
RUN useradd --create-home --shell /bin/bash apiuser \
    && chown -R apiuser:apiuser /app
USER apiuser

ENV MODEL_PATH=/app/models/best_model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    PYTHONPATH=/app/src

EXPOSE 8000

# Render sets $PORT; default to 8000 locally
ENV PORT=8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\", \"8000\")}/health', timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
