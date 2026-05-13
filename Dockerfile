FROM python:3.11-slim

WORKDIR /app

RUN useradd -r -u 1001 appuser

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir -r requirements.txt

USER appuser
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/best_model.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "scoring_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
