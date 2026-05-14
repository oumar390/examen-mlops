# MLflow server image with PostgreSQL driver.
# The official mlflow image ships without psycopg2, so we add it here.
FROM ghcr.io/mlflow/mlflow:v2.17.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.10

# Healthcheck endpoint is /health on the mlflow server.
# We use python (always present) since curl is not shipped in the base image.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=3).status == 200 else 1)" || exit 1
