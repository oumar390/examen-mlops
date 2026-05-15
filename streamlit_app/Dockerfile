# Streamlit UI for the credit scoring API.
FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY streamlit_app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY streamlit_app/app.py /app/app.py
COPY streamlit_app/profiles.json /app/profiles.json

EXPOSE 8501
HEALTHCHECK CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "/app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
