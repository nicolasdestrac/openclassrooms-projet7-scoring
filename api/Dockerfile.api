# Dockerfile.api
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# code de l'API
COPY . /app


CMD ["sh","-c","gunicorn api.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
