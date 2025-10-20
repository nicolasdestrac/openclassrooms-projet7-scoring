# ---- Base ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# OS deps (lightgbm & co)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY api/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Code + artefacts
COPY api ./api
COPY src ./src
COPY models ./models
COPY conf ./conf
COPY README.md ./README.md

# Démarrage API (FastAPI via Gunicorn + Uvicorn workers)
CMD ["sh","-c","gunicorn api.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
