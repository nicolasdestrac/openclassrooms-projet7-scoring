# ---- Base ----
FROM python:3.10-slim

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# OS deps (si besoin pour lightgbm / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Installer les deps Python
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copier le code et les artefacts
COPY api ./api
COPY src ./src
COPY models ./models
COPY conf ./conf
COPY README.md ./README.md

# Lancement (gunicorn + uvicorn workers)
CMD ["sh", "-c", "gunicorn api.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
