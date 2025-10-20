FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . /app

# Streamlit écoute sur $PORT, 0.0.0.0 pour Render
CMD ["sh","-c","streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0"]
