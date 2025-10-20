FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Dépendances Python de l'app Streamlit
COPY streamlit_app/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# On ne copie QUE l'app Streamlit
COPY streamlit_app ./streamlit_app

# Streamlit écoute sur $PORT, 0.0.0.0 pour Render
CMD ["sh","-c","streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0"]
