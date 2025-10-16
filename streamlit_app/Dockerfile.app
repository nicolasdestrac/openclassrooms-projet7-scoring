# Dockerfile.app
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501

WORKDIR /app
COPY streamlit_app/requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_app ./streamlit_app

# Passer l’URL de l’API via une variable d'env (ex: API_URL=https://ton-api.onrender.com)
ENV API_URL=""
CMD ["sh","-c","streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0"]
