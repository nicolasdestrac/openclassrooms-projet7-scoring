# streamlit_app/app.py
import os
import requests
import streamlit as st
import pandas as pd

def get_api_url():
    # 1) Render/locaux via env var
    url = os.getenv("API_URL")
    if url:
        return url.rstrip("/")
    # 2) Compat Streamlit Cloud (secrets.toml) si jamais tu en as un
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"]).rstrip("/")
    except Exception:
        pass
    # 3) fallback dev local
    return "http://127.0.0.1:8000"

API_URL = get_api_url()

st.title("Projet 7 — Scoring")
st.caption(f"API: {API_URL}")

# petit check santé
if st.button("Ping /health"):
    r = requests.get(f"{API_URL}/health", timeout=10)
    st.write(r.status_code, r.text)

# mini formulaire
with st.form("score"):
    amt_income = st.number_input("AMT_INCOME_TOTAL", value=180000, step=1000)
    amt_credit = st.number_input("AMT_CREDIT", value=450000, step=1000)
    submitted = st.form_submit_button("Prédire")

if submitted:
    payload = {"features": {"AMT_INCOME_TOTAL": amt_income, "AMT_CREDIT": amt_credit}}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    st.json(r.json())
