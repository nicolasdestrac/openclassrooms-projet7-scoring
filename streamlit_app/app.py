# streamlit_app/app.py
import os, json, io
from typing import List, Dict

import requests
import pandas as pd
import streamlit as st

# -----------------------
# Config / helpers
# -----------------------
API_URL = st.secrets.get("API_URL") or os.getenv("API_URL", "").strip()
TOP_FEATURES_SECRET = (st.secrets.get("TOP_FEATURES") or os.getenv("TOP_FEATURES", "")).strip()

st.set_page_config(page_title="Projet 7 — Scoring", layout="centered")
st.title("Projet 7 — Scoring")
if not API_URL:
    st.error("API_URL manquant (Secrets `.streamlit/secrets.toml` ou variable d'environnement).")
    st.stop()
st.caption(f"API: {API_URL}")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_json(url: str) -> Dict:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def get_schema() -> List[str]:
    try:
        js = fetch_json(f"{API_URL}/schema")
        cols = list(js.get("input_columns", []))
        return cols
    except Exception:
        return []

@st.cache_data(ttl=120, show_spinner=False)
def get_health() -> Dict:
    try:
        return fetch_json(f"{API_URL}/health")
    except Exception:
        return {}

schema = get_schema()
health = get_health()

with st.expander("État API / Schéma", expanded=False):
    st.json(health or {"status": "unknown"})
    if schema:
        st.write(f"**{len(schema)} colonnes attendues**")
        st.code("\n".join(schema))

def pick_top_features(all_cols: List[str], k: int = 10) -> List[str]:
    if TOP_FEATURES_SECRET:
        want = [c.strip() for c in TOP_FEATURES_SECRET.split(",") if c.strip()]
        return [c for c in want if c in all_cols] or want
    if not all_cols:
        return []
    # heuristique “quick & dirty” : colonnes monétaires / jours / scores / target-like
    key_words = ("AMT", "DAYS", "CREDIT", "INCOME", "EXT_SOURCE", "AGE", "SCORE", "AMT_")
    ranked = sorted(all_cols, key=lambda c: any(k in c.upper() for k in key_words), reverse=True)
    # garde l’ordre d’origine mais privilégie les colonnes “suspectes”
    seen, out = set(), []
    for c in ranked + all_cols:
        if c not in seen:
            out.append(c); seen.add(c)
        if len(out) >= k:
            break
    return out

top_cols = pick_top_features(schema, k=10)

def is_numeric_name(col: str) -> bool:
    cu = col.upper()
    return any(tag in cu for tag in ("AMT", "DAYS", "CNT", "HOUR", "YEARS", "RATIO", "SCORE"))

def call_api(endpoint: str, payload: Dict) -> Dict:
    r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------
# UI
# -----------------------
tab_simple, tab_json, tab_csv = st.tabs(["🧩 Simple", "💻 JSON avancé", "📄 CSV (1 ligne)"])

with tab_simple:
    st.write("Renseigne quelques variables utiles. Les colonnes manquantes seront imputées par le pipeline.")
    if not top_cols:
        st.info("Le schéma n'a pas été récupéré — bascule sur l’onglet **JSON avancé** ou **CSV**.")
    features = {}
    cols = st.columns(2) if len(top_cols) > 1 else [st]

    for i, colname in enumerate(top_cols):
        with cols[i % len(cols)]:
            if is_numeric_name(colname):
                val = st.number_input(colname, value=0.0, step=1.0, format="%.6f")
                if val != 0.0:
                    features[colname] = float(val)
            else:
                txt = st.text_input(colname, value="")
                if txt.strip():
                    # tente cast numérique, sinon string
                    try:
                        num = float(txt)
                        features[colname] = num
                    except Exception:
                        features[colname] = txt.strip()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Prédire (classe)", use_container_width=True):
            try:
                resp = call_api("/predict", {"features": features})
                st.success("Réponse API /predict")
                st.code(json.dumps(resp, indent=2, ensure_ascii=False))
            except Exception as e:
                st.error(f"Erreur /predict : {e}")
    with c2:
        if st.button("Probabilité (score)", use_container_width=True):
            try:
                resp = call_api("/predict_proba", {"features": features})
                st.success("Réponse API /predict_proba")
                st.code(json.dumps(resp, indent=2, ensure_ascii=False))
            except Exception as e:
                st.error(f"Erreur /predict_proba : {e}")

    if features:
        st.markdown("**Exemple `curl`**")
        st.code(
            "curl -X POST \\\n"
            f"  '{API_URL}/predict' \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps({'features': features}, ensure_ascii=False)}'"
        )

with tab_json:
    st.write("Colle un JSON complet pour `features` (toutes colonnes ou un sous-ensemble).")
    example = {"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 4430}
    raw = st.text_area("JSON", value=json.dumps({"features": example}, indent=2), height=180)
    c1, c2 = st.columns(2)
    if c1.button("Prédire (classe)"):
        try:
            payload = json.loads(raw)
            resp = call_api("/predict", payload)
            st.success("Réponse API /predict")
            st.code(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception as e:
            st.error(f"Erreur : {e}")
    if c2.button("Probabilité (score)"):
        try:
            payload = json.loads(raw)
            resp = call_api("/predict_proba", payload)
            st.success("Réponse API /predict_proba")
            st.code(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception as e:
            st.error(f"Erreur : {e}")

with tab_csv:
    st.write("Charge un CSV contenant **une seule ligne** (ou choisis la ligne à scorer). Les noms de colonnes doivent matcher au mieux `/schema`.")
    up = st.file_uploader("CSV", type=["csv"])
    if up:
        try:
            df = pd.read_csv(io.BytesIO(up.read()))
            if df.empty:
                st.error("CSV vide.")
            else:
                idx = 0
                if len(df) > 1:
                    idx = st.number_input("Index de la ligne à utiliser", min_value=0, max_value=len(df)-1, value=0, step=1)
                row = df.iloc[int(idx)].to_dict()
                st.write("Aperçu ligne sélectionnée :")
                st.json(row)
                if st.button("Scorer la ligne CSV"):
                    try:
                        resp = call_api("/predict", {"features": row})
                        st.success("Réponse API /predict")
                        st.code(json.dumps(resp, indent=2, ensure_ascii=False))
                    except Exception as e:
                        st.error(f"Erreur /predict : {e}")
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")
