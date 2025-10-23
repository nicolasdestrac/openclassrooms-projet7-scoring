# streamlit_app/app.py
# -*- coding: utf-8 -*-
import os, io, json
from pathlib import Path
from typing import List, Dict
from datetime import date

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1) TOUJOURS en premier : config de page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Projet 7 — Scoring", layout="centered")

# -----------------------------------------------------------------------------
# Helpers "secrets" sûrs (privilégie l'env, ne lit st.secrets que si le fichier existe)
# -----------------------------------------------------------------------------
def get_secret_env_first(key: str, default: str = "") -> str:
    v = os.getenv(key)
    if v:
        return v.strip()

    try:
        secrets_paths = [
            Path(".streamlit/secrets.toml"),
            Path("/opt/render/.streamlit/secrets.toml"),
            Path("/opt/render/project/src/.streamlit/secrets.toml"),
        ]
        if any(p.exists() for p in secrets_paths):
            # st.secrets.get -> None si absent
            return str(st.secrets.get(key, default)).strip()
    except Exception:
        pass
    return default

# -----------------------------------------------------------------------------
# Config (API + features à privilégier dans l'onglet Simple)
# -----------------------------------------------------------------------------
API_URL = get_secret_env_first("API_URL")
TOP_FEATURES_SECRET = get_secret_env_first("TOP_FEATURES", "")

st.title("Projet 7 — Scoring")

if not API_URL:
    st.error(
        "API_URL manquant. Ajoute la variable d'environnement **API_URL** dans "
        "Render → Settings → Environment, ou fournis `.streamlit/secrets.toml`."
    )
    st.stop()

API_BASE = API_URL.rstrip("/")  # évite //schema & co
st.caption(f"API: {API_BASE}")

# -----------------------------------------------------------------------------
# Fetch helpers (retournent (data, error_str))
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_json(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return {}, str(e)

@st.cache_data(ttl=300, show_spinner=False)
def get_schema():
    js, err = fetch_json(f"{API_BASE}/schema")
    cols = list(js.get("input_columns", [])) if js else []
    return cols, err

@st.cache_data(ttl=120, show_spinner=False)
def get_health():
    js, _ = fetch_json(f"{API_BASE}/health")
    return js if js else {}

schema, schema_err = get_schema()
health = get_health()

with st.expander("État API / Schéma", expanded=False):
    st.json(health or {"status": "unknown"})
    if schema:
        st.write(f"**{len(schema)} colonnes attendues**")
        st.code("\n".join(schema))
    if schema_err:
        st.warning(f"Erreur lors de l’appel /schema : {schema_err}")

# -----------------------------------------------------------------------------
# Sélection guidée de colonnes (et fallback si /schema vide)
# -----------------------------------------------------------------------------
def pick_top_features(all_cols: List[str], k: int = 10) -> List[str]:
    # priorité à TOP_FEATURES si fournie (ex: "AMT_INCOME_TOTAL,AMT_CREDIT,DAYS_BIRTH")
    if TOP_FEATURES_SECRET:
        want = [c.strip() for c in TOP_FEATURES_SECRET.split(",") if c.strip()]
        # si tout n'est pas dans all_cols, on renvoie quand même l'ordre demandé
        return [c for c in want if (not all_cols or c in all_cols)] or want

    if not all_cols:
        return []

    key_words = ("AMT", "DAYS", "CREDIT", "INCOME", "EXT_SOURCE", "AGE", "SCORE", "AMT_")
    ranked = sorted(all_cols, key=lambda c: any(k in c.upper() for k in key_words), reverse=True)
    seen, out = set(), []
    for c in ranked + all_cols:
        if c not in seen:
            out.append(c); seen.add(c)
        if len(out) >= k:
            break
    return out

DEFAULT_TOP = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "NAME_INCOME_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "AMT_GOODS_PRICE",
]

top_cols = pick_top_features(schema, k=10) if schema else DEFAULT_TOP

# -----------------------------------------------------------------------------
# Widgets "smart" pour l’onglet Simple
# -----------------------------------------------------------------------------
def _is_money(col: str) -> bool:
    cu = col.upper()
    return cu.startswith("AMT_") or cu.endswith("_AMT") or "AMT" in cu

def render_input_for(colname: str):
    """
    Rend un widget adapté et renvoie la valeur (ou None si non saisie).
    - Montants -> 2 décimales
    - DAYS_BIRTH -> date-picker puis conversion en jours négatifs
    - Autres DAYS_* -> entier
    - RATIO/SCORE -> 4 décimales
    - NAME_* -> texte
    - Par défaut -> numérique 6 décimales
    """
    cu = colname.upper()

    # 1) Calendrier -> DAYS_BIRTH (négatif, nb de jours avant aujourd'hui)
    if cu == "DAYS_BIRTH":
        st.markdown("**Date de naissance** → convertie en `DAYS_BIRTH` (jours négatifs)")
        dob = st.date_input("Date de naissance", value=date(1985, 1, 1))
        days = -(date.today() - dob).days
        st.caption(f"DAYS_BIRTH calculé : {days}")
        return float(days)

    # 2) Montants => 2 décimales
    if _is_money(colname):
        val = st.number_input(colname, min_value=0.0, step=100.0, format="%.2f")
        return float(val) if val != 0.0 else None

    # 3) Tous les autres DAYS_* => entier (0 décimale)
    if cu.startswith("DAYS_"):
        val = st.number_input(colname, value=0, step=1, format="%.0f")
        return float(val) if val != 0 else None

    # 4) RATIO / SCORE => 4 décimales
    if "RATIO" in cu or "SCORE" in cu:
        val = st.number_input(colname, min_value=0.0, step=0.01, format="%.4f")
        return float(val) if val != 0.0 else None

    # 5) Catégorielles NAME_* => texte
    if cu.startswith("NAME_"):
        txt = st.text_input(colname, value="")
        return txt.strip() or None

    # 6) Par défaut : numérique 6 décimales
    val = st.number_input(colname, value=0.0, step=1.0, format="%.6f")
    return float(val) if val != 0.0 else None

def call_api(endpoint: str, payload: Dict) -> Dict:
    r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------------------------------------
# UI principale
# -----------------------------------------------------------------------------
tab_simple, tab_json, tab_csv = st.tabs(["🧩 Simple", "💻 JSON avancé", "📄 CSV (1 ligne)"])

with tab_simple:
    st.write("Renseigne quelques variables utiles. Les colonnes manquantes seront imputées par le pipeline.")

    if not schema:
        st.info("Le schéma n'a pas été récupéré — bascule sur l’onglet **JSON avancé** ou **CSV**.")

    features = {}
    cols = st.columns(2) if len(top_cols) > 1 else [st]

    for i, colname in enumerate(top_cols):
        with cols[i % len(cols)]:
            val = render_input_for(colname)
            if val is not None:
                features[colname] = val

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
            f"  '{API_BASE}/predict' \\\n"
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
    st.write("Charge un CSV contenant **une seule ligne** (ou choisis la ligne à scorer). "
             "Les noms de colonnes doivent matcher au mieux `/schema`.")
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
