import os
import json
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Scoring - Démo", layout="centered")
st.title("Scoring crédit – Démo")

@st.cache_data(ttl=300)
def get_schema():
    r = requests.get(f"{API_URL}/schema", timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def get_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()

# Bandeau statut API
try:
    h = get_health()
    st.success(f"API OK • modèle: {h.get('pipeline')} • seuil: {h.get('threshold')}")
except Exception as e:
    st.error(f"API KO: {e}")

schema = get_schema()
cols = schema["input_columns"]

st.subheader("Saisir les caractéristiques")
with st.form("form"):
    # UI minimaliste : on permet de saisir quelques champs fréquents, et tout le reste via JSON facultatif
    amt_income_total = st.number_input("AMT_INCOME_TOTAL", min_value=0.0, value=180000.0, step=1000.0)
    amt_credit       = st.number_input("AMT_CREDIT",       min_value=0.0, value=450000.0, step=1000.0)
    code_gender      = st.selectbox("CODE_GENDER", options=["M", "F", "XNA"], index=0)
    name_contract    = st.selectbox("NAME_CONTRACT_TYPE", options=["Cash loans","Revolving loans"], index=0)

    st.caption("Champs avancés (optionnel) au format JSON")
    default_extra = json.dumps({"AMT_ANNUITY": 25000, "FLAG_OWN_REALTY": "Y"}, indent=2)
    extra_json = st.text_area("Extra features JSON", value=default_extra, height=140)

    submitted = st.form_submit_button("Calculer")

if submitted:
    # Construction du payload conforme à /schema : on met les colonnes absentes à blanc
    features = {c: None for c in cols}
    features.update({
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "CODE_GENDER": code_gender,
        "NAME_CONTRACT_TYPE": name_contract,
    })
    # merge JSON libre
    try:
        extra = json.loads(extra_json.strip() or "{}")
        for k, v in extra.items():
            if k in features:
                features[k] = v
    except Exception as e:
        st.warning(f"JSON invalide ignoré: {e}")

    # Appel /predict_proba puis /predict
    try:
        r1 = requests.post(f"{API_URL}/predict_proba", json={"features": features}, timeout=20)
        r1.raise_for_status()
        proba = r1.json()["probability"]

        r2 = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=20)
        r2.raise_for_status()
        pred_res = r2.json()

        st.markdown("### Résultat")
        st.metric("Probabilité de défaut (modèle)", f"{proba:.3f}")
        st.write(f"Seuil décision: **{pred_res.get('threshold')}**")
        decision = "❌ Refus" if pred_res.get("prediction", 0) == 1 else "✅ Acceptation"
        st.subheader(decision)

        # Rappel des features réellement envoyées (non vides)
        df_show = pd.DataFrame(
            [{"feature": k, "value": v} for k, v in features.items() if v is not None]
        )
        st.dataframe(df_show, use_container_width=True)

    except requests.HTTPError as e:
        st.error(f"Erreur HTTP: {e.response.text}")
    except Exception as e:
        st.error(f"Erreur: {e}")
