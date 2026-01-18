"""Frontend Streamlit pour le projet de scoring (P7).

Cette application appelle l'API de scoring (FastAPI) pour obtenir :
- le schéma d’entrée attendu,
- la santé/les métadonnées de l’API,
- une probabilité de défaut et une prédiction binaire,
- une explication locale (SHAP) des facteurs influençant la décision.

Variables d’environnement / secrets attendus
--------------------------------------------
API_URL : str
    URL racine de l’API (ex. https://openclassrooms-projet7-scoring-api.onrender.com/).
TOP_FEATURES : str, optionnel
    Liste de colonnes « mises en avant » (séparées par des virgules) à afficher
    en priorité dans le formulaire simple.
"""

import os, io, json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import date

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1) Config de page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Projet 7 — Scoring", layout="centered")

# -----------------------------------------------------------------------------
# Helpers "secrets" sûrs
# -----------------------------------------------------------------------------
def get_secret_env_first(key: str, default: str = "") -> str:
    """Récupère une valeur de configuration depuis l'environnement ou Streamlit.

    L’ordre de recherche est :
    1) Variable d’environnement (si définie),
    2) Fichiers `.streamlit/secrets.toml` (plusieurs chemins connus),
    3) Valeur par défaut.

    Args:
        key: Nom de la clé à lire.
        default: Valeur retournée si la clé n’est trouvée nulle part.

    Returns:
        La valeur trouvée (chaîne), éventuellement vide si absente partout.
    """
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
            return str(st.secrets.get(key, default)).strip()
    except Exception:
        pass
    return default

API_URL = get_secret_env_first("API_URL")
TOP_FEATURES_SECRET = get_secret_env_first("TOP_FEATURES", "")

st.title("Projet 7 — Scoring")

if not API_URL:
    st.error("API_URL manquant. Ajoutez la variable d'environnement **API_URL** (Render → Settings → Environment).")
    st.stop()

API_BASE = API_URL.rstrip("/")
st.caption(f"API: {API_BASE}")

# -----------------------------------------------------------------------------
# Fetch helpers
# -----------------------------------------------------------------------------
def fetch_json(url: str) -> Tuple[Dict, str|None]:
    """Fait une requête HTTP GET et parse la réponse JSON.

    Args:
        url: URL à interroger.

    Returns:
        Un tuple `(payload, error)` où :
        - `payload` est le dict JSON (ou `{}` en cas d’échec),
        - `error` est `None` si tout va bien, sinon un message d’erreur.
    """
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return {}, str(e)

@st.cache_data(ttl=300, show_spinner=False)
def get_schema():
    """Récupère et met en cache la liste ordonnée des colonnes d’entrée.

    Retourne les noms de colonnes exposés par l’API via `/schema`.

    Returns:
        Tuple `(columns, error)` :
        - `columns`: liste des colonnes attendues (list[str]).
        - `error`: message d’erreur éventuel (str | None).
    """
    js, err = fetch_json(f"{API_BASE}/schema")
    cols = list(js.get("input_columns", [])) if js else []
    return cols, err

@st.cache_data(ttl=120, show_spinner=False)
def get_health():
    """Récupère et met en cache les métadonnées de santé de l’API.

    Returns:
        Dict JSON renvoyé par `/health` (éventuellement vide si indisponible).
    """
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
        st.warning(f"Erreur /schema : {schema_err}")

# -----------------------------------------------------------------------------
# Colonnes à privilégier
# -----------------------------------------------------------------------------
def pick_top_features(all_cols: List[str], k: int = 6) -> List[str]:
    """Sélectionne k colonnes « mises en avant » pour le formulaire simple.

    Si `TOP_FEATURES` est défini (env/secrets), on respecte cet ordre.
    Sinon, on privilégie heuristiquement des colonnes financières/temps.

    Args:
        all_cols: Colonnes disponibles (peut être vide).
        k: Nombre désiré de colonnes.

    Returns:
        Liste de `k` noms de colonnes (ou moins si indisponibles).
    """
    if TOP_FEATURES_SECRET:
        want = [c.strip() for c in TOP_FEATURES_SECRET.split(",") if c.strip()]
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

DEFAULT_TOP = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
top_cols = pick_top_features(schema, k=6) if schema else DEFAULT_TOP

# -----------------------------------------------------------------------------
# Libellés FR & widgets
# -----------------------------------------------------------------------------
COL_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu total annuel ($)",
    "AMT_CREDIT": "Montant du crédit ($)",
    "AMT_ANNUITY": "Mensualité (annuité) ($)",
    "AMT_GOODS_PRICE": "Prix du bien ($)",
    "NAME_INCOME_TYPE": "Type de revenu",
    "DAYS_BIRTH": "Âge (date de naissance)",
    "DAYS_EMPLOYED": "Jours d'emploi (négatifs si en cours)",
    "DAYS_REGISTRATION": "Jours depuis l'enregistrement",
    "DAYS_ID_PUBLISH": "Jours depuis émission de la pièce d'identité",
    "OWN_CAR_AGE": "Âge du véhicule (années)",
}
def fr_label(colname: str) -> str:
    """Fournit un libellé lisible en français pour un nom de colonne.

    Args:
        colname: Nom brut de la feature.

    Returns:
        Libellé à afficher côté UI.
    """
    return COL_LABELS.get(colname, colname.replace("_", " ").title())

INCOME_TYPE_CHOICES = [
    ("Salarié",               "Working"),
    ("Fonctionnaire",         "State servant"),
    ("Commerçant / Associé",  "Commercial associate"),
    ("Retraité",              "Pensioner"),
    ("Étudiant",              "Student"),
    ("Sans emploi",           "Unemployed"),
    ("Entrepreneur",          "Businessman"),
    ("Congé maternité",       "Maternity leave"),
]
INCOME_TYPE_FR = [fr for fr, _ in INCOME_TYPE_CHOICES]

def _is_money(col: str) -> bool:
    """Indique si une colonne représente un montant monétaire.

    Heuristique basée sur la présence de 'AMT'.

    Args:
        col: Nom de la colonne.

    Returns:
        True si la colonne est assimilée à un montant, False sinon.
    """
    cu = col.upper()
    return cu.startswith("AMT_") or cu.endswith("_AMT") or "AMT" in cu

def render_input_for(colname: str):
    """Affiche le widget Streamlit adapté à une colonne et retourne la valeur saisie.

    Mapping principal :
    - `DAYS_BIRTH` : saisie de date → conversion en jours négatifs,
    - Montants (`AMT_*`) : `number_input` non négatif,
    - `DAYS_*` : entier (peut être 0),
    - ratios/scores : flottant,
    - `NAME_INCOME_TYPE` : liste FR → valeur anglaise ou saisie libre,
    - préfixe `NAME_` : texte,
    - fallback : flottant générique.

    Args:
        colname: Nom de la colonne à rendre.

    Returns:
        Valeur saisie (float/str) ou `None` si laissée vide.
    """
    label = fr_label(colname)
    cu = colname.upper()

    if cu == "DAYS_BIRTH":
        st.markdown(f"**{label}** → convertie en `DAYS_BIRTH` (jours négatifs)")
        dob = st.date_input("Date de naissance", value=date(1985, 1, 1), key=f"{colname}_date")
        days = -(date.today() - dob).days
        st.caption(f"DAYS_BIRTH calculé : {days}")
        return float(days)

    if _is_money(colname):
        val = st.number_input(label, min_value=0.0, step=100.0, format="%.2f", key=f"{colname}_money")
        return float(val) if val != 0.0 else None

    if cu.startswith("DAYS_"):
        val = st.number_input(label, value=0, step=1, format="%d", key=f"{colname}_int")
        return float(val) if val != 0 else None

    if "RATIO" in cu or "SCORE" in cu:
        val = st.number_input(label, min_value=0.0, step=0.01, format="%.4f", key=f"{colname}_ratio")
        return float(val) if val != 0.0 else None

    if cu == "NAME_INCOME_TYPE":
        choix_fr = st.selectbox(
            label,
            options=["— Sélectionner —"] + INCOME_TYPE_FR + ["Autre (saisie libre)"],
            index=0,
            key=f"{colname}_sel"
        )
        if choix_fr == "— Sélectionner —":
            return None
        if choix_fr == "Autre (saisie libre)":
            libre = st.text_input("Type de revenu (texte libre)", key=f"{colname}_free")
            return libre.strip() or None
        return dict(INCOME_TYPE_CHOICES).get(choix_fr)

    if cu.startswith("NAME_"):
        txt = st.text_input(label, value="", key=f"{colname}_text")
        return txt.strip() or None

    val = st.number_input(label, value=0.0, step=1.0, format="%.6f", key=f"{colname}_num")
    return float(val) if val != 0.0 else None

def call_api(endpoint: str, payload: Dict = None, method: str = "POST") -> Dict:
    """Appelle un endpoint de l’API (GET/POST) et retourne le JSON.

    Args:
        endpoint: Chemin d’endpoint (ex. '/predict').
        payload: Corps JSON pour les requêtes POST.
        method: Verbe HTTP, 'GET' ou 'POST'.

    Returns:
        Dict JSON renvoyé par l’API.

    Raises:
        RuntimeError: En cas d’échec réseau ou HTTP.
    """
    url = f"{API_BASE}{endpoint}"
    try:
        if method.upper() == "GET":
            r = requests.get(url, timeout=20)
        else:
            r = requests.post(url, json=payload or {}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Appel {endpoint} échoué : {e}")

# -----------------------------------------------------------------------------
# UI principale
# -----------------------------------------------------------------------------
tab_simple, tab_json, tab_csv = st.tabs(["🧩 Simple", "💻 JSON avancé", "📄 CSV (1 ligne)"])

def render_decision(prob: float, thr: float):
    """Affiche la décision binaire et une jauge à partir d’une probabilité.

    La décision est 1 si `prob >= thr`, sinon 0 (convention rappelée dans l’UI).

    Args:
        prob: Probabilité de défaut estimée (0–1).
        thr: Seuil métier appliqué (0–1).
    """
    pred = int(prob >= thr)
    label = "✅ ACCEPTÉ" if pred == 0 else "❌ REFUSÉ"  # selon ta convention y=1 = défaut
    color = "#16a34a" if label.startswith("✅") else "#dc2626"

    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid #e5e7eb;border-radius:12px;">
          <div style="font-size:20px;font-weight:700;color:{color};margin-bottom:6px;">{label}</div>
          <div>Probabilité de défaut estimée : <b>{prob:.1%}</b></div>
          <div>Seuil métier : <b>{thr:.1%}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0))

def render_shap_bar(contrib: Dict[str, float], top_k: int = 10):
    """Trace un bar chart horizontal des contributions locales (SHAP).

    Les contributions sont triées par valeur absolue et tronquées aux `top_k`.

    Args:
        contrib: Dictionnaire `{feature: shap_value}`.
        top_k: Nombre maximum de facteurs à afficher.

    Returns:
        None (affichage Streamlit in-place).
    """
    if not contrib:
        st.info("Explication indisponible pour cette observation.")
        return
    # tri par contribution absolue
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    labels = [fr_label(k) for k, _ in items][::-1]
    vals   = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(labels, vals)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Facteurs qui influencent la décision (SHAP)")
    ax.set_xlabel("Contribution vers REFUS ( + ) / ACCEPTATION ( – )")
    plt.tight_layout()
    st.pyplot(fig)

with tab_simple:
    st.write("Renseignez quelques variables utiles. Les colonnes manquantes seront imputées par le pipeline.")
    if not schema:
        st.info("Le schéma n'a pas été récupéré — basculez sur l’onglet **JSON avancé** ou **CSV**.")

    features = {}
    cols = st.columns(2) if len(top_cols) > 1 else [st]
    for i, colname in enumerate(top_cols):
        with cols[i % len(cols)]:
            val = render_input_for(colname)
            if val is not None:
                features[colname] = val

    col_action1, col_action2 = st.columns(2)
    with col_action1:
        predict_clicked = st.button("Prédire (classe)", use_container_width=True)
    with col_action2:
        proba_clicked = st.button("Probabilité (score)", use_container_width=True)

    if predict_clicked or proba_clicked:
        try:
            with st.spinner("Appel API..."):
                if predict_clicked:
                    resp = call_api("/predict", {"features": features})
                    prob, thr = float(resp["probability"]), float(resp["threshold"])
                else:
                    resp = call_api("/predict_proba", {"features": features})
                    prob = float(resp["probability"])
                    thr = float(health.get("threshold", 0.5))
            render_decision(prob, thr)

            # --- Tentative d'explication locale via /explain ---
            try:
                explain = call_api("/explain", {"features": features})
                # attendu: {"base_value":..., "contrib": {"feat": shap_value, ...}}
                contrib = explain.get("contrib", {})
                if contrib:
                    st.divider()
                    render_shap_bar(contrib, top_k=10)
            except Exception as e:
                st.info("Explication SHAP indisponible (endpoint /explain non exposé).")

            st.caption("La prédiction binaire utilise le seuil métier (coûts FN/FP). "
                       "Le score affiché est la probabilité estimée de défaut (`P(y=1)`).")
            st.code(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception as e:
            st.error(str(e))

    if features:
        st.markdown("**Exemple `curl`**")
        st.code(
            "curl -X POST \\\n"
            f"  '{API_BASE}/predict' \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps({'features': features}, ensure_ascii=False)}'"
        )

with tab_json:
    st.write("Collez un JSON pour `features` (toutes colonnes ou un sous-ensemble).")
    example = {"AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 4430}
    raw = st.text_area("JSON", value=json.dumps({"features": example}, indent=2), height=180)
    c1, c2 = st.columns(2)
    if c1.button("Prédire (classe)"):
        try:
            payload = json.loads(raw)
            resp = call_api("/predict", payload)
            prob, thr = float(resp["probability"]), float(resp["threshold"])
            render_decision(prob, thr)
            st.code(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception as e:
            st.error(f"Erreur : {e}")
    if c2.button("Probabilité (score)"):
        try:
            payload = json.loads(raw)
            resp = call_api("/predict_proba", payload)
            prob = float(resp["probability"])
            thr = float(health.get("threshold", 0.5))
            render_decision(prob, thr)
            st.code(json.dumps(resp, indent=2, ensure_ascii=False))
        except Exception as e:
            st.error(f"Erreur : {e}")

with tab_csv:
    st.write("Chargez un CSV (1 ligne) ou choisissez la ligne à scorer. Les noms de colonnes doivent coller à `/schema`.")
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
                        prob, thr = float(resp["probability"]), float(resp["threshold"])
                        render_decision(prob, thr)
                        st.code(json.dumps(resp, indent=2, ensure_ascii=False))
                    except Exception as e:
                        st.error(f"Erreur /predict : {e}")
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")
