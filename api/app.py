"""
Scoring API
===========

Expose une API REST (FastAPI) pour :
- obtenir le schéma d'entrée (`/schema`) ;
- calculer une probabilité (`/predict_proba`) ;
- produire une décision binaire avec seuil métier (`/predict`) ;
- expliquer localement une prédiction via SHAP (`/explain`).

Artefacts requis (dans MODEL_DIR, par défaut `models/`) :
- `scoring_model.joblib` : sklearn Pipeline (preprocessor + estimator) ;
- `decision_threshold.json` : {"threshold": float} ;
- `input_columns.json` : liste ordonnée des colonnes attendues.

Variables d’environnement utiles :
- MODEL_DIR            : dossier des artefacts ;
- FRONTEND_ORIGINS     : origines CORS autorisées, séparées par des virgules.

Exceptions
----------
- RuntimeError si les artefacts indispensables sont introuvables au démarrage.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os
import shap

# -----------------------------
# Chemins et artefacts modèle
# -----------------------------
MODEL_DIR    = os.getenv("MODEL_DIR", "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "scoring_model.joblib")
THRESH_PATH   = os.path.join(MODEL_DIR, "decision_threshold.json")
SCHEMA_PATH   = os.path.join(MODEL_DIR, "input_columns.json")

# -----------------------------
# Chargements
# -----------------------------
try:
    pipe = joblib.load(PIPELINE_PATH)  # Pipeline(preprocessor, estimator)
except Exception as e:
    raise RuntimeError(f"Impossible de charger {PIPELINE_PATH}: {e}")

try:
    with open(THRESH_PATH, "r") as f:
        decision = json.load(f)
    DECISION_THRESHOLD = float(decision.get("threshold", 0.5))
except Exception:
    DECISION_THRESHOLD = 0.5

try:
    with open(SCHEMA_PATH, "r") as f:
        INPUT_COLUMNS = list(json.load(f))
except Exception:
    raise RuntimeError(
        f"Schéma d'entrée introuvable: {SCHEMA_PATH}. "
        "Ajoute l'écriture de models/input_columns.json dans train.py."
    )

# -----------------------------
# App FastAPI + CORS
# -----------------------------
app = FastAPI(title="Scoring API", version="1.0.0")

# FRONTEND_ORIGINS peut contenir plusieurs origines séparées par des virgules.
# Exemple Render (service API > Environment Variables):
#   FRONTEND_ORIGINS = https://openclassrooms-projet7-scoring-streamlit.onrender.com
origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN")
if origins_env:
    ALLOW_ORIGINS = [o.strip().rstrip("/") for o in origins_env.split(",") if o.strip()]
else:
    # Fallback permissif (utile en dev). Pour la prod, mets la variable d'env ci-dessus.
    ALLOW_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Schéma de requête
# -----------------------------
class PredictPayload(BaseModel):
    features: dict

# -----------------------------
# Helpers
# -----------------------------
def _row_from_payload(payload: dict) -> pd.DataFrame:
    """
    Construit une unique ligne d'entrée alignée sur le schéma attendu.

    Parameters
    ----------
    payload : dict
        Dictionnaire {feature: valeur} reçu dans la requête.

    Returns
    -------
    pd.DataFrame
        DataFrame 1xN avec toutes les colonnes `INPUT_COLUMNS`
        (les colonnes manquantes sont remplies avec NaN).
    """
    df = pd.DataFrame([payload])
    for c in INPUT_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[INPUT_COLUMNS]
    return df

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Scoring API en ligne",
        "endpoints": ["/health", "/schema", "/predict_proba", "/predict", "/docs"],
        "allow_origins": ALLOW_ORIGINS,
        "model_dir": MODEL_DIR,
    }

@app.get("/health")
def health():
    """Vérifie la disponibilité de l’API et retourne quelques métadonnées (seuil, nb de colonnes, etc.)."""
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "pipeline": os.path.basename(PIPELINE_PATH),
        "threshold": DECISION_THRESHOLD,
        "n_input_columns": len(INPUT_COLUMNS),
    }

@app.get("/schema")
def schema():
    """Retourne la liste ordonnée des colonnes d’entrée attendues par le modèle."""
    return {"input_columns": INPUT_COLUMNS}

@app.post("/predict_proba")
def predict_proba(payload: PredictPayload):
    """
    Calcule la probabilité de défaut (classe 1).

    Body
    ----
    {"features": {...}}

    Returns
    -------
    {"probability": float}

    Raises
    ------
    HTTPException(400) en cas d’erreur de préparation des features ou d’inférence.
    """
    try:
        X = _row_from_payload(payload.features)
        proba = pipe.predict_proba(X)[:, 1].item()
        return {"probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {e}")

@app.post("/predict")
def predict(payload: PredictPayload):
    """
    Renvoie la décision binaire en appliquant le seuil métier.

    Returns
    -------
    {
      "probability": float,
      "prediction": int,    # 0/1
      "threshold": float
    }
    """
    try:
        X = _row_from_payload(payload.features)
        proba = pipe.predict_proba(X)[:, 1].item()
        pred = int(proba >= DECISION_THRESHOLD)
        return {"probability": float(proba), "prediction": pred, "threshold": DECISION_THRESHOLD}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {e}")

@app.post("/explain")
def explain(payload: PredictPayload):
    """
    Explique localement la prédiction via SHAP.

    Returns
    -------
    {
      "base_value": float | None,
      "contrib": {feature_name: shap_value, ...}   # Top 20 par importance absolue
    }

    Notes
    -----
    - Utilise TreeExplainer si le modèle est arborescent, sinon KernelExplainer (plus lent).
    - Les noms de features après preprocessing proviennent de `preprocessor.get_feature_names_out()`.
    """
    try:
        X_row = _row_from_payload(payload.features)
        # Récupère preprocessor + clf
        prep = pipe.named_steps.get("prep")
        clf  = pipe.named_steps.get("clf")

        # Noms de features après transformation (OHE, etc.)
        try:
            feat_names = prep.get_feature_names_out()
        except Exception:
            # fallback si indisponible
            Xt = prep.transform(X_row)
            feat_names = [f"f_{i}" for i in range(Xt.shape[1])]

        # Transforme X pour l'explainer
        Xt = prep.transform(X_row)

        # Explainer adapté
        contrib = {}
        base_value = None

        if hasattr(clf, "predict_proba"):
            # LGBM/Tree-based -> TreeExplainer
            try:
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(Xt)
                # shap >=0.41: sv est un object ; sinon liste [class0, class1]
                # On prend classe 1 (défaut)
                if isinstance(sv, list) and len(sv) > 1:
                    vals = sv[1][0]  # (n_features,)
                    base_value = float(expl.expected_value[1])
                else:
                    vals = np.array(sv.values[0]) if hasattr(sv, "values") else np.array(sv[0])
                    base_value = float(sv.base_values[0]) if hasattr(sv, "base_values") else None
            except Exception:
                # Secours: KernelExplainer (plus lent)
                f = lambda Z: clf.predict_proba(Z)[:,1]
                bg = np.zeros((50, Xt.shape[1]))  # fond neutre
                expl = shap.KernelExplainer(f, bg)
                vals = expl.shap_values(Xt, nsamples=100)[0]
                base_value = float(expl.expected_value)
        else:
            raise RuntimeError("Modèle sans predict_proba.")

        # Construit dict {feat_name: shap_value}
        for name, v in zip(feat_names, np.asarray(vals).ravel()):
            contrib[str(name)] = float(v)

        # Garde le top 20 par importance absolue
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
        contrib_top = {k: v for k, v in items}

        return {"base_value": base_value, "contrib": contrib_top}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'explication: {e}")
