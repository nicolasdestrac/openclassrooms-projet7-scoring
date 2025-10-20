from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
import os

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
    Construit une ligne dataframe à partir du dict de features.
    - Ajoute les colonnes manquantes avec NaN
    - Réordonne les colonnes pour correspondre à INPUT_COLUMNS
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
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "pipeline": os.path.basename(PIPELINE_PATH),
        "threshold": DECISION_THRESHOLD,
        "n_input_columns": len(INPUT_COLUMNS),
    }

@app.get("/schema")
def schema():
    return {"input_columns": INPUT_COLUMNS}

@app.post("/predict_proba")
def predict_proba(payload: PredictPayload):
    try:
        X = _row_from_payload(payload.features)
        proba = pipe.predict_proba(X)[:, 1].item()
        return {"probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {e}")

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        X = _row_from_payload(payload.features)
        proba = pipe.predict_proba(X)[:, 1].item()
        pred = int(proba >= DECISION_THRESHOLD)
        return {"probability": float(proba), "prediction": pred, "threshold": DECISION_THRESHOLD}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {e}")
