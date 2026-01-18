"""
Scoring API
===========

Expose une API REST (FastAPI) pour :
- obtenir le schéma d'entrée (`/schema`) ;
- calculer une probabilité (`/predict_proba`) ;
- produire une décision binaire avec seuil métier (`/predict`) ;
- expliquer localement une prédiction via SHAP (`/explain`);
- consulter un journal minimal des événements de prédiction (`/events/stats`, `/events/download`).

Artefacts requis (dans MODEL_DIR, par défaut `models/`) :
- `scoring_model.joblib` : sklearn Pipeline (preprocessor + estimator) ;
- `decision_threshold.json` : {"threshold": float} ;
- `input_columns.json` : liste ordonnée des colonnes attendues.

Variables d’environnement utiles :
- MODEL_DIR            : dossier des artefacts ;
- FRONTEND_ORIGINS     : origines CORS autorisées, séparées par des virgules ;
- EVENTS_PATH          : chemin du fichier JSONL des événements (défaut: models/events.jsonl).

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
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# -----------------------------
# Chemins et artefacts modèle
# -----------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "scoring_model.joblib")
THRESH_PATH = os.path.join(MODEL_DIR, "decision_threshold.json")
SCHEMA_PATH = os.path.join(MODEL_DIR, "input_columns.json")

# -----------------------------
# Chargements
# -----------------------------
try:
    # Pipeline scikit-learn attendu: steps "prep" et "clf" (au minimum)
    pipe = joblib.load(PIPELINE_PATH)
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
# Journalisation des événements (JSONL)
# -----------------------------
EVENTS_PATH = Path(os.getenv("EVENTS_PATH", os.path.join(MODEL_DIR, "events.jsonl")))
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _model_version_from_pipe(pipeline) -> str:
    """
    Fabrique un identifiant court de 'version' à partir de la classe du classifieur
    et de quelques hyperparamètres (hash MD5 tronqué).
    """
    try:
        clf = pipeline.named_steps.get("clf", None)
        name = clf.__class__.__name__ if clf is not None else "UnknownModel"
        params = clf.get_params() if hasattr(clf, "get_params") else {}
        key = f"{name}:{params.get('num_leaves','')}:{params.get('n_estimators','')}:{params.get('max_depth','')}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"

MODEL_VERSION = _model_version_from_pipe(pipe)

def _write_event(payload: Dict[str, Any]) -> None:
    """
    Ajoute un événement au journal JSONL. Tolérant aux erreurs disque: ne casse pas l'API.
    """
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[events] WARNING write failed: {e}")

# -----------------------------
# App FastAPI + CORS
# -----------------------------
app = FastAPI(title="Scoring API", version="1.0.0")

# FRONTEND_ORIGINS peut contenir plusieurs origines séparées par des virgules.
# Exemple Render :
#   FRONTEND_ORIGINS = https://openclassrooms-projet7-scoring-streamlit.onrender.com
origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN")
if origins_env:
    ALLOW_ORIGINS = [o.strip().rstrip("/") for o in origins_env.split(",") if o.strip()]
else:
    # Fallback permissif en dev ; en prod, définir FRONTEND_ORIGINS.
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
    features: Dict[str, Any]

# -----------------------------
# Helpers
# -----------------------------
def _row_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
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

def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    """Endpoint d'accueil: liste des routes et métadonnées de base."""
    return {
        "message": "Scoring API en ligne",
        "endpoints": [
            "/health",
            "/schema",
            "/predict_proba",
            "/predict",
            "/explain",
            "/events/stats",
            "/events/download",
            "/docs",
        ],
        "allow_origins": ALLOW_ORIGINS,
        "model_dir": MODEL_DIR,
        "model_version": MODEL_VERSION,
    }

@app.get("/health")
def health():
    """Vérifie la disponibilité de l’API et retourne quelques métadonnées (seuil, nb de colonnes, version modèle…)."""
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "pipeline": os.path.basename(PIPELINE_PATH),
        "threshold": DECISION_THRESHOLD,
        "n_input_columns": len(INPUT_COLUMNS),
        "model_version": MODEL_VERSION,
        "events_path": str(EVENTS_PATH),
    }

@app.get("/schema")
def schema():
    """Retourne la liste ordonnée des colonnes d’entrée attendues par le modèle."""
    return {"input_columns": INPUT_COLUMNS}

@app.post("/predict_proba")
def predict_proba(payload: PredictPayload):
    """
    Calcule la probabilité de défaut (classe positive = 1).

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

        # Log événement
        _write_event({
            "ts_utc": _now_utc(),
            "endpoint": "/predict_proba",
            "probability": float(proba),
            "prediction": int(proba >= DECISION_THRESHOLD),
            "threshold": DECISION_THRESHOLD,
            "model_version": MODEL_VERSION,
            "features_keys": sorted(list(payload.features.keys())),
        })

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

        # Log événement
        _write_event({
            "ts_utc": _now_utc(),
            "endpoint": "/predict",
            "probability": float(proba),
            "prediction": int(pred),
            "threshold": DECISION_THRESHOLD,
            "model_version": MODEL_VERSION,
            "features_keys": sorted(list(payload.features.keys())),
        })

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
        prep = pipe.named_steps.get("prep")
        clf = pipe.named_steps.get("clf")

        # Noms de features après transformation (OHE, etc.)
        try:
            feat_names = prep.get_feature_names_out()
        except Exception:
            Xt_tmp = prep.transform(X_row)
            feat_names = [f"f_{i}" for i in range(Xt_tmp.shape[1])]

        # Transforme X pour l'explainer
        Xt = prep.transform(X_row)

        contrib: Dict[str, float] = {}
        base_value = None

        if hasattr(clf, "predict_proba"):
            # Tree-based -> TreeExplainer en priorité
            try:
                expl = shap.TreeExplainer(clf)
                sv = expl.shap_values(Xt)
                # shap<0.41 renvoie liste [class0, class1] ; versions récentes: objet
                if isinstance(sv, list) and len(sv) > 1:
                    vals = sv[1][0]  # (n_features,)
                    base_value = float(expl.expected_value[1])
                else:
                    # support des objets Explanation
                    vals = np.array(getattr(sv, "values", sv))[0]
                    base_value = float(getattr(sv, "base_values", [None])[0])
            except Exception:
                # Secours: KernelExplainer (plus lent)
                f = lambda Z: clf.predict_proba(Z)[:, 1]
                bg = np.zeros((50, Xt.shape[1]))  # fond neutre
                expl = shap.KernelExplainer(f, bg)
                vals = expl.shap_values(Xt, nsamples=100)[0]
                base_value = float(expl.expected_value)
        else:
            raise RuntimeError("Modèle sans predict_proba.")

        # Construit dict {feat_name: shap_value}
        for name, v in zip(feat_names, np.asarray(vals).ravel()):
            contrib[str(name)] = float(v)

        # Top 20 par importance absolue
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:20]
        contrib_top = {k: v for k, v in items}

        return {"base_value": base_value, "contrib": contrib_top}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'explication: {e}")

# -----------------------------
# Endpoints événements (utiles démo/monitoring)
# -----------------------------
@app.get("/events/stats")
def events_stats():
    """
    Retourne le nombre de lignes journalisées et la version du modèle.
    """
    try:
        n = sum(1 for _ in EVENTS_PATH.open("r", encoding="utf-8"))
    except FileNotFoundError:
        n = 0
    return {"events_file": str(EVENTS_PATH), "count": n, "model_version": MODEL_VERSION}

@app.get("/events/download")
def events_download(limit: int = 1000):
    """
    Retourne les dernières lignes du journal JSONL (max `limit`).
    Paramètres
    ----------
    limit : int
        Nombre maximum de lignes renvoyées (défaut: 1000).
    """
    try:
        lines: List[str] = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
        tail = lines[-int(limit):] if limit and limit > 0 else lines
        return {"events": [json.loads(l) for l in tail]}
    except FileNotFoundError:
        return {"events": []}
