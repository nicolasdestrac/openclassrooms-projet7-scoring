# tests/conftest.py
import json, os, importlib
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def dummy_model_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    # jeu minuscule
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(50000, 10000, 200),
        "AMT_CREDIT": rng.normal(150000, 30000, 200),
    })
    # cible un peu corrélée
    y = ((X["AMT_INCOME_TOTAL"] - 0.0005 * X["AMT_CREDIT"]) > 45000).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegression(max_iter=200, random_state=42))
    ])
    pipe.fit(X, y)

    # artefacts
    joblib.dump(pipe, tmp / "scoring_model.joblib")
    (tmp / "decision_threshold.json").write_text(json.dumps({"threshold": 0.50}))
    (tmp / "input_columns.json").write_text(json.dumps(list(X.columns)))
    return tmp

@pytest.fixture()
def api_client(dummy_model_dir, monkeypatch):
    # pointe l’API vers le modèle factice
    monkeypatch.setenv("MODEL_DIR", str(dummy_model_dir))
    # Import tardif pour que MODEL_DIR soit vu
    # ATTENTION: adapter le chemin si ton module est packaging différemment
    from api import app as app_module
    importlib.reload(app_module)          # recharge api/app.py
    client = TestClient(app_module.app)   # FastAPI app
    return client
