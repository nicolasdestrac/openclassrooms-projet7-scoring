import os, json, warnings, yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from .data import load_raw
from .features import make_train_test
from .train import build_preprocessor  # réutilise ton préproc
from .metrics import make_business_scorer

warnings.filterwarnings("ignore")

@dataclass
class Config:
    data: dict
    cv: dict
    cost: dict
    model: dict
    mlflow: dict
    artifacts: dict
    tuning: dict | None = None

def read_config(path: str) -> "Config":
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))

def get_estimator(cfg: Config):
    t = cfg.model["type"]
    if t == "lgbm":
        return lgb.LGBMClassifier(**cfg.model.get("lgbm", {}))
    if t == "logreg":
        return LogisticRegression(**cfg.model.get("logreg", {}))
    if t == "rf":
        return RandomForestClassifier(**cfg.model.get("rf", {}))
    raise ValueError(f"Unknown model.type: {t}")

def setup_mlflow(cfg: Config):
    tracking_uri = os.getenv(cfg.mlflow["tracking_uri_env"], cfg.mlflow["default_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    exp_path = os.getenv(cfg.mlflow["experiment_env"], cfg.mlflow["default_experiment"])
    mlflow.set_experiment(exp_path)
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_path)
    print("Tracking URI ->", mlflow.get_tracking_uri())
    print("Experiment    ->", exp.name)

def setup_mlflow_safe(cfg):
    try:
        setup_mlflow(cfg)
    except MlflowException:
        print("[tune] Databricks indisponible → fallback local MLflow ./mlruns")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("tuning-local")

def main(config_path="conf/params.yaml", use_random=True):
    cfg = read_config(config_path)
    setup_mlflow_safe(cfg)

    # Data
    train_raw, test_raw = load_raw(cfg.data["train_csv"], cfg.data["test_csv"])
    X, y, _ = make_train_test(train_raw, test_raw)

    # Pipeline de base
    preproc = build_preprocessor(X)
    est = get_estimator(cfg)
    base = Pipeline([("prep", preproc), ("clf", est)])

    # CV + scorer métier
    cv = StratifiedKFold(
        n_splits=cfg.cv.get("n_splits", 5),
        shuffle=cfg.cv.get("shuffle", True),
        random_state=cfg.cv.get("random_state", 42),
    )
    scorer = make_business_scorer(
        fn_cost=float(cfg.cost["fn"]),
        fp_cost=float(cfg.cost["fp"]),
        grid=int(cfg.cost.get("threshold_grid", 501)),
    )

    # Espaces d’hyperparams
    grids = {
        "lgbm": {
            "clf__num_leaves": [31, 63, 127],
            "clf__min_child_samples": [10, 50, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__n_estimators": [300, 600, 1000],
        },
        "logreg": {
            "clf__C": np.logspace(-3, 2, 6),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [200, 500],
        },
        "rf": {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 8, 16, 32],
            "clf__min_samples_split": [2, 5, 10],
        },
    }

    model_type = cfg.model["type"]
    param_grid = grids[model_type]

    print(f"Hyperparam search for {model_type} | use_random={use_random}")
    with mlflow.start_run(run_name=f"{model_type}_tuning"):
        if use_random:
            search = RandomizedSearchCV(
                base, param_distributions=param_grid,
                n_iter=cfg.tuning.get("n_iter", 20) if cfg.tuning else 20,
                scoring=scorer, cv=cv, n_jobs=-1, verbose=1, random_state=42,
            )
        else:
            search = GridSearchCV(
                base, param_grid=param_grid, scoring=scorer,
                cv=cv, n_jobs=-1, verbose=1
            )

        search.fit(X, y)

        # log MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("best_score", float(search.best_score_))
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)

        # export best params -> YAML pour train.py
        best = {model_type: {}}
        for k, v in search.best_params_.items():
            if k.startswith("clf__"):
                best[model_type][k.replace("clf__", "")] = v
        out = Path("conf/best_params.yaml")
        out.write_text(yaml.safe_dump(best, sort_keys=True, allow_unicode=True))
        mlflow.log_artifact(str(out))
        print("Best params ->", best)
        print("Best (negative cost) score ->", search.best_score_)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/params.yaml")
    p.add_argument("--random", action="store_true", help="RandomizedSearch (default)")
    p.add_argument("--grid", action="store_true", help="GridSearch")
    args = p.parse_args()
    main(args.config, use_random=not args.grid)
