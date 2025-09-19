import os, json
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from .data import load_raw, ensure_dirs
from .features import make_train_test
from .metrics import evaluate_all

def cfg_get(cfg, key, default=None):
    """Accède à cfg[key] que cfg soit un dict ou un objet à attributs."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

@dataclass
class Config:
    data: dict
    cv: dict
    cost: dict
    model: dict
    mlflow: dict
    artifacts: dict

def read_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

def get_estimator(cfg: Config):
    if cfg.model["type"] == "lgbm":
        return lgb.LGBMClassifier(**cfg.model["lgbm"])
    elif cfg.model["type"] == "logreg":
        return LogisticRegression(**cfg.model["logreg"])
    elif cfg.model["type"] == "rf":
        return RandomForestClassifier(**cfg["model"]["rf"])
    else:
        raise ValueError(f"Unknown model.type: {cfg.model}")

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
    print("Artifact Loc  ->", exp.artifact_location)

def crossval_oof(X: pd.DataFrame, y: pd.Series, preprocessor, estimator, cfg: Config):
    skf = StratifiedKFold(
        n_splits=cfg.cv["n_splits"],
        shuffle=cfg.cv.get("shuffle", True),
        random_state=cfg.cv.get("random_state", 42)
    )
    oof_prob = np.zeros(len(y), dtype=float)
    fold_rows = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        prep = clone(preprocessor).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        X_tr = prep.transform(X.iloc[tr_idx])
        X_va = prep.transform(X.iloc[va_idx])
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()

        est = clone(estimator)
        if isinstance(est, lgb.LGBMClassifier):
            cv_cfg = cfg_get(cfg, "cv", {})
            esr = int(cfg_get(cv_cfg, "early_stopping_rounds", 200))
            log_period = int(cfg_get(cv_cfg, "log_period", 50))

            callbacks = [
                lgb.early_stopping(stopping_rounds=esr),
                lgb.log_evaluation(period=log_period),
            ]

            if esr:
                callbacks.append(lgb.early_stopping(stopping_rounds=esr, verbose=False))

            est.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=callbacks,
            )

            n_best = int(getattr(est, "best_iteration_", est.n_estimators_))
            best_iters.append(n_best)
            prob_va = est.predict_proba(X_va, num_iteration=n_best)[:, 1]
        else:
            est.fit(X_tr, y_tr)
            prob_va = est.predict_proba(X_va)[:, 1]

        oof_prob[va_idx] = prob_va

        from .metrics import evaluate_all
        m = evaluate_all(y_va, prob_va, cfg.cost["fn"], cfg.cost["fp"], cfg.cost["threshold_grid"])
        m.update({"fold": fold})
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    best_iter_median = int(np.median(best_iters)) if len(best_iters) else None
    return oof_prob, fold_df, best_iter_median

def fit_final_pipeline(X: pd.DataFrame, y: pd.Series, preprocessor, estimator, best_iter_median: int | None):
    prep_full = clone(preprocessor).fit(X, y)
    est = clone(estimator)
    if isinstance(est, lgb.LGBMClassifier) and best_iter_median:
        est.set_params(n_estimators=int(best_iter_median))
    pipe = Pipeline([("prep", prep_full), ("clf", est)])
    pipe.fit(X, y)
    return pipe

def main(config_path: str = "conf/params.yaml"):
    cfg = read_config(config_path)

    # MLflow
    setup_mlflow(cfg)

    # IO dirs
    ensure_dirs(cfg.artifacts["models_dir"], cfg.artifacts["reports_dir"])

    # Data
    train_raw, test_raw = load_raw(cfg.data["train_csv"], cfg.data["test_csv"])
    X, y, _ = make_train_test(train_raw, test_raw)

    # Preprocess & model
    preprocessor = build_preprocessor(X)
    estimator = get_estimator(cfg)

    # OOF
    oof_prob, fold_df, best_iter_median = crossval_oof(X, y, preprocessor, estimator, cfg)
    m_oof = evaluate_all(y.to_numpy(), oof_prob, cfg.cost["fn"], cfg.cost["fp"], cfg.cost["threshold_grid"])

    # Final fit
    final_pipe = fit_final_pipeline(X, y, preprocessor, estimator, best_iter_median)

    # Artifacts locaux
    reports_dir = Path(cfg.artifacts["reports_dir"])
    models_dir  = Path(cfg.artifacts["models_dir"])
    fold_path   = reports_dir / "cv_metrics_by_fold.csv"
    thr_path    = models_dir  / "decision_threshold.json"
    fold_df.to_csv(fold_path, index=False)
    with open(thr_path, "w") as f:
        json.dump({"threshold": m_oof["best_threshold"],
                   "source": "OOF",
                   "cost_fn": cfg.cost["fn"],
                   "cost_fp": cfg.cost["fp"]}, f)

    # MLflow logging
    with mlflow.start_run(run_name=f"{cfg.model['type']}_oof_remote") as run:
        # params
        mlflow.log_param("model_type", cfg.model["type"])
        for k, v in (cfg.model.get(cfg.model["type"], {}) or {}).items():
            mlflow.log_param(f"{cfg.model['type']}__{k}", v)
        mlflow.log_param("cv_n_splits", cfg.cv["n_splits"])
        mlflow.log_param("early_stopping_rounds", cfg.cv["early_stopping_rounds"])
        if best_iter_median:
            mlflow.log_param("best_iter_median", int(best_iter_median))
        mlflow.log_param("cost_fn", cfg.cost["fn"])
        mlflow.log_param("cost_fp", cfg.cost["fp"])

        # metrics OOF
        for k, v in m_oof.items():
            mlflow.log_metric(f"oof_{k}", float(v))

        # artifacts
        mlflow.log_artifact(str(fold_path))
        mlflow.log_artifact(str(thr_path))

        # signature d’input: échantillon réaliste (cast int -> float64 si NaN possibles)
        Xs = X.iloc[:200].copy()
        int_cols = Xs.select_dtypes(include=["int64","int32"]).columns
        Xs[int_cols] = Xs[int_cols].astype("float64")
        signature = None
        try:
            signature = infer_signature(Xs, final_pipe.predict_proba(Xs)[:, 1])
        except Exception:
            pass

        mlflow.sklearn.log_model(final_pipe, artifact_path="model", signature=signature)

    print("✅ Run MLflow terminé | AUC OOF={:.4f} | Seuil*={:.3f} | Coût={:.0f} | Coût/10k={:.1f}"
          .format(m_oof["auc"], m_oof["best_threshold"], m_oof["business_cost"], m_oof["business_cost_per10k"]))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/params.yaml")
    args = p.parse_args()
    main(args.config)
