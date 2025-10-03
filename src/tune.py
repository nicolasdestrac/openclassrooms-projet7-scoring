# src/tune.py
import os
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import mlflow

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# LightGBM (optionnel selon le modèle)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# --- util métier : on s'appuie sur tes métriques existantes
from .metrics import evaluate_all  # doit renvoyer entre autres "business_score_norm"

# ------------- Helpers config & data -------------
def read_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cfg_get(cfg: dict, dotted_key: str, default=None):
    """Accès style 'a.b.c' sur un dict imbriqué."""
    cur = cfg
    for k in dotted_key.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def load_training_df(cfg: dict) -> pd.DataFrame:
    train_csv = cfg_get(cfg, "data.train_csv")
    if not train_csv:
        raise ValueError("Chemin data.train_csv manquant dans conf/params.yaml")
    df = pd.read_csv(train_csv)
    # Essaie d'ajouter un feature engineering léger si dispo
    try:
        from .features import basic_feature_engineering
        df = basic_feature_engineering(df)
    except Exception:
        pass
    return df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Sépare X/y, détecte colonnes num/objet
    exclude = [c for c in ["TARGET", "target", "Sk_ID_CURR", "SK_ID_CURR"] if c in df.columns]
    X = df.drop(columns=[c for c in ["TARGET", "target"] if c in df.columns], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre

def build_estimator(model_type: str, cfg: dict):
    if model_type == "logreg":
        # Pour paralléliser correctement avec GridSearchCV, on met n_jobs=1 ici
        return LogisticRegression(
            penalty=cfg_get(cfg, "model.logreg.penalty", "l2"),
            solver=cfg_get(cfg, "model.logreg.solver", "saga"),
            max_iter=int(cfg_get(cfg, "model.logreg.max_iter", 2000)),
            class_weight=cfg_get(cfg, "model.logreg.class_weight", "balanced"),
            n_jobs=1,
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=int(cfg_get(cfg, "model.rf.n_estimators", 500)),
            max_depth=cfg_get(cfg, "model.rf.max_depth", None),
            min_samples_split=int(cfg_get(cfg, "model.rf.min_samples_split", 2)),
            min_samples_leaf=int(cfg_get(cfg, "model.rf.min_samples_leaf", 1)),
            max_features=cfg_get(cfg, "model.rf.max_features", "sqrt"),
            n_jobs=1,  # laisser la parallélisation à Grid/RandomizedSearchCV
            class_weight=cfg_get(cfg, "model.rf.class_weight", "balanced"),
            random_state=int(cfg_get(cfg, "cv.random_state", 42)),
        )
    elif model_type == "lgbm":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm non installé. `pip install lightgbm`")
        return lgb.LGBMClassifier(
            n_estimators=int(cfg_get(cfg, "model.lgbm.n_estimators", 1000)),
            learning_rate=float(cfg_get(cfg, "model.lgbm.learning_rate", 0.03)),
            num_leaves=int(cfg_get(cfg, "model.lgbm.num_leaves", 63)),
            max_depth=int(cfg_get(cfg, "model.lgbm.max_depth", -1)),
            min_child_samples=int(cfg_get(cfg, "model.lgbm.min_child_samples", 100)),
            subsample=float(cfg_get(cfg, "model.lgbm.subsample", 0.8)),
            colsample_bytree=float(cfg_get(cfg, "model.lgbm.colsample_bytree", 0.8)),
            reg_lambda=float(cfg_get(cfg, "model.lgbm.reg_lambda", 1.0)),
            reg_alpha=float(cfg_get(cfg, "model.lgbm.reg_alpha", 0.1)),
            class_weight=cfg_get(cfg, "model.lgbm.class_weight", "balanced"),
            n_jobs=1,      # important pour éviter la sur-souscription
            verbosity=-1,  # réduit le bruit LightGBM
        )
    else:
        raise ValueError(f"model.type inconnu: {model_type}")

# ----------- Scorer métier (business_score_norm) -----------
def make_business_scorer(fn_cost: float, fp_cost: float, threshold_grid: int):
    def _metric(y_true, y_score):
        m = evaluate_all(y_true, y_score, fn_cost, fp_cost, threshold_grid)
        return float(m["business_score_norm"])
    return make_scorer(_metric, greater_is_better=True, needs_proba=True)

# ----------- Param spaces (raisonnables et silencieux) -----------
def get_param_space(model_type: str):
    if model_type == "logreg":
        return {
            "est__C": np.logspace(-2, 2, 10),
            "est__penalty": ["l2"],
            "est__solver": ["saga", "liblinear"],
        }
    if model_type == "rf":
        return {
            "est__n_estimators": [300, 600, 900],
            "est__max_depth": [None, 10, 20, 30],
            "est__min_samples_split": [2, 5, 10],
            "est__min_samples_leaf": [1, 2, 4],
            "est__max_features": ["sqrt", "log2", 0.5],
        }
    if model_type == "lgbm":
        # on évite early stopping dans le search pour simplifier / stabiliser
        return {
            "est__num_leaves": [31, 63, 127],
            "est__max_depth": [-1, 8, 12],
            "est__min_child_samples": [50, 100, 200],
            "est__subsample": [0.7, 0.8, 0.9],
            "est__colsample_bytree": [0.7, 0.8, 0.9],
            "est__learning_rate": [0.02, 0.03, 0.05],
            "est__reg_lambda": [0.0, 1.0, 5.0],
            "est__reg_alpha": [0.0, 0.1, 0.5],
            # n_estimators fixé (assez grand) pour éviter l'early_stopping ici
            "est__n_estimators": [800, 1200, 1600],
        }
    raise ValueError(model_type)

# ------------------- Main -------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/params.yaml")
    p.add_argument("--model", choices=["lgbm", "logreg", "rf"], help="override model.type")
    p.add_argument("--random", action="store_true", help="RandomizedSearch au lieu de GridSearch")
    p.add_argument("--n-iter", type=int, default=50, help="n_iter pour RandomizedSearch")
    p.add_argument("--n-jobs", type=int, default=-1, help="parallélisme SearchCV (folds/candidats)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = read_yaml_config(args.config)

    model_type = args.model or cfg_get(cfg, "model.type", "lgbm")
    fn_cost = float(cfg_get(cfg, "cost.fn", 10.0))
    fp_cost = float(cfg_get(cfg, "cost.fp", 1.0))
    thr_grid = int(cfg_get(cfg, "cost.threshold_grid", 501))

    # lecture data
    df = load_training_df(cfg)
    target_col = "TARGET" if "TARGET" in df.columns else "target"
    if target_col not in df.columns:
        raise ValueError("Colonne cible 'TARGET' (ou 'target') absente du train.")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    pre = build_preprocessor(df)
    est = build_estimator(model_type, cfg)
    pipe = Pipeline([("prep", pre), ("est", est)])

    # CV et scorer
    skf = StratifiedKFold(
        n_splits=int(cfg_get(cfg, "cv.n_splits", 5)),
        shuffle=bool(cfg_get(cfg, "cv.shuffle", True)),
        random_state=int(cfg_get(cfg, "cv.random_state", args.seed)),
    )
    scorer = make_business_scorer(fn_cost, fp_cost, thr_grid)

    # espace d'hyperparams
    param_space = get_param_space(model_type)

    # Parallélisme côté SearchCV; **très important**: on a mis n_jobs=1 dans les estimators
    n_jobs = int(args.n_jobs)

    # MLflow
    tracking_uri = cfg_get(cfg, "mlflow.default_tracking_uri", None) or os.getenv(
        str(cfg_get(cfg, "mlflow.tracking_uri", "MLFLOW_TRACKING_URI")), None
    )
    experiment = cfg_get(cfg, "mlflow.default_experiment", None) or os.getenv(
        str(cfg_get(cfg, "mlflow.experiment", "MLFLOW_EXPERIMENT")), None
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment:
        mlflow.set_experiment(experiment)

    run_name = f"tune_{model_type}_{'random' if args.random else 'grid'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "search_type": "random" if args.random else "grid",
            "cv_splits": skf.get_n_splits(),
            "n_jobs": n_jobs,
            "random_state": args.seed,
            "model_type": model_type,
        })

        # Choix du SearchCV
        if args.random:
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_space,
                n_iter=int(args.n_iter),
                scoring=scorer,
                cv=skf,
                n_jobs=n_jobs,
                refit=True,
                verbose=1,
                random_state=args.seed,
                return_train_score=False,
            )
        else:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_space,
                scoring=scorer,
                cv=skf,
                n_jobs=n_jobs,
                refit=True,
                verbose=1,
                return_train_score=False,
            )

        # fit
        search.fit(X, y)

        # log best
        mlflow.log_metric("cv_best_business_score_norm", float(search.best_score_))
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})

        # tableau complet des essais
        cvres = pd.DataFrame(search.cv_results_)
        out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{run_name}_cv_results.csv"
        cvres.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))

        # log du meilleur pipeline complet (prep + est)
        try:
            if model_type == "lgbm" and HAS_LGBM:
                mlflow.lightgbm.log_model(search.best_estimator_, artifact_path="best_model")
            else:
                mlflow.sklearn.log_model(search.best_estimator_, artifact_path="best_model")
        except Exception:
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="best_model")

        print(f"✅ Best business_score_norm={search.best_score_:.4f}")
        print("🏆 Best params:")
        for k, v in search.best_params_.items():
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
