# src/tune.py
import os
import argparse
from pathlib import Path
import warnings

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

# LightGBM (optionnel)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ================= Helpers config & data =================
def read_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: dict, dotted_key: str, default=None):
    """Accès style 'a.b.c' dans un dict imbriqué."""
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
        raise ValueError("Chemin 'data.train_csv' manquant dans conf/params.yaml")
    df = pd.read_csv(train_csv)

    # Feature eng. léger si dispo
    try:
        from .features import basic_feature_engineering
        df = basic_feature_engineering(df)
    except Exception:
        pass
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    target_col = "TARGET" if "TARGET" in df.columns else "target"
    X = df.drop(columns=[target_col], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_estimator(model_type: str, cfg: dict):
    # IMPORTANT: n_jobs=1 ici pour éviter le double-parallélisme avec SearchCV
    if model_type == "logreg":
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
            class_weight=cfg_get(cfg, "model.rf.class_weight", "balanced"),
            random_state=int(cfg_get(cfg, "cv.random_state", 42)),
            n_jobs=1,
        )
    elif model_type == "lgbm":
        if not HAS_LGBM:
            raise RuntimeError("lightgbm non installé. `pip install lightgbm`")
        return lgb.LGBMClassifier(
            n_estimators=int(cfg_get(cfg, "model.lgbm.n_estimators", 1200)),
            learning_rate=float(cfg_get(cfg, "model.lgbm.learning_rate", 0.03)),
            num_leaves=int(cfg_get(cfg, "model.lgbm.num_leaves", 63)),
            max_depth=int(cfg_get(cfg, "model.lgbm.max_depth", -1)),
            min_child_samples=int(cfg_get(cfg, "model.lgbm.min_child_samples", 100)),
            subsample=float(cfg_get(cfg, "model.lgbm.subsample", 0.8)),
            colsample_bytree=float(cfg_get(cfg, "model.lgbm.colsample_bytree", 0.8)),
            reg_lambda=float(cfg_get(cfg, "model.lgbm.reg_lambda", 1.0)),
            reg_alpha=float(cfg_get(cfg, "model.lgbm.reg_alpha", 0.1)),
            class_weight=cfg_get(cfg, "model.lgbm.class_weight", "balanced"),
            n_jobs=1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"model.type inconnu: {model_type}")


# ================= Scorer métier =================
# On s’appuie sur evaluate_all() qui renvoie "business_score_norm" ∈ [0,1]
from .metrics import evaluate_all


def make_business_scorer(fn_cost: float, fp_cost: float, threshold_grid: int):
    def _metric(y_true, y_score):
        m = evaluate_all(y_true, y_score, fn_cost, fp_cost, threshold_grid)
        return float(m["business_score_norm"])
    return make_scorer(_metric, greater_is_better=True, needs_proba=True)


# ================= Espaces d’hyperparamètres =================
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
            "est__max_depth": [None, 12, 20, 30],
            "est__min_samples_split": [2, 5, 10],
            "est__min_samples_leaf": [1, 2, 4],
            "est__max_features": ["sqrt", "log2", 0.5],
        }
    if model_type == "lgbm":
        # Pas d'early stopping dans le search pour simplifier/stabiliser
        return {
            "est__num_leaves": [31, 63, 127],
            "est__max_depth": [-1, 8, 12],
            "est__min_child_samples": [50, 100, 200],
            "est__subsample": [0.7, 0.8, 0.9],
            "est__colsample_bytree": [0.7, 0.8, 0.9],
            "est__learning_rate": [0.02, 0.03, 0.05],
            "est__reg_lambda": [0.0, 1.0, 5.0],
            "est__reg_alpha": [0.0, 0.1, 0.5],
            "est__n_estimators": [800, 1200, 1600],
        }
    raise ValueError(model_type)


# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/params.yaml")
    ap.add_argument("--model", choices=["lgbm", "logreg", "rf"], help="override model.type")
    ap.add_argument("--random", action="store_true", help="RandomizedSearch au lieu de GridSearch")
    ap.add_argument("--n-iter", type=int, default=50, help="n_iter pour RandomizedSearch")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallélisme SearchCV (folds/candidats)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", type=int, default=1, help="0 pour silencieux")
    args = ap.parse_args()

    cfg = read_yaml_config(args.config)

    model_type = args.model or cfg_get(cfg, "model.type", "lgbm")
    fn_cost = float(cfg_get(cfg, "cost.fn", 10.0))
    fp_cost = float(cfg_get(cfg, "cost.fp", 1.0))
    thr_grid = int(cfg_get(cfg, "cost.threshold_grid", 501))

    df = load_training_df(cfg)
    target_col = "TARGET" if "TARGET" in df.columns else "target"
    if target_col not in df.columns:
        raise ValueError("Colonne cible 'TARGET' (ou 'target') absente du train.")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    pre = build_preprocessor(df)
    est = build_estimator(model_type, cfg)
    pipe = Pipeline([("prep", pre), ("est", est)])

    skf = StratifiedKFold(
        n_splits=int(cfg_get(cfg, "cv.n_splits", 5)),
        shuffle=bool(cfg_get(cfg, "cv.shuffle", True)),
        random_state=int(cfg_get(cfg, "cv.random_state", args.seed)),
    )
    scorer = make_business_scorer(fn_cost, fp_cost, thr_grid)

    param_space = get_param_space(model_type)
    n_jobs = int(args.n_jobs)

    # MLflow (local par défaut si pas de conf Databricks)
    tracking_uri = os.getenv(cfg_get(cfg, "mlflow.tracking_uri_env", "MLFLOW_TRACKING_URI")) \
                   or cfg_get(cfg, "mlflow.default_tracking_uri", None)
    experiment = os.getenv(cfg_get(cfg, "mlflow.experiment_env", "MLFLOW_EXPERIMENT")) \
                 or cfg_get(cfg, "mlflow.default_experiment", None)
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

        if args.random:
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_space,
                n_iter=int(args.n_iter),
                scoring=scorer,
                cv=skf,
                n_jobs=n_jobs,
                refit=True,
                verbose=args.verbose,
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
                verbose=args.verbose,
                return_train_score=False,
            )

        search.fit(X, y)

        # meilleurs résultats
        mlflow.log_metric("cv_best_business_score_norm", float(search.best_score_))
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})

        # dump complet des essais
        cvres = pd.DataFrame(search.cv_results_)
        out_dir = Path("reports"); out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{run_name}_cv_results.csv"
        cvres.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))

        # log du meilleur pipeline (prep + est)
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
