import os, json, re, logging, warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

import mlflow, mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# plots / explainability
import matplotlib.pyplot as plt
import shap

from .data import load_raw, ensure_dirs
from .features import make_train_test
from .metrics import evaluate_all

# --- Warnings : on fait le ménage ---
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.utils.validation",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

_BAD_CHARS = re.compile(r"[^0-9A-Za-z_]+")
_SENTINEL = object()

def make_lgbm_safe(names):
    cleaned = [_BAD_CHARS.sub("_", str(n)) for n in names]
    cleaned = [re.sub(r"_+", "_", s).strip("_") for s in cleaned]
    seen, out = {}, []
    for s in cleaned:
        i = seen.get(s, 0)
        out.append(s if i == 0 else f"{s}__{i}")
        seen[s] = i + 1
    return out

def cfg_get(obj, dotted_key: str, default=None):
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, _SENTINEL)
        else:
            cur = getattr(cur, part, _SENTINEL)
        if cur is _SENTINEL:
            return default
    return cur

@dataclass
class Config:
    data: dict
    cv: dict
    cost: dict
    model: dict
    mlflow: dict
    artifacts: dict
    tuning: dict | None = None  # optionnel

def read_config(path: str) -> "Config":
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
        return RandomForestClassifier(**cfg.model["rf"])
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
        n_splits=cfg_get(cfg, "cv.n_splits", 5),
        shuffle=cfg_get(cfg, "cv.shuffle", True),
        random_state=cfg_get(cfg, "cv.random_state", 42),
    )

    oof_prob = np.zeros(len(y), dtype=float)
    fold_rows, best_iters = [], []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="CV folds"), 1):
        prep = clone(preprocessor).fit(X.iloc[tr_idx], y.iloc[tr_idx])

        # noms de features + nettoyage pour LightGBM
        try:
            raw_feat_names = prep.get_feature_names_out()
        except Exception:
            raw_feat_names = [f"f_{i}" for i in range(prep.transform(X.iloc[tr_idx]).shape[1])]
        feat_names = make_lgbm_safe(raw_feat_names)

        # data
        X_tr_arr = prep.transform(X.iloc[tr_idx])
        X_va_arr = prep.transform(X.iloc[va_idx])
        X_tr = pd.DataFrame(X_tr_arr, index=X.index[tr_idx], columns=feat_names)
        X_va = pd.DataFrame(X_va_arr, index=X.index[va_idx], columns=feat_names)
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()

        est = clone(estimator)
        t0 = perf_counter()

        if isinstance(est, lgb.LGBMClassifier):
            esr = int(cfg_get(cfg, "cv.early_stopping_rounds", 200))
            log_period = int(cfg_get(cfg, "cv.log_period", 100))
            callbacks = [
                lgb.early_stopping(stopping_rounds=esr, verbose=False),
                lgb.log_evaluation(period=log_period),
            ]
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

        fit_secs = perf_counter() - t0
        oof_prob[va_idx] = prob_va

        # métriques fold (dont métrique métier)
        fold_auc = roc_auc_score(y_va, prob_va)
        m = evaluate_all(y_va, prob_va, cfg.cost["fn"], cfg.cost["fp"], cfg.cost["threshold_grid"])
        m.update({"fold": fold, "fit_secs": fit_secs})
        if hasattr(est, "n_iter_"):
            n_iter = est.n_iter_
            if hasattr(n_iter, "__len__"):
                n_iter = n_iter[0]
            m["n_iter"] = int(n_iter)

        fold_rows.append(m)
        print(f"Fold {fold}: AUC={fold_auc:.4f}, secs={fit_secs:.1f}")

    fold_df = pd.DataFrame(fold_rows)
    best_iter_median = int(np.median(best_iters)) if best_iters else None
    return oof_prob, fold_df, best_iter_median

def fit_final_pipeline(X: pd.DataFrame, y: pd.Series, preprocessor, estimator, best_iter_median: int | None):
    prep_full = clone(preprocessor).fit(X, y)
    est = clone(estimator)
    if isinstance(est, lgb.LGBMClassifier) and best_iter_median:
        est.set_params(n_estimators=int(best_iter_median))
    pipe = Pipeline([("prep", prep_full), ("clf", est)])
    pipe.fit(X, y)
    return pipe

def plot_and_log_feature_importance(final_pipe, X: pd.DataFrame, models_dir: Path):
    """Sauvegarde CSV + barplot des importances; SHAP pour LGBM."""
    prep = final_pipe.named_steps["prep"]
    clf = final_pipe.named_steps["clf"]

    try:
        feat_names = make_lgbm_safe(prep.get_feature_names_out())
    except Exception:
        feat_names = [f"f_{i}" for i in range(prep.transform(X.iloc[:5]).shape[1])]

    # Importances globales (tree-based ou logreg)
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        imp = np.abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
        imp = np.asarray(imp, dtype=float)
    else:
        imp = None

    fi_csv = None
    fi_png = None

    if imp is not None and len(imp) == len(feat_names):
        fi = (pd.DataFrame({"feature": feat_names, "importance": imp})
                .sort_values("importance", ascending=False))
        fi_csv = models_dir / "feature_importances.csv"
        fi.to_csv(fi_csv, index=False)

        top = fi.head(30).iloc[::-1]  # barh plus lisible
        plt.figure(figsize=(8, 10))
        plt.barh(top["feature"], top["importance"])
        plt.title("Top 30 feature importances")
        plt.tight_layout()
        fi_png = models_dir / "feature_importances_top30.png"
        plt.savefig(fi_png, dpi=150)
        plt.close()

    # SHAP (uniquement pour LGBM afin de rester rapide)
    shap_png = None
    if isinstance(clf, lgb.LGBMClassifier):
        try:
            Xs = X.sample(min(2000, len(X)), random_state=42)  # sous-échantillon
            Xs_tr = prep.transform(Xs)
            expl = shap.TreeExplainer(clf)
            shap_values = expl(Xs_tr)
            plt.figure()
            shap.plots.beeswarm(shap_values, max_display=30, show=False)
            shap_png = models_dir / "shap_beeswarm_top30.png"
            plt.tight_layout()
            plt.savefig(shap_png, dpi=150)
            plt.close()
        except Exception:
            pass

    # Log artifacts si présents
    for p in [fi_csv, fi_png, shap_png]:
        if p and Path(p).exists():
            mlflow.log_artifact(str(p))

def main(config_path: str = "conf/params.yaml"):
    cfg = read_config(config_path)

    # MLflow
    setup_mlflow(cfg)

    # IO
    ensure_dirs(cfg.artifacts["models_dir"], cfg.artifacts["reports_dir"])

    # Data
    train_raw, test_raw = load_raw(cfg.data["train_csv"], cfg.data["test_csv"])
    X, y, _ = make_train_test(train_raw, test_raw)

    # Preprocess & model
    preprocessor = build_preprocessor(X)
    estimator = get_estimator(cfg)

    # OOF CV
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
        json.dump(
            {
                "threshold": m_oof["threshold_opt"],
                "source": "OOF",
                "cost_fn": cfg.cost["fn"],
                "cost_fp": cfg.cost["fp"],
            },
            f,
        )

    # MLflow run (sécurise un run propre)
    if mlflow.active_run() is not None:
        print(f"Ending stray active run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    with mlflow.start_run(run_name=f"{cfg.model['type']}_oof_remote"):
        # Params
        mlflow.log_param("model_type", cfg.model["type"])
        for k, v in (cfg.model.get(cfg.model["type"], {}) or {}).items():
            mlflow.log_param(f"{cfg.model['type']}__{k}", v)
        mlflow.log_param("cv_n_splits", cfg.cv["n_splits"])
        mlflow.log_param("early_stopping_rounds", cfg.cv.get("early_stopping_rounds", None))
        if best_iter_median:
            mlflow.log_param("best_iter_median", int(best_iter_median))
        mlflow.log_param("cost_fn", cfg.cost["fn"])
        mlflow.log_param("cost_fp", cfg.cost["fp"])

        # Metrics OOF (préfixées "oof_")
        for k, v in m_oof.items():
            if isinstance(v, (int, float, np.floating)):
                mlflow.log_metric(f"oof_{k}", float(v))

        # Log par fold (AUC, AP, Brier, KS, coût, seuil, temps, itérations… + business_score)
        for _, r in fold_df.iterrows():
            step = int(r["fold"])
            for k in [
                "auc", "ap", "brier", "ks",
                "business_cost", "business_cost_per10k", "business_score",
                "threshold_opt",
                "fit_secs", "n_iter",
            ]:
                if k in r and pd.notnull(r[k]):
                    mlflow.log_metric(f"fold_{k}", float(r[k]), step=step)

        # Artifacts: CSV/JSON + importances/SHAP + modèle
        mlflow.log_artifact(str(fold_path))
        mlflow.log_artifact(str(thr_path))
        plot_and_log_feature_importance(final_pipe, X, models_dir)

        # Signature & modèle
        Xs = X.iloc[:200].copy()
        int_cols = Xs.select_dtypes(include=["int64","int32"]).columns
        Xs[int_cols] = Xs[int_cols].astype("float64")
        signature = None
        try:
            signature = infer_signature(Xs, final_pipe.predict_proba(Xs)[:, 1])
        except Exception:
            pass
        mlflow.sklearn.log_model(final_pipe, name="model", signature=signature)

    print(
        "✅ Run MLflow terminé | AUC OOF={:.4f} | Score={:.3f} | Seuil*={:.3f} | Coût={:.0f} | Coût/10k={:.1f}".format(
            m_oof["auc"], m_oof["business_score"], m_oof["threshold_opt"],
            m_oof["business_cost"], m_oof["business_cost_per10k"]
        )
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/params.yaml")
    args = p.parse_args()
    main(args.config)
