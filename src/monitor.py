"""
Monitoring de data drift avec Evidently.

Principe (Option 1) :
- Charge deux datasets (référence & courant)
- Ne garde que l'intersection des colonnes présentes dans les DEUX jeux
- Filtre les colonnes vides (tout NaN) ou constantes des deux côtés
- (Optionnel) Si un schéma est fourni, on l'utilise comme préférence (intersection avec les colonnes valides)
- Génère un rapport HTML Evidently + un résumé JSON + une alerte simple

Exemples :
  python -m src.monitor \
    --ref data/raw/application_train.csv \
    --cur data/raw/application_test.csv \
    --out artifacts/reports \
    --sample 50000

  # Simulation de drift pour la démo/CE2
  python -m src.monitor \
    --ref data/raw/application_train.csv \
    --cur data/raw/application_test.csv \
    --out artifacts/reports \
    --simulate \
    --money-col AMT_CREDIT --money-factor 1.10 \
    --cat-col NAME_INCOME_TYPE --cat-rate 0.25 \
    --sample 50000
"""
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional, Tuple, List

import os, yaml

import numpy as np
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# -----------------------------
# Utilities E/S
# -----------------------------
def read_any(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    raise ValueError(f"Format non supporté: {path.suffix} (CSV/Parquet)")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_params_yaml(path: Path = Path("conf/params.yaml")) -> dict:
    if yaml is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# -----------------------------
# Sélection auto des colonnes valides
# -----------------------------
def non_empty_non_constant_cols(df: pd.DataFrame) -> List[str]:
    keep: List[str] = []
    for c in df.columns:
        s = df[c]
        if s.notna().sum() == 0:
            continue  # tout NaN
        if s.dropna().nunique() <= 1:
            continue  # constant
        keep.append(c)
    return keep


def align_clean(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    prefer_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    - Intersection des colonnes ref/cur
    - Si 'prefer_cols' est fourni, intersecte aussi avec cette liste
    - Retire les colonnes vides/constantes des DEUX côtés
    """
    common = set(ref.columns) & set(cur.columns)
    if prefer_cols:
        common &= set(prefer_cols)
    common = list(common)

    ref2 = ref[common].copy()
    cur2 = cur[common].copy()

    keep_ref = set(non_empty_non_constant_cols(ref2))
    keep_cur = set(non_empty_non_constant_cols(cur2))
    keep = list(keep_ref & keep_cur)

    return ref2[keep], cur2[keep], keep


def build_colmap(df: pd.DataFrame, target: Optional[str]) -> ColumnMapping:
    num = df.select_dtypes(include=["number"]).columns.tolist()
    cat = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target and target in num:
        num.remove(target)
    if target and target in cat:
        cat.remove(target)
    return ColumnMapping(
        target=target,
        prediction=None,
        numerical_features=num,
        categorical_features=cat,
        task="classification" if target else None,
    )


# -----------------------------
# Simulation de drift
# -----------------------------
def simulate_drift(
    df: pd.DataFrame,
    money_col: Optional[str],
    factor: float,
    cat_col: Optional[str],
    cat_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cur = df.copy()
    if money_col and money_col in cur.columns:
        noise = rng.normal(loc=factor, scale=0.03, size=len(cur))
        cur[money_col] = cur[money_col].astype(float) * noise

    if cat_col and cat_col in cur.columns:
        vals = cur[cat_col].dropna()
        if len(vals):
            major = vals.mode().iloc[0]
            mask = rng.random(len(cur)) < cat_rate
            cur.loc[mask, cat_col] = major
    return cur


# -----------------------------
# Report + summary + alert
# -----------------------------
def run_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    target: Optional[str],
    out_dir: Path,
    drift_share_threshold: float,
) -> Tuple[dict, Path, Path]:
    ensure_dir(out_dir)
    colmap = build_colmap(ref_df, target)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=colmap)

    html_path = out_dir / "evidently_data_drift_report.html"
    report.save_html(str(html_path))

    # Résumé
    res = report.as_dict()["metrics"][0]["result"]  # DataDriftPreset = index 0
    summary = {
        "dataset_drift": bool(res["dataset_drift"]),
        "n_cols": int(res["number_of_columns"]),
        "n_drifted": int(res["number_of_drifted_columns"]),
        "share_drifted": float(res["share_of_drifted_columns"]),
    }
    summary_path = out_dir / "evidently_data_drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Alerte simple
    alert = bool(summary["dataset_drift"] or summary["share_drifted"] > drift_share_threshold)
    reason = (
        "dataset_drift=True"
        if summary["dataset_drift"]
        else f"share_drifted={summary['share_drifted']:.3f}>{drift_share_threshold:.2f}"
        if summary["share_drifted"] > drift_share_threshold
        else "no_alert"
    )
    alert_payload = {
        "alert": alert,
        "reason": reason,
        "threshold": drift_share_threshold,
        "metrics": summary,
        "report_html": str(html_path),
    }
    alert_path = out_dir / "alert.json"
    alert_path.write_text(json.dumps(alert_payload, indent=2), encoding="utf-8")

    return alert_payload, summary_path, html_path


# -----------------------------
# MLflow
# -----------------------------
def resolve_mlflow_config(params: dict) -> tuple[str, str]:
    """
    Renvoie (tracking_uri, experiment_name)
    Priorité :
      1) Variables d'env indiquées dans params['mlflow'] (ex: MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)
      2) Valeurs par défaut définies dans le YAML
      3) Fallback local ./mlruns
    """
    mlp = (params or {}).get("mlflow", {})
    # Noms de variables d'env (ex: "MLFLOW_TRACKING_URI")
    tracking_env_name = mlp.get("tracking_uri_env") or "MLFLOW_TRACKING_URI"
    exp_env_name      = mlp.get("experiment_env")   or "MLFLOW_EXPERIMENT"

    # Valeurs par défaut (ex: "databricks", "/Users/.../projet7")
    default_tracking  = mlp.get("default_tracking_uri") or ""
    default_experiment= mlp.get("default_experiment")   or ""

    tracking_uri = os.getenv(tracking_env_name) or default_tracking
    experiment   = os.getenv(exp_env_name)      or default_experiment

    if not tracking_uri:
        tracking_uri = "file://" + str((Path.cwd() / "mlruns").resolve())
    if not experiment:
        experiment = "Monitoring"

    return tracking_uri, experiment


def try_log_mlflow(out_dir: Path, payload: dict, tracking_uri: str, experiment: str, run_name: str = "monitoring_drift"):
    try:
        import mlflow

        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        from mlflow.tracking import MlflowClient
        try:
            MlflowClient().list_experiments(max_results=1)
        except Exception as e:
            print(f"[mlflow] Auth Databricks requise: {e}")

        print(f"[mlflow] tracking_uri = {mlflow.get_tracking_uri()}")
        print(f"[mlflow] experiment   = {experiment}")

        with mlflow.start_run(run_name=run_name) as run:
            print(f"[mlflow] run_id      = {run.info.run_id}")

            m = payload["metrics"]
            mlflow.log_metric("share_drifted", m["share_drifted"])
            mlflow.log_metric("n_drifted", m["n_drifted"])
            mlflow.log_metric("n_cols", m["n_cols"])
            mlflow.log_metric("dataset_drift", int(m["dataset_drift"]))

            for p in [
                "evidently_data_drift_report.html",
                "evidently_data_drift_summary.json",
                "alert.json",
            ]:
                fp = out_dir / p
                if fp.exists():
                    mlflow.log_artifact(str(fp))
            print("[mlflow] artifacts logged.")
    except Exception as e:
        print(f"[mlflow] WARNING: logging failed: {e}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Génère un rapport Evidently (data drift) + alerte simple.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref", required=True, help="Fichier référence (CSV/Parquet).")
    p.add_argument("--cur", required=True, help="Fichier courant (CSV/Parquet).")
    p.add_argument(
        "--schema",
        default="",
        help="Schéma d'entrée (JSON, optionnel). Si fourni, on intersecte les colonnes valides avec ce schéma.",
    )
    p.add_argument("--target", default=None, help="Nom de la cible si présente (optionnel).")
    p.add_argument("--out", default="artifacts/reports", help="Répertoire de sortie des rapports.")
    p.add_argument("--sample", type=int, default=None, help="Sous-échantillonnage (nb de lignes).")
    p.add_argument("--drift-share-threshold", type=float, default=0.30, help="Seuil d'alerte sur share_drifted.")
    p.add_argument("--simulate", action="store_true", help="Active la simulation de drift sur le dataset courant.")
    p.add_argument("--money-col", default="AMT_CREDIT", help="Colonne numérique à gonfler (si --simulate).")
    p.add_argument("--money-factor", type=float, default=1.10, help="Facteur moyen multiplicatif (si --simulate).")
    p.add_argument("--cat-col", default="NAME_INCOME_TYPE", help="Catégorielle à rebalance (si --simulate).")
    p.add_argument("--cat-rate", type=float, default=0.25, help="Part des lignes à forcer sur la modalité majoritaire.")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la simulation.")
    p.add_argument("--mlflow", action="store_true", help="Log dans MLflow si disponible.")
    return p.parse_args()


def main():
    args = parse_args()
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(args.out) / f"drift_{ts}")
    params = load_params_yaml(Path("conf/params.yaml"))

    # 1) Schéma (optionnel)
    schema_cols: Optional[List[str]] = None
    if args.schema:
        sp = Path(args.schema)
        if sp.exists():
            try:
                schema_cols = json.loads(sp.read_text())
            except Exception:
                schema_cols = None

    # 2) Données
    ref_df_raw = read_any(Path(args.ref), nrows=args.sample)
    cur_df_raw = read_any(Path(args.cur), nrows=args.sample)

    # 3) Alignement & nettoyage auto (intersection + non vide + non constant)
    ref_df, cur_df, used_cols = align_clean(ref_df_raw, cur_df_raw, prefer_cols=schema_cols)

    if len(used_cols) == 0:
        raise RuntimeError("Aucune colonne non vide/variable en commun entre ref et cur après nettoyage.")

    print(f"[monitor] Colonnes utilisées après nettoyage: {len(used_cols)}")
    # 4) Simulation de drift (optionnelle)
    if args.simulate:
        rng = np.random.default_rng(args.seed)
        cur_df = simulate_drift(
            cur_df,
            money_col=args.money_col,
            factor=args.money_factor,
            cat_col=args.cat_col,
            cat_rate=args.cat_rate,
            rng=rng,
        )

    # 5) Rapport + alerte
    payload, summary_path, html_path = run_report(
        ref_df=ref_df,
        cur_df=cur_df,
        target=args.target,
        out_dir=out_dir,
        drift_share_threshold=args.drift_share_threshold,
    )

    # 6) MLflow (optionnel)
    if args.mlflow:
        tracking_uri, experiment = resolve_mlflow_config(params)
        try_log_mlflow(
            out_dir=out_dir,
            payload=payload,
            tracking_uri=tracking_uri,
            experiment=experiment,
            run_name="monitoring_drift",
        )

    # 7) Console
    print("\n=== Evidently — Data Drift ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nHTML report : {html_path}")
    print(f"Summary     : {summary_path}")
    print(f"Alert JSON  : {out_dir/'alert.json'}")


if __name__ == "__main__":
    main()
