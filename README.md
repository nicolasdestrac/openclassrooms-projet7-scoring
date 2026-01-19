# Implémentez un modèle de scoring (Projet 7 – OpenClassrooms)

[![Tests](https://img.shields.io/github/actions/workflow/status/nicolasdestrac/openclassrooms-projet7-scoring/ci-cd.yml?label=tests&branch=main)](https://github.com/nicolasdestrac/openclassrooms-projet7-scoring/actions)
[![API Deploy](https://img.shields.io/badge/deploy-render-blue)](#)
[![MLflow](https://img.shields.io/badge/tracking-mlflow-orange)](#)
[![License](https://img.shields.io/badge/license-educational-lightgrey)](#)

Pipeline de scoring de défaut de crédit basé sur **Home Credit Default Risk**.
Le projet couvre : ingestion & features, entraînement **CV**, **métrique métier** (FN ≫ FP) et sélection de seuil, **tracking MLflow (Databricks)**, **API FastAPI** (Render), **UI Streamlit**, **CI/CD GitHub Actions**, et **monitoring Evidently**.

---

## 🔍 Vue d’ensemble

- **Modèles** : Logistic Regression, RandomForest, **LightGBM (final)**
- **Préprocess** : imputation médiane/most_frequent, OHE, normalisation, `log1p` sur colonnes monétaires, features dérivées (ratios, interactions `EXT_SOURCE_*`)
- **Validation** : Stratified K-Fold (5 folds), OOF AUC + métriques par fold
- **Score métier** : coût = `10×FN + 1×FP` → seuil optimal par grille
- **Tracking** : MLflow (Databricks) — params, métriques, artefacts, modèle
- **Serving** : API FastAPI (Render) + UI Streamlit
- **CI/CD** : tests unitaires (pytest) → déploiement Render via **Deploy Hook**
- **Monitoring** : rapport HTML Evidently + alerte JSON, exécutable en CI

---

## 📁 Arborescence

```
.
├── conf/                 # config YAML (données, CV, coûts, modèles, MLflow…)
├── models/               # artefacts (pipeline.joblib, seuil, schéma) [gitignored]
├── artifacts/            # rapports data drift [gitignored]
├── reports/              # rapports locaux [gitignored]
├── scripts/              # utilitaires (ex: download Kaggle)
├── src/
│   ├── data.py           # chargement CSV
│   ├── features.py       # feature engineering & préprocess
│   ├── metrics.py        # métriques & score métier (coût, seuil optimal)
│   ├── train.py          # CV, OOF, logging MLflow, sérialisation artefacts
│   ├── tune.py           # tuning (grid/random) piloté par conf
│   ├── monitor.py        # monitoring Evidently + (optionnel) logging MLflow
├── streamlit_app/
│   └── app.py            # application Streamlit (front démo)
├── api/
│   └── app.py            # API FastAPI (predict, predict_proba, explain)
├── tests/                # pytest : métriques, API
├── .github/workflows/
│   └── ci-cd.yml         # jobs: test | deploy
│   └── monitor.yml       # job: monitor (drift)
├── requirements.txt
└── README.md
```

---

## 🔧 Prérequis

- Python **3.10**
- Compte **Kaggle** (compétition *home-credit-default-risk*)
- Accès **Databricks** (ou autre backend MLflow)
- (Démo cloud) Compte **Render** pour l’API et la UI

---

## ⚙️ Installation

```bash
git clone https://github.com/nicolasdestrac/openclassrooms-projet7-scoring.git
cd openclassrooms-projet7-scoring

python -m pip install -U pip
pip install -r requirements.txt
```

### Variables d’environnement (`.env`)

```bash
# Kaggle (si téléchargement auto)
KAGGLE_USERNAME=...
KAGGLE_KEY=...

# MLflow (Databricks)
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT=/Users/nicolas.destrac@gmail.com/projet7
# DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
# DATABRICKS_TOKEN=<PAT>

# API / CORS (front Streamlit autorisé)
FRONTEND_ORIGINS=https://openclassrooms-projet7-scoring-streamlit.onrender.com
```

Active-les dans la session :

```bash
set -a; source .env; set +a
```

---

## ⬇️ Données

```bash
# via script
./scripts/download_data.sh

# ou manuellement
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip -o data/raw/home-credit-default-risk.zip -d data/raw
```

> Dossiers volumineux (`data/`, `models/`, `reports/`, `mlruns/`) ignorés par git.

---

## 🧪 Configuration (extrait `conf/params.yaml`)

```yaml
data:
  train_csv: data/raw/application_train.csv
  test_csv:  data/raw/application_test.csv

cv:
  n_splits: 5
  shuffle: true
  random_state: 42
  early_stopping_rounds: 200

cost:
  fn: 10.0
  fp: 1.0
  threshold_grid: 501

model:
  type: lgbm                  # lgbm | logreg | rf
  lgbm:
    n_estimators: 5000
    learning_rate: 0.03
    num_leaves: 64
    max_depth: 5
    reg_alpha: 0.5
    reg_lambda: 0.1
    subsample: 0.6
    class_weight: balanced
  logreg:
    solver: saga
    max_iter: 2000
    class_weight: balanced
  rf:
    n_estimators: 600
    class_weight: balanced_subsample

mlflow:
  tracking_uri_env: MLFLOW_TRACKING_URI
  default_tracking_uri: databricks
  experiment_env: MLFLOW_EXPERIMENT
  default_experiment: /Users/nicolas.destrac@gmail.com/projet7

artifacts:
  models_dir: models
  reports_dir: reports
```

---

## 🏃 Entraînement & artefacts

```bash
set -a; source .env; set +a
python -m src.train --config conf/params.yaml
```

Génère dans `models/` :

- `scoring_model.joblib` : **Pipeline**(preprocessor, estimator)
- `decision_threshold.json` : `{"threshold": <float>}`
- `input_columns.json` : colonnes d’entrée attendues (pour l’API)

Et loggue dans **MLflow** :
params, AUC OOF, coûts/seuils, métriques par fold, artefacts (importances, matrices), modèle + signature.

---

## 🔌 API FastAPI

**Local :**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

**Endpoints :**

- `GET /schema` → liste des colonnes d’entrée
- `POST /predict_proba` → `{"probability": float}`
- `POST /predict` → probabilité + décision binaire via seuil métier
- `POST /explain` → top-20 contributions SHAP locales (si compatible)

**Exemple de payload** :
```json
{
  "features": {
    "AMT_CREDIT": 100000.0,
    "AMT_ANNUITY": 12000.0
  }
}
```

---

## 🖥️ UI Streamlit

```bash
streamlit run streamlit_app/app.py
```

La UI appelle l’API (URL configurable) et affiche probabilité, décision et SHAP local.

---

## ✅ CI/CD GitHub Actions

Fichier : `.github/workflows/ci-cd.yml`

- **test** (PR & `main`)
  - Installe les deps
  - Lance `pytest -q`
- **deploy** (branche `main`, **optionnel**)
  - `needs: test`
  - déclenche **Render Deploy Hook** si `RENDER_DEPLOY_HOOK` est défini
- **monitor** (manuel ou planifié)
  - exécute `python -m src.monitor`
  - publie le rapport Evidently en artefact
  - **échoue** le job si `alert.json` indique du drift > seuil

> Sur Render, l’auto-deploy peut être **OFF** : le **deploy hook** devient le seul déclencheur (contrôlé par les tests).

---

## 📉 Monitoring (Evidently)

**Local** :
```bash
python -m src.monitor \
  --ref data/raw/application_train.csv \
  --cur data/raw/application_test.csv \
  --out artifacts/reports \
  --sample 50000 \
  --mlflow
```

**Options** :
- `--schema models/input_columns.json` : n’analyser que les features servies
- `--simulate --money-col AMT_CREDIT --money-factor 1.10 --cat-col NAME_INCOME_TYPE --cat-rate 0.25`
- `--drift-share-threshold 0.30` : alerte si > 30% de colonnes driftées

**Sorties** :
- `evidently_data_drift_report.html`
- `evidently_data_drift_summary.json`
- `alert.json` (clé `alert: true|false` + raison)

---

## 🧪 Tests

```bash
pytest -q
```

- **tests/test_metrics.py** : AUC, coût métier, seuil optimal
- **tests/test_api.py** : `/schema`, `/predict_proba`, `/predict` (happy paths & erreurs)

---

## 📊 Interprétabilité

- **Globale** : importances (gain/impurity pour LGBM) — logguées MLflow
- **Locale** : endpoint `/explain` (SHAP, top-20)

---

## 🔐 Bonnes pratiques & reproductibilité

- Seeds fixés (CV + LGBM)
- Sérialisation **Pipeline sklearn** + signature d’entrée (MLflow)
- Schéma contrôlé via `input_columns.json`
- Versions figées dans `requirements.txt`

---

## 🚀 Liens

- **Dépôt GitHub** : https://github.com/nicolasdestrac/openclassrooms-projet7-scoring
- **MLflow (Databricks)** : `/Users/nicolas.destrac@gmail.com/projet7`
- **API Render** : https://openclassrooms-projet7-scoring-api.onrender.com/
- **UI Streamlit** : https://openclassrooms-projet7-scoring-streamlit.onrender.com

---

## 🛠️ Dépannage (tips rapides)

- **Databricks** : vérifier `DATABRICKS_HOST` / `DATABRICKS_TOKEN` pour le tracking MLflow
- **Render** : si auto-deploy OFF, utiliser le **Deploy Hook** (secret GitHub `RENDER_DEPLOY_HOOK`)

---

## 📜 Licence

Projet pédagogique — usage non commercial.
