# Implémentez un modèle de scoring (Projet 7 – OpenClassrooms)

Pipeline de scoring de défaut de crédit basé sur le dataset **Home Credit Default Risk** (Kaggle).  
Le projet comprend : acquisition des données, feature engineering, entraînement **CV** avec **LightGBM / Logistic Regression / RandomForest**, métriques métier **FN/FP** et suivi expérimental **MLflow (Databricks)**.

## 🔧 Prérequis

- Python 3.10
- `pip`
- Compte **Kaggle** (compétition “home-credit-default-risk” rejointe)
- Accès **Databricks** (pour MLflow remote) ou autre backend MLflow si vous adaptez

## 📦 Installation

    # Cloner
    git clone git@github.com:nicolasdestrac/openclassrooms-projet7-scoring.git
    cd openclassrooms-projet7-scoring  # (= OpenClassrooms/Projet_7)

    # (optionnel) activer pyenv
    pyenv local scoring_project7
    python -V

    # Dépendances
    pip install -r requirements.txt

## 🔐 Variables d’environnement

Créez un fichier `.env` à la racine :

    # --- Kaggle ---
    KAGGLE_USERNAME=ton_login_kaggle
    KAGGLE_KEY=ta_clef_api_kaggle

    # --- MLflow Remote (Databricks) ---
    MLFLOW_TRACKING_URI=databricks
    MLFLOW_EXPERIMENT=/Users/<ton_email>/projet7_scoring

    # Option A : via variables d'env
    # DATABRICKS_HOST=https://<ton-workspace>.cloud.databricks.com
    # DATABRICKS_TOKEN=<PAT>

    # Option B : via CLI (recommandé)
    # databricks auth login

Chargez-les dans la session :

    set -a; source .env; set +a

## ⬇️ Données

Télécharger depuis Kaggle (≈700 Mo) :

    ./scripts/download_data.sh
    # => dépose les CSV dans data/raw/

Ou manuellement :

    kaggle competitions download -c home-credit-default-risk -p data/raw
    unzip -o data/raw/home-credit-default-risk.zip -d data/raw

💡 Les dossiers volumineux (`data/`, `models/`, `mlruns/`, etc.) sont **ignorés** par git (voir `.gitignore`).

## ⚙️ Configuration (conf/params.yaml)

Exemple :

    data:
      train_csv: "data/raw/application_train.csv"
      test_csv:  "data/raw/application_test.csv"

    cv:
      n_splits: 5
      shuffle: true
      random_state: 42
      early_stopping_rounds: 200
      log_period: 50

    cost:               # pondération métier (FN >>> FP)
      fn: 10.0
      fp: 1.0
      threshold_grid: 501

    model:
      type: "lgbm"      # "lgbm" | "logreg" | "rf"
      lgbm:
        n_estimators: 5000
        learning_rate: 0.03
        num_leaves: 64
        min_child_samples: 100
        subsample: 0.8
        colsample_bytree: 0.8
        reg_lambda: 1.0
        reg_alpha: 0.1
        class_weight: "balanced"
        n_jobs: -1
      logreg:
        penalty: "l2"
        solver: "saga"
        max_iter: 2000
        class_weight: "balanced"
        n_jobs: -1
      rf:
        n_estimators: 500
        max_depth: null
        min_samples_split: 2
        min_samples_leaf: 1
        n_jobs: -1
        class_weight: "balanced"

    mlflow:
      tracking_uri_env: "MLFLOW_TRACKING_URI"
      default_tracking_uri: "databricks"
      experiment_env: "MLFLOW_EXPERIMENT"
      default_experiment: "/Users/<ton_email>/projet7_scoring"

    artifacts:
      models_dir: "models"
      reports_dir: "reports"

**Changer de modèle** = modifier `model.type` (`lgbm`, `logreg`, `rf`) et ajuster ses hyperparamètres.

## 🏃‍♂️ Entraînement

    set -a; source .env; set +a
    python -m src.train --config conf/params.yaml

Pendant le run :
- Cross-validation **StratifiedKFold** (5 folds)
- Logs détaillés LightGBM (AUC, early stopping)
- Affichage **AUC par fold** et temps
- Calcul du **seuil optimal** selon les coûts `FN/FP`
- **MLflow** loggue : AUC OOF, coût, seuil*, per-fold metrics, artefacts (modèle, threshold…)

En fin de run, la console affiche les liens MLflow (expérience & run).

## 📁 Structure

    .
    ├── conf/
    │   └── params.yaml
    ├── scripts/
    │   └── download_data.sh
    ├── src/
    │   ├── data.py            # chargement CSV
    │   ├── features.py        # preprocessing / feature eng
    │   ├── metrics.py         # AUC, coût, recherche de seuil
    │   └── train.py           # pipeline CV + MLflow
    ├── data/                  # (ignored) raw/ ...
    ├── models/                # (ignored) artefacts locaux
    ├── reports/               # (ignored) rapports locaux
    ├── requirements.txt
    └── README.md

## 📊 Métriques & seuil métier

- **AUC OOF** : performance globale CV.  
- **Coût** : `fn * (#FN) + fp * (#FP)` sur une grille de seuils ∈ [0,1].  
- **Seuil*** : seuil qui **minimise** ce coût.  
Tout est loggué dans MLflow (metrics + params + artefacts).

## 🧪 Suivi expérimental (MLflow)

Le script configure MLflow pour **Databricks** et gère proprement l’ouverture/fermeture des runs :
- `mlflow.set_tracking_uri("databricks")`
- `mlflow.set_experiment("/Users/<email>/projet7_scoring")`
- `mlflow.log_params / log_metrics / log_artifacts`

Comparez facilement les runs et téléchargez les artefacts depuis l’UI Databricks.

## 🧹 Bonnes pratiques Git

- Branches : `feat/<...>`, `fix/<...>`, `chore/<...>`
- Ouvrir une PR vers `main` :

      gh pr create --fill --base main --head feat/ma-feature
      gh pr merge <num> --squash --delete-branch

- Ne jamais committer : `.env`, `data/`, `models/`, `mlruns/`, `mlflow.db`, gros CSV.

## 🛠️ Débogage (FAQ)

- **Kaggle: `Could not find kaggle.json`**  
  → Renseigner `KAGGLE_USERNAME` et `KAGGLE_KEY` dans `.env` **ou** placer `~/.kaggle/kaggle.json` (chmod 600). Vérifier que vous avez **rejoint** la compétition.

- **MLflow Databricks auth**  
  → `databricks auth login` **ou** exporter `DATABRICKS_HOST` + `DATABRICKS_TOKEN`.

- **LightGBM: “Do not support special JSON characters in feature name.”**  
  → Les features sont renommées en noms **safe** dans `train.py` (DataFrame + `make_lgbm_safe`). Si vous ajoutez des features, évitez guillemets/virgules/retours ligne.

- **sklearn ConvergenceWarning (logreg)**  
  → Augmenter `max_iter`, garder `solver="saga"`, veiller au scaling (géré dans le preprocessor).

- **“Run is already active” (MLflow)**  
  → Le script ferme maintenant les runs “orphelins”. En dernier recours : `mlflow.end_run()`.

## 📈 Roadmap (idées)

- Hyperparam tuning (Optuna/MLflow)
- API d’inférence (FastAPI)
- Monitoring data drift (Evidently)
- Dashboard des métriques

---
