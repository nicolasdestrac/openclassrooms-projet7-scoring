from pathlib import Path
import pandas as pd

from src.data import load_raw
from src.features import basic_feature_engineering

# -----------------------------
# Chemins à adapter si besoin
# -----------------------------
DATA_DIR = Path("data/raw/")

# Mets ici les bons noms de fichiers CSV de ton projet 7
TRAIN_CSV = DATA_DIR / "application_train.csv"
TEST_CSV  = DATA_DIR / "application_test.csv"   # utilisé juste pour load_raw, mais pas pour le dashboard

OUTPUT_PARQUET = DATA_DIR / "dashboard_sample.parquet"
ID_COL = "SK_ID_CURR"

# -----------------------------
# Chargement des données brutes
# -----------------------------
print(f"Chargement des données brutes depuis : {TRAIN_CSV}")
train_raw, _ = load_raw(str(TRAIN_CSV), str(TEST_CSV))

assert ID_COL in train_raw.columns, f"Colonne ID client manquante : {ID_COL}"

# -----------------------------
# Feature engineering (comme dans le train)
# -----------------------------
print("Application du basic_feature_engineering...")
train_feat = basic_feature_engineering(train_raw)

# On enlève TARGET pour ne garder que les features d'entrée
if "TARGET" in train_feat.columns:
    train_feat = train_feat.drop(columns=["TARGET"])

# -----------------------------
# Construction du dataset dashboard
# -----------------------------
# On met SK_ID_CURR en première colonne
cols = [ID_COL] + [c for c in train_feat.columns if c != ID_COL]
df_dashboard = train_feat[cols].copy()

print(f"Dataset complet pour le dashboard : {df_dashboard.shape[0]} lignes, {df_dashboard.shape[1]} colonnes.")

# Optionnel : échantillonnage pour alléger
MAX_ROWS = 8000
if len(df_dashboard) > MAX_ROWS:
    df_dashboard = df_dashboard.sample(MAX_ROWS, random_state=42)
    print(f"Échantillonnage à {len(df_dashboard)} lignes pour le dashboard.")

# -----------------------------
# Sauvegarde en Parquet
# -----------------------------
df_dashboard.to_parquet(OUTPUT_PARQUET)
print(f"✅ Fichier Parquet généré pour le dashboard : {OUTPUT_PARQUET}")
