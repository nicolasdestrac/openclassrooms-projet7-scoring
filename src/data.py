"""
Data helpers
============

- Chargement des train et test.
"""

from pathlib import Path
import pandas as pd

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def load_raw(train_csv: str, test_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les CSV bruts et renvoie (train_df, test_df)."""
    train_p, test_p = Path(train_csv), Path(test_csv)
    assert train_p.exists(), f"Fichier manquant: {train_p}"
    assert test_p.exists(),  f"Fichier manquant: {test_p}"
    train = pd.read_csv(train_p)
    test  = pd.read_csv(test_p)
    return train, test
