import numpy as np
import pandas as pd

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Nettoyage DAYS_EMPLOYED
    if "DAYS_EMPLOYED" in df.columns:
        df.loc[df["DAYS_EMPLOYED"] > 365000, "DAYS_EMPLOYED"] = np.nan

    # Age & ancienneté
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(float)
    if "DAYS_EMPLOYED" in df.columns:
        df["EMP_YEARS"] = (-df["DAYS_EMPLOYED"] / 365)

    # Ratios financiers
    def safe_div(a, b):
        return a / (b + 1e-6)

    for num, den, name in [
        ("AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_INCOME_RATIO"),
        ("AMT_ANNUITY", "AMT_CREDIT",       "ANNUITY_CREDIT_RATIO"),
        ("AMT_GOODS_PRICE","AMT_CREDIT",    "GOODS_CREDIT_RATIO"),
    ]:
        if num in df.columns and den in df.columns:
            df[name] = safe_div(df[num], df[den])

    # Interactions EXT_SOURCE
    pairs = [("EXT_SOURCE_1","EXT_SOURCE_2"),
             ("EXT_SOURCE_2","EXT_SOURCE_3"),
             ("EXT_SOURCE_1","EXT_SOURCE_3")]
    for a,b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"]    = df[a] * df[b]
            df[f"{a}_plus_{b}"] = df[a] + df[b]

    return df

def make_train_test(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    # applique les mêmes features aux deux jeux
    train = basic_feature_engineering(train_raw)
    test  = basic_feature_engineering(test_raw)
    y = train["TARGET"].astype(int)
    X = train.drop(columns=["TARGET"])
    return X, y, test
