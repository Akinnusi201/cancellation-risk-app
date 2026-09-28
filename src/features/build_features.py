import pandas as pd
from src.config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, LEAKAGE_COLUMNS


def get_model_frame(df: pd.DataFrame):
    leak = [c for c in FEATURES if c in LEAKAGE_COLUMNS]
    if leak:
        raise ValueError(f"Leakage columns cannot be features: {leak}")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    X = df[FEATURES].copy()
    y = df["is_canceled"].astype(int).copy()
    return X, y
