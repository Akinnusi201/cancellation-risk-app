"""Shared schema normalization for DataOps persistence and aggregation.

Identifier fields are intentionally stored as strings. Order/customer IDs can look numeric,
but they are keys, not quantities, and may arrive with mixed CSV inference across batches.
"""
from __future__ import annotations

import pandas as pd

from src.config import ORDER_ID_CANDIDATES, CUSTOMER_ID_CANDIDATES


def normalize_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidates = set(ORDER_ID_CANDIDATES) | set(CUSTOMER_ID_CANDIDATES) | {"order_id", "customer_id"}
    for col in candidates:
        if col in out.columns:
            # Use pandas StringDtype so missing values remain <NA> instead of the literal "nan".
            out[col] = out[col].astype("string").str.strip()
            out.loc[out[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA
    return out


def normalize_snapshot_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, Parquet-safe dataframe across cumulative dataset versions."""
    out = normalize_identifier_columns(df)

    for col in ["category_name_1", "payment_method", "status"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")

    numeric_cols = [
        "price", "qty_ordered", "grand_total", "discount_amount",
        "discount_ratio", "is_cod", "has_discount", "day_of_week", "hour",
        "prev_orders", "prev_cancellations", "customer_cancel_rate", "is_canceled",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
