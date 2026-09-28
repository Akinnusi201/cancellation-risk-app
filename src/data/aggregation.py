import pandas as pd
import numpy as np
from src.config import ORDER_ID_CANDIDATES, CUSTOMER_ID_CANDIDATES
from src.data.schema import normalize_identifier_columns


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def add_row_features(df):
    out = df.copy()
    out["day_of_week"] = out["created_at"].dt.dayofweek
    out["hour"] = out["created_at"].dt.hour.fillna(0).astype(int)
    out["discount_ratio"] = out["discount_amount"] / (out["grand_total"].abs() + 1e-5)
    out["is_cod"] = out["payment_method"].astype(str).str.lower().eq("cod").astype(int)
    out["has_discount"] = (out["discount_amount"] > 0).astype(int)
    out["is_canceled"] = out["status"].eq("canceled").astype(int)
    return out


def aggregate_to_orders(df):
    # IDs must have one stable type before pandas sort/groupby.
    df = normalize_identifier_columns(df)
    df = add_row_features(df)
    order_col = first_existing(df.columns, ORDER_ID_CANDIDATES)
    if order_col is None:
        df = df.reset_index(drop=False).rename(columns={"index": "synthetic_order_id"})
        order_col = "synthetic_order_id"
    cust_col = first_existing(df.columns, CUSTOMER_ID_CANDIDATES)
    if cust_col is None:
        df["synthetic_customer_id"] = "unknown"
        cust_col = "synthetic_customer_id"

    df = df.sort_values(["created_at", order_col]).copy()

    # Status consistency: if an order has multiple statuses, quarantine at a higher layer later if desired.
    agg = df.groupby(order_col, dropna=False).agg(
        created_at=("created_at", "min"),
        customer_id=(cust_col, "first"),
        price=("price", "mean"),
        qty_ordered=("qty_ordered", "sum"),
        grand_total=("grand_total", "max"),
        discount_amount=("discount_amount", "sum"),
        category_name_1=("category_name_1", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        payment_method=("payment_method", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        is_canceled=("is_canceled", "max"),
        status=("status", lambda s: "canceled" if (s == "canceled").any() else ("complete" if (s == "complete").any() else "received")),
    ).reset_index().rename(columns={order_col: "order_id"})

    # Stable identifier types are critical for cumulative Parquet snapshots.
    # IDs are identifiers, not quantities, so always store them as strings.
    agg["order_id"] = agg["order_id"].astype("string")
    agg["customer_id"] = agg["customer_id"].astype("string")

    agg["discount_ratio"] = agg["discount_amount"] / (agg["grand_total"].abs() + 1e-5)
    agg["is_cod"] = agg["payment_method"].astype(str).str.lower().eq("cod").astype(int)
    agg["has_discount"] = (agg["discount_amount"] > 0).astype(int)
    agg["day_of_week"] = agg["created_at"].dt.dayofweek
    agg["hour"] = agg["created_at"].dt.hour.fillna(0).astype(int)

    # Leakage-safe historical customer cancellation rate: only prior orders contribute.
    agg = agg.sort_values(["created_at", "order_id"]).reset_index(drop=True)
    grp = agg.groupby("customer_id", dropna=False)
    agg["prev_orders"] = grp.cumcount()
    agg["prev_cancellations"] = grp["is_canceled"].cumsum() - agg["is_canceled"]
    agg["customer_cancel_rate"] = np.where(agg["prev_orders"] > 0, agg["prev_cancellations"] / agg["prev_orders"], 0.0)
    return agg
