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



def recompute_customer_history(orders: pd.DataFrame) -> pd.DataFrame:
    """Recompute leakage-safe prior cancellation history after datasets are combined."""
    out = orders.sort_values(["created_at", "order_id"]).reset_index(drop=True).copy()
    grp = out.groupby("customer_id", dropna=False)
    out["prev_orders"] = grp.cumcount()
    out["prev_cancellations"] = grp["is_canceled"].cumsum() - out["is_canceled"]
    out["customer_cancel_rate"] = np.where(
        out["prev_orders"] > 0, out["prev_cancellations"] / out["prev_orders"], 0.0
    )
    return out

def _group_mode(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Deterministic group mode without slow Python lambdas per order."""
    counts = (
        df[[group_col, value_col]]
        .assign(**{value_col: df[value_col].fillna("unknown").astype(str)})
        .groupby([group_col, value_col], dropna=False, sort=False)
        .size()
        .rename("_n")
        .reset_index()
    )
    # Highest frequency wins. Lexicographic value breaks ties deterministically.
    counts = counts.sort_values([group_col, "_n", value_col], ascending=[True, False, True])
    return counts.drop_duplicates(group_col)[[group_col, value_col]]


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

    # Native groupby aggregations keep large batches fast.
    agg = df.groupby(order_col, dropna=False, sort=False).agg(
        created_at=("created_at", "min"),
        customer_id=(cust_col, "first"),
        price=("price", "mean"),
        qty_ordered=("qty_ordered", "sum"),
        grand_total=("grand_total", "max"),
        discount_amount=("discount_amount", "sum"),
        is_canceled=("is_canceled", "max"),
    ).reset_index().rename(columns={order_col: "order_id"})

    category_mode = _group_mode(df, order_col, "category_name_1").rename(columns={order_col: "order_id"})
    payment_mode = _group_mode(df, order_col, "payment_method").rename(columns={order_col: "order_id"})
    agg = agg.merge(category_mode, on="order_id", how="left").merge(payment_mode, on="order_id", how="left")
    agg["status"] = agg["is_canceled"].map({1: "canceled", 0: "complete"})

    # Stable identifier types are critical for cumulative snapshots.
    agg["order_id"] = agg["order_id"].astype("string")
    agg["customer_id"] = agg["customer_id"].astype("string")

    agg["discount_ratio"] = agg["discount_amount"] / (agg["grand_total"].abs() + 1e-5)
    agg["is_cod"] = agg["payment_method"].astype(str).str.lower().eq("cod").astype(int)
    agg["has_discount"] = (agg["discount_amount"] > 0).astype(int)
    agg["day_of_week"] = agg["created_at"].dt.dayofweek
    agg["hour"] = agg["created_at"].dt.hour.fillna(0).astype(int)

    # Leakage-safe history is calculated on the ordered batch, then recalculated again after cumulative merges.
    return recompute_customer_history(agg)
