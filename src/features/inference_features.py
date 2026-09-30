from datetime import datetime
import uuid

import pandas as pd


BASE_INPUT_COLUMNS = [
    "price", "qty_ordered", "grand_total", "discount_amount",
    "payment_method", "category_name_1",
]


def prepare_order_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in BASE_INPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required order columns: {missing}")

    if "order_id" not in out.columns:
        out["order_id"] = [f"manual_{uuid.uuid4().hex[:10]}" for _ in range(len(out))]
    out["order_id"] = out["order_id"].astype("string")

    if "created_at" not in out.columns:
        out["created_at"] = datetime.now()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce").fillna(pd.Timestamp.now())

    for c in ["price", "qty_ordered", "grand_total", "discount_amount"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[["price", "qty_ordered", "grand_total", "discount_amount"]].isna().any().any():
        raise ValueError("Price, quantity, grand total, and discount must be numeric.")

    if "customer_cancel_rate" not in out.columns:
        out["customer_cancel_rate"] = 0.0
    out["customer_cancel_rate"] = pd.to_numeric(out["customer_cancel_rate"], errors="coerce").fillna(0.0).clip(0, 1)

    out["discount_ratio"] = out["discount_amount"] / (out["grand_total"].abs() + 1e-5)
    out["is_cod"] = out["payment_method"].astype(str).str.lower().eq("cod").astype(int)
    out["has_discount"] = (out["discount_amount"] > 0).astype(int)
    out["day_of_week"] = out["created_at"].dt.dayofweek
    out["hour"] = out["created_at"].dt.hour.astype(int)
    out["category_name_1"] = out["category_name_1"].fillna("unknown").astype(str)
    out["payment_method"] = out["payment_method"].fillna("unknown").astype(str)
    return out
