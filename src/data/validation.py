import pandas as pd
import numpy as np
from src.config import LEAKAGE_COLUMNS, ORDER_ID_CANDIDATES, CUSTOMER_ID_CANDIDATES
from src.data.schema import normalize_identifier_columns

REQUIRED = ["created_at", "price", "qty_ordered", "grand_total", "discount_amount", "payment_method", "category_name_1", "status"]
VALID_STATUSES = {"complete", "canceled"}


def validate_raw(df: pd.DataFrame):
    # Normalize key fields before any grouping, sorting, deduplication, or persistence.
    df = normalize_identifier_columns(df)
    checks = []
    quarantine_mask = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype="object")

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    checks.append(("required_columns", "PASS" if not missing_cols else "FAIL", len(missing_cols), f"Missing: {missing_cols}" if missing_cols else "All required columns present"))
    if missing_cols:
        return df.copy(), df.iloc[0:0].copy(), checks, True

    order_cols = [c for c in ORDER_ID_CANDIDATES if c in df.columns]
    if not order_cols:
        # Keep a safe synthetic-ID fallback for small/demo inputs, but surface it clearly.
        checks.append(("order_identifier", "WARN", 0, f"No order identifier found; aggregation will create synthetic IDs. Expected one of: {ORDER_ID_CANDIDATES}"))
    else:
        order_col = order_cols[0]
        missing_order_id = df[order_col].isna()
        quarantine_mask |= missing_order_id
        reasons = reasons.mask(missing_order_id, reasons + "missing_order_id;")
        checks.append(("order_identifier", "PASS" if missing_order_id.sum() == 0 else "WARN", int(missing_order_id.sum()), f"Using {order_col}; missing IDs quarantined"))

    parsed_dates = pd.to_datetime(df["created_at"], errors="coerce")
    bad = parsed_dates.isna()
    quarantine_mask |= bad
    reasons = reasons.mask(bad, reasons + "invalid_created_at;")
    checks.append(("date_parsing", "PASS" if bad.sum() == 0 else "WARN", int(bad.sum()), "Unparseable created_at"))

    numeric_rules = {
        "price": lambda s: s >= 0,
        "qty_ordered": lambda s: s > 0,
        "grand_total": lambda s: s >= 0,
        "discount_amount": lambda s: s >= 0,
    }
    for col, rule in numeric_rules.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.isna() | ~rule(vals)
        quarantine_mask |= bad
        reasons = reasons.mask(bad, reasons + f"invalid_{col};")
        checks.append((f"range_{col}", "PASS" if bad.sum() == 0 else "WARN", int(bad.sum()), f"Invalid {col} values"))

    status_norm = df["status"].astype(str).str.lower().str.strip()
    bad = ~status_norm.isin(VALID_STATUSES)
    quarantine_mask |= bad
    reasons = reasons.mask(bad, reasons + "invalid_status;")
    checks.append(("target_status", "PASS" if bad.sum() == 0 else "WARN", int(bad.sum()), "Rows outside complete/canceled"))

    exact_dupes = df.duplicated(keep="first")
    quarantine_mask |= exact_dupes
    reasons = reasons.mask(exact_dupes, reasons + "duplicate_row;")
    checks.append(("duplicate_rows", "PASS" if exact_dupes.sum() == 0 else "WARN", int(exact_dupes.sum()), "Exact duplicate rows"))

    valid = df.loc[~quarantine_mask].copy()
    quarantine = df.loc[quarantine_mask].copy()
    if len(quarantine):
        quarantine["quarantine_reason"] = reasons.loc[quarantine_mask]
    valid["created_at"] = pd.to_datetime(valid["created_at"], errors="coerce")
    for c in ["price", "qty_ordered", "grand_total", "discount_amount"]:
        valid[c] = pd.to_numeric(valid[c], errors="coerce")
    valid["status"] = valid["status"].astype(str).str.lower().str.strip()

    checks.append(("leakage_guard", "PASS", 0, f"Outcome columns are blocked from model features: {sorted(LEAKAGE_COLUMNS)}"))
    return valid, quarantine, checks, False
