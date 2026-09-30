from pathlib import Path

import pandas as pd
import streamlit as st

from src.business import load_policy
from src.config import ARTIFACT_DIR

REFERENCE_PATH = ARTIFACT_DIR / "reference_orders.csv.gz"
DEMO_PATH = ARTIFACT_DIR / "demo_orders.csv.gz"
HISTORICAL_DEMO_PATH = ARTIFACT_DIR / "historical_demo_orders.csv.gz"

FEATURE_LABELS = {
    "customer_cancel_rate": "Prior customer cancellation rate",
    "is_cod": "Cash-on-delivery payment",
    "grand_total": "Order value",
    "discount_ratio": "Discount ratio",
    "price": "Average item price",
    "qty_ordered": "Quantity",
    "payment_method": "Payment method",
    "category_name_1": "Product category",
    "discount_amount": "Discount amount",
    "hour": "Order hour",
    "day_of_week": "Day of week",
    "has_discount": "Discount presence",
}


@st.cache_data(show_spinner=False)
def _read_cached_csv(path: str, mtime_ns: int):
    # mtime_ns is intentionally part of the cache key so model promotion refreshes
    # the reference and simulation data without requiring a server restart.
    return pd.read_csv(path, low_memory=False, parse_dates=["created_at"])


def load_reference():
    if not REFERENCE_PATH.exists():
        return pd.DataFrame()
    return _read_cached_csv(str(REFERENCE_PATH), REFERENCE_PATH.stat().st_mtime_ns)


def load_demo_orders():
    """Load the risk-stratified live Operations simulation queue."""
    if not DEMO_PATH.exists():
        return pd.DataFrame()
    return _read_cached_csv(str(DEMO_PATH), DEMO_PATH.stat().st_mtime_ns)


def load_historical_demo_orders():
    """Load a natural sample of the final temporal holdout for retrospective evaluation."""
    path = HISTORICAL_DEMO_PATH if HISTORICAL_DEMO_PATH.exists() else DEMO_PATH
    if not path.exists():
        return pd.DataFrame()
    return _read_cached_csv(str(path), path.stat().st_mtime_ns)


def business_policy_controls(key_prefix="policy"):
    defaults = load_policy()
    with st.expander("Economic decision assumptions", expanded=False):
        st.caption("These parameters affect the intervention recommendation, not the model's cancellation probability.")
        c1, c2 = st.columns(2)
        avoidable = c1.number_input(
            "Avoidable fulfillment cost (Rs.)", min_value=0.0, value=float(defaults["avoidable_fulfillment_cost"]),
            step=100.0, key=f"{key_prefix}_avoidable",
        )
        effectiveness = c2.slider(
            "Intervention effectiveness", 0.0, 1.0, float(defaults["intervention_effectiveness"]), 0.05,
            key=f"{key_prefix}_effectiveness",
        )
        c3, c4 = st.columns(2)
        intervention = c3.number_input(
            "Intervention cost (Rs.)", min_value=0.0, value=float(defaults["intervention_cost"]),
            step=25.0, key=f"{key_prefix}_intervention",
        )
        friction = c4.number_input(
            "False-intervention friction cost (Rs.)", min_value=0.0, value=float(defaults["false_positive_friction_cost"]),
            step=25.0, key=f"{key_prefix}_friction",
        )
    return {
        "avoidable_fulfillment_cost": avoidable,
        "intervention_effectiveness": effectiveness,
        "intervention_cost": intervention,
        "false_positive_friction_cost": friction,
    }
