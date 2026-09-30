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
    """Render business assumptions in plain operational language.

    Internal policy keys remain unchanged for database/model compatibility, while
    the UI describes each quantity in terms an operations manager can interpret.
    """
    defaults = load_policy()
    with st.expander("Business assumptions for verification", expanded=False):
        st.caption(
            "These assumptions do not change the model's cancellation risk. They only decide whether verifying an order is expected to save money."
        )
        c1, c2 = st.columns(2)
        avoidable = c1.number_input(
            "Loss if a canceled order reaches fulfillment (Rs.)",
            min_value=0.0,
            value=float(defaults["avoidable_fulfillment_cost"]),
            step=100.0,
            key=f"{key_prefix}_avoidable",
            help="Approximate picking, packing, payment, service, or other cost that could be avoided if a cancellation is caught before fulfillment.",
        )
        prevention_pct = c2.slider(
            "Loss prevented by verification (%)",
            min_value=0,
            max_value=100,
            value=int(round(float(defaults["intervention_effectiveness"]) * 100)),
            step=5,
            key=f"{key_prefix}_effectiveness_pct",
            help="The share of the cancellation-related loss that a verification step is assumed to prevent. Example: 55% means a Rs. 2,000 loss is reduced by about Rs. 1,100 when verification works.",
        )
        c3, c4 = st.columns(2)
        intervention = c3.number_input(
            "Cost to verify one order (Rs.)",
            min_value=0.0,
            value=float(defaults["intervention_cost"]),
            step=25.0,
            key=f"{key_prefix}_intervention",
            help="Direct operational cost of the verification step, such as a message, payment check, or manual review.",
        )
        unnecessary = c4.number_input(
            "Extra cost if a good order is verified (Rs.)",
            min_value=0.0,
            value=float(defaults["false_positive_friction_cost"]),
            step=25.0,
            key=f"{key_prefix}_unnecessary",
            help="Estimated cost of unnecessary delay, customer contact, or service effort when an order would have completed normally. Set this to 0 if you do not want to model that cost.",
        )

        prevented_if_canceled = avoidable * (prevention_pct / 100.0)
        st.info(
            f"Plain-English assumption: if the order would cancel, verification can prevent about **Rs. {prevented_if_canceled:,.0f}** of loss. "
            f"Every verification costs **Rs. {intervention:,.0f}**, and an unnecessary verification adds **Rs. {unnecessary:,.0f}**."
        )

    return {
        "avoidable_fulfillment_cost": avoidable,
        "intervention_effectiveness": prevention_pct / 100.0,
        "intervention_cost": intervention,
        "false_positive_friction_cost": unnecessary,
    }
