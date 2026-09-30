from pathlib import Path

import pandas as pd
import streamlit as st

from src.business import load_policy
from src.config import ARTIFACT_DIR
from src.currency import fetch_pkr_to_usd_rate, format_usd, pkr_to_usd, rate_summary, usd_to_pkr

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_currency_context():
    """Fetch and cache the latest PKR->USD rate for one hour."""
    return fetch_pkr_to_usd_rate()


def currency_caption():
    fx = get_currency_context()
    if fx.get("is_live"):
        st.caption(rate_summary(fx) + ". Refreshed automatically up to once per hour. Model features remain in PKR internally; only user-facing money is shown in USD.")
    else:
        st.warning(rate_summary(fx) + ". Live FX lookup is unavailable, so the packaged fallback rate is being used.")
    return fx


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
    """Render business assumptions in USD while preserving PKR internally.

    The historical model and existing audit tables use PKR. User-entered USD
    assumptions are converted back to PKR before the economic rule is evaluated,
    so the decision math remains unit-consistent and backward compatible.
    """
    defaults = load_policy()
    fx = get_currency_context()
    rate = float(fx["rate"])

    with st.expander("Business assumptions for verification", expanded=False):
        st.caption(
            "These assumptions do not change the model's cancellation risk. They only decide whether verifying an order is expected to save money."
        )
        st.caption(rate_summary(fx))
        c1, c2 = st.columns(2)
        avoidable_usd = c1.number_input(
            "Loss if a canceled order reaches fulfillment ($)",
            min_value=0.0,
            value=round(float(pkr_to_usd(defaults["avoidable_fulfillment_cost"], rate)), 2),
            step=0.50,
            key=f"{key_prefix}_avoidable_usd",
            help="Approximate picking, packing, payment, service, or other cost that could be avoided if a cancellation is caught before fulfillment.",
        )
        prevention_pct = c2.slider(
            "Loss prevented by verification (%)",
            min_value=0,
            max_value=100,
            value=int(round(float(defaults["intervention_effectiveness"]) * 100)),
            step=5,
            key=f"{key_prefix}_effectiveness_pct",
            help="The share of the cancellation-related loss that a verification step is assumed to prevent.",
        )
        c3, c4 = st.columns(2)
        intervention_usd = c3.number_input(
            "Cost to verify one order ($)",
            min_value=0.0,
            value=round(float(pkr_to_usd(defaults["intervention_cost"], rate)), 2),
            step=0.25,
            key=f"{key_prefix}_intervention_usd",
            help="Direct operational cost of the verification step, such as a message, payment check, or manual review.",
        )
        unnecessary_usd = c4.number_input(
            "Extra cost if a good order is verified ($)",
            min_value=0.0,
            value=round(float(pkr_to_usd(defaults["false_positive_friction_cost"], rate)), 2),
            step=0.25,
            key=f"{key_prefix}_unnecessary_usd",
            help="Estimated cost of unnecessary delay, customer contact, or service effort when an order would have completed normally. Set this to 0 if you do not want to model that cost.",
        )

        prevented_if_canceled = avoidable_usd * (prevention_pct / 100.0)
        st.info(
            f"Plain-English assumption: if the order would cancel, verification can prevent about **${prevented_if_canceled:,.2f}** of loss. "
            f"Every verification costs **${intervention_usd:,.2f}**, and an unnecessary verification adds **${unnecessary_usd:,.2f}**."
        )

    return {
        "avoidable_fulfillment_cost": float(usd_to_pkr(avoidable_usd, rate)),
        "intervention_effectiveness": prevention_pct / 100.0,
        "intervention_cost": float(usd_to_pkr(intervention_usd, rate)),
        "false_positive_friction_cost": float(usd_to_pkr(unnecessary_usd, rate)),
    }

