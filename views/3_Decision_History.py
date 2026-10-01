import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe
from src.ui import common as ui_common
from src.currency import fetch_pkr_to_usd_rate, rate_summary

@st.cache_data(ttl=3600, show_spinner=False)
def _fallback_currency_context():
    return fetch_pkr_to_usd_rate()


def currency_caption():
    helper = getattr(ui_common, "currency_caption", None)
    if callable(helper):
        return helper()
    fx = _fallback_currency_context()
    if fx.get("is_live"):
        st.caption(rate_summary(fx) + ". Refreshed automatically up to once per hour.")
    else:
        st.warning(rate_summary(fx) + ". Live FX lookup is unavailable, so the packaged fallback rate is being used.")
    return fx


require_role("manager")
st.markdown("""<div class="cr-hero"><div class="cr-eyebrow">OPERATIONS · AUDIT TRAIL</div><h1>🧾&nbsp; Decision History</h1><p>Trace every manager decision alongside the risk score, model recommendation, expected value, and model version shown at decision time.</p></div>""", unsafe_allow_html=True)
fx = currency_caption()
fx_rate = float(fx["rate"])

hist = dataframe(
    "SELECT decided_at, order_id, probability, threshold, recommendation, manager_decision, "
    "actual_outcome, net_expected_savings, model_name, model_version "
    "FROM manager_decisions ORDER BY decided_at DESC LIMIT 500"
)
if len(hist):
    hist = hist.rename(columns={
        "decided_at": "Decision time",
        "order_id": "Order ID",
        "probability": "Cancellation risk",
        "threshold": "Risk threshold",
        "recommendation": "System recommendation",
        "manager_decision": "Manager decision",
        "actual_outcome": "Known outcome",
        "net_expected_savings": "Estimated net value ($)",
        "model_name": "Model",
        "model_version": "Model version",
    })
    for col in ["Cancellation risk", "Risk threshold"]:
        if col in hist.columns:
            hist[col] = hist[col].map(lambda x: f"{x:.1%}" if x is not None else "")
    if "Estimated net value ($)" in hist.columns:
        hist["Estimated net value ($)"] = hist["Estimated net value ($)"].map(lambda x: f"{'-' if float(x) * fx_rate < 0 else ''}${abs(float(x) * fx_rate):,.2f}" if x is not None else "")
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.info("No decisions have been recorded yet.")
