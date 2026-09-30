import streamlit as st

from src.auth import require_role
from src.business import load_policy
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata
from src.ui.common import currency_caption
from src.currency import format_usd

require_role("manager")
st.title("📊 Operations Dashboard")
st.caption("Production scoring is ready at login. Data preparation and model training are isolated from day-to-day operations.")
fx = currency_caption()
fx_rate = float(fx["rate"])

meta = active_metadata()
if not meta:
    st.error("The packaged production model is missing. Contact the developer team.")
    st.stop()

m = meta.get("test_metrics", {})
cols = st.columns(5)
cols[0].metric("Production Model", f"LightGBM {meta.get('model_version', '')}")
cols[1].metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
cols[2].metric("PR-AUC", f"{m.get('pr_auc', float('nan')):.3f}")
cols[3].metric("Brier Score", f"{m.get('brier', float('nan')):.3f}")
cols[4].metric("Risk Threshold", f"{float(meta.get('threshold', .5)):.1%}")
st.caption(f"Production dataset: **{meta.get('dataset_version', 'unknown')}**. The model is loaded from a packaged artifact and does not retrain when the app starts.")

policy = load_policy()
st.subheader("Business Rules Used for Recommendations")
a, b, c, d = st.columns(4)
a.metric("Loss from a late cancellation", format_usd(policy["avoidable_fulfillment_cost"], fx_rate))
b.metric("Loss prevented by verification", f"{policy['intervention_effectiveness']:.0%}")
c.metric("Cost to verify an order", format_usd(policy["intervention_cost"], fx_rate))
d.metric("Cost of an unnecessary verification", format_usd(policy["false_positive_friction_cost"], fx_rate))
prevented = policy['avoidable_fulfillment_cost'] * policy['intervention_effectiveness']
st.info(
    f"Simple rule: verify an order only when the expected money saved is greater than the expected cost of checking it. "
    f"Under the current assumptions, a successful verification can prevent about {format_usd(prevented, fx_rate)} of a cancellation-related loss."
)

st.subheader("Recent Decisions")
hist = dataframe(
    "SELECT decided_at, order_id, probability, recommendation, manager_decision, net_expected_savings "
    "FROM manager_decisions ORDER BY decided_at DESC LIMIT 10"
)
if len(hist):
    hist = hist.rename(columns={
        "decided_at": "Decision time",
        "order_id": "Order ID",
        "probability": "Cancellation risk",
        "recommendation": "System recommendation",
        "manager_decision": "Manager decision",
        "net_expected_savings": "Estimated net value ($)",
    })
    if "Cancellation risk" in hist.columns:
        hist["Cancellation risk"] = hist["Cancellation risk"].map(lambda x: f"{x:.1%}" if x is not None else "")
    if "Estimated net value ($)" in hist.columns:
        hist["Estimated net value ($)"] = hist["Estimated net value ($)"].map(lambda x: f"{'-' if float(x) * fx_rate < 0 else ''}${abs(float(x) * fx_rate):,.2f}" if x is not None else "")
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.caption("No manager decisions have been recorded yet. Open Score Order to begin.")
