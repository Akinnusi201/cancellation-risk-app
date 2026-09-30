import streamlit as st

from src.auth import require_role
from src.business import load_policy
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata

require_role("manager")
st.title("📊 Operations Dashboard")
st.caption("Production scoring is ready at login. Data preparation and model training are isolated from day-to-day operations.")

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
st.subheader("Current Economic Policy")
a, b, c, d = st.columns(4)
a.metric("Avoidable fulfillment cost", f"Rs. {policy['avoidable_fulfillment_cost']:,.0f}")
b.metric("Intervention effectiveness", f"{policy['intervention_effectiveness']:.0%}")
c.metric("Verification cost", f"Rs. {policy['intervention_cost']:,.0f}")
d.metric("False-intervention friction", f"Rs. {policy['false_positive_friction_cost']:,.0f}")
st.info("Recommendation rule: intervene when expected avoided fulfillment cost exceeds intervention cost plus expected false-intervention friction.")

st.subheader("Recent Decisions")
hist = dataframe(
    "SELECT decided_at, order_id, probability, recommendation, manager_decision, net_expected_savings "
    "FROM manager_decisions ORDER BY decided_at DESC LIMIT 10"
)
if len(hist):
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.caption("No manager decisions have been recorded yet. Open Score Order to begin.")
