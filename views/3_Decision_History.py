import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe

require_role("manager")
st.title("🧾 Decision History")
st.caption("Audit trail of manager actions and the recommendation that was shown at decision time.")

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
        "net_expected_savings": "Estimated net value (Rs.)",
        "model_name": "Model",
        "model_version": "Model version",
    })
    for col in ["Cancellation risk", "Risk threshold"]:
        if col in hist.columns:
            hist[col] = hist[col].map(lambda x: f"{x:.1%}" if x is not None else "")
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.info("No decisions have been recorded yet.")
