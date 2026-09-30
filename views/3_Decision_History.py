import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe

require_role("manager")
st.title("🧾 Decision History")
st.caption("Audit trail of manager actions and the model recommendation in effect at decision time.")

hist = dataframe(
    "SELECT decided_at, order_id, probability, threshold, recommendation, manager_decision, "
    "actual_outcome, net_expected_savings, model_name, model_version "
    "FROM manager_decisions ORDER BY decided_at DESC LIMIT 500"
)
if len(hist):
    st.dataframe(hist, use_container_width=True, hide_index=True)
else:
    st.info("No decisions have been recorded yet.")
