import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata, candidate_metadata, list_registered_models
from src.retraining import list_retraining_requests

require_role("developer")
st.title("🧭 Developer Dashboard")
st.caption("DataOps versions data, Colab performs heavy training, ModelOps governs candidates and deployment, and Monitoring can request retraining when performance degrades.")

meta = active_metadata() or {}
versions = dataframe("SELECT dataset_version, created_at, order_count, active FROM dataset_versions ORDER BY created_at DESC")
models = list_registered_models()
candidate = candidate_metadata()
requests = [r for r in list_retraining_requests() if r.get("status") in {"RETRAINING_REQUIRED", "RUNNING", "CANDIDATE_READY"}]
recent = dataframe("SELECT created_at, event_type, status, message FROM system_events ORDER BY created_at DESC LIMIT 8")

production_name = meta.get("model_display_name") or str(meta.get("model_name", "Unknown")).replace("_", " ").title()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Production Model", production_name)
c2.metric("Production Dataset", meta.get("dataset_version", "None"))
c3.metric("Registered Models", len(models))
c4.metric("Candidate", candidate.get("model_id") if candidate else "None")
c5.metric("Retraining Requests", len(requests))

st.subheader("End-to-end workflow")
st.code(
    "New data → DataOps validation/versioning → Colab training suite → MLflow tracking → qualification gates → Candidate → developer approval → Production → Monitoring → retraining trigger",
    language=None,
)
st.success("Operations scoring stays available while experiments run. Production never changes until a developer explicitly promotes a candidate.")

st.subheader("Starter model suite")
if models:
    for model in models:
        status = model.get("status", "READY")
        st.write(f"**{model.get('display_name')}** · {status} · `{model.get('model_id')}`")
else:
    st.caption("No registered models found.")

st.subheader("Recent System Events")
if len(recent):
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
