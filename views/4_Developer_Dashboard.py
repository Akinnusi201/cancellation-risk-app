import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata, candidate_metadata, list_registered_models
from src.retraining import list_retraining_requests

require_role("developer")
st.markdown("""<div class="cr-hero"><div class="cr-eyebrow">DEVELOPER · CONTROL CENTER</div><h1>🧭&nbsp; Developer Dashboard</h1><p>Follow the complete ML lifecycle from versioned data and tracked experiments to governed deployment, monitoring, and retraining.</p></div>""", unsafe_allow_html=True)

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

st.subheader("🔁 End-to-end workflow")
st.markdown(
    """<div class="cr-flow">
    <span class="cr-flow-step">📦 DataOps</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">🧪 Experiments</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">📚 MLflow</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">⭐ Candidate</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">✅ Approval</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">🚀 Production</span><span class="cr-flow-arrow">→</span>
    <span class="cr-flow-step">📈 Monitoring</span>
    </div>""",
    unsafe_allow_html=True,
)
st.success("Operations scoring stays available while training and experiments run. Production never changes until a developer explicitly promotes a Candidate.")

st.subheader("⚡ Quick actions")
q1, q2, q3, q4 = st.columns(4)
with q1:
    st.page_link("views/5_DataOps.py", label="📦 Ingest data", use_container_width=True)
with q2:
    st.page_link("views/8_MLflow_Experiments.py", label="🧪 Run experiment", use_container_width=True)
with q3:
    st.page_link("views/6_ModelOps.py", label="🤖 Review models", use_container_width=True)
with q4:
    st.page_link("views/7_Model_Monitoring.py", label="📈 Check health", use_container_width=True)

st.subheader("🤖 Starter model suite")
if models:
    for model in models:
        status = model.get("status", "READY")
        status_icon = {"PRODUCTION": "🟢", "CANDIDATE": "🟣", "READY": "⚪"}.get(status, "⚪")
        st.write(f"{status_icon} **{model.get('display_name')}** · {status.title()} · `{model.get('model_id')}`")
else:
    st.caption("No registered models found.")

st.subheader("🕒 Recent System Events")
if len(recent):
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
