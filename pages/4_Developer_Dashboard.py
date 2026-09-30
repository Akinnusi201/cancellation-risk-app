import streamlit as st

from src.auth import require_role
from src.database.duckdb_manager import dataframe
from src.models.registry import active_metadata
from src.models.train import list_candidates

require_role("developer")
st.title("🧭 Developer Dashboard")
st.caption("DataOps creates versioned datasets. ModelOps creates candidates. Production changes only after explicit promotion.")

meta = active_metadata()
versions = dataframe("SELECT dataset_version, created_at, order_count, active FROM dataset_versions ORDER BY created_at DESC")
candidates = list_candidates()
recent = dataframe("SELECT created_at, event_type, status, message FROM system_events ORDER BY created_at DESC LIMIT 8")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Production Model", f"LightGBM {meta.get('model_version')}" if meta else "Missing")
c2.metric("Production Dataset", meta.get("dataset_version", "None") if meta else "None")
c3.metric("Dataset Versions", len(versions))
c4.metric("Candidate Models", len(candidates))

st.subheader("Workflow")
st.code("New batch → DataOps validation/versioning → ModelOps candidate training → compare metrics → explicit promotion → production inference", language=None)
st.success("Operations scoring is independent of this workflow. A failed experiment or upload does not remove the packaged production model.")

st.subheader("Recent System Events")
if len(recent):
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
