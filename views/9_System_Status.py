from pathlib import Path
import streamlit as st

from src.auth import require_role
from src.config import ARTIFACT_DIR, DB_PATH, MLFLOW_TRACKING_URI
from src.database.duckdb_manager import dataframe
from src.models.registry import PRODUCTION_MODEL_PATH, active_metadata

require_role("developer")
st.title("⚙️ System Status")
meta = active_metadata()
latest = dataframe("SELECT dataset_version, created_at FROM dataset_versions ORDER BY created_at DESC LIMIT 1")
last_event = dataframe("SELECT created_at,event_type,status,message FROM system_events ORDER BY created_at DESC LIMIT 10")

items = [
    ("Data Store", "Connected" if DB_PATH.exists() else "Ready on first use"),
    ("Packaged Model", "Available" if PRODUCTION_MODEL_PATH.exists() else "Missing"),
    ("Production Model", f"LightGBM {meta.get('model_version')}" if meta else "Missing"),
    ("Production Dataset", meta.get("dataset_version") if meta else (latest.iloc[0].dataset_version if len(latest) else "None")),
    ("Inference", "Ready" if meta and PRODUCTION_MODEL_PATH.exists() else "Blocked"),
]
cols = st.columns(len(items))
for c, (k, v) in zip(cols, items):
    c.metric(k, v)

st.success("App startup loads the packaged production artifact. It does not retrain the model.")
st.caption(f"Runtime MLflow backend: {MLFLOW_TRACKING_URI}")
st.subheader("Recent System Events")
if len(last_event):
    st.dataframe(last_event, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
