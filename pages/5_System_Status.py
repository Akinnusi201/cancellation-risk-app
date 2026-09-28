import streamlit as st
from src.models.registry import active_metadata
from src.database.duckdb_manager import dataframe
from src.config import DB_PATH, MLFLOW_TRACKING_URI

st.set_page_config(page_title="System Status", page_icon="⚙️", layout="wide")
st.title("⚙️ System Status")
meta = active_metadata()
latest = dataframe("SELECT dataset_version, created_at FROM dataset_versions ORDER BY created_at DESC LIMIT 1")
last_event = dataframe("SELECT created_at,event_type,status,message FROM system_events ORDER BY created_at DESC LIMIT 10")
items = [
    ("Data Store", "Connected" if DB_PATH.exists() else "Ready on first use"),
    ("MLflow Tracking", "Configured"),
    ("Production Model", f"LightGBM v{meta['model_version']}" if meta else "Not trained"),
    ("Dataset", meta["dataset_version"] if meta else (latest.iloc[0].dataset_version if len(latest) else "None")),
    ("Prediction Engine", "Ready" if meta else "Waiting for model"),
]
cols = st.columns(len(items))
for c, (k, v) in zip(cols, items):
    c.metric(k, v)
st.caption(f"MLflow backend: {MLFLOW_TRACKING_URI}")
st.subheader("Recent System Events")
st.dataframe(last_event, use_container_width=True, hide_index=True)
