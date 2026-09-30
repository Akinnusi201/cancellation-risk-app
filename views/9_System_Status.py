from pathlib import Path

import streamlit as st

from src.auth import require_role
from src.config import DB_PATH, MLFLOW_TRACKING_URI, ROOT
from src.database.duckdb_manager import dataframe
from src.models.registry import PRODUCTION_MODEL_PATH, active_metadata

require_role("developer")
st.title("⚙️ System Status")
meta = active_metadata()
latest = dataframe("SELECT dataset_version, created_at FROM dataset_versions ORDER BY created_at DESC LIMIT 1")
last_event = dataframe("SELECT created_at,event_type,status,message FROM system_events ORDER BY created_at DESC LIMIT 10")

workflow_dir = ROOT / ".github" / "workflows"
workflow_files = []
if workflow_dir.exists():
    workflow_files = sorted(list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml")))
dockerfile = ROOT / "Dockerfile"

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

st.subheader("DevOps Readiness")
a, b, c = st.columns(3)
a.metric("Automated CI", "Configured" if workflow_files else "Not detected")
b.metric("Docker Image", "Configured" if dockerfile.exists() else "Missing")
c.metric("Packaged Inference Test", "Enabled" if (ROOT / "tests" / "test_inference_artifact.py").exists() else "Missing")
if workflow_files:
    st.caption("GitHub Actions workflow detected: " + ", ".join(p.name for p in workflow_files) + ". It runs tests, verifies packaged inference, then builds the Docker image.")
else:
    st.warning("No .github/workflows/*.yml file is present in this deployed checkout. Copy hidden files into the GitHub repository and commit the workflow directory.")

st.subheader("Recent System Events")
if len(last_event):
    st.dataframe(last_event, use_container_width=True, hide_index=True)
else:
    st.caption("No runtime events yet.")
